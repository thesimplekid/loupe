//! LLM-driven code-review scanner.
//!
//! Pipeline: walk source files → fan out one agent session per file →
//! wait. Each session gets the full MCP tool surface — the agent reads
//! the file, optionally cross-checks `query_prior_findings` /
//! `get_finding_by_id` for duplicates, generates a regression-test
//! PoC, and (only if it's confident) calls `submit_finding`. The MCP
//! `submit_finding` tool POSTs straight to
//! `/v1/jobs/{job_id}/findings`, so submissions land on the server
//! before this scanner returns. The scanner's own return value is
//! always an empty `Vec<Finding>` — its job is orchestration, not
//! emission.
//!
//! Why no separate validation pass: the agent owns its own validation
//! loop (it has tools to read prior findings and the worktree, and is
//! asked to produce a regression-test PoC inline). A second worker-
//! side parse would be a poor model of what the agent already did.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use loupe_core::Finding;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::client::ServerClient;
use crate::llm::prompts::{self, DISCOVERY};
use crate::llm::{LlmBackend, LlmRequest, RateLimited, DEFAULT_REQUEST_TIMEOUT};
use crate::scanner::{ScanContext, Scanner};

/// Repo-relative, `/`-form path used as the progress key. Computed
/// identically here and at report time so the skip-filter and the
/// recorded rows always match.
fn rel_path(workdir: &Path, file: &Path) -> String {
	file.strip_prefix(workdir).unwrap_or(file).to_string_lossy().into_owned()
}

/// Per-file fan-out outcome. A healthy session that produced no
/// submission is still `Done` — the agent decided there was nothing to
/// report. `RateLimited` is a *soft* stop (provider throttled us):
/// the scan checkpoints and resumes rather than hard-failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunOutcome {
	Done,
	Error,
	RateLimited,
}

pub const SCANNER_ID: &str = "llm-code-review";
const CAPABILITIES: &[&str] = &["scan:llm"];

/// Config knobs operators can set via the repo's `scanner_config`
/// JSON. Defaults cover Rust, C/C++, JS/TS, Python, Go, Ruby, PHP, and
/// JVM/Swift sources, plus a broad excludelist that drops common
/// build / vendor / test dirs across those ecosystems. Tighten or
/// loosen per-repo by sending a partial JSON override (see
/// `ScannerConfigPatch`).
#[derive(Debug, Clone)]
pub struct ScannerConfig {
	pub max_concurrent_files: usize,
	pub max_file_bytes: u64,
	pub per_request_timeout: Duration,
	/// File extensions the walk will consider, lower-cased and without
	/// the leading dot (e.g. `["rs", "cpp"]`).
	pub include_extensions: Vec<String>,
	/// Directory names whose entire subtree the walk will skip. Match
	/// is by *path component* (exact equality, case-sensitive), so
	/// `"tests"` excludes any directory literally named `tests` at any
	/// depth — `crates/cdk-integration-tests/` is **not** excluded
	/// because no component of that path equals `tests`. Always
	/// includes `.git` regardless of this list.
	pub exclude_dir_components: Vec<String>,
	/// Substrings matched against the *file name only* (not the full
	/// path), used to drop common test conventions like `_test.go`,
	/// `foo.test.ts`, and `bar.spec.js`. Substring match, case-
	/// sensitive.
	pub exclude_filename_patterns: Vec<String>,
}

impl Default for ScannerConfig {
	fn default() -> Self {
		// `LOUPE_MAX_CONCURRENT_FILES` is a debug knob: a personal
		// Anthropic/OpenAI account can hit per-account concurrency
		// limits with 8-way parallel claude invocations, which manifests
		// as "every session in the first batch times out at 30s with
		// empty stdout." Setting this env var to 1–2 lets an operator
		// confirm the rate-limit theory without touching code.
		// The per-repo `scanner_config` JSON override (applied via
		// `apply_patch` later) still wins over this default.
		let max_concurrent_files = std::env::var("LOUPE_MAX_CONCURRENT_FILES")
			.ok()
			.and_then(|v| v.parse::<usize>().ok())
			.filter(|&n| n > 0)
			.unwrap_or(8);
		Self {
			max_concurrent_files,
			max_file_bytes: 64 * 1024,
			per_request_timeout: DEFAULT_REQUEST_TIMEOUT,
			include_extensions: default_extensions(),
			exclude_dir_components: default_excluded_dir_components(),
			exclude_filename_patterns: default_excluded_filename_patterns(),
		}
	}
}

/// Partial override applied on top of `ScannerConfig::default()` (or a
/// constructor-supplied baseline) when the server passes a non-null
/// `scanner_config` in the lease envelope. `None` for any field means
/// "leave the baseline alone"; `Some(...)` replaces the field
/// wholesale.
///
/// Replacing rather than merging is intentional: when an operator
/// writes `{"include_extensions":["c","h"]}` for a C-only repo they
/// almost always *don't* want our default Rust/JS/Python/etc.
/// extensions silently appended.
///
/// `exclude_path_substrings` is accepted as a back-compat alias for
/// `exclude_dir_components`. The old field was substring-based against
/// the full path, which over-matched (any path containing the
/// substring "tests" was dropped, including `cdk-integration-tests`).
/// Old server configs are routed through the new path-component
/// matcher; entries that started with `/` are stripped of their
/// leading slash so `"/target"` resolves to component `"target"`.
#[derive(Debug, Default, Clone, serde::Deserialize)]
#[serde(default)]
pub struct ScannerConfigPatch {
	pub max_concurrent_files: Option<usize>,
	pub max_file_bytes: Option<u64>,
	pub per_request_timeout_seconds: Option<u64>,
	pub include_extensions: Option<Vec<String>>,
	#[serde(alias = "exclude_path_substrings")]
	pub exclude_dir_components: Option<Vec<String>>,
	pub exclude_filename_patterns: Option<Vec<String>>,
}

impl ScannerConfig {
	pub(crate) fn apply_patch(&mut self, p: ScannerConfigPatch) {
		if let Some(v) = p.max_concurrent_files {
			self.max_concurrent_files = v;
		}
		if let Some(v) = p.max_file_bytes {
			self.max_file_bytes = v;
		}
		if let Some(v) = p.per_request_timeout_seconds {
			self.per_request_timeout = Duration::from_secs(v);
		}
		if let Some(v) = p.include_extensions {
			self.include_extensions = v;
		}
		if let Some(v) = p.exclude_dir_components {
			// Tolerate legacy entries that began with `/` (the old
			// substring-style excludes like `/target`, `/.next`).
			// Trim the leading slash and any trailing slash so they
			// collapse to a single path component name.
			self.exclude_dir_components = v
				.into_iter()
				.map(|s| s.trim_start_matches('/').trim_end_matches('/').to_owned())
				.filter(|s| !s.is_empty())
				.collect();
		}
		if let Some(v) = p.exclude_filename_patterns {
			self.exclude_filename_patterns = v;
		}
	}
}

fn default_extensions() -> Vec<String> {
	[
		// Rust.
		"rs", // C / C++ / Obj-C.
		"c", "h", "cc", "hh", "cpp", "hpp", "cxx", "hxx", "m", "mm", // JS / TS.
		"js", "jsx", "mjs", "cjs", "ts", "tsx", // Python.
		"py",  // Go.
		"go",  // Ruby.
		"rb",  // PHP.
		"php", // JVM family.
		"java", "kt", "kts", "scala", "groovy", // Swift.
		"swift",  // Misc.
		"dart", "ex", "exs", "rs.in",
	]
	.into_iter()
	.map(String::from)
	.collect()
}

fn default_excluded_dir_components() -> Vec<String> {
	[
		// VCS metadata (always; `.git` is also hard-coded in the walker
		// so this list only needs to cover other VCSes operators may
		// have).
		".git",
		".jj",
		".hg",
		".svn",
		// Test / example dirs. Policy: keep excluded by default to keep
		// scanner noise down. Operators who want fuzz harnesses or
		// example binaries reviewed override `exclude_dir_components`
		// in `scanner_config`.
		"tests",
		"test",
		"examples",
		"__tests__",
		// Build artefacts across ecosystems.
		"target",
		"build",
		"dist",
		"out",
		".next",
		".nuxt",
		"coverage",
		// Vendored deps.
		"node_modules",
		"vendor",
		".venv",
		"venv",
		"env",
		// Caches.
		"__pycache__",
		".tox",
		".gradle",
		".mypy_cache",
		".pytest_cache",
	]
	.into_iter()
	.map(String::from)
	.collect()
}

fn default_excluded_filename_patterns() -> Vec<String> {
	// Substring match against `file_name()`. Catches Go's `*_test.go`,
	// TypeScript/JS `*.test.ts` and `*.spec.js`, Python's
	// `test_*.py` (via the leading "test_" run is *not* covered here
	// to avoid hitting unrelated names — Python tests typically live
	// under a `tests/` directory, which is already excluded by the
	// dir-component list).
	["_test.", ".test.", ".spec."].into_iter().map(String::from).collect()
}

pub struct LlmCodeReviewScanner {
	backend: Arc<dyn LlmBackend>,
	config: ScannerConfig,
	/// Conditional `bkb_hint` substitution string for [`prompts::DISCOVERY`].
	/// Either [`prompts::BKB_HINT_ATTACHED`] (when the worker has
	/// attached bkb-mcp to the per-call MCP config — the agent gets
	/// the bkb tool list spelled out in the prompt's "Scope of
	/// knowledge" section) or empty (no mention of bkb at all, since
	/// the agent's tool catalog won't list bkb tools either way).
	bkb_hint: &'static str,
	/// Server client for resumable-scan progress: GET the
	/// already-scanned set before fan-out, POST each file as it
	/// finishes. `None` (e.g. unit tests with no server) disables
	/// resumption — the scan runs every walked file, as before.
	progress_client: Option<Arc<ServerClient>>,
}

impl LlmCodeReviewScanner {
	pub fn new(backend: Arc<dyn LlmBackend>) -> Self {
		Self { backend, config: ScannerConfig::default(), bkb_hint: "", progress_client: None }
	}

	pub fn with_config(mut self, config: ScannerConfig) -> Self {
		self.config = config;
		self
	}

	/// Attach the server client used for resumable-scan progress. With
	/// it set, `scan()` skips files already scanned at the current
	/// commit and reports each file as it completes, so a rate-limited
	/// scan resumes without re-spending tokens on finished files.
	pub fn with_progress_client(mut self, client: Arc<ServerClient>) -> Self {
		self.progress_client = Some(client);
		self
	}

	/// Mark this scanner as having `bkb-mcp` attached to its per-call
	/// MCP config. Toggles the conditional bkb section in the
	/// discovery prompt so the agent knows the bkb tools exist and
	/// how to honestly cite their output.
	pub fn with_bkb(mut self, attached: bool) -> Self {
		self.bkb_hint = if attached { prompts::BKB_HINT_ATTACHED } else { "" };
		self
	}
}

#[async_trait]
impl Scanner for LlmCodeReviewScanner {
	fn id(&self) -> &'static str {
		SCANNER_ID
	}

	fn capabilities(&self) -> &[&'static str] {
		CAPABILITIES
	}

	async fn scan(&self, ctx: &ScanContext) -> Result<Vec<Finding>> {
		// Apply any per-repo overrides from the lease envelope on top
		// of our baseline config.
		let mut cfg = self.config.clone();
		if !ctx.config.is_null() {
			match serde_json::from_value::<ScannerConfigPatch>(ctx.config.clone()) {
				Ok(patch) => cfg.apply_patch(patch),
				Err(e) => {
					tracing::warn!(error = %e, "ignoring scanner_config: not a ScannerConfigPatch");
				},
			}
		}

		let (mut files, stats) = walk_source_files(&ctx.workdir, &cfg);
		// Emit the walker summary up-front so an operator can see, in
		// a single log line, why the file count is what it is. The
		// `largest_dropped` list is rendered as a compact string so
		// it survives the `tracing` field formatter cleanly.
		let largest_dropped = stats
			.largest_dropped
			.iter()
			.map(|(p, s)| format!("{p} ({s}B)"))
			.collect::<Vec<_>>()
			.join(", ");
		tracing::info!(
			workdir = %ctx.workdir.display(),
			visited = stats.visited_files,
			selected = files.len(),
			dropped_ext = stats.dropped_extension,
			pruned_dirs = stats.pruned_dir_component,
			dropped_filename = stats.dropped_filename_pattern,
			dropped_size = stats.dropped_size,
			max_file_bytes = cfg.max_file_bytes,
			largest_dropped = %largest_dropped,
			"llm-code-review walker summary"
		);
		if files.is_empty() {
			return Ok(Vec::new());
		}

		// Resumable scans: drop files already scanned at this commit so
		// a continuation only spends tokens on the remainder. Strictly
		// best-effort — if the lookup fails we scan everything (a re-
		// scan wastes tokens but is never *wrong*; findings dedup on
		// the server). The progress key is `(repo_id, head_sha,
		// rel_path)`; `head_sha` is the resolved commit the runner
		// checked out.
		if let Some(client) = &self.progress_client {
			match client.scanned_files(ctx.repo_id, &ctx.head_sha).await {
				Ok(done) if !done.is_empty() => {
					let done: HashSet<String> = done.into_iter().collect();
					let before = files.len();
					files.retain(|p| !done.contains(&rel_path(&ctx.workdir, p)));
					let skipped = before - files.len();
					if skipped > 0 {
						tracing::info!(
							skipped,
							remaining = files.len(),
							commit = %ctx.head_sha,
							"llm-code-review: resuming; skipping already-scanned files",
						);
					}
				},
				Ok(_) => {},
				Err(e) => {
					tracing::warn!(
						error = %e,
						"llm-code-review: scan-progress lookup failed; scanning all files",
					);
				},
			}
			if files.is_empty() {
				tracing::info!(
					commit = %ctx.head_sha,
					"llm-code-review: all files already scanned at this commit; nothing to do",
				);
				return Ok(Vec::new());
			}
		}

		tracing::info!(
			files = files.len(),
			backend = self.backend.id(),
			"llm-code-review starting agent fan-out"
		);

		let (errors, launched) = self
			.run_all(
				&cfg,
				&ctx.workdir,
				&files,
				ctx.repo_id,
				ctx.job_id,
				&ctx.head_sha,
				self.bkb_hint,
				&ctx.cancel,
				&ctx.rate_limited,
				&ctx.resume_at,
			)
			.await;

		// A provider rate limit is a soft stop, not a broken backend.
		// Report partial (the runner turns this into
		// `CompleteOutcome::Partial`; the server checkpoints progress
		// and auto-schedules a continuation) and explicitly skip the
		// "every session errored ⇒ bail" hard-fail below — under a
		// rate limit most launched sessions error for the same benign
		// reason, and failing would re-run the whole commit from
		// scratch.
		if ctx.rate_limited.load(Ordering::SeqCst) {
			tracing::warn!(
				errored = errors,
				launched,
				remaining_files = files.len(),
				commit = %ctx.head_sha,
				"llm-code-review: stopped early on provider rate limit; \
				 reporting partial (scan will auto-resume where it left off)",
			);
			return Ok(Vec::new());
		}

		// Hard-fail when every agent session errored. Without this, an
		// LLM scanner that's completely broken (sandbox can't reach the
		// CLI, auth missing, network blocked) would silently complete
		// as "succeeded with 0 findings", which an operator can't tell
		// apart from "this is a clean repo." The agent producing no
		// findings on a healthy session is a success — only backend /
		// sandbox errors fail the scan.
		if errors > 0 && errors == launched {
			anyhow::bail!(
				"llm-code-review: every one of {n} agent sessions errored; \
				 check worker logs for the underlying error (`RUST_LOG=loupe_worker=debug` \
				 surfaces the per-call failures)",
				n = launched,
			);
		}
		if errors > 0 {
			tracing::warn!(
				errored = errors,
				total = files.len(),
				"llm-code-review: some agent sessions errored",
			);
		}
		tracing::info!("llm-code-review finished; submissions arrived via MCP");
		// The scanner's role is orchestration; agent submissions
		// already landed on the server via the MCP `submit_finding`
		// tool. The runner's `submit_findings` batch call (which
		// follows scan(...)) thus has nothing to do for this scanner.
		Ok(Vec::new())
	}
}

impl LlmCodeReviewScanner {
	/// Fan out one agent session per file with bounded concurrency.
	/// Returns `(errors, launched_sessions)`. Per-session "no finding"
	/// is a success (not an error).
	///
	/// `rate_limited` is the shared soft-stop flag: `run_one` raises it
	/// the instant a provider rate limit is seen. The launch loop
	/// checks it before each new session and stops launching once set
	/// (it still drains every already-spawned session — same as the
	/// `cancel` path — so in-flight work and its progress reports
	/// aren't lost).
	#[allow(clippy::too_many_arguments)]
	async fn run_all(
		&self, cfg: &ScannerConfig, workdir: &Path, files: &[PathBuf], repo_id: i64, job_id: i64,
		commit_sha: &str, bkb_hint: &'static str, cancel: &CancellationToken,
		rate_limited: &Arc<AtomicBool>, resume_at: &Arc<AtomicI64>,
	) -> (usize, usize) {
		let sem = Arc::new(Semaphore::new(cfg.max_concurrent_files));
		let mut handles = Vec::with_capacity(files.len());
		for path in files {
			if cancel.is_cancelled() {
				break;
			}
			if rate_limited.load(Ordering::SeqCst) {
				// A prior session hit the provider limit. Stop
				// launching new ones; the loop below still drains
				// whatever is already in flight.
				break;
			}
			let permit = sem.clone().acquire_owned().await.expect("semaphore not closed");
			// Re-check after the (blocking) permit acquire: that await
			// may have parked here while an in-flight session hit the
			// limit and raised the flag. Without this second check a
			// session could still slip through right after the limit.
			if rate_limited.load(Ordering::SeqCst) {
				break;
			}
			let backend = self.backend.clone();
			let progress = self.progress_client.clone();
			let cfg_owned = cfg.clone();
			let workdir = workdir.to_path_buf();
			let path = path.clone();
			let cancel = cancel.clone();
			let rl = rate_limited.clone();
			let ra = resume_at.clone();
			let commit = commit_sha.to_owned();
			handles.push(tokio::spawn(async move {
				let _permit = permit;
				run_one(
					backend, progress, rl, ra, &workdir, &path, &cfg_owned, repo_id, job_id,
					&commit, bkb_hint, cancel,
				)
				.await
			}));
		}

		let launched = handles.len();
		let mut errors = 0usize;
		for h in handles {
			match h.await {
				Ok(RunOutcome::Done) => {},
				Ok(RunOutcome::RateLimited) => {
					// Already mirrored into `rate_limited` by run_one;
					// not counted as an error (it's a pause, not a
					// failure) so it can't trip the all-errored bail.
				},
				Ok(RunOutcome::Error) => errors += 1,
				Err(e) => {
					tracing::warn!(error = %e, "agent session task panicked");
					errors += 1;
				},
			}
		}
		(errors, launched)
	}
}

/// Run one agent session against `file`.
///
/// - `RunOutcome::Done` — session healthy (a "no finding" result is
///   still healthy: the agent decided there was nothing to report). On
///   success the file is reported to the server's scan-progress so a
///   continuation skips it; that POST is best-effort and never changes
///   the outcome.
/// - `RunOutcome::RateLimited` — the backend failed with a provider
///   rate limit. Raises the shared `rate_limited` flag so the launch
///   loop stops, and does *not* record progress (the file must be
///   retried next run).
/// - `RunOutcome::Error` — any other session-level failure (sandbox /
///   network / CLI). Counted toward the all-errored bail; not recorded
///   as done.
#[allow(clippy::too_many_arguments)]
async fn run_one(
	backend: Arc<dyn LlmBackend>, progress: Option<Arc<ServerClient>>,
	rate_limited: Arc<AtomicBool>, resume_at: Arc<AtomicI64>, workdir: &Path, file: &Path,
	cfg: &ScannerConfig, repo_id: i64, job_id: i64, commit_sha: &str, bkb_hint: &'static str,
	cancel: CancellationToken,
) -> RunOutcome {
	let rel = rel_path(workdir, file);
	let prompt = prompts::render(DISCOVERY, &[("file", &rel), ("bkb_hint", bkb_hint)]);
	tracing::info!(file = %rel, "llm-code-review: launching agent session");
	let started = std::time::Instant::now();
	let req = LlmRequest {
		prompt,
		workdir: workdir.to_path_buf(),
		timeout: cfg.per_request_timeout,
		cancel,
		repo_id: Some(repo_id),
		job_id: Some(job_id),
		finding_id: None,
	};
	match backend.run(req).await {
		Ok(r) => {
			tracing::debug!(
				file = %rel,
				elapsed_ms = started.elapsed().as_millis() as u64,
				response_chars = r.text.chars().count(),
				"agent session finished",
			);
			// Best-effort: a lost progress report only costs a re-scan
			// of this one file next run, never correctness.
			if let Some(client) = &progress {
				if let Err(e) =
					client.report_scan_progress(job_id, commit_sha, vec![rel.clone()]).await
				{
					tracing::warn!(
						file = %rel,
						error = %e,
						"llm-code-review: failed to record scan progress; \
						 file may be re-scanned on a continuation",
					);
				}
			}
			RunOutcome::Done
		},
		Err(e) => {
			if let Some(rl) = e.chain().find_map(|c| c.downcast_ref::<RateLimited>()) {
				rate_limited.store(true, Ordering::SeqCst);
				// Record the *earliest* hint observed so the runner
				// reports the soonest legitimate resume time. Sentinel
				// 0 in the atom means "no hint yet"; a session without
				// a parseable hint is a no-op.
				if let Some(hint) = rl.resume_at {
					if hint > 0 {
						let mut cur = resume_at.load(Ordering::SeqCst);
						loop {
							let new = if cur == 0 || hint < cur { hint } else { cur };
							if new == cur {
								break;
							}
							match resume_at.compare_exchange(
								cur,
								new,
								Ordering::SeqCst,
								Ordering::SeqCst,
							) {
								Ok(_) => break,
								Err(actual) => cur = actual,
							}
						}
					}
				}
				tracing::warn!(
					file = %rel,
					resume_at = ?rl.resume_at,
					"llm-code-review: agent session stopped on provider rate limit; \
					 will resume this file on the next run",
				);
				RunOutcome::RateLimited
			} else {
				tracing::warn!(file = %rel, error = %e, "agent session failed");
				RunOutcome::Error
			}
		},
	}
}

/// Summary counters from a walk. Returned alongside the selected
/// file list so the orchestrator can log a structured diagnostic
/// describing what was kept and what was dropped — without this an
/// operator has no way to tell "the agent found nothing" from "the
/// walker silently skipped every file because of the extension /
/// exclude / size filters." The list of the largest size-dropped
/// paths is capped at five so the log line stays bounded.
#[derive(Debug, Default, Clone)]
pub(crate) struct WalkStats {
	pub visited_files: usize,
	pub dropped_extension: usize,
	pub pruned_dir_component: usize,
	pub dropped_filename_pattern: usize,
	pub dropped_size: usize,
	/// `(relative_path, size_bytes)` for up to five oversized files,
	/// largest first. Captured for the summary log so the operator
	/// has concrete paths to look at when deciding whether to bump
	/// `max_file_bytes` per-repo.
	pub largest_dropped: Vec<(String, u64)>,
}

/// Walk the worktree for source files.
///
/// Strategy is intentionally language-agnostic: walk the whole
/// worktree from `workdir`, prune any directory whose name component
/// matches `cfg.exclude_dir_components` (or is `.git`), then for each
/// remaining file apply the extension allowlist, the filename-pattern
/// exclude list, and the per-file size cap. There is **no**
/// Rust/Cargo-specific narrowing — that used to live here and was a
/// source of bugs (workspace member globs like `crates/*` resolved
/// literally to a directory that doesn't exist, so cdk and similar
/// repos silently fell through to a partial walk while pretending
/// they had targeted roots).
///
/// Returns the file list together with a [`WalkStats`] summary the
/// caller can log.
pub(crate) fn walk_source_files(workdir: &Path, cfg: &ScannerConfig) -> (Vec<PathBuf>, WalkStats) {
	let mut out: Vec<PathBuf> = Vec::new();
	let mut stats = WalkStats::default();

	let walker = walkdir::WalkDir::new(workdir).into_iter().filter_entry(|e| {
		// Prune excluded directories wholesale. Exclude checks are
		// intentionally relative to the repository root so checkout
		// paths like `/tmp/test/repo` don't cause the root to be
		// pruned because `test` is an excluded in-repo directory name.
		if e.file_type().is_dir() {
			let rel = e.path().strip_prefix(workdir).unwrap_or(e.path());
			let excluded = is_excluded_dir_component(rel, cfg);
			if excluded {
				stats.pruned_dir_component += 1;
				tracing::debug!(path = %rel.display(), "walk prune: dir component");
			}
			!excluded
		} else {
			true
		}
	});

	for entry in walker.filter_map(|r| r.ok()) {
		if !entry.file_type().is_file() {
			continue;
		}
		stats.visited_files += 1;

		let path = entry.path();
		if !has_allowed_extension(path, &cfg.include_extensions) {
			stats.dropped_extension += 1;
			tracing::debug!(path = %path.display(), "walk drop: extension");
			continue;
		}
		let rel = path.strip_prefix(workdir).unwrap_or(path);
		if is_excluded_dir_component(rel, cfg) {
			tracing::debug!(path = %path.display(), "walk drop: dir component");
			continue;
		}
		if is_excluded_filename(path, &cfg.exclude_filename_patterns) {
			stats.dropped_filename_pattern += 1;
			tracing::debug!(path = %path.display(), "walk drop: filename pattern");
			continue;
		}
		let size = match entry.metadata() {
			Ok(m) => m.len(),
			Err(_) => 0,
		};
		if size > cfg.max_file_bytes {
			stats.dropped_size += 1;
			let rel = path.strip_prefix(workdir).unwrap_or(path).to_string_lossy().into_owned();
			tracing::debug!(path = %rel, size = size, cap = cfg.max_file_bytes, "walk drop: size");
			record_largest_dropped(&mut stats.largest_dropped, rel, size);
			continue;
		}
		out.push(entry.into_path());
	}
	out.sort();
	out.dedup();
	(out, stats)
}

/// Maintain a top-5 list (largest first) of oversized paths.
/// `O(n)` insert because the list is capped at 5; a heap would be
/// overkill.
fn record_largest_dropped(top: &mut Vec<(String, u64)>, path: String, size: u64) {
	const CAP: usize = 5;
	let pos = top.iter().position(|(_, s)| size > *s).unwrap_or(top.len());
	top.insert(pos, (path, size));
	if top.len() > CAP {
		top.truncate(CAP);
	}
}

fn is_excluded_dir_component(path: &Path, cfg: &ScannerConfig) -> bool {
	// `.git` is hard-coded so a misconfigured `exclude_dir_components`
	// can't accidentally let the scanner read the git store. Other
	// VCS metadata dirs live in `default_excluded_dir_components` and
	// are removable by operators who know what they're doing.
	for component in path.components() {
		let Some(name) = component.as_os_str().to_str() else { continue };
		if name == ".git" {
			return true;
		}
		if cfg.exclude_dir_components.iter().any(|c| c == name) {
			return true;
		}
	}
	false
}

fn is_excluded_filename(path: &Path, patterns: &[String]) -> bool {
	let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return false };
	patterns.iter().any(|p| name.contains(p.as_str()))
}

fn has_allowed_extension(path: &Path, exts: &[String]) -> bool {
	path.extension()
		.and_then(|e| e.to_str())
		.map(|e| exts.iter().any(|allowed| allowed.eq_ignore_ascii_case(e)))
		.unwrap_or(false)
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
	use std::sync::Arc;

	use loupe_core::RepoSpec;
	use tokio_util::sync::CancellationToken;

	use super::*;
	use crate::llm::testing::StubLlmBackend;

	fn make_ctx(workdir: &Path) -> ScanContext {
		ScanContext {
			workdir: workdir.to_path_buf(),
			repo_id: 1,
			job_id: 1,
			repo: RepoSpec {
				host: "github.com".into(),
				owner: "a".into(),
				repo: "b".into(),
				clone_url: "https://github.com/a/b.git".into(),
				branch: None,
			},
			head_sha: "deadbeef".into(),
			base_sha: None,
			config: serde_json::Value::Null,
			cancel: CancellationToken::new(),
			rate_limited: Arc::new(AtomicBool::new(false)),
			resume_at: Arc::new(std::sync::atomic::AtomicI64::new(0)),
		}
	}

	fn write_crate(root: &Path, files: &[(&str, &str)]) {
		std::fs::write(
			root.join("Cargo.toml"),
			"[package]\nname = \"x\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
		)
		.unwrap();
		std::fs::create_dir_all(root.join("src")).unwrap();
		for (path, body) in files {
			let p = root.join(path);
			if let Some(parent) = p.parent() {
				std::fs::create_dir_all(parent).unwrap();
			}
			std::fs::write(p, body).unwrap();
		}
	}

	#[test]
	fn defaults_cover_common_languages() {
		let cfg = ScannerConfig::default();
		for ext in ["rs", "c", "cpp", "h", "hpp", "js", "ts", "py", "go", "java", "swift"] {
			assert!(cfg.include_extensions.iter().any(|e| e == ext), "missing: {ext}");
		}
		// node_modules and target are excluded out of the box (as path
		// components now, not substrings).
		assert!(cfg.exclude_dir_components.iter().any(|e| e == "node_modules"));
		assert!(cfg.exclude_dir_components.iter().any(|e| e == "target"));
		assert!(cfg.exclude_dir_components.iter().any(|e| e == "tests"));
	}

	#[test]
	fn patch_overrides_only_the_fields_present() {
		let mut cfg = ScannerConfig::default();
		let original_excludes = cfg.exclude_dir_components.clone();
		let patch: ScannerConfigPatch =
			serde_json::from_str(r#"{"include_extensions":["c","h"]}"#).unwrap();
		cfg.apply_patch(patch);
		assert_eq!(cfg.include_extensions, vec!["c".to_owned(), "h".to_owned()]);
		assert_eq!(cfg.exclude_dir_components, original_excludes);
	}

	#[test]
	fn legacy_exclude_path_substrings_patch_still_applies() {
		// Servers that haven't been re-built yet may still send
		// `exclude_path_substrings`. The patch must route to
		// `exclude_dir_components`, trimming leading/trailing slashes
		// so legacy entries like `/target` collapse to the component
		// name `target`.
		let mut cfg = ScannerConfig::default();
		let patch: ScannerConfigPatch =
			serde_json::from_str(r#"{"exclude_path_substrings":["/target","node_modules/"]}"#)
				.unwrap();
		cfg.apply_patch(patch);
		assert_eq!(
			cfg.exclude_dir_components,
			vec!["target".to_owned(), "node_modules".to_owned()]
		);
	}

	#[test]
	fn walk_picks_up_non_rust_files_without_cargo_toml() {
		let tmp = tempfile::tempdir().unwrap();
		// No Cargo.toml here — walker treats the whole tree as the root.
		std::fs::create_dir_all(tmp.path().join("src")).unwrap();
		std::fs::write(tmp.path().join("src/main.cpp"), "// stub\n").unwrap();
		std::fs::write(tmp.path().join("src/util.h"), "// stub\n").unwrap();
		std::fs::write(tmp.path().join("src/app.py"), "# stub\n").unwrap();
		std::fs::write(tmp.path().join("src/page.tsx"), "// stub\n").unwrap();

		let cfg = ScannerConfig::default();
		let (files, _stats) = walk_source_files(tmp.path(), &cfg);
		let names: Vec<String> = files
			.iter()
			.map(|p| p.strip_prefix(tmp.path()).unwrap().to_string_lossy().into_owned())
			.collect();
		for expected in ["src/main.cpp", "src/util.h", "src/app.py", "src/page.tsx"] {
			assert!(names.iter().any(|n| n == expected), "missing {expected} in {names:?}");
		}
	}

	#[test]
	fn walk_excludes_node_modules_and_build_dirs_by_default() {
		let tmp = tempfile::tempdir().unwrap();
		// Real source.
		std::fs::create_dir_all(tmp.path().join("src")).unwrap();
		std::fs::write(tmp.path().join("src/index.js"), "// real\n").unwrap();
		// Vendored deps and build artefacts that must be skipped.
		std::fs::create_dir_all(tmp.path().join("node_modules/lodash")).unwrap();
		std::fs::write(tmp.path().join("node_modules/lodash/index.js"), "// vendored\n").unwrap();
		std::fs::create_dir_all(tmp.path().join("dist")).unwrap();
		std::fs::write(tmp.path().join("dist/bundle.js"), "// built\n").unwrap();

		let (files, stats) = walk_source_files(tmp.path(), &ScannerConfig::default());
		let names: Vec<String> = files
			.iter()
			.map(|p| p.strip_prefix(tmp.path()).unwrap().to_string_lossy().into_owned())
			.collect();
		assert!(names.iter().any(|n| n == "src/index.js"), "real source missing: {names:?}");
		assert!(names.iter().all(|n| !n.contains("node_modules")), "leak: {names:?}");
		assert!(names.iter().all(|n| !n.starts_with("dist/")), "leak: {names:?}");
		assert_eq!(stats.pruned_dir_component, 2, "node_modules/ and dist/ should be pruned");
	}

	#[test]
	fn walk_picks_up_src_rs_files_only() {
		let tmp = tempfile::tempdir().unwrap();
		write_crate(
			tmp.path(),
			&[
				("src/lib.rs", "// good\n"),
				("src/util.rs", "// good\n"),
				("README.md", "ignore\n"),
				("tests/integration.rs", "// excluded by tests dir\n"),
			],
		);
		let cfg = ScannerConfig::default();
		let (files, _) = walk_source_files(tmp.path(), &cfg);
		let names: Vec<String> = files
			.iter()
			.map(|p| p.strip_prefix(tmp.path()).unwrap().to_string_lossy().into_owned())
			.collect();
		assert!(names.iter().any(|n| n.ends_with("src/lib.rs")), "names: {names:?}");
		assert!(names.iter().any(|n| n.ends_with("src/util.rs")), "names: {names:?}");
		assert!(names.iter().all(|n| !n.contains("README")), "names: {names:?}");
		// The `tests/` *directory* is excluded; the path-component
		// matcher means a *file* path containing the literal sequence
		// `tests/` is not reachable here, but be defensive.
		assert!(names.iter().all(|n| !n.split('/').any(|seg| seg == "tests")), "names: {names:?}");
	}

	#[test]
	fn walk_ignores_excluded_words_in_checkout_path() {
		// Regression: exclude matching must be scoped to paths inside
		// `workdir`. A checkout under a parent named `test`, `build`,
		// or similar should not cause the repository root itself to be
		// pruned.
		let tmp = tempfile::tempdir().unwrap();
		let workdir = tmp.path().join("test").join("repo");
		std::fs::create_dir_all(&workdir).unwrap();
		std::fs::write(workdir.join("lib.rs"), "// real source\n").unwrap();
		std::fs::create_dir_all(workdir.join("src")).unwrap();
		std::fs::write(workdir.join("src/main.rs"), "// also real\n").unwrap();

		let (files, stats) = walk_source_files(&workdir, &ScannerConfig::default());
		let names: Vec<String> = files
			.iter()
			.map(|p| p.strip_prefix(&workdir).unwrap().to_string_lossy().into_owned())
			.collect();
		assert!(names.iter().any(|n| n == "lib.rs"), "lib.rs missing: {names:?}");
		assert!(names.iter().any(|n| n == "src/main.rs"), "src/main.rs missing: {names:?}");
		assert_eq!(stats.pruned_dir_component, 0, "checkout path must not count as pruned");
	}

	#[test]
	fn walk_recovers_workspace_member_with_tests_in_its_crate_name() {
		// Regression: a Rust workspace like `cashubtc/cdk` declares
		// `members = ["crates/*"]` and ships a member crate named
		// `cdk-integration-tests`. The pre-fix walker either failed
		// to expand the glob (so the targeted-roots branch found
		// nothing) and then re-dropped the crate via a substring match
		// on "tests". The new path-component matcher keeps the crate
		// because no path *component* equals "tests".
		let tmp = tempfile::tempdir().unwrap();
		std::fs::write(tmp.path().join("Cargo.toml"), "[workspace]\nmembers = [\"crates/*\"]\n")
			.unwrap();
		std::fs::create_dir_all(tmp.path().join("crates/cdk-integration-tests/src")).unwrap();
		std::fs::write(
			tmp.path().join("crates/cdk-integration-tests/src/lib.rs"),
			"// real source in a tests-named crate\n",
		)
		.unwrap();
		std::fs::write(
			tmp.path().join("crates/cdk-integration-tests/Cargo.toml"),
			"[package]\nname = \"cdk-integration-tests\"\n",
		)
		.unwrap();
		// A *real* `tests/` directory inside the crate — still excluded.
		std::fs::create_dir_all(tmp.path().join("crates/cdk-integration-tests/tests")).unwrap();
		std::fs::write(
			tmp.path().join("crates/cdk-integration-tests/tests/it.rs"),
			"// integration test, excluded\n",
		)
		.unwrap();

		let (files, _) = walk_source_files(tmp.path(), &ScannerConfig::default());
		let names: Vec<String> = files
			.iter()
			.map(|p| p.strip_prefix(tmp.path()).unwrap().to_string_lossy().into_owned())
			.collect();
		assert!(
			names.iter().any(|n| n == "crates/cdk-integration-tests/src/lib.rs"),
			"the tests-named crate's lib.rs should be reviewed: {names:?}"
		);
		assert!(
			names.iter().all(|n| n != "crates/cdk-integration-tests/tests/it.rs"),
			"the literal tests/ dir inside that crate must still be excluded: {names:?}"
		);
	}

	#[test]
	fn walk_picks_up_go_repo_without_cargo_toml() {
		// Regression: a Go repo like cashubtc/BTCNutServer has no
		// Cargo.toml, so the walker treats the whole worktree as the
		// root. `_test.go` is dropped by the filename-pattern list;
		// `vendor/` is dropped by the dir-component list; everything
		// else is selected.
		let tmp = tempfile::tempdir().unwrap();
		std::fs::write(tmp.path().join("main.go"), "package main\n").unwrap();
		std::fs::create_dir_all(tmp.path().join("internal/api")).unwrap();
		std::fs::write(tmp.path().join("internal/api/handler.go"), "package api\n").unwrap();
		std::fs::write(tmp.path().join("internal/api/handler_test.go"), "// go test\n").unwrap();
		std::fs::create_dir_all(tmp.path().join("vendor/github.com/foo")).unwrap();
		std::fs::write(tmp.path().join("vendor/github.com/foo/bar.go"), "// vendored\n").unwrap();

		let (files, _) = walk_source_files(tmp.path(), &ScannerConfig::default());
		let names: Vec<String> = files
			.iter()
			.map(|p| p.strip_prefix(tmp.path()).unwrap().to_string_lossy().into_owned())
			.collect();
		assert!(names.iter().any(|n| n == "main.go"), "names: {names:?}");
		assert!(names.iter().any(|n| n == "internal/api/handler.go"), "names: {names:?}");
		assert!(
			names.iter().all(|n| n != "internal/api/handler_test.go"),
			"_test.go must be excluded: {names:?}"
		);
		assert!(
			names.iter().all(|n| !n.starts_with("vendor/")),
			"vendor/ must be excluded: {names:?}"
		);
	}

	#[test]
	fn walk_records_size_drops_with_largest_first() {
		let tmp = tempfile::tempdir().unwrap();
		// One file just over the 64 KiB cap (should be dropped & logged).
		let big = vec![b'x'; (64 * 1024 + 1) as usize];
		std::fs::write(tmp.path().join("big.rs"), &big).unwrap();
		// One file well within the cap (should be selected).
		std::fs::write(tmp.path().join("small.rs"), "// ok\n").unwrap();

		let (files, stats) = walk_source_files(tmp.path(), &ScannerConfig::default());
		assert_eq!(files.len(), 1, "only small.rs should be selected");
		assert_eq!(stats.dropped_size, 1, "big.rs should be counted as a size drop");
		assert_eq!(stats.largest_dropped.len(), 1);
		assert_eq!(stats.largest_dropped[0].0, "big.rs");
		assert!(stats.largest_dropped[0].1 > 64 * 1024);
	}

	#[test]
	fn record_largest_dropped_keeps_top_five_largest_first() {
		let mut top: Vec<(String, u64)> = Vec::new();
		// Push out of order; expect descending by size, capped at 5.
		for (path, size) in
			[("a", 100u64), ("b", 50), ("c", 200), ("d", 75), ("e", 300), ("f", 25), ("g", 250)]
		{
			record_largest_dropped(&mut top, path.to_owned(), size);
		}
		assert_eq!(top.len(), 5);
		assert_eq!(top[0].1, 300);
		assert_eq!(top[1].1, 250);
		assert_eq!(top[2].1, 200);
		assert_eq!(top[3].1, 100);
		assert_eq!(top[4].1, 75);
	}

	#[test]
	fn git_directory_always_excluded_even_if_user_overrides() {
		let tmp = tempfile::tempdir().unwrap();
		std::fs::create_dir_all(tmp.path().join(".git/objects")).unwrap();
		// A `.rs` file under `.git/` would be selected by extension
		// alone; the hard-coded `.git` prune in
		// `is_excluded_dir_component` must keep it out even if the
		// operator stripped `.git` from their exclude list.
		std::fs::write(tmp.path().join(".git/objects/HEAD.rs"), "// not source\n").unwrap();
		std::fs::write(tmp.path().join("lib.rs"), "// real\n").unwrap();

		let mut cfg = ScannerConfig::default();
		// Operator footgun: clear the dir-component list entirely.
		cfg.exclude_dir_components.clear();

		let (files, _) = walk_source_files(tmp.path(), &cfg);
		let names: Vec<String> = files
			.iter()
			.map(|p| p.strip_prefix(tmp.path()).unwrap().to_string_lossy().into_owned())
			.collect();
		assert!(names.iter().any(|n| n == "lib.rs"), "lib.rs missing: {names:?}");
		assert!(
			names.iter().all(|n| !n.contains(".git")),
			".git/ must be excluded regardless of user config: {names:?}"
		);
	}

	#[tokio::test]
	async fn scanner_returns_empty_findings_and_calls_backend_per_file() {
		// Submissions go via MCP, not via the return value. The
		// scanner-level test pins the orchestration contract: every
		// matching file gets one backend call, scan returns [].
		let tmp = tempfile::tempdir().unwrap();
		write_crate(tmp.path(), &[("src/lib.rs", "// a\n"), ("src/util.rs", "// b\n")]);

		let calls = Arc::new(AtomicUsize::new(0));
		let calls_for_stub = calls.clone();
		let backend = Arc::new(StubLlmBackend::new("stub", move |_req: &LlmRequest| {
			calls_for_stub.fetch_add(1, Ordering::SeqCst);
			Ok(String::new())
		}));
		let scanner = LlmCodeReviewScanner::new(backend);

		let findings = scanner.scan(&make_ctx(tmp.path())).await.unwrap();
		assert!(findings.is_empty(), "scanner returns no findings — submissions go via MCP");
		assert_eq!(
			calls.load(Ordering::SeqCst),
			2,
			"every walked file must produce one agent session"
		);
	}

	#[tokio::test]
	async fn scanner_fails_loud_when_every_session_errors() {
		// Sandbox / network / CLI being completely broken must not
		// silently complete as "0 findings" — that's
		// indistinguishable from a clean repo.
		let tmp = tempfile::tempdir().unwrap();
		write_crate(tmp.path(), &[("src/lib.rs", "// a\n")]);
		let backend = Arc::new(StubLlmBackend::new("stub", |_req: &LlmRequest| {
			Err(anyhow::anyhow!("backend exploded"))
		}));
		let scanner = LlmCodeReviewScanner::new(backend);
		let err = scanner.scan(&make_ctx(tmp.path())).await.expect_err("must fail");
		assert!(err.to_string().contains("agent session"), "unexpected error: {err}");
	}

	#[tokio::test]
	async fn rate_limit_reports_partial_not_failure() {
		// Every session rate-limited. This must NOT take the
		// all-errored hard-fail path: a rate limit is a soft stop. The
		// scan returns Ok([]) and raises ctx.rate_limited so the runner
		// reports Partial (server checkpoints + auto-resumes).
		let tmp = tempfile::tempdir().unwrap();
		write_crate(tmp.path(), &[("src/a.rs", "// a\n"), ("src/b.rs", "// b\n")]);
		let backend = Arc::new(StubLlmBackend::new("stub", |_req: &LlmRequest| {
			Err(anyhow::Error::new(RateLimited {
				detail: "429 too many requests".into(),
				resume_at: None,
			}))
		}));
		let scanner = LlmCodeReviewScanner::new(backend);
		let ctx = make_ctx(tmp.path());
		let flag = ctx.rate_limited.clone();
		let findings = scanner.scan(&ctx).await.expect("rate limit is a soft stop, not an error");
		assert!(findings.is_empty());
		assert!(flag.load(Ordering::SeqCst), "ctx.rate_limited must be raised");
	}

	#[tokio::test]
	async fn rate_limit_stops_launching_and_drains() {
		// Concurrency 1 ⇒ strictly sequential sessions. The first file
		// succeeds; the second rate-limits and raises the flag; the
		// launch loop must then stop — so with 4 files only 2 sessions
		// ever run.
		let tmp = tempfile::tempdir().unwrap();
		write_crate(
			tmp.path(),
			&[
				("src/a.rs", "// a\n"),
				("src/b.rs", "// b\n"),
				("src/c.rs", "// c\n"),
				("src/d.rs", "// d\n"),
			],
		);
		let calls = Arc::new(AtomicUsize::new(0));
		let calls_for_stub = calls.clone();
		let backend = Arc::new(StubLlmBackend::new("stub", move |_req: &LlmRequest| {
			let n = calls_for_stub.fetch_add(1, Ordering::SeqCst);
			if n == 0 {
				Ok(String::new())
			} else {
				Err(anyhow::Error::new(RateLimited {
					detail: "rate limit".into(),
					resume_at: None,
				}))
			}
		}));
		let scanner = LlmCodeReviewScanner::new(backend);
		let mut ctx = make_ctx(tmp.path());
		ctx.config = serde_json::json!({ "max_concurrent_files": 1 });
		let flag = ctx.rate_limited.clone();
		scanner.scan(&ctx).await.expect("soft stop");
		assert!(flag.load(Ordering::SeqCst));
		assert_eq!(
			calls.load(Ordering::SeqCst),
			2,
			"must stop launching once rate-limited, not run all 4 files",
		);
	}

	#[tokio::test]
	async fn rate_limit_records_resume_at_hint() {
		// A rate-limited session that carries a `resume_at` hint must
		// land in `ctx.resume_at` so the runner can thread it into the
		// partial-completion report.
		let tmp = tempfile::tempdir().unwrap();
		write_crate(tmp.path(), &[("src/a.rs", "// a\n")]);
		let backend = Arc::new(StubLlmBackend::new("stub", move |_req: &LlmRequest| {
			Err(anyhow::Error::new(RateLimited {
				detail: "rate limit".into(),
				resume_at: Some(1_500),
			}))
		}));
		let scanner = LlmCodeReviewScanner::new(backend);
		let ctx = make_ctx(tmp.path());
		let resume = ctx.resume_at.clone();
		scanner.scan(&ctx).await.expect("soft stop");
		assert_eq!(
			resume.load(Ordering::SeqCst),
			1_500,
			"hint must be recorded into ctx.resume_at"
		);
	}

	#[test]
	fn record_resume_at_keeps_the_minimum() {
		// Concurrency invariant: when multiple per-file sessions
		// report different reset-time hints, the runner reports the
		// *soonest* — that's the earliest the chain can legitimately
		// resume. `record_resume_at` enforces this directly.
		let tmp = tempfile::tempdir().unwrap();
		let ctx = make_ctx(tmp.path());
		ctx.record_resume_at(Some(2_000));
		ctx.record_resume_at(Some(1_500));
		ctx.record_resume_at(Some(3_000));
		assert_eq!(ctx.resume_at.load(Ordering::SeqCst), 1_500);
		// None is a no-op (does not reset the minimum).
		ctx.record_resume_at(None);
		assert_eq!(ctx.resume_at.load(Ordering::SeqCst), 1_500);
		// Non-positive is treated as "no hint" (defends against a
		// caller passing 0 or a negative parse artefact).
		ctx.record_resume_at(Some(0));
		ctx.record_resume_at(Some(-5));
		assert_eq!(ctx.resume_at.load(Ordering::SeqCst), 1_500);
	}

	#[tokio::test]
	async fn rate_limit_without_hint_leaves_resume_at_zero() {
		// No parseable "resets <time>" ⇒ resume_at stays at the 0
		// sentinel so the runner sends `None` and the server falls
		// back to its static backoff.
		let tmp = tempfile::tempdir().unwrap();
		write_crate(tmp.path(), &[("src/a.rs", "// a\n")]);
		let backend = Arc::new(StubLlmBackend::new("stub", |_req: &LlmRequest| {
			Err(anyhow::Error::new(RateLimited { detail: "429".into(), resume_at: None }))
		}));
		let scanner = LlmCodeReviewScanner::new(backend);
		let ctx = make_ctx(tmp.path());
		let resume = ctx.resume_at.clone();
		scanner.scan(&ctx).await.expect("soft stop");
		assert_eq!(resume.load(Ordering::SeqCst), 0);
	}

	#[tokio::test]
	async fn ctx_config_override_changes_walked_extensions() {
		// A C-only repo overrides include_extensions so a `.c` file
		// is picked up even without a Cargo.toml. Pinned at the
		// scanner level because the patch lookup is in scan().
		let tmp = tempfile::tempdir().unwrap();
		std::fs::write(tmp.path().join("widget.c"), "/* stub */\n").unwrap();
		let calls = Arc::new(AtomicUsize::new(0));
		let calls_for_stub = calls.clone();
		let backend = Arc::new(StubLlmBackend::new("stub", move |_req: &LlmRequest| {
			calls_for_stub.fetch_add(1, Ordering::SeqCst);
			Ok(String::new())
		}));
		let scanner = LlmCodeReviewScanner::new(backend);
		let mut ctx = make_ctx(tmp.path());
		ctx.config = serde_json::json!({"include_extensions":["c"]});
		scanner.scan(&ctx).await.unwrap();
		assert_eq!(
			calls.load(Ordering::SeqCst),
			1,
			"ctx.config override should have caught widget.c"
		);
	}
}
