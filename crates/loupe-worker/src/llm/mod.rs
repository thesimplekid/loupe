//! LLM backend abstraction.
//!
//! A `LlmBackend` is one provider of agentic completions: it receives a
//! prompt and a read-only working directory, manages its own internal
//! tool loop (the `claude` CLI does this for us; an HTTP backend would
//! manage one explicitly), and returns the model's final text response.
//!
//! Two concrete impls today:
//!
//! - [`ClaudeCliBackend`] shells out to Anthropic's `claude` CLI.
//!   Carries optional MCP context so each invocation can call back
//!   into `loupe-worker mcp-serve` over stdio JSON-RPC — used by the
//!   discovery scanner to query prior findings and submit new ones.
//! - [`CodexCliBackend`] shells out to OpenAI's `codex` CLI. No MCP
//!   plumbing yet; used by the cross-model verifier where the prompt
//!   is self-contained and the only output is a JSON verdict.
//!
//! Picking between them at runtime: see [`build_verifier_backend`],
//! which probes PATH for `codex` and falls back to `claude` so a
//! cross-model second opinion happens when both are available
//! without mandating both.

pub mod claude_cli;
pub mod codex_cli;
pub mod mcp;
pub mod prompts;

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
pub use claude_cli::ClaudeCliBackend;
pub use codex_cli::CodexCliBackend;
pub use mcp::{McpContext, McpTlsSource};
use tokio_util::sync::CancellationToken;

/// Default per-call wall-clock budget. Per-file LLM invocations should
/// fit comfortably within this; if they don't, the call is aborted and
/// the file is treated as having produced no findings (logged warning).
///
/// 30 minutes is generous; the goal is to be the *fallback* ceiling,
/// not the operative deadline. Auditing a 1–2k-line source file
/// end-to-end (several MCP round-trips for prior-finding dedup, a PoC
/// regression-test diff, validation) routinely takes 1–3 minutes
/// against real-world Rust repos, and the previous 180s default was
/// killing roughly 4 in 5 sessions before the agent could submit.
/// Operators can still tighten via the per-repo `scanner_config` JSON
/// (`per_request_timeout_seconds`).
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(1800);

/// Pull the first balanced JSON object out of a possibly noisy text
/// response. Tolerates prose before/after the object and a single
/// markdown fence around it. Returns the slice as an owned `String`
/// because the model occasionally emits trailing junk after the
/// closing brace; we feed only what's inside the braces.
///
/// Used by the verifier scanner, which still parses JSON from the
/// model's stdout. The discovery flow doesn't need this — submission
/// goes through the MCP `submit_finding` tool.
pub fn extract_json_object(text: &str) -> Option<String> {
	let bytes = text.as_bytes();
	let start = bytes.iter().position(|b| *b == b'{')?;
	let mut depth = 0i32;
	let mut in_str = false;
	let mut escape = false;
	for (i, b) in bytes.iter().enumerate().skip(start) {
		if in_str {
			if escape {
				escape = false;
			} else if *b == b'\\' {
				escape = true;
			} else if *b == b'"' {
				in_str = false;
			}
			continue;
		}
		match *b {
			b'"' => in_str = true,
			b'{' => depth += 1,
			b'}' => {
				depth -= 1;
				if depth == 0 {
					return std::str::from_utf8(&bytes[start..=i]).ok().map(|s| s.to_owned());
				}
			},
			_ => {},
		}
	}
	None
}

/// Marker error: a backend invocation failed because the LLM provider
/// rate-limited us, not because the backend is broken or the prompt was
/// bad. Carried inside an `anyhow::Error` so callers can
/// `err.chain().any(|e| e.is::<RateLimited>())` to distinguish a soft
/// "pause and resume" stop from a hard failure. `detail` is a short,
/// already-truncated snippet of what tripped the matcher (for logs).
///
/// `resume_at` is a best-effort Unix-seconds estimate of when the
/// provider's quota window is expected to refresh, parsed from the
/// detail text (the claude CLI's "You've hit your limit · resets
/// 3:50pm (Europe/Dublin)" form). `None` means we couldn't extract a
/// reset time — callers should fall back to the static backoff.
/// Propagated all the way to the server so the scheduler doesn't
/// retry a continuation inside the lockout window only to immediately
/// trip the same limit and burn a zero-progress slot.
#[derive(Debug, Clone)]
pub struct RateLimited {
	pub detail: String,
	pub resume_at: Option<i64>,
}

impl RateLimited {
	/// Construct a `RateLimited` from a captured snippet, auto-parsing
	/// any "resets &lt;time&gt;" hint via [`parse_reset_hint`]. `now`
	/// is the reference Unix seconds (usually `SystemTime::now()`)
	/// against which a relative reset time is anchored.
	pub fn from_detail(detail: String, now: i64) -> Self {
		let resume_at = parse_reset_hint(&detail, now);
		Self { detail, resume_at }
	}
}

impl std::fmt::Display for RateLimited {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "llm provider rate limit: {}", self.detail)
	}
}

impl std::error::Error for RateLimited {}

/// Parse the claude CLI's "resets &lt;time&gt;" hint into a Unix-seconds
/// estimate of when the provider's session-quota window refreshes.
/// Recognises the shapes seen in the wild:
///
/// - `"You've hit your limit · resets 3:50pm (Europe/Dublin)"`
/// - `"resets 15:50"` (24-hour form)
/// - `"resets at 3pm"`
///
/// Returns the next future Unix-seconds instant whose wall-clock time
/// (in UTC) matches the parsed hour:minute, plus a 15-minute safety
/// buffer so the continuation lands *after* the window closes rather
/// than racing it. The timezone in parens is intentionally ignored:
/// we don't ship a tz database and an absolute clock-time guess that's
/// up to ~12h wrong is worse than waiting at most ~24h on the wrong
/// hh:mm match. The 5-hour Claude lockout dominates either way, so a
/// best-effort "wait ≥1 cycle of the displayed time" still beats the
/// pre-parse 15-minute backoff. Returns `None` when the detail has no
/// recognisable time so the caller falls back to a static backoff.
pub fn parse_reset_hint(detail: &str, now: i64) -> Option<i64> {
	let lower = detail.to_ascii_lowercase();
	// Anchor on the literal "resets" since "limit" and digits appear
	// in plenty of unrelated error text (HTTP 429 bodies, quotas).
	let idx = lower.find("resets")?;
	// Skim past the word and an optional "at "; whatever's left is
	// the time token possibly followed by "(tz)" and prose.
	let mut rest = &lower[idx + "resets".len()..];
	rest = rest.trim_start();
	if let Some(stripped) = rest.strip_prefix("at ") {
		rest = stripped;
	}
	let rest = rest.trim_start();
	// Grab the first whitespace-or-paren-delimited token; that's the
	// time-with-suffix.
	let end = rest.find(|c: char| c.is_whitespace() || c == '(' || c == ',').unwrap_or(rest.len());
	let token = &rest[..end];
	let (hour, minute) = parse_clock_token(token)?;
	let secs_of_day = hour * 3600 + minute * 60;
	let secs_per_day: i64 = 86_400;
	let now_secs_of_day = now.rem_euclid(secs_per_day);
	let day_start = now - now_secs_of_day;
	let mut target = day_start + secs_of_day as i64;
	if target <= now {
		target += secs_per_day;
	}
	// Safety buffer: provider-side clocks and our clock can drift
	// several minutes, and the displayed minute is truncated. Wait
	// 15 min past the parsed reset so the continuation lands inside
	// the new window rather than racing the boundary.
	Some(target + 15 * 60)
}

/// Parse `"3:50pm"`, `"15:50"`, `"3pm"`, `"3:50"` etc. into
/// `(hour_0_23, minute_0_59)`. Returns `None` for anything that
/// doesn't lex cleanly.
fn parse_clock_token(token: &str) -> Option<(u32, u32)> {
	let token = token.trim_end_matches(['.', ',', ';']);
	if token.is_empty() {
		return None;
	}
	// Split off optional am/pm suffix.
	let (digits, ampm) = if let Some(stripped) = token.strip_suffix("am") {
		(stripped, Some(false))
	} else if let Some(stripped) = token.strip_suffix("pm") {
		(stripped, Some(true))
	} else if let Some(stripped) = token.strip_suffix("a.m.") {
		(stripped, Some(false))
	} else if let Some(stripped) = token.strip_suffix("p.m.") {
		(stripped, Some(true))
	} else {
		(token, None)
	};
	let digits = digits.trim_end();
	let (hour_str, minute_str) = match digits.split_once(':') {
		Some((h, m)) => (h, m),
		None => (digits, "0"),
	};
	let hour: u32 = hour_str.parse().ok()?;
	let minute: u32 = minute_str.parse().ok()?;
	if minute >= 60 {
		return None;
	}
	let hour = match (ampm, hour) {
		(Some(true), 12) => 12,              // 12pm == noon
		(Some(true), h) if h < 12 => h + 12, // 1pm..11pm
		(Some(false), 12) => 0,              // 12am == midnight
		(Some(false), h) if h < 12 => h,     // 1am..11am
		(None, h) if h < 24 => h,            // 24-hour
		_ => return None,
	};
	Some((hour, minute))
}

/// Default substrings (matched case-insensitively) that mark a provider
/// rate-limit / overload response across the `claude` CLI's stderr and
/// the common HTTP error shapes it surfaces.
const DEFAULT_RATE_LIMIT_MARKERS: &[&str] = &[
	"rate limit",
	"rate-limit",
	"ratelimit",
	"hit your limit",
	"429",
	"overloaded",
	"quota",
	"usage limit",
	"too many requests",
];

/// Heuristic: does `text` look like a provider rate-limit / overload
/// message? Case-insensitive substring match over
/// [`DEFAULT_RATE_LIMIT_MARKERS`], extendable at runtime via the
/// optional `LOUPE_RATE_LIMIT_MARKERS` env (comma-separated, debug
/// knob mirroring `LOUPE_MAX_CONCURRENT_FILES`) for when a CLI changes
/// its wording in the field before we can ship a code change.
///
/// This is a heuristic over an *unstable* CLI output format. A false
/// negative is safe (we fall back to today's hard-fail). To make a
/// silent break visible, every caller that acts on a `true` here must
/// emit the stable `loupe::rate_limit` log line — see
/// [`claude_cli`](crate::llm::claude_cli).
pub fn looks_like_rate_limit(text: &str) -> bool {
	let haystack = text.to_ascii_lowercase();
	if DEFAULT_RATE_LIMIT_MARKERS.iter().any(|m| haystack.contains(m)) {
		return true;
	}
	if let Some(extra) = std::env::var_os("LOUPE_RATE_LIMIT_MARKERS") {
		if let Some(extra) = extra.to_str() {
			return extra
				.split(',')
				.map(|m| m.trim().to_ascii_lowercase())
				.filter(|m| !m.is_empty())
				.any(|m| haystack.contains(&m));
		}
	}
	false
}

#[derive(Debug, Clone)]
pub struct LlmRequest {
	pub prompt: String,
	/// Read-only working directory the backend may inspect (e.g. the
	/// scanned worktree).
	pub workdir: PathBuf,
	pub timeout: Duration,
	pub cancel: CancellationToken,
	/// Repo id for the scan currently in progress. When `Some`, the
	/// backend may attach the loupe MCP server to its agent
	/// invocation so the model can call tools like
	/// `query_prior_findings` scoped to this repo. `None` falls back
	/// to the no-MCP behaviour (just prompt + stdout).
	pub repo_id: Option<i64>,
	/// Job id for the scan currently in progress. Required for the
	/// `submit_finding` MCP tool to POST to
	/// `/v1/jobs/{job_id}/findings`; without it, that tool is not
	/// advertised. `None` falls back to query-only MCP usage (the
	/// agent can read prior findings but can't write new ones).
	pub job_id: Option<i64>,
	/// Finding id for a verify-kind session. When `Some`, the MCP
	/// server enters verify mode: `submit_finding` is hidden;
	/// `submit_verdict`, `submit_patch`, and `validate_patch` are
	/// advertised instead. `None` keeps the discovery-mode catalog.
	pub finding_id: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
	pub text: String,
	pub backend_id: &'static str,
}

#[async_trait::async_trait]
pub trait LlmBackend: Send + Sync {
	/// Stable identifier — appears in logs and in `Finding.scanner_id`
	/// when the backend is the source of truth for a finding.
	fn id(&self) -> &'static str;

	async fn run(&self, req: LlmRequest) -> Result<LlmResponse>;
}

/// Probe PATH for `claude --version`. Returns `true` only if the
/// invocation succeeds — a missing binary, non-zero exit, or any IO
/// error all read as "not available."
///
/// Cheap to call at startup. The discovery scanner needs claude
/// specifically (its MCP `--mcp-config` surface is the contract for
/// `submit_finding`); the verifier accepts either, see
/// [`build_verifier_backend`].
pub fn claude_available() -> bool {
	std::process::Command::new("claude")
		.arg("--version")
		.stdout(Stdio::null())
		.stderr(Stdio::null())
		.status()
		.map(|s| s.success())
		.unwrap_or(false)
}

/// Return true when the worker has auth material the claude CLI can
/// use without running an interactive login during a scan.
pub fn claude_auth_available() -> bool {
	env_present("ANTHROPIC_API_KEY") || home_path(".claude.json").is_some_and(|p| p.exists())
}

/// Probe PATH for `bkb-mcp` (Bitcoin Knowledge Base MCP server).
/// Returns the resolved binary path (via `which`-style lookup) when
/// available, `None` otherwise.
///
/// Optional auto-attached MCP server: when present, the discovery
/// scanner advertises bkb's `bkb_search` / `bkb_lookup_bip` /
/// `bkb_lookup_bolt` / etc. tools to the agent so it can pull spec +
/// historical context for bitcoin/lightning code that the worktree alone won't surface. See
/// [`crate::llm::claude_cli::McpContext`] for the attachment plumbing and
/// [`crate::llm::prompts::DISCOVERY`] for the conditional prompt section.
///
/// Install via `cargo install bkb-mcp`; the binary needs to reach
/// the BKB HTTP API server (default `http://127.0.0.1:3000`,
/// override with `BKB_API_URL`).
pub fn bkb_mcp_available() -> Option<PathBuf> {
	let path = std::env::var_os("PATH")?;
	for dir in std::env::split_paths(&path) {
		let candidate = dir.join("bkb-mcp");
		if candidate.is_file() {
			let ok = std::process::Command::new(&candidate)
				.arg("--help")
				.stdout(Stdio::null())
				.stderr(Stdio::null())
				.status()
				.map(|s| s.success())
				.unwrap_or(false);
			if ok {
				return Some(candidate);
			}
		}
	}
	None
}

/// Probe PATH for `codex --version`. Returns `true` only if the
/// invocation succeeds — a missing binary, non-zero exit, or any IO
/// error all read as "not available."
///
/// Cheap to call at startup. Used by [`build_verifier_backend`] to
/// pick between codex (preferred — the verifier's whole point is a
/// *cross-model* second opinion) and a claude fallback.
pub fn codex_available() -> bool {
	std::process::Command::new("codex")
		.arg("--version")
		.stdout(Stdio::null())
		.stderr(Stdio::null())
		.status()
		.map(|s| s.success())
		.unwrap_or(false)
}

/// Directory codex should read for login-state files when env-based
/// auth is not used. `CODEX_HOME` mirrors codex's own config-home
/// override; otherwise we use `~/.codex`.
pub fn codex_home_dir() -> Option<PathBuf> {
	if let Some(home) = std::env::var_os("CODEX_HOME").filter(|v| !v.is_empty()) {
		return Some(PathBuf::from(home));
	}
	home_path(".codex")
}

/// Return true when the worker has auth material the codex CLI can use
/// without running an interactive login during a scan.
pub fn codex_auth_available() -> bool {
	env_present("OPENAI_API_KEY") || codex_home_dir().is_some_and(|p| p.join("auth.json").exists())
}

fn env_present(name: &str) -> bool {
	std::env::var_os(name).is_some_and(|v| !v.is_empty())
}

fn home_path(child: &str) -> Option<PathBuf> {
	std::env::var_os("HOME").filter(|v| !v.is_empty()).map(|h| PathBuf::from(h).join(child))
}

/// Build the verifier's [`LlmBackend`]. Prefers codex (cross-model
/// second opinion is the whole point of the verifier flow); falls
/// back to claude when codex isn't installed so single-CLI hosts
/// still get *some* verifier coverage even if it's same-family.
///
/// `mcp` (optional) attaches the loupe MCP server to the backend's
/// per-call invocation. Required for the verify-mode tool surface
/// (`submit_verdict` / `submit_patch` / `validate_patch`) — without
/// MCP, the agent has no way to commit a verdict and the runner
/// would receive no feedback to POST. Production callers should
/// always pass `Some(...)`; the `None` form is kept for tests that
/// stub the backend wholesale.
///
/// Logs the choice at info level so operators can see which backend
/// is actually verifying without having to inspect process listings.
pub fn build_verifier_backend(
	mcp: Option<McpContext>, codex_ready: bool, claude_ready: bool,
) -> Result<Arc<dyn LlmBackend>> {
	if codex_ready {
		tracing::info!("verifier backend: codex (cross-model second opinion)");
		let mut backend = CodexCliBackend::new();
		if let Some(ctx) = mcp {
			backend = backend.with_mcp_context(ctx);
		}
		Ok(Arc::new(backend))
	} else if claude_ready {
		tracing::info!("verifier backend: claude (codex unavailable; same-family fallback)");
		let mut backend = ClaudeCliBackend::new();
		if let Some(ctx) = mcp {
			backend = backend.with_mcp_context(ctx);
		}
		Ok(Arc::new(backend))
	} else {
		anyhow::bail!("no authenticated verifier backend available")
	}
}

#[cfg(test)]
mod tests {
	use std::ffi::OsString;
	use std::sync::Mutex;

	use super::*;

	static ENV_LOCK: Mutex<()> = Mutex::new(());

	struct EnvGuard {
		name: &'static str,
		old: Option<OsString>,
	}

	impl EnvGuard {
		fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
			let old = std::env::var_os(name);
			std::env::set_var(name, value);
			Self { name, old }
		}

		fn unset(name: &'static str) -> Self {
			let old = std::env::var_os(name);
			std::env::remove_var(name);
			Self { name, old }
		}
	}

	impl Drop for EnvGuard {
		fn drop(&mut self) {
			if let Some(old) = &self.old {
				std::env::set_var(self.name, old);
			} else {
				std::env::remove_var(self.name);
			}
		}
	}

	#[test]
	fn provider_auth_checks_accept_api_keys() {
		let _guard = ENV_LOCK.lock().unwrap();
		let _anthropic = EnvGuard::set("ANTHROPIC_API_KEY", "anthropic-key");
		let _openai = EnvGuard::set("OPENAI_API_KEY", "openai-key");

		assert!(claude_auth_available());
		assert!(codex_auth_available());
	}

	#[test]
	fn codex_auth_checks_codex_home_auth_json() {
		let _guard = ENV_LOCK.lock().unwrap();
		let _openai = EnvGuard::unset("OPENAI_API_KEY");
		let dir = tempfile::tempdir().unwrap();
		std::fs::write(dir.path().join("auth.json"), "{}").unwrap();
		let _codex_home = EnvGuard::set("CODEX_HOME", dir.path().as_os_str());

		assert_eq!(codex_home_dir().as_deref(), Some(dir.path()));
		assert!(codex_auth_available());
	}

	#[test]
	fn rate_limit_matcher_hits_common_phrasings() {
		assert!(looks_like_rate_limit("Error: 429 Too Many Requests"));
		assert!(looks_like_rate_limit("API rate limit exceeded, retrying"));
		assert!(looks_like_rate_limit("the model is OVERLOADED right now"));
		assert!(looks_like_rate_limit("you have exceeded your monthly quota"));
		assert!(looks_like_rate_limit("You've hit your limit · resets 3:50pm (Europe/Dublin)"));
		assert!(!looks_like_rate_limit("normal model output about a buffer overflow"));
		assert!(!looks_like_rate_limit(""));
	}

	#[test]
	fn rate_limit_matcher_honours_env_extension() {
		let _guard = ENV_LOCK.lock().unwrap();
		assert!(!looks_like_rate_limit("provider says: capacity_exhausted"));
		let _markers = EnvGuard::set("LOUPE_RATE_LIMIT_MARKERS", "capacity_exhausted, slow down");
		assert!(looks_like_rate_limit("provider says: CAPACITY_EXHAUSTED"));
		assert!(looks_like_rate_limit("please Slow Down"));
		assert!(!looks_like_rate_limit("entirely unrelated text"));
	}

	#[test]
	fn parse_reset_hint_extracts_known_phrasings() {
		// Anchor "now" at a fixed wall-clock so test outcomes are
		// deterministic: 2026-05-17 12:00:00 UTC = 1779192000.
		let now: i64 = 1_779_192_000;
		// 3:50pm (15:50) is 3h50m later → target + 15min buffer.
		let r = parse_reset_hint("You've hit your limit · resets 3:50pm (Europe/Dublin)", now)
			.expect("hint should parse");
		assert_eq!(r, now + 3 * 3600 + 50 * 60 + 15 * 60);
		// 24-hour form, same minute.
		let r = parse_reset_hint("resets 15:50", now).expect("24h hint should parse");
		assert_eq!(r, now + 3 * 3600 + 50 * 60 + 15 * 60);
		// "at 3pm" with no minutes ⇒ minute = 0.
		let r = parse_reset_hint("error: resets at 3pm", now).expect("am/pm form should parse");
		assert_eq!(r, now + 3 * 3600 + 15 * 60);
	}

	#[test]
	fn parse_reset_hint_rolls_forward_past_times() {
		// now = 12:00 UTC. "resets 3am" already happened today ⇒ next
		// occurrence is +21h, plus the 15-minute safety buffer.
		let now: i64 = 1_779_192_000;
		let r = parse_reset_hint("resets 3am", now).expect("past time should roll forward");
		assert_eq!(r, now + 15 * 3600 + 15 * 60);
		// noon == 12:00; same minute as now ⇒ rolls to tomorrow.
		let r = parse_reset_hint("resets 12pm", now).expect("equal time should roll forward");
		assert_eq!(r, now + 24 * 3600 + 15 * 60);
	}

	#[test]
	fn parse_reset_hint_ignores_unknown_text() {
		let now: i64 = 1_779_192_000;
		assert!(parse_reset_hint("", now).is_none());
		assert!(parse_reset_hint("Error: 429 Too Many Requests", now).is_none());
		assert!(parse_reset_hint("resets soon", now).is_none());
		assert!(parse_reset_hint("resets 25:99", now).is_none());
		assert!(parse_reset_hint("resets 13pm", now).is_none(), "13pm is malformed");
	}

	#[test]
	fn rate_limited_from_detail_round_trips() {
		let now: i64 = 1_779_192_000;
		let rl = RateLimited::from_detail("You've hit your limit · resets 1:00pm".to_owned(), now);
		assert_eq!(rl.resume_at, Some(now + 3600 + 15 * 60));
		// No hint ⇒ resume_at is None; detail preserved verbatim.
		let rl = RateLimited::from_detail("HTTP 429: too many requests".to_owned(), now);
		assert_eq!(rl.resume_at, None);
		assert_eq!(rl.detail, "HTTP 429: too many requests");
	}

	#[test]
	fn verifier_backend_prefers_codex_then_claude() {
		let backend = build_verifier_backend(None, true, true).unwrap();
		assert_eq!(backend.id(), "codex-cli");

		let backend = build_verifier_backend(None, false, true).unwrap();
		assert_eq!(backend.id(), "claude-cli");

		let err = match build_verifier_backend(None, false, false) {
			Ok(_) => panic!("missing verifier backend should be rejected"),
			Err(e) => e,
		};
		assert!(err.to_string().contains("no authenticated verifier backend"));
	}
}

pub mod testing {
	//! Stub backend for testing scanners without invoking a real LLM
	//! CLI / API. Tests pass a closure that produces canned responses
	//! based on the request's prompt or workdir.
	//!
	//! Lives outside `#[cfg(test)]` so integration tests in sibling
	//! crates (e.g. `loupe-server/tests/llm_dispatch.rs`) can reach it.
	//! Not intended for production wiring.
	//!
	//! Two constructors:
	//! - [`StubLlmBackend::new`] takes a sync closure — simplest for
	//!   unit tests that just need a canned text response.
	//! - [`StubLlmBackend::new_async`] takes an async closure — needed
	//!   for integration tests that simulate the agent's MCP
	//!   `submit_finding` tool by POSTing to a real loupe-server
	//!   inside the closure. The agent's tool calls happen during the
	//!   session in production; the async stub gives tests the same
	//!   "while the LLM is running" hook.

	use std::future::Future;
	use std::pin::Pin;
	use std::sync::Arc;

	use anyhow::Result;
	use async_trait::async_trait;

	use super::{LlmBackend, LlmRequest, LlmResponse};

	type AsyncStubFn = Arc<
		dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<String>> + Send>> + Send + Sync,
	>;

	pub struct StubLlmBackend {
		id: &'static str,
		f: AsyncStubFn,
	}

	impl StubLlmBackend {
		/// Create a stub whose closure is sync — good for unit tests
		/// that don't need to call back into anything async.
		pub fn new<F>(id: &'static str, f: F) -> Self
		where
			F: Fn(&LlmRequest) -> Result<String> + Send + Sync + 'static,
		{
			let f = Arc::new(f);
			Self {
				id,
				f: Arc::new(move |req: LlmRequest| {
					let f = f.clone();
					Box::pin(async move { f(&req) })
				}),
			}
		}

		/// Create a stub whose closure can `.await` — used by tests
		/// that simulate the agent calling `submit_finding` mid-
		/// session against a real server fixture.
		pub fn new_async<F, Fut>(id: &'static str, f: F) -> Self
		where
			F: Fn(LlmRequest) -> Fut + Send + Sync + 'static,
			Fut: Future<Output = Result<String>> + Send + 'static,
		{
			Self { id, f: Arc::new(move |req| Box::pin(f(req))) }
		}
	}

	#[async_trait]
	impl LlmBackend for StubLlmBackend {
		fn id(&self) -> &'static str {
			self.id
		}

		async fn run(&self, req: LlmRequest) -> Result<LlmResponse> {
			let text = (self.f)(req).await?;
			Ok(LlmResponse { text, backend_id: self.id })
		}
	}
}
