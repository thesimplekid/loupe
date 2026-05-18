//! Background tasks owned by the server: the scheduler that enqueues
//! due scans, and the reaper that reclaims expired job leases.
//!
//! Both run on the same tokio runtime as the axum handlers and shut
//! down via the same `CancellationToken` the serve loop uses, so the
//! server's `shutdown()` cleans them up too.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use loupe_core::{JobKind, JobState};
use loupe_storage::jobs::{self, JobRow, NewJob};
use loupe_storage::{findings, repos, scan_progress, Db};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

/// How often the scheduler checks for due repos. Operators with very
/// short scan intervals can shorten this; the default is fine for the
/// common case (intervals measured in minutes or hours).
pub const SCHEDULER_TICK: Duration = Duration::from_secs(30);

/// How often the reaper runs. Should be a small fraction of the lease
/// TTL so a stuck worker is reclaimed promptly.
pub const REAPER_TICK: Duration = Duration::from_secs(15);

/// Rate-limit cool-off floor before a partial scan is auto-continued.
/// Long enough that a transient per-minute throttle has a chance to
/// refresh. For long lockouts (e.g. Claude Code's 5-hour session
/// reset), the worker-supplied `jobs.resume_not_before` hint takes
/// precedence — see [`loupe_storage::jobs::resumable_partials`] for
/// the `MAX(finished_at + backoff, resume_not_before)` semantics. The
/// scheduler tick + the active-scan guard keep continuations from
/// piling up regardless of this value.
pub const DEFAULT_CONTINUATION_BACKOFF_SECS: i64 = 900;

/// Cap on *consecutive* continuations that each made zero forward
/// progress. Past this the scheduler stops auto-resuming the chain and
/// surfaces a loud, terminal `loupe::scan_stalled` signal instead of
/// churning forever — i.e. "the budget is too small for this repo at
/// the current rate" fails honestly rather than invisibly. A
/// continuation that scans ≥1 new file resets the streak.
///
/// Set to comfortably outlast Claude Code's 5-hour session lockout:
/// with the 15-minute backoff floor and the typical `resume_not_before`
/// hint deferring most retries until *after* the lockout closes, a
/// run that genuinely makes no progress for this many cycles is a
/// real stall worth alerting on — not just an unlucky multi-hour
/// window. Operators wanting tighter / looser behaviour should treat
/// this as a knob to tune per deployment.
pub const MAX_ZERO_PROGRESS_CONTINUATIONS: u32 = 24;

fn now_secs() -> i64 {
	SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
}

/// Scheduler tick: enqueue a scan job for each repo whose interval has
/// elapsed since `last_scanned_at`. Returns the number of jobs
/// enqueued. Exposed so tests can drive a tick directly without waiting
/// for a real timer.
pub fn schedule_due(db: &Db, now: i64) -> anyhow::Result<usize> {
	let due = db.with_conn(|c| Ok(repos::list_due_for_scan(c, now)?))?;
	let mut enqueued = 0;
	for repo in due {
		let since_sha = repo.last_scanned_sha.clone();
		// If a queued or leased scan already exists for this repo, skip
		// — we don't want to pile up duplicates.
		let pending: i64 = db.with_conn(|c| Ok(jobs::count_active_scans_for_repo(c, repo.id)?))?;
		if pending > 0 {
			tracing::debug!(repo = %repo.clone_url, "skipping due repo: prior scan still in flight");
			continue;
		}
		db.with_conn(|c| {
			Ok(jobs::enqueue(
				c,
				&NewJob {
					repo_id: repo.id,
					kind: JobKind::Scan,
					incremental: since_sha.is_some(),
					since_sha,
					head_sha: None,
					parent_job_id: None,
					target_finding_id: None,
				},
				now,
			)?)
		})?;
		enqueued += 1;
		tracing::info!(repo = %repo.clone_url, "scheduler enqueued periodic scan");
	}
	Ok(enqueued)
}

/// Scheduler tick: enqueue a continuation for each repo owed one
/// (its most-recent scan is a cooled-off succeeded-partial), unless the
/// chain has stalled — see [`MAX_ZERO_PROGRESS_CONTINUATIONS`]. Returns
/// the number of continuations enqueued. Driven from the same tick as
/// [`schedule_due`]; exposed for direct test invocation.
pub fn schedule_continuations(db: &Db, now: i64) -> anyhow::Result<usize> {
	let partials =
		db.with_conn(|c| Ok(jobs::resumable_partials(c, now, DEFAULT_CONTINUATION_BACKOFF_SECS)?))?;
	let mut enqueued = 0;
	for partial in partials {
		// Belt-and-suspenders against the periodic scheduler racing a
		// fresh scan into the same repo within a tick (it runs first).
		let active: i64 =
			db.with_conn(|c| Ok(jobs::count_active_scans_for_repo(c, partial.repo_id)?))?;
		if active > 0 {
			continue;
		}

		let streak = consecutive_zero_progress_streak(db, &partial)?;
		if streak >= MAX_ZERO_PROGRESS_CONTINUATIONS {
			// Stop the chain. Loud (stable, alertable target) and
			// terminal — mark the latest partial so future scheduler
			// ticks don't re-emit the same stall. The partial jobs and
			// their real findings stay untouched. A later manual or
			// periodic scan (fresh chain, parent_job_id = NULL) can
			// still retry once the budget situation changes.
			tracing::error!(
				target: "loupe::scan_stalled",
				repo_id = partial.repo_id,
				partial_job_id = partial.id,
				streak,
				"scan stalled: rate limit prevented forward progress after \
				 {streak} continuations; stopping auto-resume",
			);
			let note = format!(
				"scan stalled: rate limit prevented forward progress after {streak} \
				 continuations; auto-resume stopped"
			);
			db.with_conn(|c| {
				jobs::stop_continuation(c, partial.id)?;
				c.execute(
					"INSERT INTO scan_history
					   (repo_id, job_id, head_sha, base_sha, finding_count,
					    duration_ms, finished_at, note)
					 SELECT ?1, ?2, ?3, ?4,
					        (SELECT COUNT(*) FROM findings WHERE job_id = ?2),
					        0, ?5, ?6",
					(
						partial.repo_id,
						partial.id,
						partial.head_sha.clone().unwrap_or_default(),
						partial.since_sha.clone(),
						now,
						note,
					),
				)?;
				Ok(())
			})?;
			continue;
		}

		// Resume where it left off: the worker keys "already scanned"
		// off `(repo_id, head_sha)`, so the continuation just needs the
		// same scan shape. `parent_job_id` threads the chain for the
		// zero-progress guard above.
		db.with_conn(|c| {
			Ok(jobs::enqueue(
				c,
				&NewJob {
					repo_id: partial.repo_id,
					kind: JobKind::Scan,
					incremental: partial.incremental,
					since_sha: partial.since_sha.clone(),
					head_sha: None,
					parent_job_id: Some(partial.id),
					target_finding_id: None,
				},
				now,
			)?)
		})?;
		enqueued += 1;
		tracing::info!(
			repo_id = partial.repo_id,
			parent_job_id = partial.id,
			"scheduler enqueued rate-limit continuation",
		);
	}
	Ok(enqueued)
}

/// Length of the consecutive tail of zero-progress partial scans
/// ending at `latest`, walking the `parent_job_id` chain. A job that
/// recorded ≥1 scanned file (forward progress) ends the streak; so
/// does a non-partial / non-scan link or the chain's root.
fn consecutive_zero_progress_streak(db: &Db, latest: &JobRow) -> anyhow::Result<u32> {
	let mut streak = 0u32;
	let mut cur = Some(latest.clone());
	while let Some(job) = cur {
		if job.kind != JobKind::Scan || job.state != JobState::Succeeded || !job.partial {
			break;
		}
		let files = db.with_conn(|c| Ok(scan_progress::count_files_for_job(c, job.id)?))?;
		if files > 0 {
			break; // forward progress — the streak resets here
		}
		streak += 1;
		cur = match job.parent_job_id {
			Some(pid) => db.with_conn(|c| Ok(jobs::get(c, pid)?))?,
			None => None,
		};
	}
	Ok(streak)
}

/// Reaper tick: reclaim leases past their TTL. Re-queue if attempts <
/// MAX, fail otherwise. Wraps `loupe-storage::jobs::reap_stale_leases`.
pub fn reap_once(db: &Db, now: i64) -> anyhow::Result<usize> {
	let n = db.with_conn(|c| Ok(jobs::reap_stale_leases(c, now)?))?;
	if n > 0 {
		tracing::info!(reclaimed = n, "reaper transitioned stale leases");
	}
	// Same tick: dismiss findings whose validating_deadline has elapsed.
	// Stale validating findings sit invisible to the dispatcher (state
	// is 'validating', not 'confirmed') so without a reaper they'd
	// never escape their own state.
	let dismissed = db.with_conn(|c| Ok(findings::reap_stale_validating(c, now)?))?;
	if dismissed > 0 {
		tracing::info!(dismissed, "reaper dismissed stale validating findings");
	}
	Ok(n + dismissed)
}

/// Spawn the scheduler. Returns a JoinHandle so the caller can wait on
/// it during shutdown. Cancels cleanly when `cancel` fires. Pokes
/// `job_arrived` whenever it enqueues something so long-polling
/// workers wake immediately.
pub fn spawn_scheduler(
	db: std::sync::Arc<Db>, job_arrived: std::sync::Arc<Notify>, cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
	tokio::spawn(async move {
		let mut interval = tokio::time::interval(SCHEDULER_TICK);
		interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
		loop {
			tokio::select! {
				_ = cancel.cancelled() => return,
				_ = interval.tick() => {
					let now = now_secs();
					let mut total = 0usize;
					match schedule_due(&db, now) {
						Ok(n) => total += n,
						Err(e) => tracing::warn!(error = %e, "scheduler tick failed"),
					}
					// Same tick handles rate-limit continuations. Kept
					// separate from schedule_due so a failure in one
					// doesn't suppress the other.
					match schedule_continuations(&db, now) {
						Ok(n) => total += n,
						Err(e) => {
							tracing::warn!(error = %e, "continuation scheduler tick failed")
						},
					}
					if total > 0 {
						job_arrived.notify_waiters();
					}
				}
			}
		}
	})
}

/// Spawn the reaper. Same shape as [`spawn_scheduler`].
pub fn spawn_reaper(
	db: std::sync::Arc<Db>, cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
	tokio::spawn(async move {
		let mut interval = tokio::time::interval(REAPER_TICK);
		interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
		loop {
			tokio::select! {
				_ = cancel.cancelled() => return,
				_ = interval.tick() => {
					if let Err(e) = reap_once(&db, now_secs()) {
						tracing::warn!(error = %e, "reaper tick failed");
					}
				}
			}
		}
	})
}

#[cfg(test)]
mod tests {
	use loupe_core::ReportingDestination;
	use loupe_storage::repos::NewRepo;
	use loupe_storage::secrets::{self, SecretKind};
	use loupe_storage::workers::{self, WorkerKind};

	use super::*;

	fn fixture() -> (std::sync::Arc<Db>, i64, i64) {
		let db = std::sync::Arc::new(
			Db::open_in_memory(&loupe_storage::secrets::MasterKey::for_tests()).unwrap(),
		);
		let secret_id =
			db.with_conn(|c| Ok(secrets::insert(c, SecretKind::GithubPat, "p", b"x", 0)?)).unwrap();
		let repo_id = db
			.with_conn(|c| {
				Ok(repos::insert(
					c,
					&NewRepo {
						clone_url: "https://github.com/a/b.git".into(),
						host: "github.com".into(),
						owner: "a".into(),
						repo: "b".into(),
						default_branch: None,
						scan_interval_seconds: Some(60),
						scanner_config: serde_json::Value::Null,
						reporting: ReportingDestination::GithubIssue {
							target_owner: "x".into(),
							target_repo: "y".into(),
							pat_secret_id: secret_id,
						},
						verification_enabled: false,
						require_approval: None,
					},
					0,
				)?)
			})
			.unwrap();
		let worker_id = db
			.with_conn(|c| Ok(workers::insert(c, "w1", WorkerKind::Worker, &[1u8; 32], 0)?))
			.unwrap();
		(db, repo_id, worker_id)
	}

	#[test]
	fn scheduler_enqueues_due_repos() {
		let (db, _repo_id, _) = fixture();
		// Repo just inserted, never scanned ⇒ due immediately.
		let n = schedule_due(&db, 1_000).unwrap();
		assert_eq!(n, 1);
		// A second tick must not double-enqueue (in-flight job).
		let n = schedule_due(&db, 1_000).unwrap();
		assert_eq!(n, 0);
	}

	/// Insert a succeeded-partial scan job directly (bypassing the
	/// lease cycle) so chain/streak tests stay terse. Returns the job
	/// id. `finished_at = 0` ⇒ always past the backoff for any
	/// positive `now`.
	fn insert_partial_scan(db: &Db, repo_id: i64, parent: Option<i64>, enqueued_at: i64) -> i64 {
		db.with_conn(|c| {
			c.execute(
				"INSERT INTO jobs
				   (repo_id, kind, state, incremental, parent_job_id,
				    enqueued_at, finished_at, partial)
				 VALUES (?1, 'scan', 'succeeded', 0, ?2, ?3, 0, 1)",
				(repo_id, parent, enqueued_at),
			)?;
			Ok(c.last_insert_rowid())
		})
		.unwrap()
	}

	#[test]
	fn continuation_scheduled_only_after_backoff() {
		let (db, repo_id, worker_id) = fixture();
		let job_id = db
			.with_conn(|c| {
				Ok(jobs::enqueue(
					c,
					&NewJob {
						repo_id,
						kind: JobKind::Scan,
						incremental: false,
						since_sha: None,
						head_sha: None,
						parent_job_id: None,
						target_finding_id: None,
					},
					0,
				)?)
			})
			.unwrap();
		db.with_conn(|c| Ok(jobs::lease_next(c, worker_id, false, 10, 600)?)).unwrap();
		// Complete as partial, finished at t=1000.
		db.with_conn(|c| {
			Ok(jobs::complete(
				c,
				job_id,
				worker_id,
				JobState::Succeeded,
				true,
				Some("sha1"),
				None,
				None,
				1_000,
			)?)
		})
		.unwrap();

		// Backoff not yet elapsed (finished_at + 900 = 1900).
		assert_eq!(schedule_continuations(&db, 1_899).unwrap(), 0);
		// Elapsed ⇒ one continuation, parented to the partial.
		assert_eq!(schedule_continuations(&db, 1_900).unwrap(), 1);
		// Now a queued scan exists ⇒ the same partial is no longer the
		// most-recent scan job and the active-scan guard trips: no pile-up.
		assert_eq!(schedule_continuations(&db, 5_000).unwrap(), 0);
		let cont = db.with_conn(|c| Ok(jobs::list(c)?)).unwrap();
		let child = cont.iter().find(|j| j.parent_job_id == Some(job_id)).expect("continuation");
		assert_eq!(child.state, JobState::Queued);
		assert!(child.since_sha.is_none());
	}

	#[test]
	fn continuation_respects_resume_not_before() {
		// The worker parsed a "resets <time>" hint and reported
		// `resume_not_before` well past the static backoff. The
		// scheduler must wait until that hint elapses — even though
		// `finished_at + backoff` has already passed — so a multi-
		// hour Claude Code session lockout doesn't burn zero-progress
		// continuation slots inside the lockout window.
		let (db, repo_id, worker_id) = fixture();
		let job_id = db
			.with_conn(|c| {
				Ok(jobs::enqueue(
					c,
					&NewJob {
						repo_id,
						kind: JobKind::Scan,
						incremental: false,
						since_sha: None,
						head_sha: None,
						parent_job_id: None,
						target_finding_id: None,
					},
					0,
				)?)
			})
			.unwrap();
		db.with_conn(|c| Ok(jobs::lease_next(c, worker_id, false, 10, 600)?)).unwrap();
		// finished_at = 1_000, backoff = 900 ⇒ static cool-off elapsed
		// at t = 1_900. But the worker says "resume not before 20_000"
		// (≈5h later, mirroring a Claude session lockout).
		db.with_conn(|c| {
			Ok(jobs::complete(
				c,
				job_id,
				worker_id,
				JobState::Succeeded,
				true,
				Some("sha1"),
				None,
				Some(20_000),
				1_000,
			)?)
		})
		.unwrap();

		// Static backoff says go, but the hint hasn't elapsed yet.
		assert_eq!(schedule_continuations(&db, 1_900).unwrap(), 0);
		assert_eq!(schedule_continuations(&db, 19_999).unwrap(), 0);
		// Hint elapsed ⇒ one continuation, parented to the partial.
		assert_eq!(schedule_continuations(&db, 20_000).unwrap(), 1);
	}

	#[test]
	fn zero_progress_chain_stalls_loudly_at_cap() {
		let (db, repo_id, _) = fixture();
		// A chain of exactly MAX_ZERO_PROGRESS_CONTINUATIONS partial
		// scans, none of which recorded any scanned file.
		let mut parent = None;
		for i in 0..MAX_ZERO_PROGRESS_CONTINUATIONS as i64 {
			parent = Some(insert_partial_scan(&db, repo_id, parent, i));
		}
		// Streak == cap ⇒ no continuation; a stalled scan_history note.
		assert_eq!(schedule_continuations(&db, 10_000).unwrap(), 0);
		let (n, note): (i64, Option<String>) = db
			.with_conn(|c| {
				Ok(c.query_row(
					"SELECT COUNT(*), MAX(note) FROM scan_history WHERE repo_id = ?1",
					[repo_id],
					|r| Ok((r.get(0)?, r.get(1)?)),
				)?)
			})
			.unwrap();
		assert_eq!(n, 1, "exactly one stalled history row");
		assert!(
			note.unwrap_or_default().contains("scan stalled"),
			"history note must record the stall"
		);
		// No new scan job was created.
		let scans = db
			.with_conn(|c| Ok(jobs::list(c)?))
			.unwrap()
			.into_iter()
			.filter(|j| j.kind == JobKind::Scan)
			.count();
		assert_eq!(scans, MAX_ZERO_PROGRESS_CONTINUATIONS as usize);
		// A later scheduler tick must stay quiet: the latest partial was
		// durably marked as stopped, so the same terminal condition is not
		// logged or inserted again.
		assert_eq!(schedule_continuations(&db, 20_000).unwrap(), 0);
		let n_after: i64 = db
			.with_conn(|c| {
				Ok(c.query_row(
					"SELECT COUNT(*) FROM scan_history WHERE repo_id = ?1",
					[repo_id],
					|r| r.get(0),
				)?)
			})
			.unwrap();
		assert_eq!(n_after, 1, "stall history row must be one-shot");
	}

	#[test]
	fn forward_progress_resets_the_zero_streak() {
		let (db, repo_id, _) = fixture();
		// Long zero chain, but the most-recent partial scanned a file.
		let mut parent = None;
		for i in 0..(MAX_ZERO_PROGRESS_CONTINUATIONS as i64 + 3) {
			parent = Some(insert_partial_scan(&db, repo_id, parent, i));
		}
		let latest = parent.unwrap();
		db.with_conn(|c| {
			Ok(scan_progress::mark_scanned(c, repo_id, "sha1", "src/a.rs", latest, 1)?)
		})
		.unwrap();
		// Streak resets at the latest job ⇒ a continuation is owed.
		assert_eq!(schedule_continuations(&db, 10_000).unwrap(), 1);
	}

	#[test]
	fn reaper_reclaims_stale_leases() {
		let (db, repo_id, worker_id) = fixture();
		db.with_conn(|c| {
			Ok(jobs::enqueue(
				c,
				&NewJob {
					repo_id,
					kind: JobKind::Scan,
					incremental: false,
					since_sha: None,
					head_sha: None,
					parent_job_id: None,
					target_finding_id: None,
				},
				0,
			)?)
		})
		.unwrap();
		// Lease at t=100 with TTL=10. Reap at t=200 ⇒ requeue.
		db.with_conn(|c| Ok(jobs::lease_next(c, worker_id, false, 100, 10)?)).unwrap();
		let n = reap_once(&db, 200).unwrap();
		assert_eq!(n, 1);
	}
}
