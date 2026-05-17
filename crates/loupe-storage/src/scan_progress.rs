//! DAO for the `scan_file_progress` table — per-`(repo, commit, file)`
//! "this file was fully scanned" markers that let a rate-limited scan
//! resume without re-doing finished files.
//!
//! Writes are idempotent (`INSERT OR IGNORE` on the
//! `UNIQUE(repo_id, commit_sha, file_path)` key) so a continuation that
//! re-reports an already-recorded file is harmless, and so is a crash
//! that replays the same job.

use rusqlite::{params, Connection};

/// Record that `file_path` (repo-relative, `/`-form) has been fully
/// scanned at `commit_sha`. Idempotent: a duplicate is silently
/// ignored. Returns `true` if a new row was inserted (the file had not
/// been recorded yet), `false` if it was already present.
pub fn mark_scanned(
	conn: &Connection, repo_id: i64, commit_sha: &str, file_path: &str, job_id: i64, now: i64,
) -> rusqlite::Result<bool> {
	let n = conn.execute(
		"INSERT OR IGNORE INTO scan_file_progress
		   (repo_id, commit_sha, file_path, scanned_at, job_id)
		 VALUES (?1, ?2, ?3, ?4, ?5)",
		params![repo_id, commit_sha, file_path, now, job_id],
	)?;
	Ok(n > 0)
}

/// The repo-relative paths already scanned for `(repo_id, commit_sha)`.
/// A continuation filters these out of its file walk before fanning
/// out per-file agent sessions.
pub fn scanned_files(
	conn: &Connection, repo_id: i64, commit_sha: &str,
) -> rusqlite::Result<Vec<String>> {
	let mut stmt = conn.prepare(
		"SELECT file_path FROM scan_file_progress
		 WHERE repo_id = ?1 AND commit_sha = ?2
		 ORDER BY file_path ASC",
	)?;
	let rows = stmt.query_map(params![repo_id, commit_sha], |r| r.get::<_, String>(0))?;
	rows.collect()
}

/// How many files a given job recorded as scanned. `0` means that run
/// made zero forward progress — the signal the scheduler's
/// zero-progress guard counts to decide a continuation chain is stuck
/// on rate limits and must stop (loudly) rather than churn forever.
pub fn count_files_for_job(conn: &Connection, job_id: i64) -> rusqlite::Result<i64> {
	conn.query_row(
		"SELECT COUNT(*) FROM scan_file_progress WHERE job_id = ?1",
		params![job_id],
		|r| r.get(0),
	)
}

/// Drop all progress rows for a repo. Called once a scan covers a
/// commit completely (`jobs.partial = 0` Succeeded): the markers have
/// done their job and keeping them would let the table grow without
/// bound across the repo's lifetime.
pub fn prune_repo(conn: &Connection, repo_id: i64) -> rusqlite::Result<usize> {
	conn.execute("DELETE FROM scan_file_progress WHERE repo_id = ?1", params![repo_id])
}

#[cfg(test)]
mod tests {
	use loupe_core::ReportingDestination;

	use super::*;
	use crate::jobs::{self, NewJob};
	use crate::repos::{self, NewRepo};
	use crate::secrets::{self, SecretKind};
	use crate::Db;

	fn fixture() -> (Db, i64, i64) {
		let db = Db::open_in_memory(&crate::secrets::MasterKey::for_tests()).unwrap();
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
						scan_interval_seconds: None,
						scanner_config: serde_json::Value::Null,
						reporting: ReportingDestination::GithubIssue {
							target_owner: "a".into(),
							target_repo: "t".into(),
							pat_secret_id: secret_id,
						},
						verification_enabled: false,
						require_approval: None,
					},
					0,
				)?)
			})
			.unwrap();
		let job_id = db
			.with_conn(|c| {
				Ok(jobs::enqueue(
					c,
					&NewJob {
						repo_id,
						kind: loupe_core::JobKind::Scan,
						incremental: false,
						since_sha: None,
						parent_job_id: None,
						target_finding_id: None,
					},
					0,
				)?)
			})
			.unwrap();
		(db, repo_id, job_id)
	}

	#[test]
	fn mark_and_query_round_trip() {
		let (db, repo_id, job_id) = fixture();
		let first = db
			.with_conn(|c| Ok(mark_scanned(c, repo_id, "sha1", "src/a.rs", job_id, 10)?))
			.unwrap();
		assert!(first, "first insert is a new row");
		db.with_conn(|c| Ok(mark_scanned(c, repo_id, "sha1", "src/b.rs", job_id, 11)?)).unwrap();

		let files = db.with_conn(|c| Ok(scanned_files(c, repo_id, "sha1")?)).unwrap();
		assert_eq!(files, vec!["src/a.rs".to_string(), "src/b.rs".to_string()]);

		// A different commit doesn't see sha1's rows.
		assert!(db.with_conn(|c| Ok(scanned_files(c, repo_id, "sha2")?)).unwrap().is_empty());
	}

	#[test]
	fn mark_scanned_is_idempotent() {
		let (db, repo_id, job_id) = fixture();
		let first = db
			.with_conn(|c| Ok(mark_scanned(c, repo_id, "sha1", "src/a.rs", job_id, 10)?))
			.unwrap();
		let second = db
			.with_conn(|c| Ok(mark_scanned(c, repo_id, "sha1", "src/a.rs", job_id, 99)?))
			.unwrap();
		assert!(first);
		assert!(!second, "re-recording the same file is ignored, not an error");
		assert_eq!(db.with_conn(|c| Ok(scanned_files(c, repo_id, "sha1")?)).unwrap().len(), 1);
	}

	#[test]
	fn count_files_for_job_drives_zero_progress_guard() {
		let (db, repo_id, job_id) = fixture();
		assert_eq!(db.with_conn(|c| Ok(count_files_for_job(c, job_id)?)).unwrap(), 0);
		db.with_conn(|c| Ok(mark_scanned(c, repo_id, "sha1", "src/a.rs", job_id, 10)?)).unwrap();
		db.with_conn(|c| Ok(mark_scanned(c, repo_id, "sha1", "src/b.rs", job_id, 11)?)).unwrap();
		assert_eq!(db.with_conn(|c| Ok(count_files_for_job(c, job_id)?)).unwrap(), 2);
	}

	#[test]
	fn prune_repo_clears_all_rows() {
		let (db, repo_id, job_id) = fixture();
		db.with_conn(|c| Ok(mark_scanned(c, repo_id, "sha1", "src/a.rs", job_id, 10)?)).unwrap();
		db.with_conn(|c| Ok(mark_scanned(c, repo_id, "sha2", "src/b.rs", job_id, 11)?)).unwrap();
		let pruned = db.with_conn(|c| Ok(prune_repo(c, repo_id)?)).unwrap();
		assert_eq!(pruned, 2);
		assert!(db.with_conn(|c| Ok(scanned_files(c, repo_id, "sha1")?)).unwrap().is_empty());
		assert!(db.with_conn(|c| Ok(scanned_files(c, repo_id, "sha2")?)).unwrap().is_empty());
	}

	#[test]
	fn progress_survives_originating_job_deletion() {
		// job_id is ON DELETE SET NULL, not CASCADE: a reaped job must
		// not erase the proof that its files were scanned, or a
		// continuation would redo the work.
		let (db, repo_id, job_id) = fixture();
		db.with_conn(|c| Ok(mark_scanned(c, repo_id, "sha1", "src/a.rs", job_id, 10)?)).unwrap();
		db.with_conn(|c| {
			c.execute("DELETE FROM jobs WHERE id = ?1", [job_id])?;
			Ok(())
		})
		.unwrap();
		let files = db.with_conn(|c| Ok(scanned_files(c, repo_id, "sha1")?)).unwrap();
		assert_eq!(files, vec!["src/a.rs".to_string()], "progress must outlive its job");
	}
}
