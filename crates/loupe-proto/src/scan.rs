use loupe_core::{JobKind, JobState};
use serde::{Deserialize, Serialize};

/// Body of `POST /v1/repos/:id/scan` (admin).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanRequest {
	pub protocol_version: u16,
	#[serde(default)]
	pub incremental: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanResponse {
	pub protocol_version: u16,
	pub job_id: i64,
}

/// Listing entry for `GET /v1/jobs` and `GET /v1/jobs/:id`.
///
/// Notes on the `state` / `partial` / `continuation_stopped` triple:
/// a rate-limited scan is recorded as `state = Succeeded, partial =
/// true` (see `routes::jobs::complete`). That keeps the reaper and the
/// `attempts` machinery out of the row's way while the continuation
/// scheduler enqueues a child job (linked via `parent_job_id`) to
/// resume work. Clients listing jobs MUST consult `partial` (and
/// `continuation_stopped`) to distinguish a fully-finished scan from
/// one that is only superficially "succeeded" but still owes work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobInfo {
	pub job_id: i64,
	pub repo_id: i64,
	pub kind: JobKind,
	pub state: JobState,
	pub incremental: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub since_sha: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub head_sha: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parent_job_id: Option<i64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub target_finding_id: Option<i64>,
	pub attempts: u32,
	pub enqueued_at: i64,
	/// `state = Succeeded` but the run stopped early on a rate limit;
	/// a continuation is owed (and likely already enqueued as a child
	/// job — see `parent_job_id` on that child). Defaulted for wire
	/// back-compat with older servers.
	#[serde(default)]
	pub partial: bool,
	/// The continuation scheduler has given up auto-resuming this
	/// partial chain because it made no forward progress repeatedly.
	/// Operator action is required (e.g. cancel + manual rescan).
	#[serde(default)]
	pub continuation_stopped: bool,
	/// When the worker (or reaper) completed/terminated the job.
	/// Absent for `queued` / `leased` rows.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub finished_at: Option<i64>,
	/// Failure reason, when `state = Failed`.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub error: Option<String>,
}

#[cfg(test)]
mod tests {
	use loupe_core::{JobKind, JobState};

	use super::*;

	#[test]
	fn scan_request_defaults_incremental_false() {
		let req: ScanRequest = serde_json::from_str(r#"{"protocol_version":1}"#).unwrap();
		assert!(!req.incremental);
	}

	#[test]
	fn job_info_round_trips() {
		let info = JobInfo {
			job_id: 1,
			repo_id: 2,
			kind: JobKind::Scan,
			state: JobState::Queued,
			incremental: false,
			since_sha: None,
			head_sha: None,
			parent_job_id: None,
			target_finding_id: None,
			attempts: 0,
			enqueued_at: 1_700_000_000,
			partial: false,
			continuation_stopped: false,
			finished_at: None,
			error: None,
		};
		let s = serde_json::to_string(&info).unwrap();
		let back: JobInfo = serde_json::from_str(&s).unwrap();
		assert_eq!(info, back);
	}

	#[test]
	fn verify_job_carries_parentage() {
		let info = JobInfo {
			job_id: 5,
			repo_id: 2,
			kind: JobKind::Verify,
			state: JobState::Leased,
			incremental: false,
			since_sha: None,
			head_sha: Some("abc123".into()),
			parent_job_id: Some(1),
			target_finding_id: Some(42),
			attempts: 0,
			enqueued_at: 1_700_000_100,
			partial: false,
			continuation_stopped: false,
			finished_at: None,
			error: None,
		};
		let s = serde_json::to_string(&info).unwrap();
		let back: JobInfo = serde_json::from_str(&s).unwrap();
		assert_eq!(info, back);
	}

	#[test]
	fn partial_flags_round_trip_and_default_for_old_clients() {
		// Round-trip a partial-state row.
		let info = JobInfo {
			job_id: 39,
			repo_id: 1,
			kind: JobKind::Scan,
			state: JobState::Succeeded,
			incremental: false,
			since_sha: None,
			head_sha: Some("deadbeef".into()),
			parent_job_id: None,
			target_finding_id: None,
			attempts: 1,
			enqueued_at: 1_700_000_000,
			partial: true,
			continuation_stopped: false,
			finished_at: Some(1_700_000_500),
			error: None,
		};
		let s = serde_json::to_string(&info).unwrap();
		assert!(s.contains("\"partial\":true"), "wire form must include partial flag: {s}");
		let back: JobInfo = serde_json::from_str(&s).unwrap();
		assert_eq!(info, back);

		// An older server's payload (no partial/continuation_stopped/
		// finished_at/error) must still deserialize, defaulting the
		// new fields to false/None.
		let legacy = r#"{
			"job_id": 7,
			"repo_id": 2,
			"kind": "scan",
			"state": "queued",
			"incremental": false,
			"attempts": 0,
			"enqueued_at": 1700000000
		}"#;
		let parsed: JobInfo = serde_json::from_str(legacy).unwrap();
		assert!(!parsed.partial);
		assert!(!parsed.continuation_stopped);
		assert!(parsed.finished_at.is_none());
		assert!(parsed.error.is_none());
	}
}
