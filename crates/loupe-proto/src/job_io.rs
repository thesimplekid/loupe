use loupe_core::{Finding, Verdict};
use serde::{Deserialize, Serialize};

/// Body of `POST /v1/jobs/:id/heartbeat` (worker, lease holder).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatRequest {
	pub protocol_version: u16,
}

/// Response body of `POST /v1/jobs/:id/heartbeat`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatResponse {
	pub protocol_version: u16,
	pub lease_expires_at: i64,
}

/// Body of `POST /v1/jobs/:id/findings` (worker, scan-kind only). The
/// server rejects calls from a verify-kind job at the route layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FindingsBatch {
	pub protocol_version: u16,
	pub findings: Vec<Finding>,
}

/// Body of `POST /v1/jobs/:id/scan-progress` (worker, scan-kind only).
/// Reports source files the worker has finished scanning at
/// `commit_sha`, so a later continuation can skip them. Sent
/// incrementally as files complete — best-effort, never blocks the
/// scan. The server keys progress by `(repo_id, commit_sha, file_path)`;
/// `repo_id` is taken from the job, not the body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanProgressReport {
	pub protocol_version: u16,
	pub commit_sha: String,
	pub files: Vec<String>,
}

/// Response body of `GET /v1/repos/:repo_id/scan-progress?commit=…`.
/// The repo-relative paths already scanned at that commit, so the
/// worker can filter them out of the walk before fanning out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanProgressList {
	pub protocol_version: u16,
	pub files: Vec<String>,
}

/// Body of `POST /v1/jobs/:id/retry` (admin). Empty today apart from
/// the protocol guard; kept as a DTO so the route can grow options
/// without changing shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryJobRequest {
	pub protocol_version: u16,
}

/// Body of `POST /v1/jobs/:id/verdict` (worker, verify-kind only). One
/// verdict per verify job — that's the entire reason to split the
/// endpoint from `findings`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerdictSubmission {
	pub protocol_version: u16,
	pub verdict: Verdict,
}

/// Body of `POST /v1/jobs/:id/complete`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompleteRequest {
	pub protocol_version: u16,
	pub outcome: CompleteOutcome,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub head_sha: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub error: Option<String>,
	/// Earliest Unix-seconds instant the server scheduler should
	/// consider this job for an auto-continuation. Only meaningful
	/// with `outcome = Partial`. Populated from the LLM backend's
	/// parsed "resets &lt;time&gt;" hint so a multi-hour session
	/// lockout doesn't burn zero-progress continuation slots inside
	/// the lockout window. `None` falls back to the server's static
	/// backoff. Backward-compatible: older workers omit the field
	/// and the server treats `null` identically.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub resume_not_before: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompleteOutcome {
	Succeeded,
	Failed,
	/// The scan stopped early on a provider rate limit. Findings
	/// already submitted are real and kept; the commit is *not* fully
	/// covered, so the server records the job as succeeded-but-partial
	/// (does not advance `last_scanned_sha`) and schedules a
	/// continuation that skips the files already reported done.
	Partial,
}

#[cfg(test)]
mod tests {
	use loupe_core::{Severity, Verdict};

	use super::*;
	use crate::version::PROTOCOL_VERSION;

	#[test]
	fn findings_batch_round_trips() {
		let batch = FindingsBatch {
			protocol_version: PROTOCOL_VERSION,
			findings: vec![Finding {
				scanner_id: "x".into(),
				severity: Severity::Low,
				title: "t".into(),
				description: "d".into(),
				file_path: None,
				line_start: None,
				line_end: None,
				cwe: None,
				patch_unified: None,
				poc_unified: None,
				fingerprint: "fp".into(),
			}],
		};
		let s = serde_json::to_string(&batch).unwrap();
		let back: FindingsBatch = serde_json::from_str(&s).unwrap();
		assert_eq!(batch, back);
	}

	#[test]
	fn verdict_submission_round_trips() {
		let v = VerdictSubmission {
			protocol_version: PROTOCOL_VERSION,
			verdict: Verdict::Confirmed { notes: Some("matches".into()), patch: None },
		};
		let s = serde_json::to_string(&v).unwrap();
		let back: VerdictSubmission = serde_json::from_str(&s).unwrap();
		assert_eq!(v, back);
	}

	#[test]
	fn heartbeat_request_round_trips() {
		let req = HeartbeatRequest { protocol_version: PROTOCOL_VERSION };
		let s = serde_json::to_string(&req).unwrap();
		let back: HeartbeatRequest = serde_json::from_str(&s).unwrap();
		assert_eq!(req, back);
	}

	#[test]
	fn complete_outcome_serializes_lowercase() {
		let req = CompleteRequest {
			protocol_version: PROTOCOL_VERSION,
			outcome: CompleteOutcome::Succeeded,
			head_sha: Some("abc".into()),
			error: None,
			resume_not_before: None,
		};
		let s = serde_json::to_string(&req).unwrap();
		assert!(s.contains(r#""outcome":"succeeded""#), "got: {s}");
		assert!(!s.contains("resume_not_before"), "absent field must serialize as omitted");
		let back: CompleteRequest = serde_json::from_str(&s).unwrap();
		assert_eq!(req, back);
	}

	#[test]
	fn complete_outcome_partial_round_trips() {
		let req = CompleteRequest {
			protocol_version: PROTOCOL_VERSION,
			outcome: CompleteOutcome::Partial,
			head_sha: Some("def".into()),
			error: None,
			resume_not_before: Some(1_779_192_000),
		};
		let s = serde_json::to_string(&req).unwrap();
		assert!(s.contains(r#""outcome":"partial""#), "got: {s}");
		assert!(s.contains(r#""resume_not_before":1779192000"#), "got: {s}");
		let back: CompleteRequest = serde_json::from_str(&s).unwrap();
		assert_eq!(req, back);
	}

	#[test]
	fn complete_request_back_compat_no_resume_field() {
		// Older workers (pre-resume-hint) won't send the field.
		// The server must still accept the body verbatim.
		let json = r#"{"protocol_version":2,"outcome":"partial","head_sha":"abc"}"#;
		let back: CompleteRequest = serde_json::from_str(json).unwrap();
		assert_eq!(back.outcome, CompleteOutcome::Partial);
		assert_eq!(back.resume_not_before, None);
	}

	#[test]
	fn scan_progress_messages_round_trip() {
		let report = ScanProgressReport {
			protocol_version: PROTOCOL_VERSION,
			commit_sha: "abc123".into(),
			files: vec!["src/a.rs".into(), "src/b.rs".into()],
		};
		let s = serde_json::to_string(&report).unwrap();
		let back: ScanProgressReport = serde_json::from_str(&s).unwrap();
		assert_eq!(report, back);

		let list =
			ScanProgressList { protocol_version: PROTOCOL_VERSION, files: vec!["src/a.rs".into()] };
		let s = serde_json::to_string(&list).unwrap();
		let back: ScanProgressList = serde_json::from_str(&s).unwrap();
		assert_eq!(list, back);
	}
}
