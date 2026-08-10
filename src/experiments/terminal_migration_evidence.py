from __future__ import annotations

"""Strict scheduler evidence gates for terminal migration and runtime probing."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

MIGRATION_JOB_EVIDENCE_SCHEMA = "terminal_base_migration_job_evidence_v2"
MIGRATION_SCHEDULER_GATE_SCHEMA = "terminal_base_migration_scheduler_gate_v2"
RUNTIME_PROFILE_SCHEMA = "terminal_base_migration_runtime_profile_v1"
RUNTIME_PROBE_JOB_EVIDENCE_SCHEMA = "terminal_runtime_profile_probe_job_evidence_v1"
RUNTIME_PROBE_PREFLIGHT_SCHEMA = "terminal_runtime_profile_probe_preflight_v2"
RUNTIME_PROBE_GATE_SCHEMA = "terminal_runtime_profile_probe_scheduler_gate_v2"

# Keep the shared evidence schema self-contained so the four-file runtime-probe stage
# never needs to import the migration implementation.
AUTHORITATIVE_RUNTIME_KEYS = (
    "byteorder",
    "libc",
    "platform_machine",
    "platform_release",
    "platform_system",
    "python_build",
    "python_executable",
    "python_implementation",
    "python_version",
)
AUTHORITATIVE_DEPENDENCY_KEYS = ("numpy", "scipy")
MIGRATION_TOOL_PATHS = (
    "scripts/export_terminal_base_migration.py",
    "scripts/hoffman2_terminal_base_migration.job",
    "src/experiments/terminal_base_migration.py",
)
RUNTIME_PROBE_FILE_PATHS = (
    "scripts/collect_hoffman2_runtime_profile_probe.py",
    "scripts/hoffman2_terminal_runtime_profile_probe.job",
    "scripts/submit_hoffman2_terminal_runtime_profile_probe.sh",
    "src/experiments/terminal_migration_evidence.py",
)

MIGRATION_JOB_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "artifact_file",
        "artifact_sha256",
        "semantic_output_hash",
        "artifact_semantic_output_hash",
        "execution_approval_file_hash",
        "python_executable",
        "runtime_identity",
        "dependency_identity",
        "migration_tool_hashes",
        "hostname",
        "canonical_hostname",
        "job_id",
        "slots",
        "start_utc",
        "end_utc",
    }
)
RUNTIME_PROFILE_FIELDS = frozenset(
    {"schema", "runtime_identity", "dependency_identity"}
)
RUNTIME_PROBE_JOB_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "profile_file",
        "profile_sha256",
        "job_script_sha256",
        "python_executable",
        "hostname",
        "canonical_hostname",
        "job_id",
        "slots",
        "start_utc",
        "end_utc",
    }
)
RUNTIME_PROBE_PREFLIGHT_FIELDS = frozenset(
    {
        "schema",
        "approved_file_hashes",
        "source_file_hashes",
        "evidence_copy_hashes",
        "python_executable",
        "conda_env_path",
    }
)


def _load_terminal_base_migration():
    """Load migration-only code only when migration collection is requested."""

    from . import terminal_base_migration

    return terminal_base_migration


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_compute_hostname(value: object) -> str:
    """Canonicalize Hoffman2 hostnames to a lowercase first DNS label."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("hostname must be a nonempty string")
    hostname = value.strip().rstrip(".").lower().split(".", 1)[0]
    if not hostname or hostname.startswith("login"):
        raise ValueError("hostname is not a compute host")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in hostname):
        raise ValueError("hostname contains invalid characters")
    return hostname


def _strict_json(path: Path) -> Mapping[str, object]:
    def reject_duplicates(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _require_fields(raw: Mapping[str, object], expected: frozenset[str], context: str) -> None:
    actual = set(raw)
    if actual != expected:
        raise ValueError(
            f"{context} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_string(raw: Mapping[str, object], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _require_sha256(raw: Mapping[str, object], key: str) -> str:
    value = _require_string(raw, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a lowercase SHA-256")
    return value


def _require_positive_job_id(raw: Mapping[str, object], key: str = "job_id") -> str:
    value = _require_string(raw, key)
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"{key} must be a positive numeric scheduler ID")
    return value


def _require_one_slot(raw: Mapping[str, object]) -> int:
    value = raw["slots"]
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError("slots must be the integer 1")
    return value


def _require_utc_interval(raw: Mapping[str, object]) -> Tuple[str, str]:
    values = []
    for key in ("start_utc", "end_utc"):
        text = _require_string(raw, key)
        if not text.endswith("Z"):
            raise ValueError(f"{key} must be canonical UTC")
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00")
        except ValueError as error:
            raise ValueError(f"{key} is not ISO-8601 UTC") from error
        if parsed.tzinfo != timezone.utc:
            raise ValueError(f"{key} must be UTC")
        values.append(parsed)
    if values[1] < values[0]:
        raise ValueError("job evidence end time precedes start time")
    return _require_string(raw, "start_utc"), _require_string(raw, "end_utc")


def _parse_pairs(
    value: object,
    *,
    exact_keys: Tuple[str, ...],
    context: str,
) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    pairs = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{context} contains a malformed pair")
        key, item_value = item
        if not isinstance(key, str) or not isinstance(item_value, str) or not item_value:
            raise ValueError(f"{context} contains an invalid key/value")
        pairs.append((key, item_value))
    result = tuple(pairs)
    if tuple(sorted(result)) != result or tuple(key for key, _ in result) != exact_keys:
        raise ValueError(f"{context} keys/order mismatch")
    return result


def _parse_hash_pairs(
    value: object,
    *,
    exact_keys: Tuple[str, ...],
    context: str,
) -> Tuple[Tuple[str, str], ...]:
    result = _parse_pairs(value, exact_keys=exact_keys, context=context)
    if any(
        len(item_hash) != 64
        or any(character not in "0123456789abcdef" for character in item_hash)
        for _, item_hash in result
    ):
        raise ValueError(f"{context} contains a malformed SHA-256")
    return result


def _approved_runtime_probe_hashes(
    value: Mapping[str, str],
) -> Tuple[Tuple[str, str], ...]:
    if set(value) != set(RUNTIME_PROBE_FILE_PATHS):
        raise ValueError("external runtime-probe approved-file set mismatch")
    return _parse_hash_pairs(
        [[path, value[path]] for path in RUNTIME_PROBE_FILE_PATHS],
        exact_keys=RUNTIME_PROBE_FILE_PATHS,
        context="external runtime-probe approved hashes",
    )


def _file_hashes_at_root(
    root: Path,
    *,
    context: str,
) -> Tuple[Tuple[str, str], ...]:
    result = []
    for relative_path in RUNTIME_PROBE_FILE_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"{context} file is absent: {relative_path}")
        result.append((relative_path, sha256_file(path)))
    return tuple(result)


def validate_runtime_probe_preflight(
    raw: Mapping[str, object],
    *,
    approved_probe_file_hashes: Mapping[str, str],
) -> Dict[str, object]:
    _require_fields(raw, RUNTIME_PROBE_PREFLIGHT_FIELDS, "runtime probe preflight")
    if _require_string(raw, "schema") != RUNTIME_PROBE_PREFLIGHT_SCHEMA:
        raise ValueError("runtime probe preflight schema mismatch")
    approved = _approved_runtime_probe_hashes(approved_probe_file_hashes)
    recorded_approved = _parse_hash_pairs(
        raw["approved_file_hashes"],
        exact_keys=RUNTIME_PROBE_FILE_PATHS,
        context="preflight approved hashes",
    )
    source_hashes = _parse_hash_pairs(
        raw["source_file_hashes"],
        exact_keys=RUNTIME_PROBE_FILE_PATHS,
        context="preflight source hashes",
    )
    evidence_hashes = _parse_hash_pairs(
        raw["evidence_copy_hashes"],
        exact_keys=RUNTIME_PROBE_FILE_PATHS,
        context="preflight evidence-copy hashes",
    )
    if recorded_approved != approved or source_hashes != approved or evidence_hashes != approved:
        raise ValueError("runtime probe preflight differs from externally approved hashes")
    return {
        "approved_file_hashes": approved,
        "source_file_hashes": source_hashes,
        "evidence_copy_hashes": evidence_hashes,
        "python_executable": _require_string(raw, "python_executable"),
        "conda_env_path": _require_string(raw, "conda_env_path"),
    }


def write_runtime_probe_preflight(
    *,
    source_root: Path,
    evidence_root: Path,
    approved_probe_file_hashes: Mapping[str, str],
    python_executable: str,
    conda_env_path: str,
    preflight_path: Path,
) -> Mapping[str, object]:
    if preflight_path.exists():
        raise FileExistsError(f"runtime probe preflight already exists: {preflight_path}")
    approved = _approved_runtime_probe_hashes(approved_probe_file_hashes)
    source_hashes = _file_hashes_at_root(source_root, context="source")
    evidence_hashes = _file_hashes_at_root(evidence_root, context="evidence copy")
    if source_hashes != approved or evidence_hashes != approved:
        raise ValueError("runtime probe source/copy differs from externally approved hashes")
    if not isinstance(python_executable, str) or not python_executable:
        raise ValueError("python_executable must be a nonempty string")
    if not isinstance(conda_env_path, str) or not conda_env_path:
        raise ValueError("conda_env_path must be a nonempty string")
    payload = {
        "schema": RUNTIME_PROBE_PREFLIGHT_SCHEMA,
        "approved_file_hashes": [list(item) for item in approved],
        "source_file_hashes": [list(item) for item in source_hashes],
        "evidence_copy_hashes": [list(item) for item in evidence_hashes],
        "python_executable": python_executable,
        "conda_env_path": conda_env_path,
    }
    with preflight_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _single_hash_line(path: Path, context: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is absent: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError(f"{context} must contain exactly one line")
    value = lines[0]
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{context} is not a lowercase SHA-256")
    return value


def _single_text_line(path: Path, context: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is absent: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError(f"{context} must contain exactly one nonempty line")
    return lines[0]


def _validate_submission_job_id(run_dir: Path, submitted_job_id: str) -> None:
    evidence_dir = run_dir / "submission_evidence"
    recorded_id = _single_text_line(evidence_dir / "job_id.txt", "submitted job ID")
    terse_id = _single_text_line(evidence_dir / "qsub.stdout", "raw terse qsub output")
    qsub_status = _single_text_line(
        evidence_dir / "qsub.exit_status", "qsub exit status"
    )
    if recorded_id != submitted_job_id or terse_id != submitted_job_id:
        raise ValueError("submitted job ID differs from raw qsub evidence")
    if qsub_status != "0":
        raise ValueError("qsub exit status is not zero")
    command_path = evidence_dir / "qsub.command"
    if not command_path.is_file() or not command_path.read_text(encoding="utf-8").strip():
        raise ValueError("literal qsub command evidence is absent")


def parse_qacct(path: Path) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) == {"="}:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            raise ValueError("qacct contains a malformed line")
        key, value = parts
        if key in fields:
            raise ValueError(f"qacct contains duplicate field: {key}")
        fields[key] = value.strip()
    return fields


def _validate_qacct(
    fields: Mapping[str, str],
    *,
    submitted_job_id: str,
) -> str:
    expected = {
        "failed": "0",
        "exit_status": "0",
        "slots": "1",
        "jobnumber": submitted_job_id,
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ValueError(f"qacct gate failed: {key}={fields.get(key)!r}")
    return canonical_compute_hostname(fields.get("hostname"))


def validate_migration_job_evidence(
    raw: Mapping[str, object],
) -> Dict[str, object]:
    _require_fields(raw, MIGRATION_JOB_EVIDENCE_FIELDS, "migration job evidence")
    if _require_string(raw, "schema") != MIGRATION_JOB_EVIDENCE_SCHEMA:
        raise ValueError("migration job evidence schema mismatch")
    if _require_string(raw, "artifact_file") != "terminal_base_beliefs.candidate.json":
        raise ValueError("migration artifact filename mismatch")
    _require_sha256(raw, "artifact_sha256")
    _require_sha256(raw, "semantic_output_hash")
    _require_sha256(raw, "artifact_semantic_output_hash")
    _require_sha256(raw, "execution_approval_file_hash")
    job_id = _require_positive_job_id(raw)
    slots = _require_one_slot(raw)
    hostname = _require_string(raw, "hostname")
    canonical = canonical_compute_hostname(hostname)
    if _require_string(raw, "canonical_hostname") != canonical:
        raise ValueError("stored canonical hostname mismatch")
    runtime = _parse_pairs(
        raw["runtime_identity"],
        exact_keys=AUTHORITATIVE_RUNTIME_KEYS,
        context="job runtime identity",
    )
    dependencies = _parse_pairs(
        raw["dependency_identity"],
        exact_keys=AUTHORITATIVE_DEPENDENCY_KEYS,
        context="job dependency identity",
    )
    tools = _parse_pairs(
        raw["migration_tool_hashes"],
        exact_keys=MIGRATION_TOOL_PATHS,
        context="job migration tool hashes",
    )
    if any(value == "not-installed" for _, value in dependencies):
        raise ValueError("job dependency identity contains a missing package")
    python_executable = _require_string(raw, "python_executable")
    if dict(runtime)["python_executable"] != python_executable:
        raise ValueError("job Python executable differs from runtime identity")
    start_utc, end_utc = _require_utc_interval(raw)
    return {
        "job_id": job_id,
        "slots": slots,
        "hostname": hostname,
        "canonical_hostname": canonical,
        "runtime_identity": runtime,
        "dependency_identity": dependencies,
        "migration_tool_hashes": tools,
        "python_executable": python_executable,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "artifact_sha256": raw["artifact_sha256"],
        "semantic_output_hash": raw["semantic_output_hash"],
        "artifact_semantic_output_hash": raw["artifact_semantic_output_hash"],
        "execution_approval_file_hash": raw["execution_approval_file_hash"],
    }


def collect_migration_scheduler_evidence(
    *,
    run_dir: Path,
    qacct_path: Path,
    submitted_job_id: str,
    approved_execution_approval_file_hash: str,
    gate_path: Path,
) -> Mapping[str, object]:
    migration = _load_terminal_base_migration()
    if not submitted_job_id.isdigit() or int(submitted_job_id) <= 0:
        raise ValueError("submitted job ID must be positive and numeric")
    if gate_path.exists():
        raise FileExistsError(f"scheduler gate already exists: {gate_path}")
    if (
        len(approved_execution_approval_file_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in approved_execution_approval_file_hash
        )
    ):
        raise ValueError("external execution approval hash is malformed")
    _validate_submission_job_id(run_dir, submitted_job_id)
    qacct = parse_qacct(qacct_path)
    qacct_host = _validate_qacct(qacct, submitted_job_id=submitted_job_id)

    artifact = run_dir / "terminal_base_beliefs.candidate.json"
    matches = list(run_dir.rglob(artifact.name))
    if matches != [artifact] or not artifact.is_file() or artifact.stat().st_size == 0:
        raise ValueError("candidate artifact must exist exactly once and be nonempty")
    evidence_path = run_dir / "job_evidence.json"
    evidence = validate_migration_job_evidence(_strict_json(evidence_path))
    if evidence["job_id"] != submitted_job_id:
        raise ValueError("job evidence ID differs from submitted/qacct job ID")
    if evidence["slots"] != int(qacct["slots"]):
        raise ValueError("job evidence slots differ from qacct slots")
    if evidence["canonical_hostname"] != qacct_host:
        raise ValueError("job evidence hostname differs from qacct hostname")

    artifact_hash = sha256_file(artifact)
    artifact_hash_file = _single_hash_line(
        run_dir / "artifact_sha256.txt", "artifact hash file"
    )
    semantic_hash_file = _single_hash_line(
        run_dir / "semantic_output_hash.txt", "semantic hash file"
    )
    if artifact_hash != artifact_hash_file or artifact_hash != evidence["artifact_sha256"]:
        raise ValueError("artifact byte hashes do not agree")

    raw_artifact = _strict_json(artifact)
    parsed_artifact = migration.parse_migration(raw_artifact)
    if migration.migration_output_hash(parsed_artifact) != parsed_artifact.output_hash:
        raise ValueError("candidate artifact semantic self-hash mismatch")
    semantic_values = {
        semantic_hash_file,
        str(evidence["semantic_output_hash"]),
        str(evidence["artifact_semantic_output_hash"]),
        parsed_artifact.output_hash,
    }
    if len(semantic_values) != 1:
        raise ValueError("artifact semantic hashes do not agree")

    approval_path = run_dir / "submission_evidence" / "execution_approval.json"
    approval_hash = sha256_file(approval_path)
    if approval_hash != approved_execution_approval_file_hash:
        raise ValueError("execution approval differs from external Reviewer hash")
    submitted_approval_hash = _single_hash_line(
        run_dir / "submission_evidence" / "approved_execution_approval_sha256.txt",
        "submitted execution approval hash",
    )
    if submitted_approval_hash != approved_execution_approval_file_hash:
        raise ValueError("submitted execution approval hash differs from Reviewer hash")
    approval = migration.load_execution_approval(
        approval_path,
        approved_file_hash=approval_hash,
    )
    if (
        approval_hash != evidence["execution_approval_file_hash"]
        or approval_hash != parsed_artifact.execution_approval_file_hash
    ):
        raise ValueError("execution approval hashes do not agree")
    if evidence["migration_tool_hashes"] != approval.migration_tool_hashes:
        raise ValueError("job tool identity differs from execution approval")
    if evidence["runtime_identity"] != approval.runtime_identity:
        raise ValueError("job runtime identity differs from execution approval")
    if evidence["dependency_identity"] != approval.dependency_identity:
        raise ValueError("job dependency identity differs from execution approval")
    if parsed_artifact.migration_tool_hashes != approval.migration_tool_hashes:
        raise ValueError("artifact tool identity differs from execution approval")
    if parsed_artifact.runtime_identity != approval.runtime_identity:
        raise ValueError("artifact runtime identity differs from execution approval")
    if parsed_artifact.dependency_identity != approval.dependency_identity:
        raise ValueError("artifact dependency identity differs from execution approval")

    gate = {
        "schema": MIGRATION_SCHEDULER_GATE_SCHEMA,
        "job_id": submitted_job_id,
        "failed": 0,
        "exit_status": 0,
        "slots": 1,
        "canonical_hostname": qacct_host,
        "qacct_sha256": sha256_file(qacct_path),
        "job_evidence_sha256": sha256_file(evidence_path),
        "artifact_sha256": artifact_hash,
        "semantic_output_hash": parsed_artifact.output_hash,
        "execution_approval_file_hash": approval_hash,
        "migration_tool_hashes": [list(item) for item in approval.migration_tool_hashes],
        "runtime_identity": [list(item) for item in approval.runtime_identity],
        "dependency_identity": [list(item) for item in approval.dependency_identity],
        "candidate_only_not_reviewer_approved": True,
    }
    with gate_path.open("x", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return gate


def validate_runtime_profile(raw: Mapping[str, object]) -> Dict[str, object]:
    _require_fields(raw, RUNTIME_PROFILE_FIELDS, "runtime profile")
    if _require_string(raw, "schema") != RUNTIME_PROFILE_SCHEMA:
        raise ValueError("runtime profile schema mismatch")
    runtime = _parse_pairs(
        raw["runtime_identity"],
        exact_keys=AUTHORITATIVE_RUNTIME_KEYS,
        context="runtime profile identity",
    )
    dependencies = _parse_pairs(
        raw["dependency_identity"],
        exact_keys=AUTHORITATIVE_DEPENDENCY_KEYS,
        context="runtime profile dependencies",
    )
    if any(value == "not-installed" for _, value in dependencies):
        raise ValueError("runtime profile dependency is missing")
    return {"runtime_identity": runtime, "dependency_identity": dependencies}


def validate_runtime_probe_job_evidence(raw: Mapping[str, object]) -> Dict[str, object]:
    _require_fields(
        raw,
        RUNTIME_PROBE_JOB_EVIDENCE_FIELDS,
        "runtime probe job evidence",
    )
    if _require_string(raw, "schema") != RUNTIME_PROBE_JOB_EVIDENCE_SCHEMA:
        raise ValueError("runtime probe job evidence schema mismatch")
    if _require_string(raw, "profile_file") != "runtime_profile.candidate.json":
        raise ValueError("runtime probe profile filename mismatch")
    _require_sha256(raw, "profile_sha256")
    _require_sha256(raw, "job_script_sha256")
    job_id = _require_positive_job_id(raw)
    slots = _require_one_slot(raw)
    hostname = _require_string(raw, "hostname")
    canonical = canonical_compute_hostname(hostname)
    if _require_string(raw, "canonical_hostname") != canonical:
        raise ValueError("runtime probe canonical hostname mismatch")
    start_utc, end_utc = _require_utc_interval(raw)
    return {
        "job_id": job_id,
        "slots": slots,
        "canonical_hostname": canonical,
        "profile_sha256": raw["profile_sha256"],
        "job_script_sha256": raw["job_script_sha256"],
        "python_executable": _require_string(raw, "python_executable"),
        "start_utc": start_utc,
        "end_utc": end_utc,
    }


def _runtime_probe_gate_payload(
    *,
    run_dir: Path,
    qacct_path: Path,
    submitted_job_id: str,
    approved_probe_file_hashes: Mapping[str, str],
) -> Mapping[str, object]:
    approved = _approved_runtime_probe_hashes(approved_probe_file_hashes)
    approved_by_path = dict(approved)
    evidence_root = run_dir / "submission_evidence"
    preflight_path = evidence_root / "preflight.json"
    preflight = validate_runtime_probe_preflight(
        _strict_json(preflight_path),
        approved_probe_file_hashes=approved_probe_file_hashes,
    )
    evidence_hashes = _file_hashes_at_root(evidence_root, context="evidence copy")
    if evidence_hashes != approved or evidence_hashes != preflight["evidence_copy_hashes"]:
        raise ValueError("runtime probe evidence copies differ from approved preflight")
    _validate_submission_job_id(run_dir, submitted_job_id)
    qacct = parse_qacct(qacct_path)
    qacct_host = _validate_qacct(qacct, submitted_job_id=submitted_job_id)
    profile_path = run_dir / "runtime_profile.candidate.json"
    profile_matches = list(run_dir.rglob(profile_path.name))
    if profile_matches != [profile_path] or not profile_path.is_file():
        raise ValueError("runtime profile must exist exactly once")
    profile = validate_runtime_profile(_strict_json(profile_path))
    evidence = validate_runtime_probe_job_evidence(
        _strict_json(run_dir / "runtime_probe_job_evidence.json")
    )
    if evidence["job_id"] != submitted_job_id:
        raise ValueError("runtime probe job ID differs from submitted/qacct job ID")
    if evidence["slots"] != int(qacct["slots"]):
        raise ValueError("runtime probe slots differ from qacct slots")
    if evidence["canonical_hostname"] != qacct_host:
        raise ValueError("runtime probe hostname differs from qacct hostname")
    profile_hash = sha256_file(profile_path)
    if profile_hash != evidence["profile_sha256"] or profile_hash != _single_hash_line(
        run_dir / "runtime_profile_sha256.txt", "runtime profile hash file"
    ):
        raise ValueError("runtime profile hashes do not agree")
    job_script_path = "scripts/hoffman2_terminal_runtime_profile_probe.job"
    job_script = evidence_root / job_script_path
    submitted_job_hash = _single_hash_line(
        run_dir / "submission_evidence" / "approved_probe_job_script_sha256.txt",
        "submitted runtime-probe job-script hash",
    )
    if (
        sha256_file(job_script) != evidence["job_script_sha256"]
        or evidence["job_script_sha256"] != approved_by_path[job_script_path]
        or submitted_job_hash != approved_by_path[job_script_path]
    ):
        raise ValueError("runtime probe job-script hash mismatch")
    if dict(profile["runtime_identity"])["python_executable"] != evidence["python_executable"]:
        raise ValueError("runtime probe Python executable mismatch")
    gate = {
        "schema": RUNTIME_PROBE_GATE_SCHEMA,
        "job_id": submitted_job_id,
        "failed": 0,
        "exit_status": 0,
        "slots": 1,
        "canonical_hostname": qacct_host,
        "qacct_sha256": sha256_file(qacct_path),
        "job_evidence_sha256": sha256_file(
            run_dir / "runtime_probe_job_evidence.json"
        ),
        "profile_sha256": profile_hash,
        "job_script_sha256": evidence["job_script_sha256"],
        "preflight_sha256": sha256_file(preflight_path),
        "approved_file_hashes": [list(item) for item in approved],
        "source_file_hashes": [list(item) for item in preflight["source_file_hashes"]],
        "evidence_copy_hashes": [list(item) for item in evidence_hashes],
        "runtime_identity": [list(item) for item in profile["runtime_identity"]],
        "dependency_identity": [list(item) for item in profile["dependency_identity"]],
        "candidate_only_not_reviewer_approved": True,
    }
    return gate


def collect_runtime_probe_scheduler_evidence(
    *,
    run_dir: Path,
    qacct_path: Path,
    submitted_job_id: str,
    approved_probe_file_hashes: Mapping[str, str],
    gate_path: Path,
) -> Mapping[str, object]:
    if gate_path.exists():
        raise FileExistsError(f"runtime probe gate already exists: {gate_path}")
    gate = _runtime_probe_gate_payload(
        run_dir=run_dir,
        qacct_path=qacct_path,
        submitted_job_id=submitted_job_id,
        approved_probe_file_hashes=approved_probe_file_hashes,
    )
    with gate_path.open("x", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return gate


def validate_runtime_probe_scheduler_gate(
    *,
    run_dir: Path,
    qacct_path: Path,
    submitted_job_id: str,
    approved_probe_file_hashes: Mapping[str, str],
    gate_path: Path,
) -> Mapping[str, object]:
    expected = _runtime_probe_gate_payload(
        run_dir=run_dir,
        qacct_path=qacct_path,
        submitted_job_id=submitted_job_id,
        approved_probe_file_hashes=approved_probe_file_hashes,
    )
    recorded = _strict_json(gate_path)
    if recorded != expected:
        raise ValueError("runtime probe scheduler gate differs from source-backed evidence")
    return expected


__all__ = [
    "AUTHORITATIVE_DEPENDENCY_KEYS",
    "AUTHORITATIVE_RUNTIME_KEYS",
    "MIGRATION_JOB_EVIDENCE_SCHEMA",
    "MIGRATION_TOOL_PATHS",
    "RUNTIME_PROFILE_SCHEMA",
    "RUNTIME_PROBE_FILE_PATHS",
    "RUNTIME_PROBE_JOB_EVIDENCE_SCHEMA",
    "canonical_compute_hostname",
    "collect_migration_scheduler_evidence",
    "collect_runtime_probe_scheduler_evidence",
    "parse_qacct",
    "sha256_file",
    "validate_migration_job_evidence",
    "validate_runtime_profile",
    "validate_runtime_probe_job_evidence",
    "validate_runtime_probe_preflight",
    "validate_runtime_probe_scheduler_gate",
    "write_runtime_probe_preflight",
]
