from __future__ import annotations

"""Strict Hoffman2 evidence audit for the terminal plan-only performance gate."""

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Tuple
from xml.etree import ElementTree

from . import terminal_execution as execution


PLAN_DIAGNOSTIC_AUDIT_SCHEMA = "terminal_validation_plan_diagnostic_audit_v1"
PLAN_DIAGNOSTIC_RUN_SCHEMA = "terminal_validation_plan_diagnostic_run_v1"
EXPECTED_TASKS = tuple(range(1, 17))
MAX_WALL_SECONDS = 300.0
MAX_MEMORY_BYTES = 1024**3
FORBIDDEN_LOG_MARKERS = (
    "memoryerror",
    "killed",
    "traceback",
    "incomplete trace",
    "sidecar",
)


def _load(path: Path) -> Dict[str, Any]:
    return execution._decode(dict(execution._load_json(path)))


def _metadata(run_root: Path) -> Dict[str, Any]:
    raw = json.loads((run_root / "run_metadata.json").read_text(encoding="utf-8"))
    if raw.get("schema") != PLAN_DIAGNOSTIC_RUN_SCHEMA:
        raise RuntimeError("plan diagnostic run metadata schema mismatch")
    if raw.get("stage") != "smoke" or raw.get("descriptor_count") != len(EXPECTED_TASKS):
        raise RuntimeError("plan diagnostic run metadata scope mismatch")
    if raw.get("requested_slots") != 1:
        raise RuntimeError("plan diagnostic must request one slot")
    memory = raw.get("requested_memory_bytes")
    if type(memory) is not int or memory < 2 * MAX_MEMORY_BYTES:
        raise RuntimeError("plan diagnostic requested memory is below the audit envelope")
    if raw.get("max_wall_seconds") != int(MAX_WALL_SECONDS):
        raise RuntimeError("plan diagnostic wall-time acceptance threshold mismatch")
    jobs = raw.get("jobs")
    if not isinstance(jobs, list) or {item.get("replicate") for item in jobs} != {"a", "b"}:
        raise RuntimeError("plan diagnostic metadata must contain two replicates")
    if len(jobs) != 2 or len({str(item.get("job_id")) for item in jobs}) != 2:
        raise RuntimeError("plan diagnostic job identities are absent or duplicated")
    for name in ("source_commit", "source_tree"):
        if re.fullmatch(r"[0-9a-f]{40}", str(raw.get(name, ""))) is None:
            raise RuntimeError(f"plan diagnostic {name} is malformed")
    for name in ("source_identity_hash", "provider_hash"):
        if re.fullmatch(r"[0-9a-f]{64}", str(raw.get(name, ""))) is None:
            raise RuntimeError(f"plan diagnostic {name} is malformed")
    return raw


def validate_qstat_absence_text(text: str, job_id: str, exit_status: int) -> None:
    """Require a valid successful XML queue snapshot in which ``job_id`` is absent."""

    if exit_status != 0:
        raise RuntimeError("qstat does not authoritatively prove job absence")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise RuntimeError("qstat does not authoritatively prove job absence") from error
    local_name = lambda tag: str(tag).rsplit("}", 1)[-1]
    if local_name(root.tag) != "job_info":
        raise RuntimeError("qstat does not authoritatively prove job absence")
    child_names = {local_name(child.tag) for child in root}
    if not {"queue_info", "job_info"}.issubset(child_names):
        raise RuntimeError("qstat does not authoritatively prove job absence")
    observed_job_ids = {
        (element.text or "").strip()
        for element in root.iter()
        if local_name(element.tag) == "JB_job_number"
    }
    if str(job_id) in observed_job_ids:
        raise RuntimeError("qstat does not authoritatively prove job absence")


def _expected_source_context(project_root: Path) -> Dict[str, Any]:
    source = execution.capture_clean_source_identity(
        project_root, execution.TERMINAL_SOURCE_PATHS
    )
    provider, accepted = execution.load_accepted_canonical_base_provider()
    suites = execution.build_terminal_suites(provider, accepted, validate_contents=False)
    descriptors = execution._selected_descriptors("smoke", suites)
    return {
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "source_identity_hash": source["identity_hash"],
        "provider_hash": provider.provider_hash,
        "descriptors": tuple({
            "task_id": position,
            "suite_class": descriptor.suite_class,
            "descriptor_index": descriptor.descriptor_index,
            "descriptor_hash": descriptor.descriptor_hash,
            "source_case_id": descriptor.source_case_id,
            "profile": descriptor.profile,
            "source_identity_hash": source["identity_hash"],
            "provider_hash": provider.provider_hash,
        } for position, descriptor in enumerate(descriptors, start=1)),
    }


def _validate_job_sources(run_root: Path, job: Mapping[str, Any]) -> Tuple[str, str, str]:
    replicate = str(job["replicate"])
    job_id = str(job["job_id"])
    if not job_id.isdigit():
        raise RuntimeError("plan diagnostic job ID is malformed")
    job_path = run_root / "jobs" / f"plan_{replicate}.job"
    qsub_path = run_root / "qsub_raw" / f"plan_{replicate}.txt"
    qstat_path = run_root / "final_qstat" / f"plan_{replicate}.raw"
    qstat_status_path = run_root / "final_qstat" / f"plan_{replicate}.status"
    if (
        not job_path.is_file()
        or not qsub_path.is_file()
        or not qstat_path.is_file()
        or not qstat_status_path.is_file()
    ):
        raise RuntimeError("plan diagnostic scheduler source evidence is missing")
    qsub = qsub_path.read_text(encoding="utf-8").strip()
    if re.fullmatch(rf"{re.escape(job_id)}(?:\.[^\s]+)?", qsub) is None:
        raise RuntimeError("plan diagnostic qsub output does not match its job ID")
    script = job_path.read_text(encoding="utf-8")
    required = (
        "#$ -t 1-16",
        "#$ -tc 16",
        "#$ -l h_data=2G",
        "diagnose-plan",
        "--stage smoke",
        "--mode plan-only",
    )
    if any(token not in script for token in required):
        raise RuntimeError("plan diagnostic job script semantics are incomplete")
    status_text = qstat_status_path.read_text(encoding="utf-8").strip()
    if not status_text.isdigit():
        raise RuntimeError("qstat exit status evidence is malformed")
    validate_qstat_absence_text(
        qstat_path.read_text(encoding="utf-8"), job_id, int(status_text)
    )
    return (
        execution.sha256_file(job_path),
        execution.sha256_file(qsub_path),
        execution.logical_hash((
            execution.sha256_file(qstat_path),
            execution.sha256_file(qstat_status_path),
        )),
    )


def _validate_diagnostic(path: Path, task_id: int) -> Tuple[Dict[str, Any], bytes, str]:
    diagnostic = _load(path)
    execution._validate_self_hash(diagnostic, "diagnostic_hash", "plan diagnostic")
    if diagnostic.get("schema") != execution.PLAN_DIAGNOSTIC_SCHEMA:
        raise RuntimeError("plan diagnostic schema mismatch")
    if diagnostic.get("mode") != "plan_only":
        raise RuntimeError("P1 diagnostic entered the full evidence path")
    if diagnostic.get("full_projection") is not None or diagnostic.get("parity_pass") is not None:
        raise RuntimeError("P1 diagnostic retained formal evidence")
    plan = diagnostic.get("plan")
    if not isinstance(plan, Mapping):
        raise RuntimeError("plan diagnostic payload is missing")
    phases = dict(diagnostic.get("phase_seconds", ()))
    required_phases = {
        "provider_load",
        "suite_reconstruction",
        "source_identity_capture",
        "source_reconstruction",
        "plan_plan_computation_total",
        "plan_canonicalization_serialization",
    }
    if not required_phases.issubset(phases):
        raise RuntimeError("plan diagnostic phase profile is incomplete")
    if any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in phases.values()):
        raise RuntimeError("plan diagnostic phase profile is nonfinite")
    identity = {
        "task_id": task_id,
        "suite_class": diagnostic.get("suite_class"),
        "descriptor_index": diagnostic.get("descriptor_index"),
        "descriptor_hash": diagnostic.get("descriptor_hash"),
        "source_case_id": diagnostic.get("source_case_id"),
        "profile": diagnostic.get("profile"),
        "source_identity_hash": diagnostic.get("source_identity_hash"),
        "provider_hash": diagnostic.get("provider_hash"),
    }
    plan_bytes = execution.canonical_bytes(plan)
    return identity, plan_bytes, execution.logical_hash(plan)


def _validate_qacct(
    run_root: Path,
    job: Mapping[str, Any],
    *,
    queue: str,
    requested_memory_bytes: int,
) -> Tuple[Tuple[Dict[str, Any], ...], str]:
    replicate = str(job["replicate"])
    job_id = str(job["job_id"])
    path = run_root / "qacct" / f"plan_{replicate}.raw"
    if not path.is_file():
        raise RuntimeError("plan diagnostic qacct evidence is missing")
    records = execution.parse_qacct_records(path.read_text(encoding="utf-8"))
    if len(records) != len(EXPECTED_TASKS):
        raise RuntimeError("plan diagnostic qacct coverage is incomplete")
    normalized = []
    observed = set()
    for record in records:
        required = {
            "jobnumber", "jobname", "qname", "hostname", "taskid", "slots",
            "failed", "exit_status", "cpu", "ru_wallclock", "maxvmem",
        }
        if not required.issubset(record):
            raise RuntimeError("plan diagnostic qacct record lacks required fields")
        if record["jobnumber"] != job_id or not record["taskid"].isdigit():
            raise RuntimeError("plan diagnostic qacct identity mismatch")
        if record["jobname"] != f"tvp1_{replicate}":
            raise RuntimeError("plan diagnostic qacct job-name mismatch")
        task_id = int(record["taskid"])
        if task_id not in EXPECTED_TASKS or task_id in observed:
            raise RuntimeError("plan diagnostic qacct task coverage is invalid")
        observed.add(task_id)
        if not execution._queue_matches(queue, record["qname"]):
            raise RuntimeError("plan diagnostic qacct queue mismatch")
        host = record["hostname"].split(".", 1)[0].lower()
        if host.startswith("login") or execution._COMPUTE_HOST_RE.fullmatch(host) is None:
            raise RuntimeError("plan diagnostic did not run on a compute node")
        if record["slots"] != "1" or record["failed"] != "0" or record["exit_status"] != "0":
            raise RuntimeError("plan diagnostic qacct does not prove success")
        wall = execution._duration_seconds(record["ru_wallclock"])
        memory = execution._memory_bytes(record["maxvmem"])
        if wall > MAX_WALL_SECONDS:
            raise RuntimeError("plan diagnostic exceeded the 300-second P1 gate")
        if memory > MAX_MEMORY_BYTES or memory > requested_memory_bytes // 2:
            raise RuntimeError("plan diagnostic exceeded the P1 memory gate")
        normalized.append({
            "task_id": task_id,
            "hostname": host,
            "wall_seconds": wall,
            "cpu_seconds": execution._duration_seconds(record["cpu"]),
            "max_memory_bytes": memory,
        })
    if observed != set(EXPECTED_TASKS):
        raise RuntimeError("plan diagnostic qacct task coverage differs from 1-16")
    return tuple(sorted(normalized, key=lambda item: item["task_id"])), execution.sha256_file(path)


def audit_plan_diagnostic_run(
    run_root: Path,
    *,
    project_root: Path | None = None,
    expected_context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Audit two immutable 16-descriptor plan-only Hoffman2 replicates."""

    root = run_root.resolve()
    metadata = _metadata(root)
    context = dict(
        expected_context
        if expected_context is not None
        else _expected_source_context(
            Path(__file__).resolve().parents[2]
            if project_root is None
            else project_root.resolve()
        )
    )
    for field in ("source_commit", "source_tree", "source_identity_hash", "provider_hash"):
        if metadata.get(field) != context.get(field):
            raise RuntimeError(f"plan diagnostic {field} differs from the clean source")
    expected_descriptors = tuple(context.get("descriptors", ()))
    if len(expected_descriptors) != len(EXPECTED_TASKS):
        raise RuntimeError("expected frozen smoke descriptor mapping is incomplete")
    queue = str(metadata["queue"])
    requested_memory = int(metadata["requested_memory_bytes"])
    jobs = {str(item["replicate"]): item for item in metadata["jobs"]}
    per_replicate: Dict[str, Dict[int, Tuple[Dict[str, Any], bytes, str]]] = {}
    job_records = []
    descriptor_hashes = set()
    for replicate in ("a", "b"):
        job = jobs[replicate]
        job_script_hash, qsub_hash, qstat_hash = _validate_job_sources(root, job)
        qacct_records, qacct_hash = _validate_qacct(
            root, job, queue=queue, requested_memory_bytes=requested_memory
        )
        diagnostics = {}
        for task_id in EXPECTED_TASKS:
            output = root / f"replicate_{replicate}" / f"diagnostic_{task_id:03d}.json"
            if not output.is_file():
                raise RuntimeError("plan diagnostic output coverage is incomplete")
            diagnostics[task_id] = _validate_diagnostic(output, task_id)
            descriptor_hashes.add(diagnostics[task_id][0]["descriptor_hash"])
            log = root / "logs" / f"plan_{replicate}.{job['job_id']}.{task_id}.log"
            if not log.is_file():
                raise RuntimeError("plan diagnostic log evidence is missing")
            log_text = log.read_text(encoding="utf-8", errors="strict").lower()
            if any(marker in log_text for marker in FORBIDDEN_LOG_MARKERS):
                raise RuntimeError("plan diagnostic log contains a forbidden failure marker")
        per_replicate[replicate] = diagnostics
        job_records.append({
            "replicate": replicate,
            "job_id": str(job["job_id"]),
            "job_script_sha256": job_script_hash,
            "qsub_raw_sha256": qsub_hash,
            "final_qstat_sha256": qstat_hash,
            "qacct_raw_sha256": qacct_hash,
            "task_usage": qacct_records,
        })
    if len(descriptor_hashes) != len(EXPECTED_TASKS):
        raise RuntimeError("plan diagnostic descriptors are missing or duplicated")
    retry_records = []
    for task_id in EXPECTED_TASKS:
        a_identity, a_bytes, a_hash = per_replicate["a"][task_id]
        b_identity, b_bytes, b_hash = per_replicate["b"][task_id]
        if a_identity != b_identity or a_bytes != b_bytes or a_hash != b_hash:
            raise RuntimeError("plan diagnostic retry output is not byte-identical")
        if a_identity != expected_descriptors[task_id - 1]:
            raise RuntimeError("plan diagnostic differs from the frozen smoke mapping")
        retry_records.append({
            **a_identity,
            "plan_sha256": a_hash,
            "plan_byte_count": len(a_bytes),
        })
    payload: Dict[str, Any] = {
        "schema": PLAN_DIAGNOSTIC_AUDIT_SCHEMA,
        "run_metadata_sha256": execution.sha256_file(root / "run_metadata.json"),
        "descriptor_count": len(EXPECTED_TASKS),
        "replicate_count": 2,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_memory_bytes": MAX_MEMORY_BYTES,
        "jobs": tuple(job_records),
        "retry_records": tuple(retry_records),
        "audit_pass": True,
        "audit_hash": "",
    }
    payload["audit_hash"] = execution.logical_hash(execution._without_hash(payload, "audit_hash"))
    return payload


__all__ = [
    "EXPECTED_TASKS",
    "MAX_MEMORY_BYTES",
    "MAX_WALL_SECONDS",
    "PLAN_DIAGNOSTIC_AUDIT_SCHEMA",
    "PLAN_DIAGNOSTIC_RUN_SCHEMA",
    "audit_plan_diagnostic_run",
    "validate_qstat_absence_text",
]
