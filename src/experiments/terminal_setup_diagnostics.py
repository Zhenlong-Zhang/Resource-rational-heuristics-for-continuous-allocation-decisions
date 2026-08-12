from __future__ import annotations

"""Strict audit for dual-replicate terminal manifest planning and merge."""

import json
import hashlib
import difflib
import math
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from . import terminal_execution as execution
from .terminal_plan_diagnostics import validate_qstat_absence_text


SETUP_AUDIT_SCHEMA = "terminal_validation_manifest_setup_audit_v1"
APPROVED_PYTHON_BIN = "/u/home/z/zzl/.conda/envs/rr-allocation/bin/python"
MAX_FRAGMENT_WALL_SECONDS = 300.0
MAX_MERGE_WALL_SECONDS = 120.0
MAX_CRITICAL_PATH_SECONDS = 420.0
MAX_MEMORY_BYTES = 1024**3
FORBIDDEN_LOG_MARKERS = (
    "memoryerror",
    "killed",
    "traceback",
    "incomplete trace",
    "sidecar",
)
COMPUTE_CEILING_EVIDENCE_FILES = (
    "myresources.raw",
    "qconf_sconf_global.raw",
    "qconf_sq_campus2.raw",
    "qhost.raw",
    "qquota.raw",
    "qstat_g_c.raw",
    "qstat_user.xml",
)


def _config_value(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\s+(\S+)", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"compute-ceiling evidence lacks {name}")
    return match.group(1)


def _h_rt_seconds(value: str) -> int:
    fields = value.split(":")
    if len(fields) != 3 or any(not field.isdigit() for field in fields):
        raise RuntimeError("compute-ceiling h_rt is malformed")
    return int(fields[0]) * 3600 + int(fields[1]) * 60 + int(fields[2])


def create_compute_ceiling_report(
    evidence_root: Path,
    *,
    queue: str,
    captured_at_utc: str | None = None,
) -> Dict[str, Any]:
    """Derive a bounded report from immutable raw Hoffman2 scheduler evidence."""

    root = evidence_root.resolve()
    paths = tuple(root / name for name in COMPUTE_CEILING_EVIDENCE_FILES)
    if any(not path.is_file() for path in paths):
        raise RuntimeError("compute-ceiling raw evidence coverage is incomplete")
    global_config = (root / "qconf_sconf_global.raw").read_text(encoding="utf-8")
    queue_config = (root / "qconf_sq_campus2.raw").read_text(encoding="utf-8")
    qstat_cluster = (root / "qstat_g_c.raw").read_text(encoding="utf-8")
    qhost = (root / "qhost.raw").read_text(encoding="utf-8")
    resources = (root / "myresources.raw").read_text(encoding="utf-8")
    qquota = (root / "qquota.raw").read_text(encoding="utf-8")
    qstat_user = (root / "qstat_user.xml").read_text(encoding="utf-8")
    validate_qstat_absence_text(qstat_user, "__compute_ceiling_probe__", 0)
    queue_name = _config_value(queue_config, "qname")
    if queue_name != queue or queue not in qstat_cluster:
        raise RuntimeError("compute-ceiling queue evidence mismatch")
    max_wall = _h_rt_seconds(_config_value(queue_config, "h_rt"))
    queue_slots = int(_config_value(queue_config, "slots"))
    max_array_tasks = int(_config_value(global_config, "max_aj_tasks"))
    max_array_instances = int(_config_value(global_config, "max_aj_instances"))
    queue_line = next(
        (line for line in qstat_cluster.splitlines() if line.split()[:1] == [queue]),
        None,
    )
    if queue_line is None or len(queue_line.split()) < 7:
        raise RuntimeError("compute-ceiling live queue row is malformed")
    available_slots = int(queue_line.split()[4])
    if available_slots < 1:
        raise RuntimeError("compute-ceiling evidence reports no available queue slot")
    if "resource group(s): campus" not in resources or "24 hours run-time" not in resources:
        raise RuntimeError("compute-ceiling account evidence is incomplete")
    if "resource quota rule" not in qquota:
        raise RuntimeError("compute-ceiling quota evidence is malformed")
    host_rows = [
        line.split() for line in qhost.splitlines()
        if re.match(r"^n[0-9]+\s", line)
    ]
    if not host_rows:
        raise RuntimeError("compute-ceiling host evidence lacks CPU compute nodes")
    caps = execution.TERMINAL_NUMERICAL_RESOURCE_CAPS
    report = {
        "schema": execution.COMPUTE_CEILING_SCHEMA,
        "captured_at_utc": captured_at_utc or datetime.now(timezone.utc).isoformat(),
        "max_walltime_seconds": min(max_wall, int(caps["max_h_rt_seconds"])),
        "max_array_tasks": min(max_array_tasks, int(caps["max_array_tasks"])),
        "max_throttle": min(
            max_array_instances,
            queue_slots,
            available_slots,
            int(caps["max_throttle"]),
        ),
        "max_memory_bytes": int(caps["max_memory_bytes"]),
        "max_storage_bytes": int(caps["max_storage_bytes"]),
        "cpu_hours_quota": None,
        "allowed_queues": (queue,),
        "raw_evidence_hashes": tuple(
            (path.name, execution.sha256_file(path)) for path in paths
        ),
        "report_hash": "",
    }
    report["report_hash"] = execution.logical_hash(
        execution._without_hash(report, "report_hash")
    )
    return report


def _load(path: Path) -> Dict[str, Any]:
    return execution._decode(dict(execution._load_json(path)))


def _submissions(root: Path) -> Tuple[Dict[str, str], ...]:
    path = root / "setup_submissions.tsv"
    if not path.is_file():
        raise RuntimeError("setup submission inventory is missing")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            raise RuntimeError("setup submission inventory is malformed")
        role, job_id, job_file, qsub_file = fields
        if not job_id.isdigit() or role in {row["role"] for row in rows}:
            raise RuntimeError("setup submission identities are invalid")
        rows.append({
            "role": role,
            "job_id": job_id,
            "job_file": job_file,
            "qsub_file": qsub_file,
        })
    expected_roles = ("plan_a_001", "plan_b_002", "plan_merge")
    if tuple(row["role"] for row in rows) != expected_roles:
        raise RuntimeError("smoke setup submission roles differ from the frozen shape")
    return tuple(rows)


def _validate_profile(path: Path, *, command: str, bindings: Mapping[str, Any]) -> str:
    profile = _load(path)
    execution._validate_self_hash(profile, "profile_hash", "phase profile")
    if profile.get("schema") != "terminal_validation_phase_profile_v1":
        raise RuntimeError("setup phase profile schema mismatch")
    if profile.get("command") != command:
        raise RuntimeError("setup phase profile command mismatch")
    observed_bindings = dict(profile.get("bindings", ()))
    if any(observed_bindings.get(key) != value for key, value in bindings.items()):
        raise RuntimeError("setup phase profile binding mismatch")
    phases = dict(profile.get("phase_seconds", ()))
    expected_phases = (
        {
            "provider_load", "suite_reconstruction", "source_identity_capture",
            "planning_context_validation", "descriptor_selection", "plan_computation",
            "descriptor_reference_materialization", "fragment_canonicalization",
            "fragment_validation", "fragment_generation_total",
            "fragment_serialization", "command_total",
        }
        if command == "freeze-plan-fragment"
        else {
            "provider_load", "suite_reconstruction", "source_identity_capture",
            "fragment_loading", "merge_validation_and_assembly",
            "merge_serialization", "command_total",
        }
    )
    if set(phases) != expected_phases or any(
        type(value) is not float or not math.isfinite(value) or value < 0.0
        for value in phases.values()
    ):
        raise RuntimeError("setup phase profile timing is invalid")
    return execution.sha256_file(path)


def _validate_outputs(
    root: Path,
    compute_ceiling_path: Path,
    compute_ceiling_evidence_root: Path,
) -> Tuple[Mapping[str, Any], Tuple[Dict[str, Any], ...], Tuple[str, ...]]:
    manifest_path = root / "terminal_smoke_manifest.json"
    assembly_path = root / "manifest_plan_assembly.json"
    if not manifest_path.is_file() or not assembly_path.is_file():
        raise RuntimeError("setup merge outputs are incomplete")
    manifest = _load(manifest_path)
    assembly = _load(assembly_path)
    ceiling = _load(compute_ceiling_path)
    execution._validate_self_hash(ceiling, "report_hash", "compute ceiling report")
    rebuilt_ceiling = create_compute_ceiling_report(
        compute_ceiling_evidence_root,
        queue="campus2.q",
        captured_at_utc=str(ceiling["captured_at_utc"]),
    )
    if execution.canonical_bytes(ceiling) != execution.canonical_bytes(rebuilt_ceiling):
        raise RuntimeError("compute-ceiling report differs from bound raw evidence")
    provider, accepted = execution.load_accepted_canonical_base_provider()
    suites = execution.build_terminal_suites(provider, accepted, validate_contents=False)
    source = execution.capture_clean_source_identity(
        Path(__file__).resolve().parents[2], execution.TERMINAL_SOURCE_PATHS
    )
    fragments = {"a": [], "b": []}
    profile_hashes = []
    for replicate in ("a", "b"):
        for task_id in range(1, 17):
            fragment_path = root / f"plan_{replicate}" / f"fragment_{task_id:03d}.json"
            profile_path = root / f"profiles_{replicate}" / f"fragment_{task_id:03d}.json"
            if not fragment_path.is_file() or not profile_path.is_file():
                raise RuntimeError("setup fragment or phase-profile coverage is incomplete")
            fragment = _load(fragment_path)
            execution.validate_manifest_plan_fragment(
                fragment,
                suites,
                provider,
                accepted,
                reconstruct_expected=False,
                validate_suite_contents=False,
            )
            if (
                fragment.get("stage") != "smoke"
                or int(fragment.get("shard_index", 0)) != task_id
                or int(fragment.get("shard_count", 0)) != 16
                or execution.canonical_bytes(fragment.get("source_identity", {}))
                != execution.canonical_bytes(source)
            ):
                raise RuntimeError("setup fragment identity binding mismatch")
            fragments[replicate].append(fragment)
            profile_hashes.append(_validate_profile(
                profile_path,
                command="freeze-plan-fragment",
                bindings={
                    "stage": "smoke",
                    "shard_index": task_id,
                    "shard_count": 16,
                    "fragment_hash": fragment["fragment_hash"],
                    "source_identity_hash": source["identity_hash"],
                },
            ))
    rebuilt_manifest, rebuilt_assembly = execution.merge_manifest_plan_replicates(
        stage="smoke",
        replicate_a=tuple(fragments["a"]),
        replicate_b=tuple(fragments["b"]),
        suites=suites,
        provider=provider,
        acceptance_validator=accepted,
        source_identity=source,
        max_descriptors_per_subshard=int(manifest["max_descriptors_per_subshard"]),
        resources=dict(manifest["resources"]),
        compute_ceiling_report_hash=ceiling["report_hash"],
    )
    if execution.canonical_bytes(manifest) != execution.canonical_bytes(rebuilt_manifest):
        raise RuntimeError("stored manifest differs from exact fragment reassembly")
    if execution.canonical_bytes(assembly) != execution.canonical_bytes(rebuilt_assembly):
        raise RuntimeError("stored assembly differs from exact fragment reassembly")
    execution.validate_compute_ceiling_binding(manifest, ceiling)
    if manifest.get("resources") != {
        "queue": "campus2.q",
        "h_rt_seconds": 7200,
        "memory_bytes": 8589934592,
        "throttle": 4,
    }:
        raise RuntimeError("smoke manifest resources differ from the fail-fast P3 shape")
    profile_hashes.append(_validate_profile(
        root / "profiles_merge" / "merge.json",
        command="merge-plan-fragments",
        bindings={
            "stage": "smoke",
            "shard_count": 16,
            "manifest_hash": manifest["manifest_hash"],
            "assembly_hash": assembly["assembly_hash"],
            "source_identity_hash": source["identity_hash"],
        },
    ))
    return manifest, _submissions(root), tuple(profile_hashes)


def _validate_job_script(
    root: Path,
    row: Mapping[str, str],
    submissions: Sequence[Mapping[str, str]],
    *,
    compute_ceiling_path: Path,
) -> Tuple[str, str, str]:
    role = row["role"]
    expected_job = (root / "jobs" / f"{role}.job").resolve()
    expected_qsub = (root / "qsub_raw" / f"{role}.txt").resolve()
    if Path(row["job_file"]).resolve() != expected_job or Path(row["qsub_file"]).resolve() != expected_qsub:
        raise RuntimeError("setup submission path binding mismatch")
    if not expected_job.is_file() or not expected_qsub.is_file():
        raise RuntimeError("setup scheduler source evidence is missing")
    raw = expected_qsub.read_text(encoding="utf-8").strip()
    if re.fullmatch(rf"{re.escape(row['job_id'])}(?:\.[^\s]+)?", raw) is None:
        raise RuntimeError("setup qsub output differs from job identity")
    status_path = root / "qsub_raw" / f"{role}.status"
    if not status_path.is_file() or status_path.read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("setup qsub status does not prove success")
    script = expected_job.read_text(encoding="utf-8")
    project = Path(__file__).resolve().parents[2]
    run_tag = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:8]
    if role.startswith("plan_") and role != "plan_merge":
        replicate = "a" if role == "plan_a_001" else "b"
        segment = 1 if replicate == "a" else 2
        expected_name = f"tv_s{replicate}{segment}_{run_tag}"
        expected_script = (
            "#!/usr/bin/env bash\n"
            "#$ -cwd\n"
            f"#$ -N {expected_name}\n"
            "#$ -q campus2.q\n"
            "#$ -j y\n"
            f"#$ -o {root}/logs/{role}.$JOB_ID.$TASK_ID.log\n"
            "#$ -l h_rt=00:10:00\n"
            "#$ -l h_data=2G\n"
            "#$ -t 1-16\n"
            "#$ -tc 16\n"
            "set -euo pipefail\n"
            "export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
            f'cd "{project}"\n'
            f'"{APPROVED_PYTHON_BIN}" scripts/terminal_validation_array.py freeze-plan-fragment \\\n'
            '  --stage "smoke" --shard-index "${SGE_TASK_ID}" --shard-count "16" \\\n'
            f'  --output "{root}/plan_{replicate}/fragment_$(printf \'%03d\' ${{SGE_TASK_ID}}).json" \\\n'
            f'  --profile-output "{root}/profiles_{replicate}/fragment_$(printf \'%03d\' ${{SGE_TASK_ID}}).json"\n'
        )
    else:
        plan_ids = tuple(item["job_id"] for item in submissions if item["role"] != "plan_merge")
        if len(plan_ids) != 2:
            raise RuntimeError("setup merge dependencies are incomplete")
        expected_name = f"tv_smerge_{run_tag}"
        expected_script = (
            "#!/usr/bin/env bash\n"
            "#$ -cwd\n"
            f"#$ -N {expected_name}\n"
            "#$ -q campus2.q\n"
            "#$ -j y\n"
            f"#$ -o {root}/logs/plan_merge.$JOB_ID.log\n"
            "#$ -l h_rt=00:05:00\n"
            "#$ -l h_data=4G\n"
            f"#$ -hold_jid {','.join(plan_ids)}\n"
            "set -euo pipefail\n"
            "export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
            f'cd "{project}"\n'
            f'"{APPROVED_PYTHON_BIN}" scripts/terminal_validation_array.py merge-plan-fragments \\\n'
            f'  --stage "smoke" --replicate-a-dir "{root}/plan_a" \\\n'
            f'  --replicate-b-dir "{root}/plan_b" --shard-count "16" \\\n'
            f'  --output "{root}/terminal_smoke_manifest.json" --assembly-output "{root}/manifest_plan_assembly.json" \\\n'
            f'  --compute-ceiling "{compute_ceiling_path.resolve()}" --max-descriptors-per-subshard 450 \\\n'
            '  --queue "campus2.q" --h-rt-seconds "7200" --memory-bytes 8589934592 \\\n'
            '  --throttle "4" \\\n'
            f'  --profile-output "{root}/profiles_merge/merge.json"\n'
        )
    if script != expected_script:
        difference = "".join(difflib.unified_diff(
            expected_script.splitlines(keepends=True),
            script.splitlines(keepends=True),
            fromfile="expected",
            tofile="submitted",
            n=2,
        ))
        raise RuntimeError(f"setup job script semantics are not exact:\n{difference}")
    return (
        expected_name,
        execution.sha256_file(expected_job),
        execution.logical_hash((
            execution.sha256_file(expected_qsub),
            execution.sha256_file(status_path),
        )),
    )


def _validate_qacct(
    root: Path,
    row: Mapping[str, str],
    job_name: str,
) -> Tuple[Tuple[Dict[str, Any], ...], str]:
    role = row["role"]
    path = root / "qacct" / f"{role}.raw"
    if not path.is_file():
        raise RuntimeError("setup qacct evidence is missing")
    records = execution.parse_qacct_records(path.read_text(encoding="utf-8"))
    is_merge = role == "plan_merge"
    expected_count = 1 if is_merge else 16
    if len(records) != expected_count:
        raise RuntimeError("setup qacct coverage is incomplete")
    observed_tasks = set()
    normalized = []
    for record in records:
        required = {
            "jobnumber", "jobname", "qname", "hostname", "slots", "failed",
            "exit_status", "cpu", "ru_wallclock", "maxvmem",
        }
        if not required.issubset(record):
            raise RuntimeError("setup qacct record lacks required fields")
        if record["jobnumber"] != row["job_id"] or record["jobname"] != job_name:
            raise RuntimeError("setup qacct identity mismatch")
        if not execution._queue_matches("campus2.q", record["qname"]):
            raise RuntimeError("setup qacct queue mismatch")
        host = record["hostname"].split(".", 1)[0].lower()
        if host.startswith("login") or execution._COMPUTE_HOST_RE.fullmatch(host) is None:
            raise RuntimeError("setup task did not run on a compute node")
        if record["slots"] != "1" or record["failed"] != "0" or record["exit_status"] != "0":
            raise RuntimeError("setup qacct does not prove success")
        wall = execution._duration_seconds(record["ru_wallclock"])
        memory = execution._memory_bytes(record["maxvmem"])
        if wall > (MAX_MERGE_WALL_SECONDS if is_merge else MAX_FRAGMENT_WALL_SECONDS):
            raise RuntimeError("setup task exceeded its P2 wall-time gate")
        if memory > MAX_MEMORY_BYTES:
            raise RuntimeError("setup task exceeded its P2 memory gate")
        task_id = None
        if not is_merge:
            raw_task = record.get("taskid", "")
            if not raw_task.isdigit():
                raise RuntimeError("setup array task identity is malformed")
            task_id = int(raw_task)
            if task_id not in range(1, 17) or task_id in observed_tasks:
                raise RuntimeError("setup array task coverage is invalid")
            observed_tasks.add(task_id)
        normalized.append({
            "task_id": task_id,
            "hostname": host,
            "wall_seconds": wall,
            "cpu_seconds": execution._duration_seconds(record["cpu"]),
            "max_memory_bytes": memory,
        })
    if not is_merge and observed_tasks != set(range(1, 17)):
        raise RuntimeError("setup array task coverage differs from 1-16")
    return tuple(normalized), execution.sha256_file(path)


def audit_manifest_setup(
    setup_root: Path,
    *,
    compute_ceiling_path: Path,
    compute_ceiling_evidence_root: Path,
) -> Dict[str, Any]:
    """Audit the exact P2 smoke setup outputs and scheduler evidence."""

    root = setup_root.resolve()
    manifest, submissions, profile_hashes = _validate_outputs(
        root,
        compute_ceiling_path.resolve(),
        compute_ceiling_evidence_root.resolve(),
    )
    qstat_path = root / "final_qstat" / "snapshot.xml"
    qstat_status_path = root / "final_qstat" / "snapshot.status"
    if not qstat_path.is_file() or not qstat_status_path.is_file():
        raise RuntimeError("setup final qstat evidence is missing")
    status_text = qstat_status_path.read_text(encoding="utf-8").strip()
    if not status_text.isdigit():
        raise RuntimeError("setup qstat status is malformed")
    qstat_text = qstat_path.read_text(encoding="utf-8")
    usage = []
    source_hashes = []
    fragment_walls = []
    merge_wall = None
    for row in submissions:
        validate_qstat_absence_text(qstat_text, row["job_id"], int(status_text))
        job_name, job_hash, qsub_hash = _validate_job_script(
            root,
            row,
            submissions,
            compute_ceiling_path=compute_ceiling_path,
        )
        records, qacct_hash = _validate_qacct(root, row, job_name)
        role = row["role"]
        log_pattern = f"{role}.{row['job_id']}*.log"
        logs = tuple(sorted((root / "logs").glob(log_pattern)))
        expected_logs = 1 if role == "plan_merge" else 16
        if len(logs) != expected_logs:
            raise RuntimeError("setup log coverage is incomplete")
        for log in logs:
            lowered = log.read_text(encoding="utf-8", errors="replace").lower()
            if any(marker in lowered for marker in FORBIDDEN_LOG_MARKERS):
                raise RuntimeError("setup log contains a forbidden failure marker")
        walls = [record["wall_seconds"] for record in records]
        if role == "plan_merge":
            merge_wall = walls[0]
        else:
            fragment_walls.extend(walls)
        usage.append({
            "role": role,
            "job_id": row["job_id"],
            "job_script_sha256": job_hash,
            "qsub_raw_sha256": qsub_hash,
            "qacct_raw_sha256": qacct_hash,
            "task_usage": tuple(records),
        })
        source_hashes.extend((job_hash, qsub_hash, qacct_hash))
    if len(fragment_walls) != 32 or merge_wall is None:
        raise RuntimeError("setup resource coverage is incomplete")
    critical_path = max(fragment_walls) + merge_wall
    if critical_path > MAX_CRITICAL_PATH_SECONDS:
        raise RuntimeError("setup exceeded the P2 critical-path gate")
    result = {
        "schema": SETUP_AUDIT_SCHEMA,
        "audit_pass": True,
        "manifest_hash": manifest["manifest_hash"],
        "descriptor_count": int(manifest["expected_descriptor_count"]),
        "fragment_task_count": len(fragment_walls),
        "merge_task_count": 1,
        "maximum_fragment_wall_seconds": max(fragment_walls),
        "merge_wall_seconds": merge_wall,
        "critical_path_seconds": critical_path,
        "maximum_memory_bytes": max(
            record["max_memory_bytes"] for item in usage for record in item["task_usage"]
        ),
        "profile_sha256": tuple(profile_hashes),
        "scheduler_usage": tuple(usage),
        "final_qstat_sha256": execution.sha256_file(qstat_path),
        "compute_ceiling_sha256": execution.sha256_file(compute_ceiling_path),
        "compute_ceiling_raw_evidence_hash": execution.logical_hash(
            tuple(
                (name, execution.sha256_file(compute_ceiling_evidence_root / name))
                for name in COMPUTE_CEILING_EVIDENCE_FILES
            )
        ),
        "audit_hash": "",
    }
    result["audit_hash"] = execution.logical_hash(
        execution._without_hash(result, "audit_hash")
    )
    return result


__all__ = (
    "COMPUTE_CEILING_EVIDENCE_FILES",
    "SETUP_AUDIT_SCHEMA",
    "audit_manifest_setup",
    "create_compute_ceiling_report",
)
