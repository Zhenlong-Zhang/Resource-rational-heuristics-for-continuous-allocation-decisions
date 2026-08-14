#!/usr/bin/env python3
"""Run the StrategyMapping quadrature screening suite as 90 isolated array tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.positive_need_workflow import (  # noqa: E402
    IMPLEMENTATION_SOURCES,
    atomic_write_json,
    current_sge_task_id,
    diagnose_quadrature_orders,
    digest,
    exclusive_lock,
    git_clean,
    git_commit,
    git_tree_hash,
    package_version,
    parse_qacct_records,
    sha256_file,
    summarize_numerical_validation,
    utc_now,
)
from src.experiments.positive_need import (  # noqa: E402
    DEFAULT_SPEC_PATH,
    build_numerical_validation_cases,
    dense_numerical_validation_case_ids,
    load_positive_need_spec,
    validate_numerical_action_value_maps,
)


MANIFEST_NAME = "strategy_mapping_quadrature_diagnostic_manifest.json"
SHARD_NAME = "quadrature_case.json"
PROVISIONAL_RESULT_NAME = "strategy_mapping_quadrature_diagnostic_provisional.json"
FINAL_RESULT_NAME = "strategy_mapping_quadrature_diagnostic_final.json"
RESULT_NAME = PROVISIONAL_RESULT_NAME
QACCT_DIRECTORY_NAME = "qacct"
QACCT_EVIDENCE_NAME = "qacct_evidence.json"
ORDER_PAIRS = ((51, 101), (61, 121), (81, 161), (101, 201))
DIAGNOSTIC_SOURCES = tuple(
    sorted(
        {
            *IMPLEMENTATION_SOURCES,
            "scripts/quadrature_validation_array.py",
            "scripts/submit_hoffman2_quadrature_validation.sh",
        }
    )
)


def diagnostic_source_hashes() -> Dict[str, str]:
    missing = [
        relative
        for relative in DIAGNOSTIC_SOURCES
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise RuntimeError(f"diagnostic source files are missing: {missing}")
    return {
        relative: sha256_file(PROJECT_ROOT / relative) for relative in DIAGNOSTIC_SOURCES
    }


def current_runtime() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "hostname": platform.node(),
    }


def manifest_hash(manifest: Mapping[str, object]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return digest(payload)


def load_diagnostic_manifest(path: Path) -> Dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != manifest_hash(manifest):
        raise RuntimeError("quadrature diagnostic manifest hash mismatch")
    expected_pairs = [list(pair) for pair in ORDER_PAIRS]
    if manifest.get("analysis") != "quadrature_order_screening":
        raise RuntimeError("unexpected quadrature diagnostic analysis")
    if manifest.get("diagnostic_only") is not True:
        raise RuntimeError("quadrature screening must remain diagnostic-only")
    if manifest.get("automatic_production_order_freeze") is not False:
        raise RuntimeError("quadrature screening cannot freeze a production order")
    if manifest.get("order_pairs") != expected_pairs:
        raise RuntimeError("quadrature diagnostic order-pair set mismatch")
    cases = list(manifest.get("numerical_cases", []))
    if len(cases) != 90 or [int(case["case_id"]) for case in cases] != list(range(90)):
        raise RuntimeError("quadrature diagnostic case set must be exactly 0..89")
    if manifest.get("numerical_cases_hash") != digest(cases):
        raise RuntimeError("quadrature diagnostic case-suite hash mismatch")
    dense_case_ids = list(manifest.get("dense_reference_case_ids", []))
    if (
        len(dense_case_ids) != 36
        or len(set(map(int, dense_case_ids))) != 36
        or any(not 0 <= int(case_id) < 90 for case_id in dense_case_ids)
    ):
        raise RuntimeError("quadrature diagnostic dense-reference case set is invalid")
    if manifest.get("dense_reference_case_ids_hash") != digest(dense_case_ids):
        raise RuntimeError("quadrature diagnostic dense-reference case hash mismatch")
    tasks = list(manifest.get("tasks", []))
    expected_tasks = [
        {"task_index": case_id, "case_id": case_id} for case_id in range(90)
    ]
    if tasks != expected_tasks or int(manifest.get("task_count", -1)) != 90:
        raise RuntimeError("quadrature diagnostic task map mismatch")
    if int(manifest.get("expected_rows_per_task", -1)) != 4:
        raise RuntimeError("quadrature diagnostic row-count contract mismatch")
    return manifest


def _require_output_outside_checkout(output_dir: Path) -> None:
    resolved = output_dir.expanduser().resolve()
    root = PROJECT_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        raise RuntimeError("quadrature diagnostic output must be outside the clean checkout")


def create_diagnostic_manifest(output_dir: Path, require_clean: bool = True) -> Path:
    _require_output_outside_checkout(output_dir)
    if require_clean and not git_clean():
        raise RuntimeError("quadrature diagnostic manifest requires a clean worktree")
    spec = load_positive_need_spec()
    cases = build_numerical_validation_cases(spec)
    if len(cases) != 90 or [int(case["case_id"]) for case in cases] != list(range(90)):
        raise RuntimeError("the frozen numerical suite must contain cases 0..89 exactly")
    dense_case_ids = dense_numerical_validation_case_ids(cases, spec)
    manifest: Dict[str, object] = {
        "schema_version": 2,
        "analysis": "quadrature_order_screening",
        "diagnostic_only": True,
        "automatic_production_order_freeze": False,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "git_tree_hash": git_tree_hash(),
        "git_clean_at_creation": git_clean(),
        "spec_path": str(DEFAULT_SPEC_PATH.relative_to(PROJECT_ROOT)),
        "spec_hash": sha256_file(DEFAULT_SPEC_PATH),
        "source_hashes": diagnostic_source_hashes(),
        "runtime": current_runtime(),
        "order_pairs": [list(pair) for pair in ORDER_PAIRS],
        "frozen_numerical_settings": dict(spec["numerical_settings"]),
        "numerical_cases": cases,
        "numerical_cases_hash": digest(cases),
        "dense_reference_case_ids": dense_case_ids,
        "dense_reference_case_ids_hash": digest(dense_case_ids),
        "expected_rows_per_task": 4,
        "task_count": 90,
        "tasks": [
            {"task_index": case_id, "case_id": case_id} for case_id in range(90)
        ],
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / MANIFEST_NAME
    atomic_write_json(path, manifest)
    return path


def validate_diagnostic_source_identity(manifest: Mapping[str, object]) -> None:
    if not git_clean():
        raise RuntimeError("quadrature diagnostic execution requires a clean worktree")
    if git_commit() != manifest["git_commit"] or git_tree_hash() != manifest["git_tree_hash"]:
        raise RuntimeError("quadrature diagnostic commit/tree identity mismatch")
    if diagnostic_source_hashes() != manifest["source_hashes"]:
        raise RuntimeError("quadrature diagnostic source hashes differ from the manifest")
    if sha256_file(DEFAULT_SPEC_PATH) != manifest["spec_hash"]:
        raise RuntimeError("quadrature diagnostic frozen spec differs from the manifest")
    spec = load_positive_need_spec()
    if dict(spec["numerical_settings"]) != manifest["frozen_numerical_settings"]:
        raise RuntimeError("quadrature diagnostic numerical settings mismatch")
    cases = build_numerical_validation_cases(spec)
    if cases != manifest["numerical_cases"] or digest(cases) != manifest["numerical_cases_hash"]:
        raise RuntimeError("quadrature diagnostic frozen case suite differs from the manifest")
    dense_case_ids = dense_numerical_validation_case_ids(cases, spec)
    if (
        dense_case_ids != manifest["dense_reference_case_ids"]
        or digest(dense_case_ids) != manifest["dense_reference_case_ids_hash"]
    ):
        raise RuntimeError("quadrature diagnostic frozen dense-reference subset differs")


def shard_directory(manifest_path: Path, task_index: int) -> Path:
    return manifest_path.parent / "shards" / f"case_{task_index:03d}"


def _expected_pass(row: Mapping[str, object], numerical: Mapping[str, object]) -> bool:
    value_tolerance = float(numerical["action_value_convergence_tolerance"])
    allocation_tolerance = float(numerical["allocation_convergence_tolerance"])
    return (
        float(row["gh_max_action_value_error"]) <= value_tolerance
        and row["gh_action"] == row["gh_reference_action"]
        and float(row["terminal_grid_allocation_error"]) <= allocation_tolerance
        and float(row["terminal_grid_value_error"]) <= value_tolerance
        and row["gh_action"] == row["terminal_reference_action"]
        and float(row["dense_reference_error"]) <= value_tolerance
        and row["gh_action"] == row["dense_reference_action"]
    )


def _validate_rows(
    manifest: Mapping[str, object], case_id: int, rows: Sequence[Mapping[str, object]]
) -> None:
    if len(rows) != 4:
        raise RuntimeError(f"quadrature case {case_id} must contain exactly four pair rows")
    expected_case = dict(manifest["numerical_cases"][case_id])  # type: ignore[index]
    observed_pairs = []
    expected_dense = case_id in set(map(int, manifest["dense_reference_case_ids"]))
    numerical = dict(manifest["frozen_numerical_settings"])  # type: ignore[arg-type]
    finite_fields = (
        "gh_max_action_value_error",
        "terminal_grid_allocation_error",
        "terminal_grid_value_error",
        "dense_reference_error",
        "dense_reference_performed",
        "passed",
    )
    for row in rows:
        validate_numerical_action_value_maps(row)
        for field, expected in expected_case.items():
            if row.get(field) != expected:
                raise RuntimeError(f"quadrature case provenance mismatch: {field}")
        if row.get("diagnostic_only") is not True:
            raise RuntimeError("quadrature row is not labeled diagnostic-only")
        observed_pairs.append((int(row["gh_order"]), int(row["gh_reference_order"])))
        for field in finite_fields:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite quadrature diagnostic field: {field}")
        observed_dense = float(row["dense_reference_performed"]) >= 0.5
        if observed_dense != expected_dense:
            raise RuntimeError("quadrature dense-reference classification mismatch")
        expected_pass = 1.0 if _expected_pass(row, numerical) else 0.0
        if float(row["passed"]) != expected_pass:
            raise RuntimeError("quadrature diagnostic pass flag does not match frozen criteria")
    if observed_pairs != list(ORDER_PAIRS):
        raise RuntimeError(f"quadrature case {case_id} order-pair rows are incomplete")


def validate_diagnostic_shard(
    manifest_path: Path, task_index: int
) -> Dict[str, object]:
    manifest = load_diagnostic_manifest(manifest_path)
    if not 0 <= task_index < 90:
        raise ValueError("quadrature diagnostic task index is outside 0..89")
    path = shard_directory(manifest_path, task_index) / SHARD_NAME
    if not path.is_file():
        raise RuntimeError(f"quadrature diagnostic shard is missing: {task_index}")
    shard = json.loads(path.read_text(encoding="utf-8"))
    shard_hash = shard.pop("shard_hash", None)
    if shard_hash != digest(shard):
        raise RuntimeError(f"quadrature diagnostic shard hash mismatch: {task_index}")
    shard["shard_hash"] = shard_hash
    if (
        shard.get("diagnostic_only") is not True
        or shard.get("manifest_hash") != manifest["manifest_hash"]
        or shard.get("git_commit") != manifest["git_commit"]
        or int(shard.get("task_index", -1)) != task_index
        or int(shard.get("case_id", -1)) != task_index
    ):
        raise RuntimeError(f"quadrature diagnostic shard provenance mismatch: {task_index}")
    rows = list(shard.get("rows", []))
    if shard.get("rows_hash") != digest(rows):
        raise RuntimeError(f"quadrature diagnostic row hash mismatch: {task_index}")
    _validate_rows(manifest, task_index, rows)
    metadata = dict(shard.get("scheduler_metadata", {}))
    if metadata.get("sge_task_id") not in ("", str(task_index + 1)):
        raise RuntimeError(f"quadrature diagnostic SGE task mismatch: {task_index}")
    runtime = dict(shard.get("runtime", {}))
    frozen_runtime = dict(manifest["runtime"])  # type: ignore[arg-type]
    if runtime.get("python") != frozen_runtime.get("python"):
        raise RuntimeError(f"quadrature task {task_index} Python version mismatch")
    if runtime.get("numpy") != frozen_runtime.get("numpy"):
        raise RuntimeError(f"quadrature task {task_index} NumPy version mismatch")
    if not runtime.get("hostname") or runtime.get("hostname") != metadata.get("hostname"):
        raise RuntimeError(f"quadrature task {task_index} hostname provenance mismatch")
    return shard


def run_diagnostic_task(manifest_path: Path, task_index: int) -> None:
    manifest = load_diagnostic_manifest(manifest_path)
    validate_diagnostic_source_identity(manifest)
    if not 0 <= task_index < 90:
        raise ValueError("quadrature diagnostic task index is outside 0..89")
    output = shard_directory(manifest_path, task_index)
    lock = output.with_suffix(".lock")
    with exclusive_lock(lock):
        if output.exists():
            try:
                validate_diagnostic_shard(manifest_path, task_index)
                return
            except Exception:
                shutil.rmtree(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".case_{task_index:03d}.", dir=output.parent)
        )
        try:
            rows = diagnose_quadrature_orders(task_index, ORDER_PAIRS)
            _validate_rows(manifest, task_index, rows)
            payload: Dict[str, object] = {
                "schema_version": 1,
                "diagnostic_only": True,
                "manifest_hash": manifest["manifest_hash"],
                "git_commit": manifest["git_commit"],
                "task_index": task_index,
                "case_id": task_index,
                "scheduler_metadata": {
                    "job_id": os.environ.get("JOB_ID", "").strip(),
                    "sge_task_id": current_sge_task_id(),
                    "hostname": platform.node(),
                },
                "runtime": current_runtime(),
                "rows": rows,
                "rows_hash": digest(rows),
            }
            payload["shard_hash"] = digest(payload)
            atomic_write_json(temporary / SHARD_NAME, payload)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def _validate_scheduler_evidence(
    manifest_path: Path, manifest: Mapping[str, object]
) -> Dict[str, object]:
    run_dir = manifest_path.parent
    path = run_dir / "scheduler" / "jobs.json"
    if not path.is_file():
        raise RuntimeError("quadrature scheduler evidence is missing")
    scheduler = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "array_job_id",
        "collector_job_id",
        "array_job_name",
        "collector_job_name",
        "throttle",
        "queue",
        "task_count",
        "task_slots",
        "collector_slots",
        "array_job_path",
        "collector_job_path",
        "array_job_sha256",
        "collector_job_sha256",
        "submission_evidence",
    }
    if not required.issubset(scheduler):
        raise RuntimeError("quadrature scheduler evidence fields are incomplete")
    if int(scheduler["task_count"]) != 90 or int(scheduler["task_slots"]) != 1:
        raise RuntimeError("quadrature scheduler must use 90 one-slot tasks")
    if int(scheduler["collector_slots"]) != 1 or int(scheduler["throttle"]) < 1:
        raise RuntimeError("quadrature scheduler slot/throttle evidence is invalid")
    if scheduler["array_job_name"] == scheduler["collector_job_name"]:
        raise RuntimeError("quadrature scheduler job names must be unique")
    if not all(
        re.fullmatch(r"r6qd[ac][0-9]{10,}", str(scheduler[field]))
        for field in ("array_job_name", "collector_job_name")
    ):
        raise RuntimeError("quadrature scheduler job-name format is not unique")
    for role in ("array", "collector"):
        job_path = run_dir / str(scheduler[f"{role}_job_path"])
        if not job_path.is_file() or sha256_file(job_path) != scheduler[f"{role}_job_sha256"]:
            raise RuntimeError(f"quadrature {role} job-file evidence mismatch")
    array_text = (run_dir / str(scheduler["array_job_path"])).read_text(encoding="utf-8")
    collector_text = (run_dir / str(scheduler["collector_job_path"])).read_text(
        encoding="utf-8"
    )
    if "#$ -t 1-90" not in array_text or f"#$ -tc {scheduler['throttle']}" not in array_text:
        raise RuntimeError("quadrature array range or throttle mismatch")
    if f"#$ -hold_jid {scheduler['array_job_id']}" not in collector_text:
        raise RuntimeError("quadrature collector dependency mismatch")
    if "#$ -pe " in array_text or "#$ -pe " in collector_text:
        raise RuntimeError("quadrature diagnostics must use one-slot jobs")
    evidence = dict(scheduler["submission_evidence"])
    for relative, metadata in evidence.items():
        evidence_path = run_dir / relative
        if (
            not evidence_path.is_file()
            or sha256_file(evidence_path) != metadata["sha256"]
            or evidence_path.stat().st_size != int(metadata["bytes"])
        ):
            raise RuntimeError(f"quadrature raw qsub evidence mismatch: {relative}")
    expected_stdout = {
        "array": re.compile(rf"^{re.escape(str(scheduler['array_job_id']))}\.1-90:1\s*$"),
        "collector": re.compile(rf"^{re.escape(str(scheduler['collector_job_id']))}\s*$"),
    }
    for role, pattern in expected_stdout.items():
        relative = f"scheduler/submission_evidence/{role}.qsub.stdout"
        if relative not in evidence:
            raise RuntimeError(f"quadrature {role} terse qsub evidence is missing")
        if not pattern.fullmatch((run_dir / relative).read_text(encoding="utf-8")):
            raise RuntimeError(f"quadrature {role} terse qsub evidence mismatch")
        meta_relative = f"scheduler/submission_evidence/{role}.qsub.meta"
        if meta_relative not in evidence:
            raise RuntimeError(f"quadrature {role} qsub metadata is missing")
        meta = (run_dir / meta_relative).read_text(encoding="utf-8")
        if "qsub_exit_status=0" not in meta:
            raise RuntimeError(f"quadrature {role} qsub did not exit successfully")
        expected_name = scheduler[f"{role}_job_name"]
        if f"job_name={expected_name}" not in meta:
            raise RuntimeError(f"quadrature {role} qsub job-name evidence mismatch")
    return scheduler


def collect_diagnostic(manifest_path: Path) -> Path:
    with exclusive_lock(manifest_path.parent / ".quadrature_collect.lock"):
        return _collect_diagnostic_locked(manifest_path)


def _collect_diagnostic_locked(manifest_path: Path) -> Path:
    manifest = load_diagnostic_manifest(manifest_path)
    validate_diagnostic_source_identity(manifest)
    scheduler = _validate_scheduler_evidence(manifest_path, manifest)
    all_rows, shard_provenance = _validated_rows_and_provenance(
        manifest_path, scheduler
    )
    aggregates = _aggregate_diagnostic_rows(all_rows)
    scheduler_path = manifest_path.parent / "scheduler" / "jobs.json"
    result: Dict[str, object] = {
        "schema_version": 2,
        "analysis": "quadrature_order_screening",
        "diagnostic_only": True,
        "automatic_production_order_freeze": False,
        "finalized": False,
        "evidence_status": "provisional_pending_qacct",
        "interpretation_status": "diagnostic_evidence_only_requires_separate_review",
        "collected_at": utc_now(),
        "manifest_hash": manifest["manifest_hash"],
        "git_commit": manifest["git_commit"],
        "git_tree_hash": manifest["git_tree_hash"],
        "spec_hash": manifest["spec_hash"],
        "source_hashes": manifest["source_hashes"],
        "manifest_runtime": manifest["runtime"],
        "numerical_cases_hash": manifest["numerical_cases_hash"],
        "frozen_numerical_settings": manifest["frozen_numerical_settings"],
        "order_pairs": manifest["order_pairs"],
        "scheduler_evidence": {
            "path": str(scheduler_path.relative_to(manifest_path.parent)),
            "sha256": sha256_file(scheduler_path),
            "bytes": scheduler_path.stat().st_size,
            "record": scheduler,
        },
        "shard_provenance": shard_provenance,
        "per_case": all_rows,
        "aggregate": aggregates,
    }
    output = manifest_path.parent / PROVISIONAL_RESULT_NAME
    if output.exists():
        raise RuntimeError("quadrature provisional result already exists; refusing overwrite")
    atomic_write_json(output, result)
    return output


def _validated_rows_and_provenance(
    manifest_path: Path, scheduler: Mapping[str, object]
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    all_rows: List[Dict[str, object]] = []
    provenance: List[Dict[str, object]] = []
    for task_index in range(90):
        shard = validate_diagnostic_shard(manifest_path, task_index)
        metadata = dict(shard["scheduler_metadata"])
        if metadata.get("job_id") != str(scheduler["array_job_id"]):
            raise RuntimeError(f"quadrature task {task_index} array-job binding mismatch")
        if metadata.get("sge_task_id") != str(task_index + 1):
            raise RuntimeError(f"quadrature task {task_index} array-task binding mismatch")
        all_rows.extend(dict(row) for row in shard["rows"])  # type: ignore[union-attr]
        shard_path = shard_directory(manifest_path, task_index) / SHARD_NAME
        provenance.append(
            {
                "task_index": task_index,
                "case_id": task_index,
                "path": str(shard_path.relative_to(manifest_path.parent)),
                "file_sha256": sha256_file(shard_path),
                "bytes": shard_path.stat().st_size,
                "shard_hash": shard["shard_hash"],
                "rows_hash": shard["rows_hash"],
                "runtime": shard["runtime"],
                "scheduler_metadata": shard["scheduler_metadata"],
            }
        )
    if len(all_rows) != 360:
        raise RuntimeError("quadrature collector requires exactly 360 per-case rows")
    return all_rows, provenance


def _aggregate_diagnostic_rows(
    all_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    aggregates = []
    for primary, reference in ORDER_PAIRS:
        rows = [
            row
            for row in all_rows
            if (int(row["gh_order"]), int(row["gh_reference_order"]))
            == (primary, reference)
        ]
        if [int(row["case_id"]) for row in rows] != list(range(90)):
            raise RuntimeError("quadrature aggregate case coverage is incomplete")
        dense_count = sum(float(row["dense_reference_performed"]) >= 0.5 for row in rows)
        if dense_count != 36:
            raise RuntimeError("quadrature aggregate must contain 36 dense references")
        aggregates.append(
            {
                "diagnostic_only": True,
                "gh_order": primary,
                "gh_reference_order": reference,
                "dense_reference_case_count": dense_count,
                **summarize_numerical_validation(rows),
            }
        )
    return aggregates


def _validate_provisional_result(
    manifest_path: Path,
) -> tuple[Dict[str, object], Dict[str, object]]:
    manifest = load_diagnostic_manifest(manifest_path)
    scheduler = _validate_scheduler_evidence(manifest_path, manifest)
    path = manifest_path.parent / PROVISIONAL_RESULT_NAME
    if not path.is_file():
        raise RuntimeError("quadrature provisional result is missing")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("manifest_hash") != manifest["manifest_hash"]
        or result.get("diagnostic_only") is not True
        or result.get("automatic_production_order_freeze") is not False
        or result.get("finalized") is not False
        or result.get("evidence_status") != "provisional_pending_qacct"
    ):
        raise RuntimeError("quadrature provisional result provenance mismatch")
    scheduler_path = manifest_path.parent / "scheduler" / "jobs.json"
    observed_scheduler = dict(result.get("scheduler_evidence", {}))
    if (
        observed_scheduler.get("sha256") != sha256_file(scheduler_path)
        or observed_scheduler.get("bytes") != scheduler_path.stat().st_size
        or observed_scheduler.get("record") != scheduler
    ):
        raise RuntimeError("quadrature provisional scheduler evidence mismatch")
    rows, provenance = _validated_rows_and_provenance(manifest_path, scheduler)
    aggregates = _aggregate_diagnostic_rows(rows)
    if digest(result.get("per_case")) != digest(rows):
        raise RuntimeError("quadrature provisional rows do not match the shards")
    if digest(result.get("aggregate")) != digest(aggregates):
        raise RuntimeError("quadrature provisional aggregates do not match the shards")
    if digest(result.get("shard_provenance")) != digest(provenance):
        raise RuntimeError("quadrature provisional shard provenance mismatch")
    return result, scheduler


def _validate_qacct_records(
    records: Sequence[Mapping[str, str]],
    job_id: str,
    job_name: str,
    queue: str,
    expected_task_ids: set[str] | None,
) -> None:
    required = {
        "qname",
        "hostname",
        "jobname",
        "jobnumber",
        "slots",
        "failed",
        "exit_status",
    }
    if not records or any(not required.issubset(record) for record in records):
        raise RuntimeError(f"qacct record is missing required fields for job {job_id}")
    if any(record["jobnumber"] != job_id for record in records):
        raise RuntimeError(f"qacct job ID mismatch for job {job_id}")
    if any(record["jobname"] != job_name for record in records):
        raise RuntimeError(f"qacct job name mismatch for job {job_id}")
    if any(record["qname"].split("@", 1)[0] != queue for record in records):
        raise RuntimeError(f"qacct queue mismatch for job {job_id}")
    if any(record["failed"] != "0" or record["exit_status"] != "0" for record in records):
        raise RuntimeError(f"qacct reports a failed task for job {job_id}")
    if any(record["slots"] != "1" for record in records):
        raise RuntimeError(f"qacct slot count mismatch for job {job_id}")
    task_ids = [
        record["taskid"]
        for record in records
        if record.get("taskid") not in (None, "undefined", "NONE")
    ]
    if expected_task_ids is None:
        if len(records) != 1 or task_ids:
            raise RuntimeError("qacct collector record must be exactly one non-array task")
    elif set(task_ids) != expected_task_ids or len(task_ids) != len(expected_task_ids):
        raise RuntimeError("qacct array task coverage must be exactly 1..90")


def audit_qacct(manifest_path: Path, job_ids: Sequence[str]) -> Path:
    with exclusive_lock(manifest_path.parent / ".quadrature_qacct.lock"):
        return _audit_qacct_locked(manifest_path, job_ids)


def _audit_qacct_locked(manifest_path: Path, job_ids: Sequence[str]) -> Path:
    manifest = load_diagnostic_manifest(manifest_path)
    validate_diagnostic_source_identity(manifest)
    _, scheduler = _validate_provisional_result(manifest_path)
    expected_ids = [
        str(scheduler["array_job_id"]),
        str(scheduler["collector_job_id"]),
    ]
    if set(map(str, job_ids)) != set(expected_ids) or len(job_ids) != 2:
        raise RuntimeError("qacct audit must cover the array and collector jobs exactly")
    target = manifest_path.parent / QACCT_DIRECTORY_NAME
    if target.exists():
        raise RuntimeError("quadrature qacct evidence already exists; refusing overwrite")
    temporary = Path(
        tempfile.mkdtemp(prefix=".qacct.", dir=manifest_path.parent)
    )
    try:
        raw_dir = temporary / "raw"
        raw_dir.mkdir()
        job_evidence = []
        for role, job_id in zip(("array", "collector"), expected_ids):
            output = subprocess.check_output(["qacct", "-j", job_id], text=True)
            raw_path = raw_dir / f"{role}_job_{job_id}.txt"
            raw_path.write_text(output, encoding="utf-8")
            records = parse_qacct_records(output)
            _validate_qacct_records(
                records,
                job_id,
                str(scheduler[f"{role}_job_name"]),
                str(scheduler["queue"]),
                {str(index) for index in range(1, 91)} if role == "array" else None,
            )
            job_evidence.append(
                {
                    "role": role,
                    "job_id": job_id,
                    "job_name": scheduler[f"{role}_job_name"],
                    "raw_path": f"{QACCT_DIRECTORY_NAME}/raw/{raw_path.name}",
                    "raw_sha256": sha256_file(raw_path),
                    "bytes": raw_path.stat().st_size,
                    "records": records,
                }
            )
        evidence = {
            "schema_version": 1,
            "audited_at": utc_now(),
            "manifest_hash": manifest["manifest_hash"],
            "scheduler_evidence_sha256": sha256_file(
                manifest_path.parent / "scheduler" / "jobs.json"
            ),
            "jobs": job_evidence,
        }
        atomic_write_json(temporary / QACCT_EVIDENCE_NAME, evidence)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target / QACCT_EVIDENCE_NAME


def validate_qacct_evidence(
    manifest_path: Path, scheduler: Mapping[str, object]
) -> Dict[str, object]:
    directory = manifest_path.parent / QACCT_DIRECTORY_NAME
    evidence_path = directory / QACCT_EVIDENCE_NAME
    if not evidence_path.is_file():
        raise RuntimeError("quadrature qacct evidence is missing")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("manifest_hash") != load_diagnostic_manifest(manifest_path)["manifest_hash"]:
        raise RuntimeError("quadrature qacct manifest mismatch")
    scheduler_path = manifest_path.parent / "scheduler" / "jobs.json"
    if evidence.get("scheduler_evidence_sha256") != sha256_file(scheduler_path):
        raise RuntimeError("quadrature qacct scheduler hash mismatch")
    jobs = list(evidence.get("jobs", []))
    if len(jobs) != 2 or [job.get("role") for job in jobs] != ["array", "collector"]:
        raise RuntimeError("quadrature qacct job set is incomplete")
    for job in jobs:
        role = str(job["role"])
        raw_path = manifest_path.parent / str(job["raw_path"])
        if (
            not raw_path.is_file()
            or sha256_file(raw_path) != job["raw_sha256"]
            or raw_path.stat().st_size != int(job["bytes"])
        ):
            raise RuntimeError(f"quadrature qacct raw evidence mismatch: {role}")
        records = parse_qacct_records(raw_path.read_text(encoding="utf-8"))
        if records != job["records"]:
            raise RuntimeError(f"quadrature qacct parsed evidence mismatch: {role}")
        _validate_qacct_records(
            records,
            str(scheduler[f"{role}_job_id"]),
            str(scheduler[f"{role}_job_name"]),
            str(scheduler["queue"]),
            {str(index) for index in range(1, 91)} if role == "array" else None,
        )
    return evidence


def finalize_diagnostic(manifest_path: Path) -> Path:
    with exclusive_lock(manifest_path.parent / ".quadrature_finalize.lock"):
        manifest = load_diagnostic_manifest(manifest_path)
        validate_diagnostic_source_identity(manifest)
        provisional, scheduler = _validate_provisional_result(manifest_path)
        qacct = validate_qacct_evidence(manifest_path, scheduler)
        output = manifest_path.parent / FINAL_RESULT_NAME
        if output.exists():
            raise RuntimeError("quadrature final result already exists; refusing overwrite")
        provisional_path = manifest_path.parent / PROVISIONAL_RESULT_NAME
        qacct_path = (
            manifest_path.parent / QACCT_DIRECTORY_NAME / QACCT_EVIDENCE_NAME
        )
        final = dict(provisional)
        final.update(
            {
                "schema_version": 3,
                "finalized": True,
                "finalized_at": utc_now(),
                "evidence_status": "finalized_qacct_validated",
                "provisional_result_sha256": sha256_file(provisional_path),
                "qacct_evidence": {
                    "path": str(qacct_path.relative_to(manifest_path.parent)),
                    "sha256": sha256_file(qacct_path),
                    "bytes": qacct_path.stat().st_size,
                    "record": qacct,
                },
            }
        )
        atomic_write_json(output, final)
        return output


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--allow-dirty", action="store_true")
    run = commands.add_parser("run-task")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--manifest", type=Path, required=True)
    audit = commands.add_parser("audit-qacct")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--job-id", action="append", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--manifest", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "create":
        print(create_diagnostic_manifest(args.output_dir, require_clean=not args.allow_dirty))
    elif args.command == "run-task":
        run_diagnostic_task(args.manifest, args.task_index)
    elif args.command == "collect":
        print(collect_diagnostic(args.manifest))
    elif args.command == "audit-qacct":
        print(audit_qacct(args.manifest, args.job_id))
    elif args.command == "finalize":
        print(finalize_diagnostic(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
