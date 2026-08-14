#!/usr/bin/env python3
"""Audit two immutable targeted repeats and their raw Hoffman2 qacct evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from src.experiments.terminal_execution import (
    _duration_seconds,
    _memory_bytes,
    _validate_reference_b_runtime_evidence,
    parse_qacct_records,
)


TARGETS = ("base_72", "one_step_28517", "one_step_28715")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (_sha256(path), path.stat().st_size)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name not in {"runtime.json", "COMPLETE"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scheduler-root", type=Path, required=True)
    parser.add_argument("--qacct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requested-memory-bytes", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-queue", default="campus2.q")
    parser.add_argument("--expected-parallel-environment", default="shared")
    parser.add_argument("--expected-task-concurrency", type=int, default=4)
    parser.add_argument("--expected-execution-host", default="")
    args = parser.parse_args()

    if re.fullmatch(r"[A-Za-z0-9_.-]+\.q", args.expected_queue) is None:
        raise ValueError("expected queue name is invalid")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", args.expected_parallel_environment) is None:
        raise ValueError("expected parallel environment name is invalid")
    if args.expected_task_concurrency not in range(1, 7):
        raise ValueError("expected task concurrency must be between 1 and 6")
    if args.expected_execution_host and re.fullmatch(
        r"[A-Za-z0-9_.-]+", args.expected_execution_host
    ) is None:
        raise ValueError("expected execution host is invalid")

    qsub_raw = args.scheduler_root / "qsub.raw"
    qsub_status = args.scheduler_root / "qsub.status"
    job_id_path = args.scheduler_root / "job_id"
    job_script = args.scheduler_root / "targeted_concurrent.job"
    for required in (qsub_raw, qsub_status, job_id_path, job_script):
        if not required.is_file():
            raise RuntimeError(f"targeted scheduler evidence is missing: {required.name}")
    if qsub_status.read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("targeted qsub status is not successful")
    job_id = job_id_path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"([0-9]+)(?:\.[^\s]+)?", qsub_raw.read_text(encoding="utf-8").strip())
    if match is None or match.group(1) != job_id:
        raise RuntimeError("targeted qsub output and job identity disagree")
    job_text = job_script.read_text(encoding="utf-8")
    queue_selector = args.expected_queue + (
        f"@{args.expected_execution_host}" if args.expected_execution_host else ""
    )
    required_job_lines = (
        f"#$ -q {queue_selector}", "#$ -l h_rt=24:00:00",
        "#$ -l h_data=8589934592",
        "#$ -t 1-6", f"#$ -tc {args.expected_task_concurrency}",
        f"#$ -pe {args.expected_parallel_environment} 2",
        "scripts/run_terminal_targeted_concurrent.py",
    )
    if any(line not in job_text for line in required_job_lines):
        raise RuntimeError("targeted job script differs from the required scheduler shape")

    records = parse_qacct_records(args.qacct.read_text(encoding="utf-8"))
    by_task = {int(item["taskid"]): item for item in records if item.get("taskid", "").isdigit()}
    job_ids = {item.get("jobnumber") for item in records}
    findings = []
    targets = []
    for target_index, target in enumerate(TARGETS):
        first = args.root / target / "repeat_1"
        second = args.root / target / "repeat_2"
        inventories = []
        walls = []
        for repeat_index, path in enumerate((first, second), start=1):
            task_id = target_index * 2 + repeat_index
            runtime_path = path / "runtime.json"
            record = by_task.get(task_id)
            if not (path / "COMPLETE").is_file() or not runtime_path.is_file() or record is None:
                findings.append(f"missing_complete_runtime_or_qacct:{target}:{repeat_index}")
                continue
            runtime = json.loads(runtime_path.read_text(encoding="ascii"))
            wall = _duration_seconds(record["ru_wallclock"])
            qacct_memory = _memory_bytes(record["maxvmem"])
            walls.append(wall)
            runtime_evidence = runtime.get("reference_b_runtime_evidence", {})
            try:
                _validate_reference_b_runtime_evidence(runtime_evidence)
            except (TypeError, ValueError, RuntimeError):
                findings.append(f"worker_runtime_evidence_failed:{target}:{repeat_index}")
            worker_memory = 0
            if isinstance(runtime_evidence, dict):
                try:
                    worker_memory = (
                        int(runtime_evidence["traced_worker"]["peak_rss_bytes"])
                        + int(runtime_evidence["source_worker"]["peak_rss_bytes"])
                        + int(runtime_evidence["coordinator_peak_rss_bytes"])
                    )
                except (KeyError, TypeError, ValueError):
                    pass
            memory = max(qacct_memory, worker_memory)
            if any((
                record.get("failed") != "0", record.get("exit_status") != "0",
                record.get("qname") != args.expected_queue,
                record.get("slots") != "2",
                record.get("granted_pe") != args.expected_parallel_environment,
                runtime.get("slots") != 2,
                runtime.get("parallel_environment")
                    != args.expected_parallel_environment,
                tuple(tuple(item) for item in runtime.get("pe_host_slots", ()))
                    != ((runtime.get("hostname"), 2),),
                runtime.get("sge_task_id") != str(task_id),
                runtime.get("job_id") != record.get("jobnumber"),
                runtime.get("job_id") != job_id,
                runtime.get("hostname") != record.get("hostname", "").split(".", 1)[0].lower(),
                bool(args.expected_execution_host) and (
                    runtime.get("hostname")
                    != args.expected_execution_host.split(".", 1)[0].lower()
                ),
                wall > 21600.0, memory > 6 * 1024**3,
                memory > 0.75 * args.requested_memory_bytes,
            )):
                findings.append(f"scheduler_gate_failed:{target}:{repeat_index}")
            if target == "base_72" and wall > 18000.0:
                findings.append(f"base_72_runtime_failed:{repeat_index}")
            inventories.append(_inventory(path))
            comparable = json.loads((path / "comparable.json").read_text(encoding="ascii"))
            if comparable.get("source_commit") != args.expected_commit:
                findings.append(f"source_commit_failed:{target}:{repeat_index}")
        if len(inventories) == 2 and inventories[0] != inventories[1]:
            findings.append(f"exact_byte_mismatch:{target}")
        targets.append({"target": target, "qacct_wall_seconds": walls})
    if len(records) != 6 or set(by_task) != set(range(1, 7)):
        findings.append("qacct_coverage_not_exactly_1_to_6")
    if len(job_ids) != 1 or None in job_ids:
        findings.append("qacct_job_identity_not_unique")
    report = {
        "schema": "terminal_targeted_concurrent_audit_v1",
        "qacct_sha256": _sha256(args.qacct),
        "qsub_raw_sha256": _sha256(qsub_raw),
        "qsub_status_sha256": _sha256(qsub_status),
        "job_script_sha256": _sha256(job_script),
        "job_id": job_id,
        "expected_commit": args.expected_commit,
        "expected_queue": args.expected_queue,
        "expected_parallel_environment": args.expected_parallel_environment,
        "expected_task_concurrency": args.expected_task_concurrency,
        "expected_execution_host": args.expected_execution_host,
        "findings": findings,
        "targets": targets,
        "pass": not findings,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
