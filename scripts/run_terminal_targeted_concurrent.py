#!/usr/bin/env python3
"""Run one immutable concurrent Reference-B pathological validation task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.terminal_validation_array import load_provider
from src.experiments import terminal_execution as execution
from src.experiments.terminal_evidence_rows import (
    evaluate_terminal_evidence_descriptor,
    reconstruct_terminal_evidence_source,
    validate_terminal_evidence_bundle_structure,
)


TARGETS = {
    "base_72": ("base", 72, "393332d23deab3347e598b546595d086efaab8992714e7875be7d55f5c37398c"),
    "one_step_28517": ("one_step", 28517, "cb9fc958c198aed82b970c697755ba1c7fb2364e6b236879c1b249c462742fda"),
    "one_step_28715": ("one_step", 28715, "ff2310892899928d72c74fde885fc29c9d67a53c7856dc1d50b79366594948dc"),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-parallel-environment", default="shared")
    args = parser.parse_args()

    if not args.expected_parallel_environment or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in args.expected_parallel_environment
    ):
        raise ValueError("expected parallel environment name is invalid")

    if os.environ.get("NSLOTS") != "2" or not os.environ.get("PE_HOSTFILE"):
        raise RuntimeError("targeted validation requires a two-slot scheduler task")
    if not os.environ.get("JOB_ID", "").isdigit():
        raise RuntimeError("targeted validation requires a scheduler job ID")
    root = args.project_root.resolve()
    pe_hostfile = Path(os.environ["PE_HOSTFILE"])
    if not pe_hostfile.is_file():
        raise RuntimeError("targeted validation PE hostfile is missing")
    pe_rows = tuple(
        (fields[0].split(".", 1)[0].lower(), int(fields[1]))
        for fields in (line.split() for line in pe_hostfile.read_text(encoding="utf-8").splitlines())
        if fields
    )
    hostname = platform.node().split(".", 1)[0].lower()
    if pe_rows != ((hostname, 2),):
        raise RuntimeError(
            "targeted validation requires exactly two slots on the execution host; "
            f"hostname={hostname!r}, pe_rows={pe_rows!r}"
        )
    thread_names = execution.REFERENCE_B_THREAD_ENVIRONMENT
    if any(os.environ.get(name) != "1" for name in thread_names):
        raise RuntimeError("targeted validation numerical thread controls must equal one")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != args.expected_commit:
        raise RuntimeError("targeted source commit mismatch")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    ):
        raise RuntimeError("targeted validation requires a clean source checkout")

    suite_class, index, descriptor_hash = TARGETS[args.target]
    target = args.output_root / args.target / f"repeat_{args.repeat}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"immutable output already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    started = time.perf_counter()
    try:
        provider, accepted = load_provider()
        suites = execution.build_terminal_suites(provider, accepted, validate_contents=False)
        descriptor = next(
            item for item in suites[suite_class].descriptors if item.descriptor_index == index
        )
        if descriptor.descriptor_hash != descriptor_hash:
            raise RuntimeError("frozen targeted descriptor hash mismatch")
        source = execution.capture_clean_source_identity(root, execution.TERMINAL_SOURCE_PATHS)
        mdp, belief = reconstruct_terminal_evidence_source(descriptor, provider)
        runtime_evidence = {}
        bundle = evaluate_terminal_evidence_descriptor(
            descriptor,
            mdp,
            belief,
            concurrent_reference_b=True,
            reference_b_runtime_evidence=runtime_evidence,
        )
        failures = validate_terminal_evidence_bundle_structure(bundle, descriptor)
        if failures:
            raise RuntimeError("targeted bundle failed: " + ",".join(failures))

        rows = [execution._row_to_payload(row) for row in bundle.rows]
        rows_bytes = _canonical_bytes(rows)
        (temporary / "rows.json").write_bytes(rows_bytes)
        sidecars = []
        for relative, payload in bundle.sidecars:
            path = temporary / "sidecars" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            sidecars.append((relative, _sha256_bytes(payload), len(payload)))
        comparable = {
            "schema": "terminal_targeted_concurrent_comparable_v1",
            "source_commit": commit,
            "source_identity_hash": source["identity_hash"],
            "provider_hash": provider.provider_hash,
            "descriptor_hash": descriptor_hash,
            "methods": [row.method for row in bundle.rows],
            "rows_sha256": _sha256_bytes(rows_bytes),
            "sidecars": sidecars,
        }
        comparable_bytes = _canonical_bytes(comparable)
        (temporary / "comparable.json").write_bytes(comparable_bytes)
        runtime = {
            "schema": "terminal_targeted_concurrent_runtime_v1",
            "target": args.target,
            "repeat": args.repeat,
            "job_id": os.environ["JOB_ID"],
            "sge_task_id": os.environ.get("SGE_TASK_ID"),
            "hostname": hostname,
            "slots": 2,
            "parallel_environment": args.expected_parallel_environment,
            "pe_hostfile_sha256": execution.sha256_file(pe_hostfile),
            "pe_host_slots": pe_rows,
            "thread_environment": tuple((name, "1") for name in thread_names),
            "wall_seconds": time.perf_counter() - started,
            "comparable_sha256": _sha256_bytes(comparable_bytes),
            "reference_b_runtime_evidence": runtime_evidence,
        }
        (temporary / "runtime.json").write_bytes(_canonical_bytes(runtime))
        (temporary / "COMPLETE").write_text("complete\n", encoding="ascii")
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
