#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.terminal_migration_evidence import (  # noqa: E402
    collect_runtime_probe_scheduler_evidence,
    validate_runtime_probe_scheduler_gate,
    write_runtime_probe_preflight,
)


def _add_approved_hash_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--approved-evidence-module-hash", required=True)
    parser.add_argument("--approved-collector-hash", required=True)
    parser.add_argument("--approved-probe-job-script-hash", required=True)
    parser.add_argument("--approved-submitter-hash", required=True)


def _approved_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "scripts/collect_hoffman2_runtime_profile_probe.py": (
            args.approved_collector_hash
        ),
        "scripts/hoffman2_terminal_runtime_profile_probe.job": (
            args.approved_probe_job_script_hash
        ),
        "scripts/submit_hoffman2_terminal_runtime_profile_probe.sh": (
            args.approved_submitter_hash
        ),
        "src/experiments/terminal_migration_evidence.py": (
            args.approved_evidence_module_hash
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or verify one four-file Hoffman2 runtime-profile probe."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--source-root", type=Path, required=True)
    preflight.add_argument("--evidence-root", type=Path, required=True)
    preflight.add_argument("--python-executable", required=True)
    preflight.add_argument("--conda-env-path", required=True)
    preflight.add_argument("--preflight", type=Path, required=True)
    _add_approved_hash_arguments(preflight)

    for command in ("collect", "validate-gate"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("--run-dir", type=Path, required=True)
        command_parser.add_argument("--qacct", type=Path, required=True)
        command_parser.add_argument("--submitted-job-id", required=True)
        command_parser.add_argument("--gate", type=Path, required=True)
        _add_approved_hash_arguments(command_parser)

    args = parser.parse_args()
    approved_hashes = _approved_hashes(args)
    if args.command == "preflight":
        write_runtime_probe_preflight(
            source_root=args.source_root,
            evidence_root=args.evidence_root,
            approved_probe_file_hashes=approved_hashes,
            python_executable=args.python_executable,
            conda_env_path=args.conda_env_path,
            preflight_path=args.preflight,
        )
        print(args.preflight)
    elif args.command == "collect":
        collect_runtime_probe_scheduler_evidence(
            run_dir=args.run_dir,
            qacct_path=args.qacct,
            submitted_job_id=args.submitted_job_id,
            approved_probe_file_hashes=approved_hashes,
            gate_path=args.gate,
        )
        print(args.gate)
    else:
        validate_runtime_probe_scheduler_gate(
            run_dir=args.run_dir,
            qacct_path=args.qacct,
            submitted_job_id=args.submitted_job_id,
            approved_probe_file_hashes=approved_hashes,
            gate_path=args.gate,
        )
        print(args.gate)


if __name__ == "__main__":
    main()
