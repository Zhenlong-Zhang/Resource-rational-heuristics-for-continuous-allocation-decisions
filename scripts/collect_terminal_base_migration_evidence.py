#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.terminal_migration_evidence import (  # noqa: E402
    collect_migration_scheduler_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly cross-bind one terminal migration candidate to qacct."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qacct", type=Path, required=True)
    parser.add_argument("--submitted-job-id", required=True)
    parser.add_argument("--approved-execution-approval-file-hash", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    collect_migration_scheduler_evidence(
        run_dir=args.run_dir,
        qacct_path=args.qacct,
        submitted_job_id=args.submitted_job_id,
        approved_execution_approval_file_hash=(
            args.approved_execution_approval_file_hash
        ),
        gate_path=args.gate,
    )
    print(args.gate)


if __name__ == "__main__":
    main()
