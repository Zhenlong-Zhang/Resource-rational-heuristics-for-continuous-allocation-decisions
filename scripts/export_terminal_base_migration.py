#!/usr/bin/env python3
from __future__ import annotations

"""One-time, read-only exporter for the reviewed R6 base-belief migration."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.terminal_base_migration import (  # noqa: E402
    DEFAULT_ORIGINAL_MANIFEST_PATH,
    export_authoritative_base_migration,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export authoritative R6 terminal base beliefs exactly once."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ORIGINAL_MANIFEST_PATH,
    )
    parser.add_argument("--execution-approval", type=Path, required=True)
    parser.add_argument("--execution-approval-file-hash", required=True)
    parser.add_argument("--scheduled-job-script", type=Path, required=True)
    args = parser.parse_args()
    output_hash = export_authoritative_base_migration(
        args.output,
        args.manifest,
        execution_approval_path=args.execution_approval,
        approved_execution_approval_file_hash=(
            args.execution_approval_file_hash
        ),
        scheduled_job_script_path=args.scheduled_job_script,
    )
    print(output_hash)


if __name__ == "__main__":
    main()
