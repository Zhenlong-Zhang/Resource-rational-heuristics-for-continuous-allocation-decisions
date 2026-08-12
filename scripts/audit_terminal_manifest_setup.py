#!/usr/bin/env python3
"""Audit a completed Hoffman2 dual-replicate terminal manifest setup."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.terminal_execution import write_new_json  # noqa: E402
from src.experiments.terminal_setup_diagnostics import audit_manifest_setup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-root", type=Path, required=True)
    parser.add_argument("--compute-ceiling", type=Path, required=True)
    parser.add_argument("--compute-ceiling-evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_new_json(
        args.output,
        audit_manifest_setup(
            args.setup_root,
            compute_ceiling_path=args.compute_ceiling,
            compute_ceiling_evidence_root=args.compute_ceiling_evidence_root,
        ),
    )


if __name__ == "__main__":
    main()
