#!/usr/bin/env python3
"""Create the bounded compute-ceiling record used by terminal validation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import terminal_execution as execution  # noqa: E402
from src.experiments.terminal_setup_diagnostics import (  # noqa: E402
    create_compute_ceiling_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--queue", default="campus2.q")
    args = parser.parse_args()
    execution.write_new_json(
        args.output,
        create_compute_ceiling_report(args.evidence_root, queue=args.queue),
    )


if __name__ == "__main__":
    main()
