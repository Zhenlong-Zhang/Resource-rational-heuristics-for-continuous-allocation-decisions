#!/usr/bin/env python3
"""ENGAGEMENT schedules, JS parity/geometry and descriptive finite-model tables."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from design_validation.finite_auxiliary import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source-panel", "time-tables", "bounds-tables"))
    parser.add_argument("--design", type=Path, default=ROOT/"configs/tutoring_conditions.json")
    parser.add_argument("--aux-config", type=Path, default=ROOT/"configs/tutoring_finite_auxiliary.json")
    parser.add_argument("--config", type=Path, help="Matching time/bounds producer settings")
    parser.add_argument("--initial", type=Path, help="Generated time-initial or bounds-initial directory")
    parser.add_argument("--terminal", type=Path, help="Generated time-terminal or bounds-terminal directory")
    parser.add_argument("--root", type=Path, action="append", help="Generated bounds-root directory; repeat per template")
    parser.add_argument("--node", default="node", help="Node.js executable (source-panel only)")
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwritten")
    args = parser.parse_args()
    if args.stage != "source-panel" and args.config is None:
        name = "tutoring_terminal_time_v01.json" if args.stage == "time-tables" else "tutoring_optimality_v01.json"
        args.config = ROOT/"configs"/name
    try:
        result = run(args)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
