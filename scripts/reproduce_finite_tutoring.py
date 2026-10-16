#!/usr/bin/env python3
"""Paired horizons, conditional terminal-time sensitivity and numerical bounds."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from design_validation.finite_reproduction import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pairs", "horizon-initial", "time-initial",
                                         "time-terminal", "bounds-initial",
                                         "bounds-root", "bounds-terminal"))
    parser.add_argument("--design", type=Path, default=ROOT/"configs/tutoring_conditions.json")
    parser.add_argument("--config", type=Path, help="Scientific settings; defaults depend on stage")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--template", action="append", help="Template subset; root requires exactly one")
    parser.add_argument("--episodes", type=int, help="Pairs per template; default 400. Smaller panels are interface checks")
    parser.add_argument("--pairs", type=Path, help="Directory produced by the pairs stage; required for terminal stages")
    parser.add_argument("--max-seconds", type=float, help="Root computational cutoff, not a scientific horizon")
    parser.add_argument("--max-created-states", type=int, help="Root computational cutoff, including frontier")
    parser.add_argument("--max-depth", type=int, help="Root refinement cutoff; frontier keeps all affordable questions")
    args = parser.parse_args()
    names = {"pairs": "tutoring_paired_horizon.json", "horizon": "tutoring_paired_horizon.json",
             "time": "tutoring_terminal_time_v01.json", "bounds": "tutoring_optimality_v01.json"}
    args.config = args.config or ROOT/"configs"/names[args.stage.split("-")[0]]
    try:
        result = run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
