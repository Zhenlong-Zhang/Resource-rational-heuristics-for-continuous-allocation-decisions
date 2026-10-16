#!/usr/bin/env python3
"""Run the single prespecified C3 revision with the unchanged bounded-task engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from design_validation.revision import run_revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-file", type=Path, default=ROOT/"configs/tutoring_conditions_baseline.json")
    parser.add_argument("--patch-file", type=Path, default=ROOT/"configs/tutoring_design_revision_v02.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("screening", "confirmation"), required=True)
    args = parser.parse_args()
    gate = run_revision(args.base_file, args.patch_file, args.output_dir, args.stage, ROOT)
    print(json.dumps(gate, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
