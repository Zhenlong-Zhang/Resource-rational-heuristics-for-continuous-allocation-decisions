#!/usr/bin/env python3
"""Reproduce the historical algebra fixture or map existing public results.

The fixture stage calls the existing public allocation/inference functions on
the original deterministic 25-person/250-row recipe. The separate example stage
only writes illustrative raw histories. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fixture", "example", "map"))
    parser.add_argument("--analysis", type=Path, help="Public nominal result.json; required for map")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/h6_h7_surplus.json")
    parser.add_argument("--fixture-config", type=Path, help="Fixed historical algebra recipe; fixture only")
    parser.add_argument("--output", type=Path, required=True, help="New directory for fixture; new JSON file otherwise")
    args = parser.parse_args()
    if (args.stage == "map") != (args.analysis is not None):
        parser.error("Only map requires --analysis")
    if args.fixture_config is not None and args.stage != "fixture":
        parser.error("--fixture-config applies only to fixture")
    if args.output.exists():
        parser.error("Refuse output overwrite")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from src.tutoring_analysis.h6_h7_surplus import (
        example_trials, map_saved_analysis, reproduce_historical_fixture,
    )
    if args.stage == "fixture":
        recipe_path = args.fixture_config or ROOT / "configs/h6_h7_algebra_fixture.json"
        bundle = reproduce_historical_fixture(json.loads(recipe_path.read_text(encoding="utf-8")),
            json.loads((ROOT / "configs/expected_gpa_analysis.json").read_text(encoding="utf-8")))
        files = {"metrics.json": bundle["rows"], "result.json": bundle["analysis"],
                 "recipe.json": bundle["recipe"], "completion.json": {
                     "complete": True, "stage": "fixture", "people": 25, "rows": 250,
                     "scope": bundle["recipe"]["scope"], "scientificAcceptance": False,
                     "historicalMetadataBytesReproduced": False, "humanValidation": False}}
        bodies = {name: json.dumps(value, indent=2, allow_nan=False) + "\n" for name, value in files.items()}
        args.output.mkdir(parents=True)
        for name, body in bodies.items():
            with (args.output / name).open("x", encoding="utf-8") as stream:
                stream.write(body)
        print(json.dumps({"complete": True, "stage": "fixture", "people": 25, "rows": 250}))
        return
    if args.stage == "example":
        result = example_trials()
    else:
        result = map_saved_analysis(json.loads(args.analysis.read_text(encoding="utf-8")),
            json.loads(args.config.read_text(encoding="utf-8")),
            json.loads((ROOT / "configs/expected_gpa_analysis.json").read_text(encoding="utf-8")))
    body = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(body)
    print(json.dumps({"stage": args.stage, "output": str(args.output), "complete": True,
                      "scientificAcceptance": False}))


if __name__ == "__main__":
    main()
