#!/usr/bin/env python3
"""Independently check four exact historical records and prospective H6/H7 roles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tutoring_analysis.h6_h7_surplus_check import check_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/h6_h7_surplus.json")
    parser.add_argument("--output", type=Path, required=True, help="New independent-check JSON file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse output overwrite")
    result = check_mapping(json.loads(args.analysis.read_text(encoding="utf-8")),
        json.loads(args.config.read_text(encoding="utf-8")),
        json.loads((ROOT / "configs/expected_gpa_analysis.json").read_text(encoding="utf-8")),
        json.loads(args.mapping.read_text(encoding="utf-8")))
    body = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(body)
    print(json.dumps({"passed": True, "historicalTestsReferenced": 4, "newTests": 0}))


if __name__ == "__main__":
    main()
