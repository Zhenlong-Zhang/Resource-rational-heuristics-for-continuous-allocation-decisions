#!/usr/bin/env python3
"""Reproduce historical inference, conditional planning and bounded sensitivities."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from tutoring_analysis.reproduction import run_inference, run_planning, run_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=("inference", "planning", "bounded", "variance-adaptive"), required=True)
    parser.add_argument("--mode", choices=("fixture", "smoke", "full"), default="smoke")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--design", type=Path, default=ROOT/"configs/tutoring_conditions.json")
    parser.add_argument("--node", default="node", help="Node.js executable for the pure scientific schedule")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.analysis == "inference":
        result = run_inference(args.output, ROOT)
    else:
        names = {"planning": "tutoring_power_sensitivity_v01.json",
                 "bounded": "tutoring_bounded_robustness_v01.json",
                 "variance-adaptive": "tutoring_identifiability_robustness_v01.json"}
        config = json.loads((args.config or ROOT/"configs"/names[args.analysis]).read_text())
        if args.analysis == "planning":
            result = run_planning(config, args.output)
        else:
            design = json.loads(args.design.read_text()) if args.analysis == "variance-adaptive" else None
            result = run_calibration(args.analysis, config, args.output, args.mode, design, args.node)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
