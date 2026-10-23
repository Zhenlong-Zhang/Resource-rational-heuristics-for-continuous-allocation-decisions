#!/usr/bin/env python3
"""Selected working-population cost-aware trials and independent numerical replay."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "produce", "check"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tutoring_cost_aware.json")
    parser.add_argument("--selection", type=Path, required=True, help="JSON array of exact stable trial coordinates")
    parser.add_argument("--inputs", type=Path, help="Prepared input directory (produce/check)")
    parser.add_argument("--records", type=Path, help="Producer record directory (check)")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--node", default="node", help="Node.js executable for prepare/check source generation")
    args = parser.parse_args(argv)
    if args.stage != "prepare" and args.inputs is None:
        parser.error("--inputs is required for produce/check")
    if args.stage == "check" and args.records is None:
        parser.error("--records is required for check")
    if args.stage == "prepare" and args.inputs is not None or args.stage != "check" and args.records is not None:
        parser.error("Input flags do not belong to this stage")
    try:
        from src.design_validation.cost_aware_inputs import compact, load_context, prepare_inputs, read
        if args.stage == "prepare":
            result = prepare_inputs(load_context(args.config), read(args.selection), args.output, args.node)
            status = {"stage": "prepare", "selection": result["selection"], "output": str(args.output)}
        else:
            from src.design_validation.cost_aware_reproduction import check, produce
            if args.stage == "produce":
                result = produce(args.config, args.selection, args.inputs, args.output)
                status = {"stage": "produce", "selection": result["selection"], "output": str(args.output),
                          "statuses": [t["history"]["status"] for t in result["trials"]]}
            else:
                result = check(args.config, args.selection, args.inputs, args.records, args.output, args.node)
                status = {"stage": "check", "selection": result["selection"], "output": str(args.output),
                          "allSelectedNumericallyCompleted": result["allSelectedNumericallyCompleted"]}
        print(compact(status))
        if args.stage == "check" and not result["allSelectedNumericallyCompleted"]:
            return 2
        if args.stage == "produce" and any(t["history"]["status"] != "completed" for t in result["trials"]):
            return 2
        return 0
    except (ValueError, KeyError, TypeError, IndexError, AssertionError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
