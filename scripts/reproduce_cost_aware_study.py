#!/usr/bin/env python3
"""Full cost-aware tutoring study: prepare, independently check shards, reduce."""
import argparse
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "prepare", "run", "aggregate"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tutoring_cost_aware.json")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--people-per-shard", type=int, default=8)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path, help="New aggregate JSON file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true")
    group.add_argument("--ids", help="Comma-separated stable shard ordinals")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--node", default="node")
    args = parser.parse_args(argv)
    if args.stage != "plan" and args.output is None:
        parser.error("--output required")
    if args.stage in ("run", "aggregate") and args.prepared is None:
        parser.error("--prepared required")
    if args.stage == "run" and not (args.all or args.ids):
        parser.error("Choose --ids or --all explicitly")
    if args.stage == "aggregate" and args.summary is None:
        parser.error("--summary required")
    from src.design_validation import cost_aware_campaign as C
    from src.design_validation.cost_aware_inputs import compact
    if args.stage == "plan":
        result = C.plan(args.mode, args.people_per_shard)
    elif args.stage == "prepare":
        result = C.prepare(args.config, args.output, args.mode, args.people_per_shard, args.node)
    elif args.stage == "run":
        roster, _ = C.load_prepared(args.config, args.prepared, args.node)
        ids = list(range(len(roster["tasks"]))) if args.all else [int(i) for i in args.ids.split(",")]
        for result in C.run(args.config, args.prepared, args.output, ids, args.workers, args.node):
            print(compact(result), flush=True)
        return 0
    else:
        result = C.aggregate(args.config, args.prepared, args.output, args.summary, args.node)
        print(compact({k: v for k, v in result.items() if k != "records"}))
        return 0 if result["fullStudyComplete"] or result["mode"] == "smoke" and not result["missingOrdinals"] else 2
    print(compact({**{k: v for k, v in result.items() if k != "tasks"},
                   "taskCount": len(result["tasks"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
