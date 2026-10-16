#!/usr/bin/env python3
"""Run isolated, prospectively specified C2 synthetic identifiability checks locally."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from design_validation.bounded_task import load_design  # noqa: E402
from design_validation.identifiability import build_c2_recipe, simulate_recovery  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tutoring_identifiability_validation.json")
    parser.add_argument("--stage", choices=("smoke", "screening", "confirmation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if "src" in sys.modules or any(name.startswith(("src.", "experiments.", "mdp.")) for name in sys.modules):
        raise RuntimeError("Legacy source import isolation failed")
    config = json.loads(args.config.read_text())
    design = load_design(args.conditions)
    recipe = build_c2_recipe(design)
    paths = [args.conditions, args.config, Path(__file__), ROOT / "src/design_validation/identifiability.py", ROOT / "src/design_validation/bounded_task.py", ROOT / "src/design_validation/__init__.py"]
    source_bytes = {str(path.resolve()): path.read_bytes() for path in paths}
    hashes = {path: hashlib.sha256(content).hexdigest() for path, content in source_bytes.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.output_dir / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=False)
    snapshot_paths = {}
    for path, content in source_bytes.items():
        snapshot = snapshot_dir / Path(path).name
        with snapshot.open("xb") as file:
            file.write(content)
        snapshot_paths[path] = str(snapshot.relative_to(args.output_dir))
    manifest = {"stage": args.stage, "frozen_before_execution": True, "configuration": config,
                "sha256": hashes, "legacy_imports_absent": True, "thread_count": 1,
                "source_snapshots": snapshot_paths,
                "template_ids": recipe.template_ids,
                "scope": "no_search_C2_synthetic_recovery_not_human_experiment_or_sample_size_approval"}
    # Exclusive creation prevents accidental replacement of baseline or previous runs.
    with (args.output_dir / "manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    started = time.monotonic()
    result, rows = simulate_recovery(recipe, config, args.stage)
    result["elapsed_seconds"] = time.monotonic() - started
    result["input_sha256"] = hashes
    with (args.output_dir / "summary.json").open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    with (args.output_dir / "group_statistics.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"stage": args.stage, "cells": len(result["cells"]), "group_rows": len(rows),
                      "seconds": result["elapsed_seconds"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
