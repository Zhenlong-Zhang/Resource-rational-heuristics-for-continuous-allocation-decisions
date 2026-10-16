#!/usr/bin/env python3
"""Run isolated, bounded local validation of the proposed tutoring-task recipes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from design_validation.bounded_task import (
    BoundedTask, CERTIFICATE_TOLERANCE, DECISION_TOLERANCE, expand_templates,
    file_sha256, initial_diagnostics, load_design, numerical_sensitivity,
    simulate_episode, summarize_records,
)


def write_json(path: Path, data: object) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "screening", "confirmation", "precision"), required=True)
    parser.add_argument("--conditions", nargs="+", choices=[f"C{i}" for i in range(1, 6)])
    parser.add_argument("--episodes-per-condition", type=int)
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    expected_n = {"smoke": 30, "screening": 30, "confirmation": 120, "precision": 1200}[args.stage]
    episodes_per_condition = args.episodes_per_condition or expected_n
    if episodes_per_condition != expected_n:
        parser.error(f"Frozen {args.stage} stage requires {expected_n} episodes per condition")
    if args.stage != "smoke" and args.conditions:
        parser.error("Screening/confirmation must retain every condition")
    if args.stage in ("confirmation", "precision") and not args.sensitivity:
        parser.error("Confirmation/precision requires the prespecified resolution sensitivity")
    root_seed = {"smoke": 2026091700, "screening": 2026091701,
                 "confirmation": 2026091702, "precision": 2026091703}[args.stage]
    design = load_design(args.conditions_file)
    templates = expand_templates(design)
    if args.conditions:
        templates = [x for x in templates if x.condition_id in args.conditions]
    sources = [ROOT / "src/design_validation/bounded_task.py", Path(__file__).resolve(),
               ROOT / "src/design_validation/__init__.py"]
    hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in sources}
    config_hash = file_sha256(args.conditions_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "stage": args.stage, "root_seed": root_seed, "episodes_per_condition": episodes_per_condition,
        "input_sha256": config_hash, "input_file_name": args.conditions_file.name,
        "source_hashes": hashes, "python_version": platform.python_version(), "numpy_version": np.__version__,
        "templates": [asdict(t) for t in templates], "single_numerical_library_thread": True,
        "decision_tolerance": DECISION_TOLERANCE, "certificate_tolerance": CERTIFICATE_TOLERANCE,
        "policy": "exact_finite_one_step_myopic_not_global_metalevel_optimum",
        "sensitivity": args.sensitivity,
        "sensitivity_selection": "initial,first_post_query_if_any,terminal of episode_id=0 for every template",
        "sensitivity_scope": "1001 versus 2001 feasible allocation grid, not integration convergence",
        "original_design_unmodified": True, "human_data": False,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "frozen_conditions.json", design)
    started = time.perf_counter()
    records, diagnostics, sensitivity, timings = [], [], [], []
    with (args.output_dir / "episodes.jsonl").open("x") as stream:
        for config in templates:
            tick = time.perf_counter()
            task = BoundedTask(config)
            diag = initial_diagnostics(task)
            diagnostics.append(diag)
            count = config.repeat_count_per_30*(episodes_per_condition//30)
            for episode_id in range(count):
                record = simulate_episode(task, root_seed, episode_id)
                states = record.pop("_sensitivity_states")
                if episode_id == 0 and args.sensitivity:
                    sensitivity.extend(numerical_sensitivity(task, states))
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
                records.append(record)
            elapsed = time.perf_counter()-tick
            timings.append({"template_id": config.template_id, "episodes": count, "seconds": elapsed})
            print(f"{config.template_id}: {count} episodes in {elapsed:.2f}s; "
                  f"initial={diag['decision']['chosen_action']}; evidence={diag['acquisition_evidence']}", flush=True)
    if hashes != {str(p.relative_to(ROOT)): file_sha256(p) for p in sources}:
        raise RuntimeError("Source changed during evaluation; outputs cannot be accepted")
    if config_hash != file_sha256(args.conditions_file):
        raise RuntimeError("Input changed during evaluation")
    write_json(args.output_dir / "initial_diagnostics.json", diagnostics)
    write_json(args.output_dir / "summary.json", summarize_records(records, diagnostics))
    write_json(args.output_dir / "grid_sensitivity.json", sensitivity)
    write_json(args.output_dir / "timing.json", {"total_seconds": time.perf_counter()-started, "templates": timings})
    write_json(args.output_dir / "completed.json", {"episodes": len(records), "templates": len(templates),
        "status": "evaluation_executed_not_all_prototypes_passed", "input_sha256": config_hash,
        "source_hashes": hashes})
    print(f"Completed {len(records)} synthetic episodes in {time.perf_counter()-started:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
