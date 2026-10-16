"""Prospective one-candidate orchestration; all task mathematics live in bounded_task.

No baseline source is modified, no seeds are selected after observing results, and
passing this computational selection gate does not approve a participant study.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import numpy as np

from .bounded_task import (
    BoundedTask, CERTIFICATE_TOLERANCE, DECISION_TOLERANCE, belief_moments,
    expand_templates, file_sha256, heuristic_shares, initial_diagnostics,
    initial_log_beliefs, load_design, numerical_sensitivity, simulate_episode,
    summarize_records,
)

STAGES = {"screening": (30, 2026091901), "confirmation": (1200, 2026091902)}
EXPECTED_CHANGE = {"condition_id": "C3", "field": "prior_kernel_centres",
                   "old_value": [60, 60], "new_value": [50, 70]}
SELECTION_CRITERIA = {
    "initial_best_one_step_gain_strictly_above": DECISION_TOLERANCE,
    "target_match_wilson_lower_at_least": .80,
    "EO_ES_separation_share_distance_at_least": .08,
    "diagnostic_fraction_wilson_lower_at_least": .25,
    "cap_binding_wilson_upper_strictly_below": .01,
    "paired_gain_normal_lower_strictly_above": 0.0,
    "interval_level": .95,
    "interval_sides": "two_sided",
    "target_match_share_tolerance": .05,
}


def stage_spec(stage: str) -> tuple[int, int]:
    if stage not in STAGES:
        raise ValueError("Only the two prospectively fixed stages are permitted")
    return STAGES[stage]


def load_revision(base_path: str | Path, patch_path: str | Path) -> tuple[dict, dict, dict]:
    """Resolve exactly one declared edit, returning independent base/resolved dictionaries."""
    patch = json.loads(Path(patch_path).read_text())
    if (patch.get("schema") != "tutoring_design_single_revision_v0.2"
            or patch.get("change") != EXPECTED_CHANGE):
        raise ValueError("Only the frozen C3 [60,60] to [50,70] candidate is supported")
    if file_sha256(base_path) != patch["base_sha256"]:
        raise ValueError("Baseline fixture hash mismatch")
    engine_path = Path(__file__).with_name("bounded_task.py")
    if file_sha256(engine_path) != patch["engine_sha256"]:
        raise ValueError("Frozen bounded engine hash mismatch")
    base = load_design(base_path)
    resolved = copy.deepcopy(base)
    row = next(x for x in resolved["conditions"] if x["id"] == "C3")
    if row["prior_kernel_centres"] != patch["change"]["old_value"]:
        raise ValueError("Unexpected baseline C3 centres")
    row["prior_kernel_centres"] = list(patch["change"]["new_value"])
    # Metadata changes are explicit; the sole scientific-field edit is above.
    row["initial_information"] = "differentiated_broad_direct_priors"
    resolved["schema"] = "local_experiment_design_candidates_v0.2"
    resolved["date"] = patch["date"]
    resolved["status"] = "frozen_prospective_v0.2_candidate_computation_pending_not_human_validated"
    for condition in resolved["conditions"]:
        condition["status"] = ("prospective_revised_candidate" if condition["id"] == "C3"
                               else "unchanged_v0.1_finite_task_computational_evidence_retained")
    resolved["revision_provenance"] = {
        "base_sha256": patch["base_sha256"], "patch_sha256": file_sha256(patch_path),
        "engine_sha256": patch["engine_sha256"], "scientific_change": patch["change"],
        "metadata_changes": ["schema", "date", "status", "condition_statuses",
                             "C3_initial_information", "revision_provenance"],
        "status_note": "The input describes the fixed candidate. See the generated selection_gate.json for the tested disposition.",
    }
    return base, resolved, patch


def prior_comparison(base: dict, resolved: dict) -> list[dict[str, Any]]:
    """Actual finite moments, not an unbounded-Gaussian approximation."""
    old = {c.template_id: c for c in expand_templates(base) if c.condition_id == "C3"}
    result = []
    for new in expand_templates(resolved):
        if new.condition_id != "C3":
            continue
        pair = {}
        for label, config in (("v0.1", old[new.template_id]), ("v0.2", new)):
            logs = initial_log_beliefs(config)
            moments = [belief_moments(p) for p in logs]
            eo = heuristic_shares(*logs, config.total_millihours)["equal_outcome"]
            pair[label] = {"kernel_centres": list(config.prior_centres),
                           "kernel_spreads": list(config.prior_spreads),
                           "actual_moments_mean_sd": moments,
                           "actual_mean_sum": sum(m[0] for m in moments),
                           "actual_variance_sum": sum(m[1]**2 for m in moments),
                           "initial_EO_share_A": eo, "initial_EO_ES_gap": abs(eo-.5)}
        result.append({"template_id": new.template_id, "scale": new.scale, **pair})
    return result


def candidate_gate(summaries: list[dict], diagnostics: list[dict], stage: str) -> dict:
    """Gate arithmetic only; original point criteria remain in the unmodified summary."""
    stage_spec(stage)
    summary = next(x for x in summaries if x["condition_id"] == "C3")
    initial = [x for x in diagnostics if x["config"]["condition_id"] == "C3"]
    if {x["config"]["template_id"] for x in initial} != {f"C3_s{i}_g0" for i in range(3)} or len(initial) != 3:
        raise ValueError("All three C3 scale templates are required")
    gains = {x["config"]["template_id"]: max(x["decision"]["gain_A"], x["decision"]["gain_B"])
             for x in initial}
    measured = {
        "minimum_initial_gain": min(gains.values()),
        "target_match_lower": summary["target_match"]["wilson_95_interval"][0],
        "diagnostic_fraction_lower": summary["EO_vs_split_diagnostic"]["wilson_95_interval"][0],
        "cap_binding_upper": summary["cap_binding"]["wilson_95_interval"][1],
        "paired_gain_lower": summary["gain_vs_immediate_optimum"]["normal_approx_95_interval"][0],
    }
    checks = {
        "all_initial_gains_positive": measured["minimum_initial_gain"] > DECISION_TOLERANCE,
        "target_match_lower_ge_0_80": measured["target_match_lower"] >= .80,
        "diagnostic_fraction_lower_ge_0_25": measured["diagnostic_fraction_lower"] >= .25,
        "cap_binding_upper_lt_0_01": measured["cap_binding_upper"] < .01,
        "paired_gain_lower_gt_0": measured["paired_gain_lower"] > 0,
    }
    return {"stage": stage, "criteria": SELECTION_CRITERIA, "measured": measured,
            "initial_gains_by_template": gains, "checks": checks,
            "all_numeric_gates_met": all(checks.values()),
            "selection_evidence": stage == "confirmation",
            "selected_under_prospective_rule": all(checks.values()) if stage == "confirmation" else None,
            "scope": "Approximate two-sided95% uncertainty gates for this finite task/myopic policy; screening is descriptive; not human validation or global optimality."}


def fixed_initial_comparison(records: list[dict]) -> list[dict]:
    result = []
    for cid in sorted({r["condition_id"] for r in records}):
        rows = [r["fixed_initial_EO_comparison"] for r in records if r["condition_id"] == cid]
        result.append({"condition_id": cid, "episodes": len(rows),
            "policy_match_to_fixed_initial_EO_rate": float(np.mean([x["policy_matches_fixed_initial_EO"] for x in rows])),
            "policy_mean_absolute_gap_to_fixed_initial_EO": float(np.mean([x["policy_absolute_gap"] for x in rows])),
            "mean_absolute_terminal_EO_shift_from_initial": float(np.mean([x["terminal_EO_absolute_shift"] for x in rows])),
            "terminal_EO_shift_at_least_0_08_fraction": float(np.mean([x["terminal_EO_absolute_shift"] >= .08-1e-12 for x in rows])),
            "scope": "Descriptive only: resemblance and target movement do not uniquely identify an online cognitive rule."})
    return result


def _write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_revision(base_path: Path, patch_path: Path, output_dir: Path, stage: str,
                 source_root: Path) -> dict:
    """Run one fixed stage, retaining all outcomes and source/input provenance."""
    if source_root.resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("Source root must identify this imported implementation")
    thread_variables = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in thread_variables):
        raise ValueError("Set all numerical-library thread variables to1 before importing NumPy; use the CLI")
    n, seed = stage_spec(stage)
    base, design, patch = load_revision(base_path, patch_path)
    templates = expand_templates(design)
    sources = ["src/design_validation/__init__.py", "src/design_validation/bounded_task.py",
               "src/design_validation/revision.py", "scripts/validate_tutoring_design_revision.py"]
    hashes = {p: file_sha256(source_root/p) for p in sources}
    base_hash, patch_hash = file_sha256(base_path), file_sha256(patch_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = output_dir/"source_snapshot"
    for rel in sources:
        destination = snapshot/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root/rel, destination)
        if file_sha256(destination) != hashes[rel]:
            raise RuntimeError("Snapshot differs from pre-run source")
    for filename, source in (("base_v01.json", base_path), ("revision_patch.json", patch_path)):
        shutil.copyfile(source, output_dir/filename)
    _write_json(output_dir/"frozen_conditions.json", design)
    manifest = {"stage": stage, "root_seed": seed, "episodes_per_condition": n,
        "stage_schedule": {k: {"episodes_per_condition": v[0], "root_seed": v[1]} for k, v in STAGES.items()},
        "base_sha256": base_hash, "patch_sha256": patch_hash,
        "resolved_sha256": file_sha256(output_dir/"frozen_conditions.json"), "source_hashes": hashes,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "templates": [asdict(c) for c in templates], "single_numerical_library_thread": True,
        "decision_tolerance": DECISION_TOLERANCE, "certificate_tolerance": CERTIFICATE_TOLERANCE,
        "selection_criteria": SELECTION_CRITERIA, "policy": "exact_finite_one_step_myopic_not_certified_global_optimum",
        "sensitivity_selection": "initial,first_post_query_if_any,terminal of episode_id=0 for every template",
        "sensitivity_scope": "1001 versus 2001 feasible allocation grid, not integration convergence",
        "old_source_results_unmodified": True, "human_data": False}
    _write_json(output_dir/"manifest.json", manifest)
    started = time.perf_counter()
    _write_json(output_dir/"prior_comparison.json", prior_comparison(base, design))
    records, diagnostics, sensitivity, timings = [], [], [], []
    with (output_dir/"episodes.jsonl").open("x") as stream:
        for config in templates:
            tick = time.perf_counter()
            task = BoundedTask(config)
            diag = initial_diagnostics(task)
            diagnostics.append(diag)
            initial_eo = diag["initial_heuristics"]["equal_outcome"]
            count = config.repeat_count_per_30*(n//30)
            for episode_id in range(count):
                record = simulate_episode(task, seed, episode_id)
                states = record.pop("_sensitivity_states")
                if episode_id == 0:
                    sensitivity.extend(numerical_sensitivity(task, states))
                gap = abs(record["final_share_A"]-initial_eo)
                record["fixed_initial_EO_comparison"] = {
                    "fixed_initial_EO_share_A": initial_eo, "policy_absolute_gap": gap,
                    "policy_matches_fixed_initial_EO": gap <= .05+1e-12,
                    "terminal_EO_absolute_shift": abs(record["heuristic_shares"]["equal_outcome"]-initial_eo)}
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                records.append(record)
            elapsed = time.perf_counter()-tick
            timings.append({"template_id": config.template_id, "episodes": count, "seconds": elapsed})
            print(f"{config.template_id}: {count} episodes in {elapsed:.2f}s; initial={diag['decision']['chosen_action']}", flush=True)
    if hashes != {p: file_sha256(source_root/p) for p in sources}:
        raise RuntimeError("Source changed during evaluation")
    if (base_hash != file_sha256(base_path) or patch_hash != file_sha256(patch_path)
            or base_hash != file_sha256(output_dir/"base_v01.json")
            or patch_hash != file_sha256(output_dir/"revision_patch.json")):
        raise RuntimeError("Configuration changed during evaluation")
    if manifest["resolved_sha256"] != file_sha256(output_dir/"frozen_conditions.json"):
        raise RuntimeError("Frozen resolved configuration changed during evaluation")
    summaries = summarize_records(records, diagnostics)
    gate = candidate_gate(summaries, diagnostics, stage)
    for filename, value in (("initial_diagnostics.json", diagnostics), ("summary.json", summaries),
                            ("selection_gate.json", gate), ("grid_sensitivity.json", sensitivity),
                            ("fixed_initial_EO_comparison.json", fixed_initial_comparison(records))):
        _write_json(output_dir/filename, value)
    _write_json(output_dir/"timing.json", {"total_seconds": time.perf_counter()-started, "templates": timings})
    _write_json(output_dir/"completed.json", {"episodes": len(records), "templates": len(templates),
        "status": "evaluation_executed_selection_reported_separately", "source_hashes": hashes,
        "base_sha256": base_hash, "patch_sha256": patch_hash})
    print(f"Completed {len(records)} synthetic episodes in {time.perf_counter()-started:.2f}s.", flush=True)
    return gate
