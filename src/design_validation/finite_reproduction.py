"""Portable scientific stages for three distinct finite-tutoring analyses.

The paired panel is the only stochastic producer. Terminal consumers reconstruct
its old budget-only beliefs, never a new duration-aware policy distribution.
"""
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .bounded_task import load_design
from .budget_horizon import (BudgetTask, budget_templates, initial_diagnostics,
                             paired_episode, summarize_pairs)
from .optimality import BellmanEnvelopes, BudgetEnvelopeModel
from .terminal_time import (analysis_templates, evaluate_state, mirror_audit,
                            reference_rate, validate_settings)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=_reject_constant)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def _config_record(config):
    return {key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(config).items()}


def _selected(configs, requested):
    if not requested:
        return sorted(configs, key=lambda c: c.template_id)
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate requested template")
    found = [c for c in configs if c.template_id in requested]
    if len(found) != len(requested):
        raise ValueError("Unknown or out-of-scope template")
    return sorted(found, key=lambda c: c.template_id)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name+" must be a positive integer")
    return value


def _check_settings(stage, settings):
    if stage.startswith("time-"):
        validate_settings(settings)
        expected = {"decision_tolerance": 1e-10, "classification_margin": 1e-8,
                    "probability_tolerance": 1e-12, "posterior_parity_tolerance": 1e-10}
    elif stage.startswith("bounds-"):
        expected = {"schema": "finite-tutoring-optimality-v1",
                    "profile": "pilot-engagement-budget-v1",
                    "scientific_model": "unchanged_installed_finite_model_zero_terminal_cost",
                    "report_order": "ascending_1_to_100", "template_order": "lexicographic",
                    "numerical_guard": 1e-10, "certificate_margin": 1e-8,
                    "normalization_tolerance": 1e-12, "parity_tolerance": 1e-10,
                    "machine_certified_arithmetic": False}
    else:
        expected = {"schema": "finite-tutoring-paired-horizon-v1",
                    "active_conditions": ["C3", "C5"], "capped_assessments": 20,
                    "policy": "repeated_exact_one_step_myopic",
                    "decision_tolerance": 1e-10, "certificate_tolerance": 1e-8}
        _positive_int(settings["episodes_per_active_template"], "episodes")
        if type(settings["root_seed"]) is not int:
            raise ValueError("Integer root seed required")
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported scientific settings")


def produce_pairs(configs, settings, output, episodes):
    episodes = _positive_int(episodes, "episodes")
    rows = []
    with (output/"pairs.jsonl").open("w") as stream:
        for config in configs:
            for episode in range(episodes):
                row = paired_episode(config, settings["root_seed"], episode)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                rows.append(row)
    write_json(output/"summary.json", summarize_pairs(rows))
    panel = {"schema": "finite-tutoring-paired-panel-v1", "root_seed": settings["root_seed"],
             "episodes_per_template": episodes, "templates": [_config_record(c) for c in configs],
             "records": len(rows), "policy": settings["policy"],
             "full_panel_size_and_seed": len(configs) == 6 and episodes == 400
                 and settings["root_seed"] == 2026100901,
             "scope": "paired_cap20_and_budget_only_repeated_myopic_not_global_policy"}
    write_json(output/"panel.json", panel)
    return panel


def paired_terminals(directory, configs, tolerance=1e-10):
    """Validate panel coverage and replay old budget-only reports without draws."""
    directory = Path(directory)
    panel = read_json(directory/"panel.json")
    if (panel["schema"] != "finite-tutoring-paired-panel-v1"
            or panel["policy"] != "repeated_exact_one_step_myopic"):
        raise ValueError("Expected a paired-horizon scientific input")
    episodes = _positive_int(panel["episodes_per_template"], "episodes")
    by_id = {c.template_id: c for c in configs if c.condition_id in ("C3", "C5")}
    ids = [c["template_id"] for c in panel["templates"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate panel templates")
    for record in panel["templates"]:
        if record["template_id"] not in by_id or record != _config_record(by_id[record["template_id"]]):
            raise ValueError("Paired input scientific design differs from requested design")
    expected = {(tid, episode) for tid in ids for episode in range(episodes)}
    if panel["records"] != len(expected):
        raise ValueError("Paired input count differs from its scientific recipe")
    tasks = {tid: BudgetTask(by_id[tid]) for tid in ids}
    seen = set()
    with (directory/"pairs.jsonl").open() as stream:
        for line in stream:
            original = json.loads(line, parse_constant=_reject_constant)
            identity = (original["template_id"], original["episode_id"])
            if identity in seen or identity not in expected:
                raise ValueError("Duplicate or out-of-scope predecessor episode")
            seen.add(identity)
            task, prior, scenario = tasks[identity[0]], original["budget_only"], original["scenario"]
            if (scenario["root_seed"] != panel["root_seed"] or scenario["template_id"] != identity[0]
                    or scenario["episode_id"] != identity[1]
                    or original["condition_id"] != task.config.condition_id):
                raise ValueError("Paired scenario identity mismatch")
            potential = scenario["potential_reports"]
            if (len(potential) != 2 or any(len(reports) != task.config.max_assessments for reports in potential)
                    or any(type(report) is not int or not 1 <= report <= 100
                           for reports in potential for report in reports)):
                raise ValueError("Invalid potential report support")
            logs = [x.copy() for x in task.initial_logs]
            budget, counts, state = task.config.total_millihours, [0, 0], [0, 0, 0, 0]
            for event in prior["events"]:
                who, report = event["student"], event["report"]
                if (type(who) is not int or who not in (0, 1) or type(report) is not int
                        or not 1 <= report <= 100 or budget < task.config.cost_millihours):
                    raise ValueError("Invalid predecessor history")
                if report != potential[who][counts[who]]:
                    raise ValueError("History differs from the paired potential report stream")
                logs[who] = task.posterior(logs[who], report)
                counts[who] += 1
                state[2*who] += 1
                state[2*who+1] += report
                budget -= task.config.cost_millihours
                if event["budget_after"] != budget:
                    raise ValueError("Predecessor event budget mismatch")
            if (budget != prior["remaining_millihours"] or counts != prior["counts"]
                    or sum(counts) != prior["sample_count"]):
                raise ValueError("Predecessor terminal resource/count mismatch")
            stored = np.asarray(prior["terminal_logs"], dtype=float)
            if stored.shape != (2, 100) or not np.isfinite(stored).all():
                raise ValueError("Invalid predecessor terminal beliefs")
            error = max(float(np.max(abs(current-old))) for current, old in zip(logs, stored))
            if error > tolerance:
                raise ValueError("Predecessor posterior parity failed")
            share, value = task.terminal_solution(*logs, budget)
            if share != prior["final_share_A"] or abs(value-prior["posterior_expected_utility"]) > tolerance:
                raise ValueError("Predecessor terminal decision parity failed")
            yield original, task, tuple(logs), budget, tuple(state), error
    if seen != expected:
        raise ValueError("Incomplete predecessor panel")


def _solver(config, settings):
    model = BudgetEnvelopeModel(config, numerical_guard=settings["numerical_guard"])
    return BellmanEnvelopes(model, numerical_guard=settings["numerical_guard"],
                            normalization_tolerance=settings["normalization_tolerance"],
                            certificate_margin=settings["certificate_margin"])


def checkpoint_writer(output):
    """Alternate complete DAG files, then atomically publish their summary pointer."""
    generation = 0

    def save(solver, summary):
        nonlocal generation
        witness = output/f"witness_{generation % 2}.jsonl"
        temporary = witness.with_suffix(".jsonl.pending")
        with temporary.open("w") as stream:
            for row in solver.witness_rows():
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(witness)
        pointer = output/"checkpoint.json.pending"
        write_json(pointer, {**summary, "generation": generation, "witness_file": witness.name})
        with pointer.open("rb") as stream:
            os.fsync(stream.fileno())
        pointer.replace(output/"checkpoint.json")
        save.last_witness = witness.name
        generation += 1

    save.last_witness = None
    return save


def time_initial(configs, design, settings, output):
    reference = reference_rate(budget_templates(design))
    write_json(output/"reference.json", reference)
    manifest = []
    for config in configs:
        task = BudgetTask(config)
        analysis = evaluate_state(task, task.initial_logs, config.total_millihours,
                                  settings, reference, include_independent=config.condition_id == "C1")
        row = {"template_id": config.template_id, "condition_id": config.condition_id,
               "config": asdict(config), "working_C1_candidate": config.condition_id == "C1" and config.gap_offset == 1.,
               "analysis": analysis, "orientation_audit": mirror_audit(task, task.initial_logs, config.total_millihours),
               "scope": "initial_terminal_menu_complete_ask_once_and_optimistic_PI_upper_not_global_solution"}
        name = config.template_id+".json"
        write_json(output/name, row)
        manifest.append({"template_id": config.template_id, "condition_id": config.condition_id,
                         "file": name, "scenario_rows": len(analysis["scenarios"])})
    write_json(output/"manifest.json", manifest)
    return {"initial_states": len(configs), "scenario_rows": sum(r["scenario_rows"] for r in manifest)}


def bounds_initial(configs, settings, output):
    rows, parity = [], []
    orders = [[], [1], [100], [1, 100], [100, 1], [50, 51, 49, 50], [1, 100, 1, 100, 1, 100]]
    for config in configs:
        solver = _solver(config, settings)
        check = solver.model.posterior_parity([(who, reports) for who in (0, 1) for reports in orders],
                                              settings["parity_tolerance"])
        if not check["all_pass"]:
            raise ValueError("Sufficient-statistic posterior parity failed")
        parity.append({"template_id": config.template_id, **check})
        frontier = solver.summary()
        if config.condition_id in ("C3", "C5"):
            solver.expand(solver.model.root, 201)
        rows.append({"template_id": config.template_id, "condition_id": config.condition_id,
                     "config": asdict(config), "frontier": frontier, "one_step": solver.summary()})
    write_json(output/"initial_bounds.json", rows)
    write_json(output/"posterior_parity.json", parity)
    return {"initial_states": len(rows), "all_parity_pass": True}


def terminal_stage(stage, directory, configs, design, settings, output):
    is_time = stage == "time-terminal"
    reference = reference_rate(budget_templates(design)) if is_time else None
    if is_time:
        write_json(output/"reference.json", reference)
    counts, n, maximum_error = Counter(), 0, 0.
    models = {}
    name = "terminal_impact.jsonl" if is_time else "terminal_bounds.jsonl"
    tolerance = settings["posterior_parity_tolerance" if is_time else "parity_tolerance"]
    with (output/name).open("w") as stream:
        for original, task, logs, budget, state, error in paired_terminals(directory, configs, tolerance):
            identity = {"template_id": task.config.template_id, "episode_id": original["episode_id"]}
            if is_time:
                analysis = evaluate_state(task, logs, budget, settings, reference, acquisition=False)
                row = {**identity, "condition_id": task.config.condition_id, "posterior_parity_error": error,
                       "analysis": analysis, "old_budget_only_share_A": original["budget_only"]["final_share_A"],
                       "scope": "same_immutable_old_myopic_terminal_state_not_new_cost_aware_state_distribution"}
                for scenario in analysis["scenarios"]:
                    key = "|".join((identity["template_id"], scenario["profile"], str(scenario["rho_multiplier"]), scenario["selected_macro"]))
                    counts[key] += 1
            else:
                if identity["template_id"] not in models:
                    models[identity["template_id"]] = BudgetEnvelopeModel(task.config, numerical_guard=settings["numerical_guard"])
                model = models[identity["template_id"]]
                error = max(float(np.max(abs(a-b))) for a, b in zip(model.logs(state), original["budget_only"]["terminal_logs"]))
                if error > tolerance or model.budget(state) != budget:
                    raise ValueError("Canonical terminal posterior/resource parity failed")
                seed = model.seed(state)
                stopping = not seed.asks or all(seed.stop.lower > v.upper+settings["certificate_margin"] for v in seed.asks.values())
                status = "numerical_global_stop_at_sampled_terminal" if stopping else "inconclusive_sampled_terminal"
                counts[status] += 1
                row = {**identity, "state": state, "status": status, "log_parity_error": error,
                       "stop_lower": seed.stop.lower, "stop_upper": seed.stop.upper,
                       "ask_upper": {a: v.upper for a, v in seed.asks.items()}, "details": seed.details,
                       "scope": "sampled_terminal_stopping_only_not_earlier_or_global_policy",
                       "machine_certified_arithmetic": False}
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            maximum_error = max(maximum_error, error)
            n += 1
    panel = read_json(Path(directory)/"panel.json")
    summary = {"records": n, "maximum_posterior_parity_error": maximum_error,
               "selected_macro_counts_by_template_profile_rate" if is_time else "statuses": dict(counts),
               "input_panel": panel, "new_random_scenarios": 0,
               "new_cost_aware_trajectory_claimed": False, "whole_policy_optimality_claimed": False}
    write_json(output/"summary.json", summary)
    return summary


def run(args):
    """Run one stage. A completion file exists only after the full stage succeeds."""
    stage, output = args.stage, args.output
    settings, design = read_json(args.config), load_design(args.design)
    _check_settings(stage, settings)
    configs = budget_templates(design)
    if stage == "time-initial":
        configs = analysis_templates(design, settings)
    if stage in ("pairs", "bounds-root"):
        configs = [c for c in configs if c.condition_id in ("C3", "C5")]
    configs = _selected(configs, args.template)
    if args.episodes is not None and stage != "pairs":
        raise ValueError("--episodes is only a paired-producer setting")
    terminal = stage in ("time-terminal", "bounds-terminal")
    if terminal and (args.pairs is None or args.template):
        raise ValueError("Terminal stages require --pairs and consume the complete supplied panel")
    if not terminal and args.pairs is not None:
        raise ValueError("--pairs applies only to fixed-terminal stages")
    overrides = {key: getattr(args, key) for key in ("max_seconds", "max_created_states", "max_depth")
                 if getattr(args, key) is not None}
    if overrides and stage != "bounds-root":
        raise ValueError("Refinement cutoffs apply only to bounds-root")
    limits = None
    if stage == "bounds-root":
        if len(configs) != 1:
            raise ValueError("bounds-root requires exactly one --template")
        limits = {**settings["production"], **overrides}
        if (any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in limits.values())
                or type(limits["max_created_states"]) is not int or limits["max_created_states"] < 1
                or type(limits["max_depth"]) is not int):
            raise ValueError("Invalid computational cutoffs")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"settings.json", settings)
    write_json(output/"templates.json", [_config_record(c) for c in configs])
    if stage == "pairs":
        result = produce_pairs(configs, settings, output,
                               args.episodes if args.episodes is not None else settings["episodes_per_active_template"])
    elif stage == "horizon-initial":
        rows = [initial_diagnostics(c) for c in configs]
        write_json(output/"initial_diagnostics.json", rows)
        result = {"initial_states": len(rows), "schedule_geometry_included": False}
    elif stage == "time-initial":
        result = time_initial(configs, design, settings, output)
    elif stage == "bounds-initial":
        result = bounds_initial(configs, settings, output)
    elif terminal:
        result = terminal_stage(stage, args.pairs, configs, design, settings, output)
    else:
        started = time.perf_counter()
        solver = _solver(configs[0], settings)
        save = checkpoint_writer(output)
        result = solver.run(**limits, target_epsilon=settings["target_epsilon"],
                            rss_limit_bytes=settings["rss_limit_bytes"], started_at=started,
                            checkpoint=save, checkpoint_seconds=settings["checkpoint_seconds"])
        result.update(template=configs[0].template_id, condition=configs[0].condition_id,
                      scientific_horizon=configs[0].max_assessments, witness_file=save.last_witness,
                      bitwise_time_limited_reproduction_claimed=False)
        write_json(output/"result.json", result)
    completion = {"stage": stage, "complete": True, "machine_certified_arithmetic": False,
                  "numerical_thread_environment": {key: os.environ.get(key) for key in
                      ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}, **result}
    write_json(output/"completion.json", completion)
    return completion
