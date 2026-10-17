"""Portable finite schedules and descriptive readers for generated raw outputs."""
from pathlib import Path
import json
import subprocess

from .bounded_task import load_design
from .budget_horizon import budget_templates, schedule_geometry, source_parity
from .finite_auxiliary_inputs import (number, panel_identities, parse, read, require,
                                     stage, template_records, terminal_rows, write)
from .finite_bound_tables import ROOT_FIELDS, check_envelope, compact_initial, symmetry_bound, terminal_counts
from .finite_reproduction import _check_settings, _config_record
from .finite_time_tables import initial_summary, terminal_summary
from .terminal_time import MACROS, analysis_templates, timing_profiles


ACTIVE = [f"{condition}_s{scale}_g0" for condition in ("C3", "C5") for scale in range(3)]


def auxiliary_settings(settings):
    expected = {"schema": "finite-tutoring-auxiliary-v1", "profile": "pilot-engagement-budget-v1",
                "source_schedule_seed": 2026100902, "conditions": ["C1", "C2", "C3", "C4", "C5"],
                "schedule_variants": 9, "main_trials_per_schedule": 10,
                "reversed_trials_per_schedule": 5, "balanced_geometry_cases": 6804,
                "terminal_pooling": "equal_template_not_human_schedule_weighting",
                "terminal_templates": ACTIVE, "terminal_episodes_per_template": 400,
                "epsilon_levels": [.1, .01, .001], "machine_certified_arithmetic": False}
    require(settings == expected, "Unsupported finite auxiliary recipe")


def source_panel(design, settings, node):
    helper = Path(__file__).with_name("budget_source_panel.cjs")
    result = subprocess.run([node, str(helper)],
                            input=json.dumps({"design": design, "seed": settings["source_schedule_seed"]}),
                            text=True, capture_output=True, check=True)
    panel = parse(result.stdout)
    require(panel["schema"] == "budget-finite-source-panel-v1"
            and panel["profileId"] == settings["profile"]
            and panel["seed"] == settings["source_schedule_seed"], "Wrong source panel identity")
    identities = [(row["conditionId"], row["variant"]) for row in panel["schedules"]]
    expected = {(cid, variant) for cid in settings["conditions"] for variant in range(9)}
    require(len(identities) == 45 and set(identities) == expected, "Missing/duplicate source schedules")
    require(len(panel["beliefCases"]) == 216 and len(panel["outcomeCases"]) == 1620,
            "Incomplete JS parity panel")
    configs = budget_templates(design)
    parity = source_parity(panel, configs)
    require(parity["all_pass"] is True, "JavaScript/Python scientific parity failed")
    geometry = schedule_geometry(panel, configs)
    return {"source_panel.json": panel, "source_parity.json": parity, "geometry.json": geometry}


def check_scenarios(row, settings, initial):
    analysis = row["analysis"]
    require(analysis["global_adaptive_policy_solved"] is False
            and analysis["personal_question_time_priced"] is False
            and analysis["machine_certified_arithmetic"] is False
            and analysis["new_random_scenarios"] == 0, "Unsupported terminal sensitivity scope")
    expected = {(profile, rho) for profile in timing_profiles(settings, initial and row["condition_id"] == "C1")
                for rho in settings["rho_multipliers"]}
    keys = [(s["profile"], s["rho_multiplier"]) for s in analysis["scenarios"]]
    require(len(keys) == len(expected) and set(keys) == expected, "Missing/duplicate timing profile-rate cells")
    terminal = analysis["terminal"]
    require(set(terminal["macros"]) == set(MACROS), "Complete ten-macro menu required")
    for macro in terminal["macros"].values():
        number(macro["expected_utility"])
        number(macro["expected_combined_GPA"])
        share, index = macro["share_A"], macro["index"]
        if terminal["automatic_zero_budget"]:
            require(share is None and index is None, "Zero-budget allocation must remain undefined")
        else:
            require(type(index) is int and 0 <= index <= 1000
                    and number(share) == index/1000, "Invalid native share index")
    for scenario in analysis["scenarios"]:
        require(scenario["selected_macro"] in MACROS, "Unknown selected macro")
        require(set(scenario["duration_by_macro_seconds"]) == set(MACROS), "Missing macro duration")
        require(all(number(v) >= 0 for v in scenario["duration_by_macro_seconds"].values()),
                "Negative duration")
        if not initial:
            require(not scenario["ask_once"] and scenario["no_search_status"] == "not_checked",
                    "Old-terminal impact must not claim new acquisition")


def time_tables(args, design, settings):
    _check_settings("time-initial", settings)
    require(args.initial is not None or args.terminal is not None, "At least one complete time input required")
    require(not args.root, "Root inputs apply only to bound tables")
    expected_initial = {c.template_id: _config_record(c) for c in analysis_templates(design, settings)}
    expected_budget = {c.template_id: _config_record(c) for c in budget_templates(design)}
    active = {key: expected_budget[key] for key in ACTIVE}
    outputs, references = {}, []
    for directory, name in ((args.initial, "time-initial"), (args.terminal, "time-terminal")):
        if directory is None:
            continue
        completion = stage(directory, name, settings)
        reference = read(directory/"reference.json")
        require(number(reference["rho0_GPA_per_personal_second"]) > 0, "Positive reference rate required")
        references.append(reference)
        if name == "time-initial":
            actual = template_records(directory, expected_initial)
            require(actual == expected_initial, "Complete 36 initial templates required")
            manifest = read(directory/"manifest.json")
            ids = [item["template_id"] for item in manifest]
            require(len(ids) == len(expected_initial) and set(ids) == set(expected_initial),
                    "Missing/duplicate initial manifest identities")
            scenario_rows = 0
            for item in manifest:
                tid = item["template_id"]
                require(item["file"] == tid+".json", "Noncanonical initial input filename")
                row = read(directory/item["file"])
                c = expected_initial[tid]
                require(row["template_id"] == tid and row["config"] == c
                        and row["condition_id"] == item["condition_id"] == c["condition_id"],
                        "Initial template identity mismatch")
                require(row["working_C1_candidate"] == (c["condition_id"] == "C1" and c["gap_offset"] == 1.),
                        "Working stimulus selection changed")
                check_scenarios(row, settings, True)
                require(item["scenario_rows"] == len(row["analysis"]["scenarios"]), "Scenario denominator mismatch")
                scenario_rows += item["scenario_rows"]
            require(completion["initial_states"] == 36 and scenario_rows == completion["scenario_rows"] == 1800,
                    "Full initial scenario coverage required")
            for key, value in initial_summary(directory, settings, reference).items():
                outputs[key+".json"] = value
        else:
            template_records(directory, expected_budget)
            for row in terminal_rows(directory, "terminal_impact.jsonl", completion, active, full=True):
                require(row["condition_id"] == active[row["template_id"]]["condition_id"],
                        "Terminal condition identity mismatch")
                require(0 <= number(row["posterior_parity_error"]) <= settings["posterior_parity_tolerance"],
                        "Terminal posterior parity error")
                check_scenarios(row, settings, False)
            outputs["terminal_summary.json"] = terminal_summary(directory, settings)
    require(all(r == references[0] for r in references), "Reference changed between time stages")
    outputs["reference.json"] = references[0]
    return outputs


def bound_tables(args, design, settings):
    _check_settings("bounds-initial", settings)
    require(settings["epsilon_levels"] == [.1, .01, .001], "Unsupported epsilon levels")
    require(args.initial is not None or args.terminal is not None or args.root, "At least one bound input required")
    expected = {c.template_id: _config_record(c) for c in budget_templates(design)}
    active = {key: expected[key] for key in ACTIVE}
    initial, roots, derived, terminal = [], [], [], {}
    initial_ids, root_ids, panel = set(), set(), None
    if args.initial is not None:
        receipt = stage(args.initial, "bounds-initial", settings)
        actual = template_records(args.initial, expected)
        rows = read(args.initial/"initial_bounds.json")
        for row in rows:
            tid = row["template_id"]
            require(tid not in initial_ids and tid in actual and row["config"] == actual[tid]
                    and row["condition_id"] == actual[tid]["condition_id"], "Initial bound identity mismatch")
            initial_ids.add(tid)
        require(initial_ids == set(actual) and receipt["initial_states"] == len(rows), "Missing initial bounds")
        parity = read(args.initial/"posterior_parity.json")
        require(len(parity) == len(actual) and {r["template_id"] for r in parity} == set(actual)
                and all(r["all_pass"] is True for r in parity) and receipt["all_parity_pass"] is True,
                "Initial posterior parity not established")
        initial = compact_initial(rows)
    for directory in args.root or []:
        receipt = stage(directory, "bounds-root", settings)
        actual = template_records(directory, active)
        require(len(actual) == 1, "One template per root stage required")
        row = read(directory/"result.json")
        tid = row["template"]
        require(tid in actual and tid not in root_ids and row["condition"] == actual[tid]["condition_id"],
                "Duplicate or mismatched root identity")
        require(row["scientific_horizon"] == actual[tid]["max_assessments"], "Scientific horizon differs")
        require(all(receipt.get(key) == value for key, value in row.items()), "Root completion/result disagreement")
        check_envelope(row)
        root_ids.add(tid)
        roots.append({key: row[key] for key in ROOT_FIELDS})
        if row["condition"] == "C5":
            derived.append(symmetry_bound(row, actual[tid], settings["epsilon_levels"]))
    if args.terminal is not None:
        receipt = stage(args.terminal, "bounds-terminal", settings)
        template_records(args.terminal, expected)
        rows = terminal_rows(args.terminal, "terminal_bounds.jsonl", receipt, active)
        terminal = terminal_counts(rows, active, settings["certificate_margin"], settings["parity_tolerance"])
        panel = read(args.terminal/"summary.json")["input_panel"]
    full_panel = panel is not None and len(panel_identities(panel, active)) == 2400
    full_panel = full_panel and panel["episodes_per_template"] == 400 and panel["root_seed"] == 2026100901
    return {"bound_summary.json": {
        "initials": initial, "roots": sorted(roots, key=lambda r: r["template"]),
        "terminal_by_condition": terminal, "input_panel": panel,
        "C5_symmetry_derived_bounds": sorted(derived, key=lambda r: r["template"]),
        "coverage": {"initial_templates": sorted(initial_ids), "root_templates": sorted(root_ids),
                     "full_initial_and_root_and_terminal_coverage": initial_ids == set(expected)
                         and root_ids == set(ACTIVE) and full_panel},
        "scope": "zero_terminal_cost_budget_only_constructed_policy_not_old_myopic_regret",
        "reader_rechecks_full_DAG": False, "bitwise_time_limited_reproduction_claimed": False,
        "machine_certified_arithmetic": False, "whole_policy_optimality_claimed": False,
        "new_random_scenarios": 0}}


def run(args):
    auxiliary = read(args.aux_config)
    auxiliary_settings(auxiliary)
    design = load_design(args.design)
    require(not args.output.exists(), "New output directory required")
    if args.stage == "source-panel":
        require(args.initial is None and args.terminal is None and not args.root and args.config is None,
                "Source panel does not consume analysis stages or analysis settings")
        outputs = source_panel(design, auxiliary, args.node)
    else:
        settings = read(args.config)
        outputs = time_tables(args, design, settings) if args.stage == "time-tables" else bound_tables(args, design, settings)
    # Publish completion only after the entire input set and all derived tables pass.
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"auxiliary_settings.json", auxiliary)
    for name, value in outputs.items():
        write(args.output/name, value)
    completion = {"stage": args.stage, "complete": True, "outputs": sorted(outputs),
                  "implementation": "new_portable_adapter_of_source_accepted_algorithms",
                  "new_random_scenarios": 0, "machine_certified_arithmetic": False,
                  "human_effect_SD_or_main_schedule_estimand_claimed": False,
                  "whole_policy_optimality_claimed": False}
    if args.stage != "source-panel":
        write(args.output/"settings.json", settings)
    write(args.output/"completion.json", completion)
    return completion
