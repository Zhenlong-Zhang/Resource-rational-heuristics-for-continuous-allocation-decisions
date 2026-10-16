"""Duration-priced terminal macros; task GPA is never deducted.

The numerical finite-grid operations define idealized macros, not human mental
algorithms or measured durations. Personal question time is not priced here.
Nothing in this module changes the installed task, old policies or inference.
"""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP
import math

import numpy as np

from .bounded_task import (NEEDS, belief_moments, heuristic_shares,
                          normalize_log_weights, project_share, utility)
from .budget_horizon import BudgetTask, budget_templates, native_perfect_information, native_utility_table

MACROS = ("ES", "EO_expected_GPA", "EO_raw_surplus", "PN", "AL", "GN", "ML",
          "MFN_A", "MFN_B", "OPT_expected_GPA")
OPERATIONS = ("read_scalar", "retrieve_rule", "add_subtract", "multiply_divide", "compare", "clip")
INVENTORIES = {
    "ES": (0, 1, 0, 0, 0, 0),
    "EO_expected_GPA": (3, 1, 0, 0, 0, 1),
    "EO_raw_surplus": (3, 1, 2, 1, 0, 1),
    "PN": (2, 1, 1, 1, 0, 1),
    "AL": (2, 1, 0, 0, 1, 0), "GN": (2, 1, 0, 0, 1, 0),
    "ML": (3, 1, 1, 1, 1, 1),
    "MFN_A": (2, 1, 1, 1, 0, 1), "MFN_B": (2, 1, 1, 1, 0, 1),
    "OPT_expected_GPA": (3, 1, 0, 0, 0, 1),
}
MIRROR_MACRO = {name: name for name in MACROS}
MIRROR_MACRO.update(MFN_A="MFN_B", MFN_B="MFN_A")


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name+" must be finite and nonnegative")
    return float(value)


def validate_settings(settings):
    required = {"schema": "tutoring-terminal-personal-time-v1",
                "predecessor_profile": "pilot-engagement-budget-v1",
                "personal_question_time_priced": False,
                "C1_gap_scaled": False,
                "automatic_zero_budget_duration_seconds": 0.,
                "empirically_calibrated_cognitive_durations": False,
                "machine_certified_arithmetic": False}
    if any(settings.get(k) != v for k, v in required.items()):
        raise ValueError("Unsupported terminal personal-time semantics")
    if settings["C1_total_gaps_minutes"] != [0., .2, .5, 1.] or settings["working_C1_total_gap_minutes"] != 1.:
        raise ValueError("Frozen C1 proposals changed")
    if settings["rho_multipliers"] != [0., .01, .1, 1., 10., 100.]:
        raise ValueError("Frozen rate sensitivity changed")
    if settings["C1_independent_inference_seconds"] != [0., 5., 20., 60.]:
        raise ValueError("Frozen inference sensitivity changed")
    if set(settings["assumed_profiles"]) != {"low", "base", "high"}:
        raise ValueError("Three assumed timing profiles required")
    nonnegative(settings["common_execution_seconds"], "common duration")
    for profile in settings["assumed_profiles"].values():
        for key in (*OPERATIONS, "EO_inference_seconds", "OPT_inference_seconds"):
            nonnegative(profile[key], key)
    return settings


def analysis_templates(design, settings):
    validate_settings(settings)
    old = budget_templates(design)
    result = [c for c in old if c.condition_id != "C1"]
    for c in old:
        if c.condition_id != "C1":
            continue
        for gap in settings["C1_total_gaps_minutes"]:
            result.append(replace(c, template_id=f"C1_s{c.template_id.split('_')[1][1:]}_d{round(gap*1000):04d}",
                                  gap_offset=gap, prior_centres=(50*c.scale-gap/2, 50*c.scale+gap/2)))
    if len(result) != settings["initial_states"]:
        raise ValueError("Frozen initial-state count changed")
    return sorted(result, key=lambda c: c.template_id)


def timing_profiles(settings, include_independent=False):
    profiles = {k: dict(v) for k, v in settings["assumed_profiles"].items()}
    if include_independent:
        for eo in settings["C1_independent_inference_seconds"]:
            for opt in settings["C1_independent_inference_seconds"]:
                profiles[f"base_EO{int(eo):03d}_OPT{int(opt):03d}"] = {
                    **settings["assumed_profiles"]["base"], "EO_inference_seconds": eo,
                    "OPT_inference_seconds": opt}
    return profiles


def macro_durations(profile, common_seconds=2.65, budget=1):
    nonnegative(common_seconds, "common seconds")
    for key in (*OPERATIONS, "EO_inference_seconds", "OPT_inference_seconds"):
        nonnegative(profile[key], key)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("Integer nonnegative budget required")
    if budget == 0:
        return dict.fromkeys(MACROS, 0.)
    result = {m: common_seconds+sum(n*profile[k] for k, n in zip(OPERATIONS, INVENTORIES[m])) for m in MACROS}
    result["EO_expected_GPA"] += profile["EO_inference_seconds"]
    result["OPT_expected_GPA"] += profile["OPT_inference_seconds"]
    return result


def _probabilities(logs):
    x = np.asarray(logs, dtype=float)
    if x.shape != (100,) or np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError("Finite-support log belief required")
    return np.exp(normalize_log_weights(x))


def expected_curves(config, log_A, log_B, budget):
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("Integer nonnegative budget required")
    table = native_utility_table(budget, config.allocation_denominator, config.alpha, config.loss_multiplier)
    ua, ub = table @ _probabilities(log_A), table[::-1] @ _probabilities(log_B)
    return ua, ub, config.gpa_baseline+config.gpa_scale*ua, config.gpa_baseline+config.gpa_scale*ub


def _near(values, best, maximize=True):
    tolerance = 1e-12*max(1., abs(best))
    return np.flatnonzero(values >= best-tolerance if maximize else values <= best+tolerance)


def _grid_choice(indices, denominator):
    return int(min(indices, key=lambda j: (abs(int(j)-denominator/2), int(j))))


def terminal_summary(config, log_A, log_B, budget):
    """Raw and tolerance-selected targets/values; no timing assumptions applied."""
    ua, ub, ga, gb = expected_curves(config, log_A, log_B, budget)
    total, difference, D = ga+gb, abs(ga-gb), config.allocation_denominator
    raw_max, min_gap = float(total.max()), float(difference.min())
    opt_candidates = _near(total, raw_max)
    eo_primary = _near(difference, min_gap, maximize=False)
    eo_secondary_max = float(total[eo_primary].max())
    eo_candidates = eo_primary[total[eo_primary] >= eo_secondary_max-1e-12*max(1., abs(eo_secondary_max))]
    opt, eo = _grid_choice(opt_candidates, D), _grid_choice(eo_candidates, D)
    means = [belief_moments(log_A), belief_moments(log_B)]
    if budget:
        old = heuristic_shares(log_A, log_B, budget, D)
        indices = {"ES": D//2, "EO_expected_GPA": eo, "OPT_expected_GPA": opt,
                   "EO_raw_surplus": round(old["equal_outcome"]*D),
                   "PN": round(old["proportional_need"]*D), "AL": round(old["all_to_lower"]*D),
                   "GN": round(old["greatest_need"]*D), "ML": round(old["meet_lower_first"]*D),
                   "MFN_A": round(project_share(min(budget/1000, means[0][0])/(budget/1000), D)*D),
                   "MFN_B": round(project_share(1-min(budget/1000, means[1][0])/(budget/1000), D)*D)}
    else:
        indices = dict.fromkeys(MACROS, None)
    rows = {}
    for macro in MACROS:
        index = indices[macro]
        j = 0 if index is None else index
        rows[macro] = {"index": index, "share_A": None if index is None else index/D,
                       "amount_A_minutes": budget*j/(1000*D), "amount_B_minutes": budget*(D-j)/(1000*D),
                       "expected_utility": float(ua[j]+ub[j]), "expected_GPA_A": float(ga[j]),
                       "expected_GPA_B": float(gb[j]), "expected_combined_GPA": float(total[j]),
                       "expected_GPA_gap": float(difference[j]),
                       "regret_vs_raw_grid_GPA_max": max(0., raw_max-float(total[j]))}
    return {"budget_milliminutes": budget, "moments": [list(v) for v in means], "macros": rows,
            "raw_grid_max_combined_GPA": raw_max, "raw_EO_minimum_GPA_gap": min_gap,
            "EO_selected_gap_excess": max(0., float(difference[eo])-min_gap),
            "OPT_tied_indices": opt_candidates.tolist(), "EO_tied_indices": eo_candidates.tolist(),
            "automatic_zero_budget": budget == 0, "machine_certified_arithmetic": False}


def select_macro(summary, durations, rho):
    rho = nonnegative(rho, "GPA per personal second")
    times = np.array([nonnegative(durations[m], m+" duration") for m in MACROS])
    values = np.array([summary["macros"][m]["expected_combined_GPA"] for m in MACROS])-rho*times
    best = float(values.max())
    ties = _near(values, best)
    index = int(min(ties, key=lambda j: (times[j], int(j))))
    chosen = MACROS[index]
    allocation_classes = {}
    for m in MACROS:
        key = str(summary["macros"][m]["index"])
        allocation_classes.setdefault(key, []).append(m)
    return {"rho_GPA_per_second": rho, "selected_macro": chosen,
            "selected_index": summary["macros"][chosen]["index"],
            "selected_objective": float(values[index]), "raw_max_objective": best,
            "objective_selection_regret": max(0., best-float(values[index])),
            "tied_macros": [MACROS[j] for j in ties], "allocation_classes": allocation_classes,
            "objective_by_macro": {m: float(v) for m, v in zip(MACROS, values)},
            "duration_by_macro_seconds": {m: float(t) for m, t in zip(MACROS, times)}}


def break_even(summary, durations, rho0):
    if not math.isfinite(rho0) or rho0 <= 0:
        raise ValueError("Positive reference GPA/second required")
    pairs, lower, upper, possible = [], 0., None, True
    for i, first in enumerate(MACROS):
        for second in MACROS[i+1:]:
            g = summary["macros"][first]["expected_combined_GPA"]-summary["macros"][second]["expected_combined_GPA"]
            d = durations[first]-durations[second]
            threshold = None if d == 0 else g/d
            pairs.append({"first": first, "second": second, "GPA_difference": g, "duration_difference_seconds": d,
                          "rho_threshold": threshold, "rho0_multiplier_threshold": None if threshold is None else threshold/rho0,
                          "equal_duration_dominance": None if d else "tie" if g == 0 else first if g > 0 else second})
    for m in MACROS[1:]:
        g = summary["macros"]["ES"]["expected_combined_GPA"]-summary["macros"][m]["expected_combined_GPA"]
        d = durations["ES"]-durations[m]
        if d > 0:
            value = g/d
            upper = value if upper is None else min(upper, value)
        elif d < 0:
            lower = max(lower, g/d)
        elif g < 0:
            possible = False
    possible = possible and (upper is None or upper >= lower)
    return {"pairs": pairs, "ES_raw_preference_region": {"nonempty": possible, "minimum_rho": lower,
            "maximum_rho": upper, "minimum_rho0_multiplier": lower/rho0,
            "maximum_rho0_multiplier": None if upper is None else upper/rho0,
            "scope": "same_positive_or_automatic_terminal_menu_not_global_stopping"}}


def reference_rate(old_templates):
    configs = [c for c in old_templates if c.condition_id == "C1" and c.scale == 1.]
    if len(configs) != 1:
        raise ValueError("One old symmetric scale1 C1 required")
    c, one = configs[0], 1000
    if c.prior_centres != (50., 50.) or c.total_millihours != 120000:
        raise ValueError("Reference convention changed")
    task = BudgetTask(c)
    before = terminal_summary(c, *task.initial_logs, c.total_millihours)
    after = terminal_summary(c, *task.initial_logs, c.total_millihours+one)
    rho0 = after["raw_grid_max_combined_GPA"]-before["raw_grid_max_combined_GPA"]
    if not math.isfinite(rho0) or rho0 <= 0:
        raise ValueError("Reference tutoring benefit must be positive")
    probs = [_probabilities(logs) for logs in task.initial_logs]
    def named_extra(amounts):
        return sum(c.gpa_baseline+c.gpa_scale*float(utility((amount-NEEDS.astype(np.int64)*1000)/1000,
                                      c.alpha, c.loss_multiplier) @ p) for amount, p in zip(amounts, probs))
    return {"rho0_GPA_per_personal_second": rho0, "reference_template": c.template_id,
            "resource_before_minutes": 120., "resource_after_minutes": 121.,
            "raw_optimal_GPA_before": before["raw_grid_max_combined_GPA"],
            "raw_optimal_GPA_after": after["raw_grid_max_combined_GPA"],
            "ES_fixed_share_benefit": after["macros"]["ES"]["expected_combined_GPA"]-before["macros"]["ES"]["expected_combined_GPA"],
            "A_only_non_native_benefit": named_extra((61000, 60000))-named_extra((60000, 60000)),
            "B_only_non_native_benefit": named_extra((60000, 61000))-named_extra((60000, 60000)),
            "empirical_personal_valuation": False, "state_dependent": False}


def perfect_information_GPA(task, logs, budget):
    c = task.config
    if budget < c.cost_millihours:
        return None
    matrix = native_perfect_information(budget-c.cost_millihours, c.allocation_denominator, c.alpha, c.loss_multiplier)
    return 2*c.gpa_baseline+c.gpa_scale*float(_probabilities(logs[0]) @ matrix @ _probabilities(logs[1]))


def exact_exhaustion_GPA(task, logs, budget):
    if budget % task.config.cost_millihours:
        return None
    return terminal_summary(task.config, *logs, 0)["raw_grid_max_combined_GPA"]


def one_step_tables(task, logs, budget):
    """Complete all100-report data, independent of durations and rho."""
    if budget < task.config.cost_millihours:
        return None
    child_budget = budget-task.config.cost_millihours
    result = {}
    for who, action in enumerate(("ask_A", "ask_B")):
        probabilities, posteriors = task.predictive(logs[who])
        if len(probabilities) != 100 or np.any(probabilities <= 0) or abs(float(probabilities.sum())-1.) > 1e-12:
            raise ValueError("All100 positive normalized report probabilities required")
        summaries = []
        for post in posteriors:
            child_logs = (post, logs[1]) if who == 0 else (logs[0], post)
            summaries.append(terminal_summary(task.config, *child_logs, child_budget))
        result[action] = {"report_ids": list(range(1, 101)), "probabilities": probabilities,
                          "children": summaries, "child_budget": child_budget}
    return result


def evaluate_state(task, logs, budget, settings, reference, include_independent=False, acquisition=True):
    summary = terminal_summary(task.config, *logs, budget)
    rho0 = reference["rho0_GPA_per_personal_second"]
    branch_data = one_step_tables(task, logs, budget) if acquisition else None
    upper = perfect_information_GPA(task, logs, budget) if acquisition else None
    burn = exact_exhaustion_GPA(task, logs, budget)
    rows, thresholds = [], {}
    for profile_name, profile in timing_profiles(settings, include_independent).items():
        durations = macro_durations(profile, settings["common_execution_seconds"], budget)
        thresholds[profile_name] = break_even(summary, durations, rho0)
        for multiple in settings["rho_multipliers"]:
            rho = multiple*rho0
            selected = select_macro(summary, durations, rho)
            asks = {}
            if branch_data is not None:
                child_times = macro_durations(profile, settings["common_execution_seconds"], budget-task.config.cost_millihours)
                for action, branches in branch_data.items():
                    policies = [select_macro(child, child_times, rho) for child in branches["children"]]
                    feasible = float(branches["probabilities"] @ np.array([p["selected_objective"] for p in policies]))
                    raw = float(branches["probabilities"] @ np.array([p["raw_max_objective"] for p in policies]))
                    asks[action] = {"feasible_ask_once_objective": feasible, "raw_best_ask_once_objective": raw,
                                    "gain_vs_selected_stop": feasible-selected["selected_objective"],
                                    "child_macro_by_report": [p["selected_macro"] for p in policies],
                                    "reports": 100}
            status = "not_checked" if not acquisition else "structural_stop" if branch_data is None else (
                "numerical_all_horizon_no_search" if selected["selected_objective"] > upper+settings["classification_margin"]
                else "inconclusive")
            gain = max((v["gain_vs_selected_stop"] for v in asks.values()), default=None)
            rows.append({"profile": profile_name, "rho_multiplier": multiple, **selected,
                         "ask_once": asks, "positive_feasible_ask_once": gain is not None and gain > settings["decision_tolerance"],
                         "continuation_PI_GPA_upper": upper, "no_search_status": status,
                         "exact_exhaustion_feasible_GPA": burn,
                         "exact_exhaustion_gain_vs_stop": None if burn is None else burn-selected["selected_objective"]})
    return {"terminal": summary, "break_even": thresholds, "scenarios": rows,
            "personal_question_time_priced": False, "global_adaptive_policy_solved": False,
            "new_random_scenarios": 0, "machine_certified_arithmetic": False}


def display_number(value, places):
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def display_audit(summary):
    rows, macros = {}, summary["macros"]
    for macro, row in macros.items():
        rows[macro] = {"index": row["index"], "share_A": row["share_A"],
                      "native_amounts_minutes": [row["amount_A_minutes"], row["amount_B_minutes"]],
                      "display_amounts_minutes": [display_number(row["amount_A_minutes"], 1), display_number(row["amount_B_minutes"], 1)],
                      "raw_individual_GPA": [row["expected_GPA_A"], row["expected_GPA_B"]],
                      "display_individual_GPA": [display_number(row["expected_GPA_A"], 2), display_number(row["expected_GPA_B"], 2)]}
    es, eo = rows["ES"], rows["EO_expected_GPA"]
    distance = None if es["share_A"] is None else abs(es["share_A"]-eo["share_A"])
    means = [m[0] for m in summary["moments"]]
    rendered = [display_number(x, 1) for x in means]
    return {"actual_finite_means_minutes": means, "actual_mean_gap_minutes": means[1]-means[0],
            "display_means_minutes": rendered, "display_mean_gap_minutes": rendered[1]-rendered[0],
            "macro_targets": rows, "ES_EO_native_distinct": es["index"] != eo["index"],
            "ES_EO_rendered_amounts_distinct": es["display_amounts_minutes"] != eo["display_amounts_minutes"],
            "ES_EO_share_distance": distance, "ES_EO_old_5pp_regions_overlap": None if distance is None else distance <= .1+1e-12,
            "rounding": "decimal_half_up_proposed_not_installed_browser_parity", "human_discriminability_established": False}


def mirror_audit(task, logs, budget):
    forward = terminal_summary(task.config, *logs, budget)
    reverse = terminal_summary(task.config, logs[1], logs[0], budget)
    mapping, maximum = {}, 0.
    for name in MACROS:
        a, b = forward["macros"][name], reverse["macros"][MIRROR_MACRO[name]]
        error = abs(a["expected_combined_GPA"]-b["expected_combined_GPA"])
        maximum = max(maximum, error)
        mapping[name] = {"reverse_macro": MIRROR_MACRO[name], "forward_index": a["index"],
                         "reverse_index": b["index"], "combined_GPA_error": error}
    return {"forward": display_audit(forward), "reverse": display_audit(reverse), "mapping": mapping,
            "maximum_value_error": maximum, "smaller_index_tie_representative_may_not_mirror": True}
