"""Descriptive terminal-time tables, not new trajectories or human inference.

Aggregation arithmetic follows the source-accepted descriptive reader. The
portable input adapter is new; its checks do not certify the scientific inputs.
"""
from collections import Counter, defaultdict
import math
from .finite_auxiliary_inputs import read, read_lines


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Finite descriptive quantity required")
    return float(value)


class Group:
    def __init__(self):
        self.n = 0
        self.values = defaultdict(list)
        self.categories = defaultdict(Counter)

    def add(self, values, categories):
        self.n += 1
        for name, value in values.items():
            if value is not None:
                self.values[name].append(finite(value))
            else:
                self.values[name]  # retain explicit zero-defined denominator
        for name, value in categories.items():
            self.categories[name][str(value)] += 1

    def result(self):
        metrics = {}
        for name, values in sorted(self.values.items()):
            metrics[name] = {"defined_n": len(values), "undefined_n": self.n-len(values),
                             "mean": math.fsum(values)/len(values) if values else None,
                             "minimum": min(values) if values else None,
                             "maximum": max(values) if values else None}
        return {"records": self.n, "metrics": metrics,
                "counts": {k: dict(sorted(v.items())) for k, v in sorted(self.categories.items())}}


def scenario_values(terminal, scenario, old_share=None):
    macro = scenario["selected_macro"]
    selected = terminal["macros"][macro]
    share = selected["share_A"]
    eo, raw = terminal["macros"]["EO_expected_GPA"], terminal["macros"]["EO_raw_surplus"]
    equivalence = scenario["allocation_classes"][str(selected["index"])]
    values = {"selected_expected_utility": selected["expected_utility"],
              "selected_expected_combined_GPA": selected["expected_combined_GPA"],
              "selected_duration_seconds": scenario["duration_by_macro_seconds"][macro],
              "selected_internal_objective_GPA_minus_personal_time": scenario["selected_objective"],
              "selected_share_A": share,
              "absolute_share_change_vs_old": None if share is None or old_share is None else abs(share-old_share),
              "true_EO_vs_raw_surplus_absolute_share_distance": None if eo["share_A"] is None else abs(eo["share_A"]-raw["share_A"]),
              "true_EO_minus_raw_surplus_expected_GPA": eo["expected_combined_GPA"]-raw["expected_combined_GPA"],
              "selected_objective_tie_selection_regret": scenario["objective_selection_regret"],
              "exact_exhaustion_gain_vs_stop": scenario["exact_exhaustion_gain_vs_stop"]}
    categories = {"selected_macro": macro, "no_search_status": scenario["no_search_status"],
                  "positive_feasible_ask_once": scenario["positive_feasible_ask_once"],
                  "automatic_zero_budget": terminal["automatic_zero_budget"],
                  "multiple_tolerance_tied_macros": len(scenario["tied_macros"]) > 1,
                  "selected_allocation_multiple_macro_implementations": len(equivalence) > 1,
                  "selected_allocation_equivalence_class": "|".join(equivalence)}
    asks = scenario["ask_once"]
    for action in ("ask_A", "ask_B"):
        values[action+"_feasible_gain"] = asks[action]["gain_vs_selected_stop"] if action in asks else None
    return values, categories


def preference_region(terminal, durations, rho0):
    """Independent intersection of ES-vs-each-macro half-lines, rho>=0."""
    lower, upper, possible = 0., None, True
    es = terminal["macros"]["ES"]["expected_combined_GPA"]
    constraints = []
    for name, row in terminal["macros"].items():
        if name == "ES":
            continue
        dg, dt = es-row["expected_combined_GPA"], durations["ES"]-durations[name]
        threshold = None if dt == 0 else dg/dt
        if dt == 0 and dg < 0:
            possible = False
        elif dt > 0:
            upper = threshold if upper is None else min(upper, threshold)
        elif dt < 0:
            lower = max(lower, threshold)
        constraints.append({"competitor": name, "GPA_difference": dg, "duration_difference_seconds": dt,
                            "rho_boundary": threshold, "rho0_boundary": None if threshold is None else threshold/rho0})
    possible = possible and (upper is None or lower <= upper)
    return {"nonempty": possible, "minimum_rho": lower, "maximum_rho": upper,
            "minimum_rho0_multiplier": lower/rho0, "maximum_rho0_multiplier": None if upper is None else upper/rho0,
            "constraints": constraints, "scope": "same_state_raw_terminal_menu_only_not_global_optimal_stop"}


def initial_summary(path, settings, reference):
    manifest = read(path/"manifest.json")
    if len(manifest) != settings["initial_states"]:
        raise ValueError("Complete36 initial states required, not smoke")
    seen, groups, working, ledger, c1 = set(), defaultdict(Group), defaultdict(Group), [], []
    for item in manifest:
        row = read(path/item["file"])
        template, condition = row["template_id"], row["condition_id"]
        if template in seen:
            raise ValueError("Duplicate initial template")
        seen.add(template)
        analysis, audit = row["analysis"], row["orientation_audit"]
        terminal = analysis["terminal"]
        for scenario in analysis["scenarios"]:
            key = (condition, scenario["profile"], scenario["rho_multiplier"])
            values, categories = scenario_values(terminal, scenario)
            categories["exact_exhaustion_positive_gain"] = (
                "inapplicable" if scenario["exact_exhaustion_gain_vs_stop"] is None else
                scenario["exact_exhaustion_gain_vs_stop"] > settings["decision_tolerance"])
            groups[key].add(values, categories)
            if condition != "C1" or row["working_C1_candidate"]:
                working[key].add(values, categories)
            if scenario["no_search_status"] == "numerical_all_horizon_no_search" and (
                    scenario["positive_feasible_ask_once"] or
                    (scenario["exact_exhaustion_gain_vs_stop"] is not None and scenario["exact_exhaustion_gain_vs_stop"] > settings["decision_tolerance"])):
                raise ValueError("Conflicting global no-search and feasible ask witness")
            ledger.append({"template_id": template, "condition_id": condition,
                           "working_C1_candidate": row["working_C1_candidate"],
                           "profile": scenario["profile"], "rho_multiplier": scenario["rho_multiplier"],
                           **values, **categories})
        if condition == "C1":
            profile_times = {}
            for scenario in analysis["scenarios"]:
                name = scenario["profile"]
                times = scenario["duration_by_macro_seconds"]
                if name in profile_times and profile_times[name] != times:
                    raise ValueError("Durations changed with rho")
                profile_times[name] = times
            thresholds = {name: preference_region(terminal, times, reference["rho0_GPA_per_personal_second"])
                          for name, times in profile_times.items()}
            for name, actual in thresholds.items():
                old = analysis["break_even"][name]["ES_raw_preference_region"]
                if actual["nonempty"] != old["nonempty"]:
                    raise ValueError("ES interval disagreement")
                for bound in ("minimum_rho", "maximum_rho"):
                    if (actual[bound] is None) != (old[bound] is None) or (
                            actual[bound] is not None and abs(actual[bound]-old[bound]) > 1e-10):
                        raise ValueError("ES break-even bound disagreement")
            forward, reverse = audit["forward"], audit["reverse"]
            distance = forward["ES_EO_share_distance"]
            c1.append({"template_id": template, "scale": row["config"]["scale"],
                       "total_kernel_centre_gap_minutes": row["config"]["gap_offset"],
                       "working_candidate": row["working_C1_candidate"],
                       "forward_display_native_audit": forward, "reverse_display_native_audit": reverse,
                       "mirror_mapping": audit["mapping"], "maximum_mirror_value_error": audit["maximum_value_error"],
                       "paired_resemblance_absolute_attainability_bound_percentage_points": 100*distance,
                       "bound_derivation": "reverse_triangle_inequality_for_same_state_ES_and_expected_GPA_EO_targets",
                       "human_discrimination_or_variance_estimated": False,
                       "ES_break_even_by_timing_profile": thresholds})
    if len(c1) != 12 or sum(r["working_candidate"] for r in c1) != 3:
        raise ValueError("Frozen12 C1 sensitivity states/three working scales required")
    def compact(mapping):
        return [{"condition_id": key[0], "profile": key[1], "rho_multiplier": key[2], **group.result()}
                for key, group in sorted(mapping.items())]
    return {"initial_state_ledger": ledger, "C1_audit": c1,
            "all_sensitivity_states_equal_template_summary": compact(groups),
            "working_stimulus_equal_template_summary": compact(working),
            "pooling_warning": "equal_template_diagnostic_not_human_main10_schedule_weights; all-C1 panel also mixes declared gaps"}


def terminal_summary(path, settings):
    templates, conditions, support, seen = defaultdict(Group), defaultdict(Group), Counter(), set()
    with (path/"terminal_impact.jsonl").open() as stream:
        for row in read_lines(stream):
            identity = (row["template_id"], row["episode_id"])
            if identity in seen or row["condition_id"] not in ("C3", "C5"):
                raise ValueError("Duplicate/out-of-scope terminal record")
            seen.add(identity); support[identity[0]] += 1
            terminal = row["analysis"]["terminal"]
            if len(row["analysis"]["scenarios"]) != 18:
                raise ValueError("Three profiles by six rates required at every terminal state")
            for scenario in row["analysis"]["scenarios"]:
                if scenario["ask_once"] or scenario["no_search_status"] != "not_checked":
                    raise ValueError("Conditional terminal diagnostic is not a new trajectory or stopping proof")
                values, categories = scenario_values(terminal, scenario, row["old_budget_only_share_A"])
                categories["exact_exhaustion_positive_gain"] = (
                    "inapplicable" if scenario["exact_exhaustion_gain_vs_stop"] is None else
                    scenario["exact_exhaustion_gain_vs_stop"] > settings["decision_tolerance"])
                key = (identity[0], scenario["profile"], scenario["rho_multiplier"])
                templates[key].add(values, categories)
                conditions[(row["condition_id"], key[1], key[2])].add(values, categories)
    if len(seen) != settings["prior_terminal_records"] or len(support) != 6 or set(support.values()) != {400}:
        raise ValueError("Exactly six templates x400 frozen terminal states required")
    return {"records": len(seen), "support_by_template": dict(sorted(support.items())),
            "by_template_profile_rate": [{"template_id": k[0], "profile": k[1], "rho_multiplier": k[2], **v.result()}
                                          for k, v in sorted(templates.items())],
            "by_condition_profile_rate_equal_template_diagnostic": [{"condition_id": k[0], "profile": k[1],
                                "rho_multiplier": k[2], **v.result()} for k, v in sorted(conditions.items())],
            "pooling_warning": "400 states per template makes episode pooling equal-template; not human main10 schedule weighting",
            "distribution_warning": "same_immutable_old_budget_only_myopic_terminal_states_not_new_cost_aware_on_policy_states",
            "no_human_SD_effect_or_sampling_uncertainty_estimated": True}
