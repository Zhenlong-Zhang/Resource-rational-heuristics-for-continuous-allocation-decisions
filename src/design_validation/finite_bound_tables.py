"""Compact numerical envelopes with raw and C5 symmetry-derived bounds separate.

This new portable reader retains the source-accepted presentation algebra. It
checks input structure, not every Bellman witness or floating-point operation.
"""
from collections import Counter

from .finite_auxiliary_inputs import number, require


ROOT_FIELDS = ("template", "condition", "lower", "upper", "gap", "action_status",
               "action_bounds", "witness_action", "created_states", "expanded_states",
               "max_materialized_depth", "elapsed_seconds",
               "elapsed_including_final_checkpoint_seconds", "rss_bytes", "limit_reason",
               "attained_epsilon_levels")


def check_envelope(row):
    lower, upper = number(row["lower"]), number(row["upper"])
    require(lower <= upper and number(row["gap"]) == upper-lower, "Invalid raw root interval/gap")
    require(set(row["action_bounds"]) == {"stop", "ask_A", "ask_B"}, "Complete root action intervals required")
    for interval in row["action_bounds"].values():
        require(number(interval["lower"]) <= number(interval["upper"]), "Inverted action interval")
    require(row["machine_certified_arithmetic"] is False
            and row["whole_policy_optimality_claimed"] is False, "Numerical constructed-policy scope required")


def compact_initial(rows):
    initial = []
    for row in rows:
        seed, once = row["frontier"], row["one_step"]
        check_envelope(seed)
        check_envelope(once)
        initial.append({"template": row["template_id"], "condition": row["condition_id"],
                        "stop": seed["initial_details"]["stop_grid_max"],
                        "PI_upper_gain": seed["initial_details"]["perfect_information_upper_gain_vs_stop"],
                        "one_step_gap": once["gap"], "status": once["action_status"],
                        "best_feasible_ask_gain_lower": max(v["lower"] for a, v in once["action_bounds"].items()
                                                           if a != "stop") - once["action_bounds"]["stop"]["upper"],
                        "selector_regret": seed["initial_details"]["installed_selector_regret"]})
    return initial


def symmetry_bound(row, task, epsilon_levels):
    """Intersect equal-recipient action intervals without changing the witness."""
    check_envelope(row)
    require(row["condition"] == task["condition_id"] == "C5"
            and row["template"] == task["template_id"], "C5 root identity required")
    require(task["prior_centres"][0] == task["prior_centres"][1]
            and task["prior_spreads"][0] == task["prior_spreads"][1]
            and task["allocation_denominator"] == 1000
            and row["initial_details"]["analytical_initial_recipient_symmetry"] is True,
            "C5 initial recipient symmetry assumptions failed")
    actions = row["action_bounds"]
    common_lower = max(actions[a]["lower"] for a in ("ask_A", "ask_B"))
    common_upper = min(actions[a]["upper"] for a in ("ask_A", "ask_B"))
    require(common_lower <= common_upper, "Inverted symmetric-action intersection")
    lower = max(actions["stop"]["lower"], common_lower)
    upper = max(actions["stop"]["upper"], common_upper)
    require(lower <= upper <= row["upper"], "Invalid derived root envelope")
    require(lower == row["lower"], "Derived lower changed the concrete witness scope")
    gap = upper-lower
    return {"template": row["template"], "raw_lower": row["lower"],
            "raw_upper": row["upper"], "raw_gap": row["gap"],
            "derived_lower": lower, "derived_upper": upper, "derived_gap": gap,
            "attained_epsilon_levels": [x for x in epsilon_levels if gap <= x],
            "same_feasible_witness": True,
            "method": "explicit_postfreeze_initial_recipient_symmetry_intersection",
            "machine_certified_arithmetic": False, "whole_policy_optimality_claimed": False}


def terminal_counts(rows, configs, margin, parity_tolerance):
    counts = {}
    for row in rows:
        condition = row["template_id"].split("_", 1)[0]
        require(condition in ("C3", "C5") and row["machine_certified_arithmetic"] is False,
                "Old active-template terminal diagnostic required")
        config = configs[row["template_id"]]
        state = row["state"]
        require(len(state) == 4 and all(type(v) is int and v >= 0 for v in state),
                "Four nonnegative integer sufficient statistics required")
        require(all(state[i] <= state[i+1] <= 100*state[i] for i in (0, 2)),
                "Report sum outside count-implied support")
        budget = config["total_millihours"]-(state[0]+state[2])*config["cost_millihours"]
        require(budget >= 0 and row["details"]["budget_millihours"] == budget,
                "Terminal sufficient statistics/budget mismatch")
        require(0 <= number(row["log_parity_error"]) <= parity_tolerance, "Terminal log parity failed")
        expected = {"ask_A", "ask_B"} if budget >= config["cost_millihours"] else set()
        require(set(row["ask_upper"]) == expected, "Incomplete feasible terminal action set")
        require(number(row["stop_lower"]) <= number(row["stop_upper"]), "Inverted terminal stop interval")
        stopping = not row["ask_upper"] or all(row["stop_lower"] > number(v)+margin
                                              for v in row["ask_upper"].values())
        status = "numerical_global_stop_at_sampled_terminal" if stopping else "inconclusive_sampled_terminal"
        require(row["status"] == status, "Terminal stopping label inconsistent with supplied interval")
        counts.setdefault(condition, Counter())[row["status"]] += 1
    return {key: dict(value) for key, value in counts.items()}
