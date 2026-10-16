"""Required finite budget-horizon checks, not a global metalevel solver.

Reuse the frozen finite Bayes model, but evaluate installed integer-first surplus
arithmetic. No old capped validator, task runtime or inference profile is changed.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
from itertools import combinations
import math
from collections import Counter

import numpy as np

from .bounded_task import (BoundedTask, CERTIFICATE_TOLERANCE, DECISION_TOLERANCE,
                          NEEDS, belief_moments, expand_templates, heuristic_shares,
                          normalize_log_weights, seeded_rng, utility, wilson_interval)


@lru_cache(maxsize=64)
def native_utility_table(budget, denominator, alpha, loss_multiplier):
    """Subtract exact integer need numerator BEFORE the floating division."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("Nonnegative integer resource required")
    indices = np.arange(denominator + 1, dtype=np.int64)
    need_numerator = np.arange(1, 101, dtype=np.int64) * (1000 * denominator)
    surplus = (budget * indices[:, None] - need_numerator[None, :]) / (1000 * denominator)
    table = utility(surplus, alpha, loss_multiplier)
    table.setflags(write=False)
    return table


@lru_cache(maxsize=64)
def native_perfect_information(budget, denominator, alpha, loss_multiplier):
    table = native_utility_table(budget, denominator, alpha, loss_multiplier)
    result = np.full((100, 100), -np.inf)
    for start in range(0, denominator + 1, 64):
        vals = table[start:start + 64, :, None] + table[::-1][start:start + 64, None, :]
        result = np.maximum(result, vals.max(axis=0))
    result.setflags(write=False)
    return result


class BudgetTask(BoundedTask):
    """Finite one-step computations on the installed arithmetic/share grid."""
    def terminal_values(self, log_A, log_B, budget_millihours):
        c = self.config
        table = native_utility_table(budget_millihours, c.allocation_denominator, c.alpha, c.loss_multiplier)
        return table @ np.exp(normalize_log_weights(log_A)) + table[::-1] @ np.exp(normalize_log_weights(log_B))

    def one_step_values(self, log_A, log_B, budget_millihours, assessments=0):
        c = self.config
        share, stop = self.terminal_solution(log_A, log_B, budget_millihours)
        result = {"stop_value": stop, "stop_share": share, "ask_A_value": None,
                  "ask_B_value": None, "gain_A": None, "gain_B": None, "chosen_action": "stop"}
        if assessments >= c.max_assessments or budget_millihours < c.cost_millihours:
            return result
        table = native_utility_table(budget_millihours-c.cost_millihours, c.allocation_denominator, c.alpha, c.loss_multiplier)
        pA, pB = np.exp(normalize_log_weights(log_A)), np.exp(normalize_log_weights(log_B))
        prob_A, post_A = self.predictive(log_A)
        prob_B, post_B = self.predictive(log_B)
        vals_A = np.exp(post_A) @ table.T + (table[::-1] @ pB)[None, :]
        vals_B = (table @ pA)[None, :] + np.exp(post_B) @ table[::-1].T
        ask_A = float(prob_A @ vals_A.max(axis=1))
        ask_B = float(prob_B @ vals_B.max(axis=1))
        result.update(ask_A_value=ask_A, ask_B_value=ask_B, gain_A=ask_A-stop, gain_B=ask_B-stop)
        if max(ask_A, ask_B) > stop+DECISION_TOLERANCE:
            result["chosen_action"] = "ask_A" if ask_A >= ask_B-DECISION_TOLERANCE else "ask_B"
        return result

    def perfect_information_after_first_cost(self, log_A, log_B, budget_millihours):
        c = self.config
        if budget_millihours < c.cost_millihours or not c.max_assessments:
            return {"status": "no_request_feasible", "upper_bound_gain": None}
        matrix = native_perfect_information(budget_millihours-c.cost_millihours, c.allocation_denominator, c.alpha, c.loss_multiplier)
        full = float(np.exp(normalize_log_weights(log_A)) @ matrix @ np.exp(normalize_log_weights(log_B)))
        _, stop = self.terminal_solution(log_A, log_B, budget_millihours)
        gain = full-stop
        return {"upper_bound_gain": gain, "perfect_information_value_after_cost": full,
                "stop_value": stop, "margin": CERTIFICATE_TOLERANCE,
                "status": "numerical_all_horizon_no_search" if gain < -CERTIFICATE_TOLERANCE else "inconclusive",
                "machine_certified_arithmetic": False}


def budget_templates(design):
    return [replace(c, max_assessments=c.total_millihours//c.cost_millihours)
            for c in expand_templates(design)]


def native_realized(config, share, budget, needs):
    if budget and share is None:
        raise ValueError("Positive resource requires a share")
    index = 0 if budget == 0 else round(share*config.allocation_denominator)
    table = native_utility_table(budget, config.allocation_denominator, config.alpha, config.loss_multiplier)
    return float(table[index, needs[0]-1]+table[config.allocation_denominator-index, needs[1]-1])


def initial_diagnostics(config):
    task = BudgetTask(config)
    values = task.initial_values()
    bound = task.perfect_information_after_first_cost(*task.initial_logs, config.total_millihours)
    matrix = task.terminal_values(*task.initial_logs, config.total_millihours)
    mirror = task.terminal_values(task.initial_logs[1], task.initial_logs[0], config.total_millihours)[::-1]
    mirror_step = task.one_step_values(task.initial_logs[1], task.initial_logs[0], config.total_millihours)
    error = float(np.max(abs(matrix-mirror)))
    ask_error = max(abs(values["ask_A_value"]-mirror_step["ask_B_value"]),
                    abs(values["ask_B_value"]-mirror_step["ask_A_value"]))
    targets = heuristic_shares(*task.initial_logs, config.total_millihours)
    stop_share = values["stop_share"]
    no_search = config.condition_id in ("C1", "C2", "C4")
    gain = max(values["gain_A"], values["gain_B"])
    return {"config": asdict(config), "moments": [belief_moments(p) for p in task.initial_logs],
            "decision": values, "all_horizon_bound": bound, "heuristic_shares": targets,
            "absolute_gaps": {k: abs(stop_share-v) for k, v in targets.items()},
            "mirror_value_error": error, "mirror_ask_value_error": ask_error,
            "expected_acquisition": "no_search" if no_search else "initial_search",
            "acquisition_check": bound["upper_bound_gain"] < -CERTIFICATE_TOLERANCE if no_search else gain > DECISION_TOLERANCE,
            "acquisition_evidence": bound["status"] if no_search else "positive_one_step_feasible_policy_lower_bound" if gain > DECISION_TOLERANCE else "inconclusive_initial_search",
            "global_optimal_terminal_policy_certified": no_search and bound["status"] == "numerical_all_horizon_no_search"}


def source_parity(panel, configs):
    by_id = {c.template_id: c for c in configs}
    max_probability = max_moment = max_outcome = 0.
    for row in panel["beliefCases"]:
        task = BudgetTask(by_id[row["templateId"]])
        logs = task.initial_logs[row["student"]]
        for report in row["reports"]:
            logs = task.posterior(logs, report)
        mean, sd = belief_moments(logs)
        differences = NEEDS[:, None]-NEEDS[None, :]
        rmse = float(np.sqrt(np.sum(np.exp(normalize_log_weights(logs))[:, None]*np.exp(task.log_likelihood)*differences**2)))
        max_probability = max(max_probability, float(np.max(abs(np.exp(normalize_log_weights(logs))-row["probabilities"]))))
        max_moment = max(max_moment, abs(mean-row["mean"]), abs(sd-row["sd"]), abs(rmse-row["predictiveRMSE"]))
    for row in panel["outcomeCases"]:
        c = by_id[row["templateId"]]
        b, j, needs = row["budget"], row["index"], row["needs"]
        share = None if b == 0 else j/1000
        value = native_realized(c, share, b, needs)
        expected_amounts = [0, 0] if b == 0 else [b*j, b*(1000-j)]
        if expected_amounts != row["amountsMicrohours"]:
            raise ValueError("Native integer numerator mismatch")
        max_outcome = max(max_outcome, abs(value-row["utility"]), abs(7+.03*value-row["gpa"]))
    checks = {"probability": max_probability <= 1e-12, "moment": max_moment <= 1e-10,
              "outcome": max_outcome <= 1e-10}
    return {"checks": checks, "all_pass": all(checks.values()), "belief_cases": len(panel["beliefCases"]),
            "outcome_cases": len(panel["outcomeCases"]), "maximum_probability_error": max_probability,
            "maximum_moment_error": max_moment, "maximum_utility_gpa_error": max_outcome}


def schedule_geometry(panel, configs):
    """All nine multisets × all252 balanced side masks, order-invariant model only."""
    tasks = {c.template_id: BudgetTask(c) for c in configs}
    logical_x = {key: (belief_moments(task.initial_logs[0])[0]-belief_moments(task.initial_logs[1])[0])/(task.config.total_millihours/1000)
                 for key, task in tasks.items()}
    rows, actual, weights = [], [], {}
    if len(panel["schedules"]) != 45:
        raise ValueError("Exactly45 current schedules required")
    for item in panel["schedules"]:
        definitions = item["definitions"]
        if len(definitions) != 10 or sum(d["sideReversed"] for d in definitions) != 5:
            raise ValueError("Ten main definitions with balanced sides required")
        counts = Counter(d["templateId"] for d in definitions)
        for d in definitions:
            c = tasks[d["templateId"]].config
            if (d["budgetMillihours"], d["costMillihours"], d["capacity"]) != (c.total_millihours, c.cost_millihours, c.max_assessments):
                raise ValueError("Installed resource/capacity differs from finite template")
        cid = item["conditionId"]
        weights.setdefault(cid, Counter()).update(counts)
        base = np.array([logical_x[d["templateId"]] for d in definitions])
        x = base*np.array([-1 if d["sideReversed"] else 1 for d in definitions])
        actual.append({"condition_id": cid, "variant": item["variant"], "x": x.tolist(),
                       "Sxx": float(np.sum((x-x.mean())**2)), "span": float(np.ptp(x)),
                       "template_counts": dict(counts)})
        if cid not in ("C2", "C3", "C4"):
            continue
        for mask in combinations(range(10), 5):
            signs = np.ones(10); signs[list(mask)] = -1
            all_x = base*signs
            rows.append({"condition_id": cid, "variant": item["variant"], "reversed_indices": list(mask),
                         "x": all_x.tolist(), "Sxx": float(np.sum((all_x-all_x.mean())**2)), "span": float(np.ptp(all_x))})
    if len(rows) != 6804 or any(sum(w.values()) != 90 for w in weights.values()):
        raise ValueError("Geometry/variant weight count mismatch")
    extrema = {}
    for cid in ("C2", "C3", "C4"):
        own = [r for r in rows if r["condition_id"] == cid]
        extrema[cid] = {"minimum_Sxx": min(r["Sxx"] for r in own), "maximum_span": max(r["span"] for r in own),
                        "minimum_span": min(r["span"] for r in own), "cases": len(own)}
    return {"actual_schedules": actual, "balanced_geometry": rows, "extrema": extrema,
            "logical_weights": {cid: {k: v/90 for k, v in sorted(w.items())} for cid, w in weights.items()},
            "scope": "All current variant multisets and balanced5/5 masks; unadjusted/exchangeable/order-invariant only"}


def scenario(config, root, episode):
    task = BudgetTask(config)
    needs, streams = [], []
    for student, logs in enumerate(task.initial_logs):
        need = int(seeded_rng(root, config.template_id, episode, f"budget_latent_{student}").choice(NEEDS, p=np.exp(logs)))
        stream = seeded_rng(root, config.template_id, episode, f"budget_reports_{student}").choice(NEEDS, size=config.max_assessments,
                          p=np.exp(task.log_likelihood[need-1])).astype(int).tolist()
        needs.append(need); streams.append(stream)
    return {"root_seed": root, "template_id": config.template_id, "episode_id": episode,
            "true_needs": needs, "potential_reports": streams}


def run_path(task, supplied):
    c = task.config
    logs = [p.copy() for p in task.initial_logs]
    budget, counts, events = c.total_millihours, [0, 0], []
    while True:
        k = sum(counts)
        decision = task.initial_values() if k == 0 else task.one_step_values(*logs, budget, k)
        if decision["chosen_action"] == "stop":
            reason = "budget_insufficient" if budget < c.cost_millihours else "count_cap" if k >= c.max_assessments else "myopic_stop"
            break
        who = 0 if decision["chosen_action"] == "ask_A" else 1
        report = supplied["potential_reports"][who][counts[who]]
        logs[who] = task.posterior(logs[who], report)
        counts[who] += 1; budget -= c.cost_millihours
        events.append({"student": who, "report": report, "budget_after": budget,
                       "gain_A": decision["gain_A"], "gain_B": decision["gain_B"]})
    share, expected = task.terminal_solution(*logs, budget)
    targets = heuristic_shares(*logs, budget)
    gaps = {key: None if share is None else abs(share-value) for key, value in targets.items()}
    matches = {key: None if value is None else value <= .05+1e-12 for key, value in gaps.items()}
    lower = None if share is None else bool(matches["all_to_lower"] or matches["meet_lower_first"])
    value = native_realized(c, share, budget, supplied["true_needs"])
    immediate = native_realized(c, task.initial_values()["stop_share"], c.total_millihours, supplied["true_needs"])
    return {"sample_count": sum(counts), "counts": counts, "events": events, "stop_reason": reason,
            "remaining_millihours": budget, "budget_exhausted": budget == 0, "over_twenty": sum(counts) > 20,
            "final_share_A": share, "posterior_expected_utility": expected, "realized_utility": value,
            "gain_vs_immediate_optimum": value-immediate, "total_GPA": 7+.03*value,
            "terminal_logs": [p.tolist() for p in logs], "heuristic_shares": targets,
            "absolute_gaps": gaps, "matches": matches, "lower_family_match": lower,
            "target_match": matches["equal_outcome"] if c.condition_id == "C3" else lower,
            "EO_vs_split_diagnostic": None if share is None else abs(targets["equal_outcome"]-.5) >= .08-1e-12,
            "lower_variants_diagnostic": None if share is None else abs(targets["all_to_lower"]-targets["meet_lower_first"]) >= .08-1e-12}


def paired_episode(config, root, episode):
    shared = scenario(config, root, episode)
    full = run_path(BudgetTask(config), shared)
    capped = run_path(BudgetTask(replace(config, max_assessments=20)), shared)
    return {"condition_id": config.condition_id, "template_id": config.template_id, "episode_id": episode,
            "scenario": shared, "budget_only": full, "capped_twenty": capped,
            "paired_full_minus_capped_utility": full["realized_utility"]-capped["realized_utility"],
            "identical_policy_events": full["events"] == capped["events"],
            "identical_final_share": full["final_share_A"] == capped["final_share_A"]}


def mean_interval(records, field):
    values = np.array([r[field] for r in records], dtype=float)
    variance = 0.
    for template in {r["template_id"] for r in records}:
        own = np.array([r[field] for r in records if r["template_id"] == template], dtype=float)
        variance += (len(own)/len(values))**2 * (float(np.var(own, ddof=1))/len(own) if len(own) > 1 else 0.)
    se = math.sqrt(variance)
    return {"mean": float(values.mean()), "stratified_SE": se, "normal_95_interval": [float(values.mean()-1.96*se), float(values.mean()+1.96*se)],
            "interval_scope": "Stratified normal approximation; synthetic not human"}


def proportion(values):
    known = [v for v in values if v is not None]
    n, successes = len(known), sum(bool(v) for v in known)
    return {"valid_n": n, "undefined_n": len(values)-n, "count": successes, "rate": successes/n if n else None,
            "wilson_95_interval": wilson_interval(successes, n), "interval_scope": "Descriptive Wilson approximation for stratified independent outcomes"}


def summarize_pairs(records):
    result = []
    for label, own in [(cid, [r for r in records if r["condition_id"] == cid]) for cid in ("C3", "C5")] + [
            (tid, [r for r in records if r["template_id"] == tid]) for tid in sorted({r["template_id"] for r in records})]:
        if not own:
            continue
        policies = {}
        for policy in ("budget_only", "capped_twenty"):
            rows = [{**r[policy], "template_id": r["template_id"]} for r in own]
            stats = {key: mean_interval(rows, key) for key in ("realized_utility", "posterior_expected_utility", "gain_vs_immediate_optimum", "sample_count")}
            stats.update({key: proportion([r[key] for r in rows]) for key in ("over_twenty", "budget_exhausted", "target_match", "EO_vs_split_diagnostic", "lower_family_match", "lower_variants_diagnostic")})
            stats["stop_reasons"] = dict(Counter(r["stop_reason"] for r in rows))
            stats["question_histogram"] = dict(sorted(Counter(r["sample_count"] for r in rows).items()))
            stats["heuristics"] = {h: {"match": proportion([r["matches"][h] for r in rows]),
                                            "mean_absolute_gap": float(np.mean([r["absolute_gaps"][h] for r in rows if r["absolute_gaps"][h] is not None])) if any(r["absolute_gaps"][h] is not None for r in rows) else None}
                                   for h in rows[0]["matches"]}
            policies[policy] = stats
        full = policies["budget_only"]
        checks = {"target_match_lower_ge_0_80": full["target_match"]["wilson_95_interval"][0] >= .8,
                  "over_twenty_upper_lt_0_01": full["over_twenty"]["wilson_95_interval"][1] < .01,
                  "paired_gain_lower_gt_zero": full["gain_vs_immediate_optimum"]["normal_95_interval"][0] > 0}
        if own[0]["condition_id"] == "C3":
            checks["C3_diagnostic_lower_ge_0_25"] = full["EO_vs_split_diagnostic"]["wilson_95_interval"][0] >= .25
        result.append({"label": label, "condition_id": own[0]["condition_id"], "pairs": len(own), "policies": policies,
                       "paired_full_minus_capped_utility": mean_interval(own, "paired_full_minus_capped_utility"),
                       "different_event_paths": sum(not r["identical_policy_events"] for r in own),
                       "different_final_shares": sum(not r["identical_final_share"] for r in own),
                       "checks": checks, "all_checks_pass": all(checks.values()),
                       "policy_scope": "Repeated exact one-step myopic; NOT globally optimal adaptive terminal behavior"})
    return result
