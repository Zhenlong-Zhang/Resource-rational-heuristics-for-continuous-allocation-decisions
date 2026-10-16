"""Exact finite-task calculations and explicitly myopic acquisition validation.

This module implements the *proposed* bounded tutoring task, not the historical
Gaussian model. It has no imports from that scientific workstream. Terminal
values and one-step report expectations enumerate finite supports; a repeated
myopic policy is not a globally optimal finite-horizon metalevel policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

NEEDS = np.arange(1, 101, dtype=float)
DECISION_TOLERANCE = 1e-10
CERTIFICATE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class TaskConfig:
    condition_id: str
    template_id: str
    scale: float
    gap_offset: float
    prior_centres: tuple[float, float]
    prior_spreads: tuple[float, float]
    report_width: float
    total_millihours: int
    cost_millihours: int
    repeat_count_per_30: int
    allocation_denominator: int = 1000
    max_assessments: int = 20
    alpha: float = 0.5
    loss_multiplier: float = 2.0
    gpa_baseline: float = 3.5
    gpa_scale: float = 0.03

    def __post_init__(self) -> None:
        if min(self.prior_spreads) <= 0 or self.report_width <= 0:
            raise ValueError("Kernel spreads must be positive")
        if not 0 < self.cost_millihours < self.total_millihours:
            raise ValueError("Both requests must be initially affordable")
        if self.allocation_denominator < 2 or self.allocation_denominator % 2:
            raise ValueError("Share denominator must be positive and even")
        if self.max_assessments < 0 or self.alpha <= 0 or self.loss_multiplier <= 0:
            raise ValueError("Invalid cap or utility parameters")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_design(path: str | Path) -> dict[str, Any]:
    """Validate supported task semantics instead of silently ignoring changes."""
    data = json.loads(Path(path).read_text())
    common = data["common"]
    required = {
        "posterior": "exact_discrete_bayes_over_100_points",
        "initial_information_mode": "direct_specified_prior_not_simulated_historical_prior_samples",
    }
    if any(common.get(key) != expected for key, expected in required.items()):
        raise ValueError("Unsupported posterior/initial-information semantics")
    if common["latent_need"].get("family") != "finite_normalized_gaussian_kernel":
        raise ValueError("Unsupported need family")
    if common["observation"].get("family") != "finite_normalized_gaussian_kernel_conditional_on_true_need":
        raise ValueError("Unsupported observation family")
    for key in ("independent_recipients", "fixed_within_trial", "independent_new_students_across_trials"):
        if common["latent_need"].get(key) is not True:
            raise ValueError(f"Unsupported latent-state semantics: {key}")
    for key in ("normalization_depends_on_latent_need", "independent_given_latent_need", "assessment_does_not_teach"):
        if common["observation"].get(key) is not True:
            raise ValueError(f"Unsupported report semantics: {key}")
    allocation = common["allocation"]
    if (allocation.get("share_grid_denominator") != 1000
            or allocation.get("input_step_percentage_points") != .1
            or allocation.get("canonical_share_index_range") != [0, 1000]
            or allocation.get("initial_response") is not None
            or allocation.get("include_exact_zero_and_all_remaining") is not True
            or allocation.get("stop_irreversible") is not True):
        raise ValueError("Unsupported allocation interface semantics")
    if (data["assignment"].get("design") != "between_subjects"
            or data["assignment"].get("conditions_per_participant") != 1
            or data["assignment"].get("main_trials") != 30
            or common["variation"].get("side_assignment_counts") != [15, 15]):
        raise ValueError("Unsupported assignment/recipe semantics")
    for key in ("latent_need", "observation"):
        if tuple(common[key][x] for x in ("support_min", "support_max", "support_step")) != (1, 100, 1):
            raise ValueError("This validator requires the frozen integer 1–100 support")
    if common["stop_time_cost"] != 0 or common["wall_clock_time_cost"] != 0:
        raise ValueError("This candidate implementation has zero stop/wall-clock cost")
    if common["learning_rates"] != [1, 1] or common["pretrial_sample_count_per_student"] != 0:
        raise ValueError("Only equal-rate direct-prior candidate semantics are supported")
    if common["utility"]["gain_exponent"] != common["utility"]["loss_exponent"]:
        raise ValueError("Gain and loss exponents must match")
    if common["gpa"]["clipping"] or common["maximum_online_assessments"] != 20:
        raise ValueError("Unexpected clipping or request-cap change")
    if [c["id"] for c in data["conditions"]] != [f"C{i}" for i in range(1, 6)]:
        raise ValueError("Exactly the five ordered candidate conditions are required")
    return data


def expand_templates(design: dict[str, Any]) -> list[TaskConfig]:
    common = design["common"]
    scales = common["variation"]["scale_factors"]
    counts = common["variation"]["gap_scale_cell_counts_for_three_gap_conditions"]
    result = []
    for row in design["conditions"]:
        for si, scale in enumerate(scales):
            for gi, gap in enumerate(row["mean_gap_offsets"]):
                centre = row["prior_kernel_centres"]
                raw_time = row["total_time"] * scale * 1000
                raw_cost = row["sample_time_cost"] * scale * 1000
                if abs(raw_time-round(raw_time)) > 1e-7 or abs(raw_cost-round(raw_cost)) > 1e-7:
                    raise ValueError("Candidate time cannot be represented in millihours")
                result.append(TaskConfig(
                    condition_id=row["id"], template_id=f"{row['id']}_s{si}_g{gi}",
                    scale=float(scale), gap_offset=float(gap),
                    prior_centres=((centre[0]+gap)*scale, (centre[1]-gap)*scale),
                    prior_spreads=tuple(x*scale for x in row["prior_kernel_spreads"]),
                    report_width=row["report_kernel_width"]*scale,
                    total_millihours=round(raw_time), cost_millihours=round(raw_cost),
                    repeat_count_per_30=10 if len(row["mean_gap_offsets"]) == 1 else counts[si][gi],
                    allocation_denominator=common["allocation"]["share_grid_denominator"],
                    max_assessments=common["maximum_online_assessments"],
                    alpha=common["utility"]["gain_exponent"],
                    loss_multiplier=common["utility"]["loss_multiplier"],
                    gpa_baseline=common["gpa"]["baseline"], gpa_scale=common["gpa"]["utility_scale"],
                ))
    for cid in [f"C{i}" for i in range(1, 6)]:
        if sum(c.repeat_count_per_30 for c in result if c.condition_id == cid) != 30:
            raise ValueError("Template multiplicities must sum to 30 per condition")
    return result


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    maximum = np.max(x, axis=axis, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("Log weights must have finite maximum")
    value = maximum + np.log(np.sum(np.exp(x-maximum), axis=axis, keepdims=True))
    return np.squeeze(value, axis=axis) if axis is not None else value.reshape(())


def normalize_log_weights(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x-logsumexp(x)


def initial_log_beliefs(config: TaskConfig) -> tuple[np.ndarray, np.ndarray]:
    return tuple(normalize_log_weights(-0.5*((NEEDS-m)/s)**2)
                 for m, s in zip(config.prior_centres, config.prior_spreads))  # type: ignore[return-value]


def belief_moments(log_p: np.ndarray) -> tuple[float, float]:
    p = np.exp(normalize_log_weights(log_p))
    mean = float(p @ NEEDS)
    return mean, float(np.sqrt(p @ (NEEDS-mean)**2))


def utility(x: np.ndarray | float, alpha: float = 0.5, loss_multiplier: float = 2.0) -> np.ndarray:
    value = np.asarray(x, dtype=float)
    return np.where(value >= 0, np.maximum(value, 0)**alpha,
                    -loss_multiplier*np.maximum(-value, 0)**alpha)


def project_share(share: float, denominator: int = 1000) -> float:
    target = float(np.clip(share, 0, 1))*denominator
    lo, hi = int(math.floor(target)), int(math.ceil(target))
    minimum = min(abs(lo-target), abs(hi-target))
    candidates = [j for j in {lo, hi} if abs(j-target) <= minimum+1e-12]
    return min(candidates, key=lambda j: (abs(j-denominator/2), j))/denominator


def heuristic_shares(log_A: np.ndarray, log_B: np.ndarray, budget_millihours: int,
                     denominator: int = 1000) -> dict[str, float | None]:
    names = ("equal_split", "equal_outcome", "all_to_lower", "meet_lower_first",
             "greatest_need", "proportional_need")
    if budget_millihours <= 0:
        return dict.fromkeys(names)
    a, _ = belief_moments(log_A)
    b, _ = belief_moments(log_B)
    B = budget_millihours/1000
    tie = abs(a-b) <= 1e-9*max(1, a, b)
    raw = {"equal_split": .5, "equal_outcome": (B+a-b)/(2*B),
           "all_to_lower": .5 if tie else float(a < b),
           "meet_lower_first": .5 if tie else min(B, a)/B if a < b else 1-min(B, b)/B,
           "greatest_need": .5 if tie else float(a > b),
           "proportional_need": a/(a+b)}
    return {key: project_share(value, denominator) for key, value in raw.items()}


@lru_cache(maxsize=32)
def _utility_table(budget_millihours: int, denominator: int, alpha: float,
                   loss_multiplier: float) -> np.ndarray:
    # All quantities are exact integer microhours before floating utility evaluation.
    amount = np.arange(denominator+1, dtype=float)*budget_millihours/(1000*denominator)
    table = utility(amount[:, None]-NEEDS[None, :], alpha, loss_multiplier)
    table.setflags(write=False)
    return table


@lru_cache(maxsize=32)
def _perfect_information_matrix(budget_millihours: int, denominator: int,
                                alpha: float, loss_multiplier: float) -> np.ndarray:
    table = _utility_table(budget_millihours, denominator, alpha, loss_multiplier)
    best = np.full((100, 100), -np.inf)
    for start in range(0, denominator+1, 64):
        stop = min(denominator+1, start+64)
        values = table[start:stop, :, None] + table[::-1][start:stop, None, :]
        best = np.maximum(best, values.max(axis=0))
    best.setflags(write=False)
    return best


def seeded_rng(root_seed: int, template_id: str, episode_id: int, namespace: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{root_seed}|{template_id}|{episode_id}|{namespace}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


class BoundedTask:
    def __init__(self, config: TaskConfig):
        self.config = config
        raw = -0.5*((NEEDS[:, None]-NEEDS[None, :])/config.report_width)**2
        self.log_likelihood = raw-logsumexp(raw, axis=1)[:, None]
        self.initial_logs = initial_log_beliefs(config)
        self._initial_values: dict[str, Any] | None = None

    def posterior(self, log_p: np.ndarray, observation: int) -> np.ndarray:
        if observation < 1 or observation > 100 or int(observation) != observation:
            raise ValueError("Report must be an integer in 1–100")
        return normalize_log_weights(log_p+self.log_likelihood[:, observation-1])

    def predictive(self, log_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint = normalize_log_weights(log_p)[:, None]+self.log_likelihood
        log_probability = logsumexp(joint, axis=0)
        log_posteriors = (joint-log_probability[None, :]).T
        return np.exp(log_probability), log_posteriors

    def terminal_values(self, log_A: np.ndarray, log_B: np.ndarray,
                        budget_millihours: int) -> np.ndarray:
        if budget_millihours < 0:
            raise ValueError("Negative budget")
        c = self.config
        table = _utility_table(budget_millihours, c.allocation_denominator, c.alpha, c.loss_multiplier)
        return table @ np.exp(normalize_log_weights(log_A)) + table[::-1] @ np.exp(normalize_log_weights(log_B))

    def terminal_solution(self, log_A: np.ndarray, log_B: np.ndarray,
                          budget_millihours: int) -> tuple[float | None, float]:
        values = self.terminal_values(log_A, log_B, budget_millihours)
        if budget_millihours == 0:
            return None, float(values[0])
        best = float(values.max())
        candidates = np.flatnonzero(values >= best-1e-12*max(1, abs(best)))
        index = min(candidates, key=lambda j: (abs(int(j)-self.config.allocation_denominator/2), int(j)))
        return int(index)/self.config.allocation_denominator, float(values[index])

    def one_step_values(self, log_A: np.ndarray, log_B: np.ndarray,
                        budget_millihours: int, assessments: int = 0) -> dict[str, Any]:
        c = self.config
        share, stop_value = self.terminal_solution(log_A, log_B, budget_millihours)
        result = {"stop_value": stop_value, "stop_share": share, "ask_A_value": None,
                  "ask_B_value": None, "gain_A": None, "gain_B": None, "chosen_action": "stop"}
        if assessments >= c.max_assessments or budget_millihours < c.cost_millihours:
            return result
        next_budget = budget_millihours-c.cost_millihours
        table = _utility_table(next_budget, c.allocation_denominator, c.alpha, c.loss_multiplier)
        pA, pB = np.exp(normalize_log_weights(log_A)), np.exp(normalize_log_weights(log_B))
        probability_A, post_A = self.predictive(log_A)
        probability_B, post_B = self.predictive(log_B)
        values_A = np.exp(post_A) @ table.T + (table[::-1] @ pB)[None, :]
        values_B = (table @ pA)[None, :] + np.exp(post_B) @ table[::-1].T
        ask_A, ask_B = float(probability_A @ values_A.max(axis=1)), float(probability_B @ values_B.max(axis=1))
        result.update(ask_A_value=ask_A, ask_B_value=ask_B, gain_A=ask_A-stop_value, gain_B=ask_B-stop_value)
        if max(ask_A, ask_B) > stop_value+DECISION_TOLERANCE:
            result["chosen_action"] = "ask_A" if ask_A >= ask_B-DECISION_TOLERANCE else "ask_B"
        return result

    def initial_values(self) -> dict[str, Any]:
        if self._initial_values is None:
            self._initial_values = self.one_step_values(*self.initial_logs, self.config.total_millihours)
        return dict(self._initial_values)

    def perfect_information_after_first_cost(self, log_A: np.ndarray, log_B: np.ndarray,
                                             budget_millihours: int) -> dict[str, Any]:
        c = self.config
        if budget_millihours < c.cost_millihours or c.max_assessments == 0:
            return {"upper_bound_gain": None, "status": "no_request_feasible"}
        matrix = _perfect_information_matrix(budget_millihours-c.cost_millihours,
                                             c.allocation_denominator, c.alpha, c.loss_multiplier)
        full_info = float(np.exp(normalize_log_weights(log_A)) @ matrix @ np.exp(normalize_log_weights(log_B)))
        _, stop = self.terminal_solution(log_A, log_B, budget_millihours)
        gap = full_info-stop
        return {"upper_bound_gain": gap, "perfect_information_value_after_cost": full_info,
                "stop_value": stop, "margin": CERTIFICATE_TOLERANCE,
                "status": "numerical_no_search_certificate" if gap < -CERTIFICATE_TOLERANCE else "inconclusive"}

    def scenario(self, root_seed: int, episode_id: int) -> dict[str, Any]:
        c = self.config
        needs, streams = [], []
        for index, log_p in enumerate(self.initial_logs):
            rng = seeded_rng(root_seed, c.template_id, episode_id, f"latent_{index}")
            need = int(rng.choice(NEEDS, p=np.exp(log_p)))
            report_rng = seeded_rng(root_seed, c.template_id, episode_id, f"observations_{index}")
            reports = report_rng.choice(NEEDS, size=c.max_assessments,
                                        p=np.exp(self.log_likelihood[need-1])).astype(int)
            needs.append(need)
            streams.append(reports.tolist())
        return {"true_needs": needs, "potential_reports": streams, "root_seed": root_seed,
                "template_id": c.template_id, "episode_id": episode_id}


def scarcity_band(budget: float, need_A: float, need_B: float) -> str:
    if budget < min(need_A, need_B):
        return "neither_individually_feasible"
    if budget < max(need_A, need_B):
        return "only_lower_individually_feasible"
    if budget < need_A+need_B:
        return "either_but_not_both_feasible"
    return "both_jointly_feasible"


def realized_utility(config: TaskConfig, share: float | None, budget_millihours: int,
                     true_needs: list[int]) -> float:
    if budget_millihours == 0:
        amounts = np.zeros(2)
    else:
        if share is None:
            raise ValueError("Positive budget requires an allocation")
        index = round(share*config.allocation_denominator)
        # Form the exact integer numerator before division. Ratio-first floating
        # multiplication can turn exactly 57 hours into 56.99999999999999,
        # which falsely crosses a square-root utility kink at a need of 57.
        numerators = np.array([index, config.allocation_denominator-index], dtype=np.int64)
        numerators *= budget_millihours
        amounts = numerators.astype(float)/(1000*config.allocation_denominator)
    return float(utility(amounts-np.asarray(true_needs), config.alpha, config.loss_multiplier).sum())


def simulate_episode(task: BoundedTask, root_seed: int, episode_id: int) -> dict[str, Any]:
    c = task.config
    scenario = task.scenario(root_seed, episode_id)
    logs = [p.copy() for p in task.initial_logs]
    budget, counts, events = c.total_millihours, [0, 0], []
    sensitivity_states = [{"kind": "initial", "log_A": logs[0].copy(), "log_B": logs[1].copy(),
                           "budget_millihours": budget, "assessments": 0}]
    while True:
        k = sum(counts)
        decision = task.initial_values() if k == 0 else task.one_step_values(*logs, budget, k)
        if decision["chosen_action"] == "stop":
            reason = "cap" if k >= c.max_assessments else "insufficient_budget" if budget < c.cost_millihours else "myopic_stop"
            break
        who = 0 if decision["chosen_action"] == "ask_A" else 1
        observation = scenario["potential_reports"][who][counts[who]]
        before = [belief_moments(p) for p in logs]
        logs[who] = task.posterior(logs[who], observation)
        budget -= c.cost_millihours
        counts[who] += 1
        events.append({"action": decision["chosen_action"], "observation": observation,
                       "gain_A": decision["gain_A"], "gain_B": decision["gain_B"],
                       "budget_after_millihours": budget, "moments_before": before,
                       "moments_after": [belief_moments(p) for p in logs]})
        if sum(counts) == 1:
            sensitivity_states.append({"kind": "first_post_query", "log_A": logs[0].copy(),
                                       "log_B": logs[1].copy(), "budget_millihours": budget, "assessments": 1})
    share, expected = task.terminal_solution(*logs, budget)
    final_moments = [belief_moments(p) for p in logs]
    targets = heuristic_shares(*logs, budget, c.allocation_denominator)
    gaps = {h: abs(share-a) if share is not None and a is not None else None for h, a in targets.items()}
    matches = {h: x <= .05+1e-12 if x is not None else None for h, x in gaps.items()}
    actual = realized_utility(c, share, budget, scenario["true_needs"])
    initial_optimum = task.initial_values()["stop_share"]
    immediate_optimum = realized_utility(c, initial_optimum, c.total_millihours, scenario["true_needs"])
    immediate_equal = realized_utility(c, .5, c.total_millihours, scenario["true_needs"])
    lower_family_match = bool(matches["all_to_lower"] or matches["meet_lower_first"])
    target_match = matches["equal_split"] if c.condition_id == "C1" else matches["equal_outcome"] if c.condition_id in ("C2", "C3") else lower_family_match
    record = {
        "condition_id": c.condition_id, "template_id": c.template_id, "episode_id": episode_id,
        "scale": c.scale, "gap_offset": c.gap_offset, "policy": "exact_finite_one_step_myopic",
        "scenario": scenario, "events": events, "sample_count": sum(counts), "sample_counts": counts,
        "any_search": bool(sum(counts)), "stop_reason": reason, "cap_binding": sum(counts) == c.max_assessments,
        "remaining_millihours": budget, "final_share_A": share,
        "terminal_log_A": logs[0].tolist(), "terminal_log_B": logs[1].tolist(),
        "terminal_moments": final_moments, "posterior_expected_utility": expected,
        "realized_utility": actual, "total_GPA": 2*c.gpa_baseline+c.gpa_scale*actual,
        "immediate_optimum_utility": immediate_optimum, "immediate_equal_utility": immediate_equal,
        "gain_vs_immediate_optimum": actual-immediate_optimum, "gain_vs_equal_split": actual-immediate_equal,
        "heuristic_shares": targets, "absolute_gaps": gaps, "matches": matches,
        "target_match": bool(target_match), "lower_family_match": lower_family_match,
        "EO_vs_split_diagnostic": abs(targets["equal_outcome"]-.5) >= .08-1e-12 if budget else False,
        "lower_variants_diagnostic": abs(targets["all_to_lower"]-targets["meet_lower_first"]) >= .08-1e-12 if budget else False,
        "true_scarcity_band": scarcity_band(budget/1000, *scenario["true_needs"]),
        "mean_scarcity_band": scarcity_band(budget/1000, final_moments[0][0], final_moments[1][0]),
    }
    sensitivity_states.append({"kind": "terminal", "log_A": logs[0].copy(), "log_B": logs[1].copy(),
                               "budget_millihours": budget, "assessments": sum(counts)})
    record["_sensitivity_states"] = sensitivity_states
    return record


def initial_diagnostics(task: BoundedTask) -> dict[str, Any]:
    c = task.config
    decision = task.initial_values()
    bound = task.perfect_information_after_first_cost(*task.initial_logs, c.total_millihours)
    return {"config": asdict(c), "initial_moments": [belief_moments(p) for p in task.initial_logs],
            "decision": decision, "perfect_information_bound": bound,
            "acquisition_evidence": "positive_one_step_gain_proves_stopping_dominated"
            if decision["chosen_action"] != "stop" else bound["status"],
            "initial_heuristics": heuristic_shares(*task.initial_logs, c.total_millihours, c.allocation_denominator)}


def numerical_sensitivity(task: BoundedTask, states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fine = BoundedTask(replace(task.config, allocation_denominator=2*task.config.allocation_denominator))
    result = []
    for state in states:
        args = (state["log_A"], state["log_B"], state["budget_millihours"], state["assessments"])
        coarse_value, fine_value = task.one_step_values(*args), fine.one_step_values(*args)
        share_gap = abs(coarse_value["stop_share"]-fine_value["stop_share"]) if coarse_value["stop_share"] is not None else None
        result.append({"template_id": task.config.template_id, "state_kind": state["kind"],
                       "budget_millihours": state["budget_millihours"], "assessments": state["assessments"],
                       "coarse": coarse_value, "fine": fine_value, "terminal_share_gap": share_gap,
                       "terminal_value_gain_fine": fine_value["stop_value"]-coarse_value["stop_value"],
                       "chosen_action_agrees": coarse_value["chosen_action"] == fine_value["chosen_action"]})
    return result


def wilson_interval(successes: int, total: int) -> list[float]:
    if not total:
        return [0, 1]
    z, p = 1.959963984540054, successes/total
    denominator = 1+z*z/total
    centre = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0, centre-half), min(1, centre+half)]


def summarize_records(records: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for cid in sorted({r["condition_id"] for r in records}):
        rows = [r for r in records if r["condition_id"] == cid]
        n = len(rows)
        summary: dict[str, Any] = {"condition_id": cid, "episodes": n, "synthetic_not_human": True}
        summary["proportion_interval_note"] = "Wilson intervals are descriptive approximations for stratified independent Bernoulli outcomes, not exact-binomial or global-policy confidence proofs."
        for field in ("realized_utility", "posterior_expected_utility", "gain_vs_immediate_optimum",
                      "gain_vs_equal_split", "sample_count", "remaining_millihours"):
            values = np.array([r[field] for r in rows], dtype=float)
            variance = 0.0
            for template in {r["template_id"] for r in rows}:
                x = np.array([r[field] for r in rows if r["template_id"] == template], dtype=float)
                variance += (len(x)/n)**2 * float(x.var(ddof=1))/len(x) if len(x)>1 else 0
            se = math.sqrt(variance)
            summary[field] = {"mean": float(values.mean()), "stratified_standard_error": se,
                              "normal_approx_95_interval": [float(values.mean()-1.96*se), float(values.mean()+1.96*se)]}
        for field in ("any_search", "target_match", "lower_family_match", "EO_vs_split_diagnostic",
                      "lower_variants_diagnostic", "cap_binding"):
            count = sum(bool(r[field]) for r in rows)
            summary[field] = {"count": count, "rate": count/n, "wilson_95_interval": wilson_interval(count, n)}
        summary["heuristics"] = {}
        for h in rows[0]["matches"]:
            valid = [r for r in rows if r["matches"][h] is not None]
            count = sum(r["matches"][h] for r in valid)
            summary["heuristics"][h] = {"valid_n": len(valid), "match_count": count,
                "match_rate": count/len(valid) if valid else None,
                "wilson_95_interval": wilson_interval(count, len(valid)),
                "mean_absolute_share_gap": float(np.mean([r["absolute_gaps"][h] for r in valid])) if valid else None}
        initial = [d for d in diagnostics if d["config"]["condition_id"] == cid]
        exact_search = sum(d["config"]["repeat_count_per_30"]*(d["decision"]["chosen_action"] != "stop") for d in initial)/30
        summary["any_search"]["exact_template_weighted_probability"] = exact_search
        summary["any_search"]["note"] = "Initial policy is deterministic per frozen template; exact weighted probability is known, not estimated from episodes."
        expected_search = cid in ("C3", "C5")
        summary["point_screening"] = {
            "acquisition_target_met": (exact_search if expected_search else 1-exact_search) >= .8,
            "allocation_target_met": summary["target_match"]["rate"] >= .8,
            "EO_discriminability_met": summary["EO_vs_split_diagnostic"]["rate"] >= .25 if cid in ("C2", "C3") else None,
            "cap_rate_below_0_01": summary["cap_binding"]["rate"] < .01,
            "scope": "point estimates only; no global metalevel or human claim"}
        summary["initial_certificate_status_counts"] = {status: sum(d["acquisition_evidence"] == status for d in initial)
            for status in sorted({d["acquisition_evidence"] for d in initial})}
        summaries.append(summary)
    return summaries
