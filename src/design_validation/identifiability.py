"""Isolated synthetic C2 responsiveness and predictive-distinguishability checks.

This module does not import or execute the continuous Gaussian implementation.
Synthetic participants have 30 *no-search* C2 trials. The independent inferential unit
is a participant, never a trial. Supplied strategies are data-generating controls,
not psychological diagnoses. All model comparisons use training-only nuisance fits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class C2Recipe:
    """Exact assigned initial targets, with the original recipe's multiplicities."""

    targets: np.ndarray
    gaps: np.ndarray
    scales: np.ndarray
    folds: np.ndarray
    template_ids: tuple[str, ...]
    denominator: int = 1000

    def __post_init__(self) -> None:
        lengths = {len(self.targets), len(self.gaps), len(self.scales), len(self.folds), len(self.template_ids)}
        if lengths != {30}:
            raise ValueError("C2 validation requires exactly 30 template slots")
        if not np.all(np.isfinite(self.targets)) or np.any((self.targets < 0) | (self.targets > 1)):
            raise ValueError("Targets must be finite feasible shares")
        if set(np.unique(self.folds)) != {0, 1, 2}:
            raise ValueError("Three folds are required")
        if len(np.unique(self.gaps)) != 3 or any(np.sum(self.gaps == g) != 10 for g in np.unique(self.gaps)):
            raise ValueError("Three gaps with ten slots each are required")
        if len(np.unique(self.scales)) != 3 or any(np.sum(self.scales == s) != 10 for s in np.unique(self.scales)):
            raise ValueError("Three scales with ten slots each are required")
        if self.denominator != 1000:
            raise ValueError("The preserved v0.1 response grid has denominator 1000")
        for gap in np.unique(self.gaps):
            if set(np.unique(self.folds[self.gaps == gap])) != {0, 1, 2}:
                raise ValueError("Every heldout fold must contain every gap")


def build_c2_recipe(design: dict[str, Any]) -> C2Recipe:
    """Use the independently implemented bounded task's exact prior means and grid."""

    from .bounded_task import belief_moments, expand_templates, initial_log_beliefs

    targets: list[float] = []
    gaps: list[float] = []
    scales: list[float] = []
    template_ids: list[str] = []
    for config in expand_templates(design):
        if config.condition_id != "C2":
            continue
        log_a, log_b = initial_log_beliefs(config)
        mean_a, _ = belief_moments(log_a)
        mean_b, _ = belief_moments(log_b)
        budget = config.total_millihours / 1000.0
        if abs(mean_a - mean_b) < 1e-9:
            raise ValueError("C2 lower-oriented target requires distinct initial means")
        target = float(project_share(0.5 - abs(mean_a - mean_b) / (2.0 * budget), 1000))
        for _ in range(config.repeat_count_per_30):
            targets.append(target)
            gaps.append(float(config.gap_offset))
            scales.append(float(config.scale))
            template_ids.append(str(config.template_id))
    gap_array = np.asarray(gaps)
    folds = np.empty(len(gaps), dtype=int)
    for gap in np.unique(gap_array):
        slots = np.flatnonzero(gap_array == gap)
        folds[slots] = np.arange(len(slots)) % 3
    return C2Recipe(np.asarray(targets), gap_array, np.asarray(scales), folds, tuple(template_ids))


def project_share(value: Any, denominator: int = 1000) -> np.ndarray:
    """Project to the response grid; half ties prefer .5, then smaller share."""

    raw = np.clip(np.asarray(value, dtype=float), 0.0, 1.0) * denominator
    lower = np.floor(raw)
    upper = np.minimum(lower + 1, denominator)
    dl, du = raw - lower, upper - raw
    tied = np.abs(dl - du) <= 1e-12
    chosen = np.where(dl < du, lower, upper)
    tie_choice = np.where(np.abs(lower - denominator / 2) <= np.abs(upper - denominator / 2), lower, upper)
    return np.where(tied, tie_choice, chosen) / denominator


def constant_share_counterexample() -> dict[str, Any]:
    """Preserve the exact original arithmetic criticism, not a calibrated model claim."""

    targets = np.asarray([(150 + 40 - 80) / 300, (150 + 45 - 75) / 300, (150 + 50 - 70) / 300])
    distances = np.abs(targets - 0.4)
    return {
        "unprojected_v01_centre_targets": targets.tolist(),
        "fixed_lower_share": 0.4,
        "absolute_distances": distances.tolist(),
        "all_match_within_005": bool(np.all(distances <= 0.05)),
        "all_positive_ES_minus_EO_distance": bool(np.all(np.abs(0.4 - 0.5) - distances > 0)),
        "overlap_example": {"targets": [0.5, 0.58], "response": 0.54, "matches_both_within_005": True},
        "interpretation": "threshold_match_and_ES_minus_EO_do_not_identify_need_adaptation",
    }


def within_person_slopes(responses: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """OLS target slope with participant intercept and linear trial-index adjustment."""

    y = np.asarray(responses, dtype=float)
    x = np.asarray(targets, dtype=float)
    if y.ndim != 2 or x.shape != y.shape or y.shape[1] < 4:
        raise ValueError("Responses/targets must have identical participant by trial shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise ValueError("OLS inputs must be finite")
    time = np.linspace(-0.5, 0.5, y.shape[1])
    x_centered = x - x.mean(axis=1, keepdims=True)
    x_residual = x_centered - ((x_centered @ time) / (time @ time))[:, None] * time
    denominator = np.sum(x_residual * x_residual, axis=1)
    if np.any(denominator < 1e-14):
        raise ValueError("Assigned target variation is absent or collinear with time")
    return np.sum(x_residual * y, axis=1) / denominator


def fit_predict_fold(
    responses: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    heldout_fold: int,
    *,
    equal_outcome: bool,
) -> np.ndarray:
    """Training-only intercept/time fit; predictions returned for all trial positions.

    EO and fixed-share models each fit exactly two nuisance coefficients. The supplied
    EO target contributes a fixed slope of one; the fixed-share model contributes zero.
    Predictions are not clipped: this preserves a common linear prediction rule. Loss
    is squared prediction error, not an endpoint-unaware probability likelihood.
    """

    y = np.asarray(responses, dtype=float)
    x = np.asarray(targets, dtype=float)
    f = np.asarray(folds)
    if y.ndim != 2 or x.shape != y.shape or f.shape != y.shape:
        raise ValueError("All arrays must share participant by trial shape")
    mask = f != heldout_fold
    if np.any(mask.sum(axis=1) < 3) or np.any((~mask).sum(axis=1) < 1):
        raise ValueError("Every participant needs training and heldout trials")
    time = np.broadcast_to(np.linspace(-0.5, 0.5, y.shape[1]), y.shape)
    base = x if equal_outcome else np.zeros_like(x)
    residual = y - base
    count = mask.sum(axis=1)
    mean_t = np.sum(time * mask, axis=1) / count
    mean_y = np.sum(residual * mask, axis=1) / count
    centered_t = time - mean_t[:, None]
    denominator = np.sum(centered_t * centered_t * mask, axis=1)
    if np.any(denominator < 1e-14):
        raise ValueError("Training time covariate is singular")
    slope_t = np.sum(centered_t * residual * mask, axis=1) / denominator
    intercept = mean_y - slope_t * mean_t
    return base + intercept[:, None] + slope_t[:, None] * time


def heldout_losses(responses: np.ndarray, targets: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Three gap-stratified folds; every trial scored once and never fits itself."""

    losses = []
    for is_eo in (False, True):
        total = np.zeros(len(responses))
        for fold in range(3):
            predictions = fit_predict_fold(responses, targets, folds, fold, equal_outcome=is_eo)
            total += np.sum(np.where(folds == fold, (responses - predictions) ** 2, 0.0), axis=1)
        losses.append(total / responses.shape[1])
    return losses[0], losses[1]


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    # Lentz evaluation of the standard incomplete-beta continued fraction.
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 301):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise ArithmeticError("Incomplete beta did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0 or a <= 0 or b <= 0:
        raise ValueError("Invalid beta inputs")
    if x == 0.0 or x == 1.0:
        return x
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_upper(t: float, degrees_freedom: int) -> float:
    """Student-t upper tail, implemented without SciPy and tested against fixtures."""

    if degrees_freedom < 1 or math.isnan(t):
        raise ValueError("Invalid Student-t inputs")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * regularized_beta(degrees_freedom / (degrees_freedom + t * t), degrees_freedom / 2.0, 0.5)
    return tail if t >= 0 else 1.0 - tail


def student_t_critical(alpha: float, degrees_freedom: int) -> float:
    if not 0.0 < alpha < 0.5:
        raise ValueError("One-sided alpha must lie between zero and .5")
    lo, hi = 0.0, 1.0
    while student_t_upper(hi, degrees_freedom) > alpha:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if student_t_upper(mid, degrees_freedom) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count < 1 or not 0 <= successes <= count:
        raise ValueError("Invalid binomial count")
    p = successes / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _valid_orders(rng: np.random.Generator, recipe: C2Recipe, count: int) -> np.ndarray:
    """Uniform permutations conditional on the declared consecutive-scale constraint."""

    order = np.argsort(rng.random((count, 30)), axis=1)
    for _ in range(100):
        scales = recipe.scales[order]
        invalid = np.any((scales[:, :-3] == scales[:, 1:-2]) & (scales[:, :-3] == scales[:, 2:-1]) & (scales[:, :-3] == scales[:, 3:]), axis=1)
        if not np.any(invalid):
            return order
        order[invalid] = np.argsort(rng.random((int(invalid.sum()), 30)), axis=1)
    raise RuntimeError("Trial-order rejection did not finish")


def simulate_recovery(
    recipe: C2Recipe,
    specification: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the frozen synthetic grid in bounded chunks with paired noise across rules."""

    settings = specification["stages"][stage]
    groups, persons = int(settings["groups"]), int(settings["participants_per_group"])
    if persons < 3 or groups < 1:
        raise ValueError("Too few synthetic groups/participants")
    root_seed = int(settings["root_seed"])
    chunk_groups = int(specification["groups_per_chunk"])
    if chunk_groups < 1:
        raise ValueError("Chunk size must be positive")
    strategies = tuple(specification["strategies"])
    required = ("fixed_share", "equal_outcome", "fixed_share_drift", "learn_equal_outcome")
    if strategies != required:
        raise ValueError("Unexpected strategy grid")
    alpha = float(specification["nominal_alpha"])
    adjusted_alpha = alpha / int(specification["family_size_for_conservative_threshold"])
    critical = student_t_critical(alpha, persons - 1)
    adjusted_critical = student_t_critical(adjusted_alpha, persons - 1)
    time = np.linspace(-0.5, 0.5, 30)
    learning = np.linspace(specification["learning_start_weight"], specification["learning_end_weight"], 30)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for noise_index, noise in enumerate(specification["motor_noise_sd"]):
        for lapse_index, lapse in enumerate(specification["lapse_probability"]):
            # Independent namespaces; same arrays are used across the four strategies.
            streams = [np.random.default_rng(np.random.SeedSequence([root_seed, noise_index, lapse_index, namespace])) for namespace in range(1, 7)]
            counters = {s: {"reject": 0, "adjusted_reject": 0, "eo": 0, "fixed": 0, "ties": 0, "beta_sum": 0.0, "beta_square_sum": 0.0} for s in strategies}
            for start in range(0, groups, chunk_groups):
                batch_groups = min(chunk_groups, groups - start)
                count = batch_groups * persons
                orders = _valid_orders(streams[0], recipe, count)
                targets, folds = recipe.targets[orders], recipe.folds[orders]
                # Balanced A/B assignment is generated independently, then responses
                # are reoriented to the lower-initial-need logical recipient.
                side_order = np.argsort(streams[1].random((count, 30)), axis=1)
                lower_is_a = side_order < 15
                bias = streams[2].normal(0, specification["participant_bias_sd"], (count, 1))
                motor = streams[3].normal(0, noise, (count, 30))
                lapses = streams[4].random((count, 30)) < lapse
                random_responses = streams[5].integers(0, recipe.denominator + 1, (count, 30)) / recipe.denominator
                fixed = float(specification["fixed_lower_share"])
                for strategy in strategies:
                    if strategy == "fixed_share":
                        intended = np.broadcast_to(fixed + bias, targets.shape)
                    elif strategy == "equal_outcome":
                        intended = targets + bias
                    elif strategy == "fixed_share_drift":
                        intended = fixed + bias + float(specification["fixed_share_total_drift"]) * time
                    else:
                        intended = fixed + bias + learning * (targets - fixed)
                    lower_response = project_share(np.where(lapses, random_responses, intended + motor), recipe.denominator)
                    displayed_a = np.where(lower_is_a, lower_response, 1.0 - lower_response)
                    y = np.where(lower_is_a, displayed_a, 1.0 - displayed_a)
                    betas = within_person_slopes(y, targets)
                    fixed_loss, eo_loss = heldout_losses(y, targets, folds)
                    difference = fixed_loss - eo_loss
                    tolerance = specification["loss_tie_tolerance"]
                    eo_wins, fixed_wins = difference > tolerance, difference < -tolerance
                    ties = ~(eo_wins | fixed_wins)
                    grouped = betas.reshape(batch_groups, persons)
                    means = grouped.mean(axis=1)
                    errors = grouped.std(axis=1, ddof=1) / math.sqrt(persons)
                    statistics = np.divide(means, errors, out=np.zeros_like(means), where=errors > 0)
                    statistics = np.where((errors == 0) & (means > 0), np.inf, statistics)
                    reject, adjusted_reject = statistics > critical, statistics > adjusted_critical
                    counter = counters[strategy]
                    for key, value in (("reject", reject.sum()), ("adjusted_reject", adjusted_reject.sum()), ("eo", eo_wins.sum()), ("fixed", fixed_wins.sum()), ("ties", ties.sum())):
                        counter[key] += int(value)
                    counter["beta_sum"] += float(betas.sum())
                    counter["beta_square_sum"] += float(np.dot(betas, betas))
                    grouped_eo = eo_wins.reshape(batch_groups, persons).sum(axis=1)
                    grouped_fixed = fixed_wins.reshape(batch_groups, persons).sum(axis=1)
                    grouped_ties = ties.reshape(batch_groups, persons).sum(axis=1)
                    for index in range(batch_groups):
                        rows.append({"stage": stage, "strategy": strategy, "motor_noise_sd": noise, "lapse_probability": lapse, "group_index": start + index, "participants": persons, "mean_beta": float(means[index]), "se_beta": float(errors[index]), "reject_nominal": bool(reject[index]), "reject_conservative": bool(adjusted_reject[index]), "eo_predictive_wins": int(grouped_eo[index]), "fixed_predictive_wins": int(grouped_fixed[index]), "predictive_ties": int(grouped_ties[index])})
            for strategy in strategies:
                counter = counters[strategy]
                participants = groups * persons
                mean_beta = counter["beta_sum"] / participants
                null = strategy in ("fixed_share", "fixed_share_drift")
                correct = counter["fixed"] if null else counter["eo"] if strategy == "equal_outcome" else None
                recovery_interval = wilson_interval(correct, participants) if correct is not None else None
                detection_interval = wilson_interval(counter["reject"], groups)
                summaries.append({
                    "strategy": strategy, "motor_noise_sd": noise, "lapse_probability": lapse,
                    "groups": groups, "participants_per_group": persons,
                    "participants_total": participants, "mean_beta": mean_beta,
                    "sd_participant_beta": math.sqrt(max(0.0, (counter["beta_square_sum"] - participants * mean_beta**2) / (participants - 1))),
                    "group_rejection_rate_nominal": counter["reject"] / groups,
                    "group_rejection_wilson95": detection_interval,
                    "group_rejection_rate_conservative": counter["adjusted_reject"] / groups,
                    "group_conservative_wilson95": wilson_interval(counter["adjusted_reject"], groups),
                    "nominal_rejections": counter["reject"], "conservative_rejections": counter["adjusted_reject"],
                    "eo_predictive_win_rate": counter["eo"] / participants,
                    "fixed_predictive_win_rate": counter["fixed"] / participants,
                    "predictive_tie_rate": counter["ties"] / participants,
                    "individual_predictive_recovery": None if correct is None else correct / participants,
                    "individual_recovery_wilson95": recovery_interval,
                    "individual_recovery_lower_ge_080": None if recovery_interval is None else recovery_interval[0] >= specification["individual_recovery_wilson_lower_gate"],
                    "null_FP_upper_le_008": None if not null else detection_interval[1] <= specification["false_positive_wilson_upper_gate"],
                    "interpretation": "learning_sensitivity_no_unique_true_model" if strategy == "learn_equal_outcome" else "synthetic_null" if null else "synthetic_EO",
                })
    designated = specification["designated_sensitivity"]
    target = next(row for row in summaries if row["strategy"] == "equal_outcome" and row["motor_noise_sd"] == designated["motor_noise_sd"] and row["lapse_probability"] == designated["lapse_probability"])
    gates = {
        "stage_eligible_for_confirmation_gates": stage == "confirmation",
        "all_null_FP_upper_le_008": all(row["null_FP_upper_le_008"] for row in summaries if row["null_FP_upper_le_008"] is not None),
        "designated_group_power_lower_ge_080": target["group_rejection_wilson95"][0] >= specification["power_wilson_lower_gate"],
        "individual_recovery_failure_cells": sum(row["individual_recovery_lower_ge_080"] is False for row in summaries),
        "note": "FP gate is inflation tolerance, not proof true type-I<=.05; human N and full Holm power are not established",
    }
    result = {"schema": "tutoring_identifiability_results_v1", "stage": stage, "root_seed": root_seed,
              "recipe_targets": recipe.targets.tolist(), "nominal_alpha": alpha,
              "conservative_alpha": adjusted_alpha, "critical_t": critical,
              "conservative_critical_t": adjusted_critical,
              "counterexample": constant_share_counterexample(), "cells": summaries, "gates": gates,
              "claims": specification["claims"], "chunk_groups": chunk_groups,
              "common_randomness": "same order, bias, motor error and lapse arrays across strategies within each noise/lapse cell"}
    return result, rows
