from __future__ import annotations

"""Scarcity policies, oracle screen, pairing, and frozen inference rules."""

import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ..mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig, TrueState
from ..policies.heuristic import (
    EqualSplitBaselinePolicy,
    ImmediateAllToLowerPolicy,
    ImmediateMeetLowerFirstPolicy,
    ManualActiveSearchAllToLowerPolicy,
    ManualActiveSearchMeetLowerFirstPolicy,
    ScarcityGreatestNeedPolicy,
    all_to_lower_allocation,
    effort_to_goal,
    greatest_effort_need_allocation,
    lower_effort_identity,
    meet_lower_first_allocation,
)
from ..policies.voi import MyopicValueOfInformationPolicy
from .active_search_evaluation import (
    deterministic_realized_utility,
    full_information_utilitarian_allocation,
    wilson_interval,
)
from .randomization import EvaluationEpisode


SCARCITY_SCHEMA_VERSION = 1
SCARCITY_ROOT_SEED = 18_2026_0820
SCARCITY_STREAM_CAPACITY_PER_RECIPIENT = 60
SCARCITY_ALLOCATION_TOLERANCE = 0.05
SCARCITY_MORE_TO_LOWER_THRESHOLD = 0.55
SCARCITY_EFFORT_TIE_RELATIVE_TOLERANCE = 1e-9
SCARCITY_SUPPORT_THRESHOLD = 0.80
SCARCITY_RETAINED_GAIN_FRACTION = 0.80
SCARCITY_ORACLE_GRID_SIZE = 4001
SCARCITY_ORACLE_DENSE_GRID_SIZE = 16001
SCARCITY_DEVELOPMENT_EPISODES = 120
SCARCITY_CONFIRMATION_EPISODES = 1200
SCARCITY_VOI_DRAWS = 500

SCARCITY_DETERMINISTIC_DELTAS = (0.0, 0.2, 0.4, 0.6, 0.8)
SCARCITY_CAPACITY_RATIOS = (0.25, 0.40, 0.50, 0.70, 0.90, 1.05)
SCARCITY_UTILITY_EXPONENTS = (0.25, 0.50, 0.75)
SCARCITY_LAMBDA_SHORTFALLS = (1.0, 2.0, 4.0)
SCARCITY_SIGMA_NEEDS = (5.0, 10.0, 15.0)
SCARCITY_GAUSSIAN_CAPACITY_RATIOS = (0.25, 0.50, 0.75, 0.95, 1.05)
SCARCITY_SIGMA_SAMPLES = (2.0, 10.0, 30.0)
SCARCITY_SAMPLE_TIME_COST_PERCENTS = (0.01, 0.10, 1.00)
SCARCITY_PRIOR_SAMPLE_COUNTS = (0, 5, 20)

SCARCITY_ORACLE_MEDOID_LEVELS: Mapping[str, Tuple[float, ...]] = {
    "sigma_need": SCARCITY_SIGMA_NEEDS,
    "capacity_ratio": SCARCITY_GAUSSIAN_CAPACITY_RATIOS,
    "utility_exponent": SCARCITY_UTILITY_EXPONENTS,
    "lambda_shortfall": SCARCITY_LAMBDA_SHORTFALLS,
}

SCARCITY_ALLOCATION_METRIC_FIELDS = (
    "classification_need_1",
    "classification_need_2",
    "classification_uses_hidden_true_state",
    "need_1_positive",
    "need_2_positive",
    "both_needs_positive",
    "need_1_nonpositive",
    "need_2_nonpositive",
    "either_need_nonpositive",
    "effort_to_goal_1",
    "effort_to_goal_2",
    "lower_effort_identity",
    "lower_raw_need_identity",
    "effort_identity_differs_from_raw_need",
    "effort_tie",
    "joint_goal_feasible",
    "at_least_lower_goal_meetable",
    "exactly_one_goal_individually_meetable",
    "both_individually_but_not_jointly_meetable",
    "neither_goal_meetable",
    "all_to_lower_allocation",
    "meet_lower_first_allocation",
    "greatest_need_allocation",
    "lower_pattern_overlap",
    "lower_recipient_allocation_share",
    "more_to_lower",
    "approximately_equal_allocation",
    "all_to_lower_match",
    "meet_lower_first_match",
    "greatest_need_match",
    "all_to_lower_absolute_gap",
    "meet_lower_first_absolute_gap",
    "greatest_need_absolute_gap",
    "realized_outcome_1",
    "realized_outcome_2",
    "realized_outcome_gap",
)

SCARCITY_DENSE_CONVERGENCE_FIELDS = (
    "dense_grid_size",
    "dense_allocation",
    "dense_utility",
    "dense_utility_absolute_difference",
    "dense_allocation_absolute_difference",
    "dense_allocation_tie_within_1e-6",
)

SCARCITY_BINOMIAL_SUMMARY_SUFFIXES = (
    "count",
    "n",
    "rate",
    "ci95_low",
    "ci95_high",
    "one_sided_95_low",
    "one_sided_95_high",
    "ci95_half_width",
)

SCARCITY_PAIRED_CONTRAST_SUFFIXES = (
    "n",
    "mean",
    "standard_error",
    "ci95_low",
    "ci95_high",
    "simultaneous_one_sided_95_low",
    "simultaneous_one_sided_95_high",
    "two_sided_p_value",
)


class ScarcityError(RuntimeError):
    """Raised when a frozen scarcity invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScarcityError(message)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_seed(root_seed: int, *components: object) -> int:
    """Map a canonical seed namespace to a positive 63-bit integer."""

    payload = {"root_seed": int(root_seed), "components": list(components)}
    raw = int.from_bytes(hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:8], "big")
    value = raw & ((1 << 63) - 1)
    return value if value > 0 else 1


def scarcity_pairing_group_id(stage: str, anchor_id: str, episode_set_id: str) -> str:
    _require(stage != "" and anchor_id != "" and episode_set_id != "", "Pairing group fields must be nonblank")
    return canonical_json([stage, anchor_id, episode_set_id])


def scarcity_policy_seed(
    *,
    stage: str,
    environment_id: str,
    episode_index: int,
    policy_id: str,
    root_seed: int = SCARCITY_ROOT_SEED,
) -> int:
    return canonical_seed(
        root_seed,
        stage,
        environment_id,
        int(episode_index),
        policy_id,
        "policy_computation",
    )


@dataclass(frozen=True)
class ScarcityPairedEpisode:
    episode: EvaluationEpisode
    pairing_group_id: str
    true_state_seed_1: int
    true_state_seed_2: int
    innovation_seed_1: int
    innovation_seed_2: int
    true_state_fingerprint: str
    standardized_innovation_hash_1: str
    standardized_innovation_hash_2: str
    transformed_stream_hash_1: str
    transformed_stream_hash_2: str


def build_scarcity_paired_episode(
    config: EnvironmentConfig,
    *,
    stage: str,
    pairing_group_id: str,
    episode_index: int,
    root_seed: int = SCARCITY_ROOT_SEED,
    stream_capacity: int = SCARCITY_STREAM_CAPACITY_PER_RECIPIENT,
) -> ScarcityPairedEpisode:
    """Build paired hidden state and standardized innovations without truncation."""

    _require(episode_index >= 0, "Episode index must be nonnegative")
    _require(
        stream_capacity == SCARCITY_STREAM_CAPACITY_PER_RECIPIENT,
        "Scarcity stream capacity is frozen at 60 observations per recipient",
    )
    _require(
        max(config.prior_sample_count_1, config.prior_sample_count_2) <= 20,
        "Scarcity prior sample count exceeds 20",
    )
    _require(
        config.max_meta_samples is None or config.max_meta_samples <= 40,
        "Scarcity online sample count exceeds 40",
    )
    true_seeds = [
        canonical_seed(
            root_seed,
            stage,
            pairing_group_id,
            episode_index,
            recipient,
            "true_state",
        )
        for recipient in (1, 2)
    ]
    innovation_seeds = [
        canonical_seed(
            root_seed,
            stage,
            pairing_group_id,
            episode_index,
            recipient,
            "standardized_observation_innovation",
        )
        for recipient in (1, 2)
    ]
    needs = [
        float(random.Random(seed).gauss(config.mu_need, config.sigma_need))
        for seed in true_seeds
    ]
    innovations = []
    for seed in innovation_seeds:
        rng = random.Random(seed)
        innovations.append(
            [float(rng.gauss(0.0, 1.0)) for _ in range(stream_capacity)]
        )
    transformed = [
        [needs[index] + config.sigma_sample * value for value in innovations[index]]
        for index in (0, 1)
    ]
    true_state = TrueState(need_1=needs[0], need_2=needs[1])
    episode = EvaluationEpisode(
        episode_index=episode_index,
        true_state=true_state,
        observation_streams={
            ContinuousAllocationMetaMDP.SAMPLE_PERSON_1: transformed[0],
            ContinuousAllocationMetaMDP.SAMPLE_PERSON_2: transformed[1],
        },
    )
    return ScarcityPairedEpisode(
        episode=episode,
        pairing_group_id=pairing_group_id,
        true_state_seed_1=true_seeds[0],
        true_state_seed_2=true_seeds[1],
        innovation_seed_1=innovation_seeds[0],
        innovation_seed_2=innovation_seeds[1],
        true_state_fingerprint=canonical_hash(
            {"need_1": needs[0], "need_2": needs[1], "episode_index": episode_index}
        ),
        standardized_innovation_hash_1=canonical_hash(innovations[0]),
        standardized_innovation_hash_2=canonical_hash(innovations[1]),
        transformed_stream_hash_1=canonical_hash(transformed[0]),
        transformed_stream_hash_2=canonical_hash(transformed[1]),
    )


def gaussian_nonpositive_probabilities(mu_need: float, sigma_need: float) -> Tuple[float, float]:
    _require(sigma_need > 0.0, "sigma_need must be positive")
    standardized = mu_need / sigma_need
    individual = 0.5 * math.erfc(standardized / math.sqrt(2.0))
    if individual >= 1.0:
        either = 1.0
    else:
        either = -math.expm1(2.0 * math.log1p(-individual))
    return individual, either


def _learning_rates(config: EnvironmentConfig) -> Tuple[float, float]:
    return (
        config.learning_per_unit_of_tutoring,
        config.learning_per_unit_of_tutoring - config.delta_learning_per_unit_tutoring,
    )


def equal_outcome_allocation(
    need_1: float,
    need_2: float,
    rate_1: float,
    rate_2: float,
    remaining_time: float,
) -> float:
    if remaining_time <= 0.0:
        return 0.5
    denominator = (rate_1 + rate_2) * remaining_time
    _require(denominator > 0.0, "Equal-outcome allocation denominator is not positive")
    value = (need_1 - need_2 + rate_2 * remaining_time) / denominator
    return min(1.0, max(0.0, value))


def scarcity_allocation_metrics(
    config: EnvironmentConfig,
    true_state: TrueState,
    remaining_time: float,
    allocation_to_person1: float,
    *,
    classification_need_1: float | None = None,
    classification_need_2: float | None = None,
    classification_uses_hidden_true_state: bool | None = None,
    allocation_tolerance: float = SCARCITY_ALLOCATION_TOLERANCE,
) -> Dict[str, object]:
    """Evaluate a choice while keeping behavioral and outcome references distinct.

    Full-information rows classify the allocation against true need. Metalevel policy
    rows pass terminal posterior means as ``classification_need_*``. Positivity and
    realized outcomes always retain the hidden true state.
    """

    _require(0.0 <= allocation_to_person1 <= 1.0, "Allocation lies outside [0,1]")
    _require(remaining_time >= 0.0, "Remaining time is negative")
    _require(
        (classification_need_1 is None) == (classification_need_2 is None),
        "Classification need estimates must be supplied together",
    )
    classification_1 = (
        true_state.need_1 if classification_need_1 is None else float(classification_need_1)
    )
    classification_2 = (
        true_state.need_2 if classification_need_2 is None else float(classification_need_2)
    )
    uses_hidden_true_state = (
        classification_need_1 is None and classification_need_2 is None
        if classification_uses_hidden_true_state is None
        else bool(classification_uses_hidden_true_state)
    )
    _require(
        math.isfinite(classification_1) and math.isfinite(classification_2),
        "Classification need estimate is nonfinite",
    )
    rate_1, rate_2 = _learning_rates(config)
    effort_1 = effort_to_goal(classification_1, rate_1)
    effort_2 = effort_to_goal(classification_2, rate_2)
    lower_identity = lower_effort_identity(
        effort_1,
        effort_2,
        relative_tolerance=SCARCITY_EFFORT_TIE_RELATIVE_TOLERANCE,
    )
    raw_lower_identity = lower_effort_identity(
        max(0.0, classification_1),
        max(0.0, classification_2),
        relative_tolerance=SCARCITY_EFFORT_TIE_RELATIVE_TOLERANCE,
    )
    all_lower = all_to_lower_allocation(
        classification_1,
        classification_2,
        rate_1,
        rate_2,
    )
    meet_lower = meet_lower_first_allocation(
        classification_1,
        classification_2,
        rate_1,
        rate_2,
        remaining_time,
    )
    greatest = greatest_effort_need_allocation(
        classification_1,
        classification_2,
        rate_1,
        rate_2,
    )
    lower_share = (
        allocation_to_person1
        if lower_identity == 1
        else 1.0 - allocation_to_person1
        if lower_identity == 2
        else 0.5
    )
    effort_low = min(effort_1, effort_2)
    effort_high = max(effort_1, effort_2)
    effort_sum = effort_1 + effort_2
    overlap = lower_identity != 0 and remaining_time <= effort_low
    outcome_1 = rate_1 * allocation_to_person1 * remaining_time - true_state.need_1
    outcome_2 = rate_2 * (1.0 - allocation_to_person1) * remaining_time - true_state.need_2
    metrics = {
        "classification_need_1": classification_1,
        "classification_need_2": classification_2,
        "classification_uses_hidden_true_state": uses_hidden_true_state,
        "need_1_positive": true_state.need_1 > 0.0,
        "need_2_positive": true_state.need_2 > 0.0,
        "both_needs_positive": true_state.need_1 > 0.0 and true_state.need_2 > 0.0,
        "need_1_nonpositive": true_state.need_1 <= 0.0,
        "need_2_nonpositive": true_state.need_2 <= 0.0,
        "either_need_nonpositive": true_state.need_1 <= 0.0 or true_state.need_2 <= 0.0,
        "effort_to_goal_1": effort_1,
        "effort_to_goal_2": effort_2,
        "lower_effort_identity": lower_identity,
        "lower_raw_need_identity": raw_lower_identity,
        "effort_identity_differs_from_raw_need": (
            lower_identity != 0 and raw_lower_identity != 0 and lower_identity != raw_lower_identity
        ),
        "effort_tie": lower_identity == 0,
        "joint_goal_feasible": effort_sum <= remaining_time,
        "at_least_lower_goal_meetable": effort_low <= remaining_time,
        "exactly_one_goal_individually_meetable": effort_low <= remaining_time < effort_high,
        "both_individually_but_not_jointly_meetable": (
            effort_high <= remaining_time < effort_sum
        ),
        "neither_goal_meetable": remaining_time < effort_low,
        "all_to_lower_allocation": all_lower,
        "meet_lower_first_allocation": meet_lower,
        "greatest_need_allocation": greatest,
        "lower_pattern_overlap": overlap,
        "lower_recipient_allocation_share": lower_share,
        "more_to_lower": lower_identity != 0 and lower_share > SCARCITY_MORE_TO_LOWER_THRESHOLD,
        "approximately_equal_allocation": 0.45 <= allocation_to_person1 <= 0.55,
        "all_to_lower_match": abs(allocation_to_person1 - all_lower) <= allocation_tolerance,
        "meet_lower_first_match": abs(allocation_to_person1 - meet_lower) <= allocation_tolerance,
        "greatest_need_match": abs(allocation_to_person1 - greatest) <= allocation_tolerance,
        "all_to_lower_absolute_gap": abs(allocation_to_person1 - all_lower),
        "meet_lower_first_absolute_gap": abs(allocation_to_person1 - meet_lower),
        "greatest_need_absolute_gap": abs(allocation_to_person1 - greatest),
        "realized_outcome_1": outcome_1,
        "realized_outcome_2": outcome_2,
        "realized_outcome_gap": abs(outcome_1 - outcome_2),
    }
    _require(
        tuple(metrics) == SCARCITY_ALLOCATION_METRIC_FIELDS,
        "Scarcity allocation metric schema changed",
    )
    return metrics


def _oracle_candidate_allocations(
    config: EnvironmentConfig,
    true_state: TrueState,
    remaining_time: float,
) -> Dict[str, float]:
    rate_1, rate_2 = _learning_rates(config)
    return {
        "equal_split": 0.5,
        "equal_outcome": equal_outcome_allocation(
            true_state.need_1,
            true_state.need_2,
            rate_1,
            rate_2,
            remaining_time,
        ),
        "greatest_need": greatest_effort_need_allocation(
            true_state.need_1,
            true_state.need_2,
            rate_1,
            rate_2,
        ),
        "all_to_lower": all_to_lower_allocation(
            true_state.need_1,
            true_state.need_2,
            rate_1,
            rate_2,
        ),
        "meet_lower_first": meet_lower_first_allocation(
            true_state.need_1,
            true_state.need_2,
            rate_1,
            rate_2,
            remaining_time,
        ),
    }


def _piecewise_stationary_allocations(
    mdp: ContinuousAllocationMetaMDP,
    true_state: TrueState,
    remaining_time: float,
) -> Tuple[float, ...]:
    """Return exact stationary points from every differentiable sign stratum.

    The frozen utility is piecewise power-shaped. Within a fixed pair of
    outcome signs its first-order condition has a closed-form solution. These
    candidates complement the frozen grid and kink/boundary candidates without
    changing the objective, thresholds, or grid sizes.
    """

    alpha = mdp.utility_exponent()
    rate_1, rate_2 = mdp.learning_rates()
    amount_scale_1 = rate_1 * remaining_time
    amount_scale_2 = rate_2 * remaining_time
    if not (0.0 < alpha < 1.0 and amount_scale_1 > 0.0 and amount_scale_2 > 0.0):
        return ()
    lambda_shortfall = float(mdp.config.lambda_shortfall)
    if lambda_shortfall <= 0.0:
        return ()

    candidates: List[float] = []
    exponent = 1.0 / (1.0 - alpha)
    sign_tolerance = 1e-10
    for sign_1, sign_2 in product((1.0, -1.0), repeat=2):
        coefficient_1 = 1.0 if sign_1 > 0.0 else lambda_shortfall
        coefficient_2 = 1.0 if sign_2 > 0.0 else lambda_shortfall
        distance_ratio = (
            amount_scale_1 * coefficient_1
            / (amount_scale_2 * coefficient_2)
        ) ** exponent
        denominator = sign_1 * amount_scale_1 + distance_ratio * sign_2 * amount_scale_2
        if abs(denominator) <= 1e-15:
            continue
        allocation = (
            sign_1 * true_state.need_1
            + distance_ratio * sign_2 * (amount_scale_2 - true_state.need_2)
        ) / denominator
        if not (0.0 <= allocation <= 1.0 and math.isfinite(allocation)):
            continue
        outcome_1 = amount_scale_1 * allocation - true_state.need_1
        outcome_2 = amount_scale_2 * (1.0 - allocation) - true_state.need_2
        if sign_1 * outcome_1 < -sign_tolerance or sign_2 * outcome_2 < -sign_tolerance:
            continue
        candidates.append(float(allocation))
    return tuple(sorted(set(candidates)))


def _scarcity_full_information_oracle(
    mdp: ContinuousAllocationMetaMDP,
    true_state: TrueState,
    remaining_time: float,
    *,
    grid_size: int,
) -> Tuple[float, float]:
    """Evaluate the frozen grid/kinks plus exact piecewise stationary points."""

    grid_allocation, grid_utility = full_information_utilitarian_allocation(
        mdp,
        true_state,
        remaining_time,
        grid_size=grid_size,
    )
    allocations = sorted(
        set(
            (grid_allocation,)
            + _piecewise_stationary_allocations(
                mdp,
                true_state,
                remaining_time,
            )
        )
    )
    values = [
        deterministic_realized_utility(
            mdp,
            true_state,
            allocation,
            remaining_time,
        )
        for allocation in allocations
    ]
    best_value = max(grid_utility, max(values))
    tolerance = 1e-12 * max(1.0, abs(best_value))
    candidates = [
        index for index, value in enumerate(values) if value >= best_value - tolerance
    ]
    best_index = min(
        candidates,
        key=lambda index: (abs(allocations[index] - 0.5), allocations[index]),
    )
    return allocations[best_index], values[best_index]


def dense_oracle_convergence(
    mdp: ContinuousAllocationMetaMDP,
    true_state: TrueState,
    remaining_time: float,
    *,
    coarse_allocation: float,
    coarse_utility: float,
    dense_grid_size: int = SCARCITY_ORACLE_DENSE_GRID_SIZE,
) -> Dict[str, object]:
    """Apply the frozen dense-grid oracle gate and make allocation ties explicit."""

    dense_allocation, dense_utility = _scarcity_full_information_oracle(
        mdp,
        true_state,
        remaining_time,
        grid_size=dense_grid_size,
    )
    utility_difference = abs(dense_utility - coarse_utility)
    allocation_difference = abs(dense_allocation - coarse_allocation)
    coarse_at_dense_objective = deterministic_realized_utility(
        mdp,
        true_state,
        coarse_allocation,
        remaining_time,
    )
    dense_at_dense_objective = deterministic_realized_utility(
        mdp,
        true_state,
        dense_allocation,
        remaining_time,
    )
    utility_tie = abs(coarse_at_dense_objective - dense_at_dense_objective) <= 1e-6
    _require(utility_difference <= 1e-6, "Coarse and dense oracle utilities differ by more than 1e-6")
    _require(
        allocation_difference <= 0.0025 or utility_tie,
        "Coarse and dense oracle allocations differ without an explicit utility tie",
    )
    return {
        "dense_grid_size": dense_grid_size,
        "dense_allocation": dense_allocation,
        "dense_utility": dense_utility,
        "dense_utility_absolute_difference": utility_difference,
        "dense_allocation_absolute_difference": allocation_difference,
        "dense_allocation_tie_within_1e-6": utility_tie,
    }


def scarcity_oracle_comparison_row(
    *,
    environment_id: str,
    anchor_id: str,
    config: EnvironmentConfig,
    true_state: TrueState,
    episode_index: int,
    pairing_group_id: str,
    oracle_grid_size: int = SCARCITY_ORACLE_GRID_SIZE,
    dense_grid_size: int | None = None,
) -> Dict[str, object]:
    mdp = ContinuousAllocationMetaMDP(config)
    remaining_time = max(0.0, config.total_time - config.terminate_cost)
    oracle_allocation, oracle_utility = _scarcity_full_information_oracle(
        mdp,
        true_state,
        remaining_time,
        grid_size=oracle_grid_size,
    )
    candidate_allocations = _oracle_candidate_allocations(config, true_state, remaining_time)
    candidate_utilities = {
        policy: deterministic_realized_utility(
            mdp,
            true_state,
            allocation,
            remaining_time,
        )
        for policy, allocation in candidate_allocations.items()
    }
    tolerance = 1e-9
    _require(
        all(oracle_utility + tolerance >= value for value in candidate_utilities.values()),
        "Full-information oracle is below a feasible comparator",
    )
    metrics = scarcity_allocation_metrics(
        config,
        true_state,
        remaining_time,
        oracle_allocation,
    )
    row: Dict[str, object] = {
        "environment_id": environment_id,
        "anchor_id": anchor_id,
        "pairing_group_id": pairing_group_id,
        "episode_index": episode_index,
        "need_1": true_state.need_1,
        "need_2": true_state.need_2,
        "remaining_time": remaining_time,
        "oracle_grid_size": oracle_grid_size,
        "oracle_allocation": oracle_allocation,
        "oracle_utility": oracle_utility,
    }
    if dense_grid_size is not None:
        row.update(
            {
                f"oracle_{field}": value
                for field, value in dense_oracle_convergence(
                    mdp,
                    true_state,
                    remaining_time,
                    coarse_allocation=oracle_allocation,
                    coarse_utility=oracle_utility,
                    dense_grid_size=dense_grid_size,
                ).items()
            }
        )
    for policy, allocation in candidate_allocations.items():
        row[f"{policy}_allocation"] = allocation
        row[f"{policy}_utility"] = candidate_utilities[policy]
        row[f"oracle_regret_to_{policy}"] = max(
            0.0,
            oracle_utility - candidate_utilities[policy],
        )
    for field, value in metrics.items():
        row[f"oracle_{field}"] = value
    for comparator in ("equal_split", "greatest_need"):
        comparator_utility = candidate_utilities[comparator]
        row[f"gain_vs_{comparator}"] = oracle_utility - comparator_utility
        for policy in ("all_to_lower", "meet_lower_first"):
            row[f"retained_gain_{policy}_vs_{comparator}"] = (
                candidate_utilities[policy]
                - SCARCITY_RETAINED_GAIN_FRACTION * oracle_utility
                - (1.0 - SCARCITY_RETAINED_GAIN_FRACTION) * comparator_utility
            )
    return row


def build_deterministic_mechanism_cases() -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    for delta, capacity_ratio, exponent, lambda_shortfall in product(
        SCARCITY_DETERMINISTIC_DELTAS,
        SCARCITY_CAPACITY_RATIOS,
        SCARCITY_UTILITY_EXPONENTS,
        SCARCITY_LAMBDA_SHORTFALLS,
    ):
        need_low = 100.0 * (1.0 - delta)
        need_high = 100.0 * (1.0 + delta)
        orientations = (
            ((need_low, need_high), "lower_person1"),
            ((need_high, need_low), "lower_person2"),
        ) if delta > 0.0 else (((100.0, 100.0), "tie"),)
        for (need_1, need_2), orientation in orientations:
            remaining_time = capacity_ratio * 200.0
            config = EnvironmentConfig(
                mu_need=100.0,
                sigma_need=10.0,
                sigma_sample=10.0,
                total_time=remaining_time + 1.0,
                terminate_cost=1.0,
                sample_time_cost=1.0,
                utility_exponent=exponent,
                lambda_shortfall=lambda_shortfall,
                learning_per_unit_of_tutoring=1.0,
                delta_learning_per_unit_tutoring=0.0,
                allocation_grid_size=401,
                expected_utility_draws=SCARCITY_VOI_DRAWS,
            )
            identity = {
                "delta": delta,
                "capacity_ratio": capacity_ratio,
                "utility_exponent": exponent,
                "lambda_shortfall": lambda_shortfall,
                "orientation": orientation,
                "learning_rate_1": 1.0,
                "learning_rate_2": 1.0,
            }
            environment_id = "deterministic_" + canonical_hash(identity)[:16]
            cases.append(
                {
                    **identity,
                    "environment_id": environment_id,
                    "anchor_id": "mechanism_" + canonical_hash(identity)[:16],
                    "config": config,
                    "true_state": TrueState(need_1=need_1, need_2=need_2),
                    "analysis_role": "primary_equal_rate_mechanism_map",
                }
            )
    robustness = (
        (60.0, 140.0, 1.25, 1.0, "lower_person1"),
        (140.0, 60.0, 1.0, 1.25, "lower_person2"),
    )
    for need_1, need_2, rate_1, rate_2, orientation in robustness:
        identity = {
            "delta": 0.4,
            "capacity_ratio": 0.5,
            "utility_exponent": 0.5,
            "lambda_shortfall": 2.0,
            "orientation": orientation,
            "learning_rate_1": rate_1,
            "learning_rate_2": rate_2,
        }
        config = EnvironmentConfig(
            mu_need=100.0,
            sigma_need=10.0,
            sigma_sample=10.0,
            total_time=101.0,
            terminate_cost=1.0,
            sample_time_cost=1.0,
            utility_exponent=0.5,
            lambda_shortfall=2.0,
            learning_per_unit_of_tutoring=rate_1,
            delta_learning_per_unit_tutoring=rate_1 - rate_2,
            allocation_grid_size=401,
            expected_utility_draws=SCARCITY_VOI_DRAWS,
        )
        environment_id = "deterministic_robustness_" + canonical_hash(identity)[:16]
        cases.append(
            {
                **identity,
                "environment_id": environment_id,
                "anchor_id": "mechanism_robustness_" + canonical_hash(identity)[:16],
                "config": config,
                "true_state": TrueState(need_1=need_1, need_2=need_2),
                "analysis_role": "secondary_label_reversal_rate_robustness",
            }
        )
    _require(len(cases) == 488, "Deterministic scarcity mechanism map must contain 488 cases")
    for case_index, case in enumerate(cases):
        case["mechanism_case_index"] = case_index
    return cases


def build_gaussian_oracle_descriptors() -> List[Dict[str, object]]:
    descriptors: List[Dict[str, object]] = []
    for sigma_need, capacity_ratio, exponent, lambda_shortfall in product(
        SCARCITY_SIGMA_NEEDS,
        SCARCITY_GAUSSIAN_CAPACITY_RATIOS,
        SCARCITY_UTILITY_EXPONENTS,
        SCARCITY_LAMBDA_SHORTFALLS,
    ):
        total_time = 2.0 * 100.0 * capacity_ratio + 1.0
        identity = {
            "mu_need": 100.0,
            "sigma_need": sigma_need,
            "capacity_ratio": capacity_ratio,
            "total_time": total_time,
            "utility_exponent": exponent,
            "lambda_shortfall": lambda_shortfall,
            "learning_rate_1": 1.0,
            "learning_rate_2": 1.0,
        }
        anchor_id = "gaussian_anchor_" + canonical_hash(identity)[:20]
        environment_id = (
            f"gaussian_oracle_sigma_need={sigma_need:g}_capacity_ratio={capacity_ratio:g}_"
            f"utility_exponent={exponent:g}_lambda_shortfall={lambda_shortfall:g}"
        )
        config = EnvironmentConfig(
            mu_need=100.0,
            sigma_need=sigma_need,
            sigma_sample=10.0,
            total_time=total_time,
            terminate_cost=1.0,
            sample_time_cost=1.0,
            utility_exponent=exponent,
            lambda_shortfall=lambda_shortfall,
            learning_per_unit_of_tutoring=1.0,
            delta_learning_per_unit_tutoring=0.0,
            prior_sample_count_1=0,
            prior_sample_count_2=0,
            allocation_grid_size=401,
            expected_utility_draws=SCARCITY_VOI_DRAWS,
        )
        pairing_group = scarcity_pairing_group_id(
            "oracle_development",
            f"gaussian_sigma_need={sigma_need:g}",
            "screen_120",
        )
        individual_probability, either_probability = gaussian_nonpositive_probabilities(
            100.0,
            sigma_need,
        )
        descriptors.append(
            {
                **identity,
                "anchor_id": anchor_id,
                "environment_id": environment_id,
                "pairing_group_id": pairing_group,
                "config": config,
                "theoretical_individual_nonpositive_probability": individual_probability,
                "theoretical_either_nonpositive_probability": either_probability,
            }
        )
    _require(len(descriptors) == 135, "Gaussian scarcity oracle screen must contain 135 configurations")
    _require(
        {float(item["total_time"]) for item in descriptors} == {51.0, 101.0, 151.0, 191.0, 211.0},
        "Gaussian scarcity total-time grid changed",
    )
    return descriptors


def evaluate_deterministic_mechanism_cases(
    cases: Sequence[Mapping[str, object]],
    *,
    oracle_grid_size: int = SCARCITY_ORACLE_GRID_SIZE,
    dense_grid_size: int | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, case in enumerate(cases):
        config = case["config"]
        true_state = case["true_state"]
        _require(isinstance(config, EnvironmentConfig), "Mechanism case config is malformed")
        _require(isinstance(true_state, TrueState), "Mechanism case true state is malformed")
        row = scarcity_oracle_comparison_row(
            environment_id=str(case["environment_id"]),
            anchor_id=str(case["anchor_id"]),
            config=config,
            true_state=true_state,
            episode_index=int(case.get("mechanism_case_index", index)),
            pairing_group_id="deterministic_mechanism_map",
            oracle_grid_size=oracle_grid_size,
            dense_grid_size=dense_grid_size,
        )
        for field, value in case.items():
            if field not in {"config", "true_state"}:
                row[field] = value
        rows.append(row)
    return rows


def evaluate_gaussian_oracle_descriptor(
    descriptor: Mapping[str, object],
    *,
    n_episodes: int = SCARCITY_DEVELOPMENT_EPISODES,
    episode_start: int = 0,
    oracle_grid_size: int = SCARCITY_ORACLE_GRID_SIZE,
    dense_grid_size: int | None = None,
    root_seed: int = SCARCITY_ROOT_SEED,
) -> List[Dict[str, object]]:
    config = descriptor.get("config")
    _require(isinstance(config, EnvironmentConfig), "Gaussian oracle descriptor config is malformed")
    environment_id = str(descriptor.get("environment_id", ""))
    anchor_id = str(descriptor.get("anchor_id", ""))
    pairing_group = str(descriptor.get("pairing_group_id", ""))
    _require(environment_id and anchor_id and pairing_group, "Gaussian oracle descriptor identity is incomplete")
    rows: List[Dict[str, object]] = []
    for episode_index in range(episode_start, episode_start + n_episodes):
        paired = build_scarcity_paired_episode(
            config,
            stage="oracle_development",
            pairing_group_id=pairing_group,
            episode_index=episode_index,
            root_seed=root_seed,
        )
        row = scarcity_oracle_comparison_row(
            environment_id=environment_id,
            anchor_id=anchor_id,
            config=config,
            true_state=paired.episode.true_state,
            episode_index=episode_index,
            pairing_group_id=pairing_group,
            oracle_grid_size=oracle_grid_size,
            dense_grid_size=dense_grid_size,
        )
        row.update(
            {
                "true_state_fingerprint": paired.true_state_fingerprint,
                "standardized_innovation_hash_1": paired.standardized_innovation_hash_1,
                "standardized_innovation_hash_2": paired.standardized_innovation_hash_2,
                "transformed_stream_hash_1": paired.transformed_stream_hash_1,
                "transformed_stream_hash_2": paired.transformed_stream_hash_2,
                "theoretical_individual_nonpositive_probability": descriptor[
                    "theoretical_individual_nonpositive_probability"
                ],
                "theoretical_either_nonpositive_probability": descriptor[
                    "theoretical_either_nonpositive_probability"
                ],
                "mu_need": descriptor["mu_need"],
                "sigma_need": descriptor["sigma_need"],
                "capacity_ratio": descriptor["capacity_ratio"],
                "total_time": descriptor["total_time"],
                "utility_exponent": descriptor["utility_exponent"],
                "lambda_shortfall": descriptor["lambda_shortfall"],
                "stage": "oracle_development",
            }
        )
        rows.append(row)
    return rows


def binomial_rate_summary(values: Sequence[object], prefix: str) -> Dict[str, object]:
    binary = [bool(value) for value in values]
    trials = len(binary)
    successes = sum(binary)
    result: Dict[str, object] = {
        f"{prefix}_count": successes,
        f"{prefix}_n": trials,
    }
    if trials == 0:
        result.update(
            {
                f"{prefix}_rate": "",
                f"{prefix}_ci95_low": "",
                f"{prefix}_ci95_high": "",
                f"{prefix}_one_sided_95_low": "",
                f"{prefix}_one_sided_95_high": "",
                f"{prefix}_ci95_half_width": "",
            }
        )
        return result
    rate = successes / trials
    low, high = wilson_interval(successes, trials, confidence=0.95, one_sided=False)
    one_low, one_high = wilson_interval(successes, trials, confidence=0.95, one_sided=True)
    result.update(
        {
            f"{prefix}_rate": rate,
            f"{prefix}_ci95_low": low,
            f"{prefix}_ci95_high": high,
            f"{prefix}_one_sided_95_low": one_low,
            f"{prefix}_one_sided_95_high": one_high,
            f"{prefix}_ci95_half_width": (high - low) / 2.0,
        }
    )
    return result


def continuous_gap_summary(values: Sequence[float], prefix: str) -> Dict[str, object]:
    """Summarize nonnegative allocation/outcome distances without hiding emptiness."""

    finite = [float(value) for value in values]
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in finite),
        f"Invalid continuous gap in {prefix}",
    )
    if not finite:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean_absolute_gap": "",
            f"{prefix}_rmse": "",
        }
    return {
        f"{prefix}_n": len(finite),
        f"{prefix}_mean_absolute_gap": float(statistics.mean(finite)),
        f"{prefix}_rmse": math.sqrt(float(statistics.mean(value * value for value in finite))),
    }


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    _require(degrees_of_freedom >= 1, "Student-t degrees of freedom must be positive")
    try:
        from scipy.stats import t as student_t  # type: ignore
    except ImportError as error:
        raise ScarcityError("SciPy is required for the frozen Student-t inference") from error
    value = float(student_t.ppf(probability, degrees_of_freedom))
    _require(math.isfinite(value), "Student-t quantile is nonfinite")
    return value


def _student_t_two_sided_p_value(mean: float, standard_error: float, df: int) -> float:
    if standard_error == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    try:
        from scipy.stats import t as student_t  # type: ignore
    except ImportError as error:
        raise ScarcityError("SciPy is required for the frozen Student-t inference") from error
    return float(2.0 * student_t.sf(abs(mean / standard_error), df))


def paired_contrast_summary(values: Sequence[float], prefix: str) -> Dict[str, object]:
    finite = [float(value) for value in values]
    _require(all(math.isfinite(value) for value in finite), f"Nonfinite contrast in {prefix}")
    n = len(finite)
    result: Dict[str, object] = {f"{prefix}_n": n}
    if n == 0:
        result.update(
            {
                f"{prefix}_mean": "",
                f"{prefix}_standard_error": "",
                f"{prefix}_ci95_low": "",
                f"{prefix}_ci95_high": "",
                f"{prefix}_simultaneous_one_sided_95_low": "",
                f"{prefix}_simultaneous_one_sided_95_high": "",
                f"{prefix}_two_sided_p_value": "",
            }
        )
        return result
    mean = float(statistics.mean(finite))
    result[f"{prefix}_mean"] = mean
    if n == 1:
        result.update(
            {
                f"{prefix}_standard_error": "",
                f"{prefix}_ci95_low": "",
                f"{prefix}_ci95_high": "",
                f"{prefix}_simultaneous_one_sided_95_low": "",
                f"{prefix}_simultaneous_one_sided_95_high": "",
                f"{prefix}_two_sided_p_value": "",
            }
        )
        return result
    standard_error = float(statistics.stdev(finite) / math.sqrt(n))
    critical = _student_t_quantile(0.975, n - 1)
    radius = critical * standard_error
    result.update(
        {
            f"{prefix}_standard_error": standard_error,
            f"{prefix}_ci95_low": mean - radius,
            f"{prefix}_ci95_high": mean + radius,
            f"{prefix}_simultaneous_one_sided_95_low": mean - radius,
            f"{prefix}_simultaneous_one_sided_95_high": mean + radius,
            f"{prefix}_two_sided_p_value": _student_t_two_sided_p_value(
                mean,
                standard_error,
                n - 1,
            ),
        }
    )
    return result


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize_gaussian_oracle_environment(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    _require(bool(rows), "Cannot summarize an empty Gaussian oracle environment")
    environment_id = str(rows[0]["environment_id"])
    _require(
        all(str(row["environment_id"]) == environment_id for row in rows),
        "Gaussian oracle summary mixes environments",
    )
    positive_scarcity = [
        row
        for row in rows
        if bool(row["oracle_both_needs_positive"])
        and not bool(row["oracle_joint_goal_feasible"])
    ]
    relevant = [
        row
        for row in positive_scarcity
        if not bool(row["oracle_effort_tie"])
    ]
    nonoverlap = [row for row in relevant if not bool(row["oracle_lower_pattern_overlap"])]
    first = rows[0]
    summary: Dict[str, object] = {
        "environment_id": environment_id,
        "anchor_id": first["anchor_id"],
        "pairing_group_id": first["pairing_group_id"],
        "n_episodes": len(rows),
        "both_positive_jointly_infeasible_n": len(positive_scarcity),
        "relevant_both_positive_jointly_infeasible_nontie_n": len(relevant),
        "relevant_nonoverlap_n": len(nonoverlap),
        "effort_tie_count_in_both_positive_jointly_infeasible": (
            len(positive_scarcity) - len(relevant)
        ),
        "effort_tie_rate_in_both_positive_jointly_infeasible": (
            (len(positive_scarcity) - len(relevant)) / len(positive_scarcity)
            if positive_scarcity
            else ""
        ),
        "mu_need": first["mu_need"],
        "sigma_need": first["sigma_need"],
        "capacity_ratio": first["capacity_ratio"],
        "total_time": first["total_time"],
        "utility_exponent": first["utility_exponent"],
        "lambda_shortfall": first["lambda_shortfall"],
        "theoretical_individual_nonpositive_probability": first[
            "theoretical_individual_nonpositive_probability"
        ],
        "theoretical_either_nonpositive_probability": first[
            "theoretical_either_nonpositive_probability"
        ],
        "observed_individual_nonpositive_count": sum(
            int(bool(row["oracle_need_1_nonpositive"]))
            + int(bool(row["oracle_need_2_nonpositive"]))
            for row in rows
        ),
        "observed_individual_draw_count": 2 * len(rows),
        "observed_either_nonpositive_count": sum(
            bool(row["oracle_either_need_nonpositive"]) for row in rows
        ),
        "observed_both_positive_count": sum(
            bool(row["oracle_both_needs_positive"]) for row in rows
        ),
    }
    summary["observed_individual_nonpositive_rate"] = (
        int(summary["observed_individual_nonpositive_count"])
        / int(summary["observed_individual_draw_count"])
    )
    summary["observed_either_nonpositive_rate"] = (
        int(summary["observed_either_nonpositive_count"]) / len(rows)
    )
    summary["observed_both_positive_rate"] = (
        int(summary["observed_both_positive_count"]) / len(rows)
    )
    for field in (
        "joint_goal_feasible",
        "at_least_lower_goal_meetable",
        "exactly_one_goal_individually_meetable",
        "both_individually_but_not_jointly_meetable",
        "neither_goal_meetable",
    ):
        count = sum(bool(row[f"oracle_{field}"]) for row in rows)
        summary[f"{field}_count_full_sample"] = count
        summary[f"{field}_rate_full_sample"] = count / len(rows)
    for field in ("all_to_lower_match", "meet_lower_first_match", "more_to_lower"):
        summary.update(
            binomial_rate_summary(
                [row[f"oracle_{field}"] for row in relevant],
                f"oracle_{field}",
            )
        )
    for field in ("all_to_lower_match", "meet_lower_first_match"):
        summary.update(
            binomial_rate_summary(
                [row[f"oracle_{field}"] for row in nonoverlap],
                f"oracle_{field}_nonoverlap",
            )
        )
    for rule in ("all_to_lower", "meet_lower_first", "greatest_need"):
        summary.update(
            continuous_gap_summary(
                [float(row[f"oracle_{rule}_absolute_gap"]) for row in relevant],
                f"oracle_{rule}_allocation_gap",
            )
        )
    summary.update(
        continuous_gap_summary(
            [float(row["oracle_realized_outcome_gap"]) for row in relevant],
            "oracle_realized_outcome_gap",
        )
    )
    for comparator in ("equal_split", "greatest_need"):
        summary.update(
            paired_contrast_summary(
                [float(row[f"gain_vs_{comparator}"]) for row in relevant],
                f"gain_vs_{comparator}",
            )
        )
        for policy in ("all_to_lower", "meet_lower_first"):
            summary.update(
                paired_contrast_summary(
                    [
                        float(row[f"retained_gain_{policy}_vs_{comparator}"])
                        for row in relevant
                    ],
                    f"retained_gain_{policy}_vs_{comparator}",
                )
            )

    gain_lowers = [
        _numeric(summary[f"gain_vs_{comparator}_simultaneous_one_sided_95_low"])
        for comparator in ("equal_split", "greatest_need")
    ]
    gain_pass = all(value is not None and value > 0.0 for value in gain_lowers)
    summary["oracle_positive_gain_over_both_comparators"] = gain_pass
    direction_low = _numeric(summary["oracle_more_to_lower_one_sided_95_low"])
    direction_high = _numeric(summary["oracle_more_to_lower_one_sided_95_high"])
    direction_supported = (
        direction_low is not None
        and direction_low >= SCARCITY_SUPPORT_THRESHOLD
        and gain_pass
    )
    summary["direction_supported"] = direction_supported
    exact_label_eligible: Dict[str, bool] = {}
    for policy in ("all_to_lower", "meet_lower_first"):
        match_low = _numeric(summary[f"oracle_{policy}_match_one_sided_95_low"])
        nonoverlap_low = _numeric(
            summary[f"oracle_{policy}_match_nonoverlap_one_sided_95_low"]
        )
        retained_lowers = [
            _numeric(
                summary[
                    f"retained_gain_{policy}_vs_{comparator}_simultaneous_one_sided_95_low"
                ]
            )
            for comparator in ("equal_split", "greatest_need")
        ]
        retained_pass = all(
            value is not None and value >= 0.0 for value in retained_lowers
        )
        supported = (
            match_low is not None
            and match_low >= SCARCITY_SUPPORT_THRESHOLD
            and gain_pass
            and retained_pass
        )
        label_eligible = (
            supported
            and nonoverlap_low is not None
            and nonoverlap_low >= SCARCITY_SUPPORT_THRESHOLD
        )
        summary[f"{policy}_retained_gain_pass"] = retained_pass
        summary[f"{policy}_supported"] = supported
        summary[f"{policy}_exact_label_eligible"] = label_eligible
        exact_label_eligible[policy] = label_eligible
    if any(exact_label_eligible.values()):
        classification = "supported_exact"
    elif direction_supported:
        classification = "partial_direction_only"
    elif direction_high is not None and direction_high < SCARCITY_SUPPORT_THRESHOLD:
        classification = "not_supported_in_frozen_scope"
    else:
        classification = "inconclusive_precision"
    summary["object_level_classification"] = classification
    return summary


def summarize_gaussian_oracle_rows(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    by_environment: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        by_environment.setdefault(str(row["environment_id"]), []).append(row)
    return [
        summarize_gaussian_oracle_environment(by_environment[environment])
        for environment in sorted(by_environment)
    ]


def _normalized_medoid(
    candidates: Sequence[Mapping[str, object]],
    *,
    fields: Sequence[str],
    levels: Mapping[str, Sequence[float]],
) -> Dict[str, object]:
    _require(bool(candidates), "Cannot select an anchor medoid from an empty set")
    means: Dict[str, float] = {}
    for left in candidates:
        distances: List[float] = []
        for right in candidates:
            distance = 0.0
            for field in fields:
                ordered = tuple(float(value) for value in levels[field])
                if len(ordered) == 1:
                    continue
                left_index = ordered.index(float(left[field]))
                right_index = ordered.index(float(right[field]))
                distance += abs(left_index - right_index) / (len(ordered) - 1)
            distances.append(distance)
        means[str(left["environment_id"])] = sum(distances) / len(distances)
    minimum = min(means.values())
    tied = sorted(
        environment
        for environment, value in means.items()
        if math.isclose(value, minimum, rel_tol=0.0, abs_tol=1e-12)
    )
    selected_environment = tied[0]
    selected = next(
        dict(row) for row in candidates if row["environment_id"] == selected_environment
    )
    selected["medoid_mean_normalized_manhattan_distance"] = means[selected_environment]
    selected["medoid_tie_count"] = len(tied)
    selected["medoid_tie_break"] = "lexicographically_smallest_environment_id"
    return selected


def _capacity_band(capacity_ratio: float) -> str | None:
    if capacity_ratio <= 0.50:
        return "severe"
    if 0.75 <= capacity_ratio <= 0.95:
        return "near_feasible"
    return None


def select_gaussian_oracle_anchors(
    summaries: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Select at most four exact and two direction-only observed anchors."""

    selected: List[Dict[str, object]] = []
    for band in ("severe", "near_feasible"):
        band_rows = [
            row
            for row in summaries
            if _capacity_band(float(row["capacity_ratio"])) == band
        ]
        exact_candidates_by_rule = {
            rule: [row for row in band_rows if bool(row[f"{rule}_exact_label_eligible"])]
            for rule in ("all_to_lower", "meet_lower_first")
        }
        has_any_exact = any(exact_candidates_by_rule.values())
        for rule in ("all_to_lower", "meet_lower_first"):
            candidates = exact_candidates_by_rule[rule]
            if not candidates:
                continue
            medoid = _normalized_medoid(
                candidates,
                fields=tuple(SCARCITY_ORACLE_MEDOID_LEVELS),
                levels=SCARCITY_ORACLE_MEDOID_LEVELS,
            )
            source_anchor = str(medoid["anchor_id"])
            medoid.update(
                {
                    "source_anchor_id": source_anchor,
                    "anchor_id": (
                        "selected_exact_"
                        + canonical_hash([source_anchor, band, rule])[:20]
                    ),
                    "anchor_band": band,
                    "anchor_rule": rule,
                    "anchor_support_kind": "exact",
                }
            )
            selected.append(medoid)
        if not has_any_exact:
            direction_candidates = [row for row in band_rows if bool(row["direction_supported"])]
            if direction_candidates:
                medoid = _normalized_medoid(
                    direction_candidates,
                    fields=tuple(SCARCITY_ORACLE_MEDOID_LEVELS),
                    levels=SCARCITY_ORACLE_MEDOID_LEVELS,
                )
                source_anchor = str(medoid["anchor_id"])
                medoid.update(
                    {
                        "source_anchor_id": source_anchor,
                        "anchor_id": (
                            "selected_direction_"
                            + canonical_hash([source_anchor, band, "more_to_lower"])[:20]
                        ),
                        "anchor_band": band,
                        "anchor_rule": "more_to_lower",
                        "anchor_support_kind": "direction_only",
                    }
                )
                selected.append(medoid)
    _require(len(selected) <= 6, "Oracle anchor selection exceeded the six-anchor cap")
    _require(
        len({str(row["anchor_id"]) for row in selected}) == len(selected),
        "Selected oracle anchor identities are not unique",
    )
    return selected


def object_level_stop_decision(
    summaries: Sequence[Mapping[str, object]],
    selected_anchors: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if selected_anchors:
        exact_count = sum(
            str(anchor.get("anchor_support_kind")) == "exact"
            for anchor in selected_anchors
        )
        direction_count = sum(
            str(anchor.get("anchor_support_kind")) == "direction_only"
            for anchor in selected_anchors
        )
        _require(
            exact_count + direction_count == len(selected_anchors),
            "Selected object anchors contain an unknown support kind",
        )
        return {
            "stop_metalevel": False,
            "object_level_classification": (
                "supported_exact" if exact_count > 0 else "partial_direction_only"
            ),
            "selected_anchor_count": len(selected_anchors),
            "selected_exact_anchor_count": exact_count,
            "selected_direction_only_anchor_count": direction_count,
            "reason": (
                "At least one frozen object-level exact or direction anchor passed; "
                "metalevel execution is eligible."
            ),
        }
    relevant = [
        row
        for row in summaries
        if _capacity_band(float(row["capacity_ratio"])) is not None
    ]
    _require(bool(relevant), "Object-level stop audit has no relevant scarcity configurations")
    uppers = [
        _numeric(row.get("oracle_more_to_lower_one_sided_95_high")) for row in relevant
    ]
    all_rejected = all(
        value is not None and value < SCARCITY_SUPPORT_THRESHOLD for value in uppers
    )
    classification = (
        "not_supported_in_frozen_scope"
        if all_rejected
        else "inconclusive_precision"
    )
    return {
        "stop_metalevel": True,
        "object_level_classification": classification,
        "selected_anchor_count": 0,
        "relevant_configuration_count": len(relevant),
        "all_more_to_lower_upper_bounds_below_0.80": all_rejected,
        "reason": (
            "No eligible object-level anchor; metalevel work stops without converting "
            "imprecision into rejection."
        ),
    }


def build_development_descriptors(
    selected_anchors: Sequence[Mapping[str, object]],
    oracle_descriptors: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    source_by_anchor = {
        str(descriptor["anchor_id"]): descriptor for descriptor in oracle_descriptors
    }
    environments: List[Dict[str, object]] = []
    for selected in selected_anchors:
        source_anchor_id = str(selected["source_anchor_id"])
        _require(source_anchor_id in source_by_anchor, "Selected anchor source descriptor is absent")
        source = source_by_anchor[source_anchor_id]
        base_config = source["config"]
        _require(isinstance(base_config, EnvironmentConfig), "Selected anchor config is malformed")
        selected_anchor_id = str(selected["anchor_id"])
        pairing_group = scarcity_pairing_group_id(
            "development",
            selected_anchor_id,
            "development_120",
        )
        for sigma_sample, cost_percent, prior_count in product(
            SCARCITY_SIGMA_SAMPLES,
            SCARCITY_SAMPLE_TIME_COST_PERCENTS,
            SCARCITY_PRIOR_SAMPLE_COUNTS,
        ):
            config = replace(
                base_config,
                sigma_sample=sigma_sample,
                sample_time_cost=base_config.total_time * cost_percent / 100.0,
                prior_sample_count_1=prior_count,
                prior_sample_count_2=prior_count,
                max_meta_samples=40,
                expected_utility_draws=SCARCITY_VOI_DRAWS,
                allocation_grid_size=401,
                random_seed=None,
            )
            environment_id = (
                f"development_{selected_anchor_id}_sigma_sample={sigma_sample:g}_"
                f"sample_time_cost_percent={cost_percent:g}_prior_sample_count={prior_count}"
            )
            environments.append(
                {
                    "environment_id": environment_id,
                    "anchor_id": selected_anchor_id,
                    "source_anchor_id": source_anchor_id,
                    "anchor_rule": selected["anchor_rule"],
                    "anchor_support_kind": selected["anchor_support_kind"],
                    "anchor_band": selected["anchor_band"],
                    "pairing_group_id": pairing_group,
                    "episode_set_id": "development_120",
                    "sigma_sample": sigma_sample,
                    "sample_time_cost_percent": cost_percent,
                    "prior_sample_count": prior_count,
                    "mu_need": source["mu_need"],
                    "sigma_need": source["sigma_need"],
                    "capacity_ratio": source["capacity_ratio"],
                    "total_time": source["total_time"],
                    "utility_exponent": source["utility_exponent"],
                    "lambda_shortfall": source["lambda_shortfall"],
                    "config": config,
                }
            )
    _require(
        len(environments) == 27 * len(selected_anchors),
        "Development environment Cartesian product is incomplete",
    )
    _require(
        len({str(item["environment_id"]) for item in environments}) == len(environments),
        "Development environment identities are not unique",
    )
    return environments


SCARCITY_POLICY_ORDER = (
    "frozen_rr",
    "immediate_all_to_lower",
    "immediate_meet_lower_first",
    "manual_active_search_all_to_lower",
    "manual_active_search_meet_lower_first",
    "equal_split",
    "greatest_need",
    "initial_full_information_oracle",
)


def _observation_stream_copy(episode: EvaluationEpisode) -> Mapping[str, Sequence[float]]:
    streams = episode.observation_streams
    _require(streams is not None, "Scarcity metalevel episode lacks observation streams")
    return {action: list(values) for action, values in streams.items()}


def _sample_counts(samples: Sequence[Mapping[str, float]]) -> Tuple[int, int]:
    count_1 = sum(float(item.get("action", -1.0)) == 1.0 for item in samples)
    count_2 = sum(float(item.get("action", -1.0)) == 2.0 for item in samples)
    return count_1, count_2


def _metalevel_policy_objects() -> Mapping[str, object]:
    return {
        "frozen_rr": MyopicValueOfInformationPolicy(observation_draws=SCARCITY_VOI_DRAWS),
        "immediate_all_to_lower": ImmediateAllToLowerPolicy(),
        "immediate_meet_lower_first": ImmediateMeetLowerFirstPolicy(),
        "manual_active_search_all_to_lower": ManualActiveSearchAllToLowerPolicy(),
        "manual_active_search_meet_lower_first": ManualActiveSearchMeetLowerFirstPolicy(),
        "equal_split": EqualSplitBaselinePolicy(),
        "greatest_need": ScarcityGreatestNeedPolicy(),
    }


def evaluate_metalevel_episode(
    descriptor: Mapping[str, object],
    *,
    stage: str,
    episode_index: int,
    root_seed: int = SCARCITY_ROOT_SEED,
    oracle_grid_size: int = SCARCITY_ORACLE_GRID_SIZE,
    dense_grid_size: int | None = None,
) -> List[Dict[str, object]]:
    config = descriptor.get("config")
    _require(isinstance(config, EnvironmentConfig), "Metalevel descriptor config is malformed")
    environment_id = str(descriptor.get("environment_id", ""))
    anchor_id = str(descriptor.get("anchor_id", ""))
    pairing_group = str(descriptor.get("pairing_group_id", ""))
    _require(environment_id and anchor_id and pairing_group, "Metalevel descriptor identity is incomplete")
    paired = build_scarcity_paired_episode(
        config,
        stage=stage,
        pairing_group_id=pairing_group,
        episode_index=episode_index,
        root_seed=root_seed,
    )
    true_state = paired.episode.true_state
    base_mdp = ContinuousAllocationMetaMDP(config)
    oracle_cache: Dict[float, Tuple[float, float, Dict[str, object]]] = {}

    def oracle_at(remaining: float) -> Tuple[float, float, Dict[str, object]]:
        cache_key = float(remaining)
        if cache_key not in oracle_cache:
            allocation_value, utility_value = _scarcity_full_information_oracle(
                base_mdp,
                true_state,
                remaining,
                grid_size=oracle_grid_size,
            )
            convergence = (
                dense_oracle_convergence(
                    base_mdp,
                    true_state,
                    remaining,
                    coarse_allocation=allocation_value,
                    coarse_utility=utility_value,
                    dense_grid_size=dense_grid_size,
                )
                if dense_grid_size is not None
                else {}
            )
            oracle_cache[cache_key] = (allocation_value, utility_value, convergence)
        return oracle_cache[cache_key]

    initial_remaining = max(0.0, config.total_time - config.terminate_cost)
    initial_oracle_allocation, initial_oracle_utility, initial_convergence = oracle_at(
        initial_remaining
    )
    initial_metrics = scarcity_allocation_metrics(
        config,
        true_state,
        initial_remaining,
        initial_oracle_allocation,
    )
    policies = _metalevel_policy_objects()
    rows: List[Dict[str, object]] = []
    for policy_id in SCARCITY_POLICY_ORDER:
        policy_seed = scarcity_policy_seed(
            stage=stage,
            environment_id=environment_id,
            episode_index=episode_index,
            policy_id=policy_id,
            root_seed=root_seed,
        )
        if policy_id == "initial_full_information_oracle":
            allocation = initial_oracle_allocation
            realized_utility = initial_oracle_utility
            remaining_time = initial_remaining
            count_1 = 0
            count_2 = 0
            classification_need_1 = true_state.need_1
            classification_need_2 = true_state.need_2
            terminal_variance_1 = 0.0
            terminal_variance_2 = 0.0
            classification_basis = "hidden_true_need_full_information"
        else:
            policy = policies[policy_id]
            episode_config = replace(config, random_seed=policy_seed)
            mdp = ContinuousAllocationMetaMDP(
                episode_config,
                observation_streams=_observation_stream_copy(paired.episode),
            )
            result = mdp.run_episode(
                policy,  # type: ignore[arg-type]
                true_state=true_state,
                max_steps=42,
            )
            allocation = result.final_allocation_to_person1
            realized_utility = result.realized_utility
            remaining_time = result.remaining_time
            count_1, count_2 = _sample_counts(result.samples)
            classification_need_1 = result.final_belief.mean_1
            classification_need_2 = result.final_belief.mean_2
            terminal_variance_1 = result.final_belief.var_1
            terminal_variance_2 = result.final_belief.var_2
            classification_basis = "terminal_posterior_mean"
            _require(count_1 + count_2 <= 40, "Metalevel policy exceeded 40 online samples")
            _require(
                max(config.prior_sample_count_1 + count_1, config.prior_sample_count_2 + count_2)
                <= SCARCITY_STREAM_CAPACITY_PER_RECIPIENT,
                "Metalevel policy exhausted the 60-observation stream",
            )
        (
            time_matched_oracle_allocation,
            time_matched_oracle_utility,
            time_matched_convergence,
        ) = oracle_at(
            remaining_time
        )
        _require(
            time_matched_oracle_utility + 1e-9 >= realized_utility,
            "Time-matched oracle is below a feasible policy",
        )
        _require(
            initial_oracle_utility + 1e-9 >= realized_utility,
            "Initial oracle is below a metalevel policy",
        )
        policy_metrics = scarcity_allocation_metrics(
            config,
            true_state,
            remaining_time,
            allocation,
            classification_need_1=classification_need_1,
            classification_need_2=classification_need_2,
            classification_uses_hidden_true_state=(
                policy_id == "initial_full_information_oracle"
            ),
        )
        row: Dict[str, object] = {
            "stage": stage,
            "environment_id": environment_id,
            "environment_config_hash": canonical_hash(asdict(config)),
            "anchor_id": anchor_id,
            "source_anchor_id": descriptor.get("source_anchor_id", ""),
            "anchor_rule": descriptor.get("anchor_rule", ""),
            "anchor_support_kind": descriptor.get("anchor_support_kind", ""),
            "anchor_band": descriptor.get("anchor_band", ""),
            "pairing_group_id": pairing_group,
            "episode_set_id": descriptor.get("episode_set_id", ""),
            "confirmation_role": descriptor.get("confirmation_role", ""),
            "development_environment_id": descriptor.get(
                "development_environment_id", environment_id
            ),
            "selected_acquisition_class": descriptor.get("acquisition_class", ""),
            "selection_status": descriptor.get("selection_status", ""),
            "target_rule": descriptor.get("target_rule", ""),
            "episode_index": episode_index,
            "policy_id": policy_id,
            "policy_computation_seed": policy_seed,
            "policy_seed_fingerprint": canonical_hash(
                [stage, environment_id, episode_index, policy_id, policy_seed]
            ),
            "need_1": true_state.need_1,
            "need_2": true_state.need_2,
            "final_choice_classification_basis": classification_basis,
            "terminal_posterior_mean_1": classification_need_1,
            "terminal_posterior_mean_2": classification_need_2,
            "terminal_posterior_variance_1": terminal_variance_1,
            "terminal_posterior_variance_2": terminal_variance_2,
            "true_state_seed_1": paired.true_state_seed_1,
            "true_state_seed_2": paired.true_state_seed_2,
            "true_state_fingerprint": paired.true_state_fingerprint,
            "innovation_seed_1": paired.innovation_seed_1,
            "innovation_seed_2": paired.innovation_seed_2,
            "standardized_innovation_hash_1": paired.standardized_innovation_hash_1,
            "standardized_innovation_hash_2": paired.standardized_innovation_hash_2,
            "transformed_stream_hash_1": paired.transformed_stream_hash_1,
            "transformed_stream_hash_2": paired.transformed_stream_hash_2,
            "allocation_to_person1": allocation,
            "remaining_time": remaining_time,
            "realized_utility": realized_utility,
            "online_sample_count": count_1 + count_2,
            "sample_count_1": count_1,
            "sample_count_2": count_2,
            "immediate_termination": count_1 + count_2 == 0,
            "joint_active_search": count_1 + count_2 >= 2 and count_1 >= 1 and count_2 >= 1,
            "initial_remaining_time": initial_remaining,
            "initial_oracle_allocation": initial_oracle_allocation,
            "initial_oracle_utility": initial_oracle_utility,
            "time_matched_oracle_allocation": time_matched_oracle_allocation,
            "time_matched_oracle_utility": time_matched_oracle_utility,
            "time_matched_oracle_regret": max(
                0.0,
                time_matched_oracle_utility - realized_utility,
            ),
            "initial_oracle_regret": max(0.0, initial_oracle_utility - realized_utility),
            "sigma_sample": descriptor.get("sigma_sample", config.sigma_sample),
            "sample_time_cost_percent": descriptor.get(
                "sample_time_cost_percent",
                100.0 * config.sample_time_cost / config.total_time,
            ),
            "prior_sample_count": descriptor.get(
                "prior_sample_count",
                config.prior_sample_count_1,
            ),
            "mu_need": descriptor.get("mu_need", config.mu_need),
            "sigma_need": descriptor.get("sigma_need", config.sigma_need),
            "capacity_ratio": descriptor.get(
                "capacity_ratio",
                initial_remaining / (2.0 * config.mu_need),
            ),
            "total_time": config.total_time,
            "utility_exponent": config.utility_exponent,
            "lambda_shortfall": config.lambda_shortfall,
        }
        row.update(
            {
                f"initial_oracle_{field}": value
                for field, value in initial_convergence.items()
            }
        )
        row.update(
            {
                f"time_matched_oracle_{field}": value
                for field, value in time_matched_convergence.items()
            }
        )
        for field, value in initial_metrics.items():
            row[f"initial_{field}"] = value
        for field, value in policy_metrics.items():
            row[f"policy_{field}"] = value
        rows.append(row)
    _require(len(rows) == len(SCARCITY_POLICY_ORDER), "Metalevel policy Cartesian row is incomplete")
    _require(
        len({int(row["policy_computation_seed"]) for row in rows}) == len(rows),
        "Metalevel policy seeds are not disjoint",
    )
    return rows


def evaluate_metalevel_descriptor(
    descriptor: Mapping[str, object],
    *,
    stage: str,
    n_episodes: int,
    episode_start: int = 0,
    root_seed: int = SCARCITY_ROOT_SEED,
    oracle_grid_size: int = SCARCITY_ORACLE_GRID_SIZE,
    dense_grid_size: int | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for episode_index in range(episode_start, episode_start + n_episodes):
        rows.extend(
            evaluate_metalevel_episode(
                descriptor,
                stage=stage,
                episode_index=episode_index,
                root_seed=root_seed,
                oracle_grid_size=oracle_grid_size,
                dense_grid_size=dense_grid_size,
            )
        )
    return rows


def _policy_rows_by_episode(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[int, Mapping[str, Mapping[str, object]]]:
    indexed: Dict[int, Dict[str, Mapping[str, object]]] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        policy_id = str(row["policy_id"])
        episode = indexed.setdefault(episode_index, {})
        _require(policy_id not in episode, "Duplicate metalevel episode-policy row")
        episode[policy_id] = row
    for episode_index, policies in indexed.items():
        _require(
            set(policies) == set(SCARCITY_POLICY_ORDER),
            f"Metalevel policy coverage is incomplete for episode {episode_index}",
        )
        fingerprints = {str(row["true_state_fingerprint"]) for row in policies.values()}
        innovations = {
            (
                str(row["standardized_innovation_hash_1"]),
                str(row["standardized_innovation_hash_2"]),
            )
            for row in policies.values()
        }
        _require(len(fingerprints) == 1, "Policies do not share a true-state fingerprint")
        _require(len(innovations) == 1, "Policies do not share standardized innovations")
    return indexed


def _mean_finite(values: Sequence[float]) -> float:
    _require(bool(values) and all(math.isfinite(value) for value in values), "Mean input is empty or nonfinite")
    return float(statistics.mean(values))


def _mean_finite_or_blank(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return _mean_finite(values)


def summarize_metalevel_environment(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    _require(bool(rows), "Cannot summarize an empty metalevel environment")
    environment_id = str(rows[0]["environment_id"])
    _require(
        all(str(row["environment_id"]) == environment_id for row in rows),
        "Metalevel summary mixes environments",
    )
    indexed = _policy_rows_by_episode(rows)
    episode_indices = sorted(indexed)
    rr_rows = [indexed[index]["frozen_rr"] for index in episode_indices]
    both_positive_indices = [
        index
        for index in episode_indices
        if bool(indexed[index]["frozen_rr"]["initial_both_needs_positive"])
    ]
    positive_scarcity_indices = [
        index
        for index in both_positive_indices
        if not bool(indexed[index]["frozen_rr"]["initial_joint_goal_feasible"])
    ]
    relevant_indices = [
        index
        for index in positive_scarcity_indices
        if not bool(indexed[index]["frozen_rr"]["initial_effort_tie"])
    ]
    first = rows[0]
    summaries: List[Dict[str, object]] = []
    selected_class = str(first.get("selected_acquisition_class", ""))
    acquisition_classes = (
        (selected_class,)
        if selected_class in ("no_search", "active_search")
        else ("no_search", "active_search")
    )
    for acquisition_class in acquisition_classes:
        acquisition_field = (
            "immediate_termination" if acquisition_class == "no_search" else "joint_active_search"
        )
        summary: Dict[str, object] = {
            "stage": first["stage"],
            "environment_id": environment_id,
            "environment_config_hash": first["environment_config_hash"],
            "anchor_id": first["anchor_id"],
            "source_anchor_id": first["source_anchor_id"],
            "anchor_rule": first["anchor_rule"],
            "anchor_support_kind": first["anchor_support_kind"],
            "anchor_band": first["anchor_band"],
            "pairing_group_id": first["pairing_group_id"],
            "episode_set_id": first["episode_set_id"],
            "confirmation_role": first["confirmation_role"],
            "development_environment_id": first["development_environment_id"],
            "selection_status": first["selection_status"],
            "target_rule": first["target_rule"],
            "acquisition_class": acquisition_class,
            "n_episodes": len(episode_indices),
            "both_positive_jointly_infeasible_n": len(positive_scarcity_indices),
            "relevant_both_positive_jointly_infeasible_nontie_n": len(relevant_indices),
            "effort_tie_count_in_both_positive_jointly_infeasible": (
                len(positive_scarcity_indices) - len(relevant_indices)
            ),
            "effort_tie_rate_in_both_positive_jointly_infeasible": (
                (len(positive_scarcity_indices) - len(relevant_indices))
                / len(positive_scarcity_indices)
                if positive_scarcity_indices
                else ""
            ),
            "sigma_sample": first["sigma_sample"],
            "sample_time_cost_percent": first["sample_time_cost_percent"],
            "prior_sample_count": first["prior_sample_count"],
            "mu_need": first["mu_need"],
            "sigma_need": first["sigma_need"],
            "capacity_ratio": first["capacity_ratio"],
            "total_time": first["total_time"],
            "utility_exponent": first["utility_exponent"],
            "lambda_shortfall": first["lambda_shortfall"],
            "theoretical_individual_nonpositive_probability": (
                gaussian_nonpositive_probabilities(
                    float(first["mu_need"]),
                    float(first["sigma_need"]),
                )[0]
            ),
            "theoretical_either_nonpositive_probability": (
                gaussian_nonpositive_probabilities(
                    float(first["mu_need"]),
                    float(first["sigma_need"]),
                )[1]
            ),
            "observed_individual_nonpositive_count": sum(
                int(bool(row["initial_need_1_nonpositive"]))
                + int(bool(row["initial_need_2_nonpositive"]))
                for row in rr_rows
            ),
            "observed_individual_draw_count": 2 * len(rr_rows),
            "observed_either_nonpositive_count": sum(
                bool(row["initial_either_need_nonpositive"]) for row in rr_rows
            ),
            "observed_both_positive_count": sum(
                bool(row["initial_both_needs_positive"]) for row in rr_rows
            ),
        }
        summary["observed_individual_nonpositive_rate"] = (
            int(summary["observed_individual_nonpositive_count"])
            / int(summary["observed_individual_draw_count"])
        )
        summary["observed_either_nonpositive_rate"] = (
            int(summary["observed_either_nonpositive_count"]) / len(rr_rows)
        )
        summary["observed_both_positive_rate"] = (
            int(summary["observed_both_positive_count"]) / len(rr_rows)
        )
        for field in (
            "joint_goal_feasible",
            "at_least_lower_goal_meetable",
            "exactly_one_goal_individually_meetable",
            "both_individually_but_not_jointly_meetable",
            "neither_goal_meetable",
        ):
            count = sum(bool(row[f"initial_{field}"]) for row in rr_rows)
            summary[f"{field}_count_full_sample"] = count
            summary[f"{field}_rate_full_sample"] = count / len(rr_rows)
        for policy_id in SCARCITY_POLICY_ORDER:
            policy_rows = [indexed[index][policy_id] for index in episode_indices]
            summary[f"{policy_id}_mean_utility_full_sample"] = _mean_finite(
                [float(row["realized_utility"]) for row in policy_rows]
            )
            summary[f"{policy_id}_mean_utility_both_positive"] = _mean_finite_or_blank(
                [
                    float(indexed[index][policy_id]["realized_utility"])
                    for index in both_positive_indices
                ]
            )
            summary[
                f"{policy_id}_mean_utility_both_positive_jointly_infeasible_nontie"
            ] = _mean_finite_or_blank(
                [
                    float(indexed[index][policy_id]["realized_utility"])
                    for index in relevant_indices
                ]
            )
        summary.update(
            binomial_rate_summary(
                [row[acquisition_field] for row in rr_rows],
                "rr_acquisition",
            )
        )
        for metric in ("all_to_lower_match", "meet_lower_first_match", "more_to_lower"):
            summary.update(
                binomial_rate_summary(
                    [
                        indexed[index]["frozen_rr"][f"policy_{metric}"]
                        for index in relevant_indices
                    ],
                    f"rr_{metric}",
                )
            )
        for rule in ("all_to_lower", "meet_lower_first", "greatest_need"):
            summary.update(
                continuous_gap_summary(
                    [
                        float(indexed[index]["frozen_rr"][f"policy_{rule}_absolute_gap"])
                        for index in relevant_indices
                    ],
                    f"rr_{rule}_allocation_gap",
                )
            )
        summary.update(
            continuous_gap_summary(
                [
                    float(indexed[index]["frozen_rr"]["policy_realized_outcome_gap"])
                    for index in relevant_indices
                ],
                "rr_realized_outcome_gap",
            )
        )
        for comparator in ("equal_split", "greatest_need"):
            gain_values = [
                float(indexed[index]["frozen_rr"]["realized_utility"])
                - float(indexed[index][comparator]["realized_utility"])
                for index in episode_indices
            ]
            summary.update(
                paired_contrast_summary(gain_values, f"gain_vs_{comparator}")
            )
            for rule in ("all_to_lower", "meet_lower_first"):
                manual_policy = (
                    f"immediate_{rule}"
                    if acquisition_class == "no_search"
                    else f"manual_active_search_{rule}"
                )
                retained_values = [
                    float(indexed[index][manual_policy]["realized_utility"])
                    - SCARCITY_RETAINED_GAIN_FRACTION
                    * float(indexed[index]["frozen_rr"]["realized_utility"])
                    - (1.0 - SCARCITY_RETAINED_GAIN_FRACTION)
                    * float(indexed[index][comparator]["realized_utility"])
                    for index in episode_indices
                ]
                summary.update(
                    paired_contrast_summary(
                        retained_values,
                        f"retained_gain_{rule}_vs_{comparator}",
                    )
                )
                manual_gain_values = [
                    float(indexed[index][manual_policy]["realized_utility"])
                    - float(indexed[index][comparator]["realized_utility"])
                    for index in episode_indices
                ]
                summary.update(
                    paired_contrast_summary(
                        manual_gain_values,
                        f"manual_gain_{rule}_vs_{comparator}",
                    )
                )
        g_means = [
            float(summary[f"gain_vs_{comparator}_mean"])
            for comparator in ("equal_split", "greatest_need")
        ]
        summary["g_min"] = min(g_means)
        for rule in ("all_to_lower", "meet_lower_first"):
            d_means = [
                float(summary[f"retained_gain_{rule}_vs_{comparator}_mean"])
                for comparator in ("equal_split", "greatest_need")
            ]
            summary[f"d_min_{rule}"] = min(d_means)
        rule_scores = {
            rule: float(summary[f"d_min_{rule}"])
            for rule in ("all_to_lower", "meet_lower_first")
        }
        maximum_rule_score = max(rule_scores.values())
        diagnostic_rule = sorted(
            rule for rule, value in rule_scores.items() if value == maximum_rule_score
        )[0]
        summary["diagnostic_exact_policy"] = diagnostic_rule
        summary["max_d_min"] = maximum_rule_score
        acquisition_rate = _numeric(summary["rr_acquisition_rate"])
        more_rate = _numeric(summary["rr_more_to_lower_rate"])
        named_rule = str(summary["anchor_rule"])
        named_exact_rate = (
            _numeric(summary[f"rr_{named_rule}_match_rate"])
            if named_rule in ("all_to_lower", "meet_lower_first")
            else None
        )
        named_d_min = (
            float(summary[f"d_min_{named_rule}"])
            if named_rule in ("all_to_lower", "meet_lower_first")
            else None
        )
        exact_candidate = (
            summary["anchor_support_kind"] == "exact"
            and named_rule in ("all_to_lower", "meet_lower_first")
            and acquisition_rate is not None
            and acquisition_rate >= SCARCITY_SUPPORT_THRESHOLD
            and named_exact_rate is not None
            and named_exact_rate >= SCARCITY_SUPPORT_THRESHOLD
            and more_rate is not None
            and more_rate >= SCARCITY_SUPPORT_THRESHOLD
            and float(summary["g_min"]) > 0.0
            and named_d_min is not None
            and named_d_min >= 0.0
        )
        direction_candidate = (
            acquisition_rate is not None
            and acquisition_rate >= SCARCITY_SUPPORT_THRESHOLD
            and more_rate is not None
            and more_rate >= SCARCITY_SUPPORT_THRESHOLD
            and float(summary["g_min"]) > 0.0
        )
        summary["exact_candidate"] = exact_candidate
        summary["direction_candidate"] = direction_candidate
        summaries.append(summary)
    return summaries


def summarize_metalevel_rows(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    by_environment: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        by_environment.setdefault(str(row["environment_id"]), []).append(row)
    summaries: List[Dict[str, object]] = []
    for environment in sorted(by_environment):
        summaries.extend(summarize_metalevel_environment(by_environment[environment]))
    return summaries


SCARCITY_DEVELOPMENT_MEDOID_LEVELS: Mapping[str, Tuple[float, ...]] = {
    **SCARCITY_ORACLE_MEDOID_LEVELS,
    "sigma_sample": SCARCITY_SIGMA_SAMPLES,
    "sample_time_cost_percent": SCARCITY_SAMPLE_TIME_COST_PERCENTS,
    "prior_sample_count": tuple(float(value) for value in SCARCITY_PRIOR_SAMPLE_COUNTS),
}


def _diagnostic_fallback_key(row: Mapping[str, object]) -> Tuple[object, ...]:
    def score(field: str) -> float:
        value = _numeric(row.get(field))
        return value if value is not None else -math.inf

    return (
        -score("g_min"),
        -score("max_d_min"),
        -score("rr_acquisition_rate"),
        -score("rr_more_to_lower_rate"),
        -max(
            score("rr_all_to_lower_match_rate"),
            score("rr_meet_lower_first_match_rate"),
        ),
        str(row["environment_id"]),
    )


def _development_medoid(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return _normalized_medoid(
        rows,
        fields=tuple(SCARCITY_DEVELOPMENT_MEDOID_LEVELS),
        levels=SCARCITY_DEVELOPMENT_MEDOID_LEVELS,
    )


def _development_control_distance(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> float:
    distance = 0.0
    levels = {
        "sigma_sample": SCARCITY_SIGMA_SAMPLES,
        "sample_time_cost_percent": SCARCITY_SAMPLE_TIME_COST_PERCENTS,
        "prior_sample_count": tuple(float(value) for value in SCARCITY_PRIOR_SAMPLE_COUNTS),
    }
    for field, ordered in levels.items():
        left_index = tuple(float(value) for value in ordered).index(float(left[field]))
        right_index = tuple(float(value) for value in ordered).index(float(right[field]))
        distance += abs(left_index - right_index) / (len(ordered) - 1)
    return distance


def select_confirmation_targets(
    development_summaries: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    _require(bool(development_summaries), "Cannot freeze confirmation targets without development rows")
    targets: List[Dict[str, object]] = []
    contrasts: List[Dict[str, object]] = []
    for acquisition_class in ("no_search", "active_search"):
        class_rows = [
            row
            for row in development_summaries
            if row["acquisition_class"] == acquisition_class
        ]
        _require(bool(class_rows), f"Development rows omit acquisition class {acquisition_class}")
        exact_candidates = [row for row in class_rows if bool(row["exact_candidate"])]
        direction_candidates = [
            row for row in class_rows if bool(row["direction_candidate"])
        ]
        if exact_candidates:
            selected = _development_medoid(exact_candidates)
            status = "exact_candidate"
            target_rule = str(selected["anchor_rule"])
            rank_rule = "eligible_observed_medoid_then_environment_id"
        elif direction_candidates:
            selected = _development_medoid(direction_candidates)
            status = "direction_candidate"
            target_rule = "more_to_lower"
            rank_rule = "eligible_observed_medoid_then_environment_id"
        else:
            selected = dict(sorted(class_rows, key=_diagnostic_fallback_key)[0])
            status = "diagnostic_only"
            target_rule = str(selected["diagnostic_exact_policy"])
            rank_rule = (
                "g_min_desc,max_d_min_desc,A_q_desc,more_to_lower_desc,"
                "max_exact_rate_desc,environment_id_asc"
            )
        target = {
            "acquisition_class": acquisition_class,
            "selection_status": status,
            "target_rule": target_rule,
            "environment_id": selected["environment_id"],
            "anchor_id": selected["anchor_id"],
            "source_anchor_id": selected["source_anchor_id"],
            "pairing_group_id": scarcity_pairing_group_id(
                "confirmation",
                str(selected["anchor_id"]),
                "heldout_1200",
            ),
            "development_pairing_group_id": selected["pairing_group_id"],
            "development_row_hash": canonical_hash(selected),
            "selection_rank_rule": rank_rule,
            "medoid_mean_normalized_manhattan_distance": selected.get(
                "medoid_mean_normalized_manhattan_distance",
                "",
            ),
            "medoid_tie_count": selected.get("medoid_tie_count", ""),
            "diagnostic_exact_policy": selected["diagnostic_exact_policy"],
        }
        targets.append(target)

        same_anchor = [
            row
            for row in class_rows
            if row["anchor_id"] == selected["anchor_id"]
            and row["environment_id"] != selected["environment_id"]
        ]
        if status == "exact_candidate":
            eligible_controls = [row for row in same_anchor if not bool(row["exact_candidate"])]
            failure_predicate = "fails_exact_candidate"
        elif status == "direction_candidate":
            eligible_controls = [row for row in same_anchor if not bool(row["direction_candidate"])]
            failure_predicate = "fails_direction_candidate"
        else:
            eligible_controls = same_anchor
            failure_predicate = "closest_distinct_same_anchor"
        if eligible_controls:
            distances = [
                (_development_control_distance(selected, row), str(row["environment_id"]), row)
                for row in eligible_controls
            ]
            minimum_distance, _, control = sorted(distances, key=lambda item: (item[0], item[1]))[0]
            contrasts.append(
                {
                    "acquisition_class": acquisition_class,
                    "role": "diagnostic_contrast",
                    "target_environment_id": selected["environment_id"],
                    "environment_id": control["environment_id"],
                    "anchor_id": control["anchor_id"],
                    "pairing_group_id": scarcity_pairing_group_id(
                        "confirmation",
                        str(control["anchor_id"]),
                        "heldout_1200",
                    ),
                    "development_row_hash": canonical_hash(control),
                    "failure_predicate": failure_predicate,
                    "normalized_control_distance": minimum_distance,
                    "tie_break": "environment_id_ascending",
                    "can_establish_or_rescue_support": False,
                }
            )
        else:
            contrasts.append(
                {
                    "acquisition_class": acquisition_class,
                    "role": "diagnostic_contrast",
                    "target_environment_id": selected["environment_id"],
                    "environment_id": "",
                    "anchor_id": selected["anchor_id"],
                    "pairing_group_id": scarcity_pairing_group_id(
                        "confirmation",
                        str(selected["anchor_id"]),
                        "heldout_1200",
                    ),
                    "development_row_hash": "",
                    "failure_predicate": failure_predicate,
                    "normalized_control_distance": "",
                    "tie_break": "",
                    "can_establish_or_rescue_support": False,
                    "null_reason": "No eligible distinct same-anchor development row.",
                }
            )
    _require(len(targets) == 2, "Confirmation selection must freeze exactly two targets")
    _require(
        {str(target["acquisition_class"]) for target in targets} == {"no_search", "active_search"},
        "Confirmation targets do not cover both acquisition classes",
    )
    ordered_rows = sorted(
        [dict(row) for row in development_summaries],
        key=lambda row: (str(row["environment_id"]), str(row["acquisition_class"])),
    )
    return {
        "schema": "scarcity_confirmation_selection_v1",
        "development_rows_hash": canonical_hash(ordered_rows),
        "targets": targets,
        "contrasts": contrasts,
        "target_count": 2,
        "non_null_contrast_count": sum(bool(row["environment_id"]) for row in contrasts),
        "contrast_can_establish_or_rescue_support": False,
    }


def build_confirmation_descriptors(
    selection: Mapping[str, object],
    development_descriptors: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    by_environment = {
        str(descriptor["environment_id"]): descriptor
        for descriptor in development_descriptors
    }
    descriptors: List[Dict[str, object]] = []
    target_by_class = {
        str(row["acquisition_class"]): row for row in selection["targets"]  # type: ignore[index]
    }
    for role, selected_rows in (
        ("target", selection["targets"]),
        ("diagnostic_contrast", selection["contrasts"]),
    ):
        for selected in selected_rows:  # type: ignore[union-attr]
            development_environment = str(selected["environment_id"])
            if not development_environment:
                continue
            _require(
                development_environment in by_environment,
                "Confirmation selection references an unknown development environment",
            )
            source = dict(by_environment[development_environment])
            acquisition_class = str(selected["acquisition_class"])
            target = target_by_class[acquisition_class]
            source.update(
                {
                    "stage": "confirmation",
                    "development_environment_id": development_environment,
                    "environment_id": (
                        f"confirmation_{acquisition_class}_{development_environment}"
                    ),
                    "episode_set_id": "heldout_1200",
                    "pairing_group_id": target["pairing_group_id"],
                    "confirmation_role": role,
                    "acquisition_class": acquisition_class,
                    "selection_status": target["selection_status"],
                    "target_rule": target["target_rule"],
                }
            )
            descriptors.append(source)
    _require(2 <= len(descriptors) <= 4, "Confirmation descriptor count must lie between two and four")
    _require(
        len({str(item["environment_id"]) for item in descriptors}) == len(descriptors),
        "Confirmation environment identities are not unique",
    )
    return descriptors


def holm_adjust_p_values(values: Mapping[str, float]) -> Dict[str, float]:
    _require(
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values()),
        "Holm family contains an invalid p-value",
    )
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


SCARCITY_PAIRED_REPORTING_PREFIXES = (
    "gain_vs_equal_split",
    "gain_vs_greatest_need",
    "retained_gain_all_to_lower_vs_equal_split",
    "retained_gain_all_to_lower_vs_greatest_need",
    "retained_gain_meet_lower_first_vs_equal_split",
    "retained_gain_meet_lower_first_vs_greatest_need",
    "manual_gain_all_to_lower_vs_equal_split",
    "manual_gain_all_to_lower_vs_greatest_need",
    "manual_gain_meet_lower_first_vs_equal_split",
    "manual_gain_meet_lower_first_vs_greatest_need",
)


def attach_holm_adjustment(
    summaries: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    copied = [dict(row) for row in summaries]
    family: Dict[str, float] = {}
    for row in copied:
        row_key = f"{row['environment_id']}|{row['acquisition_class']}"
        for prefix in SCARCITY_PAIRED_REPORTING_PREFIXES:
            value = _numeric(row.get(f"{prefix}_two_sided_p_value"))
            if value is not None:
                family[f"{row_key}|{prefix}"] = value
    adjusted = holm_adjust_p_values(family)
    for row in copied:
        row_key = f"{row['environment_id']}|{row['acquisition_class']}"
        for prefix in SCARCITY_PAIRED_REPORTING_PREFIXES:
            key = f"{row_key}|{prefix}"
            row[f"{prefix}_holm_adjusted_two_sided_p_value"] = (
                adjusted[key] if key in adjusted else ""
            )
    return copied


def _simultaneous_lower_pass(
    summary: Mapping[str, object],
    prefixes: Sequence[str],
    *,
    threshold: float,
    strict: bool,
) -> bool:
    values = [
        _numeric(summary.get(f"{prefix}_simultaneous_one_sided_95_low"))
        for prefix in prefixes
    ]
    if strict:
        return all(value is not None and value > threshold for value in values)
    return all(value is not None and value >= threshold for value in values)


def confirmation_precision_check(summary: Mapping[str, object]) -> Dict[str, object]:
    prefixes = (
        "rr_acquisition",
        "rr_all_to_lower_match",
        "rr_meet_lower_first_match",
        "rr_more_to_lower",
    )
    half_widths = [
        _numeric(summary.get(f"{prefix}_ci95_half_width")) for prefix in prefixes
    ]
    complete = all(value is not None for value in half_widths)
    maximum = max((value for value in half_widths if value is not None), default=None)
    passed = complete and maximum is not None and maximum <= 0.03
    return {
        "binomial_precision_fields_complete": complete,
        "maximum_key_rate_wilson_ci95_half_width": maximum if maximum is not None else "",
        "maximum_key_rate_wilson_ci95_half_width_le_0.03": passed,
    }


def classify_metalevel_target(
    summary: Mapping[str, object],
    selection_target: Mapping[str, object],
) -> Dict[str, object]:
    acquisition_class = str(selection_target["acquisition_class"])
    _require(summary["acquisition_class"] == acquisition_class, "Target summary acquisition class changed")
    selection_status = str(selection_target["selection_status"])
    target_rule = str(selection_target["target_rule"])
    acquisition_low = _numeric(summary.get("rr_acquisition_one_sided_95_low"))
    more_low = _numeric(summary.get("rr_more_to_lower_one_sided_95_low"))
    more_high = _numeric(summary.get("rr_more_to_lower_one_sided_95_high"))
    acquisition_pass = (
        acquisition_low is not None and acquisition_low >= SCARCITY_SUPPORT_THRESHOLD
    )
    direction_pass = more_low is not None and more_low >= SCARCITY_SUPPORT_THRESHOLD
    positive_gain_pass = _simultaneous_lower_pass(
        summary,
        ("gain_vs_equal_split", "gain_vs_greatest_need"),
        threshold=0.0,
        strict=True,
    )
    exact_match_pass = False
    retained_gain_pass = False
    if target_rule in ("all_to_lower", "meet_lower_first"):
        exact_low = _numeric(summary.get(f"rr_{target_rule}_match_one_sided_95_low"))
        exact_match_pass = (
            exact_low is not None and exact_low >= SCARCITY_SUPPORT_THRESHOLD
        )
        retained_gain_pass = _simultaneous_lower_pass(
            summary,
            (
                f"retained_gain_{target_rule}_vs_equal_split",
                f"retained_gain_{target_rule}_vs_greatest_need",
            ),
            threshold=0.0,
            strict=False,
        )
    exact_supported = (
        selection_status == "exact_candidate"
        and target_rule in ("all_to_lower", "meet_lower_first")
        and acquisition_pass
        and exact_match_pass
        and direction_pass
        and positive_gain_pass
        and retained_gain_pass
    )
    partial_supported = (
        not exact_supported
        and selection_status in ("exact_candidate", "direction_candidate")
        and acquisition_pass
        and direction_pass
        and positive_gain_pass
    )
    if exact_supported:
        classification = "supported_exact"
    elif partial_supported:
        classification = "partial_direction_only"
    elif more_high is not None and more_high < SCARCITY_SUPPORT_THRESHOLD:
        classification = "not_supported_in_frozen_scope"
    else:
        classification = "inconclusive_precision"

    comparator_means = {
        comparator: float(summary[f"{comparator}_mean_utility_full_sample"])
        for comparator in ("equal_split", "greatest_need")
    }
    cstar = sorted(
        comparator_means,
        key=lambda comparator: (-comparator_means[comparator], comparator),
    )[0]
    scale_rule = (
        target_rule
        if target_rule in ("all_to_lower", "meet_lower_first")
        else str(summary["diagnostic_exact_policy"])
    )
    rr_gap = float(summary[f"gain_vs_{cstar}_mean"])
    manual_gap = float(summary[f"manual_gain_{scale_rule}_vs_{cstar}_mean"])
    denominator = max(abs(rr_gap), abs(manual_gap), 1e-12)
    rr_half_width = (
        float(summary[f"gain_vs_{cstar}_ci95_high"])
        - float(summary[f"gain_vs_{cstar}_ci95_low"])
    ) / 2.0
    manual_half_width = (
        float(summary[f"manual_gain_{scale_rule}_vs_{cstar}_ci95_high"])
        - float(summary[f"manual_gain_{scale_rule}_vs_{cstar}_ci95_low"])
    ) / 2.0
    result = {
        "acquisition_class": acquisition_class,
        "environment_id": summary["environment_id"],
        "development_environment_id": summary["development_environment_id"],
        "selection_status": selection_status,
        "target_rule": target_rule,
        "classification": classification,
        "acquisition_lower_bound_pass": acquisition_pass,
        "exact_match_lower_bound_pass": exact_match_pass,
        "more_to_lower_lower_bound_pass": direction_pass,
        "positive_gain_simultaneous_lower_bounds_pass": positive_gain_pass,
        "retained_gain_simultaneous_lower_bounds_pass": retained_gain_pass,
        "more_to_lower_upper_bound_below_0.80": (
            more_high is not None and more_high < SCARCITY_SUPPORT_THRESHOLD
        ),
        "higher_mean_simple_comparator": cstar,
        "scale_aware_denominator": denominator,
        "rr_utility_ci_half_width_ratio": rr_half_width / denominator,
        "manual_utility_ci_half_width_ratio": manual_half_width / denominator,
        "diagnostic_contrasts_can_rescue_support": False,
    }
    result.update(confirmation_precision_check(summary))
    return result


def classify_confirmation_targets(
    confirmation_summaries: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
) -> List[Dict[str, object]]:
    target_summaries = [
        row for row in confirmation_summaries if row["confirmation_role"] == "target"
    ]
    results: List[Dict[str, object]] = []
    for target in selection["targets"]:  # type: ignore[index]
        matching = [
            row
            for row in target_summaries
            if row["acquisition_class"] == target["acquisition_class"]
            and row["development_environment_id"] == target["environment_id"]
        ]
        _require(len(matching) == 1, "Each frozen target must have exactly one confirmation summary")
        results.append(classify_metalevel_target(matching[0], target))
    _require(
        {str(row["acquisition_class"]) for row in results} == {"no_search", "active_search"},
        "Confirmation classifications do not cover both acquisition classes",
    )
    return results


__all__ = (
    "SCARCITY_ALLOCATION_TOLERANCE",
    "SCARCITY_ALLOCATION_METRIC_FIELDS",
    "SCARCITY_BINOMIAL_SUMMARY_SUFFIXES",
    "SCARCITY_CONFIRMATION_EPISODES",
    "SCARCITY_DEVELOPMENT_EPISODES",
    "SCARCITY_DENSE_CONVERGENCE_FIELDS",
    "SCARCITY_MORE_TO_LOWER_THRESHOLD",
    "SCARCITY_ORACLE_DENSE_GRID_SIZE",
    "SCARCITY_ORACLE_GRID_SIZE",
    "SCARCITY_POLICY_ORDER",
    "SCARCITY_PAIRED_CONTRAST_SUFFIXES",
    "SCARCITY_RETAINED_GAIN_FRACTION",
    "SCARCITY_ROOT_SEED",
    "SCARCITY_STREAM_CAPACITY_PER_RECIPIENT",
    "SCARCITY_SUPPORT_THRESHOLD",
    "SCARCITY_VOI_DRAWS",
    "ScarcityError",
    "ScarcityPairedEpisode",
    "attach_holm_adjustment",
    "binomial_rate_summary",
    "build_confirmation_descriptors",
    "build_deterministic_mechanism_cases",
    "build_development_descriptors",
    "build_gaussian_oracle_descriptors",
    "build_scarcity_paired_episode",
    "canonical_hash",
    "canonical_seed",
    "classify_confirmation_targets",
    "classify_metalevel_target",
    "confirmation_precision_check",
    "continuous_gap_summary",
    "dense_oracle_convergence",
    "equal_outcome_allocation",
    "evaluate_deterministic_mechanism_cases",
    "evaluate_gaussian_oracle_descriptor",
    "evaluate_metalevel_descriptor",
    "evaluate_metalevel_episode",
    "gaussian_nonpositive_probabilities",
    "holm_adjust_p_values",
    "object_level_stop_decision",
    "paired_contrast_summary",
    "scarcity_allocation_metrics",
    "scarcity_oracle_comparison_row",
    "scarcity_pairing_group_id",
    "scarcity_policy_seed",
    "select_confirmation_targets",
    "select_gaussian_oracle_anchors",
    "summarize_gaussian_oracle_environment",
    "summarize_gaussian_oracle_rows",
    "summarize_metalevel_environment",
    "summarize_metalevel_rows",
)
