from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from ..mdp.meta_mdp import (
        BeliefState,
        ContinuousAllocationMetaMDP,
        EnvironmentConfig,
        MetaPolicy,
        TrueState,
    )
    from ..policies.heuristic import EqualSplitBaselinePolicy, ManualActiveSearchEqualOutcomePolicy
    from .r5 import (
        deterministic_realized_utility,
        evaluation_episode_fingerprint,
        full_information_oracle_metrics,
        full_information_utilitarian_allocation,
        wilson_interval,
    )
    from .randomization import (
        EvaluationEpisode,
        build_evaluation_episode,
        build_observation_streams,
        observation_streams_for_mdp,
    )
    from .regimes import true_outcome_metrics_for_allocation
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import (
        BeliefState,
        ContinuousAllocationMetaMDP,
        EnvironmentConfig,
        MetaPolicy,
        TrueState,
    )
    from policies.heuristic import EqualSplitBaselinePolicy, ManualActiveSearchEqualOutcomePolicy
    from experiments.r5 import (
        deterministic_realized_utility,
        evaluation_episode_fingerprint,
        full_information_oracle_metrics,
        full_information_utilitarian_allocation,
        wilson_interval,
    )
    from experiments.randomization import (
        EvaluationEpisode,
        build_evaluation_episode,
        build_observation_streams,
        observation_streams_for_mdp,
    )
    from experiments.regimes import true_outcome_metrics_for_allocation


R6_POLICY_RR = "frozen_rr"
R6_POLICY_MANUAL = "manual_active_search_equal_outcome"
R6_POLICY_EQUAL_SPLIT = "equal_split"
R6_POLICY_ORACLE = "full_information_oracle"
R6_POLICY_ORDER = (
    R6_POLICY_RR,
    R6_POLICY_MANUAL,
    R6_POLICY_EQUAL_SPLIT,
    R6_POLICY_ORACLE,
)
R6_DEFAULT_GAP_BIN_EDGES = (0.0, 10.0, 20.0, 40.0, 80.0, math.inf)
R6_DEFAULT_TOTAL_NEED_BIN_EDGES = (-math.inf, 0.0, 50.0, 100.0, 150.0, math.inf)
R6_FIXED_TOTAL_NEED_MEAN = 35.0
R6_FIXED_TOTAL_NEED_DIFFERENCES = (0.0, 10.0, 20.0, 40.0, 60.0)
R6_ALLOCATION_TIE_TOLERANCE = 1e-9
R6_POLICY_SEED_OFFSET = 300_000


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_hash(config: EnvironmentConfig) -> str:
    return _canonical_hash(asdict(config))


def _non_sigma_config_hash(config: EnvironmentConfig) -> str:
    values = asdict(config)
    values.pop("sigma_need")
    return _canonical_hash(values)


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def _ci95(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * statistics.stdev(values) / math.sqrt(len(values)))


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return math.nan
    mean_x = _mean(x)
    mean_y = _mean(y)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    variance_x = sum((a - mean_x) ** 2 for a in x)
    variance_y = sum((b - mean_y) ** 2 for b in y)
    denominator = math.sqrt(variance_x * variance_y)
    return covariance / denominator if denominator > 0.0 else math.nan


def _rate_fields(values: Sequence[float], prefix: str) -> Dict[str, float]:
    successes = sum(value >= 0.5 for value in values)
    lower, upper = wilson_interval(successes, len(values)) if values else (math.nan, math.nan)
    one_sided_low = (
        wilson_interval(successes, len(values), one_sided=True)[0] if values else math.nan
    )
    return {
        f"{prefix}_rate": successes / len(values) if values else math.nan,
        f"{prefix}_ci95_low": lower,
        f"{prefix}_ci95_high": upper,
        f"{prefix}_one_sided_95_low": one_sided_low,
    }


def _max_steps(config: EnvironmentConfig) -> int:
    if config.max_meta_samples is not None:
        return max(100, config.max_meta_samples + 2)
    if config.sample_time_cost > 0.0:
        return max(100, math.ceil(config.total_time / config.sample_time_cost) + 2)
    raise ValueError("zero sample_time_cost requires max_meta_samples")


def _terminated_belief(config: EnvironmentConfig, remaining_time: float) -> BeliefState:
    elapsed = max(0.0, config.total_time - remaining_time)
    return BeliefState(
        mean_1=config.mu_need,
        var_1=config.sigma_need**2,
        mean_2=config.mu_need,
        var_2=config.sigma_need**2,
        deliberation_time=elapsed,
        history=[{"action": 0.0, "observation": math.nan, "cost": config.terminate_cost}],
    )


def _sign_stratum(
    mdp: ContinuousAllocationMetaMDP,
    true_state: TrueState,
    allocation: float,
    remaining_time: float,
    epsilon: float = 1e-9,
) -> str:
    amount_1, amount_2 = mdp.allocation_to_learning_outcomes(allocation, remaining_time)
    outcome_1 = amount_1 - true_state.need_1
    outcome_2 = amount_2 - true_state.need_2
    if abs(outcome_1) <= epsilon or abs(outcome_2) <= epsilon:
        return "boundary_zero"
    if outcome_1 > 0.0 and outcome_2 > 0.0:
        return "both_positive"
    if outcome_1 < 0.0 and outcome_2 < 0.0:
        return "both_negative"
    return "mixed_sign"


def _sample_counts(samples: Sequence[Mapping[str, float]]) -> Tuple[int, int]:
    sample_count_1 = sum(float(item.get("action", -1.0)) == 1.0 for item in samples)
    sample_count_2 = sum(float(item.get("action", -1.0)) == 2.0 for item in samples)
    return sample_count_1, sample_count_2


def ambiguous_close_true_equal_but_closer_equal_split(
    allocation: float,
    true_equal_allocation: float,
    allocation_tolerance: float,
    tie_tolerance: float = R6_ALLOCATION_TIE_TOLERANCE,
) -> float:
    distance_to_true_equal = abs(allocation - true_equal_allocation)
    distance_to_equal_split = abs(allocation - 0.5)
    return (
        1.0
        if distance_to_true_equal <= allocation_tolerance
        and distance_to_equal_split < distance_to_true_equal - tie_tolerance
        else 0.0
    )


def _observation_residual_hashes(
    config: EnvironmentConfig,
    episode: EvaluationEpisode,
) -> Tuple[str, str]:
    streams = episode.observation_streams or {}

    def residual_hash(action: str, need: float) -> str:
        residuals = [
            # Gaussian translation can change the last floating-point bits when
            # the need scale changes. Nine decimals is far tighter than the
            # scientific tolerances while remaining invariant to that rounding.
            round((float(observation) - need) / config.sigma_sample, 9)
            for observation in streams.get(action, [])
        ]
        return _canonical_hash(residuals)

    return (
        residual_hash(ContinuousAllocationMetaMDP.SAMPLE_PERSON_1, episode.true_state.need_1),
        residual_hash(ContinuousAllocationMetaMDP.SAMPLE_PERSON_2, episode.true_state.need_2),
    )


def _seeded_observation_residual_diagnostics(
    config: EnvironmentConfig,
    episode: EvaluationEpisode,
    observation_seed: int,
    tolerance: float = 1e-10,
) -> Dict[str, object]:
    """Reconstruct innovations from their seed and verify the stored streams."""

    streams = episode.observation_streams or {}
    stream_1 = streams.get(ContinuousAllocationMetaMDP.SAMPLE_PERSON_1, [])
    stream_2 = streams.get(ContinuousAllocationMetaMDP.SAMPLE_PERSON_2, [])
    rng = random.Random(observation_seed)
    residuals_1 = [float(rng.gauss(0.0, 1.0)) for _ in stream_1]
    residuals_2 = [float(rng.gauss(0.0, 1.0)) for _ in stream_2]

    def max_error(observations: Sequence[float], need: float, residuals: Sequence[float]) -> float:
        errors = [
            abs((float(observation) - need) / config.sigma_sample - residual)
            for observation, residual in zip(observations, residuals)
        ]
        return max(errors, default=0.0)

    error_1 = max_error(stream_1, episode.true_state.need_1, residuals_1)
    error_2 = max_error(stream_2, episode.true_state.need_2, residuals_2)
    if error_1 > tolerance or error_2 > tolerance:
        raise RuntimeError(
            "Observation stream is inconsistent with its hidden true state and frozen seed"
        )
    return {
        "observation_residual_hash_1": _canonical_hash(residuals_1),
        "observation_residual_hash_2": _canonical_hash(residuals_2),
        "max_observation_reconstruction_error_1": error_1,
        "max_observation_reconstruction_error_2": error_2,
        "observation_seed": observation_seed,
    }


def _policy_episode_row(
    environment: str,
    config: EnvironmentConfig,
    episode: EvaluationEpisode,
    policy: MetaPolicy,
    policy_name: str,
    policy_role: str,
    allocation_tolerance: float,
) -> Dict[str, object]:
    fingerprint, stream_1_hash, stream_2_hash = evaluation_episode_fingerprint(episode)
    residual_hash_1, residual_hash_2 = _observation_residual_hashes(config, episode)
    policy_seed = (config.random_seed or 0) + R6_POLICY_SEED_OFFSET + episode.episode_index * 17
    episode_config = replace(config, random_seed=policy_seed)
    mdp = ContinuousAllocationMetaMDP(
        episode_config,
        observation_streams=observation_streams_for_mdp(episode),
    )
    result = mdp.run_episode(policy, true_state=episode.true_state, max_steps=_max_steps(config))
    metrics = true_outcome_metrics_for_allocation(
        mdp,
        episode.true_state,
        result.final_belief,
        result.final_allocation_to_person1,
        allocation_tolerance=allocation_tolerance,
    )
    sample_count_1, sample_count_2 = _sample_counts(result.samples)
    row: Dict[str, object] = {
        "environment": environment,
        "environment_config_hash": _config_hash(config),
        "episode_index": episode.episode_index,
        "policy": policy_name,
        "policy_role": policy_role,
        "need_1": episode.true_state.need_1,
        "need_2": episode.true_state.need_2,
        "total_true_need": episode.true_state.need_1 + episode.true_state.need_2,
        "realized_true_need_gap": abs(episode.true_state.need_1 - episode.true_state.need_2),
        "episode_fingerprint": fingerprint,
        "policy_computation_seed": policy_seed,
        "observation_stream_hash_1": stream_1_hash,
        "observation_stream_hash_2": stream_2_hash,
        "observation_residual_hash_1": residual_hash_1,
        "observation_residual_hash_2": residual_hash_2,
        "allocation_to_person1": result.final_allocation_to_person1,
        "remaining_time": result.remaining_time,
        "realized_utility": result.realized_utility,
        "online_sample_count": len(result.samples),
        "sample_count_1": sample_count_1,
        "sample_count_2": sample_count_2,
        "sampled_both_recipients": 1.0 if sample_count_1 > 0 and sample_count_2 > 0 else 0.0,
        "immediate_termination": 1.0 if len(result.samples) == 0 else 0.0,
        "abs_allocation_from_equal": abs(result.final_allocation_to_person1 - 0.5),
        "allocation_closeness_advantage": (
            abs(result.final_allocation_to_person1 - 0.5)
            - float(metrics["true_equal_outcome_allocation_gap"])
        ),
        "outcome_closeness_advantage": (
            float(metrics["equal_split_outcome_distance_to_true_equal"])
            - float(metrics["outcome_distance_to_true_equal"])
        ),
        "realized_sign_stratum": _sign_stratum(
            mdp,
            episode.true_state,
            result.final_allocation_to_person1,
            result.remaining_time,
        ),
    }
    row.update(metrics)
    return row


def _oracle_row(
    environment: str,
    config: EnvironmentConfig,
    episode: EvaluationEpisode,
    allocation_tolerance: float,
    oracle_grid_size: int,
) -> Dict[str, object]:
    oracle = full_information_oracle_metrics(
        environment,
        config,
        episode,
        allocation_tolerance=allocation_tolerance,
        grid_size=oracle_grid_size,
    )
    fingerprint, stream_1_hash, stream_2_hash = evaluation_episode_fingerprint(episode)
    residual_hash_1, residual_hash_2 = _observation_residual_hashes(config, episode)
    remaining_time = float(oracle["remaining_time"])
    allocation = float(oracle["oracle_allocation"])
    mdp = ContinuousAllocationMetaMDP(config)
    metrics = true_outcome_metrics_for_allocation(
        mdp,
        episode.true_state,
        _terminated_belief(config, remaining_time),
        allocation,
        allocation_tolerance=allocation_tolerance,
    )
    row: Dict[str, object] = {
        "environment": environment,
        "environment_config_hash": _config_hash(config),
        "episode_index": episode.episode_index,
        "policy": R6_POLICY_ORACLE,
        "policy_role": "full_information_upper_benchmark",
        "need_1": episode.true_state.need_1,
        "need_2": episode.true_state.need_2,
        "total_true_need": episode.true_state.need_1 + episode.true_state.need_2,
        "realized_true_need_gap": abs(episode.true_state.need_1 - episode.true_state.need_2),
        "episode_fingerprint": fingerprint,
        "policy_computation_seed": (config.random_seed or 0)
        + R6_POLICY_SEED_OFFSET
        + episode.episode_index * 17,
        "observation_stream_hash_1": stream_1_hash,
        "observation_stream_hash_2": stream_2_hash,
        "observation_residual_hash_1": residual_hash_1,
        "observation_residual_hash_2": residual_hash_2,
        "allocation_to_person1": allocation,
        "remaining_time": remaining_time,
        "realized_utility": float(oracle["oracle_utility"]),
        "online_sample_count": 0,
        "sample_count_1": 0,
        "sample_count_2": 0,
        "sampled_both_recipients": 0.0,
        "immediate_termination": 1.0,
        "abs_allocation_from_equal": abs(allocation - 0.5),
        "allocation_closeness_advantage": abs(allocation - 0.5)
        - float(metrics["true_equal_outcome_allocation_gap"]),
        "outcome_closeness_advantage": float(
            metrics["equal_split_outcome_distance_to_true_equal"]
        )
        - float(metrics["outcome_distance_to_true_equal"]),
        "realized_sign_stratum": str(oracle["oracle_sign_stratum"]),
        "oracle_grid_size": oracle_grid_size,
        "oracle_grid_optimality_violation": oracle["oracle_grid_optimality_violation"],
    }
    row.update(metrics)
    return row


def _time_matched_oracle_fields(
    config: EnvironmentConfig,
    true_state: TrueState,
    remaining_time: float,
    policy_utility: float,
    prefix: str,
    allocation_tolerance: float,
    oracle_grid_size: int,
) -> Dict[str, object]:
    mdp = ContinuousAllocationMetaMDP(config)
    allocation, utility = full_information_utilitarian_allocation(
        mdp,
        true_state,
        remaining_time,
        grid_size=oracle_grid_size,
    )
    metrics = true_outcome_metrics_for_allocation(
        mdp,
        true_state,
        _terminated_belief(config, remaining_time),
        allocation,
        allocation_tolerance=allocation_tolerance,
    )
    return {
        f"{prefix}_oracle_allocation": allocation,
        f"{prefix}_oracle_utility": utility,
        f"{prefix}_oracle_regret": max(0.0, utility - policy_utility),
        f"{prefix}_oracle_raw_regret": utility - policy_utility,
        f"{prefix}_oracle_optimality_violation": max(0.0, policy_utility - utility),
        f"{prefix}_oracle_realized_outcome_gap": metrics["realized_outcome_gap"],
        f"{prefix}_oracle_true_equal_outcome": metrics["true_equal_outcome"],
        f"{prefix}_oracle_closer_to_true_equal_than_equal_split": metrics[
            "closer_to_true_equal_outcome_than_equal_split"
        ],
    }


def evaluate_r6_four_way_environment(
    environment: str,
    config: EnvironmentConfig,
    evaluation_episodes: Sequence[EvaluationEpisode],
    rr_policy: MetaPolicy,
    manual_samples_per_person: int = 3,
    allocation_tolerance: float = 0.05,
    oracle_grid_size: int = 4001,
    execution_order: Sequence[str] = R6_POLICY_ORDER,
) -> List[Dict[str, object]]:
    """Evaluate four frozen strategies on identical held-out episodes."""

    if not evaluation_episodes:
        return []
    if tuple(sorted(execution_order)) != tuple(sorted(R6_POLICY_ORDER)):
        raise ValueError("execution_order must contain each Round 6 policy exactly once")
    manual = ManualActiveSearchEqualOutcomePolicy(samples_per_person=manual_samples_per_person)
    split = EqualSplitBaselinePolicy()
    rows: List[Dict[str, object]] = []
    for episode in evaluation_episodes:
        if episode.observation_streams is None:
            raise ValueError("Round 6 requires pre-generated common observation streams")
        episode_rows: Dict[str, Dict[str, object]] = {}
        for policy_name in execution_order:
            if policy_name == R6_POLICY_RR:
                episode_rows[policy_name] = _policy_episode_row(
                    environment,
                    config,
                    episode,
                    rr_policy,
                    R6_POLICY_RR,
                    "resource_rational_approximation",
                    allocation_tolerance,
                )
            elif policy_name == R6_POLICY_MANUAL:
                episode_rows[policy_name] = _policy_episode_row(
                    environment,
                    config,
                    episode,
                    manual,
                    R6_POLICY_MANUAL,
                    "manual_active_search_benchmark",
                    allocation_tolerance,
                )
            elif policy_name == R6_POLICY_EQUAL_SPLIT:
                episode_rows[policy_name] = _policy_episode_row(
                    environment,
                    config,
                    episode,
                    split,
                    R6_POLICY_EQUAL_SPLIT,
                    "equal_split_benchmark",
                    allocation_tolerance,
                )
            else:
                episode_rows[policy_name] = _oracle_row(
                    environment,
                    config,
                    episode,
                    allocation_tolerance,
                    oracle_grid_size,
                )
        rr_row = episode_rows[R6_POLICY_RR]
        manual_row = episode_rows[R6_POLICY_MANUAL]
        split_row = episode_rows[R6_POLICY_EQUAL_SPLIT]
        oracle_row = episode_rows[R6_POLICY_ORACLE]
        shared_oracle_sign = oracle_row["realized_sign_stratum"]
        initial_oracle_utility = float(oracle_row["realized_utility"])
        rr_manual_allocation_gap = abs(
            float(rr_row["allocation_to_person1"])
            - float(manual_row["allocation_to_person1"])
        )
        for row in (rr_row, manual_row, split_row, oracle_row):
            row["oracle_sign_stratum"] = shared_oracle_sign
            row["initial_oracle_allocation"] = oracle_row["allocation_to_person1"]
            row["initial_oracle_utility"] = initial_oracle_utility
            row["utility_regret_to_initial_oracle"] = max(
                0.0, initial_oracle_utility - float(row["realized_utility"])
            )
            row["raw_utility_regret_to_initial_oracle"] = (
                initial_oracle_utility - float(row["realized_utility"])
            )
            row["rr_manual_allocation_gap"] = rr_manual_allocation_gap
        rr_row.update(
            _time_matched_oracle_fields(
                config,
                episode.true_state,
                float(rr_row["remaining_time"]),
                float(rr_row["realized_utility"]),
                "rr_time_matched",
                allocation_tolerance,
                oracle_grid_size,
            )
        )
        manual_row.update(
            _time_matched_oracle_fields(
                config,
                episode.true_state,
                float(manual_row["remaining_time"]),
                float(manual_row["realized_utility"]),
                "manual_time_matched",
                allocation_tolerance,
                oracle_grid_size,
            )
        )
        rows.extend(episode_rows[policy_name] for policy_name in execution_order)
    _validate_four_way_common_randomness(rows)
    return rows


def _validate_four_way_common_randomness(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: Dict[Tuple[str, int], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["environment"]), int(row["episode_index"]))
        grouped.setdefault(key, []).append(row)
    for key, episode_rows in grouped.items():
        if len(episode_rows) != 4 or {str(row["policy"]) for row in episode_rows} != set(
            R6_POLICY_ORDER
        ):
            raise RuntimeError(f"Round 6 policy-row mismatch for {key}")
        for field in (
            "need_1",
            "need_2",
            "episode_fingerprint",
            "observation_stream_hash_1",
            "observation_stream_hash_2",
        ):
            if len({row[field] for row in episode_rows}) != 1:
                raise RuntimeError(f"Round 6 common-randomness mismatch for {key}: {field}")


def _strata(rows: Sequence[Mapping[str, object]]) -> List[Tuple[str, str, List[Mapping[str, object]]]]:
    definitions = [
        ("all", "all", lambda row: True),
        *[
            (
                "oracle_sign",
                name,
                lambda row, expected=name: str(row.get("oracle_sign_stratum", "")) == expected,
            )
            for name in ("both_positive", "mixed_sign", "both_negative", "boundary_zero")
        ],
        (
            "true_equal_feasibility",
            "feasible",
            lambda row: float(row["exact_true_equal_outcome_feasible"]) >= 0.5,
        ),
        (
            "true_equal_feasibility",
            "infeasible",
            lambda row: float(row["exact_true_equal_outcome_feasible"]) < 0.5,
        ),
        (
            "need_sign",
            "both_needs_nonnegative",
            lambda row: float(row["negative_need_either"]) < 0.5,
        ),
        (
            "need_sign",
            "any_negative_need",
            lambda row: float(row["negative_need_either"]) >= 0.5,
        ),
    ]
    return [
        (dimension, label, [row for row in rows if predicate(row)])
        for dimension, label, predicate in definitions
    ]


def _policy_summary(
    environment: str,
    policy: str,
    stratum_dimension: str,
    stratum: str,
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    sample_counts = [float(row["online_sample_count"]) for row in rows]
    summary: Dict[str, object] = {
        "environment": environment,
        "policy": policy,
        "stratum_dimension": stratum_dimension,
        "stratum": stratum,
        "n_episodes": len(rows),
        "mean_utility": _mean([float(row["realized_utility"]) for row in rows]),
        "mean_utility_ci95": _ci95([float(row["realized_utility"]) for row in rows]),
        "mean_online_sample_count": _mean(sample_counts),
        "median_online_sample_count": _percentile(sample_counts, 0.5),
        "p90_online_sample_count": _percentile(sample_counts, 0.9),
        "max_online_sample_count": max(sample_counts) if sample_counts else math.nan,
        "mean_sample_count_1": _mean([float(row["sample_count_1"]) for row in rows]),
        "mean_sample_count_2": _mean([float(row["sample_count_2"]) for row in rows]),
        "mean_allocation_to_person1": _mean(
            [float(row["allocation_to_person1"]) for row in rows]
        ),
        "mean_abs_allocation_from_equal": _mean(
            [float(row["abs_allocation_from_equal"]) for row in rows]
        ),
        "mean_realized_outcome_gap": _mean(
            [float(row["realized_outcome_gap"]) for row in rows]
        ),
        "mean_allocation_distance_to_true_equal": _mean(
            [float(row["true_equal_outcome_allocation_gap"]) for row in rows]
        ),
        "mean_outcome_distance_to_true_equal": _mean(
            [float(row["outcome_distance_to_true_equal"]) for row in rows]
        ),
        "mean_outcome_closeness_advantage": _mean(
            [float(row["outcome_closeness_advantage"]) for row in rows]
        ),
        "exact_true_equal_outcome_feasibility_rate": _mean(
            [float(row["exact_true_equal_outcome_feasible"]) for row in rows]
        ),
        "negative_need_rate": _mean([float(row["negative_need_either"]) for row in rows]),
        "mean_utility_regret_to_initial_oracle": _mean(
            [float(row["utility_regret_to_initial_oracle"]) for row in rows]
        ),
        "mean_rr_manual_allocation_gap": _mean(
            [float(row["rr_manual_allocation_gap"]) for row in rows]
        ),
    }
    for field, prefix in (
        ("true_equal_outcome", "true_equal_outcome"),
        (
            "closer_to_true_equal_outcome_than_equal_split",
            "closer_to_true_equal_than_equal_split",
        ),
        ("true_outcome_classification_tie", "true_outcome_classification_tie"),
        ("sampled_both_recipients", "sampled_both_recipients"),
        ("immediate_termination", "immediate_termination"),
    ):
        summary.update(_rate_fields([float(row[field]) for row in rows], prefix))
    return summary


def summarize_r6_four_way(
    episode_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Return stratified policy summaries and paired episode-level contrasts."""

    _validate_four_way_common_randomness(episode_rows)
    policy_summaries: List[Dict[str, object]] = []
    paired_summaries: List[Dict[str, object]] = []
    comparisons = (
        ("rr_minus_manual", R6_POLICY_RR, R6_POLICY_MANUAL),
        ("rr_minus_equal_split", R6_POLICY_RR, R6_POLICY_EQUAL_SPLIT),
        ("manual_minus_equal_split", R6_POLICY_MANUAL, R6_POLICY_EQUAL_SPLIT),
        ("oracle_minus_rr", R6_POLICY_ORACLE, R6_POLICY_RR),
    )
    environments = sorted({str(row["environment"]) for row in episode_rows})
    for environment in environments:
        environment_rows = [row for row in episode_rows if row["environment"] == environment]
        for dimension, stratum, stratum_rows in _strata(environment_rows):
            for policy in R6_POLICY_ORDER:
                policy_summaries.append(
                    _policy_summary(
                        environment,
                        policy,
                        dimension,
                        stratum,
                        [row for row in stratum_rows if row["policy"] == policy],
                    )
                )
            by_policy = {
                policy: {int(row["episode_index"]): row for row in stratum_rows if row["policy"] == policy}
                for policy in R6_POLICY_ORDER
            }
            for contrast, positive, negative in comparisons:
                shared = sorted(set(by_policy[positive]).intersection(by_policy[negative]))
                differences = [
                    float(by_policy[positive][index]["realized_utility"])
                    - float(by_policy[negative][index]["realized_utility"])
                    for index in shared
                ]
                mean_difference = _mean(differences)
                ci = _ci95(differences)
                paired_summaries.append(
                    {
                        "environment": environment,
                        "stratum_dimension": dimension,
                        "stratum": stratum,
                        "contrast": contrast,
                        "positive_policy": positive,
                        "negative_policy": negative,
                        "n_pairs": len(differences),
                        "mean_paired_utility_difference": mean_difference,
                        "paired_utility_ci95": ci,
                        "paired_utility_ci95_low": mean_difference - ci,
                        "paired_utility_ci95_high": mean_difference + ci,
                    }
                )
            rr_rows = by_policy[R6_POLICY_RR]
            manual_rows = by_policy[R6_POLICY_MANUAL]
            split_rows = by_policy[R6_POLICY_EQUAL_SPLIT]
            shared = sorted(set(rr_rows).intersection(manual_rows, split_rows))
            recovery = [
                float(rr_rows[index]["realized_utility"])
                - float(manual_rows[index]["realized_utility"])
                + 0.10
                * (
                    float(manual_rows[index]["realized_utility"])
                    - float(split_rows[index]["realized_utility"])
                )
                for index in shared
            ]
            for label, source_policy, field in (
                (
                    "rr_time_matched_oracle_minus_rr",
                    R6_POLICY_RR,
                    "rr_time_matched_oracle_raw_regret",
                ),
                (
                    "manual_time_matched_oracle_minus_manual",
                    R6_POLICY_MANUAL,
                    "manual_time_matched_oracle_raw_regret",
                ),
            ):
                values = [
                    float(by_policy[source_policy][index][field])
                    for index in sorted(by_policy[source_policy])
                ]
                mean_value = _mean(values)
                ci = _ci95(values)
                paired_summaries.append(
                    {
                        "environment": environment,
                        "stratum_dimension": dimension,
                        "stratum": stratum,
                        "contrast": label,
                        "positive_policy": label.split("_minus_")[0],
                        "negative_policy": source_policy,
                        "n_pairs": len(values),
                        "mean_paired_utility_difference": mean_value,
                        "paired_utility_ci95": ci,
                        "paired_utility_ci95_low": mean_value - ci,
                        "paired_utility_ci95_high": mean_value + ci,
                    }
                )
            recovery_mean = _mean(recovery)
            recovery_ci = _ci95(recovery)
            paired_summaries.append(
                {
                    "environment": environment,
                    "stratum_dimension": dimension,
                    "stratum": stratum,
                    "contrast": "rr_90_percent_manual_improvement_recovery",
                    "positive_policy": R6_POLICY_RR,
                    "negative_policy": R6_POLICY_MANUAL,
                    "n_pairs": len(recovery),
                    "mean_paired_utility_difference": recovery_mean,
                    "paired_utility_ci95": recovery_ci,
                    "paired_utility_ci95_low": recovery_mean - recovery_ci,
                    "paired_utility_ci95_high": recovery_mean + recovery_ci,
                    "utility_noninferior": (
                        1.0 if recovery and recovery_mean - recovery_ci >= 0.0 else 0.0
                    ),
                }
            )
    _attach_recovery_classifications(policy_summaries, paired_summaries)
    return policy_summaries, paired_summaries


def _attach_recovery_classifications(
    policy_summaries: List[Dict[str, object]],
    paired_summaries: List[Dict[str, object]],
) -> None:
    environments = sorted({str(row["environment"]) for row in policy_summaries})
    for environment in environments:
        all_policy = {
            str(row["policy"]): row
            for row in policy_summaries
            if row["environment"] == environment and row["stratum_dimension"] == "all"
        }
        all_contrasts = {
            str(row["contrast"]): row
            for row in paired_summaries
            if row["environment"] == environment and row["stratum_dimension"] == "all"
        }
        rr = all_policy[R6_POLICY_RR]
        manual = all_policy[R6_POLICY_MANUAL]
        oracle = all_policy[R6_POLICY_ORACLE]
        manual_split = all_contrasts["manual_minus_equal_split"]
        recovery = all_contrasts["rr_90_percent_manual_improvement_recovery"]
        valid = (
            float(oracle["true_equal_outcome_rate"]) >= 0.80
            and float(oracle["closer_to_true_equal_than_equal_split_rate"]) >= 0.80
            and float(manual_split["paired_utility_ci95_low"]) > 0.0
            and float(manual["true_equal_outcome_rate"]) >= 0.80
            and float(manual["closer_to_true_equal_than_equal_split_rate"]) >= 0.80
            and float(manual["sampled_both_recipients_rate"]) >= 0.80
            and float(manual["mean_abs_allocation_from_equal"]) >= 0.05
        )
        rr_behavior = (
            float(rr["true_equal_outcome_rate"]) >= 0.80
            and float(rr["closer_to_true_equal_than_equal_split_rate"]) >= 0.80
            and float(rr["mean_online_sample_count"]) > 1.0
            and float(rr["sampled_both_recipients_rate"]) >= 0.80
            and float(rr["mean_abs_allocation_from_equal"]) >= 0.05
        )
        noninferior = float(recovery["paired_utility_ci95_low"]) >= 0.0
        successful = valid and noninferior and rr_behavior
        rr_higher_utility = float(rr["mean_utility"]) >= float(manual["mean_utility"])
        if not valid:
            classification = "invalid_diagnostic_environment"
        elif successful:
            classification = "successful_strategy_recovery"
        elif rr_higher_utility and not rr_behavior:
            classification = "higher_utility_behaviorally_different_strategy"
        elif not noninferior:
            classification = "possible_approximation_or_discovery_gap"
        else:
            classification = "utility_noninferior_but_behaviorally_nonequivalent"
        fields: Dict[str, object] = {
            "valid_diagnostic_environment": 1.0 if valid else 0.0,
            "rr_utility_noninferior": 1.0 if noninferior else 0.0,
            "rr_behaviorally_equivalent": 1.0 if rr_behavior else 0.0,
            "successful_strategy_recovery": 1.0 if successful else 0.0,
            "r6_recovery_classification": classification,
        }
        for row in policy_summaries:
            if row["environment"] == environment:
                row.update(fields)
        for row in paired_summaries:
            if row["environment"] == environment:
                row.update(fields)


def build_r6_sigma_need_configs(
    base_name: str,
    base_config: EnvironmentConfig,
    sigma_need_values: Sequence[float],
) -> List[Tuple[str, EnvironmentConfig]]:
    values = [float(value) for value in sigma_need_values]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("sigma_need values must be finite and positive")
    if len(set(values)) != len(values):
        raise ValueError("sigma_need values must be unique")
    return [
        (f"{base_name}__sigma_need={value:g}", replace(base_config, sigma_need=value))
        for value in sorted(values)
    ]


def _rr_sweep_row(
    environment: str,
    config: EnvironmentConfig,
    episode: EvaluationEpisode,
    rr_policy: MetaPolicy,
    allocation_tolerance: float,
    oracle_grid_size: int,
    observation_seed: Optional[int] = None,
) -> Dict[str, object]:
    row = _policy_episode_row(
        environment,
        config,
        episode,
        rr_policy,
        R6_POLICY_RR,
        "resource_rational_approximation",
        allocation_tolerance,
    )
    oracle = full_information_oracle_metrics(
        environment,
        config,
        episode,
        allocation_tolerance=allocation_tolerance,
        grid_size=oracle_grid_size,
    )
    initial_oracle_raw_regret = float(oracle["oracle_utility"]) - float(row["realized_utility"])
    row.update(
        {
            "sigma_need": config.sigma_need,
            "non_sigma_config_hash": _non_sigma_config_hash(config),
            "standardized_need_draw_1": (episode.true_state.need_1 - config.mu_need)
            / config.sigma_need,
            "standardized_need_draw_2": (episode.true_state.need_2 - config.mu_need)
            / config.sigma_need,
            "oracle_sign_stratum": oracle["oracle_sign_stratum"],
            "oracle_allocation": oracle["oracle_allocation"],
            "oracle_utility": oracle["oracle_utility"],
            "oracle_true_equal_outcome": oracle["oracle_true_equal_outcome"],
            "oracle_closer_to_true_equal_than_equal_split": oracle[
                "oracle_closer_to_true_equal_than_equal_split"
            ],
            "oracle_grid_optimality_violation": oracle["oracle_grid_optimality_violation"],
            "initial_oracle_raw_regret": initial_oracle_raw_regret,
            "initial_oracle_regret": max(0.0, initial_oracle_raw_regret),
            "initial_oracle_optimality_violation": max(0.0, -initial_oracle_raw_regret),
            "ambiguous_close_true_equal_but_closer_equal_split": (
                ambiguous_close_true_equal_but_closer_equal_split(
                    float(row["allocation_to_person1"]),
                    float(row["true_equal_outcome_allocation"]),
                    allocation_tolerance,
                )
            ),
        }
    )
    if observation_seed is not None:
        row.update(
            _seeded_observation_residual_diagnostics(
                config,
                episode,
                observation_seed=observation_seed,
            )
        )
    return row


def evaluate_r6_sigma_need_sweep(
    base_name: str,
    base_config: EnvironmentConfig,
    sigma_need_values: Sequence[float],
    n_episodes: int,
    rr_policy: MetaPolicy,
    allocation_tolerance: float = 0.05,
    oracle_grid_size: int = 4001,
    observations_per_person: int = 100,
    observation_seed_offset: int = 100_000,
    episode_start: int = 0,
) -> List[Dict[str, object]]:
    """Vary only need variability while preserving standardized episode randomness."""

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if episode_start < 0:
        raise ValueError("episode_start must be non-negative")
    rows: List[Dict[str, object]] = []
    for environment, config in build_r6_sigma_need_configs(
        base_name, base_config, sigma_need_values
    ):
        for episode_index in range(episode_start, episode_start + n_episodes):
            episode = build_evaluation_episode(
                config,
                episode_index=episode_index,
                include_observation_streams=True,
                observations_per_person=observations_per_person,
                max_online_samples=_max_steps(config),
            )
            rows.append(
                _rr_sweep_row(
                    environment,
                    config,
                    episode,
                    rr_policy,
                    allocation_tolerance,
                    oracle_grid_size,
                    observation_seed=(config.random_seed or 0)
                    + observation_seed_offset
                    + episode_index * 17,
                )
            )
    _validate_sigma_common_randomness(rows)
    return rows


def _validate_sigma_common_randomness(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    for episode_index, episode_rows in grouped.items():
        for field in (
            "standardized_need_draw_1",
            "standardized_need_draw_2",
        ):
            values = [float(row[field]) for row in episode_rows]
            if max(values) - min(values) > 1e-10:
                raise RuntimeError(
                    f"Round 6 standardized-need mismatch for episode {episode_index}: {field}"
                )
        for field in ("observation_residual_hash_1", "observation_residual_hash_2"):
            if len({row[field] for row in episode_rows}) != 1:
                raise RuntimeError(
                    f"Round 6 standardized-observation mismatch for episode {episode_index}: {field}"
                )
    if len({row["non_sigma_config_hash"] for row in rows}) != 1:
        raise RuntimeError("A non-sigma environment field changed during the controlled sweep")


def _gap_label(value: float, edges: Sequence[float]) -> str:
    for lower, upper in zip(edges[:-1], edges[1:]):
        if lower <= value < upper:
            upper_label = "inf" if math.isinf(upper) else f"{upper:g}"
            return f"[{lower:g},{upper_label})"
    raise ValueError(f"Gap {value} is outside the declared bins")


def _interval_label(value: float, edges: Sequence[float]) -> str:
    for lower, upper in zip(edges[:-1], edges[1:]):
        if lower <= value < upper:
            lower_label = "-inf" if math.isinf(lower) and lower < 0.0 else f"{lower:g}"
            upper_label = "inf" if math.isinf(upper) else f"{upper:g}"
            return f"[{lower_label},{upper_label})"
    raise ValueError(f"Value {value} is outside the declared bins")


def _sweep_summary_row(
    rows: Sequence[Mapping[str, object]],
    group_fields: Mapping[str, object],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        **group_fields,
        "n_episodes": len(rows),
        "mean_utility": _mean([float(row["realized_utility"]) for row in rows]),
        "mean_sample_count": _mean([float(row["online_sample_count"]) for row in rows]),
        "mean_sample_count_1": _mean([float(row["sample_count_1"]) for row in rows]),
        "mean_sample_count_2": _mean([float(row["sample_count_2"]) for row in rows]),
        "mean_abs_allocation_from_equal": _mean(
            [float(row["abs_allocation_from_equal"]) for row in rows]
        ),
        "mean_allocation_closeness_advantage": _mean(
            [float(row["allocation_closeness_advantage"]) for row in rows]
        ),
        "mean_outcome_closeness_advantage": _mean(
            [float(row["outcome_closeness_advantage"]) for row in rows]
        ),
        "mean_realized_outcome_gap": _mean(
            [float(row["realized_outcome_gap"]) for row in rows]
        ),
        "mean_realized_true_need_gap": _mean(
            [float(row["realized_true_need_gap"]) for row in rows]
        ),
        "need_gap_allocation_closeness_correlation": _pearson(
            [float(row["realized_true_need_gap"]) for row in rows],
            [float(row["allocation_closeness_advantage"]) for row in rows],
        ),
        "need_gap_outcome_closeness_correlation": _pearson(
            [float(row["realized_true_need_gap"]) for row in rows],
            [float(row["outcome_closeness_advantage"]) for row in rows],
        ),
    }
    for field, prefix in (
        ("true_equal_outcome", "true_equal_outcome"),
        (
            "closer_to_true_equal_outcome_than_equal_split",
            "closer_to_true_equal_than_equal_split",
        ),
        ("ambiguous_close_true_equal_but_closer_equal_split", "ambiguous_event"),
        ("sampled_both_recipients", "sampled_both_recipients"),
    ):
        summary.update(_rate_fields([float(row[field]) for row in rows], prefix))
    return summary


def summarize_r6_sigma_need_sweep(
    episode_rows: Sequence[Mapping[str, object]],
    gap_bin_edges: Sequence[float] = R6_DEFAULT_GAP_BIN_EDGES,
    total_need_bin_edges: Sequence[float] = R6_DEFAULT_TOTAL_NEED_BIN_EDGES,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    edges = tuple(float(value) for value in gap_bin_edges)
    if len(edges) < 2 or edges[0] != 0.0 or any(
        not lower < upper for lower, upper in zip(edges[:-1], edges[1:])
    ):
        raise ValueError("gap_bin_edges must be strictly increasing from zero")
    total_edges = tuple(float(value) for value in total_need_bin_edges)
    if len(total_edges) < 2 or any(
        not lower < upper for lower, upper in zip(total_edges[:-1], total_edges[1:])
    ):
        raise ValueError("total_need_bin_edges must be strictly increasing")
    environments = sorted(
        {str(row["environment"]) for row in episode_rows},
        key=lambda name: next(
            float(row["sigma_need"]) for row in episode_rows if row["environment"] == name
        ),
    )
    environment_summaries: List[Dict[str, object]] = []
    gap_summaries: List[Dict[str, object]] = []
    previous_rows: Optional[Dict[int, Mapping[str, object]]] = None
    previous_sigma: Optional[float] = None
    for environment in environments:
        rows = [row for row in episode_rows if row["environment"] == environment]
        sigma = float(rows[0]["sigma_need"])
        summary = _sweep_summary_row(
            rows,
            {"environment": environment, "sigma_need": sigma, "stratum": "all"},
        )
        current = {int(row["episode_index"]): row for row in rows}
        paired_changes: List[float] = []
        if previous_rows is not None:
            shared = sorted(set(current).intersection(previous_rows))
            paired_changes = [
                float(current[index]["allocation_closeness_advantage"])
                - float(previous_rows[index]["allocation_closeness_advantage"])
                for index in shared
            ]
        summary.update(
            {
                "previous_sigma_need": "" if previous_sigma is None else previous_sigma,
                "mean_paired_allocation_closeness_change": _mean(paired_changes),
                "paired_allocation_closeness_change_ci95": _ci95(paired_changes),
            }
        )
        environment_summaries.append(summary)
        for dimension, stratum, predicate in (
            ("need_sign", "both_needs_nonnegative", lambda row: float(row["negative_need_either"]) < 0.5),
            ("need_sign", "any_negative_need", lambda row: float(row["negative_need_either"]) >= 0.5),
            (
                "true_equal_feasibility",
                "feasible",
                lambda row: float(row["exact_true_equal_outcome_feasible"]) >= 0.5,
            ),
            (
                "true_equal_feasibility",
                "infeasible",
                lambda row: float(row["exact_true_equal_outcome_feasible"]) < 0.5,
            ),
            *[
                (
                    "oracle_sign",
                    label,
                    lambda row, expected=label: str(row["oracle_sign_stratum"]) == expected,
                )
                for label in ("both_positive", "mixed_sign", "both_negative", "boundary_zero")
            ],
        ):
            selected = [row for row in rows if predicate(row)]
            gap_summaries.append(
                _sweep_summary_row(
                    selected,
                    {
                        "environment": environment,
                        "sigma_need": sigma,
                        "stratum_dimension": dimension,
                        "stratum": stratum,
                        "gap_bin": "all",
                    },
                )
            )
        for gap_label in [_gap_label(edges[index], edges) for index in range(len(edges) - 1)]:
            selected = [
                row
                for row in rows
                if _gap_label(float(row["realized_true_need_gap"]), edges) == gap_label
            ]
            gap_summaries.append(
                _sweep_summary_row(
                    selected,
                    {
                        "environment": environment,
                        "sigma_need": sigma,
                        "stratum_dimension": "realized_true_need_gap",
                        "stratum": gap_label,
                        "gap_bin": gap_label,
                    },
                )
            )
        total_labels = [
            _interval_label(total_edges[index], total_edges)
            for index in range(len(total_edges) - 1)
        ]
        for total_label in total_labels:
            selected = [
                row
                for row in rows
                if _interval_label(float(row["total_true_need"]), total_edges) == total_label
            ]
            gap_summaries.append(
                _sweep_summary_row(
                    selected,
                    {
                        "environment": environment,
                        "sigma_need": sigma,
                        "stratum_dimension": "total_true_need",
                        "stratum": total_label,
                        "gap_bin": "all",
                    },
                )
            )
        previous_rows = current
        previous_sigma = sigma
    return environment_summaries, gap_summaries


def evaluate_r6_fixed_total_need_diagnostic(
    environment: str,
    base_config: EnvironmentConfig,
    n_episodes_per_difference: int,
    rr_policy: MetaPolicy,
    total_need_mean: float = R6_FIXED_TOTAL_NEED_MEAN,
    need_differences: Sequence[float] = R6_FIXED_TOTAL_NEED_DIFFERENCES,
    allocation_tolerance: float = 0.05,
    oracle_grid_size: int = 4001,
    observations_per_person: int = 100,
    observation_seed_offset: int = 600_000,
    episode_start: int = 0,
) -> List[Dict[str, object]]:
    """Hold total need fixed while changing only the constructed need gap."""

    differences = [float(value) for value in need_differences]
    if n_episodes_per_difference <= 0 or n_episodes_per_difference % 2:
        raise ValueError("n_episodes_per_difference must be positive and even")
    if episode_start < 0 or episode_start % 2:
        raise ValueError("episode_start must be non-negative and even")
    if len(set(differences)) != len(differences):
        raise ValueError("need differences must be unique")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 2.0 * total_need_mean
        for value in differences
    ):
        raise ValueError("need differences must preserve nonnegative constructed needs")
    rows: List[Dict[str, object]] = []
    base_seed = base_config.random_seed or 0
    stream_length = max(observations_per_person, _max_steps(base_config) + 5)
    for difference in differences:
        for episode_index in range(
            episode_start,
            episode_start + n_episodes_per_difference,
        ):
            orientation = -1.0 if episode_index % 2 == 0 else 1.0
            true_state = TrueState(
                need_1=total_need_mean + orientation * difference / 2.0,
                need_2=total_need_mean - orientation * difference / 2.0,
            )
            streams = build_observation_streams(
                base_config,
                true_state,
                seed=base_seed + observation_seed_offset + episode_index * 17,
                observations_per_person=stream_length,
            )
            episode = EvaluationEpisode(
                episode_index=episode_index,
                true_state=true_state,
                observation_streams=streams,
            )
            row = _rr_sweep_row(
                f"{environment}__fixed_gap={difference:g}",
                base_config,
                episode,
                rr_policy,
                allocation_tolerance,
                oracle_grid_size,
                observation_seed=base_seed + observation_seed_offset + episode_index * 17,
            )
            row.update(
                {
                    "mechanism_diagnostic": "fixed_total_need_constructed_true_states",
                    "constructed_true_state": 1.0,
                    "fixed_total_need_mean": total_need_mean,
                    "constructed_need_difference": difference,
                    "orientation": orientation,
                    "orientation_rule": "negative_on_even_episode_positive_on_odd_episode",
                    "seed_namespace": observation_seed_offset,
                }
            )
            rows.append(row)
    _validate_fixed_total_common_randomness(rows)
    return rows


def _validate_fixed_total_common_randomness(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    for episode_index, episode_rows in grouped.items():
        total_needs = [float(row["need_1"]) + float(row["need_2"]) for row in episode_rows]
        if max(total_needs) - min(total_needs) > 1e-10:
            raise RuntimeError(f"Fixed-total-need mismatch for episode {episode_index}")
        for field in ("observation_residual_hash_1", "observation_residual_hash_2"):
            if len({row[field] for row in episode_rows}) != 1:
                raise RuntimeError(
                    f"Fixed-total standardized-observation mismatch for episode {episode_index}: {field}"
                )


def summarize_r6_fixed_total_need_diagnostic(
    episode_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    differences = sorted({float(row["constructed_need_difference"]) for row in episode_rows})
    previous_by_episode: Optional[Dict[int, Mapping[str, object]]] = None
    previous_difference: Optional[float] = None
    for difference in differences:
        rows = [
            row for row in episode_rows if float(row["constructed_need_difference"]) == difference
        ]
        definitions = [
            ("all", "all", lambda row: True),
            (
                "need_sign",
                "both_needs_nonnegative",
                lambda row: float(row["negative_need_either"]) < 0.5,
            ),
            (
                "need_sign",
                "any_negative_need",
                lambda row: float(row["negative_need_either"]) >= 0.5,
            ),
            (
                "true_equal_feasibility",
                "feasible",
                lambda row: float(row["exact_true_equal_outcome_feasible"]) >= 0.5,
            ),
            (
                "true_equal_feasibility",
                "infeasible",
                lambda row: float(row["exact_true_equal_outcome_feasible"]) < 0.5,
            ),
            *[
                (
                    "oracle_sign",
                    label,
                    lambda row, expected=label: str(row["oracle_sign_stratum"]) == expected,
                )
                for label in ("both_positive", "mixed_sign", "both_negative", "boundary_zero")
            ],
        ]
        for dimension, stratum, predicate in definitions:
            selected = [row for row in rows if predicate(row)]
            summary = _sweep_summary_row(
                selected,
                {
                    "mechanism_diagnostic": "fixed_total_need_constructed_true_states",
                    "constructed_need_difference": difference,
                    "fixed_total_need_mean": (
                        float(rows[0]["fixed_total_need_mean"]) if rows else math.nan
                    ),
                    "stratum_dimension": dimension,
                    "stratum": stratum,
                },
            )
            if dimension == "all":
                current_by_episode = {
                    int(row["episode_index"]): row for row in rows
                }
                shared = (
                    sorted(set(current_by_episode).intersection(previous_by_episode))
                    if previous_by_episode is not None
                    else []
                )
                allocation_changes = [
                    float(current_by_episode[index]["allocation_closeness_advantage"])
                    - float(previous_by_episode[index]["allocation_closeness_advantage"])
                    for index in shared
                ]
                ambiguous_changes = [
                    float(current_by_episode[index]["ambiguous_close_true_equal_but_closer_equal_split"])
                    - float(previous_by_episode[index]["ambiguous_close_true_equal_but_closer_equal_split"])
                    for index in shared
                ]
                summary.update(
                    {
                        "previous_constructed_need_difference": (
                            "" if previous_difference is None else previous_difference
                        ),
                        "mean_paired_allocation_closeness_change": _mean(allocation_changes),
                        "paired_allocation_closeness_change_ci95": _ci95(allocation_changes),
                        "mean_paired_ambiguous_event_change": _mean(ambiguous_changes),
                        "paired_ambiguous_event_change_ci95": _ci95(ambiguous_changes),
                    }
                )
                previous_by_episode = current_by_episode
                previous_difference = difference
            summaries.append(summary)
    if episode_rows:
        differences_all = [float(row["constructed_need_difference"]) for row in episode_rows]
        summaries.append(
            {
                "mechanism_diagnostic": "fixed_total_need_constructed_true_states",
                "constructed_need_difference": "all",
                "fixed_total_need_mean": float(episode_rows[0]["fixed_total_need_mean"]),
                "stratum_dimension": "mechanism_trend",
                "stratum": "all_differences",
                "n_episodes": len(episode_rows),
                "difference_allocation_closeness_correlation": _pearson(
                    differences_all,
                    [float(row["allocation_closeness_advantage"]) for row in episode_rows],
                ),
                "difference_outcome_closeness_correlation": _pearson(
                    differences_all,
                    [float(row["outcome_closeness_advantage"]) for row in episode_rows],
                ),
                "difference_ambiguous_event_correlation": _pearson(
                    differences_all,
                    [
                        float(row["ambiguous_close_true_equal_but_closer_equal_split"])
                        for row in episode_rows
                    ],
                ),
            }
        )
    return summaries


def select_r6_primary_environments(
    candidate_rows: Iterable[Mapping[str, object]],
    limit: int = 3,
) -> List[Mapping[str, object]]:
    """Apply the prespecified R4-only diagnostic-environment selection rule."""

    eligible = [
        row
        for row in candidate_rows
        if float(row["manual_active_minus_equal_split_utility"]) > 0.0
        and float(row["manual_active_true_equal_outcome_rate"]) >= 0.80
        and float(row["manual_active_closer_to_true_equal_rate"]) >= 0.80
        and float(row["manual_active_mean_sample_count"]) >= 6.0
        and float(row["manual_active_mean_abs_allocation_from_equal"]) >= 0.05
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["manual_active_minus_equal_split_utility"]),
            str(row["environment"]),
        )
    )
    if len(eligible) < limit:
        raise ValueError(f"Only {len(eligible)} eligible environments; {limit} required")
    return eligible[:limit]
