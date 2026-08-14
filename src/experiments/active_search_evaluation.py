from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, replace
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from ..mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig, MetaPolicy, TrueState, utility
    from ..policies.heuristic import ManualActiveSearchEqualOutcomePolicy
    from ..policies.voi import MyopicValueOfInformationPolicy
    from .randomization import (
        EvaluationEpisode,
        build_evaluation_episode,
        build_evaluation_episodes,
        observation_streams_for_mdp,
    )
    from .regimes import true_outcome_metrics_for_allocation
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig, MetaPolicy, TrueState, utility
    from policies.heuristic import ManualActiveSearchEqualOutcomePolicy
    from policies.voi import MyopicValueOfInformationPolicy
    from experiments.randomization import (
        EvaluationEpisode,
        build_evaluation_episode,
        build_evaluation_episodes,
        observation_streams_for_mdp,
    )
    from experiments.regimes import true_outcome_metrics_for_allocation


ACTIVE_SEARCH_TRUE_EQUAL_THRESHOLD = 0.80
ACTIVE_SEARCH_CLOSER_THRESHOLD = 0.80
HISTORICAL_TRUE_EQUAL_THRESHOLD = 0.90


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluation_episode_fingerprint(episode: EvaluationEpisode) -> Tuple[str, str, str]:
    """Bind a paired evaluation row to its hidden state and observation streams."""

    streams = episode.observation_streams or {}
    stream_1_hash = _canonical_hash(
        streams.get(ContinuousAllocationMetaMDP.SAMPLE_PERSON_1, [])
    )
    stream_2_hash = _canonical_hash(
        streams.get(ContinuousAllocationMetaMDP.SAMPLE_PERSON_2, [])
    )
    fingerprint = _canonical_hash(
        {
            "episode_index": episode.episode_index,
            "need_1": episode.true_state.need_1,
            "need_2": episode.true_state.need_2,
            "stream_1_hash": stream_1_hash,
            "stream_2_hash": stream_2_hash,
        }
    )
    return fingerprint, stream_1_hash, stream_2_hash


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def _ci95(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * statistics.stdev(values) / math.sqrt(len(values)))


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
    one_sided: bool = False,
) -> Tuple[float, float]:
    """Wilson score interval for a binomial rate."""

    if trials <= 0:
        return math.nan, math.nan
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials")
    quantile = confidence if one_sided else 0.5 + confidence / 2.0
    z = NormalDist().inv_cdf(quantile)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def deterministic_realized_utility(
    mdp: ContinuousAllocationMetaMDP,
    true_state: TrueState,
    allocation_to_person1: float,
    remaining_time: float,
) -> float:
    amount_1, amount_2 = mdp.allocation_to_learning_outcomes(allocation_to_person1, remaining_time)
    alpha = mdp.utility_exponent()
    return utility(amount_1 - true_state.need_1, mdp.config.lambda_shortfall, alpha) + utility(
        amount_2 - true_state.need_2,
        mdp.config.lambda_shortfall,
        alpha,
    )


def full_information_utilitarian_allocation(
    mdp: ContinuousAllocationMetaMDP,
    true_state: TrueState,
    remaining_time: float,
    grid_size: int = 4001,
) -> Tuple[float, float]:
    """Deterministically maximize realized utilitarian utility on a fine grid.

    The objective is one dimensional but can be non-concave in the shortfall
    region. A global grid is therefore safer than a concavity-based optimizer.
    The returned grid step is recorded by the caller and convergence is checked
    with a denser grid in tests and selected server diagnostics.
    """

    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    rate_1, rate_2 = mdp.learning_rates()
    special_allocations = [0.0, 0.5, 1.0]
    if remaining_time > 0.0 and rate_1 > 0.0:
        special_allocations.append(true_state.need_1 / (rate_1 * remaining_time))
    if remaining_time > 0.0 and rate_2 > 0.0:
        special_allocations.append(1.0 - true_state.need_2 / (rate_2 * remaining_time))
    equal_denominator = (rate_1 + rate_2) * remaining_time
    if equal_denominator > 0.0:
        special_allocations.append(
            (true_state.need_1 - true_state.need_2 + rate_2 * remaining_time)
            / equal_denominator
        )
    special_allocations = [min(1.0, max(0.0, value)) for value in special_allocations]

    try:
        import numpy as np  # type: ignore

        grid = np.unique(
            np.concatenate((np.linspace(0.0, 1.0, grid_size), np.asarray(special_allocations)))
        )
        outcome_1 = rate_1 * grid * remaining_time - true_state.need_1
        outcome_2 = rate_2 * (1.0 - grid) * remaining_time - true_state.need_2
        alpha = mdp.utility_exponent()
        utility_1 = np.where(
            outcome_1 < 0.0,
            -mdp.config.lambda_shortfall * np.power(np.maximum(-outcome_1, 0.0), alpha),
            np.power(np.maximum(outcome_1, 0.0), alpha),
        )
        utility_2 = np.where(
            outcome_2 < 0.0,
            -mdp.config.lambda_shortfall * np.power(np.maximum(-outcome_2, 0.0), alpha),
            np.power(np.maximum(outcome_2, 0.0), alpha),
        )
        values = utility_1 + utility_2
        best_value = float(np.max(values))
        tolerance = 1e-12 * max(1.0, abs(best_value))
        candidates = np.flatnonzero(values >= best_value - tolerance)
        best_index = min(candidates, key=lambda index: abs(float(grid[int(index)]) - 0.5))
        return float(grid[int(best_index)]), float(values[int(best_index)])
    except ImportError:
        grid = sorted(
            set([index / (grid_size - 1) for index in range(grid_size)] + special_allocations)
        )
        values = [
            deterministic_realized_utility(mdp, true_state, allocation, remaining_time)
            for allocation in grid
        ]
        best_value = max(values)
        tolerance = 1e-12 * max(1.0, abs(best_value))
        candidates = [index for index, value in enumerate(values) if value >= best_value - tolerance]
        best_index = min(candidates, key=lambda index: abs(grid[index] - 0.5))
        return grid[best_index], values[best_index]


def _outcome_values(
    mdp: ContinuousAllocationMetaMDP,
    true_state: TrueState,
    allocation: float,
    remaining_time: float,
) -> Tuple[float, float]:
    amount_1, amount_2 = mdp.allocation_to_learning_outcomes(allocation, remaining_time)
    return amount_1 - true_state.need_1, amount_2 - true_state.need_2


def _sign_stratum(outcome_1: float, outcome_2: float, epsilon: float = 1e-9) -> str:
    if abs(outcome_1) <= epsilon or abs(outcome_2) <= epsilon:
        return "boundary_zero"
    if outcome_1 > 0.0 and outcome_2 > 0.0:
        return "both_positive"
    if outcome_1 < 0.0 and outcome_2 < 0.0:
        return "both_negative"
    return "mixed_sign"


def full_information_oracle_metrics(
    environment: str,
    config: EnvironmentConfig,
    episode: EvaluationEpisode,
    allocation_tolerance: float = 0.05,
    grid_size: int = 4001,
    remaining_time: Optional[float] = None,
) -> Dict[str, float | int | str]:
    """Compare the full-information utility optimum with true equal outcome and 50/50."""

    mdp = ContinuousAllocationMetaMDP(config)
    time_available = (
        max(0.0, config.total_time - config.terminate_cost)
        if remaining_time is None
        else max(0.0, remaining_time)
    )
    true_state = episode.true_state
    oracle_allocation, oracle_utility = full_information_utilitarian_allocation(
        mdp,
        true_state,
        time_available,
        grid_size=grid_size,
    )
    rate_1, rate_2 = mdp.learning_rates()
    denominator = (rate_1 + rate_2) * time_available
    unconstrained_equal = (
        math.nan
        if denominator <= 0.0
        else (true_state.need_1 - true_state.need_2 + rate_2 * time_available) / denominator
    )
    equal_feasible = not math.isnan(unconstrained_equal) and 0.0 <= unconstrained_equal <= 1.0
    equal_allocation = min(1.0, max(0.0, unconstrained_equal)) if not math.isnan(unconstrained_equal) else 0.5
    equal_utility = deterministic_realized_utility(mdp, true_state, equal_allocation, time_available)
    split_utility = deterministic_realized_utility(mdp, true_state, 0.5, time_available)
    raw_equal_regret = oracle_utility - equal_utility
    raw_split_regret = oracle_utility - split_utility
    oracle_outcomes = _outcome_values(mdp, true_state, oracle_allocation, time_available)
    equal_outcomes = _outcome_values(mdp, true_state, equal_allocation, time_available)
    oracle_gap = abs(oracle_outcomes[0] - oracle_outcomes[1])
    equal_gap = abs(equal_outcomes[0] - equal_outcomes[1])
    split_outcomes = _outcome_values(mdp, true_state, 0.5, time_available)
    split_gap = abs(split_outcomes[0] - split_outcomes[1])
    oracle_distance = max(0.0, oracle_gap - equal_gap)
    split_distance = max(0.0, split_gap - equal_gap)
    numerical_tolerance = 1e-9 * max(1.0, time_available * (rate_1 + rate_2))
    closer = 1.0 if oracle_distance + numerical_tolerance < split_distance else 0.0
    tie = 1.0 if abs(oracle_distance - split_distance) <= numerical_tolerance else 0.0
    outcome_tolerance = allocation_tolerance * time_available * (rate_1 + rate_2)

    row: Dict[str, float | int | str] = {
        "environment": environment,
        "episode_index": episode.episode_index,
        "need_1": true_state.need_1,
        "need_2": true_state.need_2,
        "remaining_time": time_available,
        "oracle_grid_size": grid_size,
        "oracle_grid_step": 1.0 / (grid_size - 1),
        "oracle_allocation_on_regular_grid": (
            1.0
            if abs(
                oracle_allocation * (grid_size - 1)
                - round(oracle_allocation * (grid_size - 1))
            )
            <= 1e-10
            else 0.0
        ),
        "oracle_allocation": oracle_allocation,
        "true_equal_outcome_allocation": equal_allocation,
        "unconstrained_true_equal_outcome_allocation": unconstrained_equal,
        "equal_split_allocation": 0.5,
        "exact_true_equal_outcome_feasible": 1.0 if equal_feasible else 0.0,
        "oracle_utility": oracle_utility,
        "true_equal_outcome_utility": equal_utility,
        "equal_split_utility": split_utility,
        "true_equal_outcome_regret": max(0.0, raw_equal_regret),
        "equal_split_regret": max(0.0, raw_split_regret),
        "raw_true_equal_outcome_regret": raw_equal_regret,
        "raw_equal_split_regret": raw_split_regret,
        "oracle_grid_optimality_violation": max(0.0, -raw_equal_regret, -raw_split_regret),
        "oracle_allocation_gap_to_true_equal": abs(oracle_allocation - equal_allocation),
        "oracle_realized_outcome_gap": oracle_gap,
        "true_equal_outcome_solution_gap": equal_gap,
        "equal_split_realized_outcome_gap": split_gap,
        "oracle_true_equal_outcome": 1.0 if oracle_distance <= outcome_tolerance else 0.0,
        "oracle_closer_to_true_equal_than_equal_split": closer,
        "oracle_equal_split_distance_tie": tie,
        "oracle_outcome_1": oracle_outcomes[0],
        "oracle_outcome_2": oracle_outcomes[1],
        "true_equal_outcome_1": equal_outcomes[0],
        "true_equal_outcome_2": equal_outcomes[1],
        "oracle_sign_stratum": _sign_stratum(*oracle_outcomes),
        "true_equal_sign_stratum": _sign_stratum(*equal_outcomes),
        "negative_need_person1": 1.0 if true_state.need_1 < 0.0 else 0.0,
        "negative_need_person2": 1.0 if true_state.need_2 < 0.0 else 0.0,
        "negative_need_either": 1.0 if true_state.need_1 < 0.0 or true_state.need_2 < 0.0 else 0.0,
        "negative_need_both": 1.0 if true_state.need_1 < 0.0 and true_state.need_2 < 0.0 else 0.0,
        "sample_time_cost": config.sample_time_cost,
        "sample_time_cost_percent": 100.0 * config.sample_time_cost / config.total_time,
    }
    row.update({f"config_{key}": value for key, value in asdict(config).items()})
    return row


def run_active_search_oracle_map(
    configs: Iterable[Tuple[str, EnvironmentConfig]],
    n_episodes: int,
    seed_namespace_offset: int = 0,
    allocation_tolerance: float = 0.05,
    grid_size: int = 4001,
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for environment, config in configs:
        seeded_config = replace(config, random_seed=(config.random_seed or 0) + seed_namespace_offset)
        episodes = build_evaluation_episodes(seeded_config, n_episodes=n_episodes)
        rows.extend(
            full_information_oracle_metrics(
                environment,
                seeded_config,
                episode,
                allocation_tolerance=allocation_tolerance,
                grid_size=grid_size,
            )
            for episode in episodes
        )
    return rows


def _rate_summary(values: Sequence[float], prefix: str) -> Dict[str, float]:
    successes = sum(value >= 0.5 for value in values)
    lower, upper = wilson_interval(successes, len(values))
    one_sided_lower, _ = wilson_interval(successes, len(values), one_sided=True)
    return {
        f"{prefix}_rate": successes / len(values) if values else math.nan,
        f"{prefix}_ci95_low": lower,
        f"{prefix}_ci95_high": upper,
        f"{prefix}_one_sided_95_low": one_sided_lower,
    }


def summarize_active_search_oracle_map(
    episode_rows: Sequence[Mapping[str, float | int | str]],
) -> List[Dict[str, float | int | str]]:
    """Aggregate oracle behavior with uncertainty and outcome-sign diagnostics."""

    environments = sorted({str(row["environment"]) for row in episode_rows})
    summaries: List[Dict[str, float | int | str]] = []
    for environment in environments:
        rows = [row for row in episode_rows if row["environment"] == environment]
        true_equal = [float(row["oracle_true_equal_outcome"]) for row in rows]
        closer = [float(row["oracle_closer_to_true_equal_than_equal_split"]) for row in rows]
        nonnegative_rows = [row for row in rows if float(row["negative_need_either"]) == 0.0]
        summary: Dict[str, float | int | str] = {
            "environment": environment,
            "n_episodes": len(rows),
            "mean_true_equal_outcome_regret": _mean(
                [float(row["true_equal_outcome_regret"]) for row in rows]
            ),
            "mean_equal_split_regret": _mean([float(row["equal_split_regret"]) for row in rows]),
            "mean_oracle_allocation_gap_to_true_equal": _mean(
                [float(row["oracle_allocation_gap_to_true_equal"]) for row in rows]
            ),
            "exact_true_equal_outcome_feasibility_rate": _mean(
                [float(row["exact_true_equal_outcome_feasible"]) for row in rows]
            ),
            "negative_need_either_rate": _mean([float(row["negative_need_either"]) for row in rows]),
            "max_oracle_grid_optimality_violation": max(
                float(row.get("oracle_grid_optimality_violation", 0.0)) for row in rows
            ),
            "nonnegative_episode_count": len(nonnegative_rows),
            "sample_time_cost": float(rows[0]["sample_time_cost"]),
            "sample_time_cost_percent": float(rows[0]["sample_time_cost_percent"]),
        }
        summary.update(_rate_summary(true_equal, "oracle_true_equal_outcome"))
        summary.update(_rate_summary(closer, "oracle_closer_to_true_equal_than_equal_split"))
        for stratum in ("both_positive", "mixed_sign", "both_negative", "boundary_zero"):
            stratum_rows = [row for row in rows if str(row["oracle_sign_stratum"]) == stratum]
            summary[f"oracle_{stratum}_rate"] = len(stratum_rows) / len(rows)
            summary[f"oracle_true_equal_outcome_rate_given_{stratum}"] = (
                _mean([float(row["oracle_true_equal_outcome"]) for row in stratum_rows])
                if stratum_rows
                else math.nan
            )
            summary[f"oracle_closer_rate_given_{stratum}"] = (
                _mean(
                    [
                        float(row["oracle_closer_to_true_equal_than_equal_split"])
                        for row in stratum_rows
                    ]
                )
                if stratum_rows
                else math.nan
            )
        if nonnegative_rows:
            summary["nonnegative_oracle_true_equal_outcome_rate"] = _mean(
                [float(row["oracle_true_equal_outcome"]) for row in nonnegative_rows]
            )
            summary["nonnegative_oracle_closer_rate"] = _mean(
                [float(row["oracle_closer_to_true_equal_than_equal_split"]) for row in nonnegative_rows]
            )
        else:
            summary["nonnegative_oracle_true_equal_outcome_rate"] = math.nan
            summary["nonnegative_oracle_closer_rate"] = math.nan
        summary["active_search_joint_oracle_candidate"] = 1.0 if (
            float(summary["oracle_true_equal_outcome_rate"]) >= ACTIVE_SEARCH_TRUE_EQUAL_THRESHOLD
            and float(summary["oracle_closer_to_true_equal_than_equal_split_rate"])
            >= ACTIVE_SEARCH_CLOSER_THRESHOLD
        ) else 0.0
        summaries.append(summary)
    return summaries


def evaluate_active_search_rr_environment(
    environment: str,
    config: EnvironmentConfig,
    n_episodes: int,
    rr_policy: Optional[MetaPolicy] = None,
    observation_draws: int = 500,
    allocation_tolerance: float = 0.05,
    seed_namespace_offset: int = 0,
    episode_start: int = 0,
) -> List[Dict[str, float | int | str]]:
    """Evaluate one frozen RR policy with episode-specific common observation streams."""

    policy = rr_policy or MyopicValueOfInformationPolicy(observation_draws=observation_draws)
    seeded_config = replace(config, random_seed=(config.random_seed or 0) + seed_namespace_offset)
    max_steps = max(100, (seeded_config.max_meta_samples or 0) + 2)
    rows: List[Dict[str, float | int | str]] = []
    for episode_index in range(episode_start, episode_start + n_episodes):
        # Build and consume one observation stream at a time. Near-zero sampling
        # costs can require long common-random streams, so retaining all episodes
        # would make peak memory grow linearly with the evaluation size.
        episode = build_evaluation_episode(
            seeded_config,
            episode_index=episode_index,
            include_observation_streams=True,
            observations_per_person=max(100, seeded_config.max_meta_samples or 0),
            max_online_samples=max_steps,
        )
        episode_fingerprint, stream_1_hash, stream_2_hash = evaluation_episode_fingerprint(episode)
        mdp = ContinuousAllocationMetaMDP(
            replace(seeded_config, random_seed=(seeded_config.random_seed or 0) + episode.episode_index * 17 + 1),
            observation_streams=observation_streams_for_mdp(episode),
        )
        result = mdp.run_episode(policy, true_state=episode.true_state, max_steps=max_steps)
        true_metrics = true_outcome_metrics_for_allocation(
            mdp,
            episode.true_state,
            result.final_belief,
            result.final_allocation_to_person1,
            allocation_tolerance=allocation_tolerance,
        )
        sample_count = len(result.samples)
        row: Dict[str, float | int | str] = {
            "environment": environment,
            "episode_index": episode.episode_index,
            "need_1": episode.true_state.need_1,
            "need_2": episode.true_state.need_2,
            "episode_fingerprint": episode_fingerprint,
            "observation_stream_hash_1": stream_1_hash,
            "observation_stream_hash_2": stream_2_hash,
            "allocation_to_person1": result.final_allocation_to_person1,
            "realized_utility": result.realized_utility,
            "remaining_time": result.remaining_time,
            "online_sample_count": sample_count,
            "sample_count_at_least_6": 1.0 if sample_count >= 6 else 0.0,
            "abs_allocation_from_equal": abs(result.final_allocation_to_person1 - 0.5),
            "sample_time_cost": config.sample_time_cost,
            "sample_time_cost_percent": 100.0 * config.sample_time_cost / config.total_time,
            "max_meta_samples": config.max_meta_samples if config.max_meta_samples is not None else "",
            "rr_policy": policy.name,
            "observation_draws": observation_draws,
        }
        row.update(true_metrics)
        row.update({f"config_{key}": value for key, value in asdict(config).items()})
        rows.append(row)
    return rows


def summarize_active_search_rr_environments(
    episode_rows: Sequence[Mapping[str, float | int | str]],
) -> List[Dict[str, float | int | str]]:
    """Aggregate RR utility and active-search diagnostics using prespecified thresholds."""

    environments = sorted({str(row["environment"]) for row in episode_rows})
    summaries: List[Dict[str, float | int | str]] = []
    for environment in environments:
        rows = [row for row in episode_rows if row["environment"] == environment]
        true_equal = [float(row["true_equal_outcome"]) for row in rows]
        closer = [float(row["closer_to_true_equal_outcome_than_equal_split"]) for row in rows]
        six = [float(row["sample_count_at_least_6"]) for row in rows]
        summary: Dict[str, float | int | str] = {
            "environment": environment,
            "n_episodes": len(rows),
            "mean_utility": _mean([float(row["realized_utility"]) for row in rows]),
            "mean_utility_ci95": _ci95([float(row["realized_utility"]) for row in rows]),
            "mean_sample_count": _mean([float(row["online_sample_count"]) for row in rows]),
            "mean_abs_allocation_from_equal": _mean(
                [float(row["abs_allocation_from_equal"]) for row in rows]
            ),
            "exact_true_equal_outcome_feasibility_rate": _mean(
                [float(row["exact_true_equal_outcome_feasible"]) for row in rows]
            ),
            "negative_need_either_rate": _mean([float(row["negative_need_either"]) for row in rows]),
            "sample_time_cost": float(rows[0]["sample_time_cost"]),
            "sample_time_cost_percent": float(rows[0]["sample_time_cost_percent"]),
            "max_meta_samples": rows[0]["max_meta_samples"],
        }
        summary.update(_rate_summary(true_equal, "true_equal_outcome"))
        summary.update(_rate_summary(closer, "closer_to_true_equal_outcome_than_equal_split"))
        summary.update(_rate_summary(six, "sample_count_at_least_6"))
        summary["historical_true_equal_candidate"] = 1.0 if (
            float(summary["true_equal_outcome_rate"]) >= HISTORICAL_TRUE_EQUAL_THRESHOLD
        ) else 0.0
        summary["active_search_joint_discovery_candidate"] = 1.0 if (
            float(summary["true_equal_outcome_rate"]) >= ACTIVE_SEARCH_TRUE_EQUAL_THRESHOLD
            and float(summary["closer_to_true_equal_outcome_than_equal_split_rate"])
            >= ACTIVE_SEARCH_CLOSER_THRESHOLD
        ) else 0.0
        summary["active_search_joint_confirmed"] = 1.0 if (
            float(summary["true_equal_outcome_one_sided_95_low"]) >= ACTIVE_SEARCH_TRUE_EQUAL_THRESHOLD
            and float(summary["closer_to_true_equal_outcome_than_equal_split_one_sided_95_low"])
            >= ACTIVE_SEARCH_CLOSER_THRESHOLD
        ) else 0.0
        summaries.append(summary)
    return summaries


def evaluate_active_search_fixed_sampling_budgets(
    environment: str,
    config: EnvironmentConfig,
    n_episodes: int,
    total_sample_budgets: Sequence[int],
    allocation_tolerance: float = 0.05,
    seed_namespace_offset: int = 0,
    episode_start: int = 0,
) -> List[Dict[str, float | int | str]]:
    """Evaluate balanced equal-outcome policies on common episodes by sample budget."""

    budgets = sorted(set(int(value) for value in total_sample_budgets))
    if not budgets or any(value < 0 or value % 2 for value in budgets):
        raise ValueError("total_sample_budgets must contain non-negative even integers")
    seeded_config = replace(config, random_seed=(config.random_seed or 0) + seed_namespace_offset)
    max_budget = max(budgets)
    bounded_config = replace(
        seeded_config,
        max_meta_samples=max(max_budget, seeded_config.max_meta_samples or 0),
    )
    episodes = [
        build_evaluation_episode(
            bounded_config,
            episode_index=episode_index,
            include_observation_streams=True,
            observations_per_person=max(10, max_budget // 2 + 2),
        )
        for episode_index in range(episode_start, episode_start + n_episodes)
    ]
    rows: List[Dict[str, float | int | str]] = []
    for episode in episodes:
        for total_budget in budgets:
            policy = ManualActiveSearchEqualOutcomePolicy(samples_per_person=total_budget // 2)
            mdp = ContinuousAllocationMetaMDP(
                replace(
                    bounded_config,
                    random_seed=(bounded_config.random_seed or 0) + episode.episode_index * 17 + 1,
                ),
                observation_streams=observation_streams_for_mdp(episode),
            )
            result = mdp.run_episode(
                policy,
                true_state=episode.true_state,
                max_steps=total_budget + 2,
            )
            true_metrics = true_outcome_metrics_for_allocation(
                mdp,
                episode.true_state,
                result.final_belief,
                result.final_allocation_to_person1,
                allocation_tolerance=allocation_tolerance,
            )
            row: Dict[str, float | int | str] = {
                "environment": environment,
                "episode_index": episode.episode_index,
                "sampling_budget_total": total_budget,
                "sampling_budget_per_person": total_budget // 2,
                "policy": policy.name,
                "need_1": episode.true_state.need_1,
                "need_2": episode.true_state.need_2,
                "allocation_to_person1": result.final_allocation_to_person1,
                "realized_utility": result.realized_utility,
                "remaining_time": result.remaining_time,
                "online_sample_count": len(result.samples),
                "sample_time_cost": config.sample_time_cost,
                "sample_time_cost_percent": 100.0 * config.sample_time_cost / config.total_time,
            }
            row.update(true_metrics)
            row.update({f"config_{key}": value for key, value in asdict(config).items()})
            rows.append(row)
    return rows


def summarize_active_search_fixed_sampling_budgets(
    episode_rows: Sequence[Mapping[str, float | int | str]],
) -> List[Dict[str, float | int | str]]:
    """Summarize paired utility and behavior across fixed information budgets."""

    environments = sorted({str(row["environment"]) for row in episode_rows})
    summaries: List[Dict[str, float | int | str]] = []
    for environment in environments:
        environment_rows = [row for row in episode_rows if row["environment"] == environment]
        budgets = sorted({int(float(row["sampling_budget_total"])) for row in environment_rows})
        utilities_by_budget = {
            budget: {
                int(float(row["episode_index"])): float(row["realized_utility"])
                for row in environment_rows
                if int(float(row["sampling_budget_total"])) == budget
            }
            for budget in budgets
        }
        previous_budget: Optional[int] = None
        for budget in budgets:
            rows = [
                row
                for row in environment_rows
                if int(float(row["sampling_budget_total"])) == budget
            ]
            true_equal = [float(row["true_equal_outcome"]) for row in rows]
            closer = [float(row["closer_to_true_equal_outcome_than_equal_split"]) for row in rows]
            deltas: List[float] = []
            if previous_budget is not None:
                shared = sorted(
                    set(utilities_by_budget[budget]).intersection(utilities_by_budget[previous_budget])
                )
                deltas = [
                    utilities_by_budget[budget][episode_index]
                    - utilities_by_budget[previous_budget][episode_index]
                    for episode_index in shared
                ]
            summary: Dict[str, float | int | str] = {
                "environment": environment,
                "sampling_budget_total": budget,
                "sampling_budget_per_person": budget // 2,
                "previous_sampling_budget_total": "" if previous_budget is None else previous_budget,
                "n_episodes": len(rows),
                "mean_utility": _mean([float(row["realized_utility"]) for row in rows]),
                "mean_utility_ci95": _ci95([float(row["realized_utility"]) for row in rows]),
                "mean_incremental_utility_vs_previous_budget": _mean(deltas),
                "paired_incremental_utility_ci95": _ci95(deltas),
                "utility_improvement_rate_vs_previous_budget": (
                    _mean([1.0 if value > 1e-12 else 0.0 for value in deltas])
                    if deltas
                    else math.nan
                ),
                "utility_tie_rate_vs_previous_budget": (
                    _mean([1.0 if abs(value) <= 1e-12 else 0.0 for value in deltas])
                    if deltas
                    else math.nan
                ),
                "mean_online_sample_count": _mean(
                    [float(row["online_sample_count"]) for row in rows]
                ),
                "mean_abs_allocation_from_equal": _mean(
                    [abs(float(row["allocation_to_person1"]) - 0.5) for row in rows]
                ),
                "sample_time_cost": float(rows[0]["sample_time_cost"]),
                "sample_time_cost_percent": float(rows[0]["sample_time_cost_percent"]),
            }
            summary.update(_rate_summary(true_equal, "true_equal_outcome"))
            summary.update(_rate_summary(closer, "closer_to_true_equal_outcome_than_equal_split"))
            summaries.append(summary)
            previous_budget = budget
    return summaries
