from __future__ import annotations

import math
import statistics
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

try:
    from ..mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig, EpisodeResult, MetaPolicy, TrueState
    from ..policies.heuristic import build_final_choice_heuristics, build_policy_library
    from ..policies.voi import BlinkeredPolicy, MyopicValueOfInformationPolicy
    from ..solvers.dp import DiscretizedDynamicProgrammingPolicy
    from .randomization import EvaluationEpisode, ensure_evaluation_episodes, observation_streams_for_mdp
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig, EpisodeResult, MetaPolicy, TrueState
    from policies.heuristic import build_final_choice_heuristics, build_policy_library
    from policies.voi import BlinkeredPolicy, MyopicValueOfInformationPolicy
    from solvers.dp import DiscretizedDynamicProgrammingPolicy
    from experiments.randomization import EvaluationEpisode, ensure_evaluation_episodes, observation_streams_for_mdp


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def _ci95(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * statistics.stdev(values) / math.sqrt(len(values)))


def _rmse(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return math.sqrt(statistics.mean([value * value for value in values]))


def _clip_allocation(allocation: float) -> float:
    return min(1.0, max(0.0, float(allocation)))


def _policy_parameter(policy: MetaPolicy, attribute: str) -> float | str:
    value = getattr(policy, attribute, "")
    return float(value) if isinstance(value, (int, float)) else value


def _episode_seed(config: EnvironmentConfig, episode: EvaluationEpisode, offset: int = 0) -> int:
    return (config.random_seed or 0) + episode.episode_index * 17 + 1 + offset


def _mdp_for_episode(
    config: EnvironmentConfig,
    episode: EvaluationEpisode,
    seed_offset: int = 0,
    use_common_observation_streams: bool = False,
) -> ContinuousAllocationMetaMDP:
    streams = observation_streams_for_mdp(episode) if use_common_observation_streams else None
    return ContinuousAllocationMetaMDP(
        replace(config, random_seed=_episode_seed(config, episode, seed_offset)),
        observation_streams=streams,
    )


def true_equal_outcome_allocation(
    mdp: ContinuousAllocationMetaMDP,
    true_state: TrueState,
    remaining_time: float,
) -> float:
    """Allocation that equalizes true realized outcome-minus-need if feasible.

    The realized outcome for each recipient is tutoring/help received minus
    hidden true need. Equal outcome/maximin therefore means making these two
    true realized outcomes as equal as possible. If exact equalization would
    require an allocation outside [0, 1], clipping gives the closest feasible
    equal-outcome reference in this two-person single-resource model.
    """

    rate_1, rate_2 = mdp.learning_rates()
    denominator = (rate_1 + rate_2) * remaining_time
    if denominator <= 0.0:
        return 0.5
    allocation = (true_state.need_1 - true_state.need_2 + rate_2 * remaining_time) / denominator
    return _clip_allocation(allocation)


def _realized_outcome_gap(
    true_state: TrueState,
    amount_1: float,
    amount_2: float,
) -> float:
    """Absolute gap between the two recipients' true outcome-minus-need values."""

    outcome_minus_need_1 = amount_1 - true_state.need_1
    outcome_minus_need_2 = amount_2 - true_state.need_2
    return abs(outcome_minus_need_1 - outcome_minus_need_2)


def true_outcome_metrics_for_allocation(
    mdp: ContinuousAllocationMetaMDP,
    true_state: TrueState,
    belief,
    allocation_to_person1: float,
    allocation_tolerance: float,
) -> Dict[str, float]:
    """Compute true-state equity/maximin diagnostics for one final choice.

    This is intentionally true-state based rather than belief based. The final
    allocation can be chosen from a policy's beliefs, but the diagnostic asks how
    equal the hidden realized outcomes actually are after the allocation.
    """

    allocation = _clip_allocation(allocation_to_person1)
    amount_1, amount_2, remaining_time, _ = mdp.realized_utility(true_state, allocation, belief)
    realized_gap = _realized_outcome_gap(true_state, amount_1, amount_2)

    # Equal split is the main alternative Falk asked us to separate from
    # equity/maximin. It is evaluated with the same true state and remaining time
    # as the policy's chosen allocation.
    equal_split_amount_1, equal_split_amount_2, _, _ = mdp.realized_utility(true_state, 0.5, belief)
    equal_split_gap = _realized_outcome_gap(true_state, equal_split_amount_1, equal_split_amount_2)

    # The feasible true-equal-outcome reference is the best outcome gap that
    # could be achieved if the hidden true needs were known. A nonzero reference
    # gap can remain when exact equalization is infeasible within [0, 1].
    true_equal_allocation = true_equal_outcome_allocation(mdp, true_state, remaining_time)
    true_equal_amount_1, true_equal_amount_2 = mdp.allocation_to_learning_outcomes(
        true_equal_allocation,
        remaining_time,
    )
    true_equal_solution_gap = _realized_outcome_gap(
        true_state,
        true_equal_amount_1,
        true_equal_amount_2,
    )

    true_equal_allocation_gap = abs(allocation - true_equal_allocation)
    equal_split_allocation_gap = abs(allocation - 0.5)
    allocation_distance_difference = true_equal_allocation_gap - equal_split_allocation_gap

    # Outcome-distance fields are primary for equity/maximin because they compare
    # realized outcome gaps. Allocation-distance fields are secondary diagnostics:
    # two allocations can be close while their realized outcome gaps differ, and
    # vice versa when learning rates or needs differ.
    outcome_distance_to_true_equal = max(0.0, realized_gap - true_equal_solution_gap)
    equal_split_outcome_distance_to_true_equal = max(0.0, equal_split_gap - true_equal_solution_gap)
    outcome_tolerance = allocation_tolerance * remaining_time * sum(mdp.learning_rates())
    outcome_distance_difference = (
        outcome_distance_to_true_equal - equal_split_outcome_distance_to_true_equal
    )

    # This classifier implements Falk's requested logical comparison: is the
    # chosen allocation closer to the feasible true equal-outcome/maximin
    # reference than a 50/50 split is? Ties are tracked separately so they are not
    # misread as either equity/maximin success or equal-split success.
    if abs(outcome_distance_difference) <= outcome_tolerance:
        closer_true_equal_outcome = 0.0
        closer_equal_split = 0.0
        tie = 1.0
    elif outcome_distance_to_true_equal < equal_split_outcome_distance_to_true_equal:
        closer_true_equal_outcome = 1.0
        closer_equal_split = 0.0
        tie = 0.0
    else:
        closer_true_equal_outcome = 0.0
        closer_equal_split = 1.0
        tie = 0.0

    true_outcome_near_feasible_equal = (
        1.0 if outcome_distance_to_true_equal <= outcome_tolerance else 0.0
    )
    true_equal_allocation_close = 1.0 if true_equal_allocation_gap <= allocation_tolerance else 0.0

    return {
        "realized_outcome_gap": realized_gap,
        "equal_split_realized_outcome_gap": equal_split_gap,
        "true_equal_outcome_solution_gap": true_equal_solution_gap,
        "true_equal_outcome_allocation": true_equal_allocation,
        "true_equal_outcome_allocation_gap": true_equal_allocation_gap,
        "true_equal_outcome": true_outcome_near_feasible_equal,
        "true_equal_outcome_allocation_close": true_equal_allocation_close,
        "true_outcome_gap_reduction_vs_equal_split": equal_split_gap - realized_gap,
        "outcome_distance_to_true_equal": outcome_distance_to_true_equal,
        "equal_split_outcome_distance_to_true_equal": equal_split_outcome_distance_to_true_equal,
        "allocation_distance_to_true_equal_minus_equal_split": allocation_distance_difference,
        "outcome_distance_to_true_equal_minus_equal_split": outcome_distance_difference,
        "closer_to_true_equal_outcome_than_equal_split": closer_true_equal_outcome,
        "closer_to_equal_split_than_true_equal_outcome": closer_equal_split,
        "true_outcome_classification_tie": tie,
    }


def compare_rr_to_heuristics_by_final_choice(
    environment_name: str,
    n_episodes: int = 200,
    config: Optional[EnvironmentConfig] = None,
    allocation_tolerance: float = 0.05,
    rr_policy: Optional[MetaPolicy] = None,
    evaluation_episodes: Optional[Sequence[EvaluationEpisode]] = None,
    use_common_observation_streams: bool = False,
    observations_per_person: int = 100,
) -> List[Dict[str, float | str]]:
    config = config or EnvironmentConfig()
    rr_policy = rr_policy or MyopicValueOfInformationPolicy(observation_draws=24)
    episodes = ensure_evaluation_episodes(
        config=config,
        n_episodes=n_episodes,
        evaluation_episodes=evaluation_episodes,
        include_observation_streams=use_common_observation_streams,
        observations_per_person=observations_per_person,
    )
    heuristics = build_final_choice_heuristics()
    rows: List[Dict[str, float | str]] = []

    rr_utilities: List[float] = []
    rr_allocations: List[float] = []
    heuristic_utilities: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    heuristic_allocations: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    utility_gaps: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    allocation_gaps: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    allocation_matches: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    rr_realized_outcome_gaps: List[float] = []
    rr_true_equal_allocation_gaps: List[float] = []
    rr_true_equal_outcome_rates: List[float] = []
    rr_outcome_distances_to_true_equal: List[float] = []
    heuristic_realized_outcome_gaps: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    heuristic_true_equal_allocation_gaps: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    heuristic_true_equal_outcome_rates: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    heuristic_outcome_distances_to_true_equal: Dict[str, List[float]] = {
        policy.name: [] for policy in heuristics
    }
    true_gap_improvements: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    true_outcome_distance_improvements: Dict[str, List[float]] = {
        policy.name: [] for policy in heuristics
    }

    for episode in episodes:
        rr_mdp = _mdp_for_episode(
            config,
            episode,
            seed_offset=0,
            use_common_observation_streams=use_common_observation_streams,
        )
        rr_result = rr_mdp.run_episode(rr_policy, true_state=episode.true_state)
        rr_utilities.append(rr_result.realized_utility)
        rr_allocations.append(rr_result.final_allocation_to_person1)
        rr_true_metrics = true_outcome_metrics_for_allocation(
            rr_mdp,
            episode.true_state,
            rr_result.final_belief,
            rr_result.final_allocation_to_person1,
            allocation_tolerance=allocation_tolerance,
        )
        rr_realized_outcome_gaps.append(rr_true_metrics["realized_outcome_gap"])
        rr_true_equal_allocation_gaps.append(rr_true_metrics["true_equal_outcome_allocation_gap"])
        rr_true_equal_outcome_rates.append(rr_true_metrics["true_equal_outcome"])
        rr_outcome_distances_to_true_equal.append(rr_true_metrics["outcome_distance_to_true_equal"])

        for policy in heuristics:
            heuristic_allocation, _ = rr_mdp.resolve_final_allocation(policy, rr_result.final_belief)
            heuristic_allocations[policy.name].append(heuristic_allocation)
            _, _, _, heuristic_realized_utility = rr_mdp.realized_utility(
                episode.true_state,
                heuristic_allocation,
                rr_result.final_belief,
            )
            heuristic_true_metrics = true_outcome_metrics_for_allocation(
                rr_mdp,
                episode.true_state,
                rr_result.final_belief,
                heuristic_allocation,
                allocation_tolerance=allocation_tolerance,
            )
            heuristic_utilities[policy.name].append(heuristic_realized_utility)
            utility_gaps[policy.name].append(rr_result.realized_utility - heuristic_realized_utility)
            heuristic_realized_outcome_gaps[policy.name].append(
                heuristic_true_metrics["realized_outcome_gap"]
            )
            heuristic_true_equal_allocation_gaps[policy.name].append(
                heuristic_true_metrics["true_equal_outcome_allocation_gap"]
            )
            heuristic_true_equal_outcome_rates[policy.name].append(
                heuristic_true_metrics["true_equal_outcome"]
            )
            heuristic_outcome_distances_to_true_equal[policy.name].append(
                heuristic_true_metrics["outcome_distance_to_true_equal"]
            )
            true_gap_improvements[policy.name].append(
                heuristic_true_metrics["realized_outcome_gap"]
                - rr_true_metrics["realized_outcome_gap"]
            )
            true_outcome_distance_improvements[policy.name].append(
                heuristic_true_metrics["outcome_distance_to_true_equal"]
                - rr_true_metrics["outcome_distance_to_true_equal"]
            )
            allocation_gap = abs(rr_result.final_allocation_to_person1 - heuristic_allocation)
            allocation_gaps[policy.name].append(allocation_gap)
            allocation_matches[policy.name].append(1.0 if allocation_gap <= allocation_tolerance else 0.0)

    rr_mean_utility = _mean(rr_utilities)

    for policy in heuristics:
        policy_name = policy.name
        policy_mean_utility = _mean(heuristic_utilities[policy_name])
        rows.append(
            {
                "environment": environment_name,
                "n_episodes": len(episodes),
                "rr_policy": rr_policy.name,
                "rr_policy_observation_draws": _policy_parameter(rr_policy, "observation_draws"),
                "common_true_states": 1.0,
                "common_observation_streams": 1.0 if use_common_observation_streams else 0.0,
                "heuristic": policy_name,
                "rr_mean_utility": rr_mean_utility,
                "rr_mean_utility_ci95": _ci95(rr_utilities),
                "heuristic_mean_utility": policy_mean_utility,
                "heuristic_mean_utility_ci95": _ci95(heuristic_utilities[policy_name]),
                "utility_gap_rr_minus_heuristic": _mean(utility_gaps[policy_name]),
                "utility_gap_ci95": _ci95(utility_gaps[policy_name]),
                "mean_abs_allocation_gap": _mean(allocation_gaps[policy_name]),
                "mean_abs_allocation_gap_ci95": _ci95(allocation_gaps[policy_name]),
                "rmse_allocation_gap": _rmse(allocation_gaps[policy_name]),
                "final_choice_match_rate": _mean(allocation_matches[policy_name]),
                "final_choice_match_rate_ci95": _ci95(allocation_matches[policy_name]),
                "rr_mean_allocation_to_person1": _mean(rr_allocations),
                "heuristic_mean_allocation_to_person1": _mean(heuristic_allocations[policy_name]),
                "rr_mean_realized_outcome_gap": _mean(rr_realized_outcome_gaps),
                "rr_mean_true_equal_outcome_allocation_gap": _mean(rr_true_equal_allocation_gaps),
                "rr_true_equal_outcome_rate": _mean(rr_true_equal_outcome_rates),
                "rr_mean_outcome_distance_to_true_equal": _mean(rr_outcome_distances_to_true_equal),
                "heuristic_mean_realized_outcome_gap": _mean(heuristic_realized_outcome_gaps[policy_name]),
                "heuristic_mean_true_equal_outcome_allocation_gap": _mean(
                    heuristic_true_equal_allocation_gaps[policy_name]
                ),
                "heuristic_true_equal_outcome_rate": _mean(
                    heuristic_true_equal_outcome_rates[policy_name]
                ),
                "heuristic_mean_outcome_distance_to_true_equal": _mean(
                    heuristic_outcome_distances_to_true_equal[policy_name]
                ),
                "realized_outcome_gap_heuristic_minus_rr": _mean(true_gap_improvements[policy_name]),
                "realized_outcome_gap_heuristic_minus_rr_ci95": _ci95(true_gap_improvements[policy_name]),
                "outcome_distance_to_true_equal_heuristic_minus_rr": _mean(
                    true_outcome_distance_improvements[policy_name]
                ),
                "outcome_distance_to_true_equal_heuristic_minus_rr_ci95": _ci95(
                    true_outcome_distance_improvements[policy_name]
                ),
            }
        )

    return rows


def compare_rr_information_acquisition_to_heuristics(
    environment_name: str,
    n_episodes: int = 200,
    config: Optional[EnvironmentConfig] = None,
    rr_policy: Optional[MetaPolicy] = None,
    evaluation_episodes: Optional[Sequence[EvaluationEpisode]] = None,
    use_common_observation_streams: bool = False,
    observations_per_person: int = 100,
) -> List[Dict[str, float | str]]:
    config = config or EnvironmentConfig()
    rr_policy = rr_policy or MyopicValueOfInformationPolicy(observation_draws=24)
    episodes = ensure_evaluation_episodes(
        config=config,
        n_episodes=n_episodes,
        evaluation_episodes=evaluation_episodes,
        include_observation_streams=use_common_observation_streams,
        observations_per_person=observations_per_person,
    )
    heuristics = build_policy_library()
    rows: List[Dict[str, float | str]] = []

    rr_utilities: List[float] = []
    rr_sample_counts: List[int] = []
    heuristic_utilities: Dict[str, List[float]] = {policy.name: [] for policy in heuristics}
    heuristic_sample_counts: Dict[str, List[int]] = {policy.name: [] for policy in heuristics}

    for episode in episodes:
        rr_mdp = _mdp_for_episode(
            config,
            episode,
            seed_offset=0,
            use_common_observation_streams=use_common_observation_streams,
        )
        rr_result = rr_mdp.run_episode(rr_policy, true_state=episode.true_state)
        rr_utilities.append(rr_result.realized_utility)
        rr_sample_counts.append(len(rr_result.samples))

        for policy_index, policy in enumerate(heuristics):
            heuristic_mdp = _mdp_for_episode(
                config,
                episode,
                seed_offset=(policy_index + 1) * 1000,
                use_common_observation_streams=use_common_observation_streams,
            )
            heuristic_result = heuristic_mdp.run_episode(policy, true_state=episode.true_state)
            heuristic_utilities[policy.name].append(heuristic_result.realized_utility)
            heuristic_sample_counts[policy.name].append(len(heuristic_result.samples))

    rr_mean_utility = _mean(rr_utilities)
    rr_mean_samples = _mean(rr_sample_counts)

    for policy in heuristics:
        policy_name = policy.name
        utility_gaps = [
            rr_utility - heuristic_utility
            for rr_utility, heuristic_utility in zip(rr_utilities, heuristic_utilities[policy_name])
        ]
        rows.append(
            {
                "environment": environment_name,
                "n_episodes": len(episodes),
                "rr_policy": rr_policy.name,
                "rr_policy_observation_draws": _policy_parameter(rr_policy, "observation_draws"),
                "common_true_states": 1.0,
                "common_observation_streams": 1.0 if use_common_observation_streams else 0.0,
                "heuristic": policy_name,
                "rr_mean_utility": rr_mean_utility,
                "rr_mean_utility_ci95": _ci95(rr_utilities),
                "heuristic_mean_utility": _mean(heuristic_utilities[policy_name]),
                "heuristic_mean_utility_ci95": _ci95(heuristic_utilities[policy_name]),
                "utility_gap_rr_minus_heuristic": _mean(utility_gaps),
                "utility_gap_ci95": _ci95(utility_gaps),
                "rr_mean_sample_count": rr_mean_samples,
                "heuristic_mean_sample_count": _mean(heuristic_sample_counts[policy_name]),
            }
        )
    return rows


def _sample_action_count(result: EpisodeResult, action_code: float) -> int:
    return sum(1 for sample in result.samples if sample["action"] == action_code)


def _safe_share(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _behavior_indicators(
    mdp: ContinuousAllocationMetaMDP,
    result: EpisodeResult,
    allocation_tolerance: float,
    gap_threshold: float,
    small_sample_threshold: int,
) -> Dict[str, float]:
    sample_1_count = _sample_action_count(result, 1.0)
    sample_2_count = _sample_action_count(result, 2.0)
    sample_count = sample_1_count + sample_2_count
    final_gap = result.final_belief.mean_1 - result.final_belief.mean_2
    allocation = result.final_allocation_to_person1

    if final_gap > mdp.config.equal_perception_tolerance:
        all_to_greatest_need = allocation >= 1.0 - allocation_tolerance
    elif final_gap < -mdp.config.equal_perception_tolerance:
        all_to_greatest_need = allocation <= allocation_tolerance
    else:
        all_to_greatest_need = False

    equal_outcome_allocation = mdp.final_allocation_equal_outcome(result.final_belief)
    equal_outcome_gap = abs(allocation - equal_outcome_allocation)
    true_metrics = true_outcome_metrics_for_allocation(
        mdp,
        result.true_state,
        result.final_belief,
        allocation,
        allocation_tolerance=allocation_tolerance,
    )
    threshold_gap_stop = (
        bool(result.actions)
        and result.actions[-1] == mdp.TERMINATE
        and sample_count > 0
        and abs(final_gap) >= gap_threshold
    )

    return {
        "utility": result.realized_utility,
        "sample_count": float(sample_count),
        "sample_1_count": float(sample_1_count),
        "sample_2_count": float(sample_2_count),
        "person1_sampling_share": _safe_share(sample_1_count, sample_count),
        "abs_sample_count_difference": float(abs(sample_1_count - sample_2_count)),
        "immediate_termination": 1.0 if result.actions and result.actions[0] == mdp.TERMINATE else 0.0,
        "small_information": 1.0 if sample_count <= small_sample_threshold else 0.0,
        "preferential_sample_1": 1.0 if sample_1_count > sample_2_count else 0.0,
        "preferential_sample_2": 1.0 if sample_2_count > sample_1_count else 0.0,
        "near_equal_allocation": 1.0 if abs(allocation - 0.5) <= allocation_tolerance else 0.0,
        "abs_allocation_from_equal": abs(allocation - 0.5),
        "all_to_greatest_need": 1.0 if all_to_greatest_need else 0.0,
        "equal_outcome": 1.0 if equal_outcome_gap <= allocation_tolerance else 0.0,
        "equal_outcome_allocation_gap": equal_outcome_gap,
        "realized_outcome_gap": true_metrics["realized_outcome_gap"],
        "equal_split_realized_outcome_gap": true_metrics["equal_split_realized_outcome_gap"],
        "true_equal_outcome_solution_gap": true_metrics["true_equal_outcome_solution_gap"],
        "true_equal_outcome_allocation": true_metrics["true_equal_outcome_allocation"],
        "true_equal_outcome_allocation_gap": true_metrics["true_equal_outcome_allocation_gap"],
        "true_equal_outcome": true_metrics["true_equal_outcome"],
        "true_equal_outcome_allocation_close": true_metrics["true_equal_outcome_allocation_close"],
        "true_outcome_gap_reduction_vs_equal_split": true_metrics["true_outcome_gap_reduction_vs_equal_split"],
        "outcome_distance_to_true_equal": true_metrics["outcome_distance_to_true_equal"],
        "equal_split_outcome_distance_to_true_equal": true_metrics[
            "equal_split_outcome_distance_to_true_equal"
        ],
        "allocation_distance_to_true_equal_minus_equal_split": true_metrics[
            "allocation_distance_to_true_equal_minus_equal_split"
        ],
        "outcome_distance_to_true_equal_minus_equal_split": true_metrics[
            "outcome_distance_to_true_equal_minus_equal_split"
        ],
        "closer_to_true_equal_outcome_than_equal_split": true_metrics[
            "closer_to_true_equal_outcome_than_equal_split"
        ],
        "closer_to_equal_split_than_true_equal_outcome": true_metrics[
            "closer_to_equal_split_than_true_equal_outcome"
        ],
        "true_outcome_classification_tie": true_metrics["true_outcome_classification_tie"],
        "threshold_gap_stop": 1.0 if threshold_gap_stop else 0.0,
        "final_belief_need_gap": final_gap,
        "abs_final_belief_need_gap": abs(final_gap),
    }


def compare_policy_behavior_profiles(
    environment_name: str,
    n_episodes: int = 200,
    config: Optional[EnvironmentConfig] = None,
    rr_policy: Optional[MetaPolicy] = None,
    policies: Optional[Sequence[MetaPolicy]] = None,
    allocation_tolerance: float = 0.05,
    gap_threshold: float = 10.0,
    small_sample_threshold: int = 1,
    evaluation_episodes: Optional[Sequence[EvaluationEpisode]] = None,
    use_common_observation_streams: bool = False,
    observations_per_person: int = 100,
) -> List[Dict[str, float | str]]:
    config = config or EnvironmentConfig()
    rr_policy = rr_policy or MyopicValueOfInformationPolicy(observation_draws=24)
    heuristic_policies = list(build_policy_library() if policies is None else policies)
    all_policies: List[tuple[str, MetaPolicy]] = [("rr_approximation", rr_policy)] + [
        ("heuristic", policy) for policy in heuristic_policies
    ]
    episodes = ensure_evaluation_episodes(
        config=config,
        n_episodes=n_episodes,
        evaluation_episodes=evaluation_episodes,
        include_observation_streams=use_common_observation_streams,
        observations_per_person=observations_per_person,
    )
    metrics_by_policy: Dict[str, List[Dict[str, float]]] = {
        policy.name: [] for _, policy in all_policies
    }
    role_by_policy = {policy.name: role for role, policy in all_policies}

    for episode in episodes:
        for policy_index, (_, policy) in enumerate(all_policies):
            policy_mdp = _mdp_for_episode(
                config,
                episode,
                seed_offset=policy_index * 1000,
                use_common_observation_streams=use_common_observation_streams,
            )
            result = policy_mdp.run_episode(policy, true_state=episode.true_state)
            metrics_by_policy[policy.name].append(
                _behavior_indicators(
                    policy_mdp,
                    result,
                    allocation_tolerance=allocation_tolerance,
                    gap_threshold=gap_threshold,
                    small_sample_threshold=small_sample_threshold,
                )
            )

    rows: List[Dict[str, float | str]] = []
    for policy_name, metric_rows in metrics_by_policy.items():
        def metric_values(key: str) -> List[float]:
            return [row[key] for row in metric_rows if not math.isnan(row[key])]

        rows.append(
            {
                "environment": environment_name,
                "n_episodes": len(episodes),
                "policy": policy_name,
                "policy_type": role_by_policy[policy_name],
                "common_true_states": 1.0,
                "common_observation_streams": 1.0 if use_common_observation_streams else 0.0,
                "mean_utility": _mean(metric_values("utility")),
                "mean_utility_ci95": _ci95(metric_values("utility")),
                "immediate_termination_rate": _mean(metric_values("immediate_termination")),
                "small_information_rate": _mean(metric_values("small_information")),
                "mean_sample_count": _mean(metric_values("sample_count")),
                "mean_sample_1_count": _mean(metric_values("sample_1_count")),
                "mean_sample_2_count": _mean(metric_values("sample_2_count")),
                "mean_person1_sampling_share": _mean(metric_values("person1_sampling_share")),
                "preferential_sample_1_rate": _mean(metric_values("preferential_sample_1")),
                "preferential_sample_2_rate": _mean(metric_values("preferential_sample_2")),
                "mean_abs_sample_count_difference": _mean(metric_values("abs_sample_count_difference")),
                "near_equal_allocation_rate": _mean(metric_values("near_equal_allocation")),
                "mean_abs_allocation_from_equal": _mean(metric_values("abs_allocation_from_equal")),
                "all_to_greatest_need_rate": _mean(metric_values("all_to_greatest_need")),
                "equal_outcome_rate": _mean(metric_values("equal_outcome")),
                "mean_equal_outcome_allocation_gap": _mean(metric_values("equal_outcome_allocation_gap")),
                "true_equal_outcome_rate": _mean(metric_values("true_equal_outcome")),
                "true_equal_outcome_allocation_close_rate": _mean(
                    metric_values("true_equal_outcome_allocation_close")
                ),
                "mean_realized_outcome_gap": _mean(metric_values("realized_outcome_gap")),
                "mean_equal_split_realized_outcome_gap": _mean(metric_values("equal_split_realized_outcome_gap")),
                "mean_true_equal_outcome_solution_gap": _mean(metric_values("true_equal_outcome_solution_gap")),
                "mean_true_equal_outcome_allocation": _mean(metric_values("true_equal_outcome_allocation")),
                "mean_true_equal_outcome_allocation_gap": _mean(
                    metric_values("true_equal_outcome_allocation_gap")
                ),
                "mean_true_outcome_gap_reduction_vs_equal_split": _mean(
                    metric_values("true_outcome_gap_reduction_vs_equal_split")
                ),
                "mean_outcome_distance_to_true_equal": _mean(
                    metric_values("outcome_distance_to_true_equal")
                ),
                "mean_equal_split_outcome_distance_to_true_equal": _mean(
                    metric_values("equal_split_outcome_distance_to_true_equal")
                ),
                "mean_allocation_distance_to_true_equal_minus_equal_split": _mean(
                    metric_values("allocation_distance_to_true_equal_minus_equal_split")
                ),
                "mean_outcome_distance_to_true_equal_minus_equal_split": _mean(
                    metric_values("outcome_distance_to_true_equal_minus_equal_split")
                ),
                "closer_to_true_equal_outcome_than_equal_split_rate": _mean(
                    metric_values("closer_to_true_equal_outcome_than_equal_split")
                ),
                "closer_to_equal_split_than_true_equal_outcome_rate": _mean(
                    metric_values("closer_to_equal_split_than_true_equal_outcome")
                ),
                "true_outcome_classification_tie_rate": _mean(
                    metric_values("true_outcome_classification_tie")
                ),
                "threshold_gap_stop_rate": _mean(metric_values("threshold_gap_stop")),
                "mean_final_belief_need_gap": _mean(metric_values("final_belief_need_gap")),
                "mean_abs_final_belief_need_gap": _mean(metric_values("abs_final_belief_need_gap")),
            }
        )
    return rows


def summarize_rr_regimes(
    environment_names: Sequence[str],
    config_builder,
    n_episodes: int = 200,
    allocation_tolerance: float = 0.05,
    rr_policy: Optional[MetaPolicy] = None,
    use_common_observation_streams: bool = False,
    observations_per_person: int = 100,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for environment_name in environment_names:
        rows.extend(
            compare_rr_to_heuristics_by_final_choice(
                environment_name=environment_name,
                config=config_builder(environment_name),
                n_episodes=n_episodes,
                allocation_tolerance=allocation_tolerance,
                rr_policy=rr_policy,
                use_common_observation_streams=use_common_observation_streams,
                observations_per_person=observations_per_person,
            )
        )
    return rows


def compare_rr_approximation_methods(
    environment_name: str,
    n_episodes: int = 100,
    config: Optional[EnvironmentConfig] = None,
    policies: Optional[Sequence[MetaPolicy]] = None,
    evaluation_episodes: Optional[Sequence[EvaluationEpisode]] = None,
    use_common_observation_streams: bool = False,
    observations_per_person: int = 100,
) -> List[Dict[str, float | str]]:
    config = config or EnvironmentConfig()
    policies = list(policies or [
        MyopicValueOfInformationPolicy(observation_draws=16),
        BlinkeredPolicy(horizon=2, observation_draws=6),
        DiscretizedDynamicProgrammingPolicy(max_samples=2, mean_grid_size=7, observation_branches=3),
    ])
    episodes = ensure_evaluation_episodes(
        config=config,
        n_episodes=n_episodes,
        evaluation_episodes=evaluation_episodes,
        include_observation_streams=use_common_observation_streams,
        observations_per_person=observations_per_person,
    )
    policy_utilities: Dict[str, List[float]] = {policy.name: [] for policy in policies}
    policy_samples: Dict[str, List[int]] = {policy.name: [] for policy in policies}
    policy_allocations: Dict[str, List[float]] = {policy.name: [] for policy in policies}

    for episode in episodes:
        for policy_index, policy in enumerate(policies):
            policy_mdp = _mdp_for_episode(
                config,
                episode,
                seed_offset=(policy_index + 1) * 1000,
                use_common_observation_streams=use_common_observation_streams,
            )
            result = policy_mdp.run_episode(policy, true_state=episode.true_state)
            policy_utilities[policy.name].append(result.realized_utility)
            policy_samples[policy.name].append(len(result.samples))
            policy_allocations[policy.name].append(result.final_allocation_to_person1)

    best_mean_utility = max(_mean(values) for values in policy_utilities.values())
    rows: List[Dict[str, float | str]] = []
    for policy in policies:
        policy_name = policy.name
        mean_utility = _mean(policy_utilities[policy_name])
        rows.append(
            {
                "environment": environment_name,
                "n_episodes": len(episodes),
                "policy": policy_name,
                "policy_observation_draws": _policy_parameter(policy, "observation_draws"),
                "policy_horizon": _policy_parameter(policy, "horizon"),
                "policy_max_samples": _policy_parameter(policy, "max_samples"),
                "policy_mean_grid_size": _policy_parameter(policy, "mean_grid_size"),
                "policy_observation_branches": _policy_parameter(policy, "observation_branches"),
                "common_true_states": 1.0,
                "common_observation_streams": 1.0 if use_common_observation_streams else 0.0,
                "mean_utility": mean_utility,
                "mean_utility_ci95": _ci95(policy_utilities[policy_name]),
                "regret_vs_best_rr_approximation": best_mean_utility - mean_utility,
                "mean_sample_count": _mean(policy_samples[policy_name]),
                "mean_allocation_to_person1": _mean(policy_allocations[policy_name]),
            }
        )
    return rows
