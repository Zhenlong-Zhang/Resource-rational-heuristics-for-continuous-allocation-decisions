from __future__ import annotations

from typing import Dict, Optional

from mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig, EpisodeResult, MetaPolicy
from policies.voi import MyopicValueOfInformationPolicy


def run_single_episode(
    config: Optional[EnvironmentConfig] = None,
    policy: Optional[MetaPolicy] = None,
    max_steps: int = 100,
) -> EpisodeResult:
    mdp = ContinuousAllocationMetaMDP(config or EnvironmentConfig())
    policy = policy or MyopicValueOfInformationPolicy(observation_draws=16)
    return mdp.run_episode(policy=policy, max_steps=max_steps)


def simulate_many_episodes(
    config: Optional[EnvironmentConfig] = None,
    policy: Optional[MetaPolicy] = None,
    n_episodes: int = 20,
) -> Dict[str, float]:
    mdp = ContinuousAllocationMetaMDP(config or EnvironmentConfig())
    policy = policy or MyopicValueOfInformationPolicy(observation_draws=16)
    results = [mdp.run_episode(policy=policy) for _ in range(n_episodes)]
    utilities = [r.realized_utility for r in results]
    deliberations = [r.final_belief.deliberation_time for r in results]
    allocations = [r.final_allocation_to_person1 for r in results]
    remaining_times = [r.remaining_time for r in results]
    import statistics

    return {
        "policy": policy.name,
        "mean_utility": float(statistics.mean(utilities)),
        "std_utility": float(statistics.pstdev(utilities)) if len(utilities) > 1 else 0.0,
        "mean_deliberation_time": float(statistics.mean(deliberations)),
        "mean_allocation_to_person1": float(statistics.mean(allocations)),
        "mean_remaining_time": float(statistics.mean(remaining_times)),
        "termination_rate": 1.0,
    }


def episode_to_dict(result: EpisodeResult) -> Dict[str, object]:
    return {
        "true_need_1": result.true_state.need_1,
        "true_need_2": result.true_state.need_2,
        "actions": result.actions,
        "samples": result.samples,
        "final_allocation_to_person1": result.final_allocation_to_person1,
        "final_resource_person1": result.final_resource_person1,
        "final_resource_person2": result.final_resource_person2,
        "remaining_time": result.remaining_time,
        "realized_utility": result.realized_utility,
    }
