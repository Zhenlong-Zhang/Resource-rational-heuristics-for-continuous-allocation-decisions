from __future__ import annotations

from typing import Callable, Dict, Hashable, Mapping, Optional

try:
    from ..mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP, EnvironmentConfig, EpisodeResult, MetaPolicy
    from ..policies.voi import MyopicValueOfInformationPolicy
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP, EnvironmentConfig, EpisodeResult, MetaPolicy
    from policies.voi import MyopicValueOfInformationPolicy


BeliefKeyFn = Callable[[BeliefState], Hashable]


def rounded_belief_key(belief: BeliefState, digits: int = 2) -> tuple[float, float, float, float, float]:
    return (
        round(belief.mean_1, digits),
        round(belief.var_1, digits),
        round(belief.mean_2, digits),
        round(belief.var_2, digits),
        round(belief.deliberation_time, digits),
    )


class BeliefActionDictionaryPolicy:
    name = "belief_action_dictionary"

    def __init__(
        self,
        action_by_belief: Mapping[Hashable, Action],
        default_action: Action = ContinuousAllocationMetaMDP.TERMINATE,
        key_fn: Optional[BeliefKeyFn] = None,
    ):
        self.action_by_belief = dict(action_by_belief)
        self.default_action = default_action
        self.key_fn = key_fn or rounded_belief_key

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return self.action_by_belief.get(self.key_fn(belief), self.default_action)


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
