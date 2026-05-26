from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import math
import os
import random
import statistics


Action = str


@dataclass(frozen=True)
class EnvironmentConfig:
    mu_need: float = 100.0
    sigma_need: float = 20.0
    sigma_sample: float = 10.0
    total_time: float = 60.0
    lambda_shortfall: float = 2.0
    utility_exponent: float = 0.5
    alpha: Optional[float] = None
    learning_per_unit_of_tutoring: float = 1.0
    delta_learning_per_unit_tutoring: float = 0.0
    need_threshold: float = 100.0
    terminate_cost: float = 1.0
    sample_time_cost: float = 1.0
    equal_perception_tolerance: float = 1e-6
    allocation_grid_size: int = 201
    expected_utility_draws: int = 2000
    max_workers: Optional[int] = None
    initial_mean_1: Optional[float] = None
    initial_mean_2: Optional[float] = None
    initial_var_1: Optional[float] = None
    initial_var_2: Optional[float] = None
    prior_sample_count_1: int = 0
    prior_sample_count_2: int = 0
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.learning_per_unit_of_tutoring <= 0:
            raise ValueError("learning_per_unit_of_tutoring must be positive")
        if self.learning_per_unit_of_tutoring - self.delta_learning_per_unit_tutoring <= 0:
            raise ValueError(
                "learning_per_unit_of_tutoring - delta_learning_per_unit_tutoring must be positive"
            )


@dataclass(frozen=True)
class TrueState:
    need_1: float
    need_2: float


@dataclass
class BeliefState:
    mean_1: float
    var_1: float
    mean_2: float
    var_2: float
    deliberation_time: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)

    def copy(self) -> "BeliefState":
        return BeliefState(
            mean_1=self.mean_1,
            var_1=self.var_1,
            mean_2=self.mean_2,
            var_2=self.var_2,
            deliberation_time=self.deliberation_time,
            history=list(self.history),
        )


@dataclass
class EpisodeResult:
    true_state: TrueState
    final_belief: BeliefState
    actions: List[Action]
    samples: List[Dict[str, float]]
    final_allocation_to_person1: float
    final_resource_person1: float
    final_resource_person2: float
    remaining_time: float
    realized_utility: float
    terminated: bool


class MetaPolicy(Protocol):
    name: str

    def choose_action(self, mdp: "ContinuousAllocationMetaMDP", belief: BeliefState) -> Action:
        ...

    def choose_final_allocation(self, mdp: "ContinuousAllocationMetaMDP", belief: BeliefState) -> Optional[float]:
        ...


def utility(outcome_minus_need: float, lambda_shortfall: float, exponent: float = 0.5) -> float:
    if outcome_minus_need < 0:
        return -lambda_shortfall * (abs(outcome_minus_need) ** exponent)
    return outcome_minus_need ** exponent


class ContinuousAllocationMetaMDP:
    TERMINATE: Action = "terminate"
    SAMPLE_PERSON_1: Action = "sample_1"
    SAMPLE_PERSON_2: Action = "sample_2"

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)

    @staticmethod
    def default_max_workers() -> int:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - 1)

    def sample_true_state(self) -> TrueState:
        return TrueState(
            need_1=self.rng.gauss(self.config.mu_need, self.config.sigma_need),
            need_2=self.rng.gauss(self.config.mu_need, self.config.sigma_need),
        )

    def _prior_variance_after_prior_samples(self, prior_sample_count: int) -> float:
        prior_precision = 1.0 / (self.config.sigma_need ** 2)
        sample_precision = prior_sample_count / (self.config.sigma_sample ** 2)
        return 1.0 / (prior_precision + sample_precision)

    def initial_belief(self) -> BeliefState:
        prior_var_1 = self._prior_variance_after_prior_samples(self.config.prior_sample_count_1)
        prior_var_2 = self._prior_variance_after_prior_samples(self.config.prior_sample_count_2)
        return BeliefState(
            mean_1=self.config.initial_mean_1 if self.config.initial_mean_1 is not None else self.config.mu_need,
            var_1=self.config.initial_var_1 if self.config.initial_var_1 is not None else prior_var_1,
            mean_2=self.config.initial_mean_2 if self.config.initial_mean_2 is not None else self.config.mu_need,
            var_2=self.config.initial_var_2 if self.config.initial_var_2 is not None else prior_var_2,
        )

    def perceived_equal_need(self, belief: BeliefState) -> bool:
        return abs(belief.mean_1 - belief.mean_2) <= self.config.equal_perception_tolerance

    def sample_cost(self, action: Action, belief: BeliefState) -> float:
        if action in (self.SAMPLE_PERSON_1, self.SAMPLE_PERSON_2):
            return self.config.sample_time_cost
        if action == self.TERMINATE:
            return self.config.terminate_cost
        raise ValueError(f"Unknown action: {action}")

    def terminal_action_cost(self, policy: Optional[MetaPolicy] = None, belief: Optional[BeliefState] = None) -> float:
        cost = getattr(policy, "termination_time_cost", None)
        if callable(cost):
            return float(cost(self, belief))
        if cost is not None:
            return float(cost)
        return self.config.terminate_cost

    @staticmethod
    def belief_already_terminated(belief: BeliefState) -> bool:
        return bool(belief.history and belief.history[-1]["action"] == 0.0)

    def remaining_time_after_termination(
        self,
        belief: BeliefState,
        policy: Optional[MetaPolicy] = None,
    ) -> float:
        remaining = self.config.total_time - belief.deliberation_time
        if not self.belief_already_terminated(belief):
            remaining -= self.terminal_action_cost(policy, belief)
        return max(0.0, remaining)

    def utility_exponent(self) -> float:
        return self.config.alpha if self.config.alpha is not None else self.config.utility_exponent

    def learning_rates(self) -> Tuple[float, float]:
        rate_1 = self.config.learning_per_unit_of_tutoring
        rate_2 = self.config.learning_per_unit_of_tutoring - self.config.delta_learning_per_unit_tutoring
        return rate_1, rate_2

    def allocation_to_learning_outcomes(
        self,
        allocation_to_person1: float,
        remaining_time: float,
    ) -> Tuple[float, float]:
        rate_1, rate_2 = self.learning_rates()
        tutoring_time_1 = allocation_to_person1 * remaining_time
        tutoring_time_2 = (1.0 - allocation_to_person1) * remaining_time
        return rate_1 * tutoring_time_1, rate_2 * tutoring_time_2

    def posterior_update(self, mean: float, var: float, observation: float) -> Tuple[float, float]:
        obs_var = self.config.sigma_sample ** 2
        prior_precision = 1.0 / var
        obs_precision = 1.0 / obs_var
        post_var = 1.0 / (prior_precision + obs_precision)
        post_mean = post_var * (prior_precision * mean + obs_precision * observation)
        return post_mean, post_var

    def observe(self, true_state: TrueState, action: Action) -> float:
        if action == self.SAMPLE_PERSON_1:
            return float(self.rng.gauss(true_state.need_1, self.config.sigma_sample))
        if action == self.SAMPLE_PERSON_2:
            return float(self.rng.gauss(true_state.need_2, self.config.sigma_sample))
        raise ValueError(f"Action {action} does not produce an observation")

    def transition(self, belief: BeliefState, true_state: TrueState, action: Action) -> BeliefState:
        next_belief = belief.copy()
        cost = self.sample_cost(action, belief)
        next_belief.deliberation_time += cost

        if action == self.SAMPLE_PERSON_1:
            obs = self.observe(true_state, action)
            next_belief.mean_1, next_belief.var_1 = self.posterior_update(
                next_belief.mean_1, next_belief.var_1, obs
            )
            next_belief.history.append({"action": 1.0, "observation": obs, "cost": cost})
            return next_belief

        if action == self.SAMPLE_PERSON_2:
            obs = self.observe(true_state, action)
            next_belief.mean_2, next_belief.var_2 = self.posterior_update(
                next_belief.mean_2, next_belief.var_2, obs
            )
            next_belief.history.append({"action": 2.0, "observation": obs, "cost": cost})
            return next_belief

        if action == self.TERMINATE:
            return self.terminate_belief(belief)

        raise ValueError(f"Unknown action: {action}")

    def terminate_belief(self, belief: BeliefState, policy: Optional[MetaPolicy] = None) -> BeliefState:
        next_belief = belief.copy()
        cost = self.terminal_action_cost(policy, belief)
        next_belief.deliberation_time += cost
        next_belief.history.append({"action": 0.0, "observation": math.nan, "cost": cost})
        return next_belief

    def available_actions(self, belief: BeliefState, policy: Optional[MetaPolicy] = None) -> List[Action]:
        terminal_cost = self.terminal_action_cost(policy, belief)
        if belief.deliberation_time + terminal_cost > self.config.total_time:
            return []

        actions = [self.TERMINATE]
        if belief.deliberation_time + self.sample_cost(self.SAMPLE_PERSON_1, belief) + terminal_cost <= self.config.total_time:
            actions.append(self.SAMPLE_PERSON_1)
        if belief.deliberation_time + self.sample_cost(self.SAMPLE_PERSON_2, belief) + terminal_cost <= self.config.total_time:
            actions.append(self.SAMPLE_PERSON_2)
        return actions

    def draw_need_samples_from_belief(self, belief: BeliefState, draws: Optional[int] = None) -> Tuple[List[float], List[float]]:
        draws = draws or self.config.expected_utility_draws
        n1 = [self.rng.gauss(belief.mean_1, math.sqrt(belief.var_1)) for _ in range(draws)]
        n2 = [self.rng.gauss(belief.mean_2, math.sqrt(belief.var_2)) for _ in range(draws)]
        return n1, n2

    def expected_terminal_utility_from_samples(
        self,
        belief: BeliefState,
        allocation_to_person1: float,
        need_1_samples: List[float],
        need_2_samples: List[float],
    ) -> float:
        remaining_time = self.remaining_time_after_termination(belief)
        amount_1, amount_2 = self.allocation_to_learning_outcomes(allocation_to_person1, remaining_time)
        alpha = self.utility_exponent()
        utilities = [
            utility(amount_1 - n1, self.config.lambda_shortfall, alpha)
            + utility(amount_2 - n2, self.config.lambda_shortfall, alpha)
            for n1, n2 in zip(need_1_samples, need_2_samples)
        ]
        return float(statistics.mean(utilities))

    def expected_terminal_utility(self, belief: BeliefState, allocation_to_person1: float) -> float:
        need_1, need_2 = self.draw_need_samples_from_belief(belief)
        return self.expected_terminal_utility_from_samples(
            belief,
            allocation_to_person1,
            need_1,
            need_2,
        )

    def solve_terminal_allocation(self, belief: BeliefState) -> Tuple[float, float]:
        need_1_samples, need_2_samples = self.draw_need_samples_from_belief(belief)
        rate_1, rate_2 = self.learning_rates()
        if (
            abs(belief.mean_1 - belief.mean_2) <= self.config.equal_perception_tolerance
            and abs(belief.var_1 - belief.var_2) <= self.config.equal_perception_tolerance
            and abs(rate_1 - rate_2) <= self.config.equal_perception_tolerance
        ):
            symmetric_a = 0.5
            return symmetric_a, self.expected_terminal_utility_from_samples(
                belief,
                symmetric_a,
                need_1_samples,
                need_2_samples,
            )
        if self.config.allocation_grid_size <= 1:
            grid = [0.5]
        else:
            grid = [i / (self.config.allocation_grid_size - 1) for i in range(self.config.allocation_grid_size)]
        values = [
            self.expected_terminal_utility_from_samples(
                belief,
                a,
                need_1_samples,
                need_2_samples,
            )
            for a in grid
        ]
        best_index = max(range(len(values)), key=lambda idx: values[idx])
        return float(grid[best_index]), float(values[best_index])

    def final_allocation_equal_division(self, belief: BeliefState) -> float:
        return 0.5

    def final_allocation_all_to_greatest_need(self, belief: BeliefState) -> float:
        if belief.mean_1 > belief.mean_2:
            return 1.0
        if belief.mean_2 > belief.mean_1:
            return 0.0
        return 0.5

    def final_allocation_proportional_to_estimated_needs(self, belief: BeliefState) -> float:
        rate_1, rate_2 = self.learning_rates()
        time_needed_1 = belief.mean_1 / rate_1
        time_needed_2 = belief.mean_2 / rate_2
        total_estimated_time_needed = max(1e-9, time_needed_1 + time_needed_2)
        return min(1.0, max(0.0, time_needed_1 / total_estimated_time_needed))

    def final_allocation_rectify_then_split_equally(self, belief: BeliefState) -> float:
        remaining_time = self.remaining_time_after_termination(belief)
        if remaining_time <= 0:
            return 0.5
        rate_1, rate_2 = self.learning_rates()
        allocation = (belief.mean_1 - belief.mean_2 + rate_2 * remaining_time) / (
            (rate_1 + rate_2) * remaining_time
        )
        return min(1.0, max(0.0, allocation))

    def final_allocation_equal_outcome(self, belief: BeliefState) -> float:
        return self.final_allocation_rectify_then_split_equally(belief)

    def final_allocation_maximin(self, belief: BeliefState) -> float:
        return self.final_allocation_rectify_then_split_equally(belief)

    def resolve_final_allocation(self, policy: MetaPolicy, belief: BeliefState) -> Tuple[float, Optional[float]]:
        if hasattr(policy, "choose_final_allocation"):
            allocation = policy.choose_final_allocation(self, belief)
            if allocation is not None:
                return allocation, None
        allocation, expected_value = self.solve_terminal_allocation(belief)
        return allocation, expected_value

    def realized_utility(
        self,
        true_state: TrueState,
        allocation_to_person1: float,
        belief: BeliefState,
    ) -> Tuple[float, float, float, float]:
        remaining_time = self.remaining_time_after_termination(belief)
        amount_1, amount_2 = self.allocation_to_learning_outcomes(allocation_to_person1, remaining_time)
        alpha = self.utility_exponent()
        utility_1 = utility(amount_1 - true_state.need_1, self.config.lambda_shortfall, alpha)
        utility_2 = utility(amount_2 - true_state.need_2, self.config.lambda_shortfall, alpha)
        return amount_1, amount_2, remaining_time, utility_1 + utility_2

    def run_episode(
        self,
        policy: MetaPolicy,
        true_state: Optional[TrueState] = None,
        max_steps: int = 100,
    ) -> EpisodeResult:
        true_state = true_state or self.sample_true_state()
        belief = self.initial_belief()
        actions: List[Action] = []
        samples: List[Dict[str, float]] = []
        terminated = False

        for _ in range(max_steps):
            available = self.available_actions(belief, policy)
            if not available:
                action = self.TERMINATE
            else:
                action = policy.choose_action(self, belief)
                if action not in available:
                    action = self.TERMINATE
            actions.append(action)

            if action == self.TERMINATE:
                belief = self.terminate_belief(belief, policy)
                terminated = True
                break

            pre_len = len(belief.history)
            belief = self.transition(belief, true_state, action)
            if len(belief.history) > pre_len:
                samples.append(belief.history[-1])

        if not terminated:
            belief = self.terminate_belief(belief, policy)
            actions.append(self.TERMINATE)

        allocation_to_person1, _ = self.resolve_final_allocation(policy, belief)
        amount_1, amount_2, remaining_time, realized = self.realized_utility(
            true_state, allocation_to_person1, belief
        )
        return EpisodeResult(
            true_state=true_state,
            final_belief=belief,
            actions=actions,
            samples=samples,
            final_allocation_to_person1=allocation_to_person1,
            final_resource_person1=amount_1,
            final_resource_person2=amount_2,
            remaining_time=remaining_time,
            realized_utility=realized,
            terminated=True,
        )
