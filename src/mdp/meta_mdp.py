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
    need_threshold: float = 100.0
    terminate_cost: float = 1.0
    equal_perception_tolerance: float = 1e-6
    sample_cost_person2: float = 1.0
    sample_cost_person1_equal: float = 1.0
    sample_cost_person1_unequal: float = 2.0
    allocation_grid_size: int = 201
    expected_utility_draws: int = 2000
    max_workers: Optional[int] = None
    random_seed: Optional[int] = None


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


def utility(outcome_minus_need: float, lambda_shortfall: float) -> float:
    if outcome_minus_need < 0:
        return -lambda_shortfall * math.sqrt(abs(outcome_minus_need))
    return math.sqrt(outcome_minus_need)


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

    def initial_belief(self) -> BeliefState:
        prior_var = self.config.sigma_need ** 2
        return BeliefState(
            mean_1=self.config.mu_need,
            var_1=prior_var,
            mean_2=self.config.mu_need,
            var_2=prior_var,
        )

    def perceived_equal_need(self, belief: BeliefState) -> bool:
        return abs(belief.mean_1 - belief.mean_2) <= self.config.equal_perception_tolerance

    def sample_cost(self, action: Action, belief: BeliefState) -> float:
        if action == self.SAMPLE_PERSON_2:
            return self.config.sample_cost_person2
        if action == self.SAMPLE_PERSON_1:
            return (
                self.config.sample_cost_person1_equal
                if self.perceived_equal_need(belief)
                else self.config.sample_cost_person1_unequal
            )
        if action == self.TERMINATE:
            return self.config.terminate_cost
        raise ValueError(f"Unknown action: {action}")

    def remaining_time_after_termination(self, belief: BeliefState) -> float:
        return max(0.0, self.config.total_time - belief.deliberation_time - self.config.terminate_cost)

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
            next_belief.history.append({"action": 0.0, "observation": math.nan, "cost": cost})
            return next_belief

        raise ValueError(f"Unknown action: {action}")

    def available_actions(self, belief: BeliefState) -> List[Action]:
        if belief.deliberation_time + self.config.terminate_cost > self.config.total_time:
            return []

        actions = [self.TERMINATE]
        if belief.deliberation_time + self.sample_cost(self.SAMPLE_PERSON_1, belief) + self.config.terminate_cost <= self.config.total_time:
            actions.append(self.SAMPLE_PERSON_1)
        if belief.deliberation_time + self.sample_cost(self.SAMPLE_PERSON_2, belief) + self.config.terminate_cost <= self.config.total_time:
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
        amount_1 = allocation_to_person1 * remaining_time
        amount_2 = (1.0 - allocation_to_person1) * remaining_time
        utilities = [
            utility(amount_1 - n1, self.config.lambda_shortfall)
            + utility(amount_2 - n2, self.config.lambda_shortfall)
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
        if (
            abs(belief.mean_1 - belief.mean_2) <= self.config.equal_perception_tolerance
            and abs(belief.var_1 - belief.var_2) <= self.config.equal_perception_tolerance
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

    def realized_utility(
        self,
        true_state: TrueState,
        allocation_to_person1: float,
        belief: BeliefState,
    ) -> Tuple[float, float, float, float]:
        remaining_time = self.remaining_time_after_termination(belief)
        amount_1 = allocation_to_person1 * remaining_time
        amount_2 = (1.0 - allocation_to_person1) * remaining_time
        utility_1 = utility(amount_1 - true_state.need_1, self.config.lambda_shortfall)
        utility_2 = utility(amount_2 - true_state.need_2, self.config.lambda_shortfall)
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
            available = self.available_actions(belief)
            if not available:
                action = self.TERMINATE
            else:
                action = policy.choose_action(self, belief)
                if action not in available:
                    action = self.TERMINATE
            actions.append(action)

            if action == self.TERMINATE:
                belief = self.transition(belief, true_state, action)
                terminated = True
                break

            pre_len = len(belief.history)
            belief = self.transition(belief, true_state, action)
            if len(belief.history) > pre_len:
                samples.append(belief.history[-1])

        if not terminated:
            belief = self.transition(belief, true_state, self.TERMINATE)
            actions.append(self.TERMINATE)

        allocation_to_person1, _ = self.solve_terminal_allocation(belief)
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
