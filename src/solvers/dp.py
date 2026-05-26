from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List, Tuple

import math

try:
    from ..mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP, EnvironmentConfig
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP, EnvironmentConfig


@dataclass(frozen=True)
class DPState:
    mean_1: float
    mean_2: float
    samples_1: int
    samples_2: int


class FiniteHorizonDPSolver:
    """Small finite-horizon DP using the posterior precision grid suggested by Falk."""

    def __init__(
        self,
        mdp: ContinuousAllocationMetaMDP,
        max_samples: int = 2,
        mean_grid_size: int = 11,
        mean_grid_radius_sd: float = 3.0,
        observation_branches: int = 5,
    ):
        self.mdp = mdp
        self.max_samples = max_samples
        self.mean_grid_size = max(3, mean_grid_size)
        self.mean_grid_radius_sd = mean_grid_radius_sd
        self.observation_branches = max(1, observation_branches)
        self._value_cache: Dict[Tuple[DPState, int], Tuple[float, Action]] = {}
        self._terminal_cache: Dict[DPState, float] = {}
        self._mean_grid = self._build_mean_grid(mdp.config)

    @staticmethod
    def precision_step(config: EnvironmentConfig) -> float:
        return 1.0 / (config.sigma_sample ** 2)

    def _build_mean_grid(self, config: EnvironmentConfig) -> List[float]:
        low = config.mu_need - self.mean_grid_radius_sd * config.sigma_need
        high = config.mu_need + self.mean_grid_radius_sd * config.sigma_need
        if self.mean_grid_size == 1:
            return [config.mu_need]
        return [
            low + i * (high - low) / (self.mean_grid_size - 1)
            for i in range(self.mean_grid_size)
        ]

    def _snap_mean(self, mean: float) -> float:
        return min(self._mean_grid, key=lambda grid_mean: abs(grid_mean - mean))

    def variance_for_sample_count(self, sample_count: int) -> float:
        prior_precision = 1.0 / (self.mdp.config.sigma_need ** 2)
        precision = prior_precision + sample_count * self.precision_step(self.mdp.config)
        return 1.0 / precision

    def state_from_belief(self, belief: BeliefState) -> DPState:
        obs_precision = self.precision_step(self.mdp.config)
        prior_precision = 1.0 / (self.mdp.config.sigma_need ** 2)
        samples_1 = max(0, round((1.0 / belief.var_1 - prior_precision) / obs_precision))
        samples_2 = max(0, round((1.0 / belief.var_2 - prior_precision) / obs_precision))
        return DPState(
            mean_1=self._snap_mean(belief.mean_1),
            mean_2=self._snap_mean(belief.mean_2),
            samples_1=samples_1,
            samples_2=samples_2,
        )

    def belief_from_state(self, state: DPState) -> BeliefState:
        return BeliefState(
            mean_1=state.mean_1,
            var_1=self.variance_for_sample_count(state.samples_1),
            mean_2=state.mean_2,
            var_2=self.variance_for_sample_count(state.samples_2),
            deliberation_time=(state.samples_1 + state.samples_2) * self.mdp.config.sample_time_cost,
        )

    def _terminal_value(self, state: DPState) -> float:
        if state not in self._terminal_cache:
            belief = self.belief_from_state(state)
            _, value = self.mdp.solve_terminal_allocation(belief)
            self._terminal_cache[state] = value
        return self._terminal_cache[state]

    def _observation_nodes(self, mean: float, var: float) -> List[float]:
        predictive_sd = math.sqrt(var + self.mdp.config.sigma_sample ** 2)
        dist = NormalDist(mu=mean, sigma=predictive_sd)
        return [
            dist.inv_cdf((i + 0.5) / self.observation_branches)
            for i in range(self.observation_branches)
        ]

    def _next_state(self, state: DPState, action: Action, observation: float) -> DPState:
        if action == self.mdp.SAMPLE_PERSON_1:
            mean, _ = self.mdp.posterior_update(
                state.mean_1,
                self.variance_for_sample_count(state.samples_1),
                observation,
            )
            return DPState(
                mean_1=self._snap_mean(mean),
                mean_2=state.mean_2,
                samples_1=state.samples_1 + 1,
                samples_2=state.samples_2,
            )
        mean, _ = self.mdp.posterior_update(
            state.mean_2,
            self.variance_for_sample_count(state.samples_2),
            observation,
        )
        return DPState(
            mean_1=state.mean_1,
            mean_2=self._snap_mean(mean),
            samples_1=state.samples_1,
            samples_2=state.samples_2 + 1,
        )

    def _sample_action_value(self, state: DPState, action: Action, remaining_samples: int) -> float:
        belief = self.belief_from_state(state)
        if action not in self.mdp.available_actions(belief):
            return float("-inf")
        target_mean = state.mean_1 if action == self.mdp.SAMPLE_PERSON_1 else state.mean_2
        target_var = (
            self.variance_for_sample_count(state.samples_1)
            if action == self.mdp.SAMPLE_PERSON_1
            else self.variance_for_sample_count(state.samples_2)
        )
        values = [
            self.value_and_action(
                self._next_state(state, action, observation),
                remaining_samples - 1,
            )[0]
            for observation in self._observation_nodes(target_mean, target_var)
        ]
        return sum(values) / len(values)

    def value_and_action(self, state: DPState, remaining_samples: int) -> Tuple[float, Action]:
        cache_key = (state, remaining_samples)
        if cache_key in self._value_cache:
            return self._value_cache[cache_key]

        best_value = self._terminal_value(state)
        best_action = self.mdp.TERMINATE

        if remaining_samples > 0:
            sample_1_value = self._sample_action_value(state, self.mdp.SAMPLE_PERSON_1, remaining_samples)
            sample_2_value = self._sample_action_value(state, self.mdp.SAMPLE_PERSON_2, remaining_samples)
            if sample_1_value > best_value:
                best_value = sample_1_value
                best_action = self.mdp.SAMPLE_PERSON_1
            if sample_2_value > best_value:
                best_value = sample_2_value
                best_action = self.mdp.SAMPLE_PERSON_2

        self._value_cache[cache_key] = (best_value, best_action)
        return best_value, best_action

    def choose_action(self, belief: BeliefState) -> Action:
        state = self.state_from_belief(belief)
        used_samples = state.samples_1 + state.samples_2
        remaining_samples = max(0, self.max_samples - used_samples)
        return self.value_and_action(state, remaining_samples)[1]


class DiscretizedDynamicProgrammingPolicy:
    name = "discretized_dp"

    def __init__(
        self,
        max_samples: int = 2,
        mean_grid_size: int = 11,
        mean_grid_radius_sd: float = 3.0,
        observation_branches: int = 5,
    ):
        self.max_samples = max_samples
        self.mean_grid_size = mean_grid_size
        self.mean_grid_radius_sd = mean_grid_radius_sd
        self.observation_branches = observation_branches
        self._solver_key: Tuple[float, ...] | None = None
        self._solver: FiniteHorizonDPSolver | None = None

    def _config_key(self, mdp: ContinuousAllocationMetaMDP) -> Tuple[float, ...]:
        config = mdp.config
        return (
            config.mu_need,
            config.sigma_need,
            config.sigma_sample,
            config.total_time,
            config.lambda_shortfall,
            config.utility_exponent,
            config.terminate_cost,
            config.sample_time_cost,
            float(config.allocation_grid_size),
            float(config.expected_utility_draws),
            float(self.max_samples),
            float(self.mean_grid_size),
            float(self.mean_grid_radius_sd),
            float(self.observation_branches),
        )

    def _solver_for(self, mdp: ContinuousAllocationMetaMDP) -> FiniteHorizonDPSolver:
        key = self._config_key(mdp)
        if self._solver is None or self._solver_key != key:
            self._solver = FiniteHorizonDPSolver(
                mdp=mdp,
                max_samples=self.max_samples,
                mean_grid_size=self.mean_grid_size,
                mean_grid_radius_sd=self.mean_grid_radius_sd,
                observation_branches=self.observation_branches,
            )
            self._solver_key = key
        return self._solver

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        available = mdp.available_actions(belief)
        if not available:
            return mdp.TERMINATE
        action = self._solver_for(mdp).choose_action(belief)
        return action if action in available else mdp.TERMINATE
