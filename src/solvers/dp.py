from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List, Tuple

import math

from .gauss_hermite import gauss_hermite_nodes_weights

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
    online_samples_1: int = 0
    online_samples_2: int = 0


class FiniteHorizonDPSolver:
    """Small finite-horizon DP using the posterior precision grid suggested by Falk."""

    def __init__(
        self,
        mdp: ContinuousAllocationMetaMDP,
        max_samples: int = 2,
        mean_grid_size: int = 11,
        mean_grid_radius_sd: float = 3.0,
        observation_branches: int = 5,
        observation_integration: str = "quantile",
    ):
        self.mdp = mdp
        self.max_samples = max_samples
        self.mean_grid_size = max(3, mean_grid_size)
        self.mean_grid_radius_sd = mean_grid_radius_sd
        self.observation_branches = max(1, observation_branches)
        if observation_integration not in {"quantile", "gauss_hermite"}:
            raise ValueError("observation_integration must be 'quantile' or 'gauss_hermite'")
        self.observation_integration = observation_integration
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

    def variance_for_sample_count(self, sample_count: int, person: int = 1) -> float:
        configured_var = (
            self.mdp.config.initial_var_1
            if person == 1
            else self.mdp.config.initial_var_2
        )
        prior_var = configured_var if configured_var is not None else self.mdp.config.sigma_need ** 2
        prior_precision = 1.0 / prior_var
        precision = prior_precision + sample_count * self.precision_step(self.mdp.config)
        return 1.0 / precision

    def state_from_belief(self, belief: BeliefState) -> DPState:
        obs_precision = self.precision_step(self.mdp.config)
        prior_var_1 = self.mdp.config.initial_var_1 or self.mdp.config.sigma_need ** 2
        prior_var_2 = self.mdp.config.initial_var_2 or self.mdp.config.sigma_need ** 2
        samples_1 = max(0, round((1.0 / belief.var_1 - 1.0 / prior_var_1) / obs_precision))
        samples_2 = max(0, round((1.0 / belief.var_2 - 1.0 / prior_var_2) / obs_precision))
        online_samples_1 = sum(1 for event in belief.history if event.get("action") == 1.0)
        online_samples_2 = sum(1 for event in belief.history if event.get("action") == 2.0)
        return DPState(
            mean_1=self._snap_mean(belief.mean_1),
            mean_2=self._snap_mean(belief.mean_2),
            samples_1=samples_1,
            samples_2=samples_2,
            online_samples_1=online_samples_1,
            online_samples_2=online_samples_2,
        )

    def belief_from_state(self, state: DPState) -> BeliefState:
        history = [
            {"action": 1.0, "observation": math.nan, "cost": self.mdp.config.sample_time_cost}
            for _ in range(state.online_samples_1)
        ] + [
            {"action": 2.0, "observation": math.nan, "cost": self.mdp.config.sample_time_cost}
            for _ in range(state.online_samples_2)
        ]
        return BeliefState(
            mean_1=state.mean_1,
            var_1=self.variance_for_sample_count(state.samples_1, person=1),
            mean_2=state.mean_2,
            var_2=self.variance_for_sample_count(state.samples_2, person=2),
            deliberation_time=(state.online_samples_1 + state.online_samples_2)
            * self.mdp.config.sample_time_cost,
            history=history,
        )

    def _terminal_value(self, state: DPState) -> float:
        if state not in self._terminal_cache:
            belief = self.belief_from_state(state)
            _, value = self.mdp.solve_terminal_allocation(belief)
            self._terminal_cache[state] = value
        return self._terminal_cache[state]

    def _observation_nodes_weights(self, mean: float, var: float) -> List[Tuple[float, float]]:
        predictive_sd = math.sqrt(var + self.mdp.config.sigma_sample ** 2)
        if self.observation_integration == "gauss_hermite":
            nodes, weights = gauss_hermite_nodes_weights(self.observation_branches)
            return [
                (
                    mean + math.sqrt(2.0) * predictive_sd * node,
                    weight / math.sqrt(math.pi),
                )
                for node, weight in zip(nodes, weights)
            ]
        dist = NormalDist(mu=mean, sigma=predictive_sd)
        return [
            (
                dist.inv_cdf((i + 0.5) / self.observation_branches),
                1.0 / self.observation_branches,
            )
            for i in range(self.observation_branches)
        ]

    def _next_state(self, state: DPState, action: Action, observation: float) -> DPState:
        if action == self.mdp.SAMPLE_PERSON_1:
            mean, _ = self.mdp.posterior_update(
                state.mean_1,
                self.variance_for_sample_count(state.samples_1, person=1),
                observation,
            )
            return DPState(
                mean_1=self._snap_mean(mean),
                mean_2=state.mean_2,
                samples_1=state.samples_1 + 1,
                samples_2=state.samples_2,
                online_samples_1=state.online_samples_1 + 1,
                online_samples_2=state.online_samples_2,
            )
        mean, _ = self.mdp.posterior_update(
            state.mean_2,
            self.variance_for_sample_count(state.samples_2, person=2),
            observation,
        )
        return DPState(
            mean_1=state.mean_1,
            mean_2=self._snap_mean(mean),
            samples_1=state.samples_1,
            samples_2=state.samples_2 + 1,
            online_samples_1=state.online_samples_1,
            online_samples_2=state.online_samples_2 + 1,
        )

    def _sample_action_value(self, state: DPState, action: Action, remaining_samples: int) -> float:
        belief = self.belief_from_state(state)
        if action not in self.mdp.available_actions(belief):
            return float("-inf")
        target_mean = state.mean_1 if action == self.mdp.SAMPLE_PERSON_1 else state.mean_2
        target_var = (
            self.variance_for_sample_count(state.samples_1, person=1)
            if action == self.mdp.SAMPLE_PERSON_1
            else self.variance_for_sample_count(state.samples_2, person=2)
        )
        return sum(
            weight
            * self.value_and_action(
                self._next_state(state, action, observation),
                remaining_samples - 1,
            )[0]
            for observation, weight in self._observation_nodes_weights(target_mean, target_var)
        )

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
        used_samples = state.online_samples_1 + state.online_samples_2
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
        observation_integration: str = "quantile",
    ):
        self.max_samples = max_samples
        self.mean_grid_size = mean_grid_size
        self.mean_grid_radius_sd = mean_grid_radius_sd
        self.observation_branches = observation_branches
        self.observation_integration = observation_integration
        self._solver_key: Tuple[object, ...] | None = None
        self._solver: FiniteHorizonDPSolver | None = None

    def _config_key(self, mdp: ContinuousAllocationMetaMDP) -> Tuple[object, ...]:
        config = mdp.config
        return (
            config.mu_need,
            config.sigma_need,
            config.sigma_sample,
            config.total_time,
            config.lambda_shortfall,
            config.utility_exponent,
            config.alpha,
            config.terminate_cost,
            config.sample_time_cost,
            config.learning_per_unit_of_tutoring,
            config.delta_learning_per_unit_tutoring,
            config.initial_mean_1,
            config.initial_mean_2,
            config.initial_var_1,
            config.initial_var_2,
            config.prior_sample_count_1,
            config.prior_sample_count_2,
            config.max_meta_samples,
            config.expected_utility_method,
            config.gauss_hermite_order,
            float(config.allocation_grid_size),
            float(config.expected_utility_draws),
            float(self.max_samples),
            float(self.mean_grid_size),
            float(self.mean_grid_radius_sd),
            float(self.observation_branches),
            self.observation_integration,
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
                observation_integration=self.observation_integration,
            )
            self._solver_key = key
        return self._solver

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        available = mdp.available_actions(belief)
        if not available:
            return mdp.TERMINATE
        action = self._solver_for(mdp).choose_action(belief)
        return action if action in available else mdp.TERMINATE
