from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from .meta_mdp import (
        Action,
        ContinuousAllocationMetaMDP,
        EnvironmentConfig,
        TrueState,
        utility,
    )
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import (  # type: ignore
        Action,
        ContinuousAllocationMetaMDP,
        EnvironmentConfig,
        TrueState,
        utility,
    )


@dataclass(frozen=True, order=True)
class FiniteSupportAtom:
    total_need: float
    gap_fraction: float
    orientation: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_need) or self.total_need <= 0.0:
            raise ValueError("total_need must be finite and positive")
        if not math.isfinite(self.gap_fraction) or not 0.0 <= self.gap_fraction < 1.0:
            raise ValueError("gap_fraction must be finite and in [0, 1)")
        if self.orientation not in (-1, 1):
            raise ValueError("orientation must be -1 or 1")
        if self.need_1 <= 0.0 or self.need_2 <= 0.0:
            raise ValueError("every support state must imply strictly positive needs")

    @property
    def need_1(self) -> float:
        return self.total_need * (1.0 + self.orientation * self.gap_fraction) / 2.0

    @property
    def need_2(self) -> float:
        return self.total_need * (1.0 - self.orientation * self.gap_fraction) / 2.0

    @property
    def absolute_gap(self) -> float:
        return self.total_need * self.gap_fraction

    def swapped(self) -> "FiniteSupportAtom":
        return FiniteSupportAtom(self.total_need, self.gap_fraction, -self.orientation)


@dataclass(frozen=True)
class FiniteSupportPrior:
    states: Tuple[FiniteSupportAtom, ...]
    weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        states = tuple(self.states)
        weights = tuple(float(weight) for weight in self.weights)
        if not states:
            raise ValueError("finite support must contain at least one state")
        if len(states) != len(weights):
            raise ValueError("states and weights must have the same length")
        if len(set(states)) != len(states):
            raise ValueError("finite support states must be unique")
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
            raise ValueError("all support weights must be finite and positive")
        total = math.fsum(weights)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("support weights must have a finite positive sum")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "weights", tuple(weight / total for weight in weights))

    @classmethod
    def from_total_and_absolute_gaps(
        cls,
        total_needs: Sequence[float],
        absolute_gaps: Sequence[float],
        orientations: Sequence[int] = (-1, 1),
        total_weights: Optional[Sequence[float]] = None,
        gap_weights: Optional[Sequence[float]] = None,
        orientation_weights: Optional[Sequence[float]] = None,
    ) -> "FiniteSupportPrior":
        totals = tuple(float(value) for value in total_needs)
        gaps = tuple(float(value) for value in absolute_gaps)
        directions = tuple(int(value) for value in orientations)
        if not totals or not gaps or not directions:
            raise ValueError("total, gap, and orientation supports must be nonempty")

        def component_weights(
            values: Sequence[object],
            configured: Optional[Sequence[float]],
            name: str,
        ) -> Tuple[float, ...]:
            if configured is None:
                return tuple(1.0 for _ in values)
            result = tuple(float(weight) for weight in configured)
            if len(result) != len(values):
                raise ValueError(f"{name} weights must match its support")
            return result

        total_w = component_weights(totals, total_weights, "total")
        gap_w = component_weights(gaps, gap_weights, "gap")
        orientation_w = component_weights(directions, orientation_weights, "orientation")
        states: List[FiniteSupportAtom] = []
        weights: List[float] = []
        for total, weight_total in zip(totals, total_w):
            for gap, weight_gap in zip(gaps, gap_w):
                if not math.isfinite(gap) or gap < 0.0 or gap >= total:
                    raise ValueError("absolute gaps must be finite, nonnegative, and below total_need")
                for orientation, weight_orientation in zip(directions, orientation_w):
                    states.append(FiniteSupportAtom(total, gap / total, orientation))
                    weights.append(weight_total * weight_gap * weight_orientation)
        return cls(tuple(states), tuple(weights))

    @classmethod
    def from_total_gap_support(
        cls,
        total_needs: Sequence[float],
        absolute_gaps: Sequence[float],
        orientations: Sequence[int] = (-1, 1),
        total_weights: Optional[Sequence[float]] = None,
        gap_weights: Optional[Sequence[float]] = None,
        orientation_weights: Optional[Sequence[float]] = None,
    ) -> "FiniteSupportPrior":
        return cls.from_total_and_absolute_gaps(
            total_needs=total_needs,
            absolute_gaps=absolute_gaps,
            orientations=orientations,
            total_weights=total_weights,
            gap_weights=gap_weights,
            orientation_weights=orientation_weights,
        )

    def sample_atom(
        self,
        rng: Optional[random.Random] = None,
        *,
        seed: Optional[int] = None,
    ) -> FiniteSupportAtom:
        if rng is not None and seed is not None:
            raise ValueError("provide either rng or seed, not both")
        generator = rng if rng is not None else random.Random(seed)
        draw = generator.random()
        cumulative = 0.0
        for state, weight in zip(self.states, self.weights):
            cumulative += weight
            if draw < cumulative:
                return state
        return self.states[-1]

    @property
    def support_hash(self) -> str:
        rows = [
            {
                "gap_fraction": state.gap_fraction,
                "orientation": state.orientation,
                "total_need": state.total_need,
                "weight": weight,
            }
            for state, weight in zip(self.states, self.weights)
        ]
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class FiniteSupportBeliefState:
    states: Tuple[FiniteSupportAtom, ...]
    weights: Tuple[float, ...]
    deliberation_time: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.states = tuple(self.states)
        self.weights = tuple(float(weight) for weight in self.weights)
        if not self.states or len(self.states) != len(self.weights):
            raise ValueError("belief states and weights must be nonempty and aligned")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in self.weights):
            raise ValueError("posterior weights must be finite and nonnegative")
        total = math.fsum(self.weights)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("posterior weights must have a finite positive sum")
        self.weights = tuple(weight / total for weight in self.weights)

    def copy(self) -> "FiniteSupportBeliefState":
        return FiniteSupportBeliefState(
            states=self.states,
            weights=self.weights,
            deliberation_time=self.deliberation_time,
            history=list(self.history),
        )

    def _moment(self, person: int) -> Tuple[float, float]:
        needs = [state.need_1 if person == 1 else state.need_2 for state in self.states]
        mean = math.fsum(weight * need for weight, need in zip(self.weights, needs))
        variance = math.fsum(
            weight * (need - mean) ** 2 for weight, need in zip(self.weights, needs)
        )
        return float(mean), float(max(0.0, variance))

    @property
    def mean_1(self) -> float:
        return self._moment(1)[0]

    @property
    def var_1(self) -> float:
        return self._moment(1)[1]

    @property
    def mean_2(self) -> float:
        return self._moment(2)[0]

    @property
    def var_2(self) -> float:
        return self._moment(2)[1]


class FiniteSupportMetaMDP(ContinuousAllocationMetaMDP):
    def __init__(
        self,
        config: EnvironmentConfig,
        prior: FiniteSupportPrior,
        observation_streams: Optional[Mapping[Action, Sequence[float]]] = None,
    ) -> None:
        if not math.isfinite(config.sigma_sample) or config.sigma_sample <= 0.0:
            raise ValueError("sigma_sample must be finite and positive")
        super().__init__(config, observation_streams=observation_streams)
        self.prior = prior

    def sample_true_state(self) -> TrueState:
        state = self.prior.sample_atom(rng=self.rng)
        return TrueState(state.need_1, state.need_2)

    def initial_belief(self, true_state: Optional[TrueState] = None) -> FiniteSupportBeliefState:
        belief = FiniteSupportBeliefState(self.prior.states, self.prior.weights)
        if true_state is None:
            return belief
        for _ in range(self.config.prior_sample_count_1):
            belief = self.posterior_transition(
                belief,
                self.SAMPLE_PERSON_1,
                self.observe(true_state, self.SAMPLE_PERSON_1),
                advance_time=False,
                record=False,
            )
        for _ in range(self.config.prior_sample_count_2):
            belief = self.posterior_transition(
                belief,
                self.SAMPLE_PERSON_2,
                self.observe(true_state, self.SAMPLE_PERSON_2),
                advance_time=False,
                record=False,
            )
        return belief

    @staticmethod
    def _need_for_action(state: FiniteSupportAtom, action: Action) -> float:
        if action == ContinuousAllocationMetaMDP.SAMPLE_PERSON_1:
            return state.need_1
        if action == ContinuousAllocationMetaMDP.SAMPLE_PERSON_2:
            return state.need_2
        raise ValueError(f"Action {action} does not produce an observation")

    def posterior_transition(
        self,
        belief: FiniteSupportBeliefState,
        action: Action,
        observation: float,
        *,
        advance_time: bool = True,
        record: bool = True,
    ) -> FiniteSupportBeliefState:
        if action not in (self.SAMPLE_PERSON_1, self.SAMPLE_PERSON_2):
            raise ValueError(f"Action {action} does not produce an observation")
        if not math.isfinite(observation):
            raise ValueError("observation must be finite")
        if belief.states != self.prior.states:
            raise ValueError("belief support does not match this MDP's prior")

        sigma = self.config.sigma_sample
        log_scale = -math.log(sigma) - 0.5 * math.log(2.0 * math.pi)
        log_weights = []
        for state, weight in zip(belief.states, belief.weights):
            need = self._need_for_action(state, action)
            z = (observation - need) / sigma
            log_weights.append(
                -math.inf if weight == 0.0 else math.log(weight) + log_scale - 0.5 * z * z
            )
        maximum = max(log_weights)
        relative = [math.exp(value - maximum) for value in log_weights]
        normalizer = math.fsum(relative)
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise RuntimeError("posterior normalization failed")

        next_belief = FiniteSupportBeliefState(
            belief.states,
            tuple(value / normalizer for value in relative),
            deliberation_time=belief.deliberation_time,
            history=list(belief.history),
        )
        cost = self.sample_cost(action, belief) if advance_time else 0.0
        next_belief.deliberation_time += cost
        if record:
            next_belief.history.append(
                {
                    "action": 1.0 if action == self.SAMPLE_PERSON_1 else 2.0,
                    "observation": float(observation),
                    "cost": cost,
                }
            )
        return next_belief

    def posterior_weights_for_observations(
        self,
        belief: FiniteSupportBeliefState,
        action: Action,
        observations: Sequence[float],
    ):
        """Return posterior weights for many hypothetical observations."""

        if action not in (self.SAMPLE_PERSON_1, self.SAMPLE_PERSON_2):
            raise ValueError(f"Action {action} does not produce an observation")
        if belief.states != self.prior.states:
            raise ValueError("belief support does not match this MDP's prior")
        try:
            import numpy as np  # type: ignore
        except ImportError:
            rows = []
            for observation in observations:
                rows.append(
                    self.posterior_transition(
                        belief,
                        action,
                        float(observation),
                        advance_time=False,
                        record=False,
                    ).weights
                )
            return rows

        observed = np.asarray(observations, dtype=float)
        if observed.ndim != 1 or not np.all(np.isfinite(observed)):
            raise ValueError("observations must be a finite one-dimensional sequence")
        needs = np.asarray(
            [self._need_for_action(state, action) for state in belief.states],
            dtype=float,
        )
        prior = np.asarray(belief.weights, dtype=float)
        log_prior = np.full_like(prior, -np.inf)
        positive = prior > 0.0
        log_prior[positive] = np.log(prior[positive])
        standardized = (observed[:, None] - needs[None, :]) / self.config.sigma_sample
        log_weights = log_prior[None, :] - 0.5 * standardized * standardized
        row_max = np.max(log_weights, axis=1, keepdims=True)
        relative = np.exp(log_weights - row_max)
        normalizers = np.sum(relative, axis=1, keepdims=True)
        if not np.all(np.isfinite(normalizers)) or np.any(normalizers <= 0.0):
            raise RuntimeError("posterior batch normalization failed")
        return relative / normalizers

    def transition(
        self,
        belief: FiniteSupportBeliefState,
        true_state: TrueState,
        action: Action,
    ) -> FiniteSupportBeliefState:
        if action == self.TERMINATE:
            return self.terminate_belief(belief)  # type: ignore[return-value]
        observation = self.observe(true_state, action)
        return self.posterior_transition(belief, action, observation)

    def predictive_moments(
        self,
        belief: FiniteSupportBeliefState,
        action: Action,
    ) -> Tuple[float, float]:
        needs = [self._need_for_action(state, action) for state in belief.states]
        mean = math.fsum(weight * need for weight, need in zip(belief.weights, needs))
        variance = math.fsum(
            weight * (need - mean) ** 2 for weight, need in zip(belief.weights, needs)
        ) + self.config.sigma_sample ** 2
        return float(mean), float(max(0.0, variance))

    def predictive_observation_quadrature(
        self,
        belief: FiniteSupportBeliefState,
        action: Action,
        order: int,
    ) -> Tuple[Tuple[float, float], ...]:
        try:
            from ..solvers.gauss_hermite import gauss_hermite_nodes_weights
        except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
            from solvers.gauss_hermite import gauss_hermite_nodes_weights  # type: ignore

        nodes, gh_weights = gauss_hermite_nodes_weights(order)
        scale = math.sqrt(2.0) * self.config.sigma_sample
        normalization = math.sqrt(math.pi)
        points = []
        for state, posterior_weight in zip(belief.states, belief.weights):
            center = self._need_for_action(state, action)
            for node, gh_weight in zip(nodes, gh_weights):
                points.append(
                    (center + scale * node, posterior_weight * gh_weight / normalization)
                )
        return tuple(points)

    def predictive_gh_mixture(
        self,
        belief: FiniteSupportBeliefState,
        action: Action,
        order: int,
    ) -> Tuple[Tuple[float, float], ...]:
        return self.predictive_observation_quadrature(belief, action, order)

    def expected_terminal_utility(
        self,
        belief: FiniteSupportBeliefState,
        allocation_to_person1: float,
    ) -> float:
        if not 0.0 <= allocation_to_person1 <= 1.0:
            raise ValueError("allocation_to_person1 must be in [0, 1]")
        remaining_time = self.remaining_time_after_termination(belief)
        amount_1, amount_2 = self.allocation_to_learning_outcomes(
            allocation_to_person1, remaining_time
        )
        alpha = self.utility_exponent()
        return float(
            math.fsum(
                weight
                * (
                    utility(
                        amount_1 - state.need_1,
                        self.config.lambda_shortfall,
                        alpha,
                    )
                    + utility(
                        amount_2 - state.need_2,
                        self.config.lambda_shortfall,
                        alpha,
                    )
                )
                for state, weight in zip(belief.states, belief.weights)
            )
        )

    def terminal_allocation_grid(
        self,
        belief: FiniteSupportBeliefState,
    ) -> Tuple[float, ...]:
        return self.terminal_allocation_grid_at_time(
            belief,
            self.remaining_time_after_termination(belief),
        )

    def terminal_allocation_grid_at_time(
        self,
        belief: FiniteSupportBeliefState,
        remaining_time: float,
    ) -> Tuple[float, ...]:
        grid_size = self.config.allocation_grid_size
        regular = [0.5] if grid_size <= 1 else [index / (grid_size - 1) for index in range(grid_size)]
        candidates = set(regular)
        candidates.update((0.0, 0.5, 1.0))
        rate_1, rate_2 = self.learning_rates()
        if remaining_time > 0.0:
            for state in belief.states:
                candidates.add(state.need_1 / (rate_1 * remaining_time))
                candidates.add(1.0 - state.need_2 / (rate_2 * remaining_time))
                candidates.add(
                    (state.need_1 - state.need_2 + rate_2 * remaining_time)
                    / ((rate_1 + rate_2) * remaining_time)
                )
        return tuple(sorted(value for value in candidates if 0.0 <= value <= 1.0))

    def optimal_terminal_values_for_weights(
        self,
        belief: FiniteSupportBeliefState,
        posterior_weights,
        deliberation_time: float,
    ):
        """Vectorized terminal optimization for a batch of posterior weights."""

        try:
            import numpy as np  # type: ignore
        except ImportError:
            values = []
            for row in posterior_weights:
                posterior = FiniteSupportBeliefState(
                    belief.states,
                    tuple(float(weight) for weight in row),
                    deliberation_time=float(deliberation_time),
                    history=list(belief.history),
                )
                values.append(self.solve_terminal_allocation(posterior)[1])
            return values

        weights = np.asarray(posterior_weights, dtype=float)
        if weights.ndim != 2 or weights.shape[1] != len(belief.states):
            raise ValueError("posterior_weights must have shape (n, support_size)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("posterior weights must be finite and nonnegative")
        row_sums = np.sum(weights, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("each posterior-weight row must sum to one")

        terminal_belief = belief.copy()
        terminal_belief.deliberation_time = float(deliberation_time)
        remaining_time = self.remaining_time_after_termination(terminal_belief)
        allocations = np.asarray(
            self.terminal_allocation_grid_at_time(terminal_belief, remaining_time),
            dtype=float,
        )
        need_1 = np.asarray([state.need_1 for state in belief.states], dtype=float)[None, :]
        need_2 = np.asarray([state.need_2 for state in belief.states], dtype=float)[None, :]
        rate_1, rate_2 = self.learning_rates()
        amount_1 = rate_1 * allocations[:, None] * remaining_time
        amount_2 = rate_2 * (1.0 - allocations[:, None]) * remaining_time
        outcome_1 = amount_1 - need_1
        outcome_2 = amount_2 - need_2
        alpha = self.utility_exponent()
        utility_1 = np.where(
            outcome_1 < 0.0,
            -self.config.lambda_shortfall * np.power(np.maximum(-outcome_1, 0.0), alpha),
            np.power(np.maximum(outcome_1, 0.0), alpha),
        )
        utility_2 = np.where(
            outcome_2 < 0.0,
            -self.config.lambda_shortfall * np.power(np.maximum(-outcome_2, 0.0), alpha),
            np.power(np.maximum(outcome_2, 0.0), alpha),
        )
        values = weights @ (utility_1 + utility_2).T
        return np.max(values, axis=1)

    def solve_terminal_allocation(
        self,
        belief: FiniteSupportBeliefState,
    ) -> Tuple[float, float]:
        grid = self.terminal_allocation_grid(belief)
        try:
            import numpy as np  # type: ignore

            allocations = np.asarray(grid, dtype=float)
            remaining_time = self.remaining_time_after_termination(belief)
            rate_1, rate_2 = self.learning_rates()
            need_1 = np.asarray([state.need_1 for state in belief.states], dtype=float)[None, :]
            need_2 = np.asarray([state.need_2 for state in belief.states], dtype=float)[None, :]
            amount_1 = rate_1 * allocations[:, None] * remaining_time
            amount_2 = rate_2 * (1.0 - allocations[:, None]) * remaining_time
            outcome_1 = amount_1 - need_1
            outcome_2 = amount_2 - need_2
            alpha = self.utility_exponent()
            utility_1 = np.where(
                outcome_1 < 0.0,
                -self.config.lambda_shortfall * np.power(np.maximum(-outcome_1, 0.0), alpha),
                np.power(np.maximum(outcome_1, 0.0), alpha),
            )
            utility_2 = np.where(
                outcome_2 < 0.0,
                -self.config.lambda_shortfall * np.power(np.maximum(-outcome_2, 0.0), alpha),
                np.power(np.maximum(outcome_2, 0.0), alpha),
            )
            values = list((utility_1 + utility_2) @ np.asarray(belief.weights, dtype=float))
        except ImportError:
            values = [self.expected_terminal_utility(belief, allocation) for allocation in grid]
        best_value = max(values)
        tolerance = 1e-12 * max(1.0, abs(best_value))
        candidates = [
            index for index, value in enumerate(values) if value >= best_value - tolerance
        ]
        best_index = min(candidates, key=lambda index: (abs(grid[index] - 0.5), grid[index]))
        return float(grid[best_index]), float(values[best_index])


# Backward-compatible aliases for early exploratory code.
FiniteSupportState = FiniteSupportAtom
FiniteSupportBelief = FiniteSupportBeliefState
