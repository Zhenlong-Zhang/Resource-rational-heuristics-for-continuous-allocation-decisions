from __future__ import annotations

import math
from typing import Dict

try:
    from ..mdp.finite_support import FiniteSupportBeliefState, FiniteSupportMetaMDP
    from ..mdp.meta_mdp import Action
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.finite_support import FiniteSupportBeliefState, FiniteSupportMetaMDP  # type: ignore
    from mdp.meta_mdp import Action  # type: ignore


class FiniteSupportMyopicVOIPolicy:
    name = "matched_prior_myopic_rr"

    def __init__(self, quadrature_order: int = 31, tie_tolerance: float = 1e-10) -> None:
        if quadrature_order <= 0:
            raise ValueError("quadrature_order must be positive")
        if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be finite and nonnegative")
        self.quadrature_order = int(quadrature_order)
        self.tie_tolerance = float(tie_tolerance)

    def _sample_action_value(
        self,
        mdp: FiniteSupportMetaMDP,
        belief: FiniteSupportBeliefState,
        action: Action,
    ) -> float:
        points = mdp.predictive_observation_quadrature(
            belief, action, self.quadrature_order
        )
        observations = [observation for observation, _ in points]
        predictive_weights = [weight for _, weight in points]
        posterior_weights = mdp.posterior_weights_for_observations(
            belief,
            action,
            observations,
        )
        terminal_values = mdp.optimal_terminal_values_for_weights(
            belief,
            posterior_weights,
            deliberation_time=belief.deliberation_time + mdp.sample_cost(action, belief),
        )
        try:
            import numpy as np  # type: ignore

            return float(np.dot(np.asarray(predictive_weights), terminal_values))
        except ImportError:
            return float(
                math.fsum(
                    weight * value
                    for weight, value in zip(predictive_weights, terminal_values)
                )
            )

    def action_values(
        self,
        mdp: FiniteSupportMetaMDP,
        belief: FiniteSupportBeliefState,
    ) -> Dict[Action, float]:
        available = mdp.available_actions(belief, self)
        if not available:
            return {mdp.TERMINATE: mdp.solve_terminal_allocation(belief)[1]}
        values: Dict[Action, float] = {}
        for action in available:
            if action == mdp.TERMINATE:
                values[action] = mdp.solve_terminal_allocation(belief)[1]
            else:
                values[action] = self._sample_action_value(mdp, belief, action)
        return values

    def choose_action(
        self,
        mdp: FiniteSupportMetaMDP,
        belief: FiniteSupportBeliefState,
    ) -> Action:
        values = self.action_values(mdp, belief)
        stop_value = values.get(mdp.TERMINATE)
        if stop_value is None:
            return mdp.TERMINATE

        ordered_samples = (mdp.SAMPLE_PERSON_1, mdp.SAMPLE_PERSON_2)
        eligible = [
            action
            for action in ordered_samples
            if action in values and values[action] > stop_value + self.tie_tolerance
        ]
        if not eligible:
            return mdp.TERMINATE
        best_value = max(values[action] for action in eligible)
        for action in eligible:
            if values[action] >= best_value - self.tie_tolerance:
                return action
        return mdp.TERMINATE
