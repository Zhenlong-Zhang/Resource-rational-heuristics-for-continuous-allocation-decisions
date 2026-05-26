from __future__ import annotations

import math
import statistics
from typing import List

try:
    from ..mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP


class MyopicValueOfInformationPolicy:
    name = "myopic_voi"

    def __init__(self, observation_draws: int = 64):
        self.observation_draws = observation_draws

    def _estimate_action_value(
        self,
        mdp: ContinuousAllocationMetaMDP,
        belief: BeliefState,
        action: Action,
    ) -> float:
        if action == mdp.TERMINATE:
            _, value = mdp.solve_terminal_allocation(belief)
            return value

        obs_target_mean = belief.mean_1 if action == mdp.SAMPLE_PERSON_1 else belief.mean_2
        obs_target_var = belief.var_1 if action == mdp.SAMPLE_PERSON_1 else belief.var_2
        predictive_sd = math.sqrt(obs_target_var + mdp.config.sigma_sample ** 2)
        observations = [mdp.rng.gauss(obs_target_mean, predictive_sd) for _ in range(self.observation_draws)]

        values: List[float] = []
        for obs in observations:
            next_belief = belief.copy()
            next_belief.deliberation_time += mdp.sample_cost(action, belief)
            if action == mdp.SAMPLE_PERSON_1:
                next_belief.mean_1, next_belief.var_1 = mdp.posterior_update(
                    next_belief.mean_1, next_belief.var_1, float(obs)
                )
            else:
                next_belief.mean_2, next_belief.var_2 = mdp.posterior_update(
                    next_belief.mean_2, next_belief.var_2, float(obs)
                )
            _, value = mdp.solve_terminal_allocation(next_belief)
            values.append(value)
        return float(statistics.mean(values))

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        available = mdp.available_actions(belief)
        if not available:
            return mdp.TERMINATE
        values = {action: self._estimate_action_value(mdp, belief, action) for action in available}
        return max(values, key=values.get)


class BlinkeredPolicy:
    name = "blinkered"

    def __init__(self, horizon: int = 2, observation_draws: int = 12):
        self.horizon = horizon
        self.observation_draws = observation_draws

    def _sample_observations(
        self,
        mdp: ContinuousAllocationMetaMDP,
        belief: BeliefState,
        action: Action,
    ) -> List[float]:
        target_mean = belief.mean_1 if action == mdp.SAMPLE_PERSON_1 else belief.mean_2
        target_var = belief.var_1 if action == mdp.SAMPLE_PERSON_1 else belief.var_2
        predictive_sd = math.sqrt(target_var + mdp.config.sigma_sample ** 2)
        return [mdp.rng.gauss(target_mean, predictive_sd) for _ in range(self.observation_draws)]

    def _belief_after_observation(
        self,
        mdp: ContinuousAllocationMetaMDP,
        belief: BeliefState,
        action: Action,
        observation: float,
    ) -> BeliefState:
        next_belief = belief.copy()
        next_belief.deliberation_time += mdp.sample_cost(action, belief)
        if action == mdp.SAMPLE_PERSON_1:
            next_belief.mean_1, next_belief.var_1 = mdp.posterior_update(
                next_belief.mean_1,
                next_belief.var_1,
                observation,
            )
        else:
            next_belief.mean_2, next_belief.var_2 = mdp.posterior_update(
                next_belief.mean_2,
                next_belief.var_2,
                observation,
            )
        return next_belief

    def _terminal_value(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        _, value = mdp.solve_terminal_allocation(belief)
        return value

    def _value_restricted_to_action(
        self,
        mdp: ContinuousAllocationMetaMDP,
        belief: BeliefState,
        action: Action,
        depth: int,
    ) -> float:
        stop_value = self._terminal_value(mdp, belief)
        if depth <= 0 or action not in mdp.available_actions(belief):
            return stop_value

        observations = self._sample_observations(mdp, belief, action)
        continuation_values = [
            self._value_restricted_to_action(
                mdp,
                self._belief_after_observation(mdp, belief, action, float(observation)),
                action,
                depth - 1,
            )
            for observation in observations
        ]
        continue_value = float(statistics.mean(continuation_values))
        return max(stop_value, continue_value)

    def _first_action_value(
        self,
        mdp: ContinuousAllocationMetaMDP,
        belief: BeliefState,
        action: Action,
    ) -> float:
        if action == mdp.TERMINATE:
            return self._terminal_value(mdp, belief)
        observations = self._sample_observations(mdp, belief, action)
        values = [
            self._value_restricted_to_action(
                mdp,
                self._belief_after_observation(mdp, belief, action, float(observation)),
                action,
                self.horizon - 1,
            )
            for observation in observations
        ]
        return float(statistics.mean(values))

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        available = mdp.available_actions(belief)
        if not available:
            return mdp.TERMINATE
        values = {action: self._first_action_value(mdp, belief, action) for action in available}
        return max(values, key=values.get)
