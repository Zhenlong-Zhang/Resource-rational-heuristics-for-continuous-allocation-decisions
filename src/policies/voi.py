from __future__ import annotations

import math
import statistics
from typing import List

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
