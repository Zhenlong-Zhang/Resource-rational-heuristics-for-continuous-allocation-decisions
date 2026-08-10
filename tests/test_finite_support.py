from __future__ import annotations

import math
import random
import unittest

from src.mdp.finite_support import (
    FiniteSupportAtom,
    FiniteSupportMetaMDP,
    FiniteSupportPrior,
)
from src.mdp.meta_mdp import EnvironmentConfig, TrueState, utility
from src.policies.finite_support_voi import FiniteSupportMyopicVOIPolicy


class SampleOncePolicy:
    name = "sample_once"

    def choose_action(self, mdp, belief):
        return mdp.SAMPLE_PERSON_1 if not belief.history else mdp.TERMINATE


def finite_config(**changes) -> EnvironmentConfig:
    values = {
        "mu_need": 20.0,
        "sigma_need": 5.0,
        "sigma_sample": 2.0,
        "total_time": 40.0,
        "lambda_shortfall": 2.0,
        "utility_exponent": 0.35,
        "learning_per_unit_of_tutoring": 1.25,
        "terminate_cost": 1.0,
        "sample_time_cost": 0.25,
        "allocation_grid_size": 101,
        "max_meta_samples": 8,
        "random_seed": 19,
    }
    values.update(changes)
    return EnvironmentConfig(**values)


def symmetric_prior() -> FiniteSupportPrior:
    return FiniteSupportPrior.from_total_and_absolute_gaps(
        total_needs=(32.0, 36.0),
        absolute_gaps=(8.0, 16.0),
    )


class FiniteSupportTests(unittest.TestCase):
    def test_support_is_positive_normalized_and_seeded_sampling_matches_weights(self) -> None:
        states = (
            FiniteSupportAtom(20.0, 0.2, -1),
            FiniteSupportAtom(20.0, 0.2, 1),
        )
        prior = FiniteSupportPrior(states, (1.0, 3.0))
        self.assertAlmostEqual(math.fsum(prior.weights), 1.0)
        self.assertTrue(all(state.need_1 > 0.0 and state.need_2 > 0.0 for state in prior.states))
        with self.assertRaises(ValueError):
            FiniteSupportAtom(20.0, 1.0, 1)

        mdp = FiniteSupportMetaMDP(finite_config(random_seed=71), prior)
        counts = {state: 0 for state in states}
        for _ in range(8000):
            draw = mdp.sample_true_state()
            matched = next(
                state
                for state in states
                if state.need_1 == draw.need_1 and state.need_2 == draw.need_2
            )
            counts[matched] += 1
        self.assertAlmostEqual(counts[states[0]] / 8000.0, 0.25, delta=0.02)
        self.assertAlmostEqual(counts[states[1]] / 8000.0, 0.75, delta=0.02)

    def test_symmetric_prior_has_symmetric_marginal_moments(self) -> None:
        belief = FiniteSupportMetaMDP(finite_config(), symmetric_prior()).initial_belief()
        self.assertAlmostEqual(belief.mean_1, belief.mean_2)
        self.assertAlmostEqual(belief.var_1, belief.var_2)
        for state, weight in zip(belief.states, belief.weights):
            swapped_index = belief.states.index(state.swapped())
            self.assertAlmostEqual(weight, belief.weights[swapped_index])

    def test_prior_integration_api_is_deterministic(self) -> None:
        prior_a = FiniteSupportPrior.from_total_gap_support(
            total_needs=(30.0, 34.0),
            absolute_gaps=(6.0, 12.0),
            total_weights=(1.0, 2.0),
        )
        prior_b = FiniteSupportPrior.from_total_and_absolute_gaps(
            total_needs=(30.0, 34.0),
            absolute_gaps=(6.0, 12.0),
            total_weights=(1.0, 2.0),
        )
        self.assertEqual(prior_a.support_hash, prior_b.support_hash)
        self.assertEqual(prior_a.sample_atom(seed=91), prior_b.sample_atom(seed=91))
        rng_a = random.Random(17)
        rng_b = random.Random(17)
        self.assertEqual(
            [prior_a.sample_atom(rng_a) for _ in range(10)],
            [prior_b.sample_atom(rng_b) for _ in range(10)],
        )
        with self.assertRaises(ValueError):
            prior_a.sample_atom(random.Random(1), seed=1)

    def test_log_space_bayes_update_matches_direct_likelihood(self) -> None:
        prior = symmetric_prior()
        mdp = FiniteSupportMetaMDP(finite_config(sigma_sample=1.5), prior)
        belief = mdp.initial_belief()
        observation = max(state.need_1 for state in prior.states) - 0.2
        posterior = mdp.posterior_transition(
            belief,
            mdp.SAMPLE_PERSON_1,
            observation,
            advance_time=False,
            record=False,
        )
        likelihoods = [
            weight
            * math.exp(-0.5 * ((observation - state.need_1) / mdp.config.sigma_sample) ** 2)
            for state, weight in zip(prior.states, prior.weights)
        ]
        total = math.fsum(likelihoods)
        expected = [value / total for value in likelihoods]
        for actual, target in zip(posterior.weights, expected):
            self.assertAlmostEqual(actual, target, places=13)
        self.assertGreater(posterior.mean_1, belief.mean_1)
        self.assertLess(posterior.mean_2, belief.mean_2)

    def test_zero_posterior_weights_remain_valid_on_later_updates(self) -> None:
        mdp = FiniteSupportMetaMDP(finite_config(sigma_sample=0.1), symmetric_prior())
        belief = mdp.initial_belief()
        first = mdp.posterior_transition(
            belief,
            mdp.SAMPLE_PERSON_1,
            max(state.need_1 for state in belief.states) + 100.0,
            advance_time=False,
        )
        self.assertTrue(any(weight == 0.0 for weight in first.weights))
        second = mdp.posterior_transition(
            first,
            mdp.SAMPLE_PERSON_2,
            min(state.need_2 for state in belief.states),
            advance_time=False,
        )
        self.assertAlmostEqual(math.fsum(second.weights), 1.0)

    def test_vectorized_posterior_and_terminal_values_match_serial(self) -> None:
        mdp = FiniteSupportMetaMDP(finite_config(allocation_grid_size=61), symmetric_prior())
        belief = mdp.initial_belief()
        observations = [belief.mean_1 - 2.0, belief.mean_1, belief.mean_1 + 2.0]
        batch = mdp.posterior_weights_for_observations(
            belief,
            mdp.SAMPLE_PERSON_1,
            observations,
        )
        serial_beliefs = [
            mdp.posterior_transition(
                belief,
                mdp.SAMPLE_PERSON_1,
                observation,
                advance_time=True,
                record=False,
            )
            for observation in observations
        ]
        for row, serial in zip(batch, serial_beliefs):
            for actual, expected in zip(row, serial.weights):
                self.assertAlmostEqual(float(actual), expected, places=13)
        batch_values = mdp.optimal_terminal_values_for_weights(
            belief,
            batch,
            deliberation_time=belief.deliberation_time + mdp.config.sample_time_cost,
        )
        serial_values = [mdp.solve_terminal_allocation(item)[1] for item in serial_beliefs]
        for actual, expected in zip(batch_values, serial_values):
            self.assertAlmostEqual(float(actual), expected, places=11)

    def test_predictive_moments_match_weighted_mixture(self) -> None:
        mdp = FiniteSupportMetaMDP(finite_config(sigma_sample=3.0), symmetric_prior())
        belief = mdp.initial_belief()
        mean, variance = mdp.predictive_moments(belief, mdp.SAMPLE_PERSON_1)
        direct_mean = math.fsum(
            weight * state.need_1 for state, weight in zip(belief.states, belief.weights)
        )
        direct_variance = math.fsum(
            weight * (state.need_1 - direct_mean) ** 2
            for state, weight in zip(belief.states, belief.weights)
        ) + 9.0
        self.assertAlmostEqual(mean, direct_mean)
        self.assertAlmostEqual(variance, direct_variance)

    def test_predictive_gh_mixture_integrates_first_two_moments(self) -> None:
        mdp = FiniteSupportMetaMDP(finite_config(sigma_sample=2.5), symmetric_prior())
        belief = mdp.initial_belief()
        points = mdp.predictive_observation_quadrature(
            belief, mdp.SAMPLE_PERSON_1, order=31
        )
        total_weight = math.fsum(weight for _, weight in points)
        mean = math.fsum(weight * observation for observation, weight in points)
        second = math.fsum(weight * observation * observation for observation, weight in points)
        expected_mean, expected_variance = mdp.predictive_moments(
            belief, mdp.SAMPLE_PERSON_1
        )
        self.assertAlmostEqual(total_weight, 1.0, places=13)
        self.assertAlmostEqual(mean, expected_mean, places=11)
        self.assertAlmostEqual(second - mean * mean, expected_variance, places=10)

    def test_gh31_and_gh61_action_values_are_consistent(self) -> None:
        mdp = FiniteSupportMetaMDP(
            finite_config(sigma_sample=1.0, allocation_grid_size=51), symmetric_prior()
        )
        belief = mdp.initial_belief()
        values_31 = FiniteSupportMyopicVOIPolicy(31).action_values(mdp, belief)
        values_61 = FiniteSupportMyopicVOIPolicy(61).action_values(mdp, belief)
        self.assertEqual(values_31.keys(), values_61.keys())
        for action in values_31:
            self.assertAlmostEqual(values_31[action], values_61[action], delta=1e-4)
        self.assertEqual(
            FiniteSupportMyopicVOIPolicy(31).choose_action(mdp, belief),
            FiniteSupportMyopicVOIPolicy(61).choose_action(mdp, belief),
        )

    def test_terminal_utility_is_exact_weighted_enumeration(self) -> None:
        prior = symmetric_prior()
        mdp = FiniteSupportMetaMDP(finite_config(), prior)
        belief = mdp.initial_belief()
        allocation = 0.37
        remaining = mdp.remaining_time_after_termination(belief)
        amount_1, amount_2 = mdp.allocation_to_learning_outcomes(allocation, remaining)
        alpha = mdp.utility_exponent()
        direct = math.fsum(
            weight
            * (
                utility(amount_1 - state.need_1, mdp.config.lambda_shortfall, alpha)
                + utility(amount_2 - state.need_2, mdp.config.lambda_shortfall, alpha)
            )
            for state, weight in zip(prior.states, prior.weights)
        )
        self.assertAlmostEqual(mdp.expected_terminal_utility(belief, allocation), direct)

        best_allocation, best_value = mdp.solve_terminal_allocation(belief)
        enumerated = {
            candidate: mdp.expected_terminal_utility(belief, candidate)
            for candidate in mdp.terminal_allocation_grid(belief)
        }
        self.assertGreaterEqual(best_value + 1e-12, max(enumerated.values()))
        self.assertLessEqual(
            mdp.solve_terminal_allocation_result(belief).regret_upper_bound,
            2.5e-5 + 1e-12,
        )
        self.assertAlmostEqual(best_allocation, 0.5)

    def test_episode_observation_and_utility_use_the_same_true_state(self) -> None:
        prior = symmetric_prior()
        true_state = TrueState(prior.states[-1].need_1, prior.states[-1].need_2)
        observations = {
            FiniteSupportMetaMDP.SAMPLE_PERSON_1: [true_state.need_1],
            FiniteSupportMetaMDP.SAMPLE_PERSON_2: [true_state.need_2],
        }
        mdp = FiniteSupportMetaMDP(
            finite_config(), prior, observation_streams=observations
        )
        result = mdp.run_episode(SampleOncePolicy(), true_state=true_state)
        self.assertEqual(result.true_state, true_state)
        self.assertEqual(result.samples[0]["observation"], true_state.need_1)
        amount_1, amount_2 = result.final_resource_person1, result.final_resource_person2
        direct = utility(
            amount_1 - true_state.need_1,
            mdp.config.lambda_shortfall,
            mdp.utility_exponent(),
        ) + utility(
            amount_2 - true_state.need_2,
            mdp.config.lambda_shortfall,
            mdp.utility_exponent(),
        )
        self.assertAlmostEqual(result.realized_utility, direct)

    def test_swap_invariance_for_posterior_terminal_choice_and_voi(self) -> None:
        mdp = FiniteSupportMetaMDP(
            finite_config(sigma_sample=2.0, allocation_grid_size=81), symmetric_prior()
        )
        prior_belief = mdp.initial_belief()
        observation = prior_belief.mean_1 + 5.0
        belief_1 = mdp.posterior_transition(
            prior_belief, mdp.SAMPLE_PERSON_1, observation, advance_time=False
        )
        belief_2 = mdp.posterior_transition(
            prior_belief, mdp.SAMPLE_PERSON_2, observation, advance_time=False
        )
        self.assertAlmostEqual(belief_1.mean_1, belief_2.mean_2)
        self.assertAlmostEqual(belief_1.mean_2, belief_2.mean_1)
        allocation_1, value_1 = mdp.solve_terminal_allocation(belief_1)
        allocation_2, value_2 = mdp.solve_terminal_allocation(belief_2)
        self.assertAlmostEqual(allocation_1 + allocation_2, 1.0, places=12)
        self.assertAlmostEqual(value_1, value_2, places=12)

        policy = FiniteSupportMyopicVOIPolicy(31)
        values_1 = policy.action_values(mdp, belief_1)
        values_2 = policy.action_values(mdp, belief_2)
        self.assertAlmostEqual(values_1[mdp.TERMINATE], values_2[mdp.TERMINATE], places=11)
        self.assertAlmostEqual(values_1[mdp.SAMPLE_PERSON_1], values_2[mdp.SAMPLE_PERSON_2], places=10)
        self.assertAlmostEqual(values_1[mdp.SAMPLE_PERSON_2], values_2[mdp.SAMPLE_PERSON_1], places=10)

    def test_stop_wins_numerical_ties(self) -> None:
        mdp = FiniteSupportMetaMDP(
            finite_config(sample_time_cost=0.0, max_meta_samples=2),
            FiniteSupportPrior((FiniteSupportAtom(20.0, 0.0, 1),), (1.0,)),
        )
        belief = mdp.initial_belief()
        policy = FiniteSupportMyopicVOIPolicy(31, tie_tolerance=1e-10)
        self.assertEqual(policy.choose_action(mdp, belief), mdp.TERMINATE)


if __name__ == "__main__":
    unittest.main()
