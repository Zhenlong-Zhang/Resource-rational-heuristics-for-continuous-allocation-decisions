from __future__ import annotations

import unittest

from src.experiments.diagnostics import (
    EQUAL_SPLIT_BASELINE_NAME,
    MANUAL_ACTIVE_POLICY_NAME,
    identify_r4_manual_advantage_candidates,
)
from src.experiments.regimes import true_outcome_metrics_for_allocation
from src.mdp.meta_mdp import BeliefState, ContinuousAllocationMetaMDP, EnvironmentConfig, TrueState
from src.policies.heuristic import EqualSplitBaselinePolicy, ManualActiveSearchEqualOutcomePolicy


class R4DiagnosticRegressionTest(unittest.TestCase):
    def test_true_outcome_metric_prefers_true_equal_outcome_over_equal_split(self) -> None:
        config = EnvironmentConfig(
            mu_need=50.0,
            sigma_need=1.0,
            total_time=101.0,
            terminate_cost=1.0,
            sample_time_cost=1.0,
            allocation_grid_size=21,
            expected_utility_draws=10,
            random_seed=7,
        )
        mdp = ContinuousAllocationMetaMDP(config)
        belief = BeliefState(mean_1=70.0, var_1=1.0, mean_2=30.0, var_2=1.0)
        belief = mdp.terminate_belief(belief, EqualSplitBaselinePolicy())
        true_state = TrueState(need_1=70.0, need_2=30.0)

        true_equal_metrics = true_outcome_metrics_for_allocation(
            mdp,
            true_state,
            belief,
            allocation_to_person1=0.7,
            allocation_tolerance=0.01,
        )
        equal_split_metrics = true_outcome_metrics_for_allocation(
            mdp,
            true_state,
            belief,
            allocation_to_person1=0.5,
            allocation_tolerance=0.01,
        )

        self.assertAlmostEqual(true_equal_metrics["realized_outcome_gap"], 0.0)
        self.assertEqual(true_equal_metrics["true_equal_outcome"], 1.0)
        self.assertEqual(true_equal_metrics["closer_to_true_equal_outcome_than_equal_split"], 1.0)
        self.assertGreater(equal_split_metrics["realized_outcome_gap"], 0.0)
        self.assertGreater(equal_split_metrics["outcome_distance_to_true_equal"], 0.0)
        self.assertEqual(equal_split_metrics["true_outcome_classification_tie"], 1.0)

    def test_manual_active_search_policy_samples_balanced_before_equal_outcome_choice(self) -> None:
        config = EnvironmentConfig(
            mu_need=40.0,
            sigma_need=20.0,
            sigma_sample=1.0,
            total_time=100.0,
            terminate_cost=1.0,
            sample_time_cost=0.1,
            prior_sample_count_1=0,
            prior_sample_count_2=0,
            random_seed=11,
        )
        observations = {
            ContinuousAllocationMetaMDP.SAMPLE_PERSON_1: [70.0],
            ContinuousAllocationMetaMDP.SAMPLE_PERSON_2: [30.0],
        }
        mdp = ContinuousAllocationMetaMDP(config, observation_streams=observations)
        result = mdp.run_episode(
            ManualActiveSearchEqualOutcomePolicy(samples_per_person=1),
            true_state=TrueState(need_1=70.0, need_2=30.0),
        )

        self.assertEqual(result.actions[:2], [mdp.SAMPLE_PERSON_1, mdp.SAMPLE_PERSON_2])
        self.assertEqual(result.actions[-1], mdp.TERMINATE)
        self.assertEqual(len(result.samples), 2)
        self.assertGreater(result.final_allocation_to_person1, 0.5)

    def test_r4_manual_advantage_candidate_selection(self) -> None:
        base = {
            "environment": "diagnostic_env",
            "regime_grid": "r4_diagnostic_active_search",
            "grid_index": 0.0,
            "mean_abs_allocation_from_equal": 0.1,
            "true_equal_outcome_rate": 0.9,
            "closer_to_true_equal_outcome_than_equal_split_rate": 0.8,
            "mean_outcome_distance_to_true_equal": 1.0,
            "mu_need": 40.0,
            "sigma_need": 60.0,
            "sigma_sample": 1.0,
            "total_time": 100.0,
            "sample_time_cost": 0.05,
            "utility_exponent": 0.25,
            "alpha": None,
            "learning_per_unit_of_tutoring": 1.0,
            "delta_learning_per_unit_tutoring": 0.0,
            "prior_sample_count_1": 0,
            "prior_sample_count_2": 0,
            "time_to_expected_need_ratio": 1.25,
            "need_variability_ratio": 1.5,
        }
        rows = [
            {
                **base,
                "policy": MANUAL_ACTIVE_POLICY_NAME,
                "mean_utility": 4.0,
                "mean_realized_outcome_gap": 2.0,
                "mean_sample_count": 4.0,
            },
            {
                **base,
                "policy": EQUAL_SPLIT_BASELINE_NAME,
                "mean_utility": 2.0,
                "mean_realized_outcome_gap": 8.0,
                "mean_sample_count": 0.0,
                "mean_abs_allocation_from_equal": 0.0,
                "true_equal_outcome_rate": 0.0,
                "closer_to_true_equal_outcome_than_equal_split_rate": 0.0,
                "mean_outcome_distance_to_true_equal": 7.0,
            },
            {
                **base,
                "policy": "myopic_voi",
                "policy_type": "rr_approximation",
                "mean_utility": 3.5,
                "mean_realized_outcome_gap": 3.0,
                "mean_sample_count": 2.0,
                "true_equal_outcome_rate": 0.7,
                "closer_to_true_equal_outcome_than_equal_split_rate": 0.6,
                "mean_outcome_distance_to_true_equal": 2.0,
            },
        ]

        candidates = identify_r4_manual_advantage_candidates(rows)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_type"], "manual_active_search_advantage")
        self.assertGreater(candidates[0]["manual_active_minus_equal_split_utility"], 0.0)
        self.assertLess(candidates[0]["rr_minus_manual_active_utility"], 0.0)


if __name__ == "__main__":
    unittest.main()
