"""Test purpose: validate held-out strategy comparisons, common randomness, and boundary summaries."""

from __future__ import annotations

import math
import random
import unittest
from dataclasses import asdict

from src.experiments import strategy_mapping as strategy_mapping_module

from src.experiments.strategy_mapping import (
    STRATEGY_MAPPING_POLICY_EQUAL_SPLIT,
    STRATEGY_MAPPING_POLICY_MANUAL,
    STRATEGY_MAPPING_POLICY_ORACLE,
    STRATEGY_MAPPING_POLICY_RR,
    ambiguous_close_true_equal_but_closer_equal_split,
    build_strategy_mapping_sigma_need_configs,
    evaluate_strategy_mapping_fixed_total_need_diagnostic,
    evaluate_strategy_mapping_four_way_environment,
    evaluate_strategy_mapping_sigma_need_sweep,
    select_strategy_mapping_primary_environments,
    summarize_strategy_mapping_fixed_total_need_diagnostic,
    summarize_strategy_mapping_four_way,
    summarize_strategy_mapping_sigma_need_sweep,
)
from src.experiments.randomization import build_evaluation_episode
from src.experiments.regimes import true_outcome_metrics_for_allocation
from src.mdp.meta_mdp import BeliefState, ContinuousAllocationMetaMDP, EnvironmentConfig, TrueState
from src.policies.voi import MyopicValueOfInformationPolicy


class FixedAllocationPolicy:
    name = "fixed_allocation"

    def __init__(self, allocation: float = 0.5):
        self.allocation = allocation

    def choose_action(self, mdp, belief):
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp, belief):
        return self.allocation


class RNGProbePolicy(FixedAllocationPolicy):
    def __init__(self):
        super().__init__(0.5)
        self.draws = []
        self.hidden_state_visible = []

    def choose_action(self, mdp, belief):
        self.hidden_state_visible.append(hasattr(mdp, "true_state"))
        self.draws.append(mdp.rng.gauss(0.0, 1.0))
        return mdp.TERMINATE


def smoke_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        mu_need=35.0,
        sigma_need=20.0,
        sigma_sample=2.0,
        total_time=100.0,
        sample_time_cost=0.25,
        terminate_cost=1.0,
        utility_exponent=0.35,
        learning_per_unit_of_tutoring=1.25,
        expected_utility_method="gauss_hermite",
        gauss_hermite_order=5,
        allocation_grid_size=21,
        max_meta_samples=8,
        random_seed=71,
    )


class StrategyMappingTests(unittest.TestCase):
    def test_four_way_uses_identical_episodes_and_time_matched_oracles(self) -> None:
        config = smoke_config()
        episodes = [
            build_evaluation_episode(
                config,
                episode_index=index,
                include_observation_streams=True,
                observations_per_person=12,
            )
            for index in range(3)
        ]
        rows = evaluate_strategy_mapping_four_way_environment(
            "smoke",
            config,
            episodes,
            rr_policy=FixedAllocationPolicy(0.4),
            oracle_grid_size=101,
        )
        self.assertEqual(len(rows), 12)
        for episode_index in range(3):
            episode_rows = [row for row in rows if row["episode_index"] == episode_index]
            self.assertEqual(
                {row["policy"] for row in episode_rows},
                {STRATEGY_MAPPING_POLICY_RR, STRATEGY_MAPPING_POLICY_MANUAL, STRATEGY_MAPPING_POLICY_EQUAL_SPLIT, STRATEGY_MAPPING_POLICY_ORACLE},
            )
            for field in (
                "need_1",
                "need_2",
                "episode_fingerprint",
                "observation_stream_hash_1",
                "observation_stream_hash_2",
            ):
                self.assertEqual(len({row[field] for row in episode_rows}), 1)
            rr = next(row for row in episode_rows if row["policy"] == STRATEGY_MAPPING_POLICY_RR)
            manual = next(row for row in episode_rows if row["policy"] == STRATEGY_MAPPING_POLICY_MANUAL)
            self.assertGreaterEqual(
                float(rr["rr_time_matched_oracle_utility"]),
                float(rr["realized_utility"]) - 1e-10,
            )
            self.assertGreaterEqual(
                float(manual["manual_time_matched_oracle_utility"]),
                float(manual["realized_utility"]) - 1e-10,
            )
            self.assertEqual(manual["online_sample_count"], 6)
            self.assertEqual(manual["sample_count_1"], 3)
            self.assertEqual(manual["sample_count_2"], 3)
            split = next(row for row in episode_rows if row["policy"] == STRATEGY_MAPPING_POLICY_EQUAL_SPLIT)
            oracle = next(row for row in episode_rows if row["policy"] == STRATEGY_MAPPING_POLICY_ORACLE)
            self.assertEqual(split["allocation_to_person1"], 0.5)
            self.assertEqual(split["online_sample_count"], 0)
            self.assertEqual(oracle["online_sample_count"], 0)

    def test_policy_computation_seed_is_independent_of_hidden_state_seed(self) -> None:
        config = smoke_config()
        episode = build_evaluation_episode(
            config,
            episode_index=0,
            include_observation_streams=True,
            observations_per_person=12,
        )
        policy = RNGProbePolicy()
        rows = evaluate_strategy_mapping_four_way_environment(
            "seed_isolation",
            config,
            [episode],
            rr_policy=policy,
            oracle_grid_size=101,
        )
        hidden_rng = random.Random((config.random_seed or 0) + 1)
        hidden_first_standard_draw = (
            hidden_rng.gauss(config.mu_need, config.sigma_need) - config.mu_need
        ) / config.sigma_need
        expected_policy_draw = random.Random((config.random_seed or 0) + 300_000).gauss(0.0, 1.0)
        self.assertAlmostEqual(policy.draws[0], expected_policy_draw)
        self.assertNotAlmostEqual(policy.draws[0], hidden_first_standard_draw)
        self.assertEqual(policy.hidden_state_visible, [False])
        rr = next(row for row in rows if row["policy"] == STRATEGY_MAPPING_POLICY_RR)
        self.assertEqual(int(rr["policy_computation_seed"]), (config.random_seed or 0) + 300_000)

    def test_policy_execution_order_does_not_change_scientific_rows(self) -> None:
        config = smoke_config()
        episodes = [
            build_evaluation_episode(
                config,
                episode_index=index,
                include_observation_streams=True,
                observations_per_person=12,
            )
            for index in range(2)
        ]
        forward = evaluate_strategy_mapping_four_way_environment(
            "order",
            config,
            episodes,
            rr_policy=MyopicValueOfInformationPolicy(observation_draws=2),
            oracle_grid_size=101,
        )
        reverse = evaluate_strategy_mapping_four_way_environment(
            "order",
            config,
            episodes,
            rr_policy=MyopicValueOfInformationPolicy(observation_draws=2),
            oracle_grid_size=101,
            execution_order=(
                STRATEGY_MAPPING_POLICY_ORACLE,
                STRATEGY_MAPPING_POLICY_EQUAL_SPLIT,
                STRATEGY_MAPPING_POLICY_MANUAL,
                STRATEGY_MAPPING_POLICY_RR,
            ),
        )
        key = lambda row: (int(row["episode_index"]), str(row["policy"]))
        self.assertEqual(sorted(forward, key=key), sorted(reverse, key=key))

    def test_four_way_summary_is_paired_and_keeps_empty_strata(self) -> None:
        config = smoke_config()
        episodes = [
            build_evaluation_episode(
                config,
                episode_index=index,
                include_observation_streams=True,
                observations_per_person=12,
            )
            for index in range(4)
        ]
        rows = evaluate_strategy_mapping_four_way_environment(
            "smoke",
            config,
            episodes,
            rr_policy=FixedAllocationPolicy(0.4),
            oracle_grid_size=101,
        )
        policy_summary, comparisons = summarize_strategy_mapping_four_way(rows)
        all_rr = next(
            row
            for row in policy_summary
            if row["policy"] == STRATEGY_MAPPING_POLICY_RR and row["stratum_dimension"] == "all"
        )
        self.assertEqual(all_rr["n_episodes"], 4)
        self.assertTrue(
            any(
                row["stratum_dimension"] == "oracle_sign"
                and row["stratum"] == "boundary_zero"
                for row in policy_summary
            )
        )
        rr_manual = next(
            row
            for row in comparisons
            if row["contrast"] == "rr_minus_manual" and row["stratum_dimension"] == "all"
        )
        expected = sum(
            float(next(row for row in rows if row["episode_index"] == index and row["policy"] == STRATEGY_MAPPING_POLICY_RR)["realized_utility"])
            - float(next(row for row in rows if row["episode_index"] == index and row["policy"] == STRATEGY_MAPPING_POLICY_MANUAL)["realized_utility"])
            for index in range(4)
        ) / 4
        self.assertAlmostEqual(float(rr_manual["mean_paired_utility_difference"]), expected)

    def test_recovery_classification_boundaries(self) -> None:
        policy_rows = [
            {
                "environment": "e",
                "stratum_dimension": "all",
                "policy": STRATEGY_MAPPING_POLICY_RR,
                "true_equal_outcome_rate": 0.8,
                "closer_to_true_equal_than_equal_split_rate": 0.8,
                "mean_online_sample_count": 1.01,
                "sampled_both_recipients_rate": 0.8,
                "mean_abs_allocation_from_equal": 0.05,
                "mean_utility": 2.0,
            },
            {
                "environment": "e",
                "stratum_dimension": "all",
                "policy": STRATEGY_MAPPING_POLICY_MANUAL,
                "true_equal_outcome_rate": 0.8,
                "closer_to_true_equal_than_equal_split_rate": 0.8,
                "sampled_both_recipients_rate": 0.8,
                "mean_abs_allocation_from_equal": 0.05,
                "mean_utility": 1.9,
            },
            {
                "environment": "e",
                "stratum_dimension": "all",
                "policy": STRATEGY_MAPPING_POLICY_ORACLE,
                "true_equal_outcome_rate": 0.8,
                "closer_to_true_equal_than_equal_split_rate": 0.8,
                "mean_utility": 2.1,
            },
        ]
        paired = [
            {
                "environment": "e",
                "stratum_dimension": "all",
                "contrast": "manual_minus_equal_split",
                "paired_utility_ci95_low": 0.01,
            },
            {
                "environment": "e",
                "stratum_dimension": "all",
                "contrast": "rr_90_percent_manual_improvement_recovery",
                "paired_utility_ci95_low": 0.0,
            },
        ]
        strategy_mapping_module._attach_recovery_classifications(policy_rows, paired)
        self.assertEqual(policy_rows[0]["strategy_mapping_recovery_classification"], "successful_strategy_recovery")

    def test_sigma_builder_changes_only_sigma_need(self) -> None:
        config = smoke_config()
        built = build_strategy_mapping_sigma_need_configs("anchor", config, [40.0, 10.0, 20.0])
        self.assertEqual([item[1].sigma_need for item in built], [10.0, 20.0, 40.0])
        base = asdict(config)
        for _, candidate in built:
            candidate_dict = asdict(candidate)
            for field, value in base.items():
                if field != "sigma_need":
                    self.assertEqual(candidate_dict[field], value)
        with self.assertRaises(ValueError):
            build_strategy_mapping_sigma_need_configs("anchor", config, [10.0, 10.0])
        with self.assertRaises(ValueError):
            build_strategy_mapping_sigma_need_configs("anchor", config, [0.0])
        with self.assertRaises(ValueError):
            build_strategy_mapping_sigma_need_configs("anchor", config, [math.inf])

    def test_sigma_sweep_preserves_standardized_randomness(self) -> None:
        config = smoke_config()
        rows = evaluate_strategy_mapping_sigma_need_sweep(
            "anchor",
            config,
            [5.0, 20.0, 60.0],
            n_episodes=4,
            rr_policy=FixedAllocationPolicy(0.4),
            oracle_grid_size=101,
            observations_per_person=12,
        )
        self.assertEqual(len(rows), 12)
        for episode_index in range(4):
            episode_rows = [row for row in rows if row["episode_index"] == episode_index]
            self.assertLess(
                max(float(row["standardized_need_draw_1"]) for row in episode_rows)
                - min(float(row["standardized_need_draw_1"]) for row in episode_rows),
                1e-10,
            )
            self.assertEqual(len({row["observation_residual_hash_1"] for row in episode_rows}), 1)
        environments, strata = summarize_strategy_mapping_sigma_need_sweep(
            rows,
            gap_bin_edges=(0.0, 10.0, 40.0, math.inf),
        )
        self.assertEqual(len(environments), 3)
        self.assertTrue(any(row["stratum_dimension"] == "realized_true_need_gap" for row in strata))
        self.assertTrue(any(row["stratum_dimension"] == "total_true_need" for row in strata))

    def test_fixed_total_need_constructs_balanced_nonnegative_states(self) -> None:
        config = smoke_config()
        rows = evaluate_strategy_mapping_fixed_total_need_diagnostic(
            "mechanism",
            config,
            n_episodes_per_difference=4,
            rr_policy=FixedAllocationPolicy(0.4),
            need_differences=(0.0, 20.0, 60.0),
            oracle_grid_size=101,
            observations_per_person=12,
        )
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(abs(float(row["need_1"]) + float(row["need_2"]) - 70.0) < 1e-10 for row in rows))
        self.assertTrue(all(float(row["need_1"]) >= 0.0 and float(row["need_2"]) >= 0.0 for row in rows))
        for episode_index in range(4):
            episode_rows = [row for row in rows if row["episode_index"] == episode_index]
            self.assertEqual(len({row["observation_residual_hash_1"] for row in episode_rows}), 1)
        summaries = summarize_strategy_mapping_fixed_total_need_diagnostic(rows)
        all_rows = [row for row in summaries if row["stratum_dimension"] == "all"]
        self.assertEqual(
            [row["constructed_need_difference"] for row in all_rows],
            [0.0, 20.0, 60.0],
        )
        trend = next(row for row in summaries if row["stratum_dimension"] == "mechanism_trend")
        self.assertIn("difference_allocation_closeness_correlation", trend)
        by_difference = {
            float(row["constructed_need_difference"]): row
            for row in all_rows
        }
        expected_change = sum(
            float(
                next(
                    row
                    for row in rows
                    if row["constructed_need_difference"] == 20.0
                    and row["episode_index"] == episode_index
                )["allocation_closeness_advantage"]
            )
            - float(
                next(
                    row
                    for row in rows
                    if row["constructed_need_difference"] == 0.0
                    and row["episode_index"] == episode_index
                )["allocation_closeness_advantage"]
            )
            for episode_index in range(4)
        ) / 4.0
        self.assertAlmostEqual(
            float(by_difference[20.0]["mean_paired_allocation_closeness_change"]),
            expected_change,
        )
        self.assertEqual(
            float(by_difference[20.0]["mean_paired_ambiguous_event_change"]),
            0.0,
        )

        with self.assertRaises(ValueError):
            evaluate_strategy_mapping_fixed_total_need_diagnostic(
                "mechanism",
                config,
                n_episodes_per_difference=3,
                rr_policy=FixedAllocationPolicy(),
            )
        with self.assertRaises(ValueError):
            evaluate_strategy_mapping_fixed_total_need_diagnostic(
                "mechanism",
                config,
                n_episodes_per_difference=2,
                rr_policy=FixedAllocationPolicy(),
                need_differences=(10.0, 10.0),
            )

    def test_closeness_advantage_sign_convention(self) -> None:
        config = smoke_config()
        mdp = ContinuousAllocationMetaMDP(config)
        true_state = TrueState(55.0, 15.0)
        belief = BeliefState(
            mean_1=35.0,
            var_1=400.0,
            mean_2=35.0,
            var_2=400.0,
            deliberation_time=config.terminate_cost,
            history=[{"action": 0.0, "observation": math.nan, "cost": config.terminate_cost}],
        )
        true_equal_allocation = float(
            true_outcome_metrics_for_allocation(
                mdp,
                true_state,
                belief,
                0.5,
                allocation_tolerance=0.05,
            )["true_equal_outcome_allocation"]
        )
        true_equal_metrics = true_outcome_metrics_for_allocation(
            mdp,
            true_state,
            belief,
            true_equal_allocation,
            allocation_tolerance=0.05,
        )
        split_metrics = true_outcome_metrics_for_allocation(
            mdp,
            true_state,
            belief,
            0.5,
            allocation_tolerance=0.05,
        )
        true_equal_advantage = abs(true_equal_allocation - 0.5) - float(
            true_equal_metrics["true_equal_outcome_allocation_gap"]
        )
        split_advantage = 0.0 - float(split_metrics["true_equal_outcome_allocation_gap"])
        self.assertGreater(true_equal_advantage, 0.0)
        self.assertLess(split_advantage, 0.0)
        self.assertEqual(
            ambiguous_close_true_equal_but_closer_equal_split(0.51, 0.54, 0.05),
            1.0,
        )
        self.assertEqual(
            ambiguous_close_true_equal_but_closer_equal_split(0.53, 0.54, 0.05),
            0.0,
        )

    def test_primary_selection_matches_frozen_diagnostic_active_search_schema(self) -> None:
        rows = [
            {
                "environment": name,
                "manual_active_minus_equal_split_utility": utility,
                "manual_active_true_equal_outcome_rate": 0.9,
                "manual_active_closer_to_true_equal_rate": 0.85,
                "manual_active_mean_sample_count": 6.0,
                "manual_active_mean_abs_allocation_from_equal": 0.1,
            }
            for name, utility in (("b", 2.0), ("a", 2.0), ("c", 1.0), ("d", -1.0))
        ]
        selected = select_strategy_mapping_primary_environments(rows)
        self.assertEqual([row["environment"] for row in selected], ["a", "b", "c"])

if __name__ == "__main__":
    unittest.main()
