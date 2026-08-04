from __future__ import annotations

import json
import math
import tempfile
import unittest
import weakref
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from scripts.analyze_r5 import (
    build_existing_oat_slices,
    build_formal_sweep_rows,
    build_solver_comparison_rows,
    fixed_budget_evidence,
    select_existing_rr_anchor,
    validate_solver_comparison_inputs,
)
from scripts.r5_array_workflow import configs_from_json, parse_sample_budgets, rr_policy_from_manifest
from src.experiments.diagnostics import summarize_r4_diagnostic_policies
from src.experiments.r5 import (
    deterministic_realized_utility,
    evaluation_episode_fingerprint,
    evaluate_r5_rr_environment,
    evaluate_r5_fixed_sampling_budgets,
    full_information_oracle_metrics,
    full_information_utilitarian_allocation,
    summarize_r5_fixed_sampling_budgets,
    wilson_interval,
)
from src.experiments.randomization import (
    EvaluationEpisode,
    build_evaluation_episode,
    required_observations_per_person,
)
from src.experiments.regimes import true_outcome_metrics_for_allocation
from src.experiments.sweeps import build_r5_six_sample_configs
from src.mdp.meta_mdp import (
    BeliefState,
    ContinuousAllocationMetaMDP,
    EnvironmentConfig,
    TrueState,
    utility,
)
from src.solvers.gauss_hermite import (
    expected_terminal_utility_gauss_hermite,
    independent_normal_expectation_2d,
)
from src.solvers.dp import DiscretizedDynamicProgrammingPolicy, FiniteHorizonDPSolver


class AlwaysSamplePolicy:
    name = "always_sample"

    def choose_action(self, mdp, belief):
        return mdp.SAMPLE_PERSON_1

    def choose_final_allocation(self, mdp, belief):
        return 0.5


class AlwaysTerminatePolicy:
    name = "always_terminate"

    def choose_action(self, mdp, belief):
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp, belief):
        return 0.5


class R5DiagnosticsTests(unittest.TestCase):
    def test_evaluation_episode_caps_stream_at_execution_horizon(self) -> None:
        config = EnvironmentConfig(
            total_time=160.0,
            sample_time_cost=0.0016,
            sigma_sample=20.0,
            random_seed=11,
        )
        episode = build_evaluation_episode(
            config,
            episode_index=0,
            include_observation_streams=True,
            observations_per_person=100,
            max_online_samples=100,
        )
        self.assertIsNotNone(episode.observation_streams)
        self.assertEqual(len(episode.observation_streams["sample_1"]), 105)
        self.assertEqual(len(episode.observation_streams["sample_2"]), 105)

    def test_rr_environment_releases_completed_observation_streams(self) -> None:
        from src.experiments import r5 as r5_module

        original_builder = r5_module.build_evaluation_episode
        episode_refs = []
        live_before_build = []

        def tracked_builder(*args, **kwargs):
            live_before_build.append(sum(ref() is not None for ref in episode_refs))
            episode = original_builder(*args, **kwargs)
            episode_refs.append(weakref.ref(episode))
            return episode

        config = EnvironmentConfig(
            total_time=10.0,
            sample_time_cost=1.0,
            expected_utility_method="gauss_hermite",
            allocation_grid_size=11,
            random_seed=17,
        )
        with patch.object(r5_module, "build_evaluation_episode", side_effect=tracked_builder):
            rows = evaluate_r5_rr_environment(
                "streaming_test",
                config,
                n_episodes=3,
                rr_policy=AlwaysTerminatePolicy(),
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual(live_before_build, [0, 1, 1])

    def test_separable_gauss_hermite_matches_two_dimensional_formula(self) -> None:
        config = EnvironmentConfig(
            expected_utility_method="gauss_hermite",
            gauss_hermite_order=7,
            allocation_grid_size=31,
        )
        mdp = ContinuousAllocationMetaMDP(config)
        belief = BeliefState(42.0, 25.0, 55.0, 36.0, deliberation_time=2.0)
        allocation = 0.37
        remaining_time = mdp.remaining_time_after_termination(belief)
        amount_1, amount_2 = mdp.allocation_to_learning_outcomes(
            allocation,
            remaining_time,
        )
        alpha = mdp.utility_exponent()
        reference = independent_normal_expectation_2d(
            belief.mean_1,
            belief.var_1,
            belief.mean_2,
            belief.var_2,
            lambda need_1, need_2: utility(
                amount_1 - need_1,
                config.lambda_shortfall,
                alpha,
            )
            + utility(amount_2 - need_2, config.lambda_shortfall, alpha),
            order=7,
        )
        separable = expected_terminal_utility_gauss_hermite(
            mdp,
            belief,
            allocation,
            order=7,
        )
        self.assertAlmostEqual(separable, reference, places=11)

    def test_vectorized_gauss_hermite_terminal_solver_matches_grid(self) -> None:
        config = EnvironmentConfig(
            expected_utility_method="gauss_hermite",
            gauss_hermite_order=7,
            allocation_grid_size=31,
        )
        mdp = ContinuousAllocationMetaMDP(config)
        belief = BeliefState(42.0, 25.0, 55.0, 36.0, deliberation_time=2.0)
        allocation, value = mdp.solve_terminal_allocation(belief)
        grid = [index / (config.allocation_grid_size - 1) for index in range(config.allocation_grid_size)]
        reference_values = [
            expected_terminal_utility_gauss_hermite(
                mdp,
                belief,
                candidate,
                order=config.gauss_hermite_order,
            )
            for candidate in grid
        ]
        self.assertAlmostEqual(value, max(reference_values), places=11)
        self.assertAlmostEqual(value, reference_values[grid.index(allocation)], places=11)

    def test_zero_sampling_cost_requires_explicit_cap(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentConfig(sample_time_cost=0.0)

        config = EnvironmentConfig(sample_time_cost=0.0, max_meta_samples=6)
        self.assertEqual(required_observations_per_person(config, 2), 11)

    def test_zero_cost_episode_terminates_at_configured_cap(self) -> None:
        config = EnvironmentConfig(
            sample_time_cost=0.0,
            max_meta_samples=6,
            terminate_cost=1.0,
            expected_utility_draws=20,
            allocation_grid_size=21,
            random_seed=7,
        )
        result = ContinuousAllocationMetaMDP(config).run_episode(
            AlwaysSamplePolicy(),
            true_state=TrueState(40.0, 40.0),
            max_steps=8,
        )
        self.assertEqual(len(result.samples), 6)
        self.assertEqual(result.actions[-1], ContinuousAllocationMetaMDP.TERMINATE)

    def test_full_information_positive_symmetric_case_selects_equal_split(self) -> None:
        config = EnvironmentConfig(
            total_time=201.0,
            terminate_cost=1.0,
            utility_exponent=0.5,
            allocation_grid_size=101,
        )
        mdp = ContinuousAllocationMetaMDP(config)
        allocation, _ = full_information_utilitarian_allocation(
            mdp,
            TrueState(40.0, 40.0),
            remaining_time=200.0,
            grid_size=4001,
        )
        self.assertAlmostEqual(allocation, 0.5, places=4)

    def test_full_information_symmetric_shortfall_need_not_select_equal_split(self) -> None:
        config = EnvironmentConfig(
            total_time=61.0,
            terminate_cost=1.0,
            utility_exponent=0.5,
            lambda_shortfall=2.0,
        )
        mdp = ContinuousAllocationMetaMDP(config)
        allocation, _ = full_information_utilitarian_allocation(
            mdp,
            TrueState(100.0, 100.0),
            remaining_time=60.0,
            grid_size=4001,
        )
        self.assertTrue(allocation <= 0.01 or allocation >= 0.99)

    def test_full_information_oracle_includes_off_grid_equal_outcome_candidate(self) -> None:
        config = EnvironmentConfig(
            total_time=161.0,
            terminate_cost=1.0,
            utility_exponent=0.25,
            lambda_shortfall=2.25,
        )
        mdp = ContinuousAllocationMetaMDP(config)
        state = TrueState(43.17, 61.93)
        allocation, oracle_utility = full_information_utilitarian_allocation(
            mdp,
            state,
            remaining_time=160.0,
            grid_size=101,
        )
        equal_allocation = (state.need_1 - state.need_2 + 160.0) / 320.0
        equal_utility = deterministic_realized_utility(
            mdp,
            state,
            equal_allocation,
            160.0,
        )
        self.assertGreaterEqual(oracle_utility + 1e-12, equal_utility)
        self.assertTrue(0.0 <= allocation <= 1.0)

    def test_oracle_row_reports_sign_feasibility_and_negative_need(self) -> None:
        config = EnvironmentConfig(total_time=101.0, terminate_cost=1.0)
        episode = EvaluationEpisode(0, TrueState(-1.0, 40.0))
        row = full_information_oracle_metrics("fixture", config, episode, grid_size=1001)
        self.assertIn(row["oracle_sign_stratum"], {
            "both_positive", "both_negative", "mixed_sign", "boundary_zero"
        })
        self.assertEqual(row["negative_need_either"], 1.0)
        self.assertIn(row["exact_true_equal_outcome_feasible"], {0.0, 1.0})
        self.assertGreaterEqual(row["true_equal_outcome_regret"], 0.0)
        self.assertGreaterEqual(row["oracle_grid_optimality_violation"], 0.0)

    def test_six_sample_grid_contains_oracle_center_and_cost_limits(self) -> None:
        configs = [config for _, config in build_r5_six_sample_configs()]
        self.assertEqual(len(configs), 324)
        self.assertTrue(
            any(
                config.mu_need == 60.0
                and config.sigma_need == 30.0
                and config.total_time == 160.0
                and config.utility_exponent == 0.5
                for config in configs
            )
        )
        zero_cost = [config for config in configs if config.sample_time_cost == 0.0]
        self.assertTrue(zero_cost)
        self.assertTrue(all(config.max_meta_samples == 12 for config in zero_cost))

    def test_custom_confirmation_configs_are_named_and_unique(self) -> None:
        config = EnvironmentConfig(random_seed=17)
        payload = {
            "configs": [
                {"environment": "candidate_a", "config": asdict(config)},
                {"environment": "candidate_b", "config": asdict(replace(config, random_seed=18))},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configs.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = configs_from_json(path)
        self.assertEqual([name for name, _ in loaded], ["candidate_a", "candidate_b"])
        self.assertEqual([item.random_seed for _, item in loaded], [17, 18])

    def test_fixed_budget_curve_uses_common_episodes_and_exact_budgets(self) -> None:
        config = EnvironmentConfig(
            mu_need=20.0,
            sigma_need=5.0,
            sigma_sample=3.0,
            total_time=50.0,
            sample_time_cost=0.01,
            max_meta_samples=6,
            allocation_grid_size=21,
            expected_utility_draws=10,
            random_seed=19,
        )
        rows = evaluate_r5_fixed_sampling_budgets(
            "fixture",
            config,
            n_episodes=3,
            total_sample_budgets=[0, 2, 4, 6],
        )
        self.assertEqual(len(rows), 12)
        for episode_index in range(3):
            episode_rows = [row for row in rows if row["episode_index"] == episode_index]
            self.assertEqual(
                [row["online_sample_count"] for row in episode_rows],
                [0, 2, 4, 6],
            )
            self.assertEqual(len({(row["need_1"], row["need_2"]) for row in episode_rows}), 1)
        summaries = summarize_r5_fixed_sampling_budgets(rows)
        self.assertEqual([row["sampling_budget_total"] for row in summaries], [0, 2, 4, 6])
        self.assertTrue(math.isnan(summaries[0]["mean_incremental_utility_vs_previous_budget"]))
        self.assertEqual(summaries[-1]["previous_sampling_budget_total"], 4)

    def test_formal_summary_recovers_frozen_anchor_and_feature_identity(self) -> None:
        base = EnvironmentConfig()
        summary = {
            "n_episodes": 120,
            "mean_utility": 1.0,
            "mean_utility_ci95": 0.1,
            "mean_sample_count": 6.0,
            "mean_abs_allocation_from_equal": 0.1,
            "sample_time_cost": 0.1,
            "sample_time_cost_percent": 0.1,
            "true_equal_outcome_rate": 0.82,
            "true_equal_outcome_ci95_low": 0.75,
            "true_equal_outcome_ci95_high": 0.88,
            "closer_to_true_equal_outcome_than_equal_split_rate": 0.81,
            "closer_to_true_equal_outcome_than_equal_split_ci95_low": 0.74,
            "closer_to_true_equal_outcome_than_equal_split_ci95_high": 0.87,
            "sample_count_at_least_6_rate": 0.7,
            "r5_joint_discovery_candidate": 1.0,
        }
        sampling_name = "anchor_a_sample_pct=0.1"
        oat_name = "sigma_need_anchor_a_sigma_need=30"
        sampling_manifest = {
            "environments": [{"environment": sampling_name, "config": asdict(base)}]
        }
        oat_manifest = {
            "environments": [
                {
                    "environment": oat_name,
                    "config": asdict(replace(base, sigma_need=30.0)),
                }
            ]
        }
        rows = build_formal_sweep_rows(
            [{**summary, "environment": sampling_name}],
            sampling_manifest,
            [{**summary, "environment": oat_name}],
            oat_manifest,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["anchor"], "anchor_a")
        self.assertEqual(rows[0]["feature"], "sigma_need")
        self.assertEqual(rows[1]["feature"], "sample_time_cost_percent")
        self.assertEqual(rows[1]["r5_joint_0_8_0_8"], 1.0)

    def test_fixed_budget_evidence_requires_positive_paired_lower_bound(self) -> None:
        rows = [
            {
                "environment": "supported",
                "sampling_budget_total": 6,
                "previous_sampling_budget_total": 4,
                "mean_incremental_utility_vs_previous_budget": 0.10,
                "paired_incremental_utility_ci95": 0.04,
                "true_equal_outcome_rate": 0.75,
                "closer_to_true_equal_outcome_than_equal_split_rate": 0.82,
            },
            {
                "environment": "uncertain",
                "sampling_budget_total": 6,
                "previous_sampling_budget_total": 4,
                "mean_incremental_utility_vs_previous_budget": 0.02,
                "paired_incremental_utility_ci95": 0.05,
                "true_equal_outcome_rate": 0.80,
                "closer_to_true_equal_outcome_than_equal_split_rate": 0.80,
            },
        ]
        evidence = fixed_budget_evidence(rows)
        by_environment = {row["environment"]: row for row in evidence}
        self.assertEqual(by_environment["supported"]["six_observations_useful"], 1.0)
        self.assertEqual(by_environment["uncertain"]["six_observations_useful"], 0.0)

    def test_sample_budget_parser_rejects_odd_values(self) -> None:
        self.assertEqual(parse_sample_budgets("6,0,2,6"), [0, 2, 6])
        with self.assertRaises(ValueError):
            parse_sample_budgets("0,3,6")

    def test_closer_classifier_uses_only_numerical_ties(self) -> None:
        config = EnvironmentConfig(total_time=101.0, terminate_cost=1.0)
        mdp = ContinuousAllocationMetaMDP(config)
        belief = BeliefState(40.0, 1.0, 60.0, 1.0, deliberation_time=0.0)
        state = TrueState(40.0, 60.0)
        equal_allocation = (state.need_1 - state.need_2 + 100.0) / 200.0
        metrics = true_outcome_metrics_for_allocation(
            mdp,
            state,
            belief,
            equal_allocation + 0.05,
            allocation_tolerance=0.0501,
        )
        self.assertEqual(metrics["true_equal_outcome"], 1.0)
        self.assertEqual(metrics["closer_to_true_equal_outcome_than_equal_split"], 1.0)
        self.assertEqual(
            metrics["legacy_tolerance_closer_to_true_equal_outcome_than_equal_split"],
            0.0,
        )

    def test_wilson_confirmation_bound_is_conservative(self) -> None:
        lower, upper = wilson_interval(960, 1200, one_sided=True)
        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 0.8)
        stronger_lower, _ = wilson_interval(1000, 1200, one_sided=True)
        self.assertGreater(stronger_lower, 0.8)

    def test_r4_summary_uses_mean_absolute_not_absolute_mean(self) -> None:
        common = {
            "environment": "fixture",
            "regime_grid": "fixture",
            "grid_index": 0.0,
            "mean_utility": 0.0,
            "mean_realized_outcome_gap": 0.0,
            "mean_sample_count": 0.0,
            "true_equal_outcome_rate": 1.0,
            "closer_to_true_equal_outcome_than_equal_split_rate": 1.0,
            "mean_true_equal_outcome_allocation": 0.5,
            "mean_abs_true_equal_outcome_allocation_from_equal_split": 0.2,
        }
        rows = [
            {**common, "policy": "manual_active_search_equal_outcome"},
            {**common, "policy": "manual_equal_split"},
        ]
        summary = summarize_r4_diagnostic_policies(rows)[0]
        self.assertAlmostEqual(summary["mean_abs_true_equal_allocation_from_equal_split"], 0.2)

    def test_dp_prior_samples_are_free_and_do_not_consume_online_horizon(self) -> None:
        config = EnvironmentConfig(
            sigma_need=20.0,
            sigma_sample=10.0,
            sample_time_cost=1.0,
            prior_sample_count_1=5,
            prior_sample_count_2=5,
            expected_utility_draws=20,
            allocation_grid_size=21,
        )
        mdp = ContinuousAllocationMetaMDP(config)
        belief = mdp.initial_belief()
        solver = FiniteHorizonDPSolver(mdp, max_samples=10, mean_grid_size=5)
        state = solver.state_from_belief(belief)
        reconstructed = solver.belief_from_state(state)
        self.assertEqual(state.samples_1, 5)
        self.assertEqual(state.samples_2, 5)
        self.assertEqual(state.online_samples_1 + state.online_samples_2, 0)
        self.assertEqual(reconstructed.deliberation_time, 0.0)
        self.assertEqual(solver.max_samples - state.online_samples_1 - state.online_samples_2, 10)

    def test_dp_gauss_hermite_observation_weights_match_predictive_mean(self) -> None:
        config = EnvironmentConfig(
            mu_need=40.0,
            sigma_need=10.0,
            sigma_sample=5.0,
            expected_utility_draws=20,
        )
        mdp = ContinuousAllocationMetaMDP(config)
        solver = FiniteHorizonDPSolver(
            mdp,
            max_samples=2,
            mean_grid_size=7,
            observation_branches=7,
            observation_integration="gauss_hermite",
        )
        nodes_weights = solver._observation_nodes_weights(mean=42.0, var=16.0)
        self.assertAlmostEqual(sum(weight for _, weight in nodes_weights), 1.0, places=12)
        self.assertAlmostEqual(
            sum(node * weight for node, weight in nodes_weights),
            42.0,
            places=12,
        )

    def test_dp_rejects_unknown_observation_integration(self) -> None:
        mdp = ContinuousAllocationMetaMDP(EnvironmentConfig(expected_utility_draws=20))
        with self.assertRaises(ValueError):
            FiniteHorizonDPSolver(mdp, observation_integration="unknown")

    def test_r5_manifest_builds_frozen_gauss_hermite_dp(self) -> None:
        manifest = {
            "observation_draws": 500,
            "rr_policy": {
                "name": "discretized_dp",
                "dp_max_samples": 10,
                "dp_mean_grid_size": 50,
                "dp_mean_grid_radius_sd": 3.0,
                "dp_observation_branches": 7,
                "dp_observation_integration": "gauss_hermite",
            },
        }
        policy = rr_policy_from_manifest(manifest)
        self.assertIsInstance(policy, DiscretizedDynamicProgrammingPolicy)
        self.assertEqual(policy.max_samples, 10)
        self.assertEqual(policy.mean_grid_size, 50)
        self.assertEqual(policy.observation_branches, 7)
        self.assertEqual(policy.observation_integration, "gauss_hermite")

    def test_r5_episode_fingerprint_changes_with_observation_stream(self) -> None:
        state = TrueState(need_1=40.0, need_2=50.0)
        first = EvaluationEpisode(
            episode_index=3,
            true_state=state,
            observation_streams={"sample_1": [39.0], "sample_2": [51.0]},
        )
        second = EvaluationEpisode(
            episode_index=3,
            true_state=state,
            observation_streams={"sample_1": [39.5], "sample_2": [51.0]},
        )
        self.assertNotEqual(
            evaluation_episode_fingerprint(first)[0],
            evaluation_episode_fingerprint(second)[0],
        )

    def test_solver_comparison_rejects_common_randomness_mismatch(self) -> None:
        manifest = {
            "family": "custom_rr",
            "episodes_per_environment": 1,
            "allocation_tolerance": 0.05,
            "seed_namespace_offset": 8,
            "configs_source_hash": "same",
            "git_commit": "same",
            "rr_policy": {"name": "myopic_voi"},
            "environments": [
                {
                    "environment": "test",
                    "config": EnvironmentConfig().__dict__,
                }
            ],
        }
        row = {
            "environment": "test",
            "episode_index": 0,
            "need_1": 40.0,
            "need_2": 50.0,
            "episode_fingerprint": "first",
            "observation_stream_hash_1": "one",
            "observation_stream_hash_2": "two",
        }
        dp_manifest = {**manifest, "rr_policy": {"name": "discretized_dp"}}
        mismatch = {**row, "episode_fingerprint": "second"}
        with self.assertRaisesRegex(RuntimeError, "Common-randomness mismatch"):
            validate_solver_comparison_inputs([row], [mismatch], manifest, dp_manifest)

    def test_solver_comparison_uses_paired_utility_differences(self) -> None:
        def episode(policy: str, episode_index: int, utility_value: float) -> dict:
            return {
                "environment": "test",
                "episode_index": episode_index,
                "need_1": 40.0,
                "need_2": 50.0,
                "episode_fingerprint": f"episode-{episode_index}",
                "observation_stream_hash_1": f"one-{episode_index}",
                "observation_stream_hash_2": f"two-{episode_index}",
                "realized_utility": utility_value,
                "online_sample_count": 2.0 if policy == "dp" else 1.0,
                "abs_allocation_from_equal": 0.1,
                "exact_true_equal_outcome_feasible": 1.0,
                "negative_need_either": 0.0,
                "sample_time_cost": 0.1,
                "sample_time_cost_percent": 0.1,
                "max_meta_samples": "",
                "true_equal_outcome": 1.0,
                "closer_to_true_equal_outcome_than_equal_split": 1.0,
                "sample_count_at_least_6": 0.0,
            }

        pairs = [
            (episode("myopic", 0, 1.0), episode("dp", 0, 1.5)),
            (episode("myopic", 1, 3.0), episode("dp", 1, 3.5)),
        ]
        rows = build_solver_comparison_rows(pairs)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["paired_dp_minus_myopic_mean_utility"], 0.5)
        self.assertAlmostEqual(rows[0]["paired_dp_minus_myopic_utility_ci95"], 0.0)

    def test_existing_result_oat_holds_other_features_fixed(self) -> None:
        base = {
            "environment": "anchor",
            "mu_need": 35.0,
            "total_time": 100.0,
            "learning_per_unit_of_tutoring": 1.0,
            "sigma_need": 40.0,
            "sigma_sample": 2.0,
            "sample_time_cost": 0.05,
            "utility_exponent": 0.35,
            "true_equal_outcome_rate": 0.82,
            "closer_to_true_equal_outcome_than_equal_split_rate": 0.81,
            "mean_sample_count": 3.0,
            "mean_utility": 1.0,
        }
        rows = [
            base,
            {
                **base,
                "environment": "lower_cost",
                "sample_time_cost": 0.02,
                "true_equal_outcome_rate": 0.80,
                "closer_to_true_equal_outcome_than_equal_split_rate": 0.79,
            },
            {
                **base,
                "environment": "confounded",
                "sample_time_cost": 0.10,
                "sigma_sample": 4.0,
                "true_equal_outcome_rate": 0.95,
                "closer_to_true_equal_outcome_than_equal_split_rate": 0.95,
            },
        ]
        anchor = select_existing_rr_anchor(rows[:2])
        self.assertEqual(anchor["environment"], "anchor")
        slices = build_existing_oat_slices(rows, anchor)
        cost_slice = [row for row in slices if row["feature"] == "sample_time_cost"]
        self.assertEqual(
            [row["environment"] for row in cost_slice],
            ["lower_cost", "anchor"],
        )
        self.assertAlmostEqual(cost_slice[0]["sample_time_cost_percent"], 0.02)
        self.assertNotIn("confounded", [row["environment"] for row in cost_slice])


if __name__ == "__main__":
    unittest.main()
