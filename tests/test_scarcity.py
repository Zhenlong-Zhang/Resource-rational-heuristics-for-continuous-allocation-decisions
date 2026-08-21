"""Focused tests for scarcity policies, oracle rules, and frozen inference."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from src.experiments import scarcity as scarcity_module
from src.experiments.active_search_evaluation import deterministic_realized_utility
from src.experiments.scarcity import (
    SCARCITY_POLICY_ORDER,
    ScarcityError,
    attach_holm_adjustment,
    build_confirmation_descriptors,
    build_deterministic_mechanism_cases,
    build_development_descriptors,
    build_gaussian_oracle_descriptors,
    build_scarcity_paired_episode,
    canonical_hash,
    classify_metalevel_target,
    continuous_gap_summary,
    evaluate_metalevel_descriptor,
    evaluate_metalevel_episode,
    gaussian_nonpositive_probabilities,
    holm_adjust_p_values,
    object_level_stop_decision,
    paired_contrast_summary,
    scarcity_allocation_metrics,
    scarcity_oracle_comparison_row,
    scarcity_pairing_group_id,
    scarcity_policy_seed,
    select_confirmation_targets,
    select_gaussian_oracle_anchors,
)
from src.mdp.meta_mdp import BeliefState, ContinuousAllocationMetaMDP, EnvironmentConfig, TrueState
from src.policies.heuristic import (
    ImmediateAllToLowerPolicy,
    ImmediateMeetLowerFirstPolicy,
    ManualActiveSearchAllToLowerPolicy,
    ManualActiveSearchMeetLowerFirstPolicy,
    all_to_lower_allocation,
    effort_to_goal,
    greatest_effort_need_allocation,
    lower_effort_identity,
    meet_lower_first_allocation,
)


def scarcity_config(**overrides: object) -> EnvironmentConfig:
    values = {
        "mu_need": 100.0,
        "sigma_need": 10.0,
        "sigma_sample": 10.0,
        "total_time": 101.0,
        "terminate_cost": 1.0,
        "sample_time_cost": 1.0,
        "utility_exponent": 0.5,
        "lambda_shortfall": 2.0,
        "learning_per_unit_of_tutoring": 1.0,
        "delta_learning_per_unit_tutoring": 0.0,
        "expected_utility_draws": 500,
        "allocation_grid_size": 401,
        "max_meta_samples": 40,
    }
    values.update(overrides)
    return EnvironmentConfig(**values)


def oracle_summary_row(
    environment_id: str,
    *,
    capacity_ratio: float,
    exact_all: bool = False,
    exact_meet: bool = False,
    direction: bool = False,
    upper: float = 0.95,
    sigma_need: float = 10.0,
    exponent: float = 0.5,
    lambda_shortfall: float = 2.0,
) -> dict[str, object]:
    return {
        "environment_id": environment_id,
        "anchor_id": f"source_{environment_id}",
        "capacity_ratio": capacity_ratio,
        "sigma_need": sigma_need,
        "utility_exponent": exponent,
        "lambda_shortfall": lambda_shortfall,
        "all_to_lower_exact_label_eligible": exact_all,
        "meet_lower_first_exact_label_eligible": exact_meet,
        "direction_supported": direction,
        "oracle_more_to_lower_one_sided_95_high": upper,
    }


def development_row(
    environment_id: str,
    acquisition_class: str,
    *,
    exact: bool,
    direction: bool,
    sigma_sample: float,
    cost: float,
    prior: int,
    anchor_id: str = "anchor_a",
    anchor_rule: str = "all_to_lower",
    g_min: float = 1.0,
    max_d_min: float = 0.5,
    acquisition_rate: float = 0.9,
    more_rate: float = 0.9,
    exact_rate: float = 0.9,
) -> dict[str, object]:
    return {
        "environment_id": environment_id,
        "acquisition_class": acquisition_class,
        "exact_candidate": exact,
        "direction_candidate": direction,
        "g_min": g_min,
        "max_d_min": max_d_min,
        "rr_acquisition_rate": acquisition_rate,
        "rr_more_to_lower_rate": more_rate,
        "rr_all_to_lower_match_rate": exact_rate,
        "rr_meet_lower_first_match_rate": exact_rate - 0.1,
        "diagnostic_exact_policy": "all_to_lower",
        "anchor_id": anchor_id,
        "source_anchor_id": f"source_{anchor_id}",
        "pairing_group_id": f"development_pair_{anchor_id}",
        "anchor_rule": anchor_rule,
        "sigma_need": 10.0,
        "capacity_ratio": 0.5,
        "utility_exponent": 0.5,
        "lambda_shortfall": 2.0,
        "sigma_sample": sigma_sample,
        "sample_time_cost_percent": cost,
        "prior_sample_count": prior,
    }


def confirmation_summary(
    *,
    acquisition_class: str = "no_search",
    acquisition_low: float = 0.85,
    exact_low: float = 0.85,
    direction_low: float = 0.85,
    direction_high: float = 0.95,
    gain_low: float = 0.1,
    retained_low: float = 0.0,
) -> dict[str, object]:
    result: dict[str, object] = {
        "acquisition_class": acquisition_class,
        "environment_id": f"confirmation_{acquisition_class}",
        "development_environment_id": f"development_{acquisition_class}",
        "rr_acquisition_one_sided_95_low": acquisition_low,
        "rr_more_to_lower_one_sided_95_low": direction_low,
        "rr_more_to_lower_one_sided_95_high": direction_high,
        "rr_all_to_lower_match_one_sided_95_low": exact_low,
        "rr_meet_lower_first_match_one_sided_95_low": exact_low,
        "diagnostic_exact_policy": "all_to_lower",
        "equal_split_mean_utility_full_sample": 1.0,
        "greatest_need_mean_utility_full_sample": 0.5,
        "gain_vs_equal_split_mean": 0.5,
        "gain_vs_equal_split_ci95_low": 0.3,
        "gain_vs_equal_split_ci95_high": 0.7,
        "gain_vs_greatest_need_mean": 1.0,
        "gain_vs_greatest_need_ci95_low": 0.8,
        "gain_vs_greatest_need_ci95_high": 1.2,
        "manual_gain_all_to_lower_vs_equal_split_mean": 0.45,
        "manual_gain_all_to_lower_vs_equal_split_ci95_low": 0.25,
        "manual_gain_all_to_lower_vs_equal_split_ci95_high": 0.65,
        "manual_gain_all_to_lower_vs_greatest_need_mean": 0.95,
        "manual_gain_all_to_lower_vs_greatest_need_ci95_low": 0.75,
        "manual_gain_all_to_lower_vs_greatest_need_ci95_high": 1.15,
    }
    for comparator in ("equal_split", "greatest_need"):
        result[f"gain_vs_{comparator}_simultaneous_one_sided_95_low"] = gain_low
        result[
            f"retained_gain_all_to_lower_vs_{comparator}_simultaneous_one_sided_95_low"
        ] = retained_low
        result[
            f"retained_gain_meet_lower_first_vs_{comparator}_simultaneous_one_sided_95_low"
        ] = retained_low
    for prefix in (
        "rr_acquisition",
        "rr_all_to_lower_match",
        "rr_meet_lower_first_match",
        "rr_more_to_lower",
    ):
        result[f"{prefix}_ci95_half_width"] = 0.02
    return result


class FixedAllocationPolicy:
    def __init__(self, allocation: float) -> None:
        self.allocation = allocation
        self.name = f"fixed_{allocation}"

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> str:
        return mdp.TERMINATE

    def choose_final_allocation(
        self,
        mdp: ContinuousAllocationMetaMDP,
        belief: BeliefState,
    ) -> float:
        return self.allocation


class ScarcityAllocationTests(unittest.TestCase):
    def test_exact_lower_rules_direction_equal_split_and_greatest_need(self) -> None:
        self.assertEqual(effort_to_goal(-2.0, 1.0), 0.0)
        self.assertEqual(lower_effort_identity(40.0, 80.0), 1)
        self.assertEqual(lower_effort_identity(80.0, 40.0), 2)
        self.assertEqual(lower_effort_identity(40.0, 40.0), 0)
        self.assertEqual(all_to_lower_allocation(40.0, 80.0, 1.0, 1.0), 1.0)
        self.assertEqual(greatest_effort_need_allocation(40.0, 80.0, 1.0, 1.0), 0.0)
        self.assertAlmostEqual(
            meet_lower_first_allocation(40.0, 80.0, 1.0, 1.0, 100.0),
            0.4,
        )
        self.assertEqual(meet_lower_first_allocation(40.0, 80.0, 1.0, 1.0, 30.0), 1.0)
        self.assertEqual(meet_lower_first_allocation(40.0, 40.0, 1.0, 1.0, 100.0), 0.5)

    def test_effort_identity_can_reverse_raw_need_identity(self) -> None:
        config = scarcity_config(
            learning_per_unit_of_tutoring=0.5,
            delta_learning_per_unit_tutoring=-0.5,
        )
        metrics = scarcity_allocation_metrics(config, TrueState(90.0, 100.0), 100.0, 0.0)
        self.assertEqual(metrics["lower_raw_need_identity"], 1)
        self.assertEqual(metrics["lower_effort_identity"], 2)
        self.assertTrue(metrics["effort_identity_differs_from_raw_need"])
        self.assertTrue(metrics["all_to_lower_match"])
        mirrored = scarcity_allocation_metrics(
            scarcity_config(
                learning_per_unit_of_tutoring=1.0,
                delta_learning_per_unit_tutoring=0.5,
            ),
            TrueState(100.0, 90.0),
            100.0,
            1.0,
        )
        self.assertEqual(mirrored["lower_effort_identity"], 1)
        self.assertTrue(mirrored["all_to_lower_match"])

    def test_policy_labels_use_posterior_means_but_outcomes_use_hidden_truth(self) -> None:
        config = scarcity_config()
        metrics = scarcity_allocation_metrics(
            config,
            TrueState(40.0, 80.0),
            100.0,
            0.0,
            classification_need_1=120.0,
            classification_need_2=60.0,
            classification_uses_hidden_true_state=False,
        )
        self.assertEqual(metrics["lower_effort_identity"], 2)
        self.assertTrue(metrics["all_to_lower_match"])
        self.assertFalse(metrics["classification_uses_hidden_true_state"])
        self.assertAlmostEqual(float(metrics["realized_outcome_1"]), -40.0)
        self.assertAlmostEqual(float(metrics["realized_outcome_2"]), 20.0)

    def test_feasibility_overlap_and_outcome_formulas(self) -> None:
        config = scarcity_config()
        metrics = scarcity_allocation_metrics(config, TrueState(40.0, 80.0), 100.0, 0.4)
        self.assertFalse(metrics["joint_goal_feasible"])
        self.assertTrue(metrics["at_least_lower_goal_meetable"])
        self.assertFalse(metrics["exactly_one_goal_individually_meetable"])
        self.assertTrue(metrics["both_individually_but_not_jointly_meetable"])
        self.assertFalse(metrics["lower_pattern_overlap"])
        self.assertTrue(metrics["meet_lower_first_match"])
        self.assertAlmostEqual(float(metrics["realized_outcome_1"]), 0.0)
        self.assertAlmostEqual(float(metrics["realized_outcome_2"]), -20.0)
        self.assertAlmostEqual(float(metrics["realized_outcome_gap"]), 20.0)
        exactly_one = scarcity_allocation_metrics(
            config,
            TrueState(40.0, 80.0),
            60.0,
            40.0 / 60.0,
        )
        self.assertTrue(exactly_one["exactly_one_goal_individually_meetable"])
        overlap = scarcity_allocation_metrics(config, TrueState(40.0, 80.0), 30.0, 1.0)
        self.assertTrue(overlap["lower_pattern_overlap"])
        self.assertTrue(overlap["all_to_lower_match"])
        self.assertTrue(overlap["meet_lower_first_match"])

    def test_continuous_gap_reports_mean_absolute_gap_and_rmse(self) -> None:
        summary = continuous_gap_summary([0.0, 0.3, 0.4], "gap")
        self.assertAlmostEqual(float(summary["gap_mean_absolute_gap"]), 0.7 / 3.0)
        self.assertAlmostEqual(float(summary["gap_rmse"]), math.sqrt(0.25 / 3.0))
        self.assertEqual(continuous_gap_summary([], "gap")["gap_mean_absolute_gap"], "")

    def test_immediate_and_balanced_six_policies_separate_search_from_choice(self) -> None:
        config = scarcity_config(total_time=20.0, max_meta_samples=40)
        streams = {"sample_1": [40.0] * 60, "sample_2": [80.0] * 60}
        for policy, expected_allocation in (
            (ImmediateAllToLowerPolicy(), 0.5),
            (ImmediateMeetLowerFirstPolicy(), 0.5),
        ):
            result = ContinuousAllocationMetaMDP(config, observation_streams=streams).run_episode(
                policy,
                true_state=TrueState(40.0, 80.0),
            )
            self.assertEqual(len(result.samples), 0)
            self.assertEqual(result.final_allocation_to_person1, expected_allocation)
        for policy in (
            ManualActiveSearchAllToLowerPolicy(),
            ManualActiveSearchMeetLowerFirstPolicy(),
        ):
            result = ContinuousAllocationMetaMDP(config, observation_streams=streams).run_episode(
                policy,
                true_state=TrueState(40.0, 80.0),
            )
            counts = [sum(item["action"] == action for item in result.samples) for action in (1.0, 2.0)]
            self.assertEqual(counts, [3, 3])
        policy = ManualActiveSearchAllToLowerPolicy()
        mdp = ContinuousAllocationMetaMDP(config)
        tied = BeliefState(100.0, 25.0, 100.0, 25.0)
        self.assertEqual(policy.choose_action(mdp, tied), mdp.SAMPLE_PERSON_1)
        tied.history.append({"action": 1.0, "observation": 100.0, "cost": 1.0})
        self.assertEqual(policy.choose_action(mdp, tied), mdp.SAMPLE_PERSON_2)


class ScarcityOracleAndSeedTests(unittest.TestCase):
    def test_frozen_deterministic_and_gaussian_grids(self) -> None:
        deterministic = build_deterministic_mechanism_cases()
        self.assertEqual(len(deterministic), 488)
        self.assertEqual(
            sum(row["analysis_role"] == "secondary_label_reversal_rate_robustness" for row in deterministic),
            2,
        )
        gaussian = build_gaussian_oracle_descriptors()
        self.assertEqual(len(gaussian), 135)
        self.assertEqual({row["total_time"] for row in gaussian}, {51.0, 101.0, 151.0, 191.0, 211.0})
        self.assertTrue(all(float(row["theoretical_either_nonpositive_probability"]) > 0.0 for row in gaussian))

    def test_unbounded_gaussian_probability_formula(self) -> None:
        individual, either = gaussian_nonpositive_probabilities(100.0, 10.0)
        self.assertGreater(individual, 0.0)
        self.assertAlmostEqual(either, 1.0 - (1.0 - individual) ** 2)

    def test_oracle_dominates_comparators_and_dense_grid_converges(self) -> None:
        config = scarcity_config()
        state = TrueState(60.0, 140.0)
        row = scarcity_oracle_comparison_row(
            environment_id="dense",
            anchor_id="anchor_dense",
            config=config,
            true_state=state,
            episode_index=0,
            pairing_group_id="dense",
            oracle_grid_size=4001,
            dense_grid_size=16001,
        )
        dense = scarcity_oracle_comparison_row(
            environment_id="dense",
            anchor_id="anchor_dense",
            config=config,
            true_state=state,
            episode_index=0,
            pairing_group_id="dense",
            oracle_grid_size=16001,
        )
        self.assertLessEqual(abs(float(row["oracle_utility"]) - float(dense["oracle_utility"])), 1e-6)
        self.assertLessEqual(float(row["oracle_dense_utility_absolute_difference"]), 1e-6)
        self.assertTrue(
            float(row["oracle_dense_allocation_absolute_difference"]) <= 0.0025
            or bool(row["oracle_dense_allocation_tie_within_1e-6"])
        )
        for comparator in ("equal_split", "greatest_need", "all_to_lower", "meet_lower_first"):
            self.assertGreaterEqual(float(row["oracle_utility"]) + 1e-9, float(row[f"{comparator}_utility"]))

    def test_piecewise_stationary_candidates_close_known_dense_gate_failure(self) -> None:
        config = scarcity_config(
            total_time=181.0,
            utility_exponent=0.75,
            lambda_shortfall=4.0,
        )
        forward = scarcity_oracle_comparison_row(
            environment_id="stationary_forward",
            anchor_id="stationary_anchor",
            config=config,
            true_state=TrueState(80.0, 120.0),
            episode_index=0,
            pairing_group_id="stationary_pair",
            oracle_grid_size=4001,
            dense_grid_size=16001,
        )
        reverse = scarcity_oracle_comparison_row(
            environment_id="stationary_reverse",
            anchor_id="stationary_anchor",
            config=config,
            true_state=TrueState(120.0, 80.0),
            episode_index=0,
            pairing_group_id="stationary_pair",
            oracle_grid_size=4001,
            dense_grid_size=16001,
        )
        self.assertLessEqual(float(forward["oracle_dense_utility_absolute_difference"]), 1e-12)
        self.assertLessEqual(float(reverse["oracle_dense_utility_absolute_difference"]), 1e-12)
        self.assertAlmostEqual(
            float(forward["oracle_allocation"]),
            1.0 - float(reverse["oracle_allocation"]),
            places=12,
        )

    def test_symmetric_boundary_tie_is_explicitly_utility_equivalent(self) -> None:
        config = scarcity_config(total_time=61.0)
        mdp = ContinuousAllocationMetaMDP(config)
        state = TrueState(100.0, 100.0)
        utility_left = deterministic_realized_utility(mdp, state, 0.0, 60.0)
        utility_right = deterministic_realized_utility(mdp, state, 1.0, 60.0)
        self.assertAlmostEqual(utility_left, utility_right, places=12)

    def test_pairing_seed_regression_across_controls_and_noise(self) -> None:
        pairing = scarcity_pairing_group_id("development", "anchor", "development_120")
        base = scarcity_config(sigma_sample=10.0, prior_sample_count_1=0, prior_sample_count_2=0)
        control = replace(
            base,
            sample_time_cost=0.1,
            prior_sample_count_1=20,
            prior_sample_count_2=20,
        )
        noisy = replace(base, sigma_sample=30.0)
        episodes = [
            build_scarcity_paired_episode(config, stage="development", pairing_group_id=pairing, episode_index=7)
            for config in (base, control, noisy)
        ]
        self.assertEqual(len({episode.true_state_fingerprint for episode in episodes}), 1)
        self.assertEqual(len({episode.standardized_innovation_hash_1 for episode in episodes}), 1)
        self.assertEqual(episodes[0].transformed_stream_hash_1, episodes[1].transformed_stream_hash_1)
        self.assertNotEqual(episodes[0].transformed_stream_hash_1, episodes[2].transformed_stream_hash_1)
        for recipient in ("sample_1", "sample_2"):
            need = getattr(episodes[0].episode.true_state, "need_1" if recipient == "sample_1" else "need_2")
            base_z = [(value - need) / 10.0 for value in episodes[0].episode.observation_streams[recipient]]
            noisy_z = [(value - need) / 30.0 for value in episodes[2].episode.observation_streams[recipient]]
            for left, right in zip(base_z, noisy_z):
                self.assertAlmostEqual(left, right, places=12)

    def test_policy_seeds_are_full_environment_disjoint(self) -> None:
        seeds = {
            scarcity_policy_seed(
                stage="development",
                environment_id=environment,
                episode_index=0,
                policy_id=policy,
            )
            for environment in ("env_a", "env_b")
            for policy in SCARCITY_POLICY_ORDER
        }
        self.assertEqual(len(seeds), 2 * len(SCARCITY_POLICY_ORDER))
        self.assertTrue(all(0 < seed < 2**63 for seed in seeds))

    def test_pairing_is_order_and_shard_invariant(self) -> None:
        config = scarcity_config()
        pairing = scarcity_pairing_group_id("development", "anchor", "development_120")
        forward = {
            index: build_scarcity_paired_episode(
                config,
                stage="development",
                pairing_group_id=pairing,
                episode_index=index,
            ).true_state_fingerprint
            for index in range(6)
        }
        reverse = {
            index: build_scarcity_paired_episode(
                config,
                stage="development",
                pairing_group_id=pairing,
                episode_index=index,
            ).true_state_fingerprint
            for index in reversed(range(6))
        }
        shards = {}
        for indices in ((0, 2, 4), (1, 3, 5)):
            for index in indices:
                shards[index] = build_scarcity_paired_episode(
                    config,
                    stage="development",
                    pairing_group_id=pairing,
                    episode_index=index,
                ).true_state_fingerprint
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, shards)

    def test_stream_contract_rejects_prior_or_online_overflow(self) -> None:
        pairing = scarcity_pairing_group_id("development", "anchor", "development_120")
        with self.assertRaises(ScarcityError):
            build_scarcity_paired_episode(
                scarcity_config(prior_sample_count_1=21),
                stage="development",
                pairing_group_id=pairing,
                episode_index=0,
            )
        with self.assertRaises(ScarcityError):
            build_scarcity_paired_episode(
                scarcity_config(max_meta_samples=41),
                stage="development",
                pairing_group_id=pairing,
                episode_index=0,
            )


class ScarcitySelectionAndInferenceTests(unittest.TestCase):
    def test_anchor_selection_uses_exact_then_direction_by_band(self) -> None:
        summaries = [
            oracle_summary_row("severe_all", capacity_ratio=0.5, exact_all=True),
            oracle_summary_row("severe_direction", capacity_ratio=0.5, direction=True),
            oracle_summary_row("near_direction_a", capacity_ratio=0.75, direction=True),
            oracle_summary_row("near_direction_b", capacity_ratio=0.95, direction=True),
            oracle_summary_row("feasible", capacity_ratio=1.05, exact_all=True),
        ]
        anchors = select_gaussian_oracle_anchors(summaries)
        self.assertEqual(len(anchors), 2)
        self.assertEqual(
            {(row["anchor_band"], row["anchor_support_kind"]) for row in anchors},
            {("severe", "exact"), ("near_feasible", "direction_only")},
        )
        self.assertNotIn("feasible", {row["environment_id"] for row in anchors})
        direction = next(row for row in anchors if row["anchor_support_kind"] == "direction_only")
        self.assertEqual(direction["environment_id"], "near_direction_a")
        self.assertEqual(direction["medoid_tie_break"], "lexicographically_smallest_environment_id")
        gate = object_level_stop_decision(summaries, anchors)
        self.assertFalse(gate["stop_metalevel"])
        self.assertEqual(gate["object_level_classification"], "supported_exact")

    def test_object_no_anchor_rejection_and_imprecision_both_stop(self) -> None:
        rejected = object_level_stop_decision(
            [oracle_summary_row("a", capacity_ratio=0.5, upper=0.79)],
            [],
        )
        self.assertTrue(rejected["stop_metalevel"])
        self.assertEqual(rejected["object_level_classification"], "not_supported_in_frozen_scope")
        imprecise = object_level_stop_decision(
            [
                oracle_summary_row("a", capacity_ratio=0.5, upper=0.79),
                oracle_summary_row("b", capacity_ratio=0.75, upper=0.80),
            ],
            [],
        )
        self.assertTrue(imprecise["stop_metalevel"])
        self.assertEqual(imprecise["object_level_classification"], "inconclusive_precision")

    def test_development_grid_is_exact_27_way_cross_per_anchor(self) -> None:
        oracle_descriptors = build_gaussian_oracle_descriptors()
        source = oracle_descriptors[0]
        selected = {
            "source_anchor_id": source["anchor_id"],
            "anchor_id": "selected_anchor",
            "anchor_rule": "all_to_lower",
            "anchor_support_kind": "exact",
            "anchor_band": "severe",
        }
        descriptors = build_development_descriptors([selected], oracle_descriptors)
        self.assertEqual(len(descriptors), 27)
        self.assertEqual(len({row["pairing_group_id"] for row in descriptors}), 1)
        self.assertEqual(len({row["environment_id"] for row in descriptors}), 27)

    def test_target_selection_freezes_exactly_one_per_class_and_nonrescuing_contrasts(self) -> None:
        rows = [
            development_row("no_exact", "no_search", exact=True, direction=True, sigma_sample=10.0, cost=0.1, prior=5),
            development_row("no_control", "no_search", exact=False, direction=False, sigma_sample=10.0, cost=1.0, prior=5),
            development_row("active_direction", "active_search", exact=False, direction=True, sigma_sample=2.0, cost=0.1, prior=5),
            development_row("active_control", "active_search", exact=False, direction=False, sigma_sample=2.0, cost=1.0, prior=5),
        ]
        selection = select_confirmation_targets(rows)
        self.assertEqual(selection["target_count"], 2)
        self.assertEqual({row["acquisition_class"] for row in selection["targets"]}, {"no_search", "active_search"})
        self.assertTrue(all(not row["can_establish_or_rescue_support"] for row in selection["contrasts"]))
        statuses = {row["acquisition_class"]: row["selection_status"] for row in selection["targets"]}
        self.assertEqual(statuses, {"no_search": "exact_candidate", "active_search": "direction_candidate"})

    def test_diagnostic_fallback_uses_frozen_lexicographic_rank(self) -> None:
        rows = [
            development_row("no_a", "no_search", exact=False, direction=False, sigma_sample=2.0, cost=0.1, prior=0, g_min=0.5),
            development_row("no_b", "no_search", exact=False, direction=False, sigma_sample=10.0, cost=0.1, prior=0, g_min=0.7),
            development_row("active_a", "active_search", exact=False, direction=False, sigma_sample=2.0, cost=0.1, prior=0, g_min=0.4),
            development_row("active_b", "active_search", exact=False, direction=False, sigma_sample=10.0, cost=0.1, prior=0, g_min=0.6),
        ]
        selection = select_confirmation_targets(rows)
        chosen = {row["acquisition_class"]: row["environment_id"] for row in selection["targets"]}
        self.assertEqual(chosen, {"no_search": "no_b", "active_search": "active_b"})
        self.assertTrue(all(row["selection_status"] == "diagnostic_only" for row in selection["targets"]))

    def test_confirmation_descriptors_preserve_pairing_and_class_labels(self) -> None:
        summaries = [
            development_row("no_exact", "no_search", exact=True, direction=True, sigma_sample=10.0, cost=0.1, prior=5),
            development_row("no_control", "no_search", exact=False, direction=False, sigma_sample=10.0, cost=1.0, prior=5),
            development_row("active_direction", "active_search", exact=False, direction=True, sigma_sample=2.0, cost=0.1, prior=5),
            development_row("active_control", "active_search", exact=False, direction=False, sigma_sample=2.0, cost=1.0, prior=5),
        ]
        configs = {
            "no_exact": scarcity_config(),
            "no_control": scarcity_config(sample_time_cost=1.01),
            "active_direction": scarcity_config(sigma_sample=2.0),
            "active_control": scarcity_config(sigma_sample=2.0, sample_time_cost=1.01),
        }
        development_descriptors = [
            {**row, "config": configs[str(row["environment_id"])]}
            for row in summaries
        ]
        selection = select_confirmation_targets(summaries)
        descriptors = build_confirmation_descriptors(selection, development_descriptors)
        self.assertEqual({row["acquisition_class"] for row in descriptors if row["confirmation_role"] == "target"}, {"no_search", "active_search"})
        target_pairing = {
            row["acquisition_class"]: row["pairing_group_id"]
            for row in descriptors
            if row["confirmation_role"] == "target"
        }
        for row in descriptors:
            self.assertEqual(row["pairing_group_id"], target_pairing[row["acquisition_class"]])

    def test_paired_contrast_holm_and_retained_gain_arithmetic(self) -> None:
        contrast = paired_contrast_summary([1.0, 2.0, 3.0], "contrast")
        self.assertEqual(contrast["contrast_mean"], 2.0)
        self.assertGreater(float(contrast["contrast_standard_error"]), 0.0)
        adjusted = holm_adjust_p_values({"a": 0.01, "b": 0.03, "c": 0.2})
        self.assertEqual(adjusted, {"a": 0.03, "b": 0.06, "c": 0.2})
        summaries = [{"environment_id": "e", "acquisition_class": "no_search", "gain_vs_equal_split_two_sided_p_value": 0.01}]
        attached = attach_holm_adjustment(summaries)
        self.assertEqual(attached[0]["gain_vs_equal_split_holm_adjusted_two_sided_p_value"], 0.01)
        self.assertAlmostEqual(9.0 - 0.8 * 10.0 - 0.2 * 4.0, 0.2)

    def test_classification_precedence_exact_partial_rejection_inconclusive(self) -> None:
        exact_target = {"acquisition_class": "no_search", "selection_status": "exact_candidate", "target_rule": "all_to_lower"}
        exact = classify_metalevel_target(confirmation_summary(), exact_target)
        self.assertEqual(exact["classification"], "supported_exact")
        partial = classify_metalevel_target(
            confirmation_summary(exact_low=0.70),
            exact_target,
        )
        self.assertEqual(partial["classification"], "partial_direction_only")
        rejected = classify_metalevel_target(
            confirmation_summary(direction_low=0.70, direction_high=0.79),
            exact_target,
        )
        self.assertEqual(rejected["classification"], "not_supported_in_frozen_scope")
        inconclusive = classify_metalevel_target(
            confirmation_summary(direction_low=0.70, direction_high=0.85),
            exact_target,
        )
        self.assertEqual(inconclusive["classification"], "inconclusive_precision")
        diagnostic_target = {"acquisition_class": "no_search", "selection_status": "diagnostic_only", "target_rule": "all_to_lower"}
        diagnostic = classify_metalevel_target(confirmation_summary(), diagnostic_target)
        self.assertEqual(diagnostic["classification"], "inconclusive_precision")


class ScarcityExecutionInvarianceTests(unittest.TestCase):
    def descriptor(self) -> dict[str, object]:
        return {
            "environment_id": "development_anchor_sigma_sample=10_cost=0.1_prior=0",
            "anchor_id": "anchor",
            "source_anchor_id": "source_anchor",
            "anchor_rule": "all_to_lower",
            "anchor_support_kind": "exact",
            "anchor_band": "severe",
            "pairing_group_id": scarcity_pairing_group_id("development", "anchor", "development_120"),
            "episode_set_id": "development_120",
            "sigma_sample": 10.0,
            "sample_time_cost_percent": 0.1,
            "prior_sample_count": 0,
            "mu_need": 100.0,
            "sigma_need": 10.0,
            "capacity_ratio": 0.5,
            "total_time": 101.0,
            "utility_exponent": 0.5,
            "lambda_shortfall": 2.0,
            "config": scarcity_config(allocation_grid_size=21),
        }

    @staticmethod
    def policy_objects() -> dict[str, object]:
        return {
            "frozen_rr": FixedAllocationPolicy(0.4),
            "immediate_all_to_lower": FixedAllocationPolicy(1.0),
            "immediate_meet_lower_first": FixedAllocationPolicy(0.4),
            "manual_active_search_all_to_lower": FixedAllocationPolicy(1.0),
            "manual_active_search_meet_lower_first": FixedAllocationPolicy(0.4),
            "equal_split": FixedAllocationPolicy(0.5),
            "greatest_need": FixedAllocationPolicy(0.0),
        }

    def test_policy_order_does_not_change_scientific_rows(self) -> None:
        descriptor = self.descriptor()
        with patch.object(scarcity_module, "_metalevel_policy_objects", return_value=self.policy_objects()):
            forward = evaluate_metalevel_episode(descriptor, stage="development", episode_index=0, oracle_grid_size=101)
            with patch.object(scarcity_module, "SCARCITY_POLICY_ORDER", tuple(reversed(SCARCITY_POLICY_ORDER))):
                reverse = evaluate_metalevel_episode(descriptor, stage="development", episode_index=0, oracle_grid_size=101)
        forward_by_policy = {row["policy_id"]: row for row in forward}
        reverse_by_policy = {row["policy_id"]: row for row in reverse}
        self.assertEqual(canonical_hash(forward_by_policy), canonical_hash(reverse_by_policy))
        self.assertEqual(
            forward_by_policy["frozen_rr"]["final_choice_classification_basis"],
            "terminal_posterior_mean",
        )
        self.assertEqual(
            forward_by_policy["initial_full_information_oracle"][
                "final_choice_classification_basis"
            ],
            "hidden_true_need_full_information",
        )

    def test_shard_partition_does_not_change_rows(self) -> None:
        descriptor = self.descriptor()
        with patch.object(scarcity_module, "_metalevel_policy_objects", return_value=self.policy_objects()):
            serial = evaluate_metalevel_descriptor(descriptor, stage="development", n_episodes=4, oracle_grid_size=101)
            sharded = evaluate_metalevel_descriptor(descriptor, stage="development", n_episodes=2, episode_start=0, oracle_grid_size=101)
            sharded.extend(evaluate_metalevel_descriptor(descriptor, stage="development", n_episodes=2, episode_start=2, oracle_grid_size=101))
        serial_sorted = sorted(serial, key=lambda row: (row["episode_index"], row["policy_id"]))
        shard_sorted = sorted(sharded, key=lambda row: (row["episode_index"], row["policy_id"]))
        self.assertEqual(canonical_hash(serial_sorted), canonical_hash(shard_sorted))


if __name__ == "__main__":
    unittest.main()
