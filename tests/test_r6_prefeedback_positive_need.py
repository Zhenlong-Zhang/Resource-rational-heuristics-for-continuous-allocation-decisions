from __future__ import annotations

from dataclasses import replace
import unittest

from src.experiments.r6_prefeedback_positive_need import (
    POLICY_ORACLE,
    POLICY_RR,
    POLICY_SPLIT,
    PositiveNeedEnvironment,
    build_development_environments,
    build_finite_support_episodes,
    build_latent_support_table,
    build_numerical_validation_cases,
    evaluate_fixed_budgets,
    evaluate_serious_environment,
    load_positive_need_spec,
    select_target_control_pair,
    solver_diagnosis_trigger,
    summarize_serious,
    validate_serious_common_randomness,
)


class R6PreFeedbackPositiveNeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_positive_need_spec()
        self.environments = build_development_environments(self.spec)

    def test_frozen_grid_has_72_positive_matched_environments(self) -> None:
        self.assertEqual(len(self.environments), 72)
        self.assertEqual(len({environment.name for environment in self.environments}), 72)
        for environment in self.environments:
            self.assertEqual(len(environment.prior.states), 18)
            self.assertAlmostEqual(sum(environment.prior.weights), 1.0)
            self.assertTrue(
                all(atom.need_1 > 0.0 and atom.need_2 > 0.0 for atom in environment.prior.states)
            )

    def test_numerical_suite_freezes_90_cases_and_36_dense_references(self) -> None:
        cases = build_numerical_validation_cases(self.spec)
        self.assertEqual(len(cases), 90)
        self.assertEqual([case["case_id"] for case in cases], list(range(90)))
        by_name = {environment.name: environment for environment in self.environments}
        dense_kinds = {
            "uniform_prior",
            "person1_predictive_mean",
            "person1_minimum_support",
            "person1_maximum_support",
        }
        dense = [
            case
            for case in cases
            if by_name[case["environment"]].sample_time_cost == 0.02
            and case["belief_kind"] in dense_kinds
        ]
        self.assertEqual(len(dense), 36)
        support = build_latent_support_table(self.spec)
        self.assertEqual(len(support), 54)
        self.assertEqual({row["gap_class"] for row in support}, {"low", "medium", "high"})

    def test_cost_conditions_share_latent_states_and_residuals(self) -> None:
        target = next(
            item
            for item in self.environments
            if item.gap_class == "high" and item.sigma_sample == 2.0 and item.sample_time_cost == 0.02
        )
        control = next(
            item
            for item in self.environments
            if item.gap_class == "high" and item.sigma_sample == 2.0 and item.sample_time_cost == 8.0
        )
        target_episodes = build_finite_support_episodes(
            target,
            n_episodes=18,
            stage="test",
            seed_namespace=900,
            observations_per_person=8,
            balanced_atoms=True,
        )
        control_episodes = build_finite_support_episodes(
            control,
            n_episodes=18,
            stage="test",
            seed_namespace=900,
            observations_per_person=8,
            balanced_atoms=True,
        )
        for target_episode, control_episode in zip(target_episodes, control_episodes):
            self.assertEqual(target_episode.atom, control_episode.atom)
            self.assertEqual(target_episode.true_state, control_episode.true_state)
            self.assertEqual(target_episode.residual_hash_1, control_episode.residual_hash_1)
            self.assertEqual(target_episode.residual_hash_2, control_episode.residual_hash_2)
            self.assertEqual(target_episode.fingerprint, control_episode.fingerprint)

    def test_selection_uses_lowest_target_and_lowest_higher_control(self) -> None:
        rows = []
        for gap, advantage in (("low", 1.0), ("high", 2.0)):
            for cost in (0.02, 0.1, 1.0, 2.0):
                rows.append(
                    {
                        "gap_class": gap,
                        "sigma_sample": 2.0,
                        "sample_time_cost": cost,
                        "environment": f"{gap}-{cost}",
                        "environment_hash": f"hash-{gap}-{cost}",
                        "target_gate_pass": 1.0 if cost in (0.02, 0.1) else 0.0,
                        "control_manual_gate_pass": 1.0 if cost >= 1.0 else 0.0,
                        "manual_6_minus_split_ci95_low": advantage,
                        "manual_6_minus_manual_4_ci95_low": 0.5,
                    }
                )
        selected = select_target_control_pair(rows)
        self.assertEqual(selected["selection_status"], "selected_without_rr_behavior")
        self.assertEqual(selected["gap_class"], "high")
        self.assertEqual(selected["target_sample_time_cost"], 0.02)
        self.assertEqual(selected["control_sample_time_cost"], 1.0)
        csv_round_trip = [
            {key: str(value) for key, value in row.items()} for row in rows
        ]
        self.assertEqual(select_target_control_pair(csv_round_trip), selected)

    def test_solver_trigger_requires_every_prespecified_gate(self) -> None:
        values = {
            "recovery_ci95_high": -0.1,
            "time_matched_regret_ci95_low": 0.1,
            "mean_time_matched_regret": 2.0,
            "mean_time_matched_oracle_utility": 100.0,
            "mean_rr_sample_count": 1.0,
            "fixed_budget_evidence_favors_6_over_4": True,
            "development_gate_pass": True,
        }
        self.assertTrue(solver_diagnosis_trigger(**values))
        boundaries = {
            "recovery_ci95_high": 0.0,
            "time_matched_regret_ci95_low": 0.0,
            "mean_time_matched_regret": 1.0,
            "mean_rr_sample_count": 1.01,
            "fixed_budget_evidence_favors_6_over_4": False,
            "development_gate_pass": False,
        }
        for field, boundary in boundaries.items():
            changed = dict(values)
            changed[field] = boundary
            self.assertFalse(solver_diagnosis_trigger(**changed), field)

    def test_fixed_budget_and_serious_rows_share_episode_scaffolds(self) -> None:
        original = next(
            item
            for item in self.environments
            if item.gap_class == "medium"
            and item.sigma_sample == 2.0
            and item.sample_time_cost == 0.1
        )
        environment = PositiveNeedEnvironment(
            name="small-test",
            gap_class=original.gap_class,
            sigma_sample=original.sigma_sample,
            sample_time_cost=original.sample_time_cost,
            config=replace(
                original.config,
                allocation_grid_size=21,
                max_meta_samples=4,
            ),
            prior=original.prior,
        )
        episodes = build_finite_support_episodes(
            environment,
            n_episodes=2,
            stage="test",
            seed_namespace=901,
            observations_per_person=8,
            balanced_atoms=True,
        )
        fixed = evaluate_fixed_budgets(
            environment,
            episodes,
            samples_per_person=(0, 1),
            allocation_tolerance=0.05,
            oracle_grid_size=101,
        )
        self.assertEqual(len(fixed), 6)
        self.assertEqual(sum(row["policy"] == POLICY_ORACLE for row in fixed), 2)

        serious = evaluate_serious_environment(
            environment,
            episodes,
            quadrature_order=3,
            manual_samples_per_person=1,
            allocation_tolerance=0.05,
            oracle_grid_size=101,
        )
        self.assertEqual(len(serious), 8)
        validate_serious_common_randomness(serious)
        self.assertEqual(sum(row["policy"] == POLICY_RR for row in serious), 2)
        self.assertEqual(sum(row["policy"] == POLICY_SPLIT for row in serious), 2)
        for episode_index in range(2):
            episode_rows = [row for row in serious if row["episode_index"] == episode_index]
            self.assertEqual(len({row["episode_fingerprint"] for row in episode_rows}), 1)

        target_rows = [{**row, "environment": "target"} for row in serious]
        control_rows = [{**row, "environment": "control"} for row in serious]
        _, _, smoke = summarize_serious(
            target_rows + control_rows,
            target_environment="target",
            control_environment="control",
            selection={
                "selection_status": "selected_without_rr_behavior",
                "target_gate_pass": 1.0,
                "fixed_budget_evidence_favors_6_over_4": 1.0,
            },
            scientific_confirmation=False,
        )
        self.assertEqual(smoke["readiness_classification"], "invalid_evidence")
        self.assertEqual(smoke["evidence_status"], "smoke_only_not_scientific_evidence")


if __name__ == "__main__":
    unittest.main()
