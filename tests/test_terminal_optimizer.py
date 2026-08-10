from __future__ import annotations

from dataclasses import replace
import math
import unittest

from src.experiments.r6_prefeedback_positive_need import (
    _numerical_belief,
    build_development_environments,
    build_numerical_validation_cases,
    load_positive_need_spec,
)
from src.mdp.finite_support import (
    FiniteSupportAtom,
    FiniteSupportBeliefState,
    FiniteSupportMetaMDP,
    FiniteSupportPrior,
)
from src.mdp.meta_mdp import EnvironmentConfig
from src.solvers.terminal import (
    STRUCTURAL_SYMMETRY_INVARIANT_FIELDS,
    StructuralSymmetry,
    build_structural_symmetry_hashes,
    classify_terminal_tie,
    diagnose_terminal_performance,
    optimal_terminal_results_for_weight_rows,
    optimize_terminal_allocation,
    prove_recipient_swap_symmetry,
    rational_power_bounds_7_20,
    structural_mirror_tie_supported,
    terminal_breakpoints,
    terminal_objective_upper_bound,
    validate_structural_symmetry_proof,
)


def terminal_config(**changes) -> EnvironmentConfig:
    values = {
        "mu_need": 20.0,
        "sigma_need": 1.0,
        "sigma_sample": 2.0,
        "total_time": 40.0,
        "lambda_shortfall": 2.0,
        "utility_exponent": 0.35,
        "learning_per_unit_of_tutoring": 1.0,
        "terminate_cost": 1.0,
        "sample_time_cost": 0.25,
        "allocation_grid_size": 11,
        "max_meta_samples": 2,
        "random_seed": 19,
    }
    values.update(changes)
    return EnvironmentConfig(**values)


def one_atom_mdp(atom: FiniteSupportAtom, **config_changes) -> FiniteSupportMetaMDP:
    return FiniteSupportMetaMDP(
        terminal_config(**config_changes),
        FiniteSupportPrior((atom,), (1.0,)),
    )


def swapped_two_atom_prior(
    *,
    total_need: float = 80.0,
    gap_fraction: float = 0.5,
    weights=(0.5, 0.5),
) -> FiniteSupportPrior:
    return FiniteSupportPrior(
        (
            FiniteSupportAtom(total_need, gap_fraction, -1),
            FiniteSupportAtom(total_need, gap_fraction, 1),
        ),
        tuple(weights),
    )


class TerminalOptimizerTests(unittest.TestCase):
    def assert_bound_dominates_dense_values(
        self,
        mdp: FiniteSupportMetaMDP,
        belief: FiniteSupportBeliefState,
        lower: float,
        upper: float,
        count: int = 101,
    ) -> None:
        bound = terminal_objective_upper_bound(mdp, belief, lower, upper)
        for index in range(count):
            allocation = lower + (upper - lower) * index / (count - 1)
            self.assertGreaterEqual(
                bound,
                mdp.expected_terminal_utility(belief, allocation),
            )

    def test_unique_boundary_maximum_is_searched(self) -> None:
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief = mdp.initial_belief()
        result = mdp.solve_terminal_allocation_result(belief)

        self.assertEqual(result.allocation, 1.0)
        self.assertEqual(result.tie_status, "unique")
        self.assertLessEqual(result.regret_upper_bound, 2.5e-5 + 1e-12)
        self.assertEqual(result.value, mdp.expected_terminal_utility(belief, 1.0))

    def test_support_derived_kink_can_be_the_unique_maximum(self) -> None:
        prior = FiniteSupportPrior(
            (
                FiniteSupportAtom(30.0, 0.4, -1),
                FiniteSupportAtom(60.0, 0.0, -1),
            ),
            (0.5071635969746414, 0.4928364030253586),
        )
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief = mdp.initial_belief()
        result = mdp.solve_terminal_allocation_result(belief)
        kink = 9.0 / 39.0

        self.assertAlmostEqual(result.allocation, kink, places=15)
        self.assertEqual(result.tie_status, "unique")
        self.assertEqual(result.value, mdp.expected_terminal_utility(belief, result.allocation))
        self.assertGreater(result.value, mdp.expected_terminal_utility(belief, kink - 1e-6))
        self.assertGreater(result.value, mdp.expected_terminal_utility(belief, kink + 1e-6))

    def test_symmetric_center_maximum_is_unique(self) -> None:
        mdp = one_atom_mdp(FiniteSupportAtom(20.0, 0.0, 1))
        belief = mdp.initial_belief()
        result = mdp.solve_terminal_allocation_result(belief)

        self.assertTrue(result.structural_symmetry.valid)
        self.assertTrue(result.structural_symmetry.proof_hash)
        self.assertAlmostEqual(result.allocation, 0.5, places=12)
        self.assertEqual(result.tie_status, "unique")
        self.assertEqual(result.value, mdp.expected_terminal_utility(belief, 0.5))

    def test_structural_off_center_mirrored_maxima_use_lower_canonical_side(self) -> None:
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.0, 1))
        belief = mdp.initial_belief()
        result = mdp.solve_terminal_allocation_result(belief)

        self.assertTrue(result.structural_symmetry.valid)
        self.assertEqual(result.tie_status, "structural_symmetry_tie")
        self.assertEqual(result.allocation, 0.0)
        self.assertEqual(result.value, mdp.expected_terminal_utility(belief, 0.0))
        self.assertEqual(
            mdp.expected_terminal_utility(belief, 0.0),
            mdp.expected_terminal_utility(belief, 1.0),
        )

    def test_single_mirror_orbit_and_interval_pair_is_structural(self) -> None:
        symmetry = StructuralSymmetry(True, (0,), "test")
        self.assertTrue(
            structural_mirror_tie_supported(
                symmetry,
                (0.2, 0.8),
                ((0.19, 0.21), (0.79, 0.81)),
                5e-4,
            )
        )
        self.assertEqual(
            classify_terminal_tie(
                symmetry,
                (0.2, 0.8),
                ((0.19, 0.21), (0.79, 0.81)),
                5e-4,
            ),
            "structural_symmetry_tie",
        )

    def test_cross_orbit_ties_are_never_structural(self) -> None:
        symmetry = StructuralSymmetry(True, (0,), "test")
        cases = {
            "center_plus_pair": (
                (0.2, 0.5, 0.8),
                ((0.19, 0.21), (0.49, 0.51), (0.79, 0.81)),
            ),
            "multiple_pairs": (
                (0.1, 0.2, 0.8, 0.9),
                ((0.09, 0.11), (0.19, 0.21), (0.79, 0.81), (0.89, 0.91)),
            ),
            "unmatched_interval": (
                (0.2, 0.8),
                ((0.19, 0.21), (0.70, 0.72)),
            ),
            "cross_orbit_pair": (
                (0.2, 0.7),
                ((0.19, 0.21), (0.69, 0.71)),
            ),
        }
        for name, (allocations, intervals) in cases.items():
            with self.subTest(name=name):
                self.assertFalse(
                    structural_mirror_tie_supported(
                        symmetry,
                        allocations,
                        intervals,
                        5e-4,
                    )
                )
                self.assertEqual(
                    classify_terminal_tie(
                        symmetry,
                        allocations,
                        intervals,
                        5e-4,
                    ),
                    "ordinary_tie_provisional",
                )

    def test_forged_symmetry_metadata_cannot_certify_asymmetric_weights(self) -> None:
        prior = swapped_two_atom_prior()
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief = FiniteSupportBeliefState(prior.states, (0.9, 0.1))
        setattr(belief, "posterior_expression_tracked", True)
        setattr(belief, "likelihood_observations_1", ("forged",))
        setattr(belief, "likelihood_observations_2", ("forged",))

        proof = prove_recipient_swap_symmetry(mdp, belief)
        result = mdp.solve_terminal_allocation_result(belief)

        self.assertFalse(proof.valid)
        self.assertEqual(proof.reason, "posterior_weights_not_exactly_swap_invariant")
        self.assertFalse(result.structural_symmetry.valid)
        self.assertEqual(
            result.value,
            mdp.expected_terminal_utility(belief, result.allocation),
        )

    def test_structural_hashes_bind_every_named_section_5_4_invariant(self) -> None:
        base_prior = FiniteSupportPrior.from_total_and_absolute_gaps(
            total_needs=(80.0, 100.0),
            absolute_gaps=(20.0,),
        )

        def proof_for(
            *,
            prior=base_prior,
            weights=None,
            **config_changes,
        ):
            mdp = FiniteSupportMetaMDP(terminal_config(**config_changes), prior)
            belief = (
                mdp.initial_belief()
                if weights is None
                else FiniteSupportBeliefState(prior.states, tuple(weights))
            )
            proof = prove_recipient_swap_symmetry(mdp, belief)
            self.assertTrue(proof.valid, proof.reason)
            self.assertTrue(validate_structural_symmetry_proof(mdp, belief, proof))
            return proof

        base = proof_for()
        base_fields = dict(base.invariant_field_hashes)
        self.assertEqual(
            tuple(base_fields),
            STRUCTURAL_SYMMETRY_INVARIANT_FIELDS,
        )

        support_prior = FiniteSupportPrior.from_total_and_absolute_gaps(
            total_needs=(82.0, 102.0),
            absolute_gaps=(20.0,),
        )
        reordered = (base_prior.states[0], base_prior.states[2], base_prior.states[1], base_prior.states[3])
        reordered_prior = FiniteSupportPrior(reordered, (1.0, 1.0, 1.0, 1.0))
        weighted_prior = FiniteSupportPrior(
            base_prior.states,
            (0.2, 0.2, 0.3, 0.3),
        )
        termination_collision = proof_for(total_time=41.0, terminate_cost=2.0)
        self.assertEqual(
            dict(termination_collision.invariant_field_hashes)["remaining_time"],
            base_fields["remaining_time"],
        )

        variants = {
            "support_atoms": proof_for(prior=support_prior),
            "swap_permutation": proof_for(prior=reordered_prior),
            "prior_weights": proof_for(prior=weighted_prior),
            "posterior_weights": proof_for(weights=(0.2, 0.2, 0.3, 0.3)),
            "learning_rate_1": proof_for(learning_per_unit_of_tutoring=1.2),
            "learning_rate_2": proof_for(learning_per_unit_of_tutoring=1.2),
            "observation_noise_sigma": proof_for(sigma_sample=3.0),
            "sampling_time_cost": proof_for(sample_time_cost=0.5),
            "termination_cost": termination_collision,
            "remaining_time": proof_for(total_time=41.0),
            "utility_lambda_shortfall": proof_for(lambda_shortfall=3.0),
            "utility_exponent": proof_for(utility_exponent=0.36),
        }
        for field_name in STRUCTURAL_SYMMETRY_INVARIANT_FIELDS:
            with self.subTest(field_name=field_name):
                variant = variants[field_name]
                self.assertNotEqual(
                    dict(variant.invariant_field_hashes)[field_name],
                    base_fields[field_name],
                )
                self.assertNotEqual(variant.invariant_hash, base.invariant_hash)
                self.assertNotEqual(variant.proof_hash, base.proof_hash)

    def test_tampered_named_field_is_rejected_even_after_outer_hash_recomputation(self) -> None:
        prior = swapped_two_atom_prior()
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief = mdp.initial_belief()
        proof = prove_recipient_swap_symmetry(mdp, belief)
        self.assertTrue(validate_structural_symmetry_proof(mdp, belief, proof))

        tampered_fields = tuple(
            (name, "f" * 64 if name == "termination_cost" else field_hash)
            for name, field_hash in proof.invariant_field_hashes
        )
        invariant_hash, proof_hash = build_structural_symmetry_hashes(
            proof.permutation,
            tampered_fields,
        )
        self_consistent_tamper = replace(
            proof,
            invariant_field_hashes=tampered_fields,
            invariant_hash=invariant_hash,
            proof_hash=proof_hash,
        )
        self.assertFalse(
            validate_structural_symmetry_proof(mdp, belief, self_consistent_tamper)
        )
        self.assertFalse(
            validate_structural_symmetry_proof(
                mdp,
                belief,
                replace(proof, proof_hash="0" * 64),
            )
        )
        self.assertEqual(prove_recipient_swap_symmetry(mdp, belief), proof)

    def test_result_value_is_always_the_public_stored_posterior_objective(self) -> None:
        prior = swapped_two_atom_prior(total_need=64.0, gap_fraction=0.25)
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        beliefs = (
            mdp.initial_belief(),
            FiniteSupportBeliefState(prior.states, (0.83, 0.17)),
            FiniteSupportBeliefState(prior.states, (1.0, 0.0)),
        )
        for belief in beliefs:
            with self.subTest(weights=belief.weights):
                result = mdp.solve_terminal_allocation_result(belief)
                self.assertEqual(
                    result.value,
                    mdp.expected_terminal_utility(belief, result.allocation),
                )
                self.assertGreaterEqual(result.global_upper_bound, result.value)

    def test_interval_upper_bounds_dominate_stored_objective(self) -> None:
        prior = FiniteSupportPrior.from_total_and_absolute_gaps(
            total_needs=(30.0, 60.0),
            absolute_gaps=(0.0, 12.0, 24.0),
        )
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief = FiniteSupportBeliefState(
            prior.states,
            (0.0, 1e-250, 0.19, 0.0, 0.07, 0.31, 0.0, 0.11, 0.0, 0.32, 0.0, 0.0),
        )
        points = terminal_breakpoints(mdp, belief)
        for lower, upper in zip(points[:-1], points[1:]):
            self.assert_bound_dominates_dense_values(mdp, belief, lower, upper, 31)
        self.assert_bound_dominates_dense_values(mdp, belief, 0.123, 0.876, 101)

    def test_rational_power_brackets_are_certified_by_exact_integer_ratios(self) -> None:
        values = (
            0.0,
            math.ulp(0.0),
            1e-300,
            0.125,
            math.nextafter(1.0, 0.0),
            1.0,
            math.nextafter(1.0, math.inf),
            2.0,
            1e8,
            1e300,
        )
        for value in values:
            with self.subTest(value=value):
                lower, upper = rational_power_bounds_7_20(value)
                value_numerator, value_denominator = value.as_integer_ratio()
                lower_numerator, lower_denominator = lower.as_integer_ratio()
                upper_numerator, upper_denominator = upper.as_integer_ratio()
                exact_right_lower = (
                    value_numerator**7 * lower_denominator**20
                )
                exact_left_lower = (
                    lower_numerator**20 * value_denominator**7
                )
                exact_right_upper = (
                    value_numerator**7 * upper_denominator**20
                )
                exact_left_upper = (
                    upper_numerator**20 * value_denominator**7
                )
                self.assertLessEqual(exact_left_lower, exact_right_lower)
                self.assertGreaterEqual(exact_left_upper, exact_right_upper)
                runtime_value = value**0.35
                self.assertLessEqual(lower, runtime_value)
                self.assertGreaterEqual(upper, runtime_value)

    def test_unsupported_utility_exponent_fails_closed(self) -> None:
        mdp = one_atom_mdp(
            FiniteSupportAtom(20.0, 0.0, 1),
            utility_exponent=0.5,
        )
        with self.assertRaisesRegex(ValueError, "only utility exponent 0.35"):
            mdp.solve_terminal_allocation_result(mdp.initial_belief())

    def test_near_zero_cancellation_is_searched_without_derivative_pruning(self) -> None:
        prior = swapped_two_atom_prior(total_need=28.0, gap_fraction=0.2)
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief = FiniteSupportBeliefState(
            prior.states,
            (0.5 + 2e-14, 0.5 - 2e-14),
        )
        result = mdp.solve_terminal_allocation_result(belief)
        dense = [
            (index / 4000.0, mdp.expected_terminal_utility(belief, index / 4000.0))
            for index in range(4001)
        ]
        dense_best = max(value for _, value in dense)

        self.assertGreaterEqual(result.value + 2.5e-5, dense_best)
        self.assertLess(abs(result.allocation - 0.5), 0.01)
        self.assert_bound_dominates_dense_values(mdp, belief, 0.45, 0.55, 101)

    def test_tiny_and_zero_weights_at_kinks_remain_finite(self) -> None:
        prior = FiniteSupportPrior.from_total_and_absolute_gaps(
            total_needs=(30.0, 70.0),
            absolute_gaps=(0.0, 20.0),
        )
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        weights = [0.0] * len(prior.states)
        weights[0] = 1.0 - 1e-200
        weights[-1] = 1e-200
        belief = FiniteSupportBeliefState(prior.states, tuple(weights))
        result = mdp.solve_terminal_allocation_result(belief)

        self.assertTrue(math.isfinite(result.value))
        self.assertTrue(math.isfinite(result.global_upper_bound))
        self.assertEqual(result.value, mdp.expected_terminal_utility(belief, result.allocation))
        for lower, upper in zip(
            terminal_breakpoints(mdp, belief)[:-1],
            terminal_breakpoints(mdp, belief)[1:],
        ):
            self.assert_bound_dominates_dense_values(mdp, belief, lower, upper, 11)

    def test_large_dynamic_range_uses_conservative_bound(self) -> None:
        mdp = one_atom_mdp(
            FiniteSupportAtom(1e8, 0.2, 1),
            total_time=1e8,
            learning_per_unit_of_tutoring=1.5,
            utility_exponent=0.35,
        )
        belief = mdp.initial_belief()
        result = mdp.solve_terminal_allocation_result(belief)

        self.assertTrue(math.isfinite(result.value))
        self.assertEqual(result.value, mdp.expected_terminal_utility(belief, result.allocation))
        self.assert_bound_dominates_dense_values(mdp, belief, 0.0, 1.0, 201)

    def test_multi_atom_nonconcave_objective_with_multiple_local_maxima(self) -> None:
        prior = FiniteSupportPrior.from_total_and_absolute_gaps(
            total_needs=(70.0, 90.0),
            absolute_gaps=(10.0, 40.0),
        )
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief = mdp.initial_belief()
        result = mdp.solve_terminal_allocation_result(belief)
        values = [
            mdp.expected_terminal_utility(belief, index / 2000.0)
            for index in range(2001)
        ]
        local_maxima = [
            index
            for index in range(1, 2000)
            if values[index] >= values[index - 1] and values[index] >= values[index + 1]
        ]
        if values[0] >= values[1]:
            local_maxima.insert(0, 0)
        if values[-1] >= values[-2]:
            local_maxima.append(2000)

        self.assertGreaterEqual(len(local_maxima), 2)
        self.assertGreaterEqual(result.value + 2.5e-5, max(values))
        self.assertEqual(result.value, mdp.expected_terminal_utility(belief, result.allocation))

    def test_every_support_kink_is_a_breakpoint(self) -> None:
        prior = FiniteSupportPrior.from_total_and_absolute_gaps(
            total_needs=(24.0, 32.0),
            absolute_gaps=(0.0, 8.0),
        )
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief = mdp.initial_belief()
        remaining = mdp.remaining_time_after_termination(belief)
        rate_1, rate_2 = mdp.learning_rates()
        expected = {0.0, 1.0}
        for state in prior.states:
            expected.add(state.need_1 / (rate_1 * remaining))
            expected.add(1.0 - state.need_2 / (rate_2 * remaining))
        expected = {value for value in expected if 0.0 <= value <= 1.0}

        self.assertEqual(terminal_breakpoints(mdp, belief), tuple(sorted(expected)))

    def test_symmetry_proof_rejects_support_prior_and_learning_asymmetry(self) -> None:
        support_mdp = one_atom_mdp(FiniteSupportAtom(50.0, 0.3, -1))
        self.assertEqual(
            prove_recipient_swap_symmetry(support_mdp, support_mdp.initial_belief()).reason,
            "support_not_closed_under_swap",
        )

        prior = swapped_two_atom_prior(weights=(0.6, 0.4))
        prior_mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        symmetric_posterior = FiniteSupportBeliefState(prior.states, (0.5, 0.5))
        self.assertEqual(
            prove_recipient_swap_symmetry(prior_mdp, symmetric_posterior).reason,
            "prior_weights_not_exactly_swap_invariant",
        )

        rate_mdp = FiniteSupportMetaMDP(
            terminal_config(delta_learning_per_unit_tutoring=0.2),
            swapped_two_atom_prior(),
        )
        self.assertEqual(
            prove_recipient_swap_symmetry(rate_mdp, rate_mdp.initial_belief()).reason,
            "learning_rates_not_symmetric",
        )

    def test_ordinary_near_tie_is_only_provisional_and_deterministic(self) -> None:
        prior = swapped_two_atom_prior(weights=(0.5 + 1e-15, 0.5 - 1e-15))
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief = mdp.initial_belief()
        results = [mdp.solve_terminal_allocation_result(belief) for _ in range(3)]

        self.assertFalse(results[0].structural_symmetry.valid)
        self.assertIn(results[0].tie_status, {"unique", "ordinary_tie_provisional"})
        self.assertNotEqual(results[0].tie_status, "structural_symmetry_tie")
        self.assertTrue(all(result.allocation == results[0].allocation for result in results))
        self.assertTrue(all(result.value == results[0].value for result in results))

    def test_frozen_cases_72_and_77_reject_unstored_symbolic_symmetry(self) -> None:
        spec = load_positive_need_spec()
        cases = build_numerical_validation_cases(spec)
        environments = {
            environment.name: environment
            for environment in build_development_environments(spec)
        }
        for case_id in (72, 77):
            case = cases[case_id]
            environment = environments[str(case["environment"])]
            belief = _numerical_belief(environment, str(case["belief_kind"]))
            mdp = FiniteSupportMetaMDP(environment.config, environment.prior)
            first = mdp.solve_terminal_allocation_result(belief)
            second = mdp.solve_terminal_allocation_result(belief)

            self.assertFalse(first.structural_symmetry.valid)
            self.assertEqual(
                first.structural_symmetry.reason,
                "posterior_weights_not_exactly_swap_invariant",
            )
            self.assertNotEqual(first.tie_status, "structural_symmetry_tie")
            self.assertEqual(first.allocation, second.allocation)
            self.assertEqual(first.value, second.value)
            self.assertEqual(
                first.value,
                mdp.expected_terminal_utility(belief, first.allocation),
            )

    def test_evaluation_cap_exhaustion_fails_closed(self) -> None:
        mdp = one_atom_mdp(FiniteSupportAtom(20.0, 0.0, 1))
        with self.assertRaisesRegex(RuntimeError, "evaluation cap"):
            optimize_terminal_allocation(mdp, mdp.initial_belief(), max_evaluations=3)

    def test_scalar_and_batch_results_have_identical_value_and_allocation_semantics(self) -> None:
        prior = FiniteSupportPrior.from_total_and_absolute_gaps(
            total_needs=(32.0, 36.0),
            absolute_gaps=(8.0, 16.0),
        )
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief = mdp.initial_belief()
        observations = (belief.mean_1 - 2.0, belief.mean_1, belief.mean_1 + 2.0)
        rows = mdp.posterior_weights_for_observations(
            belief,
            mdp.SAMPLE_PERSON_1,
            observations,
        )
        deliberation_time = belief.deliberation_time + mdp.config.sample_time_cost
        batch_results = optimal_terminal_results_for_weight_rows(
            mdp,
            belief,
            rows,
            deliberation_time,
        )
        batch_values = mdp.optimal_terminal_values_for_weights(
            belief,
            rows,
            deliberation_time=deliberation_time,
        )

        for row, batch_result, batch_value in zip(rows, batch_results, batch_values):
            posterior = FiniteSupportBeliefState(
                states=belief.states,
                weights=tuple(float(weight) for weight in row),
                deliberation_time=deliberation_time,
                history=list(belief.history),
            )
            scalar_result = mdp.solve_terminal_allocation_result(posterior)
            self.assertEqual(batch_result.allocation, scalar_result.allocation)
            self.assertEqual(batch_result.value, scalar_result.value)
            self.assertEqual(float(batch_value), scalar_result.value)
            self.assertEqual(
                batch_result.value,
                mdp.expected_terminal_utility(posterior, batch_result.allocation),
            )

    def test_performance_diagnostic_is_repeatable_and_counts_evaluations(self) -> None:
        mdp = one_atom_mdp(FiniteSupportAtom(20.0, 0.0, 1))
        diagnostic = diagnose_terminal_performance(
            mdp,
            mdp.initial_belief(),
            repeats=2,
        )

        self.assertTrue(diagnostic.deterministic)
        self.assertEqual(diagnostic.repeats, 2)
        self.assertEqual(diagnostic.row_count, 1)
        self.assertEqual(len(set(diagnostic.objective_evaluations_per_repeat)), 1)
        self.assertGreater(diagnostic.objective_evaluations_per_repeat[0], 0)
        self.assertGreaterEqual(diagnostic.total_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
