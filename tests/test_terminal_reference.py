"""Test purpose: validate the first independent terminal reference solver and its certificates."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
import unittest
from unittest.mock import patch

from src.mdp.finite_support import (
    FiniteSupportAtom,
    FiniteSupportMetaMDP,
    FiniteSupportPrior,
)
from src.solvers.terminal import prove_recipient_swap_symmetry
from src.solvers.terminal_reference import (
    REFERENCE_A_BRANCH_RULE,
    REFERENCE_PRECISION_LADDER,
    TERMINAL_TIE_SCALE,
    CandidateIsolationEvidence,
    _AObjective,
    _CandidateSummary,
    _SearchSnapshot,
    _absolute_difference_interval,
    _add_up,
    _regret_interval,
    _resolve_candidates,
    _run_reference_a_level,
    _strictly_dominates,
    _tau_bounds,
    _validate_terminal_reference_record_shape,
    solve_terminal_reference_a,
    source_validate_terminal_reference_record,
    terminal_belief_identity_hash,
    terminal_mdp_identity_hash,
    terminal_reference_a_numerical_method_config_hash,
    terminal_reference_certificate_hash,
    terminal_scientific_spec_hash,
    validate_production_against_reference_a,
    validate_terminal_reference_record,
)
from tests.test_terminal_optimizer import one_atom_mdp, terminal_config


class TerminalReferenceATests(unittest.TestCase):
    @staticmethod
    def identity_hashes(mdp, evaluation_cap=200_000):
        return (
            terminal_scientific_spec_hash(mdp),
            terminal_reference_a_numerical_method_config_hash(evaluation_cap),
        )

    @staticmethod
    def rehash(record):
        cleared = replace(record, certificate_hash="")
        return replace(
            cleared,
            certificate_hash=terminal_reference_certificate_hash(cleared),
        )

    def solve(self, mdp, *, evaluation_cap=200_000):
        belief = mdp.initial_belief()
        production = mdp.solve_terminal_allocation_result(belief)
        reference = solve_terminal_reference_a(
            mdp,
            belief,
            production.allocation,
            evaluation_cap=evaluation_cap,
        )
        return belief, production, reference

    def validate_record(self, mdp, belief, reference):
        scientific_hash, numerical_hash = self.identity_hashes(
            mdp,
            reference.evaluation_cap,
        )
        return validate_terminal_reference_record(
            reference,
            mdp,
            belief,
            scientific_spec_hash=scientific_hash,
            numerical_method_config_hash=numerical_hash,
        )

    def validate_production(self, mdp, belief, production, reference):
        scientific_hash, numerical_hash = self.identity_hashes(
            mdp,
            reference.evaluation_cap,
        )
        return validate_production_against_reference_a(
            mdp,
            belief,
            production,
            reference,
            scientific_spec_hash=scientific_hash,
            numerical_method_config_hash=numerical_hash,
        )

    def test_source_validation_proof_is_identity_bound_and_public_default_recomputes(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference = self.solve(mdp)
        scientific_hash, numerical_hash = self.identity_hashes(
            mdp, reference.evaluation_cap
        )
        proof = source_validate_terminal_reference_record(
            reference,
            mdp,
            belief,
            scientific_spec_hash=scientific_hash,
            numerical_method_config_hash=numerical_hash,
        )
        with patch(
            "src.solvers.terminal_reference.validate_terminal_reference_record",
            side_effect=AssertionError("proof path recomputed the reference"),
        ):
            accepted = validate_production_against_reference_a(
                mdp,
                belief,
                production,
                reference,
                scientific_spec_hash=scientific_hash,
                numerical_method_config_hash=numerical_hash,
                _source_validation_proof=proof,
            )
            copied_record = replace(reference)
            rejected = validate_production_against_reference_a(
                mdp,
                belief,
                production,
                copied_record,
                scientific_spec_hash=scientific_hash,
                numerical_method_config_hash=numerical_hash,
                _source_validation_proof=proof,
            )
            forged_proof = replace(proof, _seal=object())
            forged = validate_production_against_reference_a(
                mdp,
                belief,
                production,
                reference,
                scientific_spec_hash=scientific_hash,
                numerical_method_config_hash=numerical_hash,
                _source_validation_proof=forged_proof,
            )
        self.assertEqual(accepted.status, "accepted")
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(forged.status, "rejected")
        self.assertIn("reference_a_source_recomputation", rejected.failures)
        self.assertIn("reference_a_source_recomputation", forged.failures)

        original_time = belief.deliberation_time
        try:
            object.__setattr__(belief, "deliberation_time", original_time + 1.0)
            mutated = validate_production_against_reference_a(
                mdp,
                belief,
                production,
                reference,
                scientific_spec_hash=scientific_hash,
                numerical_method_config_hash=numerical_hash,
                _source_validation_proof=proof,
            )
        finally:
            object.__setattr__(belief, "deliberation_time", original_time)
        self.assertEqual(mutated.status, "rejected")
        self.assertIn("reference_a_source_recomputation", mutated.failures)

        with patch(
            "src.solvers.terminal_reference.validate_terminal_reference_record",
            wraps=validate_terminal_reference_record,
        ) as recompute:
            default = validate_production_against_reference_a(
                mdp,
                belief,
                production,
                reference,
                scientific_spec_hash=scientific_hash,
                numerical_method_config_hash=numerical_hash,
            )
        self.assertEqual(default.status, "accepted")
        self.assertEqual(recompute.call_count, 1)

    def assert_resolved_and_valid(self, mdp, belief, production, reference):
        self.assertEqual(reference.status, "resolved")
        validation = self.validate_production(
            mdp,
            belief,
            production,
            reference,
        )
        self.assertEqual(validation.status, "accepted", validation.failures)
        self.assertFalse(validation.failures)
        self.assertLessEqual(
            reference.global_value_interval[1]
            - reference.global_value_interval[0],
            reference.precision_level + 1e-15,
        )

    @staticmethod
    def candidate(
        allocation_interval,
        value_interval,
        witness_allocation,
        witness_value,
    ):
        return _CandidateSummary(
            allocation_interval=allocation_interval,
            value_interval=value_interval,
            witness_allocation=witness_allocation,
            witness_value=witness_value,
            partition_count=1,
            maximum_depth=1,
            isolation_rule=REFERENCE_A_BRANCH_RULE,
        )

    def test_unique_boundary_reference_binds_all_source_identities(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference = self.solve(mdp)

        self.assert_resolved_and_valid(mdp, belief, production, reference)
        self.assertEqual(reference.tie_status, "unique")
        self.assertEqual(production.allocation, 1.0)
        self.assertEqual(reference.mdp_identity_hash, terminal_mdp_identity_hash(mdp))
        self.assertEqual(
            reference.belief_identity_hash,
            terminal_belief_identity_hash(belief),
        )
        self.assertEqual(
            reference.scientific_spec_hash,
            terminal_scientific_spec_hash(mdp),
        )
        self.assertEqual(
            reference.numerical_method_config_hash,
            terminal_reference_a_numerical_method_config_hash(),
        )

    def test_nearly_coincident_kinks_fail_closed_when_bound_cannot_tighten(self):
        prior = FiniteSupportPrior(
            (
                FiniteSupportAtom(30.0, 0.4, -1),
                FiniteSupportAtom(60.0, 0.0, -1),
            ),
            (0.5071635969746414, 0.4928364030253586),
        )
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief, _, reference = self.solve(mdp)

        self.assertEqual(reference.status, "reference_unresolved")
        self.assertEqual(reference.tie_status, "reference_unresolved")
        self.assertEqual(
            reference.stopping_reason,
            "global_value_interval_precision_ladder_exhausted",
        )
        self.assertTrue(self.validate_record(mdp, belief, reference))

    def test_unique_interior_reference_refines_until_isolated(self):
        mdp = one_atom_mdp(FiniteSupportAtom(10.0, 0.3, 1))
        belief, production, reference = self.solve(mdp)

        self.assert_resolved_and_valid(mdp, belief, production, reference)
        self.assertEqual(reference.tie_status, "unique")
        self.assertEqual(reference.precision_level, 1e-8)
        self.assertLessEqual(
            reference.canonical_allocation_interval[1]
            - reference.canonical_allocation_interval[0],
            1e-4,
        )

    def test_structural_mirror_pair_always_uses_lower_canonical_side(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.0, 1))
        belief, production, reference = self.solve(mdp)

        self.assert_resolved_and_valid(mdp, belief, production, reference)
        self.assertEqual(reference.tie_status, "structural_symmetry_tie")
        self.assertEqual(reference.canonical_allocation_interval, min(
            reference.candidate_allocation_intervals
        ))
        self.assertLess(reference.representative_allocation, 0.5)

    def test_structural_canonical_side_ignores_asymmetric_witness_distances(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.0, 1))
        belief = mdp.initial_belief()
        right = self.candidate((0.79, 0.80), (1.0, 1.0), 0.79, 1.0)
        left = self.candidate((0.20, 0.21), (1.0, 1.0), 0.20, 1.0)
        snapshot = _SearchSnapshot((1.0, 1.0), (right, left))

        status, _, canonical_index, _ = _resolve_candidates(
            snapshot,
            prove_recipient_swap_symmetry(mdp, belief),
        )

        self.assertEqual(status, "structural_symmetry_tie")
        self.assertEqual(canonical_index, 1)

    def test_ordinary_near_tie_is_provisional_before_adaptive_refinement(self):
        mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.0, 1),
            delta_learning_per_unit_tutoring=1e-8,
        )
        belief = mdp.initial_belief()
        objective = _AObjective(mdp, belief, 200_000)
        snapshot = _run_reference_a_level(objective, REFERENCE_PRECISION_LADDER[0])
        tie_status, candidates, canonical_index, reason = _resolve_candidates(
            snapshot,
            prove_recipient_swap_symmetry(mdp, belief),
        )

        self.assertIsNone(tie_status)
        self.assertIsNone(canonical_index)
        self.assertEqual(reason, "ordinary_tie_provisional")
        self.assertEqual(len(candidates), 2)

    def test_adaptive_ladder_refines_near_tie_to_unique(self):
        mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.0, 1),
            delta_learning_per_unit_tutoring=1e-8,
        )
        belief, production, reference = self.solve(mdp)

        self.assert_resolved_and_valid(mdp, belief, production, reference)
        self.assertEqual(reference.tie_status, "unique")
        self.assertEqual(reference.precision_level, 1e-8)

    def test_constant_objective_is_unresolved_without_canonical_allocation(self):
        mdp = one_atom_mdp(
            FiniteSupportAtom(20.0, 0.0, 1),
            total_time=1.0,
            terminate_cost=1.0,
        )
        belief, production, reference = self.solve(mdp)

        self.assertEqual(
            mdp.expected_terminal_utility(belief, 0.0),
            mdp.expected_terminal_utility(belief, 1.0),
        )
        self.assertEqual(reference.status, "reference_unresolved")
        self.assertEqual(reference.tie_status, "reference_unresolved")
        self.assertIsNone(reference.canonical_allocation_interval)
        self.assertIsNone(reference.representative_allocation)
        self.assertEqual(
            reference.stopping_reason,
            "connected_plateau_requires_multiple_maximizer_rule",
        )
        self.assertTrue(self.validate_record(mdp, belief, reference))
        self.assertEqual(
            self.validate_production(mdp, belief, production, reference).status,
            "rejected",
        )

    def test_broad_connected_candidate_is_not_unique(self):
        candidate = self.candidate((0.2, 0.8), (1.0, 1.0), 0.5, 1.0)
        mdp = one_atom_mdp(FiniteSupportAtom(20.0, 0.0, 1))
        status, _, canonical_index, reason = _resolve_candidates(
            _SearchSnapshot((1.0, 1.0), (candidate,)),
            prove_recipient_swap_symmetry(mdp, mdp.initial_belief()),
        )
        self.assertIsNone(status)
        self.assertIsNone(canonical_index)
        self.assertEqual(reason, "connected_maximizer_region_provisional")

    def test_evaluation_cap_fails_closed_with_finite_bounds(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference = self.solve(mdp, evaluation_cap=1)

        self.assertEqual(reference.status, "reference_unresolved")
        self.assertEqual(reference.stopping_reason, "evaluation_cap_exhausted")
        self.assertTrue(all(math.isfinite(value) for value in reference.global_value_interval))
        self.assertTrue(self.validate_record(mdp, belief, reference))
        validation = self.validate_production(
            mdp,
            belief,
            production,
            reference,
        )
        self.assertEqual(validation.status, "rejected")
        self.assertIn("reference_a_resolved", validation.failures)

    def test_production_value_and_regret_intervals_are_checked(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference = self.solve(mdp)

        self.assert_resolved_and_valid(mdp, belief, production, reference)
        self.assertLessEqual(reference.production_value_interval[0], production.value)
        self.assertGreaterEqual(reference.production_value_interval[1], production.value)
        self.assertGreaterEqual(reference.production_regret_interval[0], 0.0)
        self.assertLessEqual(reference.production_regret_interval[1], 1e-4)

    def test_stale_certificate_tamper_is_rejected(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, _, reference = self.solve(mdp)
        tampered = replace(
            reference,
            production_allocation=reference.production_allocation - 0.01,
        )

        self.assertFalse(self.validate_record(mdp, belief, tampered))

    def test_actual_mdp_belief_and_config_identity_mismatches_are_rejected(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, _, reference = self.solve(mdp)
        scientific_hash, numerical_hash = self.identity_hashes(mdp)

        altered_mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.5, -1),
            total_time=41.0,
        )
        altered_belief = belief.copy()
        altered_belief.history.append({"action": 1.0})

        self.assertFalse(validate_terminal_reference_record(
            reference,
            altered_mdp,
            altered_mdp.initial_belief(),
            scientific_spec_hash=terminal_scientific_spec_hash(altered_mdp),
            numerical_method_config_hash=numerical_hash,
        ))
        self.assertFalse(validate_terminal_reference_record(
            reference,
            mdp,
            altered_belief,
            scientific_spec_hash=scientific_hash,
            numerical_method_config_hash=numerical_hash,
        ))
        self.assertFalse(validate_terminal_reference_record(
            reference,
            mdp,
            belief,
            scientific_spec_hash="0" * 64,
            numerical_method_config_hash=numerical_hash,
        ))
        self.assertFalse(validate_terminal_reference_record(
            reference,
            mdp,
            belief,
            scientific_spec_hash=scientific_hash,
            numerical_method_config_hash="f" * 64,
        ))

    def test_self_rehashed_suboptimal_fabrication_fails_source_recomputation(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference = self.solve(mdp)
        suboptimal_allocation = 0.0
        suboptimal_value = mdp.expected_terminal_utility(
            belief,
            suboptimal_allocation,
        )
        value_interval = (
            suboptimal_value,
            math.nextafter(suboptimal_value, math.inf),
        )
        production_interval = (
            math.nextafter(suboptimal_value, -math.inf),
            math.nextafter(suboptimal_value, math.inf),
        )
        allocation_interval = (0.0, 1e-6)
        evidence = CandidateIsolationEvidence(
            allocation_interval=allocation_interval,
            value_interval=value_interval,
            witness_allocation=suboptimal_allocation,
            witness_value=suboptimal_value,
            partition_count=1,
            maximum_depth=1,
            isolation_rule=REFERENCE_A_BRANCH_RULE,
        )
        forged = self.rehash(replace(
            reference,
            global_value_interval=value_interval,
            candidate_allocation_intervals=(allocation_interval,),
            candidate_value_intervals=(value_interval,),
            candidate_isolation_evidence=(evidence,),
            canonical_allocation_interval=allocation_interval,
            representative_allocation=suboptimal_allocation,
            production_allocation=suboptimal_allocation,
            production_value_interval=production_interval,
            production_regret_interval=_regret_interval(
                value_interval,
                production_interval,
            ),
            precision_level=1e-6,
            stopping_reason="unique_candidate_isolated",
        ))
        forged_production = replace(
            production,
            allocation=suboptimal_allocation,
            value=suboptimal_value,
        )

        self.assertTrue(_validate_terminal_reference_record_shape(forged))
        self.assertGreater(production.value - suboptimal_value, 1.0)
        self.assertFalse(self.validate_record(mdp, belief, forged))
        validation = self.validate_production(
            mdp,
            belief,
            forged_production,
            forged,
        )
        self.assertEqual(validation.status, "rejected")
        self.assertIn("reference_a_source_recomputation", validation.failures)

    def test_rehashed_nonfinite_and_semantically_invalid_evidence_is_rejected(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, _, reference = self.solve(mdp)
        evidence = reference.candidate_isolation_evidence[0]

        infinite_interval = (-math.inf, math.inf)
        invalid_records = {
            "nonfinite_candidate": replace(
                reference,
                candidate_value_intervals=(infinite_interval,),
                candidate_isolation_evidence=(
                    replace(evidence, value_interval=infinite_interval),
                ),
            ),
            "fabricated_rule": replace(
                reference,
                candidate_isolation_evidence=(
                    replace(evidence, isolation_rule="fabricated_rule"),
                ),
            ),
            "partition_depth": replace(
                reference,
                candidate_isolation_evidence=(
                    replace(
                        evidence,
                        partition_count=reference.objective_evaluation_count + 1,
                        maximum_depth=reference.objective_evaluation_count + 1,
                    ),
                ),
            ),
            "altered_witness": replace(
                reference,
                candidate_isolation_evidence=(
                    replace(
                        evidence,
                        witness_value=math.nextafter(
                            evidence.witness_value,
                            math.inf,
                        ),
                    ),
                ),
            ),
        }

        for name, candidate in invalid_records.items():
            with self.subTest(name=name):
                rehashed = self.rehash(candidate)
                self.assertFalse(
                    _validate_terminal_reference_record_shape(rehashed)
                )
                self.assertFalse(self.validate_record(mdp, belief, rehashed))

    def test_directed_tau_bounds_enclose_exact_threshold(self):
        tau_low, tau_high = _tau_bounds((2.0, 2.0))
        exact = Fraction.from_float(TERMINAL_TIE_SCALE) * 2

        self.assertLessEqual(Fraction.from_float(tau_low), exact)
        self.assertGreaterEqual(Fraction.from_float(tau_high), exact)
        self.assertLess(tau_low, float(exact))
        self.assertGreater(tau_high, float(exact))

    def test_one_ulp_tie_threshold_is_conservative(self):
        mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.0, 1),
            delta_learning_per_unit_tutoring=1e-8,
        )
        symmetry = prove_recipient_swap_symmetry(mdp, mdp.initial_belief())
        tau_low, _ = _tau_bounds((1.0, 1.0))
        tie_difference = math.nextafter(tau_low, 0.0)
        provisional_difference = tau_low

        tied = _SearchSnapshot(
            (1.0, 1.0),
            (
                self.candidate((0.1, 0.2), (0.0, 0.0), 0.1, 0.0),
                self.candidate(
                    (0.8, 0.9),
                    (tie_difference, tie_difference),
                    0.8,
                    tie_difference,
                ),
            ),
        )
        provisional = _SearchSnapshot(
            (1.0, 1.0),
            (
                self.candidate((0.1, 0.2), (0.0, 0.0), 0.1, 0.0),
                self.candidate(
                    (0.8, 0.9),
                    (provisional_difference, provisional_difference),
                    0.8,
                    provisional_difference,
                ),
            ),
        )

        self.assertEqual(_resolve_candidates(tied, symmetry)[0], "certified_value_tie")
        self.assertIsNone(_resolve_candidates(provisional, symmetry)[0])

    def test_one_ulp_non_tie_threshold_is_conservative(self):
        _, tau_high = _tau_bounds((1.0, 1.0))
        second = self.candidate((0.8, 0.9), (0.0, 0.0), 0.8, 0.0)
        directed_boundary = _add_up(0.0, tau_high)
        at_boundary = self.candidate(
            (0.1, 0.2),
            (directed_boundary, directed_boundary),
            0.1,
            directed_boundary,
        )
        one_ulp_above = math.nextafter(directed_boundary, math.inf)
        above_boundary = self.candidate(
            (0.1, 0.2),
            (one_ulp_above, one_ulp_above),
            0.1,
            one_ulp_above,
        )

        self.assertFalse(_strictly_dominates(at_boundary, second, tau_high))
        self.assertTrue(_strictly_dominates(above_boundary, second, tau_high))
        lower, upper = _absolute_difference_interval(
            above_boundary.value_interval,
            second.value_interval,
        )
        self.assertLessEqual(lower, one_ulp_above)
        self.assertGreaterEqual(upper, one_ulp_above)


if __name__ == "__main__":
    unittest.main()
