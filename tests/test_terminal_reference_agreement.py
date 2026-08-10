from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from src.mdp.finite_support import FiniteSupportAtom
from src.solvers.terminal_reference import (
    solve_terminal_reference_a,
    terminal_reference_a_numerical_method_config_hash,
    terminal_reference_certificate_hash,
    terminal_scientific_spec_hash,
)
from src.solvers.terminal_reference_agreement import (
    TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE,
    _directed_point_distance,
    _interval_distance,
    terminal_reference_agreement_certificate_hash,
    validate_terminal_reference_agreement,
)
from src.solvers.terminal_reference_b import (
    solve_terminal_reference_b,
    terminal_reference_b_numerical_method_config_hash,
)
from tests.test_terminal_optimizer import one_atom_mdp


class TerminalReferenceAgreementTests(unittest.TestCase):
    @staticmethod
    def reference_certificate(record):
        cleared = replace(record, certificate_hash="")
        return replace(
            cleared,
            certificate_hash=terminal_reference_certificate_hash(cleared),
        )

    @staticmethod
    def solve_records(mdp, *, a_cap=200_000, b_cap=500_000, production=None):
        belief = mdp.initial_belief()
        if production is None:
            production = mdp.solve_terminal_allocation_result(belief)
        reference_a = solve_terminal_reference_a(
            mdp,
            belief,
            production.allocation,
            evaluation_cap=a_cap,
        )
        reference_b = solve_terminal_reference_b(
            mdp,
            belief,
            production.allocation,
            evaluation_cap=b_cap,
        )
        return belief, production, reference_a, reference_b

    @staticmethod
    def validate(mdp, belief, production, reference_a, reference_b):
        return validate_terminal_reference_agreement(
            mdp,
            belief,
            production,
            reference_a,
            reference_b,
            scientific_spec_hash=terminal_scientific_spec_hash(mdp),
            reference_a_numerical_method_config_hash=(
                terminal_reference_a_numerical_method_config_hash(
                    reference_a.evaluation_cap
                )
            ),
            reference_b_numerical_method_config_hash=(
                terminal_reference_b_numerical_method_config_hash(
                    reference_b.evaluation_cap
                )
            ),
        )

    def assert_accepted(self, mdp):
        belief, production, reference_a, reference_b = self.solve_records(mdp)
        result = self.validate(mdp, belief, production, reference_a, reference_b)
        self.assertEqual(result.status, "accepted", result.failure_reasons)
        self.assertEqual(result.failure_count, 0)
        self.assertFalse(result.failure_reasons)
        self.assertEqual(
            result.certificate_hash,
            terminal_reference_agreement_certificate_hash(result),
        )
        return result

    def assert_rejected_deterministically(
        self,
        mdp,
        belief,
        production,
        reference_a,
        reference_b,
    ):
        first = self.validate(
            mdp,
            belief,
            production,
            reference_a,
            reference_b,
        )
        second = self.validate(
            mdp,
            belief,
            production,
            reference_a,
            reference_b,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.status, "rejected")
        self.assertEqual(first.failure_count, len(first.failure_reasons))
        self.assertEqual(
            sum(count for _, count in first.failure_reason_counts),
            first.failure_count,
        )
        self.assertEqual(
            tuple(name for name, passed in first.checks if not passed),
            first.failure_reasons,
        )
        self.assertEqual(
            first.certificate_hash,
            terminal_reference_agreement_certificate_hash(first),
        )
        return first

    def test_accepts_unique_boundary(self):
        result = self.assert_accepted(
            one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        )
        self.assertEqual(result.tie_status, "unique")
        self.assertEqual(result.production_allocation, 1.0)

    def test_accepts_unique_kink(self):
        result = self.assert_accepted(
            one_atom_mdp(
                FiniteSupportAtom(0.01, 0.5, -1),
                total_time=1.01,
                terminate_cost=1.0,
            )
        )
        self.assertEqual(result.tie_status, "unique")
        self.assertAlmostEqual(result.production_allocation, 0.25, places=14)

    def test_accepts_unique_smooth_interior(self):
        result = self.assert_accepted(
            one_atom_mdp(FiniteSupportAtom(10.0, 0.3, 1))
        )
        self.assertEqual(result.tie_status, "unique")
        self.assertGreater(result.production_allocation, 0.0)
        self.assertLess(result.production_allocation, 1.0)

    def test_accepts_structural_mirror_pair_with_lower_canonical_side(self):
        result = self.assert_accepted(
            one_atom_mdp(FiniteSupportAtom(80.0, 0.0, 1))
        )
        self.assertEqual(result.tie_status, "structural_symmetry_tie")
        self.assertLess(result.agreed_canonical_allocation_interval[1], 0.5)

    def test_structural_distances_are_directed_at_one_ulp_boundaries(self):
        tolerance = TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE
        below = math.nextafter(tolerance, 0.0)
        above = math.nextafter(tolerance, math.inf)

        self.assertLessEqual(_directed_point_distance(0.0, below), tolerance)
        self.assertEqual(_directed_point_distance(0.0, tolerance), tolerance)
        self.assertGreater(_directed_point_distance(0.0, above), tolerance)
        self.assertLessEqual(_interval_distance((0.0, 0.0), (below, below)), tolerance)
        self.assertEqual(
            _interval_distance((0.0, 0.0), (tolerance, tolerance)),
            tolerance,
        )
        self.assertGreater(
            _interval_distance((0.0, 0.0), (above, above)),
            tolerance,
        )

        # Plain binary64 subtraction rounds this exact stored-float distance inward.
        left = 1.1773756886705876e-06
        right = 0.0002511773756886706
        self.assertEqual(abs(left - right), tolerance)
        self.assertGreater(_directed_point_distance(left, right), tolerance)
        self.assertGreater(
            _interval_distance((left, left), (right, right)),
            tolerance,
        )

    def test_mutually_self_rehashed_bad_structural_proof_fails_closed(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.0, 1))
        belief, production, reference_a, reference_b = self.solve_records(mdp)
        malformed_proof = replace(
            reference_a.structural_symmetry,
            proof_hash="0" * 64,
        )
        malformed_a = self.reference_certificate(
            replace(reference_a, structural_symmetry=malformed_proof)
        )
        malformed_b = self.reference_certificate(
            replace(reference_b, structural_symmetry=malformed_proof)
        )

        result = self.assert_rejected_deterministically(
            mdp,
            belief,
            production,
            malformed_a,
            malformed_b,
        )

        self.assertIn("reference_a_source_valid", result.failure_reasons)
        self.assertIn("reference_b_source_valid", result.failure_reasons)
        self.assertNotIn("structural_symmetry_shape", result.failure_reasons)
        self.assertNotIn("structural_symmetry_same_proof", result.failure_reasons)
        self.assertIn("structural_symmetry_source_valid", result.failure_reasons)

    def test_malformed_structural_types_shapes_and_missing_fields_fail_closed(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.0, 1))
        belief, production, reference_a, reference_b = self.solve_records(mdp)

        missing_candidates = SimpleNamespace(**reference_a.__dict__)
        del missing_candidates.candidate_allocation_intervals
        cases = (
            (
                "wrong_symmetry_type",
                replace(reference_a, structural_symmetry=None),
                replace(reference_b, structural_symmetry=None),
                "structural_symmetry_shape",
            ),
            (
                "missing_candidate_attribute",
                missing_candidates,
                reference_b,
                "structural_candidate_pair_shape",
            ),
            (
                "malformed_candidate_intervals",
                replace(
                    reference_a,
                    candidate_allocation_intervals=((0.1,), None),
                ),
                replace(
                    reference_b,
                    candidate_allocation_intervals=("not-an-interval",),
                ),
                "structural_candidate_pair_shape",
            ),
            (
                "wrong_representative_type",
                replace(reference_a, representative_allocation="lower"),
                reference_b,
                "structural_lower_representatives",
            ),
            (
                "malformed_production_fields",
                replace(
                    reference_a,
                    production_allocation=None,
                    production_value_interval=("low", "high"),
                ),
                reference_b,
                "production_allocation_matches_reference_a",
            ),
        )

        for name, malformed_a, malformed_b, expected_failure in cases:
            with self.subTest(name=name):
                result = self.assert_rejected_deterministically(
                    mdp,
                    belief,
                    production,
                    malformed_a,
                    malformed_b,
                )
                self.assertIn("reference_a_source_valid", result.failure_reasons)
                if malformed_b is not reference_b:
                    self.assertIn("reference_b_source_valid", result.failure_reasons)
                self.assertIn(expected_failure, result.failure_reasons)

    def test_rejects_resolved_requirement_failure(self):
        mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.0, 1),
            delta_learning_per_unit_tutoring=1e-13,
        )
        belief, production, reference_a, reference_b = self.solve_records(
            mdp,
            a_cap=1,
            b_cap=1,
        )

        result = self.validate(mdp, belief, production, reference_a, reference_b)

        self.assertEqual(result.status, "rejected")
        self.assertIn("reference_a_resolved", result.failure_reasons)
        self.assertIn("reference_b_resolved", result.failure_reasons)
        self.assertIn(
            "unresolved_or_incompatible_tie_blocks_acceptance",
            result.failure_reasons,
        )

    def test_rejects_global_interval_disagreement(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference_a, reference_b = self.solve_records(mdp)
        shifted = tuple(value + 0.01 for value in reference_b.global_value_interval)
        forged_b = self.reference_certificate(
            replace(reference_b, global_value_interval=shifted)
        )

        result = self.validate(mdp, belief, production, reference_a, forged_b)

        self.assertEqual(result.status, "rejected")
        self.assertIn("reference_b_source_valid", result.failure_reasons)
        self.assertIn("global_interval_agreement", result.failure_reasons)

    def test_rejects_canonical_interval_disagreement(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference_a, reference_b = self.solve_records(mdp)
        old_evidence = reference_b.candidate_isolation_evidence[0]
        forged_evidence = replace(
            old_evidence,
            allocation_interval=(0.0, 0.0),
            witness_allocation=0.0,
        )
        forged_b = self.reference_certificate(
            replace(
                reference_b,
                candidate_allocation_intervals=((0.0, 0.0),),
                candidate_isolation_evidence=(forged_evidence,),
                canonical_allocation_interval=(0.0, 0.0),
                representative_allocation=0.0,
            )
        )

        result = self.validate(mdp, belief, production, reference_a, forged_b)

        self.assertEqual(result.status, "rejected")
        self.assertIn("reference_b_source_valid", result.failure_reasons)
        self.assertIn("ordinary_canonical_interval_agreement", result.failure_reasons)

    def test_rejects_cross_case_identity_mixing(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference_a, _ = self.solve_records(mdp)
        other_mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.5, -1),
            sigma_sample=3.0,
        )
        _, _, _, other_reference_b = self.solve_records(other_mdp)

        result = self.validate(
            mdp,
            belief,
            production,
            reference_a,
            other_reference_b,
        )

        self.assertEqual(result.status, "rejected")
        self.assertIn("reference_b_source_valid", result.failure_reasons)
        self.assertIn("cross_reference_mdp_identity", result.failure_reasons)
        self.assertIn("cross_reference_scientific_identity", result.failure_reasons)

    def test_rejects_separate_numerical_identity_mixing(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference_a, reference_b = self.solve_records(mdp)

        result = validate_terminal_reference_agreement(
            mdp,
            belief,
            production,
            reference_a,
            reference_b,
            scientific_spec_hash=terminal_scientific_spec_hash(mdp),
            reference_a_numerical_method_config_hash=(
                reference_b.numerical_method_config_hash
            ),
            reference_b_numerical_method_config_hash=(
                reference_a.numerical_method_config_hash
            ),
        )

        self.assertEqual(result.status, "rejected")
        self.assertIn(
            "reference_a_numerical_identity_matches_source",
            result.failure_reasons,
        )
        self.assertIn(
            "reference_b_numerical_identity_matches_source",
            result.failure_reasons,
        )

    def test_rejects_incompatible_tie_classification(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.0, 1))
        belief, production, reference_a, reference_b = self.solve_records(mdp)
        forged_b = self.reference_certificate(replace(reference_b, tie_status="unique"))

        result = self.validate(mdp, belief, production, reference_a, forged_b)

        self.assertEqual(result.status, "rejected")
        self.assertIn("reference_b_source_valid", result.failure_reasons)
        self.assertIn("tie_classification_compatible", result.failure_reasons)

    def test_rejects_self_rehashed_fabricated_reference(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference_a, reference_b = self.solve_records(mdp)
        fabricated_b = self.reference_certificate(
            replace(reference_b, stopping_reason="fabricated_source_trace")
        )

        result = self.validate(mdp, belief, production, reference_a, fabricated_b)

        self.assertEqual(result.status, "rejected")
        self.assertIn("reference_b_source_valid", result.failure_reasons)

    def test_rejects_source_valid_references_for_suboptimal_production(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief = mdp.initial_belief()
        optimal = mdp.solve_terminal_allocation_result(belief)
        suboptimal = replace(
            optimal,
            allocation=0.0,
            value=mdp.expected_terminal_utility(belief, 0.0),
        )
        reference_a = solve_terminal_reference_a(mdp, belief, 0.0)
        reference_b = solve_terminal_reference_b(mdp, belief, 0.0)

        result = self.validate(mdp, belief, suboptimal, reference_a, reference_b)

        self.assertEqual(result.status, "rejected")
        self.assertNotIn("reference_a_source_valid", result.failure_reasons)
        self.assertNotIn("reference_b_source_valid", result.failure_reasons)
        self.assertIn("production_value_global_distance", result.failure_reasons)
        self.assertIn("production_regret_upper_bound", result.failure_reasons)
        self.assertIn("production_allocation_canonical_distance", result.failure_reasons)

    def test_failure_counts_cover_every_failed_check(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference_a, reference_b = self.solve_records(
            mdp,
            a_cap=1,
            b_cap=1,
        )

        result = self.validate(mdp, belief, production, reference_a, reference_b)

        self.assertEqual(result.failure_count, len(result.failure_reasons))
        self.assertEqual(
            sum(count for _, count in result.failure_reason_counts),
            result.failure_count,
        )
        self.assertEqual(
            tuple(name for name, passed in result.checks if not passed),
            result.failure_reasons,
        )
        self.assertTrue(all(count == 1 for _, count in result.failure_reason_counts))
        self.assertTrue(math.isfinite(result.production_value_interval[0]))


if __name__ == "__main__":
    unittest.main()
