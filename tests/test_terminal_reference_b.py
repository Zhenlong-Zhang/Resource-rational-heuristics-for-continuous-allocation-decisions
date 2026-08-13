from __future__ import annotations

from dataclasses import replace
import ast
import inspect
import math
from unittest import TestCase
from unittest.mock import patch

from src.mdp.finite_support import (
    FiniteSupportAtom,
    FiniteSupportMetaMDP,
    FiniteSupportPrior,
)
from src.solvers.terminal_reference import (
    terminal_belief_identity_hash,
    terminal_mdp_identity_hash,
    terminal_reference_certificate_hash,
    terminal_scientific_spec_hash,
)
from src.solvers.terminal_reference_b import (
    REFERENCE_B_CANONICAL_ORDER,
    REFERENCE_B_CANDIDATE_ORDER,
    REFERENCE_B_CHILD_ORDER,
    REFERENCE_B_BRANCH_RULE,
    REFERENCE_B_HEAP_ORDER,
    REFERENCE_B_WITNESS_ORDER,
    _BCandidate,
    _BObjective,
    _BSnapshot,
    _add_up,
    _make_b_node,
    _resolve_b_candidates,
    _tau_bounds,
    solve_terminal_reference_b,
    terminal_reference_b_numerical_method_config_hash,
    validate_terminal_reference_b_record,
)
import src.solvers.terminal_reference_b as reference_b_module
from tests.test_terminal_optimizer import one_atom_mdp, terminal_config


_REFERENCE_B_ALLOWED_MODULE_IMPORTS = {"hashlib", "heapq", "json", "math"}
_REFERENCE_B_ALLOWED_FROM_IMPORTS = {
    "__future__": {"annotations"},
    "dataclasses": {"dataclass", "replace"},
    "fractions": {"Fraction"},
    "typing": {"Any", "Dict", "List", "Mapping", "Optional", "Sequence", "Tuple"},
    "terminal": {"StructuralSymmetry", "prove_recipient_swap_symmetry"},
    "terminal_reference": {
        "_SOURCE_VALIDATION_PROOF_SEAL",
        "CandidateIsolationEvidence",
        "TerminalReferenceRecord",
        "TerminalReferenceSourceValidationProof",
        "terminal_belief_identity_hash",
        "terminal_mdp_identity_hash",
        "terminal_reference_certificate_hash",
        "terminal_scientific_spec_hash",
    },
}
_REFERENCE_B_FORBIDDEN_CALLS = {
    "optimize_terminal_allocation",
    "solve_terminal_reference_a",
    "_run_reference_a_level",
    "rational_power_bounds_7_20",
    "terminal_objective_upper_bound",
    "expected_terminal_utility",
}


def _reference_b_dependency_violations(source: str):
    """Return imports/calls that cross Reference B's permanent dependency boundary."""

    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _REFERENCE_B_ALLOWED_MODULE_IMPORTS:
                    violations.append(f"module_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[-1]
            imported = {alias.name for alias in node.names}
            allowed = _REFERENCE_B_ALLOWED_FROM_IMPORTS.get(module)
            if allowed is None:
                violations.append(f"from_module:{node.module or ''}")
            elif imported != allowed:
                for name in sorted(imported - allowed):
                    violations.append(f"from_name:{node.module or ''}:{name}")
                for name in sorted(allowed - imported):
                    violations.append(f"missing_allowed_name:{module}:{name}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            else:
                call_name = ""
            if call_name in _REFERENCE_B_FORBIDDEN_CALLS:
                violations.append(f"forbidden_call:{call_name}")
            if call_name in {"__import__", "import_module"} and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    module_name = argument.value
                    if module_name not in _REFERENCE_B_ALLOWED_MODULE_IMPORTS:
                        violations.append(f"dynamic_import:{module_name}")
    return tuple(violations)


class TerminalReferenceBTests(TestCase):
    @staticmethod
    def candidate(
        allocation_interval,
        value_interval,
        witness_allocation,
        witness_value,
    ):
        return _BCandidate(
            allocation_interval=allocation_interval,
            value_interval=value_interval,
            witness_allocation=witness_allocation,
            witness_value=witness_value,
            partition_count=1,
            maximum_depth=1,
        )

    @staticmethod
    def solve(mdp, *, evaluation_cap=500_000):
        belief = mdp.initial_belief()
        production = mdp.solve_terminal_allocation_result(belief)
        reference = solve_terminal_reference_b(
            mdp,
            belief,
            production.allocation,
            evaluation_cap=evaluation_cap,
        )
        return belief, production, reference

    @staticmethod
    def validate(mdp, belief, record):
        return validate_terminal_reference_b_record(
            record,
            mdp,
            belief,
            scientific_spec_hash=terminal_scientific_spec_hash(mdp),
            numerical_method_config_hash=(
                terminal_reference_b_numerical_method_config_hash(
                    record.evaluation_cap
                )
            ),
        )

    @staticmethod
    def rehash(record):
        cleared = replace(record, certificate_hash="")
        return replace(
            cleared,
            certificate_hash=terminal_reference_certificate_hash(cleared),
        )

    def assert_valid(self, mdp, belief, record):
        self.assertTrue(self.validate(mdp, belief, record))
        self.assertEqual(record.mdp_identity_hash, terminal_mdp_identity_hash(mdp))
        self.assertEqual(
            record.belief_identity_hash,
            terminal_belief_identity_hash(belief),
        )
        self.assertEqual(
            record.numerical_method_config_hash,
            terminal_reference_b_numerical_method_config_hash(
                record.evaluation_cap
            ),
        )

    def test_b_is_independent_of_production_and_reference_a_entrypoints(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief = mdp.initial_belief()
        with (
            patch(
                "src.solvers.terminal.optimize_terminal_allocation",
                side_effect=AssertionError("production called"),
            ),
            patch(
                "src.solvers.terminal_reference.solve_terminal_reference_a",
                side_effect=AssertionError("Reference A called"),
            ),
            patch(
                "src.solvers.terminal_reference._run_reference_a_level",
                side_effect=AssertionError("Reference-A search called"),
            ),
            patch(
                "src.solvers.terminal.terminal_objective_upper_bound",
                side_effect=AssertionError("production bound called"),
            ),
            patch(
                "src.solvers.terminal.rational_power_bounds_7_20",
                side_effect=AssertionError("production power bound called"),
            ),
            patch.object(
                FiniteSupportMetaMDP,
                "expected_terminal_utility",
                side_effect=AssertionError("public objective called"),
            ),
        ):
            reference = solve_terminal_reference_b(mdp, belief, 1.0)
            self.assertTrue(self.validate(mdp, belief, reference))

        self.assertEqual(reference.status, "resolved")
        self.assertEqual(reference.tie_status, "unique")

    def test_unique_boundary_is_resolved_and_source_bound(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, production, reference = self.solve(mdp)

        self.assertEqual(reference.status, "resolved")
        self.assertEqual(reference.tie_status, "unique")
        self.assertEqual(production.allocation, 1.0)
        self.assertIn(1.0, reference.canonical_allocation_interval)
        self.assertLessEqual(
            reference.global_value_interval[1]
            - reference.global_value_interval[0],
            1e-6,
        )
        self.assert_valid(mdp, belief, reference)

    def test_unique_smooth_interior_is_resolved(self):
        mdp = one_atom_mdp(FiniteSupportAtom(10.0, 0.3, 1))
        belief, production, reference = self.solve(mdp)

        self.assertEqual(reference.status, "resolved")
        self.assertEqual(reference.tie_status, "unique")
        self.assertLess(production.allocation, 1.0)
        self.assertGreater(production.allocation, 0.0)
        self.assertLessEqual(
            reference.canonical_allocation_interval[0],
            production.allocation,
        )
        self.assertGreaterEqual(
            reference.canonical_allocation_interval[1],
            production.allocation,
        )
        self.assert_valid(mdp, belief, reference)

    def test_structural_mirror_pair_uses_lower_canonical_interval(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.0, 1))
        belief, _, reference = self.solve(mdp)

        self.assertEqual(reference.status, "resolved")
        self.assertEqual(reference.tie_status, "structural_symmetry_tie")
        self.assertEqual(
            reference.canonical_allocation_interval,
            min(reference.candidate_allocation_intervals),
        )
        self.assertLess(reference.representative_allocation, 0.5)
        self.assert_valid(mdp, belief, reference)

    def test_ordinary_near_tie_is_certified_without_structural_symmetry(self):
        mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.0, 1),
            delta_learning_per_unit_tutoring=1e-13,
        )
        belief, _, reference = self.solve(mdp)

        self.assertFalse(reference.structural_symmetry.valid)
        self.assertEqual(reference.status, "resolved")
        self.assertEqual(reference.tie_status, "certified_value_tie")
        self.assertEqual(len(reference.candidate_allocation_intervals), 2)
        self.assert_valid(mdp, belief, reference)

    def test_unrepresentable_kink_interval_fails_closed_with_finite_evidence(self):
        prior = FiniteSupportPrior(
            (
                FiniteSupportAtom(30.0, 0.4, -1),
                FiniteSupportAtom(60.0, 0.0, -1),
            ),
            (0.5071635969746414, 0.4928364030253586),
        )
        mdp = FiniteSupportMetaMDP(terminal_config(), prior)
        belief, _, reference = self.solve(mdp)
        kink = 9.0 / 39.0

        self.assertEqual(reference.status, "reference_unresolved")
        self.assertEqual(
            reference.stopping_reason,
            "global_value_interval_precision_ladder_exhausted",
        )
        self.assertTrue(
            any(lower <= kink <= upper for lower, upper in reference.candidate_allocation_intervals)
        )
        self.assertTrue(
            all(
                math.isfinite(value)
                for interval in reference.candidate_value_intervals
                for value in interval
            )
        )
        self.assert_valid(mdp, belief, reference)

    def test_constant_plateau_is_unresolved_without_unearned_canonical(self):
        mdp = one_atom_mdp(
            FiniteSupportAtom(20.0, 0.0, 1),
            total_time=1.0,
            terminate_cost=1.0,
        )
        belief, _, reference = self.solve(mdp)

        self.assertEqual(reference.status, "reference_unresolved")
        self.assertEqual(reference.tie_status, "reference_unresolved")
        self.assertIsNone(reference.canonical_allocation_interval)
        self.assertIsNone(reference.representative_allocation)
        self.assertEqual(
            reference.stopping_reason,
            "connected_plateau_requires_multiple_maximizer_rule",
        )
        self.assert_valid(mdp, belief, reference)

    def test_evaluation_cap_is_bounded_and_source_valid(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, _, reference = self.solve(mdp, evaluation_cap=1)

        self.assertEqual(reference.status, "reference_unresolved")
        self.assertEqual(reference.stopping_reason, "evaluation_cap_exhausted")
        self.assertEqual(reference.objective_evaluation_count, 1)
        self.assertTrue(
            all(math.isfinite(value) for value in reference.global_value_interval)
        )
        self.assert_valid(mdp, belief, reference)

    def test_independent_fixed_value_intervals_and_bounds_contain_public_values(self):
        mdp = one_atom_mdp(FiniteSupportAtom(10.0, 0.3, 1))
        belief = mdp.initial_belief()
        objective = _BObjective(mdp, belief, 10_000)
        intervals = ((0.0, 0.25), (0.25, 0.75), (0.75, 1.0))

        for lower, upper in intervals:
            bound = objective.upper_bound(lower, upper)
            for index in range(21):
                allocation = lower + (upper - lower) * index / 20
                point = objective.fixed_value(allocation)
                public_value = mdp.expected_terminal_utility(belief, allocation)
                self.assertLessEqual(point.lower, public_value)
                self.assertGreaterEqual(point.upper, public_value)
                self.assertGreaterEqual(bound, point.upper)

    def test_exact_depth_50_stored_objective_counterexample_is_enclosed(self):
        prior = FiniteSupportPrior(
            (
                FiniteSupportAtom(76.84838606823047, 0.6800095506418942, -1),
                FiniteSupportAtom(175.80513802378826, 0.7210628632779053, -1),
                FiniteSupportAtom(63.7183980952953, 0.2143784616568827, -1),
            ),
            (0.8673576780725962, 0.3575742105767925, 0.27793346353764253),
        )
        mdp = FiniteSupportMetaMDP(
            terminal_config(
                total_time=74.04535348250742,
                terminate_cost=1.2617703123633988,
                learning_per_unit_of_tutoring=1.1103354853079421,
                delta_learning_per_unit_tutoring=0.353722054571448,
                lambda_shortfall=3.780813301365045,
            ),
            prior,
        )
        belief = mdp.initial_belief()
        lower = 0.29744321776213223
        upper = 0.2974432177621331
        interior = 0.2974432177621323
        node = _make_b_node(_BObjective(mdp, belief, 10_000), lower, upper, 50)
        stored_value = mdp.expected_terminal_utility(belief, interior)

        self.assertEqual(node.depth, 50)
        self.assertGreaterEqual(node.upper_bound, stored_value)
        self.assertGreater(node.upper_bound, -11.539149375225781)

    def test_deep_dyadic_near_kink_and_flat_nodes_enclose_every_float(self):
        cases = []
        kink_mdp = one_atom_mdp(
            FiniteSupportAtom(40.0, 0.5, -1),
            total_time=41.0,
        )
        kink = 0.25
        cases.append(
            (
                "kink",
                kink_mdp,
                math.nextafter(kink, -math.inf),
                math.nextafter(kink, math.inf),
            )
        )
        flat_mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.0, 1),
            delta_learning_per_unit_tutoring=1e-13,
        )
        center = 0.5
        lower = center - 2.0 ** -48
        upper = center + 2.0 ** -48
        cases.append(("flat", flat_mdp, lower, upper))

        for name, mdp, lower, upper in cases:
            with self.subTest(name=name):
                belief = mdp.initial_belief()
                objective = _BObjective(mdp, belief, 10_000)
                node = _make_b_node(objective, lower, upper, 50)
                allocation = lower
                checked = 0
                while allocation <= upper and checked < 256:
                    self.assertGreaterEqual(
                        node.upper_bound,
                        mdp.expected_terminal_utility(belief, allocation),
                    )
                    checked += 1
                    next_allocation = math.nextafter(allocation, math.inf)
                    if next_allocation == allocation:
                        break
                    allocation = next_allocation
                self.assertGreater(checked, 1)

    def test_b_specific_one_ulp_tie_and_dominance_thresholds(self):
        tau_low, tau_high = _tau_bounds((1.0, 1.0))
        tied_difference = math.nextafter(tau_low, 0.0)
        tie_boundary = tau_low
        base = self.candidate((0.1, 0.2), (0.0, 0.0), 0.1, 0.0)

        tied = _BSnapshot(
            (1.0, 1.0),
            (
                base,
                self.candidate(
                    (0.8, 0.9),
                    (tied_difference, tied_difference),
                    0.8,
                    tied_difference,
                ),
            ),
        )
        provisional = _BSnapshot(
            (1.0, 1.0),
            (
                base,
                self.candidate(
                    (0.8, 0.9),
                    (tie_boundary, tie_boundary),
                    0.8,
                    tie_boundary,
                ),
            ),
        )
        asymmetric_mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.0, 1),
            delta_learning_per_unit_tutoring=1e-8,
        )
        symmetry = asymmetric_mdp.initial_belief()
        proof = reference_b_module.prove_recipient_swap_symmetry(
            asymmetric_mdp,
            symmetry,
        )

        self.assertEqual(
            _resolve_b_candidates(tied, proof)[0],
            "certified_value_tie",
        )
        self.assertIsNone(_resolve_b_candidates(provisional, proof)[0])

        directed_boundary = _add_up(0.0, tau_high)
        other = self.candidate((0.8, 0.9), (0.0, 0.0), 0.8, 0.0)
        at_boundary = _BSnapshot(
            (1.0, 1.0),
            (
                self.candidate(
                    (0.1, 0.2),
                    (directed_boundary, directed_boundary),
                    0.1,
                    directed_boundary,
                ),
                other,
            ),
        )
        above = math.nextafter(directed_boundary, math.inf)
        above_boundary = _BSnapshot(
            (1.0, 1.0),
            (
                self.candidate((0.1, 0.2), (above, above), 0.1, above),
                other,
            ),
        )
        self.assertIsNone(_resolve_b_candidates(at_boundary, proof)[0])
        self.assertEqual(_resolve_b_candidates(above_boundary, proof)[0], "unique")

    def test_numerical_method_hash_binds_every_work_and_order_control(self):
        baseline = terminal_reference_b_numerical_method_config_hash()
        changes = {
            "_ROOT_BRACKET_LIMIT": 1,
            "REFERENCE_B_HEAP_ORDER": REFERENCE_B_HEAP_ORDER + "_changed",
            "REFERENCE_B_CHILD_ORDER": REFERENCE_B_CHILD_ORDER + "_changed",
            "REFERENCE_B_WITNESS_ORDER": REFERENCE_B_WITNESS_ORDER + ("changed",),
            "REFERENCE_B_CANDIDATE_ORDER": REFERENCE_B_CANDIDATE_ORDER + "_changed",
            "REFERENCE_B_CANONICAL_ORDER": REFERENCE_B_CANONICAL_ORDER + "_changed",
        }
        for name, changed in changes.items():
            with self.subTest(name=name), patch.object(reference_b_module, name, changed):
                self.assertNotEqual(
                    terminal_reference_b_numerical_method_config_hash(),
                    baseline,
                )
        self.assertNotEqual(
            terminal_reference_b_numerical_method_config_hash(499_999),
            baseline,
        )

    def test_reference_b_dependency_allowlist_and_forbidden_call_scan(self):
        self.assertEqual(
            _reference_b_dependency_violations(
                inspect.getsource(reference_b_module)
            ),
            (),
        )

    def test_dependency_scan_rejects_aliased_reference_and_production_imports(self):
        forbidden_snippets = {
            "reference_a_module_alias": (
                "import src.solvers.terminal_reference as alias"
            ),
            "production_module_alias": "import src.solvers.terminal as production",
            "reference_a_helper_alias": (
                "from src.solvers.terminal_reference "
                "import _run_reference_a_level as hidden"
            ),
            "dynamic_reference_a_import": (
                "alias = __import__('src.solvers.terminal_reference')"
            ),
        }
        for name, snippet in forbidden_snippets.items():
            with self.subTest(name=name):
                self.assertTrue(
                    _reference_b_dependency_violations(snippet),
                    f"dependency scan accepted forbidden snippet: {snippet}",
                )

    def test_self_rehashed_forged_b_record_is_rejected_by_source_recompute(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, _, reference = self.solve(mdp)
        evidence = reference.candidate_isolation_evidence[0]
        shifted_interval = (0.0, 1e-5)
        forged = self.rehash(
            replace(
                reference,
                candidate_allocation_intervals=(shifted_interval,),
                candidate_isolation_evidence=(
                    replace(
                        evidence,
                        allocation_interval=shifted_interval,
                        isolation_rule=REFERENCE_B_BRANCH_RULE,
                        witness_allocation=0.0,
                    ),
                ),
                canonical_allocation_interval=shifted_interval,
                representative_allocation=0.0,
            )
        )

        self.assertFalse(self.validate(mdp, belief, forged))

    def test_mdp_belief_and_numerical_identity_mismatches_are_rejected(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief, _, reference = self.solve(mdp)
        altered_mdp = one_atom_mdp(
            FiniteSupportAtom(80.0, 0.5, -1),
            total_time=41.0,
        )
        altered_belief = belief.copy()
        altered_belief.history.append({"action": 1.0})

        self.assertFalse(
            validate_terminal_reference_b_record(
                reference,
                altered_mdp,
                altered_mdp.initial_belief(),
                scientific_spec_hash=terminal_scientific_spec_hash(altered_mdp),
                numerical_method_config_hash=reference.numerical_method_config_hash,
            )
        )
        self.assertFalse(
            validate_terminal_reference_b_record(
                reference,
                mdp,
                altered_belief,
                scientific_spec_hash=terminal_scientific_spec_hash(mdp),
                numerical_method_config_hash=reference.numerical_method_config_hash,
            )
        )
        self.assertFalse(
            validate_terminal_reference_b_record(
                reference,
                mdp,
                belief,
                scientific_spec_hash=terminal_scientific_spec_hash(mdp),
                numerical_method_config_hash="f" * 64,
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
