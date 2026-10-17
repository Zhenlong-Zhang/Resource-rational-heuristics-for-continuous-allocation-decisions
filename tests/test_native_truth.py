"""Exact ideal laws retain native binary64 roundoff and diagnostic limitations."""
from fractions import Fraction as F
import math
import struct
import unittest

from src.tutoring_analysis import native_truth as T
from src.tutoring_analysis.expected_outcomes import digest


def projection_fixture():
    rows = [{"conditionId": "C2", "mainIndex": i + 1, "remainingIntegerBudget": 1000,
             "x0": float(i), "jEGPA": 500, "terminalMeans": [40., 60.]} for i in range(10)]
    centre, sxx = F(9, 2), F(165, 2)
    weights = [float((F(i) - centre) / sxx) for i in range(10)]
    payload = {"schema": "frozen-projection-plan-v1",
               "operationOrder": "ordered_float64_multiply_then_add_mainIndex_1_to_10",
               "predictorBinary64": [struct.pack(">d", row["x0"]).hex() for row in rows],
               "weightBinary64": [struct.pack(">d", w).hex() for w in weights]}
    plan = {**payload, "planHash": digest(payload)}
    states = {str(i): {"metrics": row} for i, row in enumerate(rows)}
    index = {"|".join(map(str, ("working", "C2", 0, i + 1, (0, 1, 3)[i % 3]))): str(i) for i in range(10)}
    lookup = {"states": states, "index": index,
              "geometry": {"working|C2": {"M": 9., "supportedVariants": [0], "projectionPlans": {"0": plan}}}}
    return lookup


class NativeTruthTests(unittest.TestCase):
    def test_decimal_law_is_not_binary64_input(self):
        self.assertEqual(T.decimal(.3), F(3, 10))
        self.assertNotEqual(T.binary(.3), F(3, 10))
        with self.assertRaises(ValueError):
            T.decimal(True)
        with self.assertRaises(ValueError):
            T.unpack("7ff0000000000000")

    def test_ordered_rounding_is_not_algebraic_cancellation(self):
        terms = [F(2 ** 54), F(1), F(-2 ** 54)]
        self.assertEqual(T.native_mean(terms), 0)
        self.assertEqual(sum(terms) / 3, F(1, 3))
        self.assertEqual(T.native_projection(terms, [1, 1, 1]), 0)

    def test_all_exchangeable_patterns_have_exact_unit_mass(self):
        for law, q, rho in (("beta", .3, 0), ("beta", .3, .6),
                             ("selected_uniform", None, None), ("centered_beta", None, None)):
            moments = T.probability_moments(law, q, rho)
            masses = [T.bernoulli_pattern_mass(k, moments) for k in range(11)]
            self.assertTrue(all(p >= 0 for p in masses))
            self.assertEqual(sum(math.comb(10, k) * p for k, p in enumerate(masses)), 1)
        self.assertEqual(T.slope_moments("centered_beta", F(2))[1], 0)
        self.assertEqual(T.slope_moments("selected_uniform", F(2))[1], F(3, 64))

    def test_native_Q_nonzero_contrast_is_not_snapped_to_null(self):
        cell = {"id": "B03", "layer": "B", "family": "Q", "subcase": "native", "q": .3, "rho": .3,
                "armOverrides": {"C3": {"rho": .1}, "C2": {"rho": .6}}}
        spec = {"id": "H01_Q_C3_minus_C2", "endpoint": "Q", "weights": {"C3": 1, "C2": -1}, "tail": "greater"}
        value, ideal, _, status, classification = T.derive_target(T.Oracle({}), cell, spec)
        self.assertEqual(ideal, 0)
        self.assertNotEqual(value, 0)
        self.assertEqual(classification, "exact_native_nonzero_not_boundary_null")
        self.assertEqual(status, "exact_rational_native_endpoint_under_declared_ideal_probability_law")

    def test_frozen_plan_is_bound_and_checked_independently(self):
        lookup = projection_fixture()
        self.assertEqual(len(T.Oracle(lookup).plan("working", "C2", 0)), 10)
        plan = lookup["geometry"]["working|C2"]["projectionPlans"]["0"]
        plan["weightBinary64"][0] = struct.pack(">d", 5.).hex()
        plan["planHash"] = digest({k: v for k, v in plan.items() if k != "planHash"})
        with self.assertRaisesRegex(ValueError, "centered-OLS"):
            T.Oracle(lookup).plan("working", "C2", 0)

    def test_C04_diagnostic_remains_underived(self):
        result = T.derive_target(T.Oracle({}), {"id": "C04", "layer": "C", "subcase": "Q"},
                                 {"endpoint": "Q", "weights": {"C2": 1}})
        self.assertEqual(result, (None, None, {}, "diagnostic_not_derived", "not_derived_diagnostic_only"))

    def test_abstract_reference_preserves_exact_declared_decimal(self):
        value, ideal, _, _, classification = T.derive_target(T.Oracle({}),
            {"id": "A05", "layer": "A", "subcase": "correct_direction"},
            {"id": "H05_initial_C4", "endpoint": "initial_slope", "tail": "less"})
        self.assertEqual(value, F(-3, 10))
        self.assertEqual(value, ideal)
        self.assertEqual(classification, "correct_direction_alternative")


if __name__ == "__main__":
    unittest.main()
