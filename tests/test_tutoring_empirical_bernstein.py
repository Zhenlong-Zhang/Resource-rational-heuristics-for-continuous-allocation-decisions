"""Deterministic MP11/cohort interface checks, not calibration or human evidence."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutoring_analysis.empirical_bernstein import (
    METHOD, cohort_identification, empirical_bernstein_contrast, inverse_tail,
)


def person_values(condition, numbers):
    return [{"participantId": f"formula-{condition}-{i}", "conditionId": condition,
             "status": "available", "value": value} for i, value in enumerate(numbers)]


class EmpiricalBernsteinTests(unittest.TestCase):
    def test_weighted_unbiased_variance_and_two_sided_budget(self):
        values = person_values("C2", [.2, .4, .6]) + person_values("C3", [.7, .8])
        weights = {"C3": 1., "C2": -1.}
        bounds = {"C3": (0., 1.), "C2": (0., 1.)}
        result = empirical_bernstein_contrast(values, weights, bounds)
        A = math.sqrt(2*.005/2) + math.sqrt(2*.04/3)
        B = 7/3 + 7/6
        z = math.log(8/.05)
        self.assertEqual(result["method"], METHOD)
        self.assertAlmostEqual(result["estimate"], .35)
        self.assertAlmostEqual(result["groups"]["C2"]["variance"], .04)
        self.assertAlmostEqual(result["groups"]["C3"]["variance"], .005)
        self.assertAlmostEqual(result["varianceComponentA"], A)
        self.assertAlmostEqual(result["rangeComponentB"], B)
        self.assertAlmostEqual(result["pointwiseRadius"], A*math.sqrt(z)+B*z)
        self.assertEqual(result["legalContrastBounds"], [-1., 1.])
        self.assertEqual(result["interval"], [-1., 1.])
        self.assertFalse(result["intervalSimultaneous"])
        self.assertFalse(result["knownBoundsEstimatedFromSample"])
        reflected = empirical_bernstein_contrast(values, {"C3": -1., "C2": 1.}, bounds, "less")
        self.assertEqual(reflected["pValue"], result["pValue"])
        self.assertAlmostEqual(reflected["pointwiseRadius"], result["pointwiseRadius"])

    def test_inverse_solves_declared_monotone_radius(self):
        for A, B, k, z in ((.12, .013, 1, 8.), (0., .2, 2, 6.), (3., .001, 4, 10.)):
            deviation = A*math.sqrt(z)+B*z
            result = inverse_tail(deviation, 0., "greater", A, B, k)
            self.assertAlmostEqual(result["inverseLogRadius"], z)
            self.assertAlmostEqual(result["logPValue"], min(0., math.log(2*k)-z))
            self.assertAlmostEqual(result["pValue"], math.exp(result["logPValue"]))
            less = inverse_tail(-deviation, 0., "less", A, B, k)
            self.assertEqual(less, result)

    def test_zero_wrong_sign_and_capped_inverse(self):
        for estimate in (0., -.5):
            result = inverse_tail(estimate, 0., "greater", .1, .2, 1)
            self.assertEqual(result["pValue"], 1.)
            self.assertEqual(result["logPValue"], 0.)
            self.assertFalse(result["pValueUnderflow"])
        capped = inverse_tail(.0001, 0., "greater", 0., 1., 1)
        self.assertEqual(capped["pValue"], 1.)

    def test_missing_zero_one_counts_public_unavailable_no_fallback(self):
        for values in ([], person_values("C2", [.8]),
                       person_values("C2", [.8, .9])+person_values("C3", [.1])):
            weights = {"C2": 1.} if len(values) < 2 else {"C2": 1., "C3": -1.}
            result = empirical_bernstein_contrast(values, weights, {c: (0., 1.) for c in weights})
            self.assertEqual(result["status"], "unavailable")
            self.assertIn("fewer_than_two", result["reason"])
            for key in ("pValue", "logPValue", "interval", "pointwiseRadius"):
                self.assertIsNone(result[key])
        values = person_values("C2", [.8, .9])
        values.append({"participantId": "missing-C2", "conditionId": "C2", "status": "unavailable", "value": None})
        result = empirical_bernstein_contrast(values, {"C2": 1.}, {"C2": (0., 1.)})
        self.assertEqual(result["groups"]["C2"]["participants"], 2)

    def test_constants_keep_known_range_penalty_and_n_two(self):
        result = empirical_bernstein_contrast(person_values("C2", [.8, .8]), {"C2": 1.}, {"C2": (0., 1.)})
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["groups"]["C2"]["variance"], 0.)
        self.assertEqual(result["varianceComponentA"], 0.)
        self.assertAlmostEqual(result["rangeComponentB"], 7/3)
        self.assertGreater(result["pointwiseRadius"], 0.)
        self.assertNotEqual(result["interval"], [.8, .8])

    def test_underflow_retains_finite_log_and_tiny_known_width(self):
        result = empirical_bernstein_contrast(person_values("C2", [1.]*5000), {"C2": 1.}, {"C2": (0., 1.)})
        self.assertEqual(result["pValue"], 0.)
        self.assertTrue(result["pValueUnderflow"])
        self.assertTrue(math.isfinite(result["logPValue"]))
        tiny = empirical_bernstein_contrast(person_values("C2", [1e-300]*2), {"C2": 1.}, {"C2": (0., 2e-300)})
        self.assertEqual(tiny["status"], "available")
        self.assertGreater(tiny["rangeComponentB"], 0.)
        self.assertTrue(math.isfinite(tiny["inverseLogRadius"]))

    def test_parameters_reject_bool_nonfinite_enum_and_invalid_domains(self):
        own = person_values("C2", [.2, .4]); w = {"C2": 1.}; bounds = {"C2": (0., 1.)}
        for key, value in (("null", True), ("null", np.bool_(False)), ("null", "0"),
                           ("null", math.nan), ("confidence", True), ("confidence", 1.),
                           ("confidence", 0.), ("side", "two_sided")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                empirical_bernstein_contrast(own, w, bounds, **{key: value})
        for value in (True, np.bool_(True), "1", 0., math.inf):
            with self.assertRaises(ValueError): empirical_bernstein_contrast(own, {"C2": value}, bounds)
        for domain in ((0., 0.), (1., 0.), (False, 1.), ("0", 1.), (0., math.inf), None, "01"):
            with self.assertRaises(ValueError): empirical_bernstein_contrast(own, w, {"C2": domain})
        with self.assertRaises(ValueError): empirical_bernstein_contrast(own, w, {"C3": (0., 1.)})
        with self.assertRaises(ValueError): empirical_bernstein_contrast(own, {"C6": 1.}, {"C6": (0., 1.)})

    def test_outcome_identity_and_domain_errors_halt(self):
        own = person_values("C2", [.2, .4])
        bad_rows = (own+own[:1], [{**own[0], "participantId": " leading"}],
                    [{**own[0], "conditionId": "C6"}], [{**own[0], "value": True}],
                    [{**own[0], "value": math.nan}], [{**own[0], "status": "imputed"}],
                    [{**own[0], "status": "unavailable", "value": .2}], [{**own[0], "value": 1.01}])
        for rows in bad_rows:
            with self.assertRaises(ValueError): empirical_bernstein_contrast(rows, {"C2": 1.}, {"C2": (0., 1.)})

    def test_inverse_rejects_invalid_components(self):
        good = dict(estimate=.3, null=0., side="greater", A=.1, B=.01, arms=2)
        for name, value in (("estimate", True), ("null", math.inf), ("A", -.1),
                            ("B", 0.), ("arms", True), ("arms", 1.5), ("side", "auto")):
            with self.assertRaises(ValueError): inverse_tail(**{**good, name: value})


class CohortIdentificationTests(unittest.TestCase):
    target = "finite_enrolled_cohort_defined_full_potential_endpoint"

    def test_signed_weights_and_wholly_unobserved_people(self):
        observed = [{"participantId": "a", "conditionId": "C2", "value": .2},
                    {"participantId": "b", "conditionId": "C3", "value": .8}]
        result = cohort_identification(observed, {"C2": 2, "C3": 2},
            {"C2": (0., 1.), "C3": (0., 1.)}, {"C2": -1., "C3": 1.}, self.target)
        np.testing.assert_allclose(result["interval"], [-.2, .8], atol=1e-15)
        self.assertFalse(result["confidenceInterval"])
        self.assertFalse(result["superpopulationRepair"])
        self.assertTrue(result["zeroCompatible"])
        self.assertEqual(result["arms"]["C2"]["whollyUnobservedPeople"], 1)

    def test_partial_trial_means_are_bounded_before_person_aggregation(self):
        observed = [{"participantId": "partial", "conditionId": "C2", "value": .8,
                     "observedTrialCount": 5, "endpointKind": "mean"}]
        result = cohort_identification(observed, {"C2": 2}, {"C2": (0., 1.)}, {"C2": 1.}, self.target)
        np.testing.assert_allclose(result["interval"], [.2, .95], atol=1e-15)
        arm = result["arms"]["C2"]
        self.assertEqual(arm["fullyObservedPeople"], 0)
        self.assertEqual(arm["partiallyObservedPeople"], 1)
        self.assertEqual(arm["observedTrialEndpointCount"], 5)
        self.assertEqual(arm["potentialTrialEndpointCount"], 20)

    def test_zero_observations_whole_range_and_zero_enrollment_unavailable(self):
        result = cohort_identification([], {"C2": 2}, {"C2": (-3., 3.)}, {"C2": -2.}, self.target)
        self.assertEqual(result["interval"], [-6., 6.])
        result = cohort_identification([], {"C2": 0}, {"C2": (0., 1.)}, {"C2": 1.}, self.target)
        self.assertEqual(result["status"], "unavailable"); self.assertIsNone(result["interval"])
        empty = [{"participantId": "zero", "conditionId": "C2", "value": None, "observedTrialCount": 0}]
        result = cohort_identification(empty, {"C2": 1}, {"C2": (0., 1.)}, {"C2": 1.}, self.target)
        self.assertEqual(result["interval"], [0., 1.])
        self.assertEqual(result["arms"]["C2"]["identifiedPeopleWithZeroObservedEndpoints"], 1)

    def test_invalid_enrollment_target_duplicate_and_partial_slope(self):
        good = [{"participantId": "one", "conditionId": "C2", "value": .4}]
        for N in (True, -1, 1.5, "1", 0):
            with self.assertRaises(ValueError): cohort_identification(good, {"C2": N}, {"C2": (0., 1.)}, {"C2": 1.}, self.target)
        with self.assertRaises(ValueError): cohort_identification(good, {"C2": 1}, {"C2": (0., 1.)}, {"C2": 1.}, "unknown_population")
        for observed in (good+good, [{**good[0], "value": True}], [{**good[0], "value": 1.1}],
                         [{**good[0], "observedTrialCount": True}], [{**good[0], "observedTrialCount": 11}],
                         [{**good[0], "observedTrialCount": 0}], [{**good[0], "endpointKind": "slope", "observedTrialCount": 9}]):
            with self.assertRaises(ValueError): cohort_identification(observed, {"C2": 2}, {"C2": (0., 1.)}, {"C2": 1.}, self.target)
        mixed = good+[{"participantId": "two", "conditionId": "C2", "value": .3, "endpointKind": "slope"}]
        with self.assertRaises(ValueError): cohort_identification(mixed, {"C2": 2}, {"C2": (0., 1.)}, {"C2": 1.}, self.target)


if __name__ == "__main__": unittest.main()
