"""Deterministic source-context/corner parity tests, not error calibration."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
import warnings
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from design_validation.bounded_task import project_share
from design_validation.c1_identifiability import audit_candidates
from tutoring_analysis.empirical_bernstein import (
    ATOL, RTOL, attest_initial_rows, fit_variance_adaptive,
    initial_projection_values, source_slope_domain, _schedule_reason,
)
from tutoring_analysis.identifiability_robustness import (
    CELL_IDS, cell_truths, coefficient_table, projection_truth, schedule_rows, validate_config,
    generate_batch, core_statistics, scalar_parity, prepare_context, vector_contrast,
)
from tutoring_analysis.inference import CONDITIONS, CORE, POLICIES, participant_slope

def deterministic_definitions(cid):
    """A manually declared legal multiset/order; no schedule or observation RNG."""
    if cid in ("C2", "C4"):
        ids = [f"{cid}_s{s}_g{g}" for g in range(3) for s in range(3)] + [f"{cid}_s0_g0"]
    else:
        ids = [f"{cid}_s{s}_g0" for s in [0, 1, 2]*3+[0]]
    return [{"templateId": template, "trialId": f"main-{i:02d}", "stageIndex": i,
             "scheduleIndex": i+3, "sideReversed": bool(i % 2 == 0)}
            for i, template in enumerate(ids, 1)]


def deterministic_metric_rows(design, people=2):
    """Zero-question finite initial geometry, with shared projection function."""
    out = []
    for cid in CONDITIONS:
        for p in range(people):
            rows = schedule_rows(deterministic_definitions(cid), design, cid, f"deterministic-{cid}-{p}")
            for i, row in enumerate(rows):
                share = [0., .25, .5, .75, 1.][(i+p) % 5]
                a, b, budget = row["meanAInitial"], row["meanBInitial"], row["initialBudget"]
                eo = project_share(.5+(a-b)/(2*budget))
                d0, d1 = 100*abs(share-.5), 100*abs(share-eo)
                tied = abs(a-b) <= 1e-9*max(1., a, b)
                row.update(dataRole="synthetic_design", exclusionReasons="", shareA=share,
                    questions=0, anyQuestion=0, remainingBudget=budget, meanATerminal=a, meanBTerminal=b,
                    target_equal_split=.5, target_equal_outcome=eo, distancePP_equal_split=d0,
                    distancePP_equal_outcome=d1, EOResemblance=d0-d1, terminalTie=tied,
                    lowerNeedDirectional=0. if tied else share-.5 if a < b else .5-share,
                    xTerminal=row["xInitial"])
                out.append(row)
    return out


class SourceContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads((ROOT/"configs/tutoring_conditions.json").read_text())
        cls.domain = source_slope_domain(cls.design)
        cls.rows = deterministic_metric_rows(cls.design)
        cls.config = json.loads((ROOT/"configs/tutoring_identifiability_robustness_v01.json").read_text())

    def test_source_global_safe_bound_is_not_observed_extrema(self):
        domain = self.domain
        canonical = min(abs(row["xInitial"]) for row in domain["rows"])
        safe = min(abs(row["xInitial"])-ATOL-RTOL*abs(row["xInitial"]) for row in domain["rows"])
        self.assertEqual(domain["sourceTemplateOrientationCount"], 42)
        self.assertEqual(domain["canonicalMinimumMagnitude"], canonical)
        self.assertAlmostEqual(domain["acceptedInputSafeMagnitude"], safe)
        self.assertGreater(safe, 0.)
        self.assertLess(safe, canonical)
        self.assertAlmostEqual(domain["SxxLowerBound"], 10*safe**2)
        self.assertAlmostEqual(domain["idealSlopeAbsoluteBound"], 1/(2*safe))
        self.assertEqual(domain["numericSlopeAbsoluteBound"], math.ceil(1/(2*safe))+1)
        self.assertFalse(domain["machineCertified"])
        self.assertFalse(domain["rawSolverModified"])

    def test_legal_rows_fit_all_nine_fixed_slots_all_policies(self):
        for policy in POLICIES:
            fit = fit_variance_adaptive(self.rows, self.design, "synthetic_design", "synthetic_validation", policy=policy)
            self.assertFalse(fit["primaryReplacement"])
            tests = fit["correction"]["tests"]
            self.assertEqual([t["id"] for t in tests], [entry[0] for entry in CORE])
            self.assertEqual(len(tests), 9)
            self.assertEqual(fit["correction"]["retainedUnavailableSlots"], 0)
            for test in tests:
                self.assertEqual(test["status"], "available")
            self.assertEqual(fit["correction"]["globalAnyGuarantee"], policy != "within_family_holm")

    def test_full_endpoint_projection_is_unchanged_raw_solver(self):
        rows = attest_initial_rows(self.rows, self.domain)
        estimates = initial_projection_values(rows, self.domain)
        for person in estimates:
            own = [r for r in rows if r["participantId"] == person["participantId"]]
            if person["conditionId"] in ("C1", "C5"):
                self.assertIsNone(person["value"])
                self.assertIn("structurally_constant", person["reason"])
            else:
                old = participant_slope(own, "xInitial")
                self.assertEqual(person["value"], old["value"])
                self.assertEqual(person["observations"], 10)
                self.assertTrue(person["eligibilityChangedFromPrimary"])

    def test_missing_endpoint_retains_mean_but_makes_projection_unavailable(self):
        rows = copy.deepcopy(self.rows)
        first = next(r for r in rows if r["conditionId"] == "C2" and r["stageIndex"] == 1)
        for key in ("shareA", "EOResemblance", "lowerNeedDirectional", "distancePP_equal_split", "distancePP_equal_outcome"):
            first[key] = None
        fit = fit_variance_adaptive(rows, self.design, "synthetic_design", "synthetic_validation")
        slope = next(p for p in fit["participantEstimates"] if p["participantId"] == first["participantId"] and p["testId"] == "initial_C2")
        mean = next(p for p in fit["participantEstimates"] if p["participantId"] == first["participantId"] and p["testId"] == "resemblance_C2")
        self.assertEqual(slope["status"], "unavailable")
        self.assertEqual(slope["observations"], 9)
        self.assertTrue(slope["endpointMissingWithinCompleteHistory"])
        self.assertEqual(mean["status"], "available"); self.assertEqual(mean["observations"], 9)
        test = next(t for t in fit["correction"]["tests"] if t["id"] == "initial_C2")
        self.assertIsNone(test["pValue"]); self.assertIsNone(test["adjustedP"])
        self.assertFalse(test["reject"])

    def test_missing_record_not_reinterpreted_as_full_endpoint_target(self):
        p = "deterministic-C2-0"
        rows = [r for r in self.rows if r["participantId"] != p or r["stageIndex"] != 10]
        fit = fit_variance_adaptive(rows, self.design, "synthetic_design", "synthetic_validation")
        for person in fit["participantEstimates"]:
            if person["participantId"] == p:
                self.assertEqual(person["status"], "unavailable")
                self.assertFalse(person["completeTen"])

    def test_corrupt_source_metadata_duplicates_and_provenance_raise(self):
        changes = (("templateId", "practice_C2"), ("xInitial", 0.), ("sideReversed", True),
                   ("meanAInitial", 100.), ("scale", 1.01), ("initialBudget", 1.),
                   ("scheduleIndex", 9), ("initialTie", True), ("target_equal_outcome", .123),
                   ("dataRole", "software_check"), ("stageIndex", True))
        for key, value in changes:
            rows = copy.deepcopy(self.rows)
            first = next(r for r in rows if r["conditionId"] == "C2" and r["stageIndex"] == 1)
            first[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit_variance_adaptive(rows, self.design, "synthetic_design", "synthetic_validation")
        with self.assertRaises(ValueError):
            fit_variance_adaptive(self.rows+self.rows[:1], self.design, "synthetic_design", "synthetic_validation")
        with self.assertRaises(ValueError):
            fit_variance_adaptive(self.rows, self.design, "software_check", "synthetic_validation")

    def test_multiset_balance_order_and_external_variant_guards(self):
        own = attest_initial_rows([r for r in self.rows if r["participantId"] == "deterministic-C2-0"], self.domain)
        self.assertIsNone(_schedule_reason(own))
        unbalanced = copy.deepcopy(own); unbalanced[0]["sideReversed"] = True
        self.assertIn("balanced", _schedule_reason(unbalanced))
        multiset = copy.deepcopy(own); multiset[-1]["templateId"] = "C2_s1_g0"
        for row in multiset: row["scheduleVariant"] = 0
        self.assertIn("variant", _schedule_reason(multiset))
        wrong = copy.deepcopy(own); wrong[-1]["templateId"] = "C2_s0_g1"
        wrong[-2]["templateId"] = "C2_s0_g1"
        self.assertIn("multiset", _schedule_reason(wrong))
        consecutive = copy.deepcopy(own)
        for row in consecutive[:4]: row["scale"] = .95
        self.assertIn("four_consecutive", _schedule_reason(consecutive))
        bad_variant = copy.deepcopy(own); bad_variant[0]["scheduleVariant"] = True
        with self.assertRaises(ValueError): attest_initial_rows(bad_variant, self.domain)

    def test_partial_terminal_adjusted_contexts_unavailable(self):
        own = attest_initial_rows(self.rows, self.domain)
        for kwargs in ({"predictor": "xTerminal"}, {"model": "order_only"}, {"context": "mean_only_unsupported_slopes"}):
            estimates = initial_projection_values(own, self.domain, **kwargs)
            self.assertTrue(all(p["status"] == "unavailable" and p["value"] is None for p in estimates))
        fit = fit_variance_adaptive(self.rows, self.design, "synthetic_design", "synthetic_validation", context="mean_only_unsupported_slopes")
        self.assertEqual(fit["correction"]["retainedUnavailableSlots"], 3)

    def test_supported_out_of_domain_raw_coefficient_hard_halts(self):
        own = attest_initial_rows([r for r in self.rows if r["participantId"] == "deterministic-C2-0"], self.domain)
        with patch("tutoring_analysis.empirical_bernstein.participant_slope", return_value={
                "status": "available", "reason": None, "value": self.domain["numericBounds"][1]+1.}):
            with self.assertRaisesRegex(ValueError, "HALT"):
                initial_projection_values(own, self.domain)

    def test_literal_config_seal_and_c1_prespecified_batch_guard(self):
        cells = validate_config(self.config)
        self.assertEqual([c["id"] for c in cells], list(CELL_IDS))
        self.assertEqual([c["seed"] for c in cells], list(range(720071, 720083)))
        self.assertEqual(sum(self.config["replicationsPerCell"] for _ in cells), 120000)
        for mutation in (("alpha", .1), ("unrecognizedIgnoredField", True)):
            bad = copy.deepcopy(self.config); bad[mutation[0]] = mutation[1]
            with self.assertRaisesRegex(ValueError, "Frozen literal config object changed"):
                validate_config(bad)
        for key, value in (("candidateDeltas", [6, 8, 12, 16]), ("orientations", [False]), ("scales", [1])):
            spec = copy.deepcopy(self.config["c1"]); spec[key] = value
            with patch("design_validation.c1_identifiability.BoundedTask", side_effect=AssertionError("must stop before evaluation")):
                with self.assertRaises(ValueError): audit_candidates(self.design, spec)

    def test_binary_corner_raw_solver_parity_and_analytic_truth(self):
        tables = {}
        for cid in ("C2", "C3", "C4"):
            rows = attest_initial_rows([r for r in self.rows if r["participantId"] == f"deterministic-{cid}-0"], self.domain)
            table = coefficient_table(rows, self.domain); tables[cid] = table
            self.assertEqual(table["patterns"], 1024)
            self.assertLess(table["maximumRawAlgebraDifference"], 1e-10)
            self.assertFalse(table["oldSolverModified"])
            self.assertGreaterEqual(table["Sxx"], self.domain["SxxLowerBound"])
            self.assertGreaterEqual(table["rawCoefficientRange"][0], self.domain["numericBounds"][0])
            self.assertLessEqual(table["rawCoefficientRange"][1], self.domain["numericBounds"][1])
            ordinary = projection_truth(table, .2, .5, 0.)
            selected = projection_truth(table, .2, .5, 0., selected=True)
            self.assertAlmostEqual(ordinary["ideal"], .2)
            self.assertAlmostEqual(selected["ideal"], .35)
            self.assertLess(abs(ordinary["difference"]), 1e-10)
            self.assertFalse(ordinary["rawTruthSnapped"])
            constant = projection_truth(table, 0., .5, 0., constant=True)
            self.assertEqual(constant["raw"], table["zeroPatternCoefficient"])
            with self.assertRaises(ValueError): projection_truth(table, 100., .5, 0.)
        null_cell = validate_config(self.config)[0]
        truth = cell_truths(null_cell, tables)
        core = truth["selectedCore"]
        self.assertEqual(len(core["rawTruth"]), 9)
        self.assertEqual(set(core["idealNullRoster"]), {r[0] for r in CORE})
        expected_raw_nulls = {CORE[i][0] for i, x in enumerate(core["rawSignedTruth"]) if x <= 0}
        self.assertEqual(set(core["rawNullRoster"]), expected_raw_nulls)

    def test_vector_constant_tiny_width_matches_scalar_supported_inverse(self):
        from tutoring_analysis.empirical_bernstein import empirical_bernstein_contrast
        numbers = np.full((2, 2), 1e-300)
        groups = {"C2": {"Q": {"values": numbers, "available": np.ones((2, 2), bool)}}}
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = vector_contrast(groups, "Q", {"C2": 1.}, {"C2": (0., 2e-300)}, "greater")
        scalar = empirical_bernstein_contrast([
            {"participantId": str(i), "conditionId": "C2", "status": "available", "value": 1e-300}
            for i in range(2)], {"C2": 1.}, {"C2": (0., 2e-300)})
        self.assertTrue(np.all(result["available"]))
        np.testing.assert_allclose(result["p"], scalar["pValue"], rtol=1e-10)

    def test_vector_zero_one_two_people_unavailable_and_domain_masks(self):
        from tutoring_analysis.empirical_bernstein import empirical_bernstein_contrast
        values = np.asarray([[np.nan]*3, [.4, np.nan, np.nan], [.2, .4, np.nan], [.2, .4, .6]])
        available = np.isfinite(values)
        groups = {"C2": {"Q": {"values": values, "available": available}}}
        result = vector_contrast(groups, "Q", {"C2": 1.}, {"C2": (0., 1.)}, "greater")
        np.testing.assert_array_equal(result["available"], [False, False, True, True])
        np.testing.assert_array_equal(result["counts"]["C2"], [0, 1, 2, 3])
        self.assertTrue(np.isnan(result["p"][:2]).all())
        self.assertTrue(np.isnan(result["low"][:2]).all())
        self.assertTrue(np.isnan(result["estimate"][0]))
        self.assertEqual(result["estimate"][1], .4)
        for index, numbers in enumerate(values):
            person = [{"participantId": str(i), "conditionId": "C2", "value": float(v), "status": "available"}
                      for i, v in enumerate(numbers) if np.isfinite(v)]
            scalar = empirical_bernstein_contrast(person, {"C2": 1.}, {"C2": (0., 1.)})
            if scalar["status"] == "available":
                self.assertAlmostEqual(result["p"][index], scalar["pValue"])
                np.testing.assert_allclose([result["low"][index], result["high"][index]], scalar["interval"])
        for changed in (np.asarray([[1.1, .5]]), np.asarray([[np.inf, .5]])):
            bad = {"C2": {"Q": {"values": changed, "available": np.ones((1, 2), bool)}}}
            with self.assertRaises(ValueError): vector_contrast(bad, "Q", {"C2": 1.}, {"C2": (0., 1.)}, "greater")
        bad = {"C2": {"Q": {"values": values, "available": np.ones_like(available)}}}
        with self.assertRaises(ValueError): vector_contrast(bad, "Q", {"C2": 1.}, {"C2": (0., 1.)}, "greater")
        with self.assertRaises(ValueError): vector_contrast(groups, "Q", {"C2": 1.}, {"C2": (0., 1.)}, "greater", "implicit_fallback")

    def test_tiny_fixed_seed_functional_generator_and_scalar_parity(self):
        # Gate permits these separately labelled six tiny functional replicas.
        # Frozen seeds with different RNG call shapes are not production prefixes;
        # this is functionality/reproduction, not new calibration evidence.
        context = prepare_context(self.config, self.design)
        for index in (0, 10):
            cell = context["cells"][index]
            batch = generate_batch(cell, 0, context, batch_size=3)
            repeated = generate_batch(cell, 0, context, batch_size=3)
            self.assertEqual(batch["attempts"], 3)
            for cid in CONDITIONS:
                for endpoint in batch["groups"][cid]:
                    np.testing.assert_array_equal(batch["groups"][cid][endpoint]["values"],
                                                  repeated["groups"][cid][endpoint]["values"])
            statistics = core_statistics(batch, context)
            for policy in POLICIES:
                for replica in range(3):
                    receipt = scalar_parity(cell, batch, statistics, context, policy, replica, self.config)
                    self.assertEqual(receipt["status"], "PASS")
                    self.assertEqual(receipt["fixedNineRoutes"], 9)
                    self.assertTrue(receipt["matchedSixChecked"])
            if cell["selection"]:
                for cid in CONDITIONS:
                    raw = batch["raw"][cid]
                    self.assertTrue(np.all(raw["slopeU"][raw["includedPeople"]] > .75))
        for bad_index, bad_size in ((True, 3), (-1, 3), (0, True), (0, 0)):
            with self.assertRaises(ValueError): generate_batch(context["cells"][0], bad_index, context, bad_size)


if __name__ == "__main__": unittest.main()
