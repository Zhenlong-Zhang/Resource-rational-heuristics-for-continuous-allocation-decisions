"""Portable known-bound/finite-geometry tests, not stochastic/human calibration."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutoring_analysis.bounded_inference import (BOUNDS, GEOMETRY, PERSON_R_ENVELOPE,
    TRIAL_R_ENVELOPE, bounded_contrast, fit_bounded, source_fixture_rows, validate_geometry)
from tutoring_analysis.inference import CORE, POLICIES, participant_values


def values(cid, data):
    return [{"participantId": f"direct-{cid}-{i}", "conditionId": cid, "status": "available", "value": v}
            for i, v in enumerate(data)]


class BoundedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rows = source_fixture_rows()

    def test_weighted_formula_and_fixed_less_side(self):
        own = values("C2", [.2, .4, .6])+values("C3", [.7, .8])
        r = bounded_contrast(own, {"C3": 1., "C2": -1.}, {"C3": (0., 1.), "C2": (0., 1.)})
        V = 1/2+1/3; estimate = .75-.4
        self.assertAlmostEqual(r["estimate"], estimate); self.assertAlmostEqual(r["rangeProxyV"], V)
        self.assertAlmostEqual(r["pValue"], math.exp(-2*estimate**2/V))
        self.assertAlmostEqual(r["pointwiseRadius"], math.sqrt(V*math.log(40)/2))
        self.assertEqual(r["legalContrastBounds"], [-1., 1.])
        less = bounded_contrast(own, {"C3": -1., "C2": 1.}, {"C3": (0., 1.), "C2": (0., 1.)}, "less")
        self.assertAlmostEqual(less["pValue"], r["pValue"])

    def test_n_one_constants_and_wrong_sign(self):
        r = bounded_contrast(values("C2", [50.]), {"C2": 1.}, {"C2": (-50., 50.)})
        self.assertEqual(r["status"], "available"); self.assertAlmostEqual(r["pValue"], math.exp(-.5))
        self.assertGreater(r["pointwiseRadius"], 0)
        r = bounded_contrast(values("C2", [.3]*3), {"C2": 1.}, {"C2": (0., 1.)})
        self.assertEqual(r["status"], "available"); self.assertEqual(r["groups"]["C2"]["observedVarianceDiagnostic"], 0)
        self.assertGreater(r["pointwiseRadius"], 0)
        for v in (0., -.5):
            r = bounded_contrast(values("C2", [v]), {"C2": 1.}, {"C2": (-1., 1.)})
            self.assertEqual(r["pValue"], 1.); self.assertEqual(r["logPValue"], 0.)

    def test_empty_group_no_renormalization_and_log_underflow(self):
        r = bounded_contrast(values("C2", [.5]), {"C2": -1., "C3": 1.}, {"C2": (0., 1.), "C3": (0., 1.)})
        self.assertEqual(r["reason"], "empty_required_condition"); self.assertIsNone(r["pValue"])
        r = bounded_contrast(values("C2", [1.]*2000), {"C2": 1.}, {"C2": (0., 1.)})
        self.assertEqual(r["pValue"], 0.); self.assertTrue(r["pValueUnderflow"])
        self.assertTrue(math.isfinite(r["logPValue"])); self.assertLess(r["logPValue"], -1000)

    def test_bad_generic_parameters_and_independent_identities(self):
        own = values("C2", [.3]); w = {"C2": 1.}; b = {"C2": (0., 1.)}
        for key, bad in (("null", True), ("null", "0"), ("null", float("nan")), ("confidence", True), ("confidence", "0.95"), ("side", "auto")):
            with self.assertRaises(ValueError): bounded_contrast(own, w, b, **{key: bad})
        for weight in (True, "1", 0, float("inf")):
            with self.assertRaises(ValueError): bounded_contrast(own, {"C2": weight}, b)
        for domain in ((0, 0), (1, 0), (False, 1), ("0", 1), (0, float("inf")), "01", None):
            with self.assertRaises(ValueError): bounded_contrast(own, w, {"C2": domain})
        for data in (values("C2", [1.1]), own+own, [{**own[0], "participantId": " "}], [{**own[0], "value": "0.3"}]):
            with self.assertRaises(ValueError): bounded_contrast(data, w, b)

    def test_finite_source_person_estimands_and_nine_slots(self):
        checked = validate_geometry(self.rows, "synthetic_design", "synthetic_validation", GEOMETRY)
        for policy in POLICIES:
            fit = fit_bounded(self.rows, "synthetic_design", "synthetic_validation", GEOMETRY, policy)
            self.assertEqual(len(fit["tests"]), 9); self.assertFalse(fit["multiplicity"]["allCoreSuccess"])
            self.assertEqual(fit["practiceRowsExcluded"], 10)
            self.assertEqual(fit["multiplicity"]["retainedUnavailableSlots"], 3)
            for test in fit["tests"]:
                if test["endpoint"] not in BOUNDS:
                    self.assertEqual(test["reason"], "method_unsupported_initial_projection")
                    self.assertIsNone(test["pValue"]); self.assertIsNone(test["interval"]); self.assertFalse(test["reject"])
            for endpoint in BOUNDS:
                actual = [v for v in fit["participantEstimates"] if v["endpoint"] == endpoint]
                self.assertEqual(actual, participant_values(checked, endpoint))

    def test_record_complete_vs_missing_endpoint_and_partial(self):
        rows = copy.deepcopy(self.rows)
        first = next(r for r in rows if r["conditionId"] == "C2" and r["stageIndex"] == 1 and r["stage"] == "main")
        for k in ("shareA", "EOResemblance", "lowerNeedDirectional", "distancePP_equal_split", "distancePP_equal_outcome"): first[k] = None
        fit = fit_bounded(rows, "synthetic_design", "synthetic_validation", GEOMETRY)
        p = next(v for v in fit["participantEstimates"] if v["participantId"] == first["participantId"] and v["endpoint"] == "EOResemblance")
        self.assertTrue(p["completeTen"]); self.assertEqual(p["observations"], 9)
        self.assertTrue(p["endpointMissingWithinCompleteHistory"])
        partial = [r for r in rows if r["participantId"] != first["participantId"] or r["stage"] == "practice" or r["stageIndex"] <= 4]
        fit = fit_bounded(partial, "synthetic_design", "synthetic_validation", GEOMETRY)
        p = next(v for v in fit["participantEstimates"] if v["participantId"] == first["participantId"] and v["endpoint"] == "EOResemblance")
        self.assertEqual(p["status"], "unavailable"); self.assertEqual(p["mainTrials"], 4)
        self.assertEqual(next(t for t in fit["tests"] if t["id"] == "resemblance_C2")["groups"]["C2"]["participants"], 1)

    def test_geometry_and_provenance_corruption(self):
        edits = (lambda r: r.update(target_equal_split=.51), lambda r: r.update(target_equal_outcome=1.1),
                 lambda r: r.update(target_equal_outcome=.123), lambda r: r.update(shareA=.00001),
                 lambda r: r.update(distancePP_equal_split=60), lambda r: r.update(EOResemblance=51),
                 lambda r: r.update(lowerNeedDirectional=.51), lambda r: r.update(terminalTie=not r["terminalTie"]),
                 lambda r: r.update(meanATerminal=101), lambda r: r.update(remainingBudget=-1),
                 lambda r: r.update(dataRole="software_check"), lambda r: r.update(dataRole="unclassified"))
        for edit in edits:
            bad = copy.deepcopy(self.rows); edit(bad[0])
            with self.assertRaises(ValueError): fit_bounded(bad, "synthetic_design", "synthetic_validation", GEOMETRY)
        with self.assertRaises(ValueError): fit_bounded(self.rows, "synthetic_design", "synthetic_validation", "surrogate")
        with self.assertRaises(ValueError): fit_bounded(self.rows+self.rows[:1], "synthetic_design", "synthetic_validation", GEOMETRY)

    def test_full_grid_and_up_to_ten_raw_mean_envelope(self):
        p = np.arange(1001)/1000; targets = p[:, None]
        raw = 100*np.abs(p[None, :]-.5)-100*np.abs(p[None, :]-targets)
        self.assertEqual(raw.size, 1002001); self.assertEqual(int((raw < -50).sum()), 173)
        self.assertGreaterEqual(float(raw.min()), -50-TRIAL_R_ENVELOPE)
        self.assertLessEqual(float(raw.max()), 50+TRIAL_R_ENVELOPE)
        for n in range(1, 11):
            value = float(np.mean([float(raw.min())]*n))
            self.assertGreaterEqual(value, BOUNDS["EOResemblance"][0])
        with self.assertRaises(ValueError): bounded_contrast(values("C2", [-50-2*PERSON_R_ENVELOPE]), {"C2": 1.}, {"C2": BOUNDS["EOResemblance"]})

    def test_genuine_clipped_c5_raw_geometry_not_clamped(self):
        from design_validation.bounded_task import BoundedTask, belief_moments, expand_templates
        import json
        root = Path(__file__).resolve().parents[1]
        design = json.loads((root/"configs/tutoring_conditions.json").read_text())
        c = next(c for c in expand_templates(design) if c.condition_id == "C5" and c.scale == .95)
        task = BoundedTask(c); logs = [x.copy() for x in task.initial_logs]
        for _ in range(10): logs[0] = task.posterior(logs[0], 100); logs[1] = task.posterior(logs[1], 1)
        a, b = [belief_moments(x)[0] for x in logs]; budget = (c.total_millihours-20*c.cost_millihours)/1000
        row = copy.deepcopy(next(r for r in self.rows if r["conditionId"] == "C5" and r["stage"] == "main"))
        row.update(shareA=.001, meanATerminal=a, meanBTerminal=b, remainingBudget=budget,
                   target_equal_split=.5, target_equal_outcome=1., terminalTie=False,
                   distancePP_equal_split=100*abs(.001-.5), distancePP_equal_outcome=100*abs(.001-1),
                   lowerNeedDirectional=.5-.001, questions=20, anyQuestion=1)
        row["EOResemblance"] = row["distancePP_equal_split"]-row["distancePP_equal_outcome"]
        self.assertLess(row["EOResemblance"], -50)
        checked = validate_geometry([row], "synthetic_design", "synthetic_validation", GEOMETRY)
        self.assertEqual(checked[0]["EOResemblance"], row["EOResemblance"])
        row["EOResemblance"] = -50-2*TRIAL_R_ENVELOPE
        with self.assertRaises(ValueError): validate_geometry([row], "synthetic_design", "synthetic_validation", GEOMETRY)


if __name__ == "__main__": unittest.main()
