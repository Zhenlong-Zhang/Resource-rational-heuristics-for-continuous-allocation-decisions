"""Purpose: preserve fixed batching, native truths and independent calibration readback."""
import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import beta, nct, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tutoring_analysis import native_calibration as C, native_readback as R
from src.tutoring_analysis.expected_outcomes import digest


class NativeCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = json.loads((ROOT / "configs/expected_gpa_analysis.json").read_text())
        cls.config = json.loads((ROOT / "configs/native_calibration.json").read_text())
        cls.cells = {c["id"] + ":" + c["subcase"]: c for c in C.requests(cls.analysis, cls.config)}

    def test_frozen_roster_seeds_and_batch_resume(self):
        self.assertEqual(len(self.cells), 40)
        self.assertEqual(self.cells, R.cells_for(self.analysis, self.config))
        for cell in self.cells.values():
            for i in (0, 39):
                self.assertEqual(str(C._seed(cell, i)), R.seed(cell, i))
        self.assertEqual(C._seed(self.cells["B12:forward"], 0), C._seed(self.cells["B12:sign_mirror"], 0))
        cell = {**self.cells["A02:null"], "onlyBatchIds": [0, 1, 39]}
        self.assertEqual(C._pending_batch_ids(cell, {0: {}}), [1, 39])
        for ids in ([0, 0], [-1], [40], [True]):
            with self.assertRaises(ValueError):
                C._pending_batch_ids({**cell, "onlyBatchIds": ids}, {})

    def test_exact_coverage_does_not_round_truth_to_boundary(self):
        truth = {"coverageEligible": True, "exactNativeExpectation": {"numerator": "1", "denominator": str(10**100)}}
        result = {"available": np.array([True, True, False]), "low": np.array([0., 0., 0.]),
                  "high": np.array([0., 1., 1.])}
        self.assertEqual(C._coverage_mask(result, truth).tolist(), [False, True, False])
        truth["coverageEligible"] = False
        self.assertFalse(C._coverage_mask(result, truth).any())

    def test_independent_cp_and_a05_reference(self):
        for successes, n in ((0, 250), (250, 250), (12, 250), (500, 10000), (9500, 10000)):
            actual = R.cp(successes, n)
            expected = [0 if successes == 0 else beta.ppf((1-.95)/2, successes, n-successes+1),
                        1 if successes == n else beta.ppf(1-(1-.95)/2, successes+1, n-successes)]
            np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)
        for key in ("A05:correct_direction", "A05:separate_pooled_reference"):
            cell = self.cells[key]
            specs = C._specs_for(cell, self.analysis)
            spec = next(s for s in specs if len(s["weights"]) == (2 if "pooled" in key else 1))
            one = len(spec["weights"]) == 1
            n = cell["n"]
            df = n - 1 if one else 2*n - 2
            expected = nct.sf(t.ppf(.95, df), df, .3*np.sqrt(n if one else n/2))
            self.assertAlmostEqual(R.normal_reference_power(cell, spec), expected, places=9)

    def test_readback_uses_public_spec_without_private_progress_metadata(self):
        cell = self.cells["A02:null"]
        specs = R.specs_for(cell, self.analysis)
        accepted = {"targets": [{"testId": s["id"], "numericalValue": 0.,
                    "truthStatus": "exact", "classification": "boundary_null",
                    "coverageEligible": True, "typeIEligible": True} for s in specs]}
        candidate = {"targets": [{"testId": s["id"], "selectedNativeFloatingTruth": 0.} for s in specs]}
        result = R.summary([], cell, specs, candidate, accepted, "binding", "truth", self.analysis)
        self.assertFalse(result["complete"])
        self.assertEqual(result["completedAttempts"], 0)
        for row in result["targets"]:
            self.assertTrue(row["primary"])
            self.assertEqual(row["status"], "PROVISIONAL")
            self.assertIsNone(row["availableCoverageRate"])
            self.assertNotIn("model_premise_status", row)

    def test_empty_producer_summary_uses_public_spec(self):
        cell = {**self.cells["A02:null"], "onlyBatchIds": []}
        specs = C._specs_for(cell, self.analysis)
        accepted = {"targets": [{"testId": s["id"], "numericalValue": 0.,
                    "truthStatus": "exact", "classification": "boundary_null",
                    "coverageEligible": True, "typeIEligible": True} for s in specs]}
        candidate = {"targets": [{"testId": s["id"], "selectedNativeFloatingTruth": 0.} for s in specs]}
        # Isolate empty-summary formatting; real prepared-input checks have separate coverage.
        with patch.object(C, "_check_execution"), patch.object(C, "_validate_truth_bundle", return_value=(candidate, accepted)):
            result = C.run_cell(cell, {"receipt": {"sourceReceiptHash": "source", "lookupHash": "lookup"}},
                                {}, {"nativeTruthReceiptHash": "truth"}, [])
        self.assertFalse(result["complete"])
        for row in result["targets"]:
            self.assertTrue(row["primary"])
            self.assertEqual(row["status"], "PROVISIONAL")
            self.assertNotIn("model_premise_status", row)

    def test_normal_scalar_vector_audit(self):
        cell = self.cells["A02:null"]
        specs = C._specs_for(cell, self.analysis)
        endpoints, _ = C._normal_batch(cell, specs, np.random.default_rng(C._seed(cell, 0)), 250)
        for spec in specs:
            vector = C._vector_contrast(endpoints, spec, False)
            for i in (0, 1, 17, 249):
                scalar = R.contrast({a: endpoints[a][spec["endpoint"]][i].tolist() for a in spec["weights"]}, spec)
                for field in ("effect", "SE", "df", "statistic", "p", "low", "high"):
                    self.assertAlmostEqual(vector[field][i], scalar[field], delta=1e-10)
                self.assertEqual(bool(vector["reject"][i]), scalar["reject"])

    def test_cli_help_and_immutable_write(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_native_calibration.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        spec = importlib.util.spec_from_file_location("native_cli", ROOT / "scripts/reproduce_native_calibration.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch.json"
            cli.write(path, {"a": 1})
            cli.write(path, {"a": 1})
            with self.assertRaises(ValueError):
                cli.write(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            ordered = Path(folder) / "analysis.json"
            cli.write(ordered, self.analysis)
            self.assertEqual(list(cli.read(ordered)["claims"][0]["weights"].values()), [1, -1])


if __name__ == "__main__":
    unittest.main()
