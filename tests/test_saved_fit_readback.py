"""Fixed fixtures for independent eight-contrast readback and input rejection.

No producer import, generated inputs, model fit or package calculation is used.
The useful readback job separately supplies the actual saved-fit positive case.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.tutoring_statistics import independent_saved_fit as reader
READER = Path(__file__).resolve().parents[1] / "scripts/reproduce_saved_fit_magnitude.py"


def references():
    # Hand-specified probabilities test composition, not t calibration.
    result = {}
    for key in reader.IDS:
        for kind, less, greater in (("D", .001, .999), ("S", .999, .001)):
            name = key + ":" + kind
            result[name] = dict(linearID=name, status="available", reason=None,
                L=[0.] * 9, coefficientNames=list(reader.NAMES), estimate=.1, SE=.2,
                df=20. if kind == "D" else 35., statistic=.5,
                lessP=less, greaterP=greater, twoSidedP=.6, CI=[-.3, .5])
    return result


def candidate_method():
    refs = references()
    tests = {key: dict(status="available", inputP=dict(D_less=.001, D_greater=.999, S_less=.999, S_greater=.001),
        orientationP=dict(reference_positive=.001, reference_negative=.999), pMagnitude=.002,
        holmPaddingP=.002, holmP=.008, holmReject=True, nominalReject=True) for key in reader.IDS}
    return dict(covariance=[[1. if i == j else 0. for j in range(9)] for i in range(9)],
        contrastCovariance=[[1. if i == j else 0. for j in range(8)] for i in range(8)], references=refs,
        composition=dict(methodID=reader.METHOD, familySize=4, alpha=.05, tests=tests,
            status="complete", holmComputed=True, anyMagnitudeRejects=True, completeFamilyAvailable=True,
            counts=dict(planned=4, technicalCompleted=4, available=4, statisticalUnavailable=0, technicalUnresolved=0)))


class MagnitudeIndependentReadbackTests(unittest.TestCase):
    def test_population_exact_clusters_and_rejections(self):
        rows = [dict(participant=arm + "person", condition=arm, need_A=1., need_B=2., y=3.)
                for arm in ("C1", "C2", "C3") for _ in range(10)]
        self.assertEqual(reader.population(rows), dict(C1=1, C2=1, C3=1))
        for damaged in (rows[:-1], [dict(row, condition="C2") if i == 0 else row for i, row in enumerate(rows)],
                        [dict(row, need_A=True) if i == 0 else row for i, row in enumerate(rows)]):
            with self.assertRaises(ValueError):
                reader.population(damaged)

    def test_exact_and_continuous_boundaries(self):
        reader.near(.5 + 1e-11, .5)
        for a, b in ((True, 1), (1., 1), ([1], [1, 2])):
            with self.assertRaises(ValueError):
                reader.exact(a, b)
        for a in (True, float("nan"), float("inf"), .51):
            with self.assertRaises(ValueError):
                reader.near(a, .5)

    def test_manual_composition_partial_and_technical(self):
        refs = references()
        outcome = reader.independent_composition(refs)
        self.assertTrue(outcome["anyMagnitudeRejects"])
        self.assertEqual([v["holmP"] for v in outcome["tests"].values()], [.008] * 4)
        refs[reader.IDS[-1] + ":S"]["status"] = "statistical_unavailable"
        outcome = reader.independent_composition(refs)
        self.assertTrue(outcome["anyMagnitudeRejects"])
        self.assertIsNone(outcome["tests"][reader.IDS[-1]]["holmP"])
        self.assertFalse(outcome["tests"][reader.IDS[-1]]["holmReject"])
        refs[reader.IDS[-1] + ":S"]["status"] = "technical_unresolved"
        outcome = reader.independent_composition(refs)
        self.assertIsNone(outcome["anyMagnitudeRejects"])
        self.assertTrue(all(v["holmReject"] is None for v in outcome["tests"].values()))
        del refs[reader.IDS[-1] + ":D"]
        with self.assertRaises(ValueError):
            reader.independent_composition(refs)

    def test_package_method_comparison_rejects_sum_and_vector_changes(self):
        method = candidate_method()
        expected = {key: copy.deepcopy(method[key]) for key in ("covariance", "contrastCovariance", "references")}
        reader.compare_method(method, expected, "fixture")
        for field, value in (("df", 20.), ("L", [1e-16] + [0.] * 8), ("lessP", .2),
                             ("coefficientNames", list(reversed(reader.NAMES))), ("status", "statistical_unavailable")):
            damaged = copy.deepcopy(method)
            damaged["references"][reader.IDS[0] + ":S"][field] = value
            with self.assertRaises(ValueError):
                reader.compare_method(damaged, expected, "fixture")
        damaged = copy.deepcopy(method)
        damaged["references"][reader.IDS[0] + ":S"]["L"][0] = True
        with self.assertRaises(ValueError):
            reader.compare_method(damaged, expected, "fixture")

    def test_full_guard_components_not_just_usable(self):
        raw = dict(status="fitted", coefficients={"x": 1.}, vcov=[[1.]],
            diagnostics=dict(converged=True, singular=False, usableCandidateInference=True,
                             gradient=0., hessianEigenvalues=[1., 2.]))
        self.assertTrue(reader.full_guard(raw)["usableCandidateInference"])
        raw["diagnostics"].update(usableCandidateInference=False, gradient=None)
        a = reader.full_guard(raw)
        raw["diagnostics"].update(gradient=0., hessianEigenvalues=[0., 2.])
        b = reader.full_guard(raw)
        self.assertNotEqual(a, b)
        with self.assertRaises(ValueError):
            reader.exact(a, b)

    def test_strict_source_membership_and_no_overwrite(self):
        class Bound:
            def read(self, pin):
                if pin["sha256"] == reader.AGG_SHA:
                    return dict(schema="h10-power-design-aggregate-v1", complete=True,
                                expectedTotal=80000, completedTotal=80000, independentBlocks=[])
                return dict(schema="h10-power-design-independent-aggregate-v1", passed=True, sourceClosure=True,
                    complete=True, aggregate=dict(sha256=reader.AGG_SHA), manifest=dict(sha256=reader.MANIFEST_SHA), independentBlocks=[])
        request = dict(target=reader.TARGET, aggregate=dict(path="/a", sha256=reader.AGG_SHA),
            independentAggregate=dict(path="/b", sha256=reader.INDEP_SHA), blockProof=dict(path="/c", sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Block not accepted"):
            reader.admit_history(request, Bound())
        with self.assertRaises(ValueError):
            reader.decode('{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            reader.decode('{"x":NaN}')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            reader.write_new(path, dict(retained=True))
            with self.assertRaises(ValueError):
                reader.write_new(path, dict(retained=False))
            self.assertEqual(path.read_text(), '{\n  "retained": true\n}\n')

    def test_actual_cli_rejects_wrong_request_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            request, produced, binding, output = [folder / name for name in ("request.json", "producer.json", "binding.json", "proof.json")]
            request.write_text('{"schema":"wrong"}')
            produced.write_text('{}')
            process = subprocess.run([sys.executable, str(READER), "verify", "--input", str(request), "--producer-result", str(produced),
                "--output", str(output)], text=True, capture_output=True, check=False)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Wrong scientific request", process.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
