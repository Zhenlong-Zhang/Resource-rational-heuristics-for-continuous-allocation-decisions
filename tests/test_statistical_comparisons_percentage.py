"""Percentage-unit equivalence, ordered cancellation and synthetic-input admission.

These are executable unittest software tests, not calibration or human evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tutoring_statistics import comparison_percentage as percentage
from tutoring_statistics.joint_design import lower_need_scores, prepare_population
from src.tutoring_analysis.nominal_inference import test_contrast, canonical_mean
from tutoring_statistics.finite_means import canonical_mean as shared_mean


class PercentageSoftwareTests(unittest.TestCase):
    def test_packaged_nominal_kernel_import(self):
        self.assertEqual(percentage.frozen_contrast.__module__, 'src.tutoring_analysis.nominal_inference')

    def test_exact_frozen_Student_then_unit_transform(self):
        result = percentage.percentage_analysis(percentage.example_payload())
        for hypothesis in ("H8", "H9"):
            actual = result[hypothesis]
            reference = result["equivalence"][hypothesis]["fractionReference"]
            self.assertEqual(actual["status"], "available")
            for name in ("status", "reason", "statistic", "df", "p"):
                self.assertEqual(actual[name], reference[name])
            self.assertEqual(actual["effect"], 100 * reference["effect"])
            self.assertEqual(actual["SE"], 100 * reference["SE"])
            self.assertEqual(actual["candidateReject"], reference["candidateNominalReject"])
            self.assertEqual(actual["meanCI95"], [50 + 100 * x for x in reference["CI"]])
            self.assertEqual(result["equivalence"][hypothesis]["status"], "pass")

    def test_ordered_cancellation_cannot_be_replaced_by_display_retesting(self):
        payload = percentage.example_payload(cancellation=True)
        result = percentage.percentage_analysis(payload)
        self.assertEqual(float(shared_mean([.1, .2, -.3] + [0.] * 7)), float(canonical_mean([.1, .2, -.3] + [0.] * 7)))
        for key in ("H8", "H9"):
            self.assertEqual(result[key]["status"], "available")
            self.assertEqual(result[key]["mean"], 50.)
            self.assertNotEqual(result[key]["effect"], 0.)
            self.assertTrue(result[key]["displayRoundingErasesEffect"])
            direct = result["directPercentDiagnostics"][key]
            self.assertEqual(direct["status"], "numerical_representation_changes_availability")
            self.assertEqual(direct["result"]["reason"], "zero_or_nonfinite_SE")
            self.assertFalse(direct["controlsMainInference"])
        reverse = percentage.percentage_analysis(dict(payload, rows=list(reversed(payload["rows"]))))
        self.assertEqual(result["H8"], reverse["H8"])
        changed = deepcopy(payload)
        for row, j in zip(changed["rows"][:3], (600, 200, 700)):
            row.update(shareIndex=j, share=j / 1000)
        altered = percentage.percentage_analysis(changed)
        self.assertNotEqual(result["H8"]["effect"], altered["H8"]["effect"])

    def test_complete_positive_budget_person_eligibility_and_tie(self):
        payload = percentage.example_payload()
        participant = payload["rows"][0]["participant"]
        own = [r for r in payload["rows"] if r["participant"] == participant]
        for reason, rows in (("incomplete_ten_main_history", own[:-1]),
                             ("duplicate_main_index", own + [own[0]])):
            result = prepare_population(dict(rows=rows))
            self.assertFalse(result["eligibleParticipants"])
            self.assertIn(reason, result["participants"][0]["reasons"])
        exhausted = deepcopy(own)
        exhausted[0].update(share=None, shareIndex=None, remainingBudgetMs=0)
        result = prepare_population(dict(rows=exhausted))
        self.assertEqual(len(result["completeHistoryRows"]), 10)
        self.assertFalse(result["eligibleParticipants"])
        tie = dict(own[0], meanATerminal=50., meanBTerminal=50. + 1e-9)
        self.assertEqual(lower_need_scores(tie)["centeredShare"], 0.)

    def test_zero_variance_and_small_N_are_unavailable_not_nonrejection_evidence(self):
        payload = percentage.example_payload()
        one = [r for r in payload["rows"] if r["participant"] == payload["rows"][0]["participant"]]
        small = percentage.percentage_analysis(dict(rows=one))
        self.assertEqual(small["H8"]["reason"], "fewer_than_two_eligible_people_in_required_arm")
        for row in payload["rows"]:
            row.update(shareIndex=600, share=.6)
        zero = percentage.percentage_analysis(payload)
        self.assertEqual(zero["H8"]["reason"], "zero_or_nonfinite_aggregate_SE")
        self.assertIsNone(zero["H8"]["p"])

    def test_input_origin_bad_replay_and_missing_design_refused(self):
        for data in (dict(rows=[]), dict(origin="human", rows=[])):
            with self.assertRaisesRegex(ValueError, "synthetic"):
                percentage.synthetic_input(data)
        records = dict(origin="synthetic_software_fixture", schema="tutoring-statistics-fixture-records-v1", records=[{}])
        with self.assertRaisesRegex(ValueError, "matching design"):
            percentage.synthetic_input(records)
        fake = subprocess.CompletedProcess([], 0, stdout=json.dumps(dict(ok=True, normalized=dict(errors=["bad replay"]))))
        with patch.object(percentage.subprocess, "run", return_value=fake), self.assertRaisesRegex(ValueError, "Normalizer rejected"):
            percentage.synthetic_input(records, design={})
        with patch.object(percentage.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=json.dumps(dict(ok=False, error="corrupt transport")))), self.assertRaises(ValueError):
            percentage.synthetic_input(records, design={})

    def test_actual_example_CLI_and_human_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "percentage.json"
            env = dict(__import__("os").environ, PYTHONPATH=str(ROOT / "src"))
            subprocess.run([sys.executable, "-m", "tutoring_statistics.comparison_percentage", "--example", "cancellation",
                "--design-only", "--output", str(output)], env=env, capture_output=True, text=True, check=True)
            result = json.loads(output.read_text())
            self.assertFalse(result["fittingRequested"])
            self.assertFalse(result["scientificAdoption"])
            source = Path(tmp) / "human.json"
            source.write_text(json.dumps(dict(origin="human", rows=[])))
            refused = subprocess.run([sys.executable, "-m", "tutoring_statistics.comparison_percentage", "--input", str(source),
                "--output", str(Path(tmp) / "forbidden.json")], env=env, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertFalse((Path(tmp) / "forbidden.json").exists())


if __name__ == "__main__":
    unittest.main()
