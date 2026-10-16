"""Fresh-input publication wiring and preservation of negative scientific outcomes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutoring_analysis import reproduction as R
from tutoring_analysis.identifiability_robustness import prepare_context, source_call

def read(path):
    return json.loads(Path(path).read_text())


class ReproductionTests(unittest.TestCase):
    def test_historical_failure_enumeration_keeps_all_routes(self):
        metric = {"numerator": 7, "denominator": 10, "rate": .7, "interval": [.3, .9]}
        test = {"id": "initial_C3", "coverageCriterionPassed": False,
            "rawErrorCriterionPassed": False, "designatedLowTailCriterionPassed": False,
            "coverage": metric, "rawFalseReject": metric, "firstGlobalThresholdReject": metric}
        policy = {"globalErrorCriterionPassed": False, "trueNullFWER": metric,
                  "families": {"responsiveness": {"criterionPassed": False, "trueNullFWER": metric}}}
        pop = {"tests": [test], "policies": {"global_holm": policy}}
        data = {"attempts": 10, "criteria": {"coverageLower": .93, "ordinaryErrorUpper": .06,
                "lowTailErrorAllowance": .001}, "cells": [{"cell": {"id": "skew", "nominalStatus": "stress"},
                "populations": {"primary": pop, "early": pop}}]}
        self.assertEqual(len(R.historical_failures(data)), 10)
        with tempfile.TemporaryDirectory() as directory:
            R.write_historical_failures(directory, data)
            result = read(Path(directory)/"criteria_readback.json")
            self.assertIn("MIXED", result["status"])
            self.assertEqual(len(result["failureRecords"]), 10)
            self.assertEqual(len((Path(directory)/"all_criterion_failures.csv").read_text().splitlines()), 11)

    def test_bounded_fresh_entry_retains_unsupported_slopes(self):
        config = read(ROOT/"configs/tutoring_bounded_robustness_v01.json")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"bounded"
            result = R.run_calibration("bounded", config, out, "fixture")
            self.assertEqual(result["attempts"], 30)
            self.assertEqual(len(list((out/"batches").glob("*.json"))), 10)
            rows = read(out/"summary.json")["cells"][0]["routes"]
            self.assertEqual(sum(row["methodUnsupported"] for row in rows), 3)
            self.assertTrue(read(out/"settings.json")["fixtureIsAlternateSizedNotScientificPrefix"])
            with self.assertRaises(ValueError):
                R.run_calibration("bounded", config, out, "fixture")

    def test_variance_context_fresh_native_schedule_and_raw_truths(self):
        config = read(ROOT/"configs/tutoring_identifiability_robustness_v01.json")
        design = read(ROOT/"configs/tutoring_conditions.json")
        context = prepare_context(config, design)
        self.assertEqual(context["cornerCoefficients"], 3072)
        self.assertEqual(len(context["cells"]), 12)
        self.assertEqual(len(R.cohort_examples(config, context["domain"])), 135)
        for row in R.cohort_examples(config, context["domain"]):
            self.assertFalse(row["confidenceInterval"])
            self.assertFalse(row["superpopulationRepair"])
        for cell in context["cells"]:
            truth = cell["truths"]["selectedCore"]
            self.assertEqual(truth["rawNullRoster"],
                [definition[0] for definition, value in zip(R.CORE, truth["rawSignedTruth"]) if value <= 0])
        for variant in range(9):
            spec = {**config["projectionSchedules"], "conditions": list(R.CONDITIONS), "scheduleVariant": variant}
            schedules = source_call("schedules", design, spec)
            self.assertEqual(len(schedules), 5)
            for schedule in schedules:
                definitions = schedule["definitions"]
                self.assertEqual([d["stageIndex"] for d in definitions], list(range(1, 11)))
                self.assertEqual(sum(d["sideReversed"] for d in definitions), 5)
                self.assertEqual([d["scheduleIndex"] for d in definitions], list(range(4, 14)))
                self.assertFalse(any(len({d["scale"] for d in definitions[i:i+4]}) == 1 for i in range(7)))
        with self.assertRaises(ValueError):
            source_call("panel", design, config["projectionSchedules"])

    def test_variance_actual_cli_no_private_context(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"adaptive"
            subprocess.run([sys.executable, str(ROOT/"scripts/reproduce_tutoring_statistics.py"),
                "--analysis", "variance-adaptive", "--mode", "fixture", "--output", str(out)],
                cwd=directory, check=True, capture_output=True, text=True)
            self.assertEqual(read(out/"summary.json")["attempts"], 36)
            self.assertEqual(len(read(out/"cohort_identification.json")), 135)
            audit = read(out/"C1_states.json")
            self.assertEqual(len(audit["states"]), 30)
            self.assertEqual(audit["candidateStates"], 18)
            self.assertEqual(audit["comparatorStates"], 12)
            self.assertFalse(audit["candidateAdopted"])
            self.assertFalse(audit["batchExpandedAfterOutcomes"])
            for state in audit["states"]:
                self.assertFalse(state["candidateAdopted"])
                self.assertFalse(state["historicalEightPointGate"])
                self.assertFalse(state["strictlyDisjointClosedFivePointMatchNeighborhoods"])
                self.assertEqual(state["numericalNoSearchBoundPass"],
                    state["acquisitionOnlyUpperBound"]["upper_bound_gain"] < -1e-8)
                if state["delta"] == 0:
                    self.assertEqual(state["kind"], "symmetric_comparator")
                    self.assertAlmostEqual(state["EOESGapShare"], 0., places=12)
                if state["delta"] == 4:
                    self.assertEqual(state["kind"], "old_failed_delta4_comparator")
                    self.assertFalse(state["fivePointGate"])
            self.assertEqual(read(out/"prepared_context.json")["cornerCoefficients"], 3072)

    def test_inference_and_planning_entrypoints(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            R.run_inference(base/"inference", ROOT)
            self.assertEqual(len(read(base/"inference/finite_bounded_fit.json")["tests"]), 9)
            R.run_planning(read(ROOT/"configs/tutoring_power_sensitivity_v01.json"), base/"planning")
            self.assertEqual(len(read(base/"planning/planning.json")["scenarios"]), 28)


if __name__ == "__main__":
    unittest.main()
