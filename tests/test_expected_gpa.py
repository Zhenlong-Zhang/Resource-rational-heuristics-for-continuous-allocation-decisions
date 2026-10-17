"""Test finite expected-GPA geometry, nominal inference and portable entrypoints."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce_expected_gpa import templates, convert_trials
from src.design_validation.budget_horizon import BudgetTask
from src.design_validation.terminal_time import terminal_summary
from src.tutoring_analysis import conditional_power as P, nominal_inference as I
from src.tutoring_analysis.expected_outcomes import allocation_metrics, trial_row

SPEC = json.loads((ROOT / "configs/expected_gpa_analysis.json").read_text())


class ExpectedGPATests(unittest.TestCase):
    def test_target_and_metric_definition(self):
        config = templates()["C3_s0_g0"]
        task = BudgetTask(config)
        target = terminal_summary(config, *task.initial_logs, config.total_millihours)
        j = target["macros"]["EO_expected_GPA"]["index"]
        row = trial_row(config, [], j, participant_id="p", main_index=1)
        self.assertEqual(row["D_EGPA_pp"], 0)
        self.assertEqual(row["R_EGPA_pp"], abs(j - 500) / 10)
        at_split = allocation_metrics(row, 500)
        self.assertEqual(at_split["R_EGPA_pp"], -abs(j - 500) / 10)

    def test_raw_history_and_mirror(self):
        template = "C1_s1_d1000"
        record = {"templateId": template, "participantId": "p", "mainIndex": 1,
                  "humanObservations": False, "paid": [{"student": 0, "report": 45}],
                  "shareIndex": 600, "sideReversed": True}
        row = convert_trials([record], templates())[0]
        self.assertGreater(row["initialMeans"][0], row["initialMeans"][1])
        self.assertEqual(row["paidCount"], 1)
        self.assertEqual(row["remainingIntegerBudget"], row["initialIntegerBudget"] - row["paidIntegerCost"])
        with self.assertRaises(ValueError):
            convert_trials([{**record, "humanObservations": True}], templates())
        with self.assertRaises(ValueError):
            convert_trials([{**record, "paid": [{"student": True, "report": 45}]}], templates())

    def fixture(self):
        configs = templates()
        keys = sorted(k for k, c in configs.items() if c.condition_id == "C2")
        return [trial_row(configs[keys[(i - 1) % len(keys)]], [], 500 + i,
                          participant_id="p", main_index=i) for i in range(1, 11)]

    def test_person_complete_ten_and_missingness(self):
        rows = self.fixture()
        person = I.participant_endpoints(rows, SPEC)[0]
        self.assertEqual(person["endpoints"]["R_EGPA_pp"]["status"], "available")
        rows[0]["recordPresent"] = False
        person = I.participant_endpoints(rows, SPEC)[0]
        self.assertEqual(person["endpoints"]["R_EGPA_pp"]["status"], "unavailable")
        self.assertEqual(person["endpoints"]["available_history_R_EGPA_pp"]["status"], "available")

    def test_primary_roster_and_no_missing_arm_rescue(self):
        result = I.analyze(self.fixture(), SPEC)
        self.assertEqual(sum(x["namespace"] == "primary" for x in result["tests"]), 9)
        self.assertFalse(next(x for x in result["tests"] if x["testId"] == "D10_R_EGPA_C1")["primary"])
        self.assertIsNone(result["globalErrorGuarantee"])
        self.assertTrue(all(x["status"] == "unavailable" for x in result["tests"]))
        bad = copy.deepcopy(SPEC)
        bad["claims"][5]["tail"] = "less"
        with self.assertRaises(ValueError):
            I.validate_spec(bad)
        with self.assertRaises(ValueError):
            I.validate_spec({**SPEC, "questionLimitRule": "cap20"})

    def test_exhaustion_uncommitted_ties_and_invalid_people(self):
        config = next(c for c in templates().values()
                      if c.total_millihours % c.cost_millihours == 0)
        paid = [{"student": i % 2, "report": 50}
                for i in range(config.total_millihours // config.cost_millihours)]
        exhausted = trial_row(config, paid, None, participant_id="p", main_index=1)
        rows = [{**copy.deepcopy(exhausted), "mainIndex": i} for i in range(1, 11)]
        person = I.participant_endpoints(rows, SPEC)[0]
        self.assertEqual(person["endpoints"]["Q"]["value"], 1.)
        self.assertEqual(person["endpoints"]["R_EGPA_pp"]["status"], "unavailable")
        uncommitted = trial_row(config, [], None, participant_id="p", main_index=1)
        self.assertFalse(uncommitted["choiceAvailable"])
        self.assertIsNone(uncommitted["R_EGPA_pp"])
        with self.assertRaises(ValueError):
            trial_row(config, paid, 500, participant_id="p", main_index=1)
        tie = {**self.fixture()[0], "terminalMeans": [50., 50.]}
        self.assertEqual(allocation_metrics(tie, 1000)["L_lower_mean_share"], 0.)
        for mixed in (False, True):
            invalid = self.fixture()
            if mixed:
                invalid[0]["sessionId"] = "different"
            else:
                invalid.append(copy.deepcopy(invalid[0]))
            self.assertTrue(I.participant_endpoints(invalid, SPEC)[0]["eligibility"]["quarantineReasons"])
        self.assertTrue(all(t["status"] == "unavailable" for t in I.analyze([], SPEC)["tests"]))

    def test_nominal_contrast_matches_scipy(self):
        from scipy.stats import ttest_1samp
        values = [1., 2., 3., 5.]
        people = [{"participantId": str(i), "conditionId": "C2",
                   "endpoints": {"R_EGPA_pp": {"status": "available", "value": value}}}
                  for i, value in enumerate(values)]
        actual = I.test_contrast(people, SPEC["claims"][5])
        expected = ttest_1samp(values, 0., alternative="greater")
        self.assertAlmostEqual(actual["p"], expected.pvalue)
        self.assertEqual(actual["alpha"], .05)
        opposite = I.test_contrast(people, {**SPEC["claims"][5], "id": "negative_diagnostic",
                                           "primary": False, "tail": "less"})
        self.assertAlmostEqual(actual["p"] + opposite["p"], 1.)
        for person in people:
            person["endpoints"]["R_EGPA_pp"]["value"] = 1.
        self.assertEqual(I.test_contrast(people, SPEC["claims"][5])["status"], "unavailable")

    def test_conditional_power_boundary_and_population(self):
        data = P.report()
        self.assertEqual(len(data["rows"]), 24)
        focus = [r for r in data["rows"] if r["d"] == .3 and r["targetPower"] == .8
                 and r["tail"] == "one_sided_primary"]
        self.assertEqual([r["boundary"]["minimum"] for r in focus], [71, 139])
        self.assertFalse(data["finalNSelected"])
        for row in data["rows"]:
            self.assertLess(row["boundary"]["previousProbability"], row["targetPower"])
            self.assertGreaterEqual(row["boundary"]["probability"], row["targetPower"])

    def test_real_entrypoints(self):
        runner = ROOT / "scripts/reproduce_expected_gpa.py"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for stage, options in (("targets", ["--template", "C3_s0_g0"]), ("power", [])):
                command = [sys.executable, str(runner), stage, "--output", str(root / stage), *options]
                subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
                self.assertTrue(json.loads((root / stage / "completion.json").read_text())["complete"])
                self.assertNotEqual(subprocess.run(command, cwd=root, capture_output=True).returncode, 0)
            trials = [{"templateId": "C2_s0_g-8", "participantId": "p", "mainIndex": i,
                       "humanObservations": False, "paid": [], "shareIndex": 500 + i} for i in range(1, 11)]
            first = next(k for k, c in templates().items() if c.condition_id == "C2")
            for row in trials:
                row["templateId"] = first
            raw = root / "trials.json"
            raw.write_text(json.dumps(trials))
            subprocess.run([sys.executable, str(runner), "analyze", "--trials", str(raw),
                            "--output", str(root / "analysis")], cwd=root, check=True, capture_output=True)
            data = json.loads((root / "analysis/result.json").read_text())
            self.assertEqual(len(data["people"]), 1)


if __name__ == "__main__":
    unittest.main()
