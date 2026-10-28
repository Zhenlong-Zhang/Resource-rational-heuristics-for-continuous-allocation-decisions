"""Preserve historical H6/H7 payloads and verify the portable synthetic route.

Hand-authored schema fixtures are not numerical evidence. The one integration
test separately exercises the existing public raw-history/inference generator.
No private results, scheduler environment or optional fixture skips are used.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tutoring_analysis import h6_h7_surplus as M
from src.tutoring_analysis.h6_h7_surplus_check import check_mapping
from src.tutoring_analysis.expected_outcomes import digest

OLD_SPEC = json.loads((ROOT / "configs/expected_gpa_analysis.json").read_text())
SPEC = json.loads((ROOT / "configs/h6_h7_surplus.json").read_text())


def fixture():
    """Minimal public result shape; supplied numbers only test reference copying."""
    people, tests = [], []
    for arm, old_id in (("C2", "H06_R_EGPA_C2"), ("C3", "H07_R_EGPA_C3")):
        for i in range(2):
            people.append(dict(participantId=arm + str(i), sessionId=arm + "session" + str(i),
                conditionId=arm, eligibility=dict(recordedRows=10, presentRows=10,
                completeChoiceTen=True, completePaidTen=True, quarantineReasons=[]),
                endpoints={endpoint: dict(status="available", value=float(i), reason=None, observations=10)
                           for endpoint in ("R_surplus_pp", "R_EGPA_pp")}))
        for endpoint, test_id, primary in (("R_surplus_pp", "S_surplus_" + arm, False),
                                           ("R_EGPA_pp", old_id, True)):
            tests.append(dict(testId=test_id, endpointId=endpoint, weights={arm: 1},
                tail="greater", null=0, alpha=.05, confidence=.95, primary=primary,
                method="Student", personPopulation=M.POPULATION, requiredArmCounts={arm: 2},
                namespace="primary" if primary else "supporting_descriptive",
                status="available", effect=1., SE=1., df=1., statistic=1., p=.25,
                CI=[-12., 14.], reason=None, candidateNominalReject=False,
                writer={"kind": "self_contained_schema_fixture", "identity": "original"}))
    return dict(schema="person-level-nominal-result-v1", analysisId="person-level-nominal-v1",
        analysisSpecHash=digest(OLD_SPEC), humanObservations=False, globalErrorGuarantee=None,
        scope="fixed_person_level_statistics_not_full_calibration_or_human_validation",
        people=people, tests=tests, supportingSensitivities=[])


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.analysis = fixture()

    def mapped(self, analysis=None, spec=None, old_spec=None):
        return M.map_saved_analysis(self.analysis if analysis is None else analysis,
            SPEC if spec is None else spec, OLD_SPEC if old_spec is None else old_spec)

    def rejected_by_both(self, analysis=None, spec=None, old_spec=None):
        produced = self.mapped()
        analysis = self.analysis if analysis is None else analysis
        spec = SPEC if spec is None else spec
        old_spec = OLD_SPEC if old_spec is None else old_spec
        with self.assertRaises(ValueError):
            M.map_saved_analysis(analysis, spec, old_spec)
        with self.assertRaises(ValueError):
            check_mapping(analysis, spec, old_spec, produced)

    def test_exact_historical_payload_and_roles_are_preserved(self):
        before = deepcopy(self.analysis)
        output = self.mapped()
        self.assertEqual([row["prospectiveClaimId"] for row in output["mappings"]],
                         ["H06_R_SURPLUS_C2", "S_R_EGPA_C2", "H07_R_SURPLUS_C3", "S_R_EGPA_C3"])
        old = {row["testId"]: row for row in self.analysis["tests"]}
        for row in output["mappings"]:
            self.assertEqual(row["historicalResult"], old[row["historicalTestId"]])
            self.assertEqual(row["historicalResult"]["primary"], row["prospectiveRole"] == "secondary")
        self.assertTrue(check_mapping(self.analysis, SPEC, OLD_SPEC, output)["passed"])
        output["mappings"][0]["historicalResult"]["writer"]["identity"] = "changed"
        self.assertEqual(self.analysis, before)

    def test_definition_changes_rejected(self):
        for key, value in (("referenceState", "initial"), ("tail", "less"),
                           ("noSupportingRescue", False), ("minimumPeoplePerArm", True),
                           ("humanAdapterReleased", True), ("alpha", .10)):
            with self.subTest(key=key):
                bad = deepcopy(SPEC)
                bad[key] = value
                self.rejected_by_both(spec=bad)
        bad = deepcopy(SPEC)
        bad["claims"][0]["endpoint"] = "R_EGPA_pp"
        self.rejected_by_both(spec=bad)

    def test_public_identity_scope_and_duplicates_rejected(self):
        for key, value in (("humanObservations", True), ("scope", "human"),
                           ("analysisSpecHash", "b" * 64), ("schema", "r9-candidate-inference-v1")):
            with self.subTest(key=key):
                bad = deepcopy(self.analysis)
                bad[key] = value
                self.rejected_by_both(analysis=bad)
        for collection in ("people", "tests"):
            bad = deepcopy(self.analysis)
            bad[collection].append(deepcopy(bad[collection][0]))
            self.rejected_by_both(analysis=bad)

    def test_population_counts_and_endpoint_status_rejected(self):
        for key, value in (("completeChoiceTen", False), ("recordedRows", 9), ("presentRows", True)):
            bad = deepcopy(self.analysis)
            bad["people"][0]["eligibility"][key] = value
            self.rejected_by_both(analysis=bad)
        bad = deepcopy(self.analysis)
        bad["tests"][0]["requiredArmCounts"]["C2"] = 3
        self.rejected_by_both(analysis=bad)
        bad = deepcopy(self.analysis)
        bad["people"][0]["endpoints"]["R_surplus_pp"].update(
            status="unavailable", value=None, reason="incomplete_ten_positive_defined_choices")
        self.rejected_by_both(analysis=bad)

    def test_unavailable_surplus_cannot_be_rescued(self):
        self.analysis["tests"][0].update(status="unavailable", reason="zero_or_nonfinite_aggregate_SE",
            p=None, CI=None, SE=None, df=None, statistic=None, candidateNominalReject=False)
        self.analysis["tests"][1].update(p=.01, candidateNominalReject=True)
        output = self.mapped()
        self.assertEqual(output["mappings"][0]["historicalResult"]["status"], "unavailable")
        self.assertTrue(output["mappings"][1]["historicalResult"]["candidateNominalReject"])
        self.assertTrue(output["noSupportingRescue"])
        self.assertTrue(check_mapping(self.analysis, SPEC, OLD_SPEC, output)["passed"])

    def test_missing_arm_remains_unavailable(self):
        self.analysis["people"] = [p for p in self.analysis["people"] if p["conditionId"] == "C2"]
        for row in self.analysis["tests"]:
            if "C3" in row["weights"]:
                row.update(requiredArmCounts={"C3": 0}, status="unavailable",
                    reason="fewer_than_two_eligible_people_in_required_arm", effect=None,
                    SE=None, df=None, statistic=None, p=None, CI=None, candidateNominalReject=False)
        output = self.mapped()
        self.assertTrue(check_mapping(self.analysis, SPEC, OLD_SPEC, output)["passed"])
        self.assertEqual(output["mappings"][2]["historicalResult"]["requiredArmCounts"], {"C3": 0})

    def test_old_semantics_and_malformed_uncertainty_rejected(self):
        for key, value in (("primary", True), ("tail", "less"), ("endpointId", "R_EGPA_pp"),
                           ("null", True), ("method", "normal"), ("namespace", "primary"),
                           ("p", -1), ("SE", 0), ("df", float("nan")), ("CI", [2, 1]),
                           ("candidateNominalReject", True)):
            with self.subTest(key=key):
                bad = deepcopy(self.analysis)
                bad["tests"][0][key] = value
                self.rejected_by_both(analysis=bad)
        bad = deepcopy(self.analysis)
        bad["tests"].pop()
        self.rejected_by_both(analysis=bad)
        bad_spec = deepcopy(OLD_SPEC)
        next(t for t in bad_spec["supporting"] if t["id"] == "S_surplus_C2")["tail"] = "less"
        self.rejected_by_both(old_spec=bad_spec)

    def test_independent_checker_rejects_payload_role_and_extra_claims(self):
        original = self.mapped()
        for field, value in (("prospectiveRole", "secondary"), ("historicalResultRecomputed", True),
                             ("historicalTestId", "S_surplus_C3")):
            bad = deepcopy(original)
            bad["mappings"][0][field] = value
            with self.assertRaises(ValueError):
                check_mapping(self.analysis, SPEC, OLD_SPEC, bad)
        for change in ("p", "writer"):
            bad = deepcopy(original)
            bad["mappings"][0]["historicalResult"][change] = .1 if change == "p" else {"identity": "new"}
            with self.assertRaises(ValueError):
                check_mapping(self.analysis, SPEC, OLD_SPEC, bad)
        bad = deepcopy(original)
        bad["acceptedScientificResult"] = True
        with self.assertRaises(ValueError):
            check_mapping(self.analysis, SPEC, OLD_SPEC, bad)

    def test_mapper_and_checker_do_not_call_numerical_generators(self):
        with patch("src.tutoring_analysis.nominal_inference.test_contrast", side_effect=AssertionError), \
                patch("src.tutoring_analysis.nominal_inference.participant_endpoints", side_effect=AssertionError):
            output = self.mapped()
            with patch.object(M, "map_saved_analysis", side_effect=AssertionError):
                proof = check_mapping(self.analysis, SPEC, OLD_SPEC, output)
        self.assertTrue(proof["passed"])
        for key in ("humanObservations", "humanAdapterReleased", "humanDirectionEstablished",
                    "humanPowerEstablished", "historicalResultsModified"):
            self.assertIs(output[key], False)
        for key in ("newInputs", "newPersonEndpoints", "newTests"):
            self.assertEqual(output[key], 0)
        self.assertTrue(output["noGlobalErrorGuarantee"])

    def test_portable_cli_chain_with_existing_public_generator(self):
        mapper = ROOT / "scripts/map_h6_h7_surplus.py"
        reader = ROOT / "scripts/read_h6_h7_surplus_mapping.py"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def run(script, *args, success=True):
                result = subprocess.run([sys.executable, str(script), *map(str, args)],
                    cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
                return result
            for script in (mapper, reader):
                run(script, "--help")
            raw, mapped, proof = root / "trials.json", root / "mapping.json", root / "check.json"
            run(mapper, "example", "--output", raw)
            self.assertEqual(json.loads(raw.read_text()), M.example_trials())
            run(ROOT / "scripts/reproduce_expected_gpa.py", "analyze", "--trials", raw,
                "--output", root / "analysis")
            analysis_path = root / "analysis/result.json"
            before = analysis_path.read_bytes()
            run(mapper, "map", "--analysis", analysis_path, "--output", mapped)
            run(reader, "--analysis", analysis_path, "--mapping", mapped, "--output", proof)
            self.assertTrue(json.loads(proof.read_text())["passed"])
            output = json.loads(mapped.read_text())
            historical = json.loads(before)
            old = {row["testId"]: row for row in historical["tests"]}
            for row in output["mappings"]:
                self.assertEqual(row["historicalResult"], old[row["historicalTestId"]])
                self.assertEqual(row["historicalResult"]["status"], "available")
            self.assertEqual(analysis_path.read_bytes(), before)
            mapped_before, proof_before = mapped.read_bytes(), proof.read_bytes()
            run(mapper, "map", "--analysis", analysis_path, "--output", mapped, success=False)
            run(reader, "--analysis", analysis_path, "--mapping", mapped, "--output", proof, success=False)
            self.assertEqual(mapped.read_bytes(), mapped_before)
            self.assertEqual(proof.read_bytes(), proof_before)
            output["mappings"][0]["historicalResult"]["p"] = .123
            tampered = root / "tampered.json"
            tampered.write_text(json.dumps(output))
            rejected = root / "rejected.json"
            run(reader, "--analysis", analysis_path, "--mapping", tampered, "--output", rejected, success=False)
            self.assertFalse(rejected.exists())
            historical["humanObservations"] = True
            wrong = root / "wrong.json"
            wrong.write_text(json.dumps(historical))
            run(mapper, "map", "--analysis", wrong, "--output", rejected, success=False)
            self.assertFalse(rejected.exists())

    def test_notebook_is_source_calling_and_disabled_by_default(self):
        notebook = json.loads((ROOT / "notebooks/statistical_methods/expected_gpa_and_nominal_inference.ipynb").read_text())
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertIn("'pip', 'install'", "".join(code[0]["source"]))
        for cell in code:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
        text = "\n".join("".join(cell["source"]) for cell in code)
        for fragment in ("RUN_SURPLUS = False", "reproduce_expected_gpa.py",
                         "map_h6_h7_surplus.py", "read_h6_h7_surplus_mapping.py", "'fixture'"):
            self.assertIn(fragment, text)


class HistoricalFixtureTests(unittest.TestCase):
    """One real CLI reproduction of the fixed 25-person algebra input recipe."""
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "fixture"
        cls.mapper = ROOT / "scripts/map_h6_h7_surplus.py"
        command = [sys.executable, str(cls.mapper), "fixture", "--output", str(cls.output)]
        result = subprocess.run(command, cwd=cls.root, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.rows = json.loads((cls.output / "metrics.json").read_text())
        cls.analysis = json.loads((cls.output / "result.json").read_text())
        cls.recipe = json.loads((cls.output / "recipe.json").read_text())

    def test_exact_recipe_roster_geometry_and_projection_inputs(self):
        self.assertEqual(len(self.rows), 250)
        self.assertEqual(len(self.analysis["people"]), 25)
        self.assertEqual(self.recipe, M.historical_fixture_specification())
        for position, row in enumerate(self.rows):
            arm_index, within_arm = divmod(position, 50)
            person, trial = divmod(within_arm, 10)
            main = trial + 1
            arm = ("C1", "C2", "C3", "C4", "C5")[arm_index]
            self.assertEqual((row["conditionId"], row["participantId"], row["sessionId"], row["mainIndex"]),
                             (arm, arm + "-algebra-" + str(person), arm + "-algebra-" + str(person), main))
            gap = 0. if arm == "C5" else (main - 5.5) * (1 + person / 10)
            terminal_gap = gap + (-1) ** (main + person) * (person + 1) / 5
            count = (0, 1, 3)[(main + person) % 3]
            expected_index = 500 + (main - 5) * (person + 1) * (3 if arm_index % 2 == 0 else -3) + 2 * person
            target = 500 + ((main + arm_index) % 5 - 2) * 7
            self.assertEqual(row["initialMeans"], [50. + gap, 50.])
            self.assertEqual(row["terminalMeans"], [50. + terminal_gap, 50.])
            self.assertEqual((row["paidCount"], row["remainingIntegerBudget"]), (count, 100000 - count * 1000))
            self.assertEqual((row["shareIndex"], row["jEGPA"], row["jSurplus"]),
                             (expected_index, target, target + 3))
            self.assertEqual(row["R_EGPA_pp"], (abs(expected_index - 500) - abs(expected_index - target)) / 10)
            self.assertEqual(row["R_surplus_pp"], (abs(expected_index - 500) - abs(expected_index - target - 3)) / 10)
            self.assertEqual("initialProjectionPlan" in row, arm != "C5")
            self.assertEqual(row["fixtureScope"], "nonengine_algebra_fixture;not_native_or_forward_generated")
            self.assertNotIn("nativeProfileClaim", row)
            self.assertNotIn("originReceiptHash", row)
        self.assertEqual(self.rows[0]["initialProjectionPlan"]["predictorBinary64"][0], "bfa70a3d70a3d70a")
        self.assertEqual(self.rows[0]["initialProjectionPlan"]["weightBinary64"][0], "c015d1745d1745d1")

    def test_four_original_numerical_payloads_and_mapping_cli(self):
        import numpy
        import scipy

        # Golden numbers from the original four deterministic API-test records,
        # not fitted human data or a replacement result generator.
        expected = {
            "S_surplus_C2": (-.06, .046475800154489016, 4., -1.2909944487358052,
                              .8668676860168449, [-.18903750784908213, .06903750784908214]),
            "H06_R_EGPA_C2": (-.040000000000000015, .039999999999999994, 4., -1.0000000000000004,
                               .8130495168499706, [-.15105780420791196, .07105780420791191]),
            "S_surplus_C3": (-.07600000000000003, .10457533169921096, 4., -.7267488303895426,
                              .7461853419721631, [-.3663476678207105, .21434766782071046]),
            "H07_R_EGPA_C3": (-.13599999999999995, .08930845424706443, 4., -1.522812158676123,
                               .8987650855784974, [-.3839600206470436, .11196002064704366]),
        }
        exact_runtime = (sys.version_info[:3] == (3, 10, 9)
                         and numpy.__version__ == "1.23.5" and scipy.__version__ == "1.10.0")
        tests = {row["testId"]: row for row in self.analysis["tests"]}
        for test_id, values in expected.items():
            row = tests[test_id]
            for key, value in zip(("effect", "SE", "df", "statistic"), values[:4]):
                self.assertEqual(row[key], value)
            self.assertEqual(row["armMeans"], {test_id[-2:]: values[0]})
            self.assertEqual(row["requiredArmCounts"], {test_id[-2:]: 5})
            self.assertEqual(row["status"], "available")
            self.assertFalse(row["candidateNominalReject"])
            self.assertIsNone(row["reason"])
            for actual, reference in zip([row["p"], *row["CI"]], [values[4], *values[5]]):
                if exact_runtime:
                    self.assertEqual(actual, reference)
                else:
                    # The original independent challenger's finite tolerance;
                    # this does not assert cross-version bitwise equivalence.
                    self.assertLessEqual(abs(actual - reference), 1e-10 + 1e-9 * abs(reference))
                    self.assertTrue(math.isfinite(actual))
        path = self.output / "result.json"
        original = path.read_bytes()
        mapped, proof = self.root / "historical_mapping.json", self.root / "historical_check.json"
        for command in (
            [sys.executable, str(self.mapper), "map", "--analysis", str(path), "--output", str(mapped)],
            [sys.executable, str(ROOT / "scripts/read_h6_h7_surplus_mapping.py"),
             "--analysis", str(path), "--mapping", str(mapped), "--output", str(proof)],
        ):
            result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(json.loads(proof.read_text())["passed"])
        for row in json.loads(mapped.read_text())["mappings"]:
            self.assertEqual(row["historicalResult"], tests[row["historicalTestId"]])

    def test_fixture_changes_and_overwrites_rejected(self):
        bad_recipe = deepcopy(self.recipe)
        bad_recipe["peoplePerArm"] = 4
        path = self.root / "bad_recipe.json"
        path.write_text(json.dumps(bad_recipe))
        refused = self.root / "refused_fixture"
        before = (self.output / "result.json").read_bytes()
        for options in (("--fixture-config", str(path), "--output", str(refused)),
                        ("--output", str(self.output))):
            result = subprocess.run([sys.executable, str(self.mapper), "fixture", *options],
                                    cwd=self.root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse(refused.exists())
        self.assertEqual((self.output / "result.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
