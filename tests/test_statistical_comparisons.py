"""Original comparison rosters, repaired counts, negative results and portable CLI.

Synthetic events exercise reducers, not numerical calibration. Actual RDS/CR2
entry validation is separately selected through the public CLI in its pinned
runtime; these unittest cases never launch a formal campaign.
"""
from collections import Counter
from copy import deepcopy
from contextlib import redirect_stdout
import io as text_io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tutoring_statistics import comparison_design as design
from tutoring_statistics import comparison_reduction as reduction
from tutoring_statistics import comparison_runner as runner
from tutoring_statistics import joint_generation as generation
from tutoring_statistics import cluster_inference as inference
from tutoring_statistics import reproduction as storage


def event(family, ordinal, *, unavailable=True, technical=True, candidate_covered=True):
    case = design.case_for(family, ordinal)
    if unavailable:
        raw = dict(status="unavailable", reason="design_support_unavailable")
        candidate = raw
    else:
        def fitted(covered):
            refs = {}
            for key in generation.H10_IDS:
                estimate = case["delta"][key] + (0 if covered else 10)
                refs[key] = dict(estimate=estimate, SE=1., df=20., statistic=estimate,
                    CI=[estimate - 2., estimate + 2.], coefficientNames=["fixture"], L=[1.])
            return dict(status="fitted", diagnostics=dict(usableCandidateInference=True), references=refs)
        raw, candidate = fitted(True), fitted(candidate_covered)
    branches = [generation.branch_result(case, mode, raw) for mode in generation.TAILS]
    row = dict(ordinal=ordinal, case=case, caseID=case["caseID"], repID=case["repID"],
        fitKey=design.fit_key(family, case), technicalCompleted=technical,
        technicalError=None if technical else dict(phase="CR2_covariance", message="synthetic failure"),
        baselineBranches=branches, status=raw["status"], diagnostics=raw.get("diagnostics"), failureReason=raw.get("reason"))
    if family != "baseline":
        other = [generation.branch_result(case, mode, candidate) for mode in generation.TAILS] if technical else None
        row.update(candidateBranches=other, paired=inference.paired_result(branches, other),
            candidateMetadata=None if unavailable else dict(designDiagnostics=dict(
                oracle=dict(contrastSE={k: 2. for k in generation.H10_IDS}))))
    return row


class ComparisonRosterTests(unittest.TestCase):
    def test_original_sizes_case_reps_and_boundary(self):
        expected = {"baseline": (40000, 24), "diagnostic": (480, 24), "fresh": (26000, 13)}
        for family, (total, cases) in expected.items():
            ids = [design.coordinates(family, n) for n in range(total)]
            self.assertEqual(len(ids), len(set(ids)))
            counts = Counter(index for index, _ in ids)
            self.assertEqual(len(counts), cases)
            for index, n in counts.items():
                self.assertEqual(n, 20 if family == "diagnostic" else
                    1000 if family == "baseline" and (index - 1) % 6 >= 4 else 2000)
        self.assertEqual(design.coordinates("baseline", 23999), (24, 1000))
        self.assertEqual(design.coordinates("baseline", 24000), (1, 1001))
        self.assertEqual(design.coordinates("baseline", 39999), (22, 2000))
        self.assertEqual(design.coordinates("fresh", 25999), (22, 2000))

    def test_independent_case_seeds_and_prefix_fit_reuse(self):
        for family, ordinals in (("baseline", (0, 23, 24, 479, 23999, 24000, 39999)),
                                 ("diagnostic", (0, 23, 24, 479)), ("fresh", (0, 12, 13, 25999))):
            for ordinal in ordinals:
                with self.subTest(family=family, ordinal=ordinal):
                    case = design.case_for(family, ordinal)
                    self.assertEqual(case, design.independent_case(family, ordinal))
                    self.assertEqual(case["N"], 154)
                    db, rb = (402610190, 502610190) if family == "fresh" else (202610190, 302610190)
                    self.assertEqual(case["designSeed"], db + 10000 * case["scenarioIndex"] + case["repID"])
                    self.assertEqual(case["responseSeed"], rb + 10000 * case["caseIndex"] + case["repID"])
        for ordinal in (0, 23, 24, 479):
            case = design.case_for("baseline", ordinal)
            self.assertEqual(case, design.case_for("diagnostic", ordinal))
            self.assertEqual(design.fit_key("baseline", case), design.fit_key("diagnostic", case))
        self.assertNotEqual(design.case_for("baseline", 0)["responseSeed"], design.case_for("fresh", 0)["responseSeed"])
        from tutoring_statistics.paired_reference import case_for
        self.assertNotEqual(design.case_for("fresh", 11)["responseSeed"], case_for(1)["responseSeed"])

    def test_selection_refuses_duplicates_types_and_overflow(self):
        self.assertEqual(list(design.selection("baseline", start=23, limit=2)), [23, 24])
        self.assertEqual(len(design.selection("fresh", full=True)), 26000)
        for kwargs in (dict(start=-1), dict(limit=0), dict(start=25999, limit=2),
                       dict(start=True), dict(full=True, start=1), dict(full=True, limit=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                design.selection("fresh", **kwargs)
        for ordinal in (-1, True, 480):
            with self.assertRaises(ValueError):
                design.case_for("diagnostic", ordinal)


class ComparisonReductionTests(unittest.TestCase):
    def test_corrected_M_T_are_assigned_for_full_partial_and_zero(self):
        for completed in (20, 17, 0):
            rows = [event("diagnostic", 24 * i) for i in range(completed)]
            result = reduction.reduce_rows("diagnostic", rows)
            case = result["cases"][0]
            for method in ("baseline", "candidate"):
                counts = case[method]["branches"][0]["counts"]
                self.assertEqual((counts["M"], counts["T"], counts["technicalMissing"]), (20, completed, 20 - completed))
                self.assertNotEqual(counts["M"], 40)
                self.assertIsNone(case[method]["branches"][0]["rates"]["falseRejectionAvailable"])
                if completed < 20:
                    self.assertIsNone(case[method]["branches"][0]["rates"]["falseRejectionAll"])
                else:
                    self.assertEqual(case[method]["branches"][0]["rates"]["falseRejectionAll"]["denominator"], 20)

    def test_full_480_fixture_is_exact_roster_not_new_fits(self):
        result = reduction.reduce_rows("diagnostic", [event("diagnostic", n) for n in range(480)], require_full=True)
        self.assertTrue(result["complete"])
        self.assertEqual(result["completedTotal"], 480)
        self.assertEqual(result["technicalMissing"], 0)
        self.assertTrue(all(c["T"] == c["M"] == 20 for c in result["cases"]))

    def test_technical_unknown_is_distinct_from_guard_unavailable(self):
        guard = event("diagnostic", 0, unavailable=False)
        for name in ("baselineBranches", "candidateBranches"):
            raw = dict(status="fitted", diagnostics=dict(usableCandidateInference=False), references={
                k: b["reference"] for k, b in guard[name][0]["tests"].items()})
            guard[name] = [generation.branch_result(guard["case"], mode, raw) for mode in generation.TAILS]
        guard["diagnostics"]["usableCandidateInference"] = False
        guard["paired"] = inference.paired_result(guard["baselineBranches"], guard["candidateBranches"])
        unknown = event("diagnostic", 24, technical=False)
        result = reduction.reduce_rows("diagnostic", [guard, unknown])
        case = result["cases"][0]
        self.assertEqual((case["present"], case["T"], case["technicalUnresolved"], case["technicalMissing"]), (2, 1, 1, 19))
        self.assertEqual(case["originalGuardCounts"]["false"], 1)
        self.assertEqual(case["candidate"]["branches"][0]["counts"]["A4"], 0)
        self.assertEqual(case["unresolvedRepIDs"], [2])
        self.assertNotIn(2, case["missingRepIDs"])

    def test_negative_coverage_and_fixed_W_ratios_retained(self):
        row = event("diagnostic", 0, unavailable=False, candidate_covered=False)
        result = reduction.reduce_rows("diagnostic", [row])
        self.assertEqual(len(result["negativeCoverageChanges"]), 4)
        self.assertTrue(all(v["net"] == -1 for v in result["negativeCoverageChanges"]))
        ratios = result["cases"][0]["fixedWorkingWeightSERatios"]["C1-C2:A"]
        self.assertEqual(ratios["baselineOverOracleSE"]["mean"], .5)
        self.assertEqual(ratios["CR2OverOracleSE"]["mean"], .5)
        self.assertIn("estimated_working_W_held_fixed", result["oracleScope"])

    def test_missing_full_duplicate_cross_case_and_count_corruption_refused(self):
        row = event("fresh", 0)
        with self.assertRaisesRegex(ValueError, "Full aggregate"):
            reduction.reduce_rows("fresh", [row], require_full=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reduction.reduce_rows("fresh", [row, row])
        for mutate in (lambda r: r.update(caseID="wrong"), lambda r: r["case"].update(responseSeed=1),
                       lambda r: r["baselineBranches"][0]["counts"].update(M=2),
                       lambda r: r["candidateBranches"][0]["counts"].update(F=1)):
            broken = deepcopy(row); mutate(broken)
            with self.assertRaises(ValueError):
                reduction.reduce_rows("fresh", [broken])

    def test_fresh_practical_target_never_established_from_partial(self):
        result = reduction.reduce_rows("fresh", [event("fresh", 11, unavailable=False)])
        self.assertEqual(result["expectedTotal"], 26000)
        self.assertFalse(result["complete"])
        self.assertTrue(all(not a["established"] for c in result["cases"] for a in c["practicalTarget"]))

    def test_original_tail_truth_and_four_member_padding(self):
        case = design.case_for("baseline", 4)
        refs = {k: dict(estimate=case["delta"][k], SE=.01, df=100., statistic=case["delta"][k] / .01,
                       CI=[case["delta"][k] - .02, case["delta"][k] + .02]) for k in generation.H10_IDS}
        raw = dict(status="fitted", diagnostics=dict(usableCandidateInference=True), references=refs)
        literal = generation.branch_result(case, "literal", raw)
        sign = generation.branch_result(case, "sign_consistent", raw)
        self.assertEqual(literal["tests"]["C1-C2:B"]["tail"], "less")
        self.assertEqual(sign["tests"]["C1-C2:B"]["tail"], "greater")
        self.assertEqual(literal["familySize"], 4)
        self.assertEqual(literal["coverage"], sign["coverage"])
        raw["references"]["C1-C2:A"]["SE"] = 0.
        partial = generation.branch_result(case, "literal", raw)
        self.assertEqual(partial["tests"]["C1-C2:A"]["holmPaddingP"], 1.)
        self.assertEqual(partial["familySize"], 4)
        self.assertEqual(partial["counts"]["A4"], 0)


class ComparisonInterfaceTests(unittest.TestCase):
    def test_portable_plan_and_help(self):
        env = os.environ.copy()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not_created"
            process = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_statistical_comparisons.py"),
                "--family", "fresh", "--output", str(output)], capture_output=True, text=True, env=env, check=True)
            plan = json.loads(process.stdout)
            self.assertEqual((plan["expectedTotal"], plan["selected"], plan["draws"], plan["fits"]), (26000, 1, 0, 0))
            self.assertFalse(output.exists())
        help_result = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_statistical_comparisons.py"), "--help"],
                                     capture_output=True, text=True, env=env, check=True)
        self.assertIn("--full", help_result.stdout)

    def test_execution_and_source_output_refusals(self):
        with self.assertRaisesRegex(ValueError, "requires --execute"):
            runner.main(["--family", "baseline", "--stage", "all"])
        for target in (ROOT, ROOT / "src", ROOT / "src/new", ROOT / "configs/new"):
            with self.assertRaisesRegex(ValueError, "protected"):
                runner.output_root(target)

    def test_completed_input_resume_does_not_redraw_and_changed_seed_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            case = design.case_for("baseline", 0)
            directory = cache / "inputs" / design.fit_key("baseline", case)
            manifest = dict(path="unused-manifest", sha256="unused")
            spec = dict(case=case, caseID=case["caseID"], repID=1, fitKey=case["caseID"], manifest=manifest, independentGeneration=True,
                request=storage.write(directory / "actual_request.json", dict(rows=[])),
                generation=storage.write(directory / "generation.json", dict(case=case)))
            storage.write(directory / "spec.json", spec)
            with patch.object(generation, "generate", side_effect=AssertionError("must not redraw")):
                resumed, _ = runner.prepare_member("diagnostic", 0, cache, Path(tmp) / "absent_native", manifest)
            self.assertEqual(resumed, spec)
            altered = deepcopy(spec); altered["case"]["responseSeed"] += 1
            (directory / "spec.json").write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "Changed resumed"):
                runner.prepare_member("diagnostic", 0, cache, Path(tmp), manifest)

    def test_atomic_nonoverwrite_event_hash_and_partial_input_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "value.json"
            storage.write(path, dict(x=1))
            before = path.read_bytes()
            storage.write(path, dict(x=1))
            self.assertEqual(before, path.read_bytes())
            with self.assertRaisesRegex(ValueError, "Refusing changed"):
                storage.write(path, dict(x=2))
            row = event("baseline", 0)
            manifest = dict(path="unused", sha256="unused")
            commit = dict(schema=runner.SCHEMA, family="baseline", manifest=manifest,
                          event=row, eventDigest=runner.event_digest(row), artifacts=[], independentReadback=True)
            target = root / "00000.json"
            storage.write(target, commit)
            self.assertEqual(runner.admit_event(target, "baseline", manifest), row)
            commit["event"]["case"]["responseSeed"] += 1
            target.write_text(json.dumps(commit))
            with self.assertRaisesRegex(ValueError, "commitment"):
                runner.admit_event(target, "baseline", manifest)
            case = design.case_for("baseline", 0)
            storage.write(root / "inputs" / case["caseID"] / "actual_request.json", dict(rows=[]))
            with self.assertRaisesRegex(ValueError, "Uncommitted input"):
                runner.prepare_member("baseline", 0, root, root, manifest)

    def test_support_roundoff_only_with_original_stable_rank(self):
        support = dict(arms={a: dict(participants=154, rows=1540) for a in ("C1", "C2", "C3")},
            rank=9, columns=9, fitEligible=True, reason=None, scaledSingularValues=[1.] * 9)
        rows = [None] * 4620
        changed = deepcopy(support); changed["scaledSingularValues"][0] += 2.**-52
        runner.support_equal(changed, support, rows)
        too_far = deepcopy(support); too_far["scaledSingularValues"][0] += 1e-8
        with self.assertRaisesRegex(ValueError, "rank resolution"):
            runner.support_equal(too_far, support, rows)
        unstable = deepcopy(support); unstable["scaledSingularValues"][-1] = 4620 * 2.**-52
        unstable["rank"] = 8
        with self.assertRaisesRegex(ValueError, "not stable"):
            runner.support_equal(unstable, unstable, rows)


if __name__ == "__main__":
    unittest.main()
