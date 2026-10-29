"""Portable joint/interval identities, independent reductions and refusal checks.

Small deterministic fixtures test wiring only, not population calibration.
No private data, model fitting, random input generation or R is required.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import stats

from src.tutoring_statistics import joint_interval_design, joint_interval_reproduction as runner
from src.tutoring_statistics import magnitude_design as design, saved_fit_magnitude as core
from src.tutoring_statistics import magnitude_composition, magnitude_reduction
from src.tutoring_statistics import independent_magnitude_reduction as independent
from src.tutoring_statistics import joint_reduction, independent_joint_reduction
from src.tutoring_statistics import saved_joint_reduction, independent_saved_joint_reduction
from src.tutoring_statistics import interval_reduction, independent_interval_reduction
from src.tutoring_statistics import known_covariance_reduction, independent_known_covariance_reduction
from src.tutoring_statistics import reproduction as io
from test_known_covariance import fixture


def magnitude_fixture(mode, ordinal):
    plan = design.manifest(mode)
    case = design.case_for(plan, ordinal)
    L = np.asarray(core.eight_contrasts(core.NAMES), dtype=float)
    V = np.eye(9)
    covariance = L @ V @ L.T
    refs = core.reference_rows((L @ np.arange(9, dtype=float)).tolist(),
        np.sqrt(np.diag(covariance)).tolist(), [100.]*8, L.tolist(), stats)
    method = dict(status="complete", covariance=V.tolist(), contrastCovariance=covariance.tolist(),
                  references=refs, composition=core.compose_references(refs, magnitude_composition))
    record = dict(ordinal=ordinal, caseID=case["caseID"], repID=case["repID"],
                  populationCounts=dict(C1=2, C2=2, C3=2), truth=design.truth(case),
                  methods={key: copy.deepcopy(method) for key in ("baseline", "candidate")})
    event = magnitude_reduction.compact_event(record)
    result = dict(record=record, event=event, independentEvent=copy.deepcopy(event))
    references = {name: {key: copy.deepcopy(method[key]) for key in
                         ("covariance", "contrastCovariance", "references")} for name in ("baseline", "candidate")}
    return plan, result, dict(methods=references)


def coverage_fixture(ordinal):
    plan = design.manifest("fresh_coverage_validation")
    case = design.case_for(plan, ordinal)
    targets = joint_interval_design.COVERAGE_TARGETS[case["caseIndex"]]
    refs = {key: dict(L=value["L"], truth=value["truth"], status="available", reason=None,
            estimate=value["truth"], SE=1., df=100., CI=[value["truth"]-1., value["truth"]+1.],
            covered=True, coverageNumericallyUnresolved=False) for key, value in targets.items()}
    return dict(ordinal=ordinal, repID=case["repID"], caseID=case["caseID"], caseIndex=case["caseIndex"],
        actualK=dict(C1=2, C2=2, C3=2), methods={key: dict(status="complete", references=copy.deepcopy(refs))
        for key in ("baseline", "candidate", "KR")}, oracle=dict(status="complete", references=copy.deepcopy(refs)))


class JointIntervalIdentityTests(unittest.TestCase):
    def test_three_fixed_rosters_and_independent_seed_mapping(self):
        for mode, total, width in (("fresh_joint_validation", 10645, 5),
                                   ("fresh_coverage_validation", 4258, 2), ("saved_coverage_s1", 2000, 1)):
            plan = design.manifest(mode)
            self.assertEqual((plan["expectedTotal"], len(plan["cases"])), (total, width))
            self.assertEqual(plan["expectedOrdinals"], independent.expected_ids(mode))
            ctx = dict(manifest=plan, expectedSet=set(plan["expectedOrdinals"]))
            for ordinal in plan["expectedOrdinals"][:width]+plan["expectedOrdinals"][-width:]:
                self.assertEqual(design.case_for(plan, ordinal), independent.case_for(ctx, ordinal))
            for ordinal in (-1, True, 1.5, 80000):
                with self.assertRaises(ValueError):
                    design.case_for(plan, ordinal)
        self.assertEqual(design.manifest("saved_coverage_s1")["expectedOrdinals"][-1], 79992)
        last = design.case_for(design.manifest("fresh_joint_validation"), 10644)
        self.assertEqual(last["responseSeed"], 1602610270+500000+2129)
        self.assertEqual(last["parameters"]["slopes"]["C3"]["A"], 2*.8288338510706277)
        self.assertEqual(list(v["trueNull"] for v in design.truth(last).values()), [True, True, False, False])

    def test_cli_plan_has_no_initialization_or_numerical_runtime(self):
        with patch.object(runner, "initialize") as initialize, patch.object(io, "numerical_runtime") as runtime:
            runner.main(["--mode", "joint_fresh", "--input", "not_read", "--output", "not_created", "--start", "0", "--count", "5"])
        initialize.assert_not_called()
        runtime.assert_not_called()
        with patch.object(runner, "initialize") as initialize, self.assertRaises(ValueError):
            runner.main(["--mode", "joint_fresh", "--input", "unused", "--output", "unused", "--all", "--aggregate", "unused.json"])
        initialize.assert_not_called()

    def test_descriptor_mutation_and_output_overlap_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            pin = io.write(path, dict(input=1))
            obs = runner.Observations()
            self.assertEqual(obs.read(pin), dict(input=1))
            path.write_text('{"input":2}\n')
            with self.assertRaises(ValueError):
                obs.fresh()
            with self.assertRaises(ValueError):
                io.write(path, dict(input=3))
            with self.assertRaises(ValueError):
                io.validate_output(Path(tmp) / "new", tmp)

    def test_wrong_upstream_law_is_not_accepted(self):
        manifest = dict(schema=runner.SCHEMA, plan=runner.plan("joint_fresh"), sources=[],
                        inputManifest=dict(path="synthetic", sha256="wrong"), observedInputs={})
        with patch.object(runner.Observations, "read", return_value=manifest), \
             patch.object(runner.upstream, "source_files", return_value=[]), \
             patch.object(runner.upstream, "observe_manifest", return_value=dict(plan=runner.plan("joint_saved")["inputPlan"])):
            with self.assertRaises(ValueError):
                runner.context({}, runner.Observations())

    def test_source_closure_includes_every_statistics_helper_and_data_file(self):
        root = runner.ROOT
        expected = {p.resolve() for p in root.rglob("*") if p.is_file()
                    and p.suffix in {".py", ".R", ".cjs", ".json"}
                    and "__pycache__" not in p.parts}
        actual = {Path(p["path"]).resolve() for p in runner.upstream.source_files()}
        self.assertEqual(actual, expected)
        for name in ("joint_interval_design.py", "joint_interval_reproduction.py",
                     "saved_joint_reduction.py", "independent_saved_joint_reduction.py",
                     "interval_fit.R", "data/interval_environment.json",
                     "data/numerical_environment.json"):
            self.assertIn((root / name).resolve(), actual)


class JointIntervalReductionTests(unittest.TestCase):
    def test_saved_joint_is_CR2_only_and_keeps_original_report_schema(self):
        for ordinal in (1, 15):
            plan, result, references = magnitude_fixture("saved_power_core", ordinal)
            calculated = runner.joint_analysis(result, references, saved=True)
            self.assertEqual(set(calculated["methods"]), {"candidate"})
            events = [dict({k: calculated[field][k] for k in ("ordinal", "caseID", "repID", "truth")},
                           **calculated[field]["methods"]["candidate"])
                      for field in ("event", "independentEvent")]
            first = saved_joint_reduction.reduce_events(plan, [events[0]])
            second = independent_saved_joint_reduction.reduce_events(plan, [events[1]])
            independent.compare_report(first, second)
            case = next(v for v in first if v["present"])
            self.assertEqual((case["M"], case["T"], case["A4"]), (2000, 1, 1))
            self.assertIsNone(case["family"]["F"])
            self.assertIsNone(case["family"]["D"]["rate"])
            self.assertEqual(set(case["comparisonTransitions"]), set(design.IDS))
            for reducer in (saved_joint_reduction.reduce_events, independent_saved_joint_reduction.reduce_events):
                with self.assertRaises(ValueError):
                    reducer(plan, [events[0], events[0]])

    def test_joint_independent_mapping_family_diagnostics_and_saved_denominator(self):
        for mode, ordinal in (("fresh_joint_validation", 4), ("saved_power_core", 7)):
            plan, result, references = magnitude_fixture(mode, ordinal)
            calculated = runner.joint_analysis(result, references)
            ctx = dict(manifest=plan, modules=dict(audit=design), originalReader=independent)
            first = joint_reduction.reduce_events(ctx, [calculated["event"]])
            second = independent_joint_reduction.reduce_events(ctx, [calculated["independentEvent"]])
            independent.compare_report(first, second)
            case = next(v for v in first if v["present"])
            self.assertEqual(case["methods"]["candidate"]["M"], plan["B"])
            self.assertEqual(case["paired"]["candidate"]["jointCompleted"], 1)
            self.assertFalse(calculated["independentNegativeCorrelationSolver"])
            bad = dict(calculated["event"], repID=999)
            for reduce in (joint_reduction.reduce_events, independent_joint_reduction.reduce_events):
                with self.assertRaises(ValueError):
                    reduce(ctx, [bad])
                with self.assertRaises(ValueError):
                    reduce(ctx, [calculated["event"], calculated["event"]])

    def test_all_null_discovery_is_not_zero_power(self):
        plan, result, references = magnitude_fixture("fresh_joint_validation", 0)
        calculated = runner.joint_analysis(result, references)
        self.assertIsNone(calculated["event"]["methods"]["candidate"]["events"]["D"])

    def test_oracle_hybrids_and_independent_reducer(self):
        actual, case = fixture("s1_rep1")
        oracle, checked = runner.oracle_analysis("oracle_s1", dict(case=case), actual)
        _, result, _ = magnitude_fixture("saved_coverage_s1", 32)
        events = runner.oracle_event(result, oracle, checked)
        first = known_covariance_reduction.reduce_events([events["event"]])
        second = independent_known_covariance_reduction.reduce_events([events["independentEvent"]])
        independent.compare_report(first, second)
        self.assertEqual(len(first["hybrids"]), 6)
        self.assertEqual(first["M"], 2000)
        self.assertFalse(first["complete"])
        self.assertIsNone(first["oldCoverageCountsCalculated"]["candidate"])

    def test_interval_unknowns_and_separate_Holm3(self):
        rows = [coverage_fixture(0), coverage_fixture(1)]
        rows[0]["methods"]["KR"]["references"]["C1-C2:B:D"].update(covered=None, coverageNumericallyUnresolved=True)
        for row in rows:
            independent_interval_reduction.audit_event(row)
        first = interval_reduction.reduce_rows(rows)
        second = independent_interval_reduction.reduce_independently(rows)
        independent.compare_report(first, second)
        value = first["1"]["methods"]["KR"]["C1-C2:B:D"]
        self.assertEqual(value["endpointUnknown"], 1)
        self.assertEqual(value["A"], 0)
        self.assertEqual(value["T"], 1)
        self.assertIsNone(value["fullCoverage"])
        self.assertIsNone(first["1"]["pairedKRminusCR2"]["C1-C2:B:D"]["holm3P"])
        invalid = copy.deepcopy(rows[0]); invalid["caseIndex"] = 2
        with self.assertRaises(ValueError):
            independent_interval_reduction.audit_event(invalid)

    def test_exact_mcnemar_and_endpoint_unknown_rule(self):
        rows = [coverage_fixture(2*i) for i in range(5)]
        for i, row in enumerate(rows):
            row["methods"]["candidate"]["references"]["C1-C2:B:D"]["covered"] = i >= 3
            row["methods"]["KR"]["references"]["C1-C2:B:D"]["covered"] = i != 4
        left = interval_reduction.compare_pair(rows, "C1-C2:B:D", "candidate", "KR")
        right = independent_interval_reduction.table_for(rows, "C1-C2:B:D", "candidate", "KR")
        independent.compare_report(left, right)
        self.assertEqual(left["table"], {"00": 0, "01": 3, "10": 1, "11": 1})
        ref = dict(status="available", L=[0]*9, estimate=1., SE=1., df=100., CI=[0., 2.])
        classified = runner.linear_interval(ref, 0.)
        self.assertIsNone(classified["covered"])
        self.assertTrue(classified["coverageNumericallyUnresolved"])

    def test_notebook_public_routes_disabled_and_no_saved_outputs(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/statistical_methods/magnitude_validation_and_power.ipynb").read_text())
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertIn("pip", code[0])
        self.assertTrue(all(not c.get("outputs") and c.get("execution_count") is None for c in notebook["cells"]))
        all_code = "\n".join(code)
        for mode in runner.INPUT_MODES:
            self.assertIn(mode, all_code)
        self.assertIn("RUN_JOINT_INTERVAL = False", all_code)
        self.assertIn("FULL_JOINT_INTERVAL = False", all_code)
        self.assertIn("reproduce_joint_and_interval_statistics.py", all_code)


if __name__ == "__main__":
    unittest.main()
