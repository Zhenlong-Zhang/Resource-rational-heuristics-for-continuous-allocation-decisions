"""Fixed roster/reduction fixtures, not simulations or method-calibration evidence.

No private results, source snapshots, random draws, model fits or RDS reads.
Synthetic scalar records test report semantics, not constituent-test validity.
"""
import copy
import unittest

from src.tutoring_statistics import independent_magnitude_reduction as reader
from src.tutoring_statistics import magnitude_design as design
from src.tutoring_statistics import magnitude_reduction as producer
from src.tutoring_statistics import power_reference


def context(manifest):
    return dict(manifest=manifest, expectedSet=set(manifest["expectedOrdinals"]))


def record(manifest, ordinal, *, unavailable=(), rejected=(), technical=False):
    case = design.case_for(manifest, ordinal)
    fixed_truth = design.truth(case)
    result = dict(ordinal=ordinal, caseID=case["caseID"], repID=case["repID"],
                  populationCounts=dict(C1=149, C2=156, C3=147), truth=fixed_truth, methods={})
    for name in ("baseline", "candidate"):
        tests, references = {}, {}
        for key in design.IDS:
            available = key not in unavailable and not technical
            status = "technical_unresolved" if technical else (
                "available" if available else "statistical_unavailable")
            reject = available and key in rejected
            tests[key] = dict(status=status,
                pMagnitude=(.001 if reject else .8) if available else None,
                holmP=(.004 if reject else 1.) if available else None,
                holmReject=None if technical else reject)
            for kind in ("D", "S"):
                target = fixed_truth[key][kind]
                references[key+":"+kind] = dict(status=status,
                    estimate=target if available else None, SE=1. if available else None,
                    df=10. if available else None,
                    CI=[target-1., target+1.] if available else None,
                    lessP=.4 if available else None, greaterP=.6 if available else None)
        result["methods"][name] = dict(status="technical_unresolved" if technical else (
            "statistical_unavailable" if unavailable else "complete"), references=references,
            composition=dict(status="technical_unresolved" if technical else "complete", tests=tests))
    return result


class MagnitudeDesignTests(unittest.TestCase):
    def test_complete_fresh_roster_and_exact_seed_endpoints(self):
        manifest = design.manifest("fresh_validation")
        self.assertEqual(manifest["B"], 2129)
        self.assertEqual(manifest["expectedTotal"], 10645)
        self.assertEqual(manifest["expectedOrdinals"], list(range(10645)))
        self.assertEqual([case["caseIndex"] for case in manifest["cases"]], [1, 2, 3, 4, 6])
        self.assertEqual(reader.expected_ids(manifest["mode"]), manifest["expectedOrdinals"])
        first, last = design.case_for(manifest, 0), design.case_for(manifest, 10644)
        self.assertEqual((last["caseIndex"], last["repID"]), (6, 2129))
        for role, base in (("retention", 1102610260), ("design", 1202610260), ("response", 1302610260)):
            self.assertEqual(first[role+"Seed"], base+100000+1)
            self.assertEqual(last[role+"Seed"], base+600000+2129)
        for ordinal in (0, 1, 2, 3, 4, 10640, 10641, 10642, 10643, 10644):
            self.assertEqual(design.case_for(manifest, ordinal), reader.case_for(context(manifest), ordinal))

    def test_fresh_truth_templates_and_generating_law(self):
        manifest = design.manifest("fresh_validation")
        expected = {
            1: ((.25, -.25), (.25, -.25), (.25, -.25)),
            2: ((.25, -.25), (-.25, .25), (-.25, .25)),
            3: ((0., 0.), (0., 0.), (0., 0.)),
            4: ((.025, -.025), (.025, -.025), (-.025, .025)),
            6: ((.25, -.25), (.25, -.25), (.5, -.5)),
        }
        for case in manifest["cases"]:
            with self.subTest(case=case["caseIndex"]):
                self.assertEqual(tuple(tuple(case["parameters"]["slopes"][arm][need] for need in ("A", "B"))
                                       for arm in design.ARMS), expected[case["caseIndex"]])
                self.assertEqual(case["view"], "initial_amount")
                self.assertEqual((case["N"], case["recruitedPerArm"], case["eligibilityProbability"]), (193, 193, .8))
                self.assertIsNone(case["fixedEligiblePerArm"])
                self.assertEqual((case["parameters"]["alpha"], case["parameters"]["sdU"]), (60., 8.))
                self.assertEqual(case["parameters"]["sigmaByArm"], {arm: 12. for arm in design.ARMS})
                expected_nulls = [True, True, False, False] if case["caseIndex"] == 6 else [True]*4
                self.assertEqual([value["trueNull"] for value in design.truth(case).values()], expected_nulls)
                self.assertEqual(design.truth(case), reader.truth(case))

    def test_saved_roster_is_original_public_cells_and_sparse_ids(self):
        manifest = design.manifest("saved_power_core")
        self.assertEqual((manifest["B"], manifest["expectedTotal"]), (2000, 16000))
        self.assertEqual(manifest["expectedOrdinals"][:8], [1, 3, 5, 7, 9, 11, 13, 15])
        self.assertEqual(manifest["expectedOrdinals"][-1], 79975)
        self.assertEqual(manifest["cases"], [case for case in power_reference.cells() if case["cellIndex"] in design.CORE])
        self.assertEqual(reader.expected_ids(manifest["mode"]), manifest["expectedOrdinals"])
        for ordinal in manifest["expectedOrdinals"][:8]+manifest["expectedOrdinals"][-8:]:
            case = design.case_for(manifest, ordinal)
            self.assertEqual(case, power_reference.case_for(ordinal, 2000))
            self.assertEqual(case, reader.case_for(context(manifest), ordinal))
            self.assertEqual(case["populationRegime"], "recruit193_mcar08")

    def test_saved_magnitude_truth_does_not_use_signed_null_labels(self):
        manifest = design.manifest("saved_power_core")
        for cell in design.CORE:
            case = design.case_for(manifest, cell-1)
            expected = [False, True, False, True] if cell in (10, 12, 14) else [False]*4
            self.assertEqual([value["trueNull"] for value in design.truth(case).values()], expected)
            changed = copy.deepcopy(case)
            changed["trueNullIDs"] = {"literal": list(design.IDS), "sign_consistent": list(design.IDS)}
            self.assertEqual(design.truth(changed), design.truth(case))
            self.assertEqual(reader.truth(changed), design.truth(case))
        self.assertAlmostEqual(design.truth(design.case_for(manifest, 15))["C1-C2:B"]["comparator"], .35)

    def test_case_and_manifest_mutation_is_not_shared(self):
        first = design.manifest("fresh_validation")
        case = design.case_for(first, 0)
        case["parameters"]["slopes"]["C1"]["A"] = 99.
        self.assertEqual(first["cases"][0]["parameters"]["slopes"]["C1"]["A"], .25)
        first["cases"][0]["parameters"]["alpha"] = 99.
        self.assertEqual(design.manifest("fresh_validation")["cases"][0]["parameters"]["alpha"], 60.)
        saved = design.manifest("saved_power_core")
        saved["cases"][0]["parameters"]["alpha"] = 99.
        self.assertEqual(design.manifest("saved_power_core")["cases"][0]["parameters"]["alpha"], 60.)

    def test_reject_optional_cells_reindexing_and_noninteger_ids(self):
        with self.assertRaises(ValueError):
            design.manifest("joint_saved_power")
        for mode, invalid in (("fresh_validation", (-1, 10645, True, 1.0)),
                              ("saved_power_core", (0, 16, 32, 79976, True, 1.0))):
            manifest = design.manifest(mode)
            for ordinal in invalid:
                with self.subTest(mode=mode, ordinal=ordinal), self.assertRaises(ValueError):
                    design.case_for(manifest, ordinal)
                with self.assertRaises(ValueError):
                    reader.case_for(context(manifest), ordinal)


class MagnitudeReductionTests(unittest.TestCase):
    def assert_reports_agree(self, manifest, rows):
        actual = producer.reduce_rows(manifest, rows)
        expected = reader.reduce_rows(context(manifest), rows)
        reader.compare_report(actual, expected)
        return actual

    def test_compact_shape_and_partial_family_has_no_A4_gate(self):
        manifest = design.manifest("fresh_validation")
        raw = record(manifest, 4, unavailable=("C1-C2:B",), rejected=("C1-C2:A", "C1-C3:A"))
        event = producer.compact_event(raw)
        independent = reader.compact_event(raw,
            {name: raw["methods"][name]["references"] for name in ("baseline", "candidate")},
            {name: raw["methods"][name]["composition"] for name in ("baseline", "candidate")})
        self.assertEqual(event, independent)
        self.assertEqual(set(event), {"ordinal", "caseID", "repID", "populationCounts", "truth", "methods"})
        for value in event["methods"].values():
            self.assertTrue(value["technicalCompleted"])
            self.assertFalse(value["allAvailable"])
            self.assertEqual(value["availableCount"], 3)
            self.assertEqual((value["R"], value["F"], value["D"]), (True, True, True))
            self.assertIsNone(value["tests"]["C1-C2:B"]["pMagnitude"])
        report = self.assert_reports_agree(manifest, [event])[-1]["methods"]["candidate"]
        self.assertEqual((report["M"], report["T"], report["A4"]), (2129, 1, 0))
        self.assertEqual(report["family"]["D"]["all"]["count"], 1)
        self.assertEqual(report["family"]["D"]["allAvailable"]["count"], 0)

    def test_all_thirteen_empty_cases_keep_truth_and_missing_denominators(self):
        total = 0
        for mode in design.MODES:
            manifest = design.manifest(mode)
            reports = self.assert_reports_agree(manifest, [])
            total += len(reports)
            for report in reports:
                self.assertEqual(report["measurement"]["M"], manifest["B"])
                self.assertEqual(report["measurement"]["missing"], manifest["B"])
                self.assertEqual(report["measurement"]["candidateT"], 0)
                method = report["methods"]["candidate"]
                self.assertIsNone(method["family"]["R"]["all"]["fullRate"])
                self.assertEqual(method["family"]["R"]["all"]["unknownEventBounds"], [0., 1.])
                for key, applies in (("F", any(value["trueNull"] for value in report["truth"].values())),
                                     ("D", any(not value["trueNull"] for value in report["truth"].values()))):
                    self.assertEqual(method["family"][key] is not None, applies)
        self.assertEqual(total, 13)

    def test_statistical_unavailable_and_technical_unknown_are_distinct(self):
        manifest = design.manifest("fresh_validation")
        rows = [producer.compact_event(record(manifest, 4, rejected=("C1-C3:A",))),
                producer.compact_event(record(manifest, 9, unavailable=design.IDS)),
                producer.compact_event(record(manifest, 14, technical=True))]
        report = self.assert_reports_agree(manifest, rows)[-1]
        value = report["methods"]["candidate"]
        self.assertEqual((value["M"], value["T"], value["A4"]), (2129, 2, 1))
        self.assertEqual((value["missing"], value["technicalUnresolved"], value["statisticalUnavailableFamilies"]), (2126, 1, 1))
        self.assertIsNone(value["family"]["D"]["all"]["fullRate"])
        self.assertEqual(value["family"]["D"]["all"]["completedRate"], .5)
        self.assertEqual(value["family"]["D"]["all"]["unknownEventBounds"], [1/2129, 2128/2129])
        self.assertEqual(report["paired"]["jointCompleted"], 2)
        self.assertIsNone(rows[-1]["methods"]["candidate"]["R"])
        self.assertFalse(rows[1]["methods"]["candidate"]["R"])

    def test_full_v6_fixture_preserves_M_T_and_A4(self):
        manifest = design.manifest("fresh_validation")
        available = producer.compact_event(record(manifest, 4))
        unavailable = producer.compact_event(record(manifest, 4, unavailable=design.IDS))
        rows = [dict(available if rep < 2129 else unavailable, ordinal=5*(rep-1)+4, repID=rep)
                for rep in range(1, 2130)]
        value = self.assert_reports_agree(manifest, rows)[-1]["methods"]["candidate"]
        self.assertEqual((value["M"], value["T"], value["A4"]), (2129, 2129, 2128))
        full = value["family"]["F"]["all"]
        available_only = value["family"]["F"]["allAvailable"]
        self.assertEqual((full["count"], full["fullRate"], available_only["count"]), (0, 0., 0))
        self.assertLess(full["fullCI"][1], available_only["CI"][1])
        self.assertEqual(value["contrasts"]["C1-C3:A"]["available"], 2128)

    def test_saved_sum_coverage_fixture_retains_adverse_pointwise_interval(self):
        manifest = design.manifest("saved_power_core")
        covered = producer.compact_event(record(manifest, 5))
        missed = copy.deepcopy(covered)
        for name in ("baseline", "candidate"):
            for key in ("C1-C2:A:S", "C1-C3:A:S"):
                missed["methods"][name]["references"][key]["covered"] = False
        rows = [dict(covered if rep <= 1879 else missed, ordinal=40*(rep-1)+5, repID=rep)
                for rep in range(1, 2001)]
        value = self.assert_reports_agree(manifest, rows)[2]["methods"]["candidate"]
        self.assertIsNone(value["family"]["F"])
        self.assertEqual((value["M"], value["T"], value["A4"]), (2000, 2000, 2000))
        for key in ("C1-C2:A:S", "C1-C3:A:S"):
            coverage = value["linearReferences"][key]["coverageAll"]
            self.assertEqual((coverage["count"], coverage["fullRate"]), (1879, .9395))
            self.assertAlmostEqual(coverage["fullCI"][1], .9495479849066164, places=12)
            self.assertLess(coverage["fullCI"][1], .95)
        self.assertEqual(value["linearReferences"]["C1-C2:A:D"]["coverageAll"]["count"], 2000)

    def test_independent_CP_endpoints_and_unknown_bounds(self):
        for counts in ((0, 0), (0, 1), (1, 1), (0, 2129), (1, 2000), (1879, 2000), (2002, 2129)):
            reader.near(producer.interval(*counts), reader.interval(*counts))
        self.assertIsNone(producer.interval(0, 0))
        self.assertEqual(producer.interval(0, 1)[0], 0.)
        self.assertEqual(reader.interval(1, 1)[1], 1.)
        self.assertGreater(reader.interval(2002, 2129)[1], .95)
        reader.compare_report(producer.rate(1, 2, 3), reader.rate(1, 2, 3))
        self.assertEqual(producer.rate(1, 2, 3)["unknownEventBounds"], [1/3, 2/3])
        for invalid in ((True, 2), (1.0, 2), (2, 1), (-1, 1)):
            with self.assertRaises(ValueError):
                producer.interval(*invalid)
            with self.assertRaises(ValueError):
                reader.interval(*invalid)

    def test_duplicate_unplanned_and_changed_identity_or_truth_rejected(self):
        manifest = design.manifest("fresh_validation")
        good = producer.compact_event(record(manifest, 4))
        for rows in ([good, good], [dict(good, ordinal=10645)], [dict(good, repID=2)]):
            with self.assertRaises(ValueError):
                producer.reduce_rows(manifest, rows)
            with self.assertRaises(ValueError):
                reader.reduce_rows(context(manifest), rows)
        changed = copy.deepcopy(good)
        changed["truth"]["C1-C2:A"]["trueNull"] = False
        with self.assertRaises(ValueError):
            producer.reduce_rows(manifest, [changed])
        with self.assertRaises(ValueError):
            reader.reduce_rows(context(manifest), [changed])
        for changed in (dict(good, ordinal=True), dict(good, caseID="unknown")):
            with self.assertRaises(ValueError):
                reader.reduce_rows(context(manifest), [changed])

    def test_paired_transitions_and_exact_count_comparison(self):
        manifest = design.manifest("fresh_validation")
        raw = record(manifest, 4)
        rejecting = record(manifest, 4, rejected=("C1-C3:A",))
        raw["methods"]["candidate"] = rejecting["methods"]["candidate"]
        report = self.assert_reports_agree(manifest, [producer.compact_event(raw)])[-1]
        self.assertEqual(report["paired"]["familyRejectionTransitions"], {"00": 0, "01": 1, "10": 0, "11": 0})
        for actual, expected in (({"count": 1.0}, {"count": 1}), ({"count": True}, {"count": 1}),
                                 ({"count": 2}, {"count": 1}), ({"F": 0}, {"F": None})):
            with self.assertRaises(ValueError):
                reader.compare_report(actual, expected)
        reader.compare_report({"rate": .5+1e-10}, {"rate": .5})

    def test_tail_thresholds_are_strict_and_keep_fixed_nulls(self):
        manifest = design.manifest("fresh_validation")
        raw = record(manifest, 0)
        raw["methods"]["candidate"]["references"]["C1-C2:A:D"]["lessP"] = .05
        report = self.assert_reports_agree(manifest, [producer.compact_event(raw)])[0]
        tails = report["methods"]["candidate"]["linearReferences"]["C1-C2:A:D"]["tails"]
        self.assertTrue(tails["less"]["trueNull"])
        self.assertTrue(tails["greater"]["trueNull"])
        self.assertEqual([row["threshold"] for row in tails["less"]["rejectionByThreshold"]], [.05, .025, .0125, .05/6, .00625])
        self.assertEqual(tails["less"]["rejectionByThreshold"][0]["all"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
