"""Finite native reconstruction, integer payoff, controls and input rejection."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from src.tutoring_analysis import native_lookup as L
from src.tutoring_analysis import native_reconstruction as I
from src.tutoring_analysis.expected_outcomes import PROFILE, ROW_SCHEMA, digest


ROOT = Path(__file__).resolve().parents[1]


def specs():
    return tuple(json.loads((ROOT / "configs" / name).read_text())
                 for name in ("expected_gpa_analysis.json", "native_calibration.json"))


def state(arm="C1", variant=0, main=1, prefix=0):
    meta = L._source_meta(arm, variant)
    definition = I.schedule(meta)[main + 3]
    streams = I.scenario(meta, definition)[1]
    paid = L._paid(definition, L._prefix(definition, main, prefix), streams)
    initial, terminal = L._independent_logs(definition, paid)
    return {"definition": definition, "conditionId": arm, "mainIndex": main,
            "scheduleVariant": variant, "paid": paid, "prefixLength": prefix,
            "initialLogs": [p.tolist() for p in initial], "terminalLogs": [p.tolist() for p in terminal],
            "remainingMillihours": definition["budgetMillihours"] - prefix * definition["costMillihours"],
            "semanticFingerprint": "synthetic-test-only"}


class NativeLookupTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is needed for the native transition interface")
    def test_clean_design_and_engine_schedule_match_independent_reconstruction(self):
        program = """
const E=require('./src/tutoring_analysis/native_engine.cjs');
const N=require('./src/tutoring_analysis/native_normalizer.cjs');
const design=require('./configs/native_design.json');
const meta=JSON.parse(process.argv[1]);
const state=E.createSession(design,{...meta,profileId:E.R9,startedAt:'2026-10-11T00:00:00.000Z'});
let rejected=false;
try { N.normalize({rr_final_meta:'not a synthetic export'},design); } catch(e) { rejected=true; }
process.stdout.write(JSON.stringify({schedule:state.schedule,rejected}));
"""
        meta = L._source_meta("C1", 0)
        result = subprocess.run(["node", "-e", program, json.dumps(meta)], cwd=ROOT,
                                text=True, capture_output=True, check=True)
        actual = json.loads(result.stdout)
        self.assertTrue(actual["rejected"])
        L._Replay().near(actual["schedule"], I.schedule(meta), "Native schedule parity")

    def test_full_schedule_roster_and_unscaled_gap(self):
        schedules = {(a, v): I.schedule(L._source_meta(a, v)) for a in L.ARMS for v in range(9)}
        for (arm, _), rows in schedules.items():
            self.assertEqual([r["stage"] for r in rows], ["practice"] * 4 + ["main"] * 10)
            self.assertEqual([r["stageIndex"] for r in rows[4:]], list(range(1, 11)))
            if arm == "C1":
                for d in rows:
                    self.assertEqual(d["priorCentres"][1] - d["priorCentres"][0], 1)
                    self.assertEqual(d["gap"], 0)
        indices = L._expected_indices(schedules, [0, .2, .5, 1])
        self.assertTrue(all(("working", a, v, i, p) in indices
                            for a in L.ARMS for v in range(9) for i in range(1, 11) for p in (0, 1, 3)))
        self.assertTrue(all(a == "C1" for population, a, _, _, _ in indices if population != "working"))

    def test_prefix_and_exhaustion_identity(self):
        d = I.schedule(L._source_meta("C1", 0))[4]
        n = d["budgetMillihours"] // d["costMillihours"]
        self.assertEqual(L._prefix(d, 1, n), [i % 2 for i in range(n)])
        with self.assertRaises(ValueError):
            L._prefix(d, 2, n)
        with self.assertRaises(ValueError):
            L._prefix(d, 1, 2)

    def test_independent_likelihood_includes_latent_dependent_normalizer(self):
        s = state("C3", prefix=3)
        initial, terminal = L._Replay().logs(s["definition"], s["paid"])
        np.testing.assert_allclose(s["initialLogs"], initial, rtol=0, atol=1e-10)
        np.testing.assert_allclose(s["terminalLogs"], terminal, rtol=0, atol=1e-10)
        self.assertNotEqual(I.channel(6)[0][0], I.channel(6)[49][49])

    def test_integer_kink_and_independent_target(self):
        table = L._independent_utility_table(120000)
        self.assertEqual(table[500, 59], 0.)
        s = state("C2", prefix=1)
        j, ties, minimum = L._independent_target(s["terminalLogs"], s["remainingMillihours"])
        target, _ = L._Replay().targets(s["terminalLogs"], s["remainingMillihours"])
        self.assertEqual(j, target["jEGPA"])
        self.assertEqual(ties, target["EO_tied_indices"])
        self.assertAlmostEqual(minimum, target["raw_EO_minimum_GPA_gap"], places=10)

    def test_conditional_controls_keep_ancestor_reports(self):
        s = state(prefix=3)
        original = copy.deepcopy(s)
        for gap in (0, .2, .5, 1):
            control = L._synthetic_control(s, gap)
            self.assertEqual(control["paid"], s["paid"])
            self.assertEqual(control["remainingMillihours"], s["remainingMillihours"])
            self.assertFalse(control["engineReplayClaim"])
            self.assertEqual(control["definition"]["c1TotalCentreGapMinutes"], gap)
        self.assertEqual(s, original)

    def test_public_row_uses_native_logs_and_uncommitted_semantics(self):
        analysis, _ = specs()
        design = json.loads((ROOT / "configs/native_design.json").read_text())
        s = state("C2", prefix=3)
        identity = {"valid": True, "meta": {"conditionId": "C2", "participantId": "p", "sessionId": "s"},
                    "common": design["common"], "analysisId": analysis["analysisId"],
                    "analysisSpecHash": digest(analysis), "sourceReceiptHash": "test-content"}
        trial = {**s, "result": {"remainingMillihours": s["remainingMillihours"], "shareIndex": None}}
        row = L._trial_metrics(trial, identity, "test-target-content")
        self.assertEqual(row["metric_schema"], ROW_SCHEMA)
        self.assertEqual(row["profileId"], PROFILE)
        self.assertFalse(row["humanObservations"])
        self.assertIsNone(row["R_EGPA_pp"])
        self.assertEqual(row["DeltaNumerator"], abs(row["jEGPA"] - 500))
        trial["result"]["remainingMillihours"] -= 1
        with self.assertRaisesRegex(ValueError, "paid-budget"):
            L._trial_metrics(trial, identity, "test-target-content")

    def test_receipts_cannot_substitute_for_source_or_lookup(self):
        analysis, calibration = specs()
        with patch.object(L.subprocess, "run") as process:
            with self.assertRaises(ValueError):
                L.validate_source({"schema": "private-receipt", "status": "accepted"}, analysis, calibration)
            process.assert_not_called()
        with self.assertRaises(ValueError):
            L.validate_lookup({"schema": L.LOOKUP_SCHEMA, "receipt": {"status": "accepted"}}, analysis, calibration)


if __name__ == "__main__":
    unittest.main()
