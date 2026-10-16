"""Small fixed-seed functionality/parity, NOT the frozen full calibration."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutoring_analysis.robustness import (CELL_IDS, aggregate_batch, combine_batches,
    core_statistics, generate_batch, resolved_cells, scalar_parity, vector_bounded_contrast)
from tutoring_analysis.inference import POLICIES


class BoundedRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.config = json.loads((root/"configs/tutoring_bounded_robustness_v01.json").read_text())
        cls.cells = resolved_cells(cls.config)

    def test_literal_truth_roster_and_no_precision_override(self):
        self.assertEqual([c["id"] for c in self.cells], list(CELL_IDS))
        self.assertEqual(sum(c["replications"] for c in self.cells), 100000)
        selected = self.cells[-1]["analyticTruths"]["selected"]["C4"]
        self.assertEqual(selected["Q"], 0); self.assertEqual(selected["R"], 15); self.assertAlmostEqual(selected["L"], .225)
        self.assertEqual(self.cells[-1]["analyticTruths"]["original"]["C4"]["R"], 0)
        for key, value in (("batchSize", 251), ("replicationsPerCell", 100001), ("alpha", .06)):
            bad = copy.deepcopy(self.config); bad[key] = value
            with self.assertRaises(ValueError): resolved_cells(bad)
        bad = copy.deepcopy(self.config); bad["scalarParity"]["absoluteTolerance"] = .01
        with self.assertRaises(ValueError): resolved_cells(bad)

    def test_all_generators_ranges_replay_and_independent_arms(self):
        for cell in self.cells:
            batch = generate_batch(cell, 0, 7); replay = generate_batch(cell, 0, 7)
            for cid in batch["groups"]:
                for key, bounds in (("Q", (0, 1)), ("R", (-50, 50)), ("L", (-.5, .5))):
                    a, b = batch["groups"][cid][key]["values"], replay["groups"][cid][key]["values"]
                    np.testing.assert_equal(a, b)
                    self.assertTrue(np.all(np.isnan(a) | ((a >= bounds[0]) & (a <= bounds[1]))))
            if cell["people"]["C2"] == cell["people"]["C3"]:
                self.assertFalse(np.array_equal(batch["raw"]["C2"]["ownU"], batch["raw"]["C3"]["ownU"]))

    def test_all_policy_scalar_and_matched_production_parity(self):
        for cell in self.cells:
            batch = generate_batch(cell, 0, 3); statistics = core_statistics(batch["groups"])
            self.assertTrue(np.all(~statistics["bounded"]["available"][:, 2:5]))
            for policy in POLICIES:
                self.assertEqual(scalar_parity(cell, batch, statistics, policy, 0)["status"], "PASS")

    def test_mcar_complete_records_and_mnar_selected_target(self):
        mcar = generate_batch(self.cells[-2], 0, 25)
        for cid, raw in mcar["raw"].items():
            self.assertTrue(np.all(~raw["includedPeople"] | (raw["recordCounts"] == 10)))
        mnar = generate_batch(self.cells[-1], 0, 5)
        for cid, raw in mnar["raw"].items():
            self.assertTrue(np.all(~raw["includedPeople"] | (raw["ownU"] > .75)))
            q = mnar["groups"][cid]["Q"]["values"]
            self.assertTrue(np.all(np.isnan(q) | (q == 0)))
        stats = core_statistics(mnar["groups"])["bounded"]
        np.testing.assert_equal(stats["originalTruth"][[5, 6, 7, 8]], 0)
        np.testing.assert_allclose(stats["selectedTruth"][[5, 6, 7, 8]], [15, 15, .225, .225])

    def test_attempted_denominators_and_no_six_slot_promotion(self):
        cell = self.cells[3]; batch = generate_batch(cell, 0, 25); stats = core_statistics(batch["groups"])
        record = aggregate_batch(cell, batch, stats, self.config); summary = combine_batches(cell, [record], self.config)
        self.assertEqual(summary["attempts"], 25)
        for policy in summary["policies"].values():
            self.assertEqual(policy["allFixedNineReject"]["numerator"], 0)
            self.assertEqual(policy["selectedTrueNullFalseEvent"]["denominator"], 25)
            self.assertIsNone(policy["familyFalseEvents"]["responsiveness"]["rate"])
        self.assertEqual(len(summary["routes"]), 9)
        self.assertEqual(sum(r["methodUnsupported"] for r in summary["routes"]), 3)
        with self.assertRaises(ValueError): combine_batches(cell, [record, record], self.config)

    def test_no_alternative_discovery_and_bound_corruption(self):
        cell = self.cells[0]; batch = generate_batch(cell, 0, 3); stats = core_statistics(batch["groups"])
        summary = combine_batches(cell, [aggregate_batch(cell, batch, stats, self.config)], self.config)
        self.assertIsNone(summary["policies"]["global_holm"]["anyTrueAlternativeDiscovery"]["rate"])
        for bad_value in (51, float("inf")):
            bad = copy.deepcopy(batch["groups"]); bad["C2"]["R"]["values"][0, 0] = bad_value
            with self.assertRaises(ValueError): vector_bounded_contrast(bad, "EOResemblance", {"C2": 1.})


if __name__ == "__main__": unittest.main()
