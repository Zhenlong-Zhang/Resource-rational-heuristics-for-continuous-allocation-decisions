"""Purpose: fixed-card arithmetic, independent checks and refusal on synthetic inputs.

These fixtures do not run adaptive paths or establish original-root agreement.
"""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from src.design_validation.strategy_cards import fixed as P
from src.design_validation.strategy_cards import fixed_reference as R


def root(budget=126000, cost=1260, displays=("50.5", "50.5")):
    return {
        "id": ["C1", 0, 1],
        "task": {"total_millihours": budget, "cost_millihours": cost,
                 "loss_multiplier": 2., "gpa_baseline": 3.5, "gpa_scale": .03,
                 "alpha": .5, "allocation_denominator": 1000, "report_width": 6.},
        "logsHex": [[0.0.hex()] * 100 for _ in range(2)],
        "display": {"root": list(displays), "orderEquivalent": True,
                    "singleReport": [[value] * 100 for value in displays]},
        "orderProof": copy.deepcopy(P.ORDER_PROOF),
    }


class FixedCardTests(unittest.TestCase):
    def test_fresh_four_cards_ties_and_independent_reader(self):
        inp = root()
        before = copy.deepcopy(inp)
        result = P.evaluate(inp, fixed=False)
        self.assertEqual(inp, before)
        self.assertEqual([r["indices"] for r in result["rows"]], [[500]] * 4)
        self.assertTrue(all(r["expectedQuestions"] == r["expectedTutoringMinutes"] == 0
                            for r in result["rows"]))
        check = R.verify(inp, result)
        self.assertEqual(check["independentlyRescoredGPARows"], 4)
        self.assertEqual(check["unresolvedCards"], [])
        self.assertEqual(check["GPAIntervalScope"], "payoff_support")
        self.assertLessEqual(check["maximumAbsoluteGPADisagreement"], 2e-11)

    def test_display_ranks_and_mirror(self):
        inp = root(displays=("40.0", "60.0"))
        first = P.evaluate(inp, fixed=False)
        self.assertEqual([r["indices"] for r in first["rows"][2:]], [[0], [1000]])
        R.verify(inp, first)
        mirrored = root(displays=("60.0", "40.0"))
        second = P.evaluate(mirrored, fixed=False)
        R.verify(mirrored, second)
        for left, right in zip(first["rows"], second["rows"]):
            self.assertEqual(left["indices"], sorted(1000 - j for j in right["indices"]))
            self.assertAlmostEqual(left["expectedGPA"], right["expectedGPA"], places=11)

    def test_rejects_private_cache_including_empty(self):
        for cache in ({}, {"500": {"directUnpricedG": 123.}}):
            inp = root()
            result = P.evaluate(inp, fixed=False)
            inp["oldScores"] = cache
            with self.subTest(cache=cache), self.assertRaisesRegex(ValueError, "oldScores"):
                P.evaluate(inp, fixed=False)
            with self.assertRaisesRegex(ValueError, "oldScores"):
                R.verify(inp, result)

    def test_complete_fixed_two_integral_and_exact_debits(self):
        inp = root()
        result = P.evaluate(inp)
        self.assertEqual(len(result["rows"]), 5)
        self.assertEqual(result["fixedEvidence"]["remainingBudget"], 123480)
        self.assertEqual(result["fixedEvidence"]["ambiguousPairCount"], 0)
        self.assertEqual(result["rows"][4]["expectedQuestions"], 2)
        self.assertEqual(result["rows"][4]["expectedTutoringMinutes"], 2.52)
        self.assertEqual(R.verify(inp, result)["independentlyRescoredGPARows"], 5)

    def test_skip_both_does_not_construct_report_law(self):
        inp = root(budget=1500, cost=1000)
        with patch.object(P, "shapes", side_effect=AssertionError("No reports")):
            result = P.evaluate(inp)
        self.assertEqual(result["rows"][4], dict(result["rows"][1], card=P.CARDS[4]))
        self.assertEqual(result["fixedEvidence"], dict(
            orderMasses=[], remainingBudget=1500, skippedBoth=True))
        with patch.object(R, "weights", side_effect=AssertionError("No reports")):
            self.assertEqual(R.verify(inp, result)["independentlyRescoredGPARows"], 4)

    def test_exact_zero_remaining_budget(self):
        inp = root(budget=2000, cost=1000)
        result = P.evaluate(inp)
        self.assertEqual(result["fixedEvidence"]["remainingBudget"], 0)
        self.assertEqual(result["rows"][4]["expectedQuestions"], 2)
        self.assertEqual(result["fixedEvidence"]["ambiguousPairCount"], 0)
        R.verify(inp, result)

    def test_unresolved_root_target_is_not_filled_by_skip_both(self):
        inp = root(budget=1500, cost=1000)
        with patch.object(P, "similar", return_value=(500, 500, True)):
            result = P.evaluate(inp)
        self.assertIsNone(result["rows"][1]["expectedGPA"])
        self.assertIsNone(result["rows"][4]["expectedGPA"])
        with patch.object(R, "targets", return_value=(500, True)):
            check = R.verify(inp, result)
        self.assertEqual(check["unresolvedCards"], [P.CARDS[1], P.CARDS[4]])
        self.assertEqual(check["independentlyRescoredGPARows"], 3)

    def test_order_display_and_input_rejections(self):
        for mutate in (
            lambda r: r["orderProof"].update(weights=["1", "0"]),
            lambda r: r["display"].update(orderEquivalent=False),
            lambda r: r["display"].update(root=["50.50", "50.5"]),
            lambda r: r["display"].update(singleReport=[["50.5"] * 99] * 2),
            lambda r: r["logsHex"][0].__setitem__(0, "nan"),
            lambda r: r["task"].update(total_millihours=True),
        ):
            inp = root()
            mutate(inp)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                P.evaluate(inp)

    def test_reader_rejects_value_roster_charge_and_provenance_changes(self):
        inp = root()
        result = P.evaluate(inp)
        for mutate in (
            lambda r: r.update(id=["C1", 0, 2]),
            lambda r: r["rows"].__setitem__(1, copy.deepcopy(r["rows"][0])),
            lambda r: r["rows"][0].update(expectedGPA=r["rows"][0]["expectedGPA"] + 1e-5),
            lambda r: r["rows"][0].update(valueSources=[{"source": "historical_reuse"}]),
            lambda r: r["rows"][4].update(expectedTutoringMinutes=0),
            lambda r: r["fixedEvidence"].update(remainingBudget=126000),
            lambda r: r["fixedEvidence"].update(orderMasses=[1., .9]),
            lambda r: r["fixedEvidence"].update(ambiguousPairCount=1),
            lambda r: r["rows"][0].update(expectedGPAInterval=["7", "7"]),
        ):
            changed = copy.deepcopy(result)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                R.verify(inp, changed)

    def test_ambiguous_supported_pair_survives_probability_underflow(self):
        inp = root()
        p = np.full(100, 1e-200)
        p[0] = 1.
        prior = np.full(100, .01)
        law = (prior, p, np.tile(prior, (100, 1)))
        unknown = np.zeros((100, 100), dtype=bool)
        unknown[1, 1] = True
        indices = np.full((100, 100), 500, dtype=np.int64)
        original_similar, original_targets = P.similar, R.targets

        def producer_target(a, b, budget, loss):
            if np.ndim(a) == 0:
                return original_similar(a, b, budget, loss)
            return indices, indices, unknown

        def reader_target(a, b, budget, loss):
            if np.ndim(a) == 0:
                return original_targets(a, b, budget, loss)
            return indices, unknown

        with patch.object(P, "shapes", return_value=law), \
                patch.object(P, "similar", side_effect=producer_target):
            result = P.evaluate(inp)
        self.assertEqual(result["fixedEvidence"]["unresolvedMass"], 0.)
        self.assertEqual(result["fixedEvidence"]["ambiguousPairCount"], 1)
        self.assertEqual(result["rows"][4]["status"], "unresolved")
        self.assertIsNone(result["rows"][4]["expectedGPA"])
        with patch.object(R, "weights", return_value=law), \
                patch.object(R, "targets", side_effect=reader_target):
            check = R.verify(inp, result)
            self.assertEqual(check["unresolvedCards"], [P.CARDS[4]])
            changed = copy.deepcopy(result)
            changed["fixedEvidence"]["ambiguousPairCount"] = 0
            changed["rows"][4].update(status="scored", expectedGPA=7.)
            with self.assertRaises(ValueError):
                R.verify(inp, changed)


if __name__ == "__main__":
    unittest.main()
