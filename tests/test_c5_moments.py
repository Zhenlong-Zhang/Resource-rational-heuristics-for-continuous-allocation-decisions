"""Independent finite-grid law, G, signed ranges and original EB-floor checks."""
import copy
from fractions import Fraction as F
import unittest

import numpy as np

from src.design_validation import c5_moments as P
from src.design_validation import c5_reference as R
from src.design_validation.bounded_task import normalize_log_weights


def roots():
    counts = [P.GRID//4]*4 + [0]*96
    return [dict(rootKey=['C5', 0, main], budget=100000+1000*main,
                 laws=[dict(counts=list(counts)), dict(counts=list(counts))])
            for main in range(1, 11)]


class C5MomentTests(unittest.TestCase):
    def test_strict_cdf_grid_and_exact_real_distinction(self):
        logs = np.zeros(100)
        actual = P.finite_law(logs, np, normalize_log_weights)
        self.assertEqual(actual, R.finite_law(logs, np, normalize_log_weights))
        self.assertEqual(sum(actual['counts']), P.GRID)
        self.assertTrue(actual['lastCategoryIncludesFallback'])
        self.assertGreater(len(set(actual['counts'])), 1)  # 100 does not divide 2^53.
        half = np.full(100, -np.inf)
        half[:2] = np.log(.5)
        exact = P.finite_law(half, np, lambda x: x)
        self.assertEqual(exact, R.finite_law(half, np, lambda x: x))
        self.assertEqual(exact['counts'], [P.GRID//2, P.GRID//2] + [0]*98)

    def test_final_fallback_and_invalid_law(self):
        logs = np.full(100, -np.inf)
        logs[0] = -1e-14
        actual = P.finite_law(logs, np, lambda x: x)
        self.assertEqual(actual, R.finite_law(logs, np, lambda x: x))
        self.assertGreater(actual['counts'][-1], 0)
        with self.assertRaises(ValueError):
            P.finite_law(np.zeros(99), np, normalize_log_weights)
        for bad in ([1]*100, [P.GRID]+[0]*98, [-1, P.GRID+1]+[0]*98):
            with self.assertRaises(ValueError):
                P.validate_counts(bad)

    def test_two_matrix_algorithms_and_tie_ranges(self):
        fixture = roots()
        actual = P.design_payload(fixture)
        self.assertEqual(actual, R.design_reference(fixture))
        self.assertEqual(actual['partial_low']['status'], 'available')
        fixture[0]['laws'] = [dict(counts=[P.GRID]+[0]*99)]*2
        actual = P.design_payload(fixture)
        self.assertEqual(actual, R.design_reference(fixture))
        self.assertEqual(actual['gap']['rootRanges'][0]['singlePrice'], [P.rational(0)]*2)
        self.assertEqual(P.inverse([[F(1), F(1)], [F(1), F(1)]]), (None, 0))
        with self.assertRaises(ValueError):
            P.design_payload(fixture[:-1])

    def test_signed_not_absolute_bounds(self):
        counts = [P.GRID//2, P.GRID//2]+[0]*98
        inv = [[F(2), F(-1)], [F(-1), F(1)]]
        actual = P.root_range(counts, counts, 2000, inv, 'gap')
        self.assertEqual(actual['weightMinimum'], P.rational(F(-1, 2)))
        self.assertEqual(actual['singlePrice'], [P.rational(F(-1, 2)), P.rational(0)])
        self.assertEqual(actual['pairedDifference'], [P.rational(F(-1, 2)), P.rational(F(1, 2))])

    def test_original_floor_enclosure_and_negative_result(self):
        lo, hi = P.log_interval(800)
        rlo, rhi = R.log_reference(800)
        self.assertLessEqual(lo, rlo)
        self.assertGreaterEqual(hi, rhi)
        design = P.design_payload(roots())
        designs = [dict(payload=copy.deepcopy(design)) for _ in range(9)]
        summary = P.summarize_design(designs, True)
        R.verify_summary(summary, [d['payload'] for d in designs], True)
        self.assertEqual(summary['activeJ'], 2)
        self.assertEqual(summary['highPriceSlopes'], [P.rational(0)]*2)
        self.assertFalse(summary['floorFeasible'])
        self.assertFalse(summary['policyDrawsReleased'])
        self.assertFalse(summary['precisionEstablished'])
        for row in summary['ranges']:
            self.assertEqual([f['replicates'] for f in row['zeroVarianceFloors']], list(P.LOOKS))
            self.assertTrue(all(f['status'] == 'unattainable' for f in row['zeroVarianceFloors']))
        changed = copy.deepcopy(summary)
        changed['floorFeasible'] = True
        with self.assertRaises(ValueError):
            R.verify_summary(changed, [d['payload'] for d in designs], True)

    def test_incomplete_or_nonexchangeable_is_not_zero(self):
        value = P.summarize_design([], False)
        self.assertIsNone(value['highPriceSlopes'])
        self.assertIsNone(value['floorFeasible'])
        R.verify_summary(value, [], False)
        designs = [dict(payload=P.design_payload(roots())) for _ in range(9)]
        value = P.summarize_design(designs, False)
        self.assertIsNone(value['activeJ'])
        self.assertEqual(value['ranges'][0]['zeroVarianceFloors'], [])
        R.verify_summary(value, [d['payload'] for d in designs], False)


if __name__ == '__main__':
    unittest.main()
