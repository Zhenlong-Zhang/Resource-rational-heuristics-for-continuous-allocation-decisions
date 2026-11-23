"""Exact saved-design geometry and no-refit boundaries on synthetic endpoints."""
import copy
from fractions import Fraction as Q
import unittest
from unittest.mock import patch

from src.design_validation import coefficient_diagnostics as D
from src.design_validation import coefficient_reference as R


def complete_fixture():
    """Synthetic complete roster with 66 ties, not retained scientific results."""
    rows, schedules, cells = [], [], []
    for number, (condition, price, variant) in enumerate(D.schedule_roster()):
        for main in range(1, 11):
            needs = [30+main*main//2, 10+main]
            if number < 66 and main == 1:
                needs = [20, 20]
            rows.append(dict(condition=condition, price=price, variant=variant, main=main,
                             share=.5, index=500, initial_budget_milli=100000,
                             needs_displayed=needs, logical_order=['B', 'A'],
                             rank_x=main/10, rank_y=.5, initial_ask=0, mean_paid=0,
                             terminal_action='ES', selected_rule='ES'))
        ties = int(number < 66)
        schedules.append(dict(condition=condition, price=price, schedule=variant,
                              complete=True, present_mains=list(range(1, 11)), missing_mains=[],
                              available_rows=10, missing=0, ties=ties, eligible=10-ties,
                              gap_rank=2, partial_rank=3, gap_slope=0., lower_need_partial=0.,
                              initial_mean_slope=0., initial_mean_eligible=10))
    for condition in D.CONDITIONS:
        for price in D.PRICES:
            metric = dict(mean=0., sample_sd=0., defined_schedules=9, expected_schedules=9)
            cells.append(dict(condition=condition, price=price, complete=True,
                              gap_slope=dict(metric), lower_need_partial=dict(metric)))
    return rows, dict(schema=D.T.SCHEMA, fullFiniteTarget=True, verifiedTrials=2700,
                      uncertainty='sample_sd_across_nine_fixed_schedules_ddof1',
                      schedules=schedules, conditions=cells)


class CoefficientDiagnosticsTests(unittest.TestCase):
    def test_pair_differences_and_invariants(self):
        points = [(Q(1, 10), Q(3, 10)), (Q(2, 10), Q(4, 10)),
                  (Q(3, 10), Q(8, 10)), (Q(2, 10), Q(9, 10))]
        actual = D.design_geometry(points)
        self.assertEqual(actual, R.independent_geometry(points))
        R.verify_geometry_invariants(D, points, actual)
        self.assertEqual(actual['status'], 'full_rank')

    def test_empty_constant_and_collinear(self):
        for points, ranks in (([], (0, 0)), ([(Q(1), Q(2))]*3, (1, 1)),
                              ([(Q(i), Q(2*i)) for i in range(3)], (2, 2))):
            actual = D.design_geometry(points)
            self.assertEqual(actual, R.independent_geometry(points))
            self.assertEqual((actual['gapRank'], actual['partialRank']), ranks)

    def test_display_swap_and_exact_tie(self):
        row = dict(needs_displayed=[80, 20], initial_budget_milli=100000, index=750)
        other = dict(row, needs_displayed=[20, 80], index=250)
        self.assertEqual(D.project(row), D.project(other))
        self.assertIsNone(D.project(dict(row, needs_displayed=[20, 20])))
        self.assertEqual(D.project(row)[2], Q(3, 4))

    def test_complete_no_refit_and_independent_receiving(self):
        rows, summary = complete_fixture()
        old = copy.deepcopy((rows, summary))
        with patch.object(D.T.np.linalg, 'lstsq', side_effect=AssertionError('No coefficient fit')):
            result = D.diagnose(rows, summary)
            checked = R.verify(rows, summary, result)
        self.assertEqual((rows, summary), old)
        self.assertTrue(checked['passed'])
        self.assertEqual(result['scope']['eligible'], 2634)
        self.assertEqual(result['counters']['newFitInvocations'], 0)
        self.assertEqual(result['cells'][0]['metrics']['gap_slope']['accepted']['sample_sd'], 0.)

    def test_missing_duplicate_tie_count_and_saved_sd_refused(self):
        rows, summary = complete_fixture()
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                D.diagnose(bad, summary)
        changed = copy.deepcopy(rows)
        changed[0]['needs_displayed'] = [21, 20]
        with self.assertRaises(ValueError):
            D.diagnose(changed, summary)
        changed_summary = copy.deepcopy(summary)
        changed_summary['conditions'][0]['gap_slope']['sample_sd'] = .1
        with self.assertRaises(ValueError):
            D.diagnose(rows, changed_summary)

    def test_independent_receiver_rejects_changed_geometry(self):
        rows, summary = complete_fixture()
        result = D.diagnose(rows, summary)
        result['schedules'][0]['geometry']['partialRank'] = 2
        with self.assertRaises(ValueError):
            R.verify(rows, summary, result)


if __name__ == '__main__':
    unittest.main()
