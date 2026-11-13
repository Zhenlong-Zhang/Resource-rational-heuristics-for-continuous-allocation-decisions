"""Purpose: verify fixed-roster true-need fits and saved-action diagnostics.

Synthetic fixtures exercise units, estimands, missingness, rank, paired C3
additivity and C5 display identity. They contain no private results or sources;
historical parity and source authentication are separate verification gates.
"""
import copy
import json
import math
import unittest

import numpy as np

from src.design_validation import true_need_diagnostics as D


def schedule(condition='C1', price='rho_0', variant=0):
    rows = []
    for m in range(1, 11):
        lo, hi = 10 + m, 30 + m * m // 2
        index = 200 - 2 * lo + 3 * hi
        rows.append({'condition': condition, 'price': price, 'variant': variant,
                     'main': m, 'index': index, 'share': index / 1000,
                     'initial_budget_milli': 100000, 'needs_displayed': [hi, lo],
                     'logical_order': ['A', 'B'], 'rank_x': m / 10,
                     'rank_y': index / 1000, 'initial_ask': 0, 'mean_paid': 0,
                     'terminal_action': 'OPT_expected_GPA', 'selected_rule': 'OPT_expected_GPA'})
    return rows


def cell(condition='C1', price='rho_0'):
    return [r for v in range(9) for r in schedule(condition, price, v)]


def c3_rows():
    rows = []
    for v in range(9):
        for price in (D.BASE_PRICE, D.MIDPOINT_PRICE):
            for r in schedule('C3', price, v):
                m = r['main']
                index = 100 + 20 * m
                if price == D.MIDPOINT_PRICE:
                    r['initial_ask'] = int(m % 2 == 0)
                    r['mean_paid'] = 2 * r['initial_ask']
                    index += 100 if r['initial_ask'] else -50
                r.update(index=index, share=index / 1000, rank_y=index / 1000)
                rows.append(r)
    return rows


class TestTrueNeedDiagnostics(unittest.TestCase):
    def test_intercepts_initial_units_and_independent_centered_gap_fit(self):
        rows = schedule()
        for r in rows:
            r['budget_milli'] = 50000
            r['need_a'], r['need_b'] = 90., 10.
        got = D.summarize(rows)['schedules'][0]
        x = np.array([(r['needs_displayed'][0] - r['needs_displayed'][1]) / 100 for r in rows])
        y = np.array([r['share'] for r in rows])
        expected_gap = np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean())
        self.assertAlmostEqual(got['gap_slope'], expected_gap)
        self.assertAlmostEqual(got['lower_need_partial'], -.2)
        self.assertEqual((got['gap_rank'], got['partial_rank'], got['eligible']), (2, 3, 10))
        changed = copy.deepcopy(rows)
        for r in changed:
            r['initial_budget_milli'] *= 2
        other = D.summarize(changed)['schedules'][0]
        self.assertAlmostEqual(other['gap_slope'], 2 * got['gap_slope'])
        self.assertAlmostEqual(other['lower_need_partial'], -.4)

    def test_display_swap_preserves_true_and_legacy_coefficients(self):
        rows, swapped = schedule(), schedule()
        for r in swapped:
            r['needs_displayed'].reverse()
            r['logical_order'].reverse()
            r['index'] = 1000 - r['index']
            r['share'] = r['index'] / 1000
        a, b = (D.summarize(values)['schedules'][0] for values in (rows, swapped))
        for metric in D.METRICS:
            self.assertAlmostEqual(a[metric], b[metric])

    def test_true_ties_missing_needs_and_initial_ties_are_distinct(self):
        rows = schedule()
        rows[0]['needs_displayed'] = [20, 20]
        rows[1]['needs_displayed'] = None
        rows[2]['needs_displayed'] = [None, 30]
        rows[3]['rank_x'] = rows[3]['rank_y'] = None
        got = D.summarize(rows)['schedules'][0]
        self.assertEqual((got['available_rows'], got['missing'], got['ties'], got['eligible']), (10, 2, 1, 7))
        self.assertEqual(got['initial_mean_eligible'], 9)
        self.assertAlmostEqual(got['lower_need_partial'], -.2)
        self.assertIsNotNone(got['initial_mean_slope'])

    def test_empty_tied_constant_and_collinear_design_ranks(self):
        for needs, ranks in ((None, (0, 0)), ([20, 20], (0, 0)), ([40, 20], (1, 1))):
            rows = schedule()
            for r in rows:
                r['needs_displayed'] = needs
            got = D.summarize(rows)['schedules'][0]
            self.assertEqual((got['gap_rank'], got['partial_rank']), ranks)
            self.assertIsNone(got['gap_slope'])
            self.assertIsNone(got['lower_need_partial'])
        rows = schedule()
        for r in rows:
            r['needs_displayed'] = [30 + r['main'], 20]
        got = D.summarize(rows)['schedules'][0]
        self.assertEqual((got['gap_rank'], got['partial_rank']), (2, 2))
        self.assertIsNotNone(got['gap_slope'])
        self.assertIsNone(got['lower_need_partial'])

    def test_legacy_slope_keeps_its_ties_and_centered_ss_floor(self):
        rows = schedule()
        x = [r['rank_x'] for r in rows]
        y = [r['rank_y'] for r in rows]
        mx, my = sum(x) / 10, sum(y) / 10
        beta = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx)**2 for a in x)
        self.assertAlmostEqual(D.summarize(rows)['schedules'][0]['initial_mean_slope'], beta)
        for r in rows:
            r['rank_x'] = 1 + r['main'] * 1e-9
        got = D.summarize(rows)['schedules'][0]
        self.assertIsNone(got['initial_mean_slope'])
        self.assertIsNotNone(got['gap_slope'])
        for r in rows:
            r['rank_x'] = r['rank_y'] = None
        got = D.summarize(rows)['schedules'][0]
        self.assertEqual(got['initial_mean_eligible'], 0)
        self.assertIsNone(got['initial_mean_slope'])
        self.assertIsNotNone(got['lower_need_partial'])

    def test_equal_nine_means_and_sample_sd_not_available_case_weighting(self):
        rows = cell()
        for r in rows:
            hi, lo = r['needs_displayed']
            index = 100 + (r['variant'] + 1) * (hi - lo)
            r.update(index=index, share=index / 1000, rank_y=index / 1000)
        got = D.summarize(rows)['conditions'][0]['gap_slope']
        self.assertAlmostEqual(got['mean'], .5)
        self.assertAlmostEqual(got['sample_sd'], math.sqrt(.075))
        self.assertEqual(got['defined_schedules'], 9)
        for r in rows[:10]:
            r['needs_displayed'] = [40, 20]
        got = D.summarize(rows)['conditions'][0]['gap_slope']
        self.assertEqual(got['defined_schedules'], 8)
        self.assertIsNone(got['mean'])
        self.assertIsNone(got['sample_sd'])

    def test_partial_schedule_and_cell_expose_exact_missing_rosters(self):
        got = D.summarize(cell()[:-1])
        schedule_row = got['schedules'][-1]
        self.assertEqual(schedule_row['missing_mains'], [10])
        self.assertEqual(schedule_row['available_rows'], 9)
        self.assertEqual(schedule_row['partial_rank'], 3)
        self.assertIsNone(schedule_row['lower_need_partial'])
        self.assertIsNone(schedule_row['initial_mean_slope'])
        condition = got['conditions'][0]
        self.assertEqual(condition['complete_schedules'], 8)
        self.assertEqual(condition['gap_slope']['defined_schedules'], 8)
        self.assertIsNone(condition['gap_slope']['mean'])
        self.assertIn(['C1', 8, 10], got['missingCoordinates']['rho_0'])
        condition = D.summarize(cell()[:-10])['conditions'][0]
        self.assertEqual(condition['missing_schedules'], [8])
        self.assertIsNone(condition['gap_slope']['sample_sd'])

    def test_full_roster_permutation_purity_and_strict_json(self):
        rows = [r for c in D.CONDITIONS for p in D.LABELS for r in cell(c, p)]
        original = copy.deepcopy(rows)
        got = D.summarize(rows, full=True)
        self.assertEqual((got['verifiedTrials'], len(got['schedules']), len(got['conditions'])), (2700, 270, 30))
        self.assertTrue(got['fullFiniteTarget'])
        self.assertEqual(got, D.summarize(reversed(rows), full=True))
        self.assertEqual(rows, original)
        self.assertEqual(json.loads(json.dumps(got, allow_nan=False)), got)
        self.assertFalse(got['rowSourceAuthenticated'])
        with self.assertRaises(ValueError):
            D.summarize(rows[:-1], full=True)

    def test_malformed_missing_foreign_duplicate_and_nonfinite_refused(self):
        row = schedule()[0]
        changes = [('condition', 'C6'), ('price', 'rho_unknown'), ('variant', True),
                   ('variant', 9), ('main', 0), ('main', 1.), ('index', True),
                   ('index', 1001), ('share', .9), ('share', float('nan')),
                   ('initial_budget_milli', 0), ('initial_budget_milli', 100000.),
                   ('initial_budget_milli', True), ('needs_displayed', [0, 20]),
                   ('needs_displayed', [1., 20]), ('needs_displayed', [True, 20]),
                   ('needs_displayed', [101, 20]), ('needs_displayed', [20]),
                   ('logical_order', ['A', 'A']), ('logical_order', 'AB'),
                   ('rank_x', float('inf')), ('rank_x', None), ('rank_x', -1.),
                   ('rank_y', .12345), ('rank_y', None), ('rank_y', float('nan')),
                   ('initial_ask', 1), ('initial_ask', True), ('mean_paid', -1),
                   ('mean_paid', .5), ('terminal_action', ''), ('terminal_action', 1),
                   ('selected_rule', ' MFN_A'), ('selected_rule', {})]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                D.summarize([{**row, key: value}])
        for key in D.FIELDS:
            bad = {k: v for k, v in row.items() if k != key}
            with self.subTest(missing=key), self.assertRaises(ValueError):
                D.summarize([bad])
        for rows in ([row, row], [None], None, {}, 'rows'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                D.summarize(rows)
        with self.assertRaises(ValueError):
            D.summarize([], full=1)

    def test_empty_input_has_no_fabricated_fits_or_counts(self):
        got = D.summarize([])
        self.assertFalse(got['fullFiniteTarget'])
        self.assertEqual(got['schedules'], [])
        self.assertEqual(got['conditions'], [])
        self.assertEqual(len(got['c3']['partial']), 9)
        self.assertEqual(len(got['c5']['partial']), 90)
        self.assertEqual(got['c5']['counts']['endpoints'], 0)
        self.assertIsNone(got['c3']['summary']['slope_change']['mean'])
        json.dumps(got, allow_nan=False)

    def test_c3_common_denominator_and_two_contributions(self):
        got = D.summarize(c3_rows())['c3']
        self.assertFalse(got['causal'])
        self.assertEqual(got['partial'], [])
        self.assertEqual(got['summary']['paired_targets'], 90)
        self.assertEqual(got['summary']['search_targets'], 45)
        for r in got['schedules']:
            self.assertAlmostEqual(r['base_slope'], .2)
            self.assertAlmostEqual(r['centered_predictor_ss'], .825)
            self.assertAlmostEqual(r['search_contribution'], .025 / .825)
            self.assertAlmostEqual(r['stop_contribution'], .0125 / .825)
            self.assertAlmostEqual(r['slope_change'], .0375 / .825)
            self.assertTrue(r['additivity_verified'])
        self.assertAlmostEqual(got['summary']['slope_change']['sample_sd'], 0.)
        self.assertEqual(got, D.summarize(c3_rows()[::-1])['c3'])

    def test_c3_uses_initial_estimand_even_for_true_ties_and_empty_subgroups(self):
        rows = c3_rows()
        for r in rows:
            r['needs_displayed'] = [20, 20]
            r['initial_ask'] = r['mean_paid'] = 0
        got = D.summarize(rows)
        self.assertTrue(all(s['gap_slope'] is None for s in got['schedules']))
        for r in got['c3']['schedules']:
            self.assertEqual(r['search_contribution'], 0.)
            self.assertAlmostEqual(r['stop_contribution'], r['slope_change'])
        for r in rows:
            if r['price'] == D.MIDPOINT_PRICE:
                r['initial_ask'] = r['mean_paid'] = 1
        for r in D.summarize(rows)['c3']['schedules']:
            self.assertEqual(r['stop_contribution'], 0.)
            self.assertAlmostEqual(r['search_contribution'], r['slope_change'])

    def test_c3_partial_unavailable_and_rank_deficient_are_not_zero_filled(self):
        rows = c3_rows()
        got = D.summarize(rows[:-1])['c3']
        self.assertEqual(got['partial'][0]['missing_midpoint_mains'], [10])
        self.assertEqual(got['summary']['available_schedules'], 8)
        self.assertIsNone(got['summary']['slope_change']['mean'])
        for r in rows:
            if r['variant'] == 0 and r['main'] == 1:
                r['rank_x'] = r['rank_y'] = None
        got = D.summarize(rows)['c3']
        self.assertEqual(got['schedules'][0]['status'], 'unavailable_initial_mean')
        self.assertEqual(got['partial'][0]['unavailable_initial_mean_mains'], [1])
        for r in rows:
            r['rank_x'] = 1.
            r['rank_y'] = r['share']
        got = D.summarize(rows)['c3']
        self.assertEqual(got['summary']['available_schedules'], 0)
        self.assertTrue(all(s['status'] == 'rank_deficient' for s in got['schedules']))
        self.assertTrue(all(s['search_contribution'] is None for s in got['schedules']))

    def test_c3_refuses_pair_predictor_identity_and_orientation_changes(self):
        for field, value in (('rank_x', .15), ('rank_y', .78),
                             ('logical_order', ['B', 'A']), ('initial_budget_milli', 120000)):
            rows = c3_rows()
            target = next(r for r in rows if r['price'] == D.MIDPOINT_PRICE and r['main'] == 1)
            target[field] = value
            if field == 'rank_y':
                target[field] = 1 - target['share']
            with self.subTest(field=field), self.assertRaises(ValueError):
                D.summarize(rows)
        rows = c3_rows()
        target = next(r for r in rows if r['price'] == D.MIDPOINT_PRICE and r['main'] == 1)
        target['rank_x'] = target['rank_y'] = None
        with self.assertRaises(ValueError):
            D.summarize(rows)

    def test_c5_saved_actions_are_distinct_from_allocation_and_selected_rules(self):
        rows = cell('C5', D.HIGH_PRICE)
        for r in rows:
            r.update(index=1000, share=1., rank_y=1., terminal_action='MFN_A', selected_rule='MFN_A')
            r['logical_order'] = ['A', 'B'] if r['main'] % 2 else ['B', 'A']
        got = D.summarize(rows)['c5']
        self.assertTrue(got['complete'])
        self.assertEqual(got['partial'], [])
        self.assertEqual(got['counts']['terminal_action_counts'], {'MFN_A': 90})
        self.assertEqual(got['counts']['selected_rule_counts'], {'MFN_A': 90})
        self.assertEqual(got['counts']['mfn_a_index_1000_logical_counts'], {'A': 45, 'B': 45})
        self.assertEqual(got['counts']['saved_action_displayed_recipient_counts'], {'first': 90})
        rows[0].update(terminal_action='OPT_expected_GPA', selected_rule='MFN_A', expected_u=1.)
        rows[1].update(terminal_action=None, selected_rule=None, expected_u=1.)
        rows[2].update(terminal_action='unrecognized', selected_rule='new_rule')
        rows[3].update(terminal_action='MFN_B', selected_rule='MFN_B', index=0, share=0., rank_y=0.)
        rows[4].update(terminal_action='ES', selected_rule='ES', index=500, share=.5, rank_y=.5)
        got = D.summarize(rows)['c5']
        counts = got['counts']
        self.assertEqual(counts['terminal_action_counts']['MFN_A'], 85)
        self.assertEqual(counts['selected_rule_counts']['MFN_A'], 86)
        self.assertEqual(counts['mfn_a_index_1000'], 85)
        self.assertEqual(counts['missing_terminal_actions'], 1)
        self.assertEqual(counts['unknown_terminal_actions'], 2)
        self.assertEqual(counts['missing_selected_rules'], 1)
        self.assertEqual(counts['unknown_selected_rules'], 2)
        self.assertEqual(counts['allocation_displayed_recipient_counts'], {'first': 88, 'second': 1, 'split': 1})
        self.assertEqual(got['endpoints'][1]['saved_action_logical_recipient'], 'unknown')
        self.assertEqual(got['endpoints'][3]['saved_action_logical_recipient'], 'A')
        self.assertFalse(got['action_identity_inferred_from_allocation'])
        self.assertEqual(D.summarize(rows[:-1])['c5']['partial'], [[8, 10]])


if __name__ == '__main__':
    unittest.main()
