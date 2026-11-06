"""Purpose: fixed-outcome identity, rank estimands, complete-roster and CLI checks.

Fixtures test software invariants; retained full-reference parity is checked
separately without repeating the policy campaign.
"""
import copy
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from src.design_validation import personal_time_diagnostics as D
from src.design_validation import personal_time_sensitivity as S

ROOT = Path(__file__).resolve().parents[1]


def rows_for(c='C3', label='rho_0.001'):
    return [{'condition': c, 'variant': v, 'main': m, 'price': label,
             'rank_x': float(m), 'rank_y': .1*m,
             **{k: -float(v) if k.endswith('_u') else 0. for k in D.FIELDS}}
            for v in range(9) for m in range(1, 11)]


def endpoint_fixture():
    cfg = {'candidateId': 'fixture', 'task': {'alpha': .5, 'loss_multiplier': 2.,
           'gpa_baseline': 3.5, 'gpa_scale': .03, 'total_millihours': 120000}}
    root = {'conditionId': 'C3', 'scheduleVariant': 0, 'mainIndex': 1,
            'initialLogs': [[-math.log(100)]*100 for _ in range(2)],
            'logicalToDisplayed': ['A', 'B'], 'definition': {'budgetMillihours': 120000}}
    means = [D.P.G.finite_mean(logs) for logs in root['initialLogs']]
    ctx = {'identity': D.P.stable_identity(cfg, S.RECIPE['seed'], S.RECIPE['populationId'], 'C3', 0, 0, 1),
           'bindings': {k: S.RECIPE[k] for k in ('populationId', 'randomizationGroup')}}
    stop = {'action': 'ES', 'index': 500, 'legacySelection': {'allocation_classes':
            {'500': ['ES', 'EO_raw_surplus', 'OPT_expected_GPA'], '0': ['AL']}}}
    payload = {'context': ctx, 'finalState': {'budget': 120000, 'queryIndex': 0},
               'terminal': {'budget': 120000, 'intendedIndex': 500, 'action': 'ES',
                            'realizedCombinedGPA': None, 'actionExpectedCombinedGPA': 7.1},
               'metrics': {'choiceAvailable': True, 'remainingIntegerBudget': 120000,
                           'initialIntegerBudget': 120000, 'shareIndex': 500, 'share': .5,
                           'paidCount': 0, 'Q': 0, 'jSurplus': 500, 'jAL': 0,
                           'initialMeans': means, 'rawInitialGap': 0., 'x0': 0.},
               'events': [{'name': 'decision_000000', 'value': {'value': {'stop': stop}}}],
               'endpointKind': 'expected_only', 'initialDecision': {'action': 'ES'}}
    return root, cfg, payload


class TestAllocationDiagnostics(unittest.TestCase):
    def test_row_refuses_identity_budget_index_and_donor_changes(self):
        root, cfg, original = endpoint_fixture()
        result = D.row(root, 'rho_0.001', cfg, original)
        self.assertFalse(result['historicalEventClaimed'])
        for section, field, value in (
            ('identity', 'trialId', 'foreign'), ('identity', 'mainIndex', 2),
            ('terminal', 'budget', 119000), ('terminal', 'intendedIndex', 499),
            ('metrics', 'initialMeans', [10., 90.]), ('metrics', 'share', .6),
        ):
            modified = copy.deepcopy(original)
            target = modified['context']['identity'] if section == 'identity' else modified[section]
            target[field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                D.row(root, 'rho_0.001', cfg, modified)

    def test_row_trajectory_latents_are_target_specific(self):
        root, cfg, original = endpoint_fixture()
        draws = D.target_latents(root, original['context'])
        original['events'] += [{'name': 'latent_'+k, 'value': v} for k, v in draws.items()]
        original['endpointKind'] = 'realized_trajectory'
        original['terminal']['realizedCombinedGPA'] = D.fixed_utility(
            120000, 500, [draws[k]['category'] for k in ('A', 'B')], cfg)[0]
        D.row(root, 'rho_0.001', cfg, original)
        for field, value in (('category', 0), ('key', draws['B']['key']),
                             ('uniformHex', (0.).hex()), ('roundingFallback', True)):
            modified = copy.deepcopy(original)
            modified['events'][1]['value'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                D.row(root, 'rho_0.001', cfg, modified)

    def test_fixed_native_arithmetic_and_independent_scalar(self):
        cfg = {'task': {'alpha': .5, 'loss_multiplier': 2., 'gpa_baseline': 3.5, 'gpa_scale': .03}}
        for budget in (1000, 60000, 177450):
            for index in (0, 370, 500, 1000):
                for needs in ([1, 100], [41, 55]):
                    got = D.fixed_utility(budget, index, needs, cfg)
                    ref = D.fixed_utility(budget, index, needs, cfg, independent=True)
                    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)
                    self.assertEqual(got[1], (got[0]-7.)/.03)

    def test_recipient_swap_preserves_total_not_individual_assignment(self):
        cfg = {'task': {'alpha': .5, 'loss_multiplier': 2., 'gpa_baseline': 3.5, 'gpa_scale': .03}}
        a = D.fixed_utility(120000, 300, [20, 70], cfg)
        b = D.fixed_utility(120000, 700, [70, 20], cfg)
        self.assertEqual(a, b)
        self.assertNotEqual(a, D.fixed_utility(120000, 300, [70, 20], cfg))

    def test_refuse_invalid_budget_index_or_needs(self):
        cfg = {'task': {}}
        for b, j, ns in ((0, 500, [10, 20]), (True, 500, [10, 20]),
                         (100, 1001, [10, 20]), (100, 500, [0, 20])):
            with self.assertRaises(ValueError):
                D.fixed_utility(b, j, ns, cfg)

    def test_logical_key_and_initial_law_orientation(self):
        ctx = {'identity': {'seed': 20261019, 'conditionId': 'C3', 'variant': 0, 'replicateId': 0, 'mainIndex': 1},
               'bindings': {k: S.RECIPE[k] for k in ('populationId', 'randomizationGroup')}}
        logs = [[-math.inf]*100 for _ in range(2)]
        logs[0][9], logs[1][89] = 0., 0.
        root = {'initialLogs': logs, 'logicalToDisplayed': ['B', 'A']}
        a = D.target_latents(root, ctx)
        b = D.target_latents(root, ctx, independent=True)
        self.assertEqual(a, b)
        self.assertEqual((a['A']['category'], a['B']['category']), (90, 10))
        reversed_root = {'initialLogs': logs[::-1], 'logicalToDisplayed': ['A', 'B']}
        self.assertEqual(a, D.target_latents(reversed_root, ctx))
        self.assertNotEqual(a['A']['keySHA256'], a['B']['keySHA256'])

    def test_schedule_intercept_and_negative_utilities(self):
        values = rows_for()
        for r in values:
            r['rank_y'] += r['variant']*100
        result = D.summarize(values)
        self.assertAlmostEqual(result['conditions'][0]['beta_high_need'], .1)
        self.assertEqual(result['conditions'][0]['realized_u'], -4.)
        self.assertFalse(result['fullFiniteTarget'])
        ref = D.summarize(values, independent=True)
        self.assertAlmostEqual(ref['conditions'][0]['beta_high_need'], .1)

    def test_rank_ties_and_constant_predictor_are_undefined(self):
        values = rows_for('C5')
        for r in values:
            r['rank_x'] = r['rank_y'] = None
        cell = D.summarize(values)['conditions'][0]
        self.assertIsNone(cell['beta_high_need'])
        self.assertEqual(cell['defined_schedules'], 0)
        self.assertEqual(cell['rank_eligible'], 0)
        self.assertIsNone(D.slope([(1., 0.), (1., 1.)]))

    def test_one_undefined_schedule_is_not_dropped(self):
        values = rows_for()
        for r in values[:10]:
            r['rank_x'] = 1.
        result = D.summarize(values)['conditions'][0]
        self.assertEqual(result['defined_schedules'], 8)
        self.assertIsNone(result['beta_high_need'])

    def test_partial_schedule_never_contributes_a_condition_mean(self):
        result = D.summarize(rows_for()[:-1])
        self.assertEqual(len(result['persons']), 8)
        self.assertEqual(result['conditions'], [])
        self.assertEqual(result['partialPersons'][0]['present'], 9)

    def test_duplicate_foreign_and_nonfinite_rows_refused(self):
        values = rows_for()
        for altered in (values+[values[0]], [{**values[0], 'main': 11}],
                        [{**values[0], 'expected_u': float('nan')}],
                        [{**values[0], 'rank_x': None}]):
            with self.assertRaises(ValueError):
                D.summarize(altered)

    def test_full_and_permutation(self):
        values = [r for c in ('C1', 'C2', 'C3', 'C4', 'C5')
                  for label, _ in S.RATES for r in rows_for(c, label)]
        full = D.summarize(values, full=True)
        self.assertTrue(full['fullFiniteTarget'])
        self.assertEqual((len(full['persons']), len(full['conditions'])), (270, 30))
        self.assertEqual(full, D.summarize(values[::-1], full=True))
        with self.assertRaises(ValueError):
            D.summarize(values[:-1], full=True)

    def test_real_help_without_numerical_import(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT/'scripts/analyze_personal_time_allocations.py'),
                                 '--help'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--figures', result.stdout)

    def test_configuration_identity(self):
        self.assertEqual(D.I.read(ROOT/'configs/personal_time_diagnostics.json'), D.RECIPE)


if __name__ == '__main__':
    unittest.main()
