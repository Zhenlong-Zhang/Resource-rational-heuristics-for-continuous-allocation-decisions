"""Purpose: frozen proxy identity, portable CLI, independent math and refusal tests.

Synthetic fixtures establish software invariants, not historical numerical
agreement. A separately scoped entry check compares original source/evidence.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.design_validation import personal_time_inputs as I
from src.design_validation import personal_time_sensitivity as S
from src.design_validation import personal_time_sensitivity_streams as P
from src.design_validation import personal_time_sensitivity_reference as R
from src.design_validation import personal_time_sensitivity_reduction as D

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'scripts/reproduce_personal_time_sensitivity.py'


def fixture():
    context = S.load_context(ROOT / 'configs/personal_time_sensitivity.json')
    cfg = I.materialize(context['reference']['base'], context['reference']['instance'])
    logs = [[-math.log(100.)]*100 for _ in range(2)]
    cfg['state'] = {'budget': cfg['task']['total_millihours'], 'logs': copy.deepcopy(logs)}
    root = {'conditionId': 'C1', 'scheduleVariant': 0, 'mainIndex': 1,
            'definition': {'budgetMillihours': cfg['task']['total_millihours']},
            'initialLogs': logs, 'logicalToDisplayed': ['B', 'A']}
    return context, root, cfg


def metric(main, variant=0, constant=False):
    return {'mainIndex': main, 'Q': 0, 'paidCount': 0, 'expectedFinalUtility': -13.+variant,
            'expectedCombinedGPA': 6.6, 'expectedNetObjective': 6.5, 'totalPersonalSeconds': 3.,
            'searchSeconds': 0., 'terminalSeconds': 3., 'realizedCombinedGPA': None,
            'D_ES_pp': .5, 'R_surplus_pp': -2., 'R_EGPA_pp': -1., 'L_lower_mean_share': -.1,
            'share': .4+.01*main, 'x0': 1. if constant else float(main),
            'choiceAvailable': True, 'remainingIntegerBudget': 1000}


class TestPortableSensitivity(unittest.TestCase):
    def test_real_cli_help_and_parser_do_not_need_numerical_imports(self):
        for args, code in ((['--help'], 0), ([], 2), (['all', '--output', 'unused'], 2),
                           (['all', '--output', 'unused', '--full', '--only', 'C3:0:1'], 2),
                           (['all', '--output', str(ROOT / 'src'), '--full'], 2)):
            result = subprocess.run([sys.executable, '-S', str(CLI), *args], text=True, capture_output=True)
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertNotIn('ModuleNotFoundError', result.stderr)
        self.assertIn('--selection', subprocess.run(
            [sys.executable, '-S', str(CLI), '--help'], text=True, capture_output=True).stdout)

    def test_actual_installed_namespace_import_and_shared_kernel(self):
        source = """
import sys
sys.path.insert(0, sys.argv[1])
from design_validation import personal_time_sensitivity as s
from design_validation import bounded_task as b
assert callable(s.prepare) and callable(s.produce) and callable(s.check)
assert s.P.G.normalize_log_weights is b.normalize_log_weights
assert 'src' not in sys.modules
assert len(s.select_roots(full=True)) == 450
"""
        result = subprocess.run([sys.executable, '-c', source, str(ROOT / 'src')],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_api_modes_and_selection(self):
        spec = importlib.util.spec_from_file_location('proxy_cli_test', CLI)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        calls = []
        selected = [{'conditionId': 'C3', 'scheduleVariant': 0, 'mainIndex': 1}]
        api = SimpleNamespace(select_roots=lambda **kw: calls.append(('select', kw)) or selected,
            load_context=lambda p: calls.append(('context', p)) or {'fixture': True},
            prepare=lambda *args, **kw: calls.append(('prepare', args, kw)),
            produce=lambda *args: calls.append(('produce', args)),
            check=lambda *args: calls.append(('check', args)) or {'checked': True})
        package = SimpleNamespace(personal_time_sensitivity=api)
        old_path = list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'design_validation': package}):
                got = cli.main(['all', '--output', directory, '--only', 'C3:0:1', '--node', 'selected-node'])
            self.assertEqual(got, {'checked': True})
            self.assertEqual([c[0] for c in calls], ['select', 'context', 'prepare', 'produce', 'check'])
            self.assertEqual(calls[2][2], {'node': 'selected-node'})
        finally:
            sys.path[:] = old_path

    def test_fixed_recipe_bits_labels_and_original_config_description(self):
        context, root, cfg = fixture()
        self.assertEqual(dict(S.RATES)['rho_0.00001'].hex(), '0x1.4f8b588e368f1p-17')
        self.assertEqual(dict(S.RATES)['rho_geometric_0.001_0.01'].hex(), '0x1.9e7c6e43390b7p-9')
        with patch.object(I, 'make_config', side_effect=lambda *_: copy.deepcopy(cfg)):
            reference = S.rate_config(context, root, 'rho_0.01')
            low = S.rate_config(context, root, 'rho_0.00001')
        self.assertEqual(reference, cfg)
        self.assertEqual(low['candidateId'], cfg['candidateId'])
        self.assertEqual(low['valuation'], {'rhoGPAperSecond': .00001, 'description': S.DESCRIPTION})
        self.assertEqual(S.select_roots(only='C5:0:1,C3:0:1'), I.select_roots(only='C5:0:1,C3:0:1'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            changed = copy.deepcopy(S.RECIPE)
            changed['rates'][4][1] = .00316
            I.write(path, changed)
            with self.assertRaisesRegex(ValueError, 'frozen six-price'):
                S.load_context(path)

    def test_binding_covers_new_science_and_detects_changed_source_runtime(self):
        context, root, _ = fixture()
        selected = [{k: root[k] for k in I.FIELDS}]
        sources = S.implementation_identity()
        for name in ('personal_time_sensitivity.py', 'personal_time_sensitivity_streams.py',
                     'personal_time_sensitivity_reference.py', 'personal_time_sensitivity_reduction.py',
                     'personal_time_sensitivity_analysis.py', 'configs/personal_time_sensitivity.json',
                     '../tutoring_analysis/native_engine.cjs', '../tutoring_analysis/nominal_inference.py'):
            self.assertIn(name, sources)
        with patch.object(S, 'runtime_identity', return_value={'fixture': 1}):
            first = S.binding(context, selected)
            with patch.object(S, 'implementation_identity', return_value={**sources, 'changed.py': 'different'}):
                self.assertNotEqual(S.binding(context, selected), first)
        with patch.object(S, 'runtime_identity', return_value={'fixture': 2}):
            self.assertNotEqual(S.binding(context, selected), first)


class TestOriginalStreams(unittest.TestCase):
    def test_typed_keys_and_price_free_report_streams(self):
        context, root, cfg = fixture()
        recipe = context['recipe']
        ids = P.stable_identity(cfg, 20261019, recipe['populationId'], 'C1', 0, 0, 1)
        self.assertEqual(ids, R.stable_identity(cfg, 20261019, recipe['populationId'], 'C1', 0, 0, 1))
        self.assertNotEqual(P.canonical(1), P.canonical(1.))
        self.assertNotEqual(P.canonical(-0.), P.canonical(0.))
        self.assertEqual(P.canonical(float('-inf')), R.key_value(float('-inf')))
        ctx = {'identity': ids, 'bindings': {k: recipe[k] for k in ('populationId', 'randomizationGroup')}}
        low = copy.deepcopy(cfg)
        low['valuation']['rhoGPAperSecond'] = .00001
        other = {**ctx, 'identity': P.stable_identity(low, 20261019, recipe['populationId'], 'C1', 0, 0, 1)}
        key = P.make_key(ctx, 'A', 'report', 0)
        self.assertEqual(key, P.make_key(other, 'A', 'report', 0))
        self.assertEqual(key, R.make_key(ctx, 'A', 'report', 0))
        self.assertNotEqual(ctx['identity']['personId'], other['identity']['personId'])
        law = P.A.normalize_log_weights(np.asarray(root['initialLogs'][0]))
        draw = P.stream_draw(key, law)
        self.assertEqual(draw, R.stream_draw(key, law))
        self.assertEqual(len(draw['seedHex']), 32)
        self.assertLessEqual(float.fromhex(draw['cdfPreviousHex']), float.fromhex(draw['uniformHex']))
        self.assertGreater(float.fromhex(draw['cdfSelectedHex']), float.fromhex(draw['uniformHex']))
        for field in ('key', 'uniformHex', 'probabilityHex', 'cdfSelectedHex', 'category'):
            altered = copy.deepcopy(draw)
            altered[field] = 'changed'
            with self.assertRaises(ValueError):
                S.compare(altered, R.stream_draw(key, law))
        with self.assertRaises(ValueError):
            P.stream_draw(key, law + .1)

    def test_paid_update_exactly_once_and_correct_slot(self):
        _, root, cfg = fixture()
        task = P.A.BudgetTask(P.A.TaskConfig(**cfg['task']))
        state = {'budget': cfg['state']['budget'], 'logHex': [[float(v).hex() for v in row] for row in root['initialLogs']],
                 'counts': [0, 0], 'queryIndex': 0, 'searchSecondsHex': (0.).hex()}
        before = copy.deepcopy(state)
        p = P.paid_value(cfg, task, state, {'action': 'ask_B'}, {'category': 51})
        r = R.paid_value(cfg, task, state, {'action': 'ask_B'}, {'category': 51})
        self.assertEqual(p, r)
        self.assertEqual(state, before)
        self.assertEqual(p['after']['counts'], [0, 1])
        self.assertEqual(p['after']['queryIndex'], 1)
        self.assertEqual(p['after']['budget'], state['budget']-task.config.cost_millihours)
        self.assertEqual(float.fromhex(p['after']['searchSecondsHex']), 4.65)
        self.assertEqual(p['after']['logHex'][0], state['logHex'][0])

    def test_expected_reference_never_draws_or_calls_supplied_path_runner(self):
        context, root, cfg = fixture()
        stop = {'action': 'ES', 'forced': False, 'index': 500, 'seconds': 3.,
                'expectedCombinedGPA': 7., 'expectedUtility': 0., 'objective': 6.97}
        chosen = {'action': 'ES', 'stop': stop, 'asks': []}
        for kernels in (P, R):
            with patch.object(kernels, 'decision', return_value=chosen) as evaluate, \
                 patch.object(kernels, 'stream_draw', side_effect=AssertionError('No expected-stop draws')), \
                 patch.object(kernels, 'metric_row', return_value={'expectedFinalUtility': 'not_from_GPA'}), \
                 patch.object(P.A, 'run', side_effect=AssertionError('No supplied path runner')):
                result = S.trial(context, root, cfg, {}, 'rho_0.01', kernels)
            self.assertEqual(result['endpointKind'], 'expected_only')
            self.assertEqual(result['metrics']['expectedFinalUtility'], 0.)
            self.assertIsNone(result['terminal']['realizedCombinedGPA'])
            self.assertEqual(evaluate.call_args.args[2]['logHex'], [[float(x).hex() for x in row] for row in root['initialLogs']])


class TestSensitivityReductions(unittest.TestCase):
    def test_original_ordered_means_and_independent_centered_slope(self):
        rows = [metric(m) for m in range(1, 11)]
        p, r = D.person(rows), R.independent_person(rows)
        for key in ('means', 'choiceMeans', 'choiceEligibleWholeTen', 'realizedAvailableMains', 'availableRealizedMean'):
            S.compare(p[key], r[key])
        S.compare(p['initialSlope'].get('value'), r['initialSlopeValue'])
        self.assertLess(p['means']['expectedFinalUtility'], 0)
        self.assertLess(p['choiceMeans']['R_surplus_pp'], 0)
        self.assertIsNone(p['monteCarloSE'])
        self.assertIsNone(p['means']['realizedCombinedGPA'])

    def test_undefined_slope_and_missing_choice_stay_undefined(self):
        rows = [metric(m, constant=True) for m in range(1, 11)]
        p, r = D.person(rows), R.independent_person(rows)
        self.assertIsNone(p['initialSlope'].get('value'))
        self.assertEqual(p['initialSlope']['reason'], 'insufficient_predictor_support')
        self.assertIsNone(r['initialSlopeValue'])
        rows[0].update(choiceAvailable=False, remainingIntegerBudget=0, share=None)
        p = D.person(rows)
        self.assertFalse(p['choiceEligibleWholeTen'])
        self.assertTrue(all(x is None for x in p['choiceMeans'].values()))
        with self.assertRaises(ValueError):
            D.person(rows[:-1])

    def test_nonfinite_or_bool_endpoints_are_not_silently_aggregated(self):
        for key, value in (('expectedFinalUtility', float('nan')), ('paidCount', True),
                           ('realizedCombinedGPA', float('inf')), ('mainIndex', True)):
            rows = [metric(m) for m in range(1, 11)]
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                D.person(rows)

    def test_partial_person_never_becomes_a_person_or_full_table(self):
        rows = [{'coordinates': ['C3', 0, 1], 'rateLabel': 'rho_0', 'metrics': metric(1)}]
        summary = D.aggregate(rows, ['rho_0'])
        self.assertEqual(summary['persons'], [])
        self.assertEqual(summary['conditions'], [])
        self.assertEqual(len(summary['missingCoordinates']['rho_0']), 449)
        self.assertFalse(summary['fullFiniteTarget'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            D.aggregate(rows+rows, ['rho_0'])
        changed = copy.deepcopy(rows)
        changed[0]['coordinates'][1] = True
        with self.assertRaises(ValueError):
            D.aggregate(changed, ['rho_0'])

    def test_equal_nine_denominators_and_independent_condition_reduction(self):
        rows = [{'coordinates': ['C3', v, m], 'rateLabel': 'rho_0', 'metrics': metric(m, v)}
                for v in range(9) for m in range(1, 11)]
        summary = D.aggregate(rows, ['rho_0'])
        value = summary['conditions'][0]
        self.assertTrue(value['completeOriginalNineSchedules'])
        self.assertEqual(value['means']['expectedFinalUtility']['fullNineMean'], -9.)
        self.assertFalse(summary['completePerRate'])
        people = [{'condition': 'C3', 'variant': v, 'rateLabel': 'rho_0',
                   **R.independent_person([metric(m, v) for m in range(1, 11)])} for v in range(9)]
        S.compare(summary['conditions'], R.independent_conditions(people, ['rho_0']))
        partial = D.aggregate(rows[:-10], ['rho_0'])
        self.assertIsNone(partial['conditions'][0]['means']['expectedFinalUtility']['fullNineMean'])
        self.assertEqual(partial['conditions'][0]['means']['expectedFinalUtility']['availableSchedules'], 8)
        with self.assertRaises(ValueError):
            D.condition_rows(summary['persons']+[summary['persons'][0]], ['rho_0'])

    def test_five_complete_prices_are_not_the_six_price_target(self):
        rows = [{'coordinates': [c, v, m], 'rateLabel': label, 'metrics': metric(m, v, c == 'C5')}
                for label in D.LABELS[:-1] for c, v, m in sorted(I.ORIGINAL)]
        summary = D.aggregate(rows, D.LABELS[:-1])
        self.assertTrue(summary['completePerRate'])
        self.assertFalse(summary['fullFiniteTarget'])
        self.assertEqual(len(summary['persons']), 225)
        self.assertEqual(len(summary['conditions']), 25)


class TestImmutableTrials(unittest.TestCase):
    def trial_record(self):
        context, root, cfg = fixture()
        stop = {'action': 'ES', 'forced': False, 'index': 500, 'seconds': 3.,
                'expectedCombinedGPA': 7., 'expectedUtility': 0., 'objective': 6.97}
        with patch.object(P, 'decision', return_value={'action': 'ES', 'stop': stop, 'asks': []}), \
             patch.object(P, 'metric_row', return_value={'expectedFinalUtility': 0.}):
            value = S.trial(context, root, cfg, {}, 'rho_0.01', P)
        binding = S.record_binding(root, cfg, {}, 'rho_0.01')
        return S.envelope(binding, value), binding

    def test_changed_binding_missing_events_and_rehashed_foreign_events_rejected(self):
        record, binding = self.trial_record()
        S.validate_record(record, binding)
        for mutate in (lambda p: p['events'].pop(), lambda p: p['events'].reverse(),
                       lambda p: p['events'].append({'name': 'foreign', 'value': {}}),
                       lambda p: p['finalState'].update(queryIndex=1),
                       lambda p: p['context'].update(runSHA256='foreign')):
            changed = copy.deepcopy(record['payload'])
            mutate(changed)
            with self.assertRaises(ValueError):
                S.validate_record(S.envelope(binding, changed), binding)
        with self.assertRaises(ValueError):
            S.validate_record(record, {**binding, 'configSHA256': 'different'})

    def test_producer_closed_resume_never_calls_scientific_trial(self):
        record, binding = self.trial_record()
        context, root, cfg = fixture()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(S, 'load_roots', return_value=([root], {}, Path(directory))), \
             patch.object(S, 'rate_config', return_value=cfg), \
             patch.object(S, 'trial', side_effect=AssertionError('Closed resume must not replay')):
            for label, _ in S.RATES:
                bound = S.record_binding(root, cfg, {}, label)
                I.write(S.record_path(directory, label, root), S.envelope(bound, record['payload']))
            before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(directory).rglob('*.json')}
            S.produce(context, [root], directory)
            self.assertTrue(all((Path(p).read_bytes(), Path(p).stat().st_mtime_ns) == value for p, value in before.items()))

    def test_checker_refuses_missing_producer_without_generating_it(self):
        context, root, _ = fixture()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(S, 'load_roots', return_value=([root], {}, Path(directory))), \
             patch.object(S, 'trial', side_effect=AssertionError('Missing producers must fail before replay')):
            with self.assertRaisesRegex(ValueError, 'producer trial inventory'):
                S.check(context, [root], directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
