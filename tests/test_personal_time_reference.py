"""Purpose: portable finite-reference interfaces, accounting and refusal cases.

Small synthetic fixtures test wiring and invariants, not historical agreement.
The separate release gate regenerates all 450 original roots, computes the two
selected original decisions with independent readback, and checks four EO indices.
"""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.design_validation import personal_time_policy as A
from src.design_validation import personal_time_reference as R
from src.design_validation import personal_time_inputs as I
from src.design_validation import personal_time_reproduction as P
from src.design_validation.personal_time_examples import EXPECTED, target_rows

ROOT = Path(__file__).resolve().parents[1]
SMALL = [{'conditionId': c, 'scheduleVariant': 0, 'mainIndex': 1} for c in ('C3', 'C5')]


class TestPersonalTimeInputs(unittest.TestCase):
    def test_exact_original_selection_and_no_implicit_compute(self):
        self.assertEqual(I.select_roots(only='C5:0:1,C3:0:1'), SMALL)
        self.assertEqual(I.selection(I.read(ROOT / 'configs/personal_time_selection.json')), SMALL)
        self.assertEqual(len(I.select_roots(full=True)), 450)
        for kwargs in ({}, {'full': True, 'only': 'C3:0:1'}, {'selected': []},
                       {'only': 'C3:0:1,C3:0:1'}, {'only': 'C3:9:1'}, {'only': 'C3:0:0'},
                       {'selected': [{**SMALL[0], 'mainIndex': True}]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                I.select_roots(**kwargs)

    def test_original_model_identity_and_no_silent_law_normalization(self):
        context = I.load_context(ROOT / 'configs/personal_time_reference.json')
        cfg = I.materialize(context['base'], context['instance'])
        self.assertEqual(cfg['candidateId'], 'r10-native-D1000-stream-endpoint-validation-not-adopted-v1')
        self.assertEqual(cfg['random']['streamNamespace'], 'r10-random-terminal:native-validation-v1')
        self.assertEqual(cfg['random']['probabilities'], [1. / 1001] * 1001)
        self.assertEqual(A.request_seconds(cfg, 0, 1), 4.65)
        self.assertEqual(A.request_seconds(cfg, 1, 100), R.seconds(cfg, 1, 100))
        cfg['random']['probabilities'][0] += .01
        with self.assertRaisesRegex(ValueError, 'unit probability mass'):
            A.validate(cfg)

    def test_binary64_projection_keeps_signed_zero_but_not_json_integer_spelling(self):
        self.assertEqual(I.numeric_bits(1), I.numeric_bits(1.0))
        self.assertNotEqual(I.numeric_bits(-0.), I.numeric_bits(0.))
        self.assertEqual(I.numeric_bits(True), True)
        with self.assertRaises(ValueError):
            I.numeric_bits(float('nan'))

    def test_immutable_resume_and_refusal_do_not_replace_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'output.json'
            I.write(path, {'value': 1})
            before = path.stat().st_mtime_ns, path.read_bytes()
            I.write(path, {'value': 1})
            self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                I.write(path, {'value': 2})
            self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before)
            link = Path(directory) / 'alias.json'
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                I.write(link, {'value': 1})

    def test_duplicate_or_nonfinite_json_is_rejected(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                I.parse(value)

    def test_prepare_is_public_subprocess_and_validated_immutable_resume(self):
        context = {'recipe': {'analysis': {}, 'construction': {}}, 'root_identity': {},
                   'design': {'z_original_first': 1, 'a_original_second': 2}}
        fake = {'schema': 'test-only', 'roots': []}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(I, 'binding', return_value={'testBinding': True}), \
             patch.object(I, 'validate_roots', return_value=[]), \
             patch.object(I.subprocess, 'run', return_value=SimpleNamespace(stdout=I.encoded(fake).decode())) as node:
            I.prepare(context, SMALL, directory, node='chosen-node')
            args, kwargs = node.call_args
            self.assertEqual(args[0][0], 'chosen-node')
            self.assertEqual(Path(args[0][1]).name, 'personal_time_source.cjs')
            self.assertNotIn('env', kwargs)
            self.assertEqual(I.parse(kwargs['input'])['selection'], SMALL)
            self.assertEqual(list(I.parse(kwargs['input'])['design']), list(context['design']))
            I.prepare(context, SMALL, directory, node='chosen-node')
            self.assertEqual(node.call_count, 1)

    def test_native_generation_error_retains_child_diagnostic(self):
        context = {'recipe': {'analysis': {}, 'construction': {}}, 'root_identity': {}, 'design': {}}
        failure = I.subprocess.CalledProcessError(1, ['node'], stderr='Original snapshot mismatch\n')
        with tempfile.TemporaryDirectory() as directory, patch.object(I, 'binding', return_value={}), \
             patch.object(I.subprocess, 'run', side_effect=failure):
            with self.assertRaisesRegex(ValueError, 'Original snapshot mismatch'):
                I.prepare(context, SMALL, directory)

    def test_input_reader_rejects_changed_bits_snapshots_and_rosters(self):
        root = {**SMALL[0], 'seed': 'synthetic-test-only', 'definition': {}, 'common': {},
                'logicalToDisplayed': ['A', 'B'], 'initialLogs': [[-0.]],
                'stateSnapshotSHA256': 'test-snapshot', 'prefixLength': 0, 'prefixKind': 'fixed', 'paid': []}
        identities = [{'condition': c, 'variant': v, 'main': m,
                       'scientificSHA256': 'unused-test-only', 'snapshotSHA256': 'unused-test-only'}
                      for c, v, m in sorted(I.ORIGINAL)]
        for row in identities:
            if (row['condition'], row['variant'], row['main']) == I.coordinates(root):
                row.update(scientificSHA256=I.scientific_digest(root), snapshotSHA256='test-snapshot')
        context = {'root_identity': {'coordinates': identities}}
        data = {'schema': 'personal-time-original-roots-v1', 'selection': [SMALL[0]], 'roots': [root],
                'nodeVersion': '22.14.0', 'realizedPolicyHistoriesGenerated': False}
        I.validate_roots(data, [SMALL[0]], context)
        for mutate in (lambda d: d['roots'][0]['initialLogs'][0].__setitem__(0, 0.),
                       lambda d: d['roots'][0].update(stateSnapshotSHA256='changed'),
                       lambda d: d['roots'].append(copy.deepcopy(root)),
                       lambda d: d.update(realizedPolicyHistoriesGenerated=True)):
            changed = copy.deepcopy(data)
            mutate(changed)
            with self.assertRaises(ValueError):
                I.validate_roots(changed, [SMALL[0]], context)


class TestPersonalTimeMathAndReduction(unittest.TestCase):
    def test_full100_enumeration_and_tutoring_cost_are_separate(self):
        cfg = I.read(ROOT / 'configs/personal_time_base.json')
        cfg['valuation']['rhoGPAperSecond'] = .01
        logs = [np.zeros(100), np.zeros(100)]
        task = SimpleNamespace(config=SimpleNamespace(cost_millihours=3),
                               predictive=lambda _: (np.full(100, .01), [np.zeros(100)] * 100))
        terminal = {'action': 'ES', 'objective': 1.0}
        with patch.object(A, 'terminal', return_value=terminal) as stop, \
             patch.object(A, 'request_seconds', return_value=2.):
            decision = A.evaluate(cfg, task, logs, 10)
        self.assertEqual(stop.call_count, 201)
        self.assertEqual([call.args[-1] for call in stop.call_args_list[1:]], [7] * 200)
        self.assertEqual(decision['action'], 'ES')
        for ask in decision['asks']:
            self.assertAlmostEqual(ask['objective'], .98)
            self.assertEqual(ask['tutoringIntegerCost'], 3)
        ticks = []
        with patch.object(R, 'stop_value', return_value=terminal), patch.object(R, 'seconds', return_value=2.):
            independent = R.decision_value(cfg, task, logs, 10, lambda: ticks.append(None))
        self.assertEqual(len(ticks), 200)
        P.compare(decision, independent)

    def test_forced_random_boundary_has_positive_personal_time_without_draw(self):
        cfg = I.read(ROOT / 'configs/personal_time_base.json')
        task = A.BudgetTask(A.validate(cfg))
        producer = A.evaluate(cfg, task, task.initial_logs, 0)
        independent = R.decision_value(cfg, task, task.initial_logs, 0,
                                       lambda: self.fail('No reports affordable at zero budget'))
        P.compare(producer, independent)
        self.assertEqual(producer['action'], 'random')
        self.assertTrue(producer['stop']['forced'])
        self.assertGreater(producer['stop']['seconds'], 0)
        self.assertFalse(producer['stop']['realizedAllocationDrawn'])
        self.assertEqual(producer['asks'], [])

    def test_supplied_path_charges_incremental_and_sunk_time_once(self):
        cfg = I.read(ROOT / 'configs/personal_time_base.json')
        cfg['state']['budget'] = cfg['task']['total_millihours']
        cfg['pathReports'] = [{'slot': 0, 'report': 50}]
        stop = {'action': 'ES', 'seconds': 3., 'expectedCombinedGPA': 7., 'objective': 6.97}
        with patch.object(A, 'evaluate', side_effect=[{'action': 'ask_A'}, {'action': 'ES', 'stop': stop}]), \
             patch.object(A, 'request_seconds', return_value=2.):
            result = A.run(cfg)
        self.assertEqual(result['reportsConsumed'], 1)
        self.assertEqual(result['finalBudget'], cfg['state']['budget'] - cfg['task']['cost_millihours'])
        self.assertEqual(result['totalPersonalSeconds'], 5.)
        self.assertAlmostEqual(result['finalNetObjective'], 6.95)

    def test_partial_selection_never_becomes_a_condition_estimate(self):
        row = {'coordinates': ['C3', 0, 1], **dict.fromkeys((*P.SCALARS, 'share'), 0.)}
        summary = P.aggregate([row])
        self.assertFalse(summary['fullFiniteTarget'])
        self.assertEqual(summary['missingTrials'], 449)
        self.assertEqual(summary['conditions'], {})
        with self.assertRaises(ValueError):
            P.aggregate([row, row])

    def test_full_aggregation_requires_whole_people_and_never_invents_mcse(self):
        rows = [{'coordinates': [c, v, m], **{key: float(v) for key in (*P.SCALARS, 'share')}}
                for c, v, m in sorted(I.ORIGINAL)]
        summary = P.aggregate(rows)
        self.assertEqual(summary['assignedPeople'], 45)
        self.assertEqual(summary['verifiedTrials'], 450)
        self.assertEqual(len(summary['persons']), 45)
        for condition in summary['conditions'].values():
            for scalar in condition['scalarEstimates'].values():
                self.assertEqual(scalar['estimate']['mean'], 4.)
                self.assertIsNone(scalar['estimate']['MCSE'])
        self.assertFalse(summary['realizedOutcomesGenerated'])
        self.assertFalse(summary['monteCarloPrecisionEstablished'])

    def test_reader_emits_all100_margins_not_two_display_rows(self):
        roots = [dict(zip(I.FIELDS, key)) for key in sorted(I.ORIGINAL)]
        cfg = I.read(ROOT / 'configs/personal_time_base.json')
        cfg['state']['logs'] = [[0.] * 100, [0.] * 100]
        decision = {'action': 'ES', 'stop': {'action': 'ES', 'expectedCombinedGPA': 7.,
                    'objective': 6.97, 'seconds': 3., 'index': 1},
                    'asks': [{'slot': s, 'objective': 6.9, 'expectedIncrementalPersonalSeconds': 4.65}
                             for s in (0, 1)]}
        with tempfile.TemporaryDirectory() as directory, patch.object(P, 'reference_runtime'), \
             patch.object(P, 'load_roots', return_value=roots), patch.object(P, 'binding', return_value={}), \
             patch.object(P, 'make_config', return_value=cfg), patch.object(P, 'validate_record'), \
             patch.object(P, 'BudgetTask', return_value=object()), \
             patch.object(P, 'read', return_value={'decision': decision}), \
             patch.object(P.R, 'decision_value', return_value=decision):
            result = P.check({}, roots, directory)
            self.assertEqual(result['marginRows'], 100)
            self.assertEqual(result['originalVariant0Roots'], 50)
            self.assertEqual(len(I.read(Path(directory) / 'margin_rows.json')), 100)
            self.assertEqual(len((Path(directory) / 'acquisition_margins.csv').read_text().splitlines()), 101)
            self.assertEqual(len(I.read(Path(directory) / 'trial_rows.json')), 450)

    def test_comparison_preserves_discrete_decisions_and_finite_tolerance(self):
        P.compare({'value': 1. + 1e-11}, {'value': 1.})
        for got, expected in ((True, 1), (1., 1), (float('nan'), 1.), (1.1, 1.), ('AL', 'ES')):
            with self.subTest(got=got), self.assertRaises(ValueError):
                P.compare(got, expected)

    def test_checkpoint_tampering_and_unexpected_action_are_refused(self):
        decision = {'action': 'ES', 'stop': {'action': 'ES'}}
        record = {'schema': 'personal-time-expected-stop-v1', 'binding': {'x': 1}, 'status': 'computed',
                  'decision': decision, 'decisionSHA256': I.digest(decision),
                  'realizedOutcomesGenerated': False, 'newRandomDraws': 0}
        P.validate_record(record, {'x': 1})
        for mutate in (lambda r: r['binding'].update(x=2),
                       lambda r: r.update(status='unexpected_selected_action'),
                       lambda r: r['decision'].update(action='ask_A')):
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.assertRaises(ValueError):
                P.validate_record(changed, {'x': 1})

    def test_equal_target_route_calls_existing_kernels(self):
        configs = [SimpleNamespace(template_id=t, total_millihours=100, allocation_denominator=1000)
                   for t in ('C2_s1_g0', 'C3_s1_g0')]
        def summary(config, *args):
            return {'macros': {rule: {'index': EXPECTED[config.template_id, rule]}
                               for rule in ('EO_expected_GPA', 'EO_raw_surplus')}}
        module = 'src.design_validation.personal_time_examples'
        with patch(module + '.budget_templates', return_value=configs), \
             patch(module + '.BudgetTask', return_value=SimpleNamespace(initial_logs=[[], []])), \
             patch(module + '.terminal_summary', side_effect=summary) as terminal:
            rows = target_rows({})
        self.assertEqual(terminal.call_count, 2)
        self.assertEqual([row['index'] for row in rows], [367, 367, 451, 446])
        self.assertTrue(all(row['exact_index_agreement'] for row in rows))


if __name__ == '__main__':
    unittest.main()
