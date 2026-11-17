"""Purpose: preserve the card roster, weights, refusal rules and portable entry."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation.strategy_cards import adaptive_path as A, inputs as I
from src.design_validation.strategy_cards import reproduction as R, summary as S

ROOT = Path(__file__).resolve().parents[1]


def row(condition='C3', variant=0, main=1):
    return {'id': [condition, variant, main], 'replicate': 0,
            'fixed': [{'card': name, 'status': 'scored', 'expectedGPA': 7.,
                       'expectedGPAInterval': ['5', '9'],
                       'expectedQuestions': 2 if index == 4 else 0,
                       'expectedTutoringMinutes': 1 if index == 4 else 0}
                      for index, name in enumerate(S.FIXED)],
            'adaptive': {'card': S.ADAPTIVE, 'status': 'terminal', 'GPAInterval': ['7', '71/10'],
                         'questions': [2, 2], 'tutoringMinutes': ['1', '1'], 'target': 501}}


class SummaryTests(unittest.TestCase):
    def test_full_weights_and_uncertainty_labels(self):
        records = [row(*key) for key in sorted(I.ORIGINAL)]
        value = S.summarize(records, full=True)
        self.assertEqual(len(value['cells']), 30)
        self.assertTrue(value['fullFiniteTarget'])
        self.assertFalse(value['rankingCertified'])
        self.assertFalse(value['humanTargetsEstablished'])
        self.assertFalse(value['rowSourceAuthenticated'])
        self.assertIsNone(value['withinRootMCSE'])
        self.assertEqual(value['cells'][0]['expectedCombinedGPA'], 7)
        self.assertEqual(value['cells'][5]['means']['GPAInterval'], ['7', '71/10'])

    def test_partial_not_renormalized(self):
        value = S.summarize([row()])
        self.assertEqual(value['missingRoots'], 449)
        self.assertIsNone(value['cells'][0]['expectedCombinedGPA'])
        self.assertAlmostEqual(value['cells'][0]['pointContribution'], 7/90)
        self.assertEqual(value['cells'][0]['missingRoots'], 89)
        self.assertIsNone(value['cells'][5]['means']['GPAInterval'])
        with self.assertRaises(ValueError):
            S.summarize([row()], full=True)

    def test_duplicate_and_foreign_coordinates(self):
        for records in ([row(), row()], [row('C6')], [row(variant=True)]):
            with self.assertRaises(ValueError):
                S.summarize(records)

    def test_wrong_replicate_and_card_roster(self):
        item = row()
        for field, value in [('replicate', 1), ('replicate', False), ('fixed', item['fixed'][:-1])]:
            changed = dict(item, **{field: value})
            with self.assertRaises(ValueError):
                S.summarize([changed])

    def test_unresolved_not_imputed_to_zero(self):
        records = [row('C3', v, m) for v in range(9) for m in range(1, 11)]
        records[0]['fixed'][0].update(status='unresolved', expectedGPA=None)
        value = S.summarize(records)
        self.assertIsNone(value['cells'][0]['expectedCombinedGPA'])
        self.assertEqual(value['cells'][0]['statusCounts']['unresolved'], 1)
        self.assertEqual(value['cells'][0]['meanPayoffSupport'], ['5', '9'])

    def test_nonfinite_and_invalid_intervals(self):
        bad = row()
        bad['fixed'][0]['expectedGPA'] = float('nan')
        with self.assertRaises(ValueError):
            S.summarize([bad])
        bad = row()
        bad['adaptive']['GPAInterval'] = ['8', '7']
        with self.assertRaises(ValueError):
            S.summarize([bad])

    def test_unavailable_costs_remain_unknown(self):
        for omitted in (False, True):
            records = [row('C3', v, m) for v in range(9) for m in range(1, 11)]
            unavailable = records[0]['fixed'][0]
            unavailable.update(status='unavailable', expectedGPA=None, expectedGPAInterval=None,
                               expectedQuestions=None, expectedTutoringMinutes=None)
            if omitted:
                del unavailable['expectedQuestions']
                del unavailable['expectedTutoringMinutes']
            cell = S.summarize(records)['cells'][0]
            self.assertEqual(cell['denominator'], 90)
            self.assertEqual(cell['observedRoots'], 90)
            self.assertEqual(cell['statusCounts']['unavailable'], 1)
            self.assertIsNone(cell['expectedCombinedGPA'])
            self.assertIsNone(cell['meanPayoffSupport'])
            self.assertIsNone(cell['meanQuestions'])
            self.assertIsNone(cell['meanTutoringMinutes'])


class EntryTests(unittest.TestCase):
    def test_recipe_is_exact(self):
        recipe = json.loads((ROOT / 'configs/strategy_cards.json').read_text())
        self.assertEqual(recipe, I.RECIPE)
        self.assertEqual(recipe['stream_domain'], 'R13-one-more-v1')

    def test_public_help_outside_checkout(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce_strategy_cards.py'), '--help'],
                                cwd=tempfile.gettempdir(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('aggregate', result.stdout)

    def test_no_implicit_execution(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce_strategy_cards.py'),
                                 'produce', '--output', 'not-created'], cwd=tempfile.gettempdir(),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('no implicit', result.stderr)

    def test_changed_recipe_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'recipe.json'
            value = copy.deepcopy(I.RECIPE)
            value['replicate'] = 1
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                I.context(path)

    def test_selection_and_concurrent_lock(self):
        roots = [{'id': ['C3', 0, 1]}]
        self.assertEqual(R.selected_roots(roots, only='C3:0:1'), roots)
        with self.assertRaises(ValueError):
            R.selected_roots(roots, full=True)
        with tempfile.TemporaryDirectory() as folder:
            with R.lock(folder, 'produce', roots[0]):
                with self.assertRaises(FileExistsError):
                    with R.lock(folder, 'produce', roots[0]):
                        self.fail('Concurrent root production admitted')

    def test_production_wiring_without_running_a_policy(self):
        root = {'id': ['C3', 0, 1]}
        with patch.object(R.fixed, 'evaluate', return_value={'rows': []}) as fixed:
            with patch.object(R.adaptive, 'evaluate', return_value={'status': 'terminal'}) as adaptive:
                value = R.produce_root(root)
        fixed.assert_called_once_with(root)
        self.assertEqual(adaptive.call_args.args[:2], (root, 0))
        self.assertEqual(value['replicate'], 0)

    def test_saved_entry_binding_and_partial_reduction(self):
        root = {'id': ['C3', 0, 1], 'task': {'total_millihours': 1000, 'cost_millihours': 100,
                'gpa_baseline': 3.5, 'gpa_scale': .03, 'loss_multiplier': 2}}
        fixture = row()
        record = {'id': root['id'], 'replicate': 0,
                  'fixed': {'id': root['id'], 'rows': fixture['fixed']},
                  'adaptive': {'status': 'terminal', 'target': 501,
                               'externalConditionalExpectedGPAInterval': ['7', '71/10']}}
        check = {'fixed': {}, 'adaptive': {'status': 'PASS', 'disposition': 'terminal',
                 'paidReports': 2, 'terminal': {'cursors': [1, 1], 'budget': 800,
                 'target': 501, 'independentlyEnclosedGPA': ['7', '71/10']}}}
        binding = {'runtime': {'fixture': True}}
        with tempfile.TemporaryDirectory() as folder, patch.object(I, 'load', return_value=(binding, [root])):
            with patch.object(I, 'runtime', return_value=binding['runtime']):
                with patch.object(R, 'produce_root', return_value=record) as produce:
                    R.process(folder, None, phase='produce', only='C3:0:1')
                    R.process(folder, None, phase='produce', only='C3:0:1')
                    produce.assert_called_once()
                with patch.object(R, 'check_root', return_value=check) as verify:
                    R.process(folder, None, phase='check', only='C3:0:1')
                    R.process(folder, None, phase='check', only='C3:0:1')
                    verify.assert_called_once()
            summary = R.aggregate(folder, None)
            value = I.read(summary['summary'])
            self.assertTrue(value['rowSourceAuthenticated'])
            self.assertFalse(value['fullFiniteTarget'])
            self.assertEqual(value['observedRoots'], 1)
            with self.assertRaises(ValueError):
                R.aggregate(folder, None, full=True)
            path = Path(folder) / 'producer' / R.name(root)
            changed = I.read(path)
            changed['result']['replicate'] = 1
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                R.aggregate(folder, None)

    def test_input_source_mapping_and_display_rejection(self):
        source = {'fixture': True}
        root = {'id': ['C3', 0, 1], 'logBits': [['abc'], ['def']], 'orderedPhysical': ['A', 'B'],
                'display': {'id': ['C3', 0, 1], 'priorLogBits': [['abc'], ['def']], 'root': ['1.0', '2.0']}}
        adapter = {k: v for k, v in root.items() if k != 'display'}
        binding = {'recipe': I.RECIPE, 'sources': source, 'selection': [], 'rootSHA256': I.digest([root])}
        with tempfile.TemporaryDirectory() as folder, patch.object(I, 'context', return_value={}), \
                patch.object(I, 'sources', return_value=source), \
                patch.object(I.I, 'load_roots', return_value=[{}]), \
                patch.object(I, 'root_adapter', return_value=adapter):
            run, data = Path(folder) / 'run.json', Path(folder) / 'roots.json'
            I.write(run, binding)
            I.write(data, [root])
            self.assertEqual(I.load(folder, None)[1], [root])
            changed = copy.deepcopy(root)
            changed['orderedPhysical'].reverse()
            data.write_text(json.dumps([changed]))
            rebound = dict(binding, rootSHA256=I.digest([changed]))
            run.write_text(json.dumps(rebound))
            with self.assertRaises(ValueError):
                I.load(folder, None)
            data.write_text(json.dumps([root]))
            run.write_text(json.dumps(binding))
            with patch.object(I, 'render', return_value=[changed]):
                with self.assertRaises(ValueError):
                    I.load(folder, None, node='fixture-node')
            run.write_text(json.dumps(dict(binding, sources={'changed': True})))
            with self.assertRaises(ValueError):
                I.load(folder, None)

    def test_strict_comparison_and_physical_rng(self):
        self.assertEqual(A.choose(['1', '1'], {0: ['1', '1'], 1: ['1', '1']}), ('stop', None))
        self.assertEqual(A.choose(['1', '1'], {0: ['2', '2'], 1: ['2', '2']}), ('ask_tie', None))
        self.assertEqual(A.choose(['1', '1'], {0: ['2', '2'], 1: ['1', '1']}), ('ask', 0))
        key, u = A.key_uniform('report', ['C3', 0, 1], 0, 'B', 1)
        self.assertEqual(key, ['R13-one-more-v1', 'report', ['C3', 0, 1], 0, 'B', 1])
        self.assertNotEqual(u, A.key_uniform('report', ['C3', 0, 1], 0, 'B', 0)[1])


if __name__ == '__main__':
    unittest.main()
