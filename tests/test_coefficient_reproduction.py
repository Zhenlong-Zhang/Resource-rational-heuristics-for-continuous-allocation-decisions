"""Portable entry, exact checkpoints, notebook defaults and rejection boundaries."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation import coefficient_io as IO
from src.design_validation import coefficient_reproduction as D
from src.design_validation import personal_time_inputs as I
from src.design_validation import c5_reproduction as C5
from src.design_validation import c5_streams as K
from src.design_validation import c5_moments as M
from test_coefficient_diagnostics import complete_fixture

ROOT = Path(__file__).resolve().parents[1]


class CoefficientReproductionTests(unittest.TestCase):
    def test_help_and_no_implicit_population(self):
        cli = ROOT / 'scripts/reproduce_coefficient_diagnostics.py'
        for args, code in ((['--help'], 0), (['fresh', '--output', 'unused'], 2),
                           (['c5', '--output', 'unused'], 2),
                           (['diagnose', '--output', 'unused', '--full'], 2),
                           (['check-c5', '--output', 'unused', '--only', 'C5:0:1', '--workers', '2'], 2),
                           (['import-original', '--output', 'unused'], 2),
                           (['diagnose-original', '--input', 'unused', '--output', 'unused2', '--full'], 2)):
            result = subprocess.run([sys.executable, '-S', str(cli), *args],
                                    cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_exact_checkpoint_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'stage.json'
            value, new = IO.stage(path, {'input': 'one'}, lambda: {'answer': 7})
            self.assertTrue(new)
            before = path.read_bytes(), path.stat().st_mtime_ns
            def forbidden():
                raise AssertionError('Closed numerical stage repeated')
            self.assertEqual(IO.stage(path, {'input': 'one'}, forbidden), (value, False))
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
            with self.assertRaises(ValueError):
                IO.stage(path, {'input': 'two'}, forbidden)
            with self.assertRaises(ValueError):
                IO.stage(path.with_name('missing.json'), {}, forbidden, readonly=True)
            changed = I.read(path)
            changed['payload']['answer'] = 8
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                IO.stage(path, {'input': 'one'}, forbidden)

    def test_malformed_checkpoint_and_foreign_member(self):
        for value in (None, [], {}, {'payload': 1}):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'stage.json'
                I.write(path, value)
                with self.assertRaises(ValueError):
                    IO.stage(path, {}, lambda: self.fail('Corrupt is not missing'))
        with tempfile.TemporaryDirectory() as temp:
            I.write(Path(temp) / 'laws/foreign.json', {})
            with self.assertRaises(ValueError):
                IO.inventory(temp, set())

    def test_saved_diagnosis_resume_keeps_saved_rows_and_no_policy(self):
        rows, summary = complete_fixture()
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'input', Path(temp) / 'output'
            with patch.object(D, 'admit_bundle', return_value=(rows, summary, 'synthetic-seal')), \
                 patch.object(IO, 'source_identity', return_value={'synthetic': True}):
                result = D.diagnose(source, output, ROOT / 'configs')
                self.assertEqual(result['newDesignSchedules'], 270)
                with patch.object(D.D, 'diagnose', side_effect=AssertionError('No resumed calculation')):
                    resumed = D.diagnose(source, output, ROOT / 'configs', checking=True)
                self.assertEqual(resumed['newDesignSchedules'], 0)
                self.assertTrue(resumed['passed'])

    def test_unsealed_or_changed_input_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            I.write(Path(temp) / 'figure_bundle.json', {})
            with self.assertRaises(ValueError):
                D.admit_bundle(temp, ROOT / 'configs')
            with self.assertRaises(ValueError):
                IO.separate(temp, Path(temp) / 'nested')

    def test_c5_roster_key_domain_and_zero_boundary(self):
        self.assertEqual(len(C5.selection(full=True)), 90)
        self.assertEqual(len(C5.selection(only='C5:0:1')), 1)
        with self.assertRaises(ValueError):
            C5.selection(only='C1:0:1')
        with self.assertRaises(ValueError):
            K.private_key(0, 1, 0, 'A')
        a, b = K.private_key(0, 1, 1, 'A'), K.private_key(0, 1, 1, 'B')
        self.assertNotEqual(a, b)
        self.assertNotIn('price', a)
        self.assertEqual(a['seed'], 20261019)
        self.assertEqual(a['randomizationGroup'], K.GROUP)
        self.assertNotEqual(a, K.private_key(0, 1, 1, 'A', 'report', 0))

    def test_display_symmetry_is_conditional_not_implied_by_keys(self):
        root = dict(conditionId='C5', scheduleVariant=0, mainIndex=1, seed=1,
                    definition={'budgetMillihours': 100000}, logicalToDisplayed=['B', 'A'],
                    initialLogs=[[0.]*100, [0.]*100], common={})
        high = dict(decision={'action': 'MFN_A', 'stop': {'action': 'MFN_A', 'index': 1000}},
                    terminal={'action': 'MFN_A', 'intendedIndex': 1000, 'budget': 100000,
                              'newScientificDraws': 0, 'realizedCombinedGPA': None})
        record = C5.law_record(root, high)
        self.assertTrue(record['symmetryCertified'])
        self.assertEqual(record, C5.law_record(root, high, independent=True))
        changed = copy.deepcopy(root)
        changed['initialLogs'][1][0] = 1.
        self.assertFalse(C5.law_record(changed, high)['symmetryCertified'])
        bad_high = copy.deepcopy(high)
        bad_high['terminal']['intendedIndex'] = 0
        with self.assertRaises(ValueError):
            C5.law_record(root, bad_high)

    def test_source_to_law_to_reader_and_missing_only_resume_fixture(self):
        root = dict(conditionId='C5', scheduleVariant=0, mainIndex=1, seed=1,
                    definition={'budgetMillihours': 100000}, logicalToDisplayed=['A', 'B'],
                    initialLogs=[[0.]*100, [0.]*100], common={})
        high = dict(decision={'action': 'MFN_A', 'stop': {'action': 'MFN_A', 'index': 1000}},
                    terminal={'action': 'MFN_A', 'intendedIndex': 1000, 'budget': 100000,
                              'newScientificDraws': 0, 'realizedCombinedGPA': None})
        run = dict(highSource='fresh_public_initial_decisions')
        domain = dict(newScientificDraws=0, terminalDomainPending=True)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            setup = (output, {}, [root], [], run, None, 0)
            with patch.object(C5, '_setup', return_value=setup), \
                 patch.object(C5.S, 'rate_config', return_value={'fixture': True}), \
                 patch.object(C5.K, 'audit', return_value=domain), \
                 patch.object(C5, '_high_source', return_value=high) as source:
                made = C5.produce(output, None, None)
                got = C5.check(output, None, None)
                self.assertEqual(source.call_count, 2)
                self.assertEqual(made['counters']['newRootLaws'], 1)
                self.assertEqual(got['verifiedRoots'], 1)
                self.assertEqual(got['verifiedDesigns'], 0)
                self.assertIsNone(got['summary']['floorFeasible'])
                self.assertFalse(got['fullFiniteTarget'])
                with patch.object(C5, 'law_record', side_effect=AssertionError('Closed law repeated')):
                    repeated = C5.produce(output, None, None)
                    checked = C5.check(output, None, None)
                self.assertEqual(source.call_count, 2)
                self.assertEqual(repeated['counters']['newRootLaws'], 0)
                self.assertEqual(checked['counters']['newIndependentRootLaws'], 0)
                (output / 'independent_laws/C5_v0_m1.json').unlink()
                repaired = C5.check(output, None, None)
                self.assertEqual(repaired['counters']['newIndependentRootLaws'], 1)
                (output / 'laws/C5_v0_m1.json').unlink()
                with self.assertRaises(ValueError):
                    C5.check(output, None, None)

    def test_notebook_is_output_free_and_installs_first(self):
        nb = json.loads((ROOT / 'notebooks/finite_tutoring/coefficient_precision.ipynb').read_text())
        code = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
        self.assertIn("'pip', 'install'", ''.join(code[0]['source']))
        self.assertTrue(all(cell['execution_count'] is None and cell['outputs'] == [] for cell in code))
        self.assertIn('RUN = False', ''.join(code[1]['source']))
        self.assertNotIn('rep1', ''.join(cell['source'][0] for cell in code))


if __name__ == '__main__':
    unittest.main()
