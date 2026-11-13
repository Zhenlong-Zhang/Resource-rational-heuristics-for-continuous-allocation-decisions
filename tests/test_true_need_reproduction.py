"""Purpose: portable CLI, frozen recipe and saved terminal-routine source linkage.

The real checked-input entry is also verified against retained endpoint outputs;
these fixtures alone do not authenticate a historical scientific result.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation import true_need_reproduction as R

ROOT = Path(__file__).resolve().parents[1]


class TrueNeedReproductionTests(unittest.TestCase):
    def test_recipe_matches_config(self):
        self.assertEqual(json.loads((ROOT / 'configs/true_need_diagnostics.json').read_text()),
                         R.RECIPE)

    def test_csv_metrics_are_scalar_columns_and_missing_stays_missing(self):
        got = R.table_rows([{'condition': 'C3', 'gap': {'mean': None, 'sample_sd': None},
                            'missing_mains': [2, 3]}])[0]
        self.assertIsNone(got['gap_mean'])
        self.assertIsNone(got['gap_sample_sd'])
        self.assertEqual(json.loads(got['missing_mains']), [2, 3])

    def test_cli_help_from_outside_checkout(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_true_need_allocations.py'),
                                     '--help'], cwd=cwd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--full', result.stdout)

    def test_terminal_fields_use_saved_selection_not_index_guess(self):
        payload = {'terminal': {'action': 'MFN_A', 'intendedIndex': 1000},
                   'events': [{'name': 'decision_000000', 'value': {'value': {'stop': {
                       'action': 'MFN_A', 'index': 1000,
                       'legacySelection': {'selected_macro': 'MFN_A', 'selected_index': 1000}}}}}]}
        self.assertEqual(R.terminal_fields(payload),
                         {'terminal_action': 'MFN_A', 'selected_rule': 'MFN_A'})
        for mutate in ('action', 'index', 'selected_index', 'missing_selection', 'missing_decision'):
            bad = copy.deepcopy(payload)
            if mutate == 'action':
                bad['terminal']['action'] = 'AL'
            elif mutate == 'index':
                bad['terminal']['intendedIndex'] = 0
            elif mutate == 'missing_selection':
                bad['events'][0]['value']['value']['stop']['legacySelection'].clear()
            elif mutate == 'selected_index':
                bad['events'][0]['value']['value']['stop']['legacySelection']['selected_index'] = 0
            else:
                bad['events'].clear()
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                R.terminal_fields(bad)

    def test_changed_recipe_rejected_before_endpoint_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'recipe.json'
            config.write_text(json.dumps({**R.RECIPE, 'tieRule': 'approximate'}))
            with patch.object(R.D, 'analyze') as downstream:
                with self.assertRaises(ValueError):
                    R.analyze(root / 'input', root / 'output', config)
                downstream.assert_not_called()

    def test_nested_output_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'input'
            with self.assertRaises(ValueError):
                R.analyze(source, source / 'out', ROOT / 'configs/true_need_diagnostics.json')


if __name__ == '__main__':
    unittest.main()
