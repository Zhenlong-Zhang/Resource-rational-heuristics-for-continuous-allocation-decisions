"""Purpose: refuse foreign historical bytes and keep redraw provenance explicit.

Synthetic fixtures exercise software boundaries only. Accepted-summary parity
and visual output are verified separately with the pinned original inputs.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.design_validation import paper_figure_original as O
from src.design_validation import paper_figures_plot as F
from src.design_validation import personal_time_inputs as I
from test_paper_figures_plot import full_fixture

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/paper_figures.json'


class OriginalFigureTests(unittest.TestCase):
    def test_foreign_input_is_not_a_policy_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'inputs'
            I.write(source / 'saved_summary.json', {})
            with self.assertRaisesRegex(ValueError, 'Changed historical scientific bytes'):
                O.redraw(source, Path(temp) / 'out', CONFIG)

    def test_same_or_nested_destination_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            for source, output in ((Path(temp), Path(temp)),
                                   (Path(temp), Path(temp) / 'out'),
                                   (Path(temp) / 'in', Path(temp))):
                with self.subTest(source=source, output=output), self.assertRaises(ValueError):
                    O.redraw(source, output, CONFIG)

    def test_explicit_fixture_redraw_never_creates_a_policy_seal(self):
        allocation, true, _, _ = full_fixture()
        with tempfile.TemporaryDirectory() as temp:
            source, out = Path(temp) / 'in', Path(temp) / 'out'
            I.write(source / 'saved_summary.json', allocation)
            I.write(source / 'public_summary.json', true)
            pins = {name: O.S.file_sha(source / name) for name in O.INPUTS}
            with patch.dict(O.INPUTS, pins, clear=True), \
                 patch.object(F, 'render', return_value={}):
                result = O.redraw(source, out, CONFIG)
                repeated = O.redraw(source, out, CONFIG)
            self.assertEqual(result, repeated)
            self.assertFalse(result['newPublicPolicyEpoch'])
            self.assertFalse(result['historicalAcceptanceRevalidated'])
            self.assertFalse(result['actionQuestionTableIncluded'])
            self.assertFalse((out / 'figure_bundle.json').exists())
            self.assertEqual(I.read(out / 'historical_redraw.json')['inputs'], pins)

    def test_old_policy_and_coefficient_kernels_stay_outside_adapter(self):
        self.assertNotIn('paper_figure_original.py', O.S.SOURCE_FILES)
        self.assertEqual(O.INPUTS['public_summary.json'], O.O.INPUTS['public_summary.json'])


if __name__ == '__main__':
    unittest.main()
