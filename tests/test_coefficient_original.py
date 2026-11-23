"""Original compact evidence import without synthesizing policy provenance.

All data here are synthetic. Only the test replaces the fixed byte identities;
production has no override for the accepted scientific dataset or recipe.
"""
import copy
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation import coefficient_original as O
from src.design_validation import coefficient_reproduction as C
from src.design_validation import coefficient_io as IO
from src.design_validation import personal_time_inputs as I
from test_coefficient_diagnostics import complete_fixture

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    rows, summary = complete_fixture()
    summary.update(rowSourceAuthenticated=False, historicalEventClaimed=False,
                   policyRerunByAnalysis=False, monteCarloPrecisionEstablished=False,
                   humanInferenceAdopted=False)
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=O.CSV_FIELDS, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows({key: row[key] for key in O.CSV_FIELDS} for row in summary['schedules'])
    raw = {'admitted_rows.json': I.encoded(rows), 'public_summary.json': I.encoded(summary),
           'true_need_schedules.csv': text.getvalue().encode()}
    return rows, summary, text.getvalue(), raw


class OriginalCoefficientImportTests(unittest.TestCase):
    def test_original_public_names_and_source_recipe(self):
        self.assertEqual(set(O.INPUTS), {'admitted_rows.json', 'public_summary.json', 'true_need_schedules.csv'})
        self.assertEqual(O.INPUTS['true_need_schedules.csv'],
                         'ec15975a68e2640573dcf70ecd2ba02a5b330a57395196f0cb37b8b00e5e4fc9')
        identity = O.scientific_identity(ROOT / 'configs')
        self.assertEqual(identity['referencePublicCommit'], '290ef18893c17c0e25add5f4ab450096f8a8a66d')
        self.assertEqual(identity['proxyRecipe']['seed'], 20261019)
        self.assertEqual(identity['scope']['endpoints'], 2700)

    def test_scientific_csv_checks_and_original_false_flags(self):
        rows, summary, text, _ = fixture()
        check = O.admit_compact(rows, summary, text)
        self.assertTrue(check['csvSummaryAgreement'])
        self.assertFalse(check['independentGeometryChecked'])
        with self.assertRaises(ValueError):
            O.admit_compact(rows, summary, text.replace('gap_slope', 'different_slope', 1))
        with self.assertRaises(ValueError):
            O.admit_compact(rows, summary, text + text.splitlines()[1] + '\r\n')
        bad = copy.deepcopy(summary)
        bad['rowSourceAuthenticated'] = True
        with self.assertRaises(ValueError):
            O.admit_compact(rows, bad, text)

    def test_import_then_independent_diagnosis_without_paper_epoch(self):
        rows, summary, _, raw = fixture()
        pins = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, imported, output = (root / name for name in ('scientific', 'imported', 'diagnosed'))
            for name, value in raw.items():
                I.write_bytes(source / name, value)
            with patch.object(O, 'INPUTS', pins), \
                 patch.object(O, 'scientific_identity', return_value={'fixture': 'not historical evidence'}), \
                 patch.object(IO, 'source_identity', return_value={'fixture': True}), \
                 patch.object(C.Paper, 'run', side_effect=AssertionError('No policy source execution')), \
                 patch.object(C.T.T.np.linalg, 'lstsq', side_effect=AssertionError('No refit')):
                prepared = O.prepare(source, imported, ROOT / 'configs')
                self.assertTrue(prepared['newImport'])
                self.assertFalse(prepared['paperBundleCreated'])
                self.assertFalse(prepared['historicalAcceptanceRevalidated'])
                self.assertFalse(prepared['newPublicPolicyEpoch'])
                self.assertEqual(O.load(imported, ROOT / 'configs')[:2], (rows, summary))
                before = (imported / 'original_import.json').read_bytes()
                self.assertFalse(O.prepare(source, imported, ROOT / 'configs')['newImport'])
                self.assertEqual(before, (imported / 'original_import.json').read_bytes())
                got = C.diagnose_original(imported, output, ROOT / 'configs')
                self.assertTrue(got['passed'])
                self.assertTrue(got['originalInputsByteMatched'])
                self.assertFalse(got['historicalParityEstablished'])
                self.assertFalse(got['historicalExecutionReconstructed'])
                self.assertEqual(got['newFitInvocations'], 0)
                again = C.diagnose_original(imported, output, ROOT / 'configs', checking=True)
                self.assertEqual(again['newDesignSchedules'], 0)
                self.assertFalse((imported / 'figure_bundle.json').exists())
                self.assertFalse((imported / 'policy_check.json').exists())
                stored = I.read(imported / 'inputs/public_summary.json')
                self.assertFalse(stored['rowSourceAuthenticated'])
                self.assertEqual(stored, summary)

    def test_wrong_scientific_bytes_cannot_be_self_sealed(self):
        _, _, _, raw = fixture()
        with tempfile.TemporaryDirectory() as temp:
            for name, value in raw.items():
                I.write_bytes(Path(temp) / name, value)
            # Unpatched production pins reject a complete but different fixture.
            with self.assertRaisesRegex(ValueError, 'Changed original scientific bytes'):
                O._read_inputs(temp)

    def test_mutated_import_or_recipe_is_rejected(self):
        _, _, _, raw = fixture()
        pins = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
        with tempfile.TemporaryDirectory() as temp:
            source, imported = Path(temp) / 'source', Path(temp) / 'imported'
            for name, value in raw.items():
                I.write_bytes(source / name, value)
            with patch.object(O, 'INPUTS', pins), \
                 patch.object(O, 'scientific_identity', return_value={'recipe': 'original'}), \
                 patch.object(IO, 'source_identity', return_value={'fixture': True}):
                O.prepare(source, imported, ROOT / 'configs')
                with patch.object(O, 'scientific_identity', return_value={'recipe': 'changed'}):
                    with self.assertRaises(ValueError):
                        O.load(imported, ROOT / 'configs')
                manifest = imported / 'original_import.json'
                original = manifest.read_bytes()
                forged = I.read(manifest)
                forged['binding']['boundary']['newPublicPolicyEpoch'] = True
                forged['contentSHA256'] = I.digest({key: value for key, value in forged.items()
                                                  if key != 'contentSHA256'})
                manifest.write_bytes(I.encoded(forged))
                with self.assertRaises(ValueError):
                    O.load(imported, ROOT / 'configs')
                manifest.write_bytes(original)
                forged = I.read(manifest)
                forged['payload']['independentGeometryChecked'] = True
                forged['contentSHA256'] = I.digest({key: value for key, value in forged.items()
                                                  if key != 'contentSHA256'})
                manifest.write_bytes(I.encoded(forged))
                with self.assertRaises(ValueError):
                    O.prepare(source, imported, ROOT / 'configs')
                manifest.write_bytes(original)
                path = imported / 'inputs/true_need_schedules.csv'
                path.write_bytes(path.read_bytes() + b'\n')
                with self.assertRaisesRegex(ValueError, 'Changed original scientific bytes'):
                    O.load(imported, ROOT / 'configs')


if __name__ == '__main__':
    unittest.main()
