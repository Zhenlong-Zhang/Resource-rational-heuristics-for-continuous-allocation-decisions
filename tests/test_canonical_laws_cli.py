"""Check public no-draw entry, fixed-roster admission and refusal behavior."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/'scripts/reproduce_canonical_laws.py'


class CanonicalEntryTests(unittest.TestCase):
    def test_unmatched_product_is_labeled_unverified(self):
        classify = runpy.run_path(str(ENTRY))['classify_product']
        self.assertEqual(classify('0x1.9333333333334p+1', '0'*64),
                         (False, 'new_unverified_table_bytes'))

    def test_changed_adopted_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            shutil.copytree(ROOT/'src', repo/'src')
            (repo/'configs').mkdir()
            shutil.copy(ROOT/'configs/canonical_c5_latent.json', repo/'configs')
            (repo/'configs/canonical_c5_latent.sha256').write_text('0'*64)
            (repo/'scripts').mkdir()
            shutil.copy(ENTRY, repo/'scripts')
            result = subprocess.run([sys.executable, str(repo/'scripts'/ENTRY.name),
                                     '--config', str(ROOT/'configs/canonical_c5_latent.json'),
                                     '--digest', str(ROOT/'configs/canonical_c5_latent.sha256'),
                                     '--output', str(repo/'result')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Adopted scientific configuration pin mismatch', result.stderr)

    def test_entry_and_refusals(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            law = {'counts': [2**53] + [0]*99, 'cumulativeHex': [float(1).hex()]*100}
            config = {'schema': 'fixed-c5-latent-53bit-v1', 'denominator': 2**53,
                      'recipientOrder': ['A', 'B'], 'lastCategoryIncludesFallback': True,
                      'identity': 'adopted-fixed-integer-scientific-input-not-fresh-exp-log',
                      'roots': [{'rootKey': ['C5', v, m], 'recipientLaws': [law, law]}
                                for v in range(9) for m in range(1, 11)]}
            source, digest, output = folder/'config.json', folder/'config.sha', folder/'output'

            def write(value):
                data = json.dumps(value).encode()
                source.write_bytes(data)
                digest.write_text(hashlib.sha256(data).hexdigest())

            def run(destination, allow=True):
                extra = ['--allow-unverified-config'] if allow else []
                return subprocess.run([sys.executable, str(ENTRY), '--config', str(source),
                                       '--digest', str(digest), '--output', str(destination), *extra],
                                      capture_output=True, text=True)

            write(config)
            refused = run(folder/'adoption-refusal', allow=False)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('not the adopted public scientific input', refused.stderr)
            result = run(output)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual((summary['fixedRootCount'], summary['fixedRecipientLaws']), (90, 180))
            self.assertEqual(summary['newPolicyDraws'], 0)
            self.assertFalse(summary['fixedInputAdopted'])
            self.assertEqual(len(summary['tables']), 3)
            self.assertTrue(all(x['historicalPayloadParity'] is None for x in summary['tables']))
            accepted = {
                '0x1.9333333333334p+1': 'ba563fd341874d99c0a74df66a2dee4c7aea19d7eb52ce64b7569dccfebba85c',
                '0x1.6ccccccccccccp+1': 'f85a6c3b83cca15bc83b8b35c57754a3a43beeb59f31cb0fd4128d2d4453d95c',
                '0x1.8000000000000p+1': 'e3a4632ba2924afae5df9be893c9c2e3b209e8250d31c9c386d6c3e922728bb6',
            }
            for table in summary['tables']:
                actual = hashlib.sha256((output/table['filename']).read_bytes()).hexdigest()
                self.assertEqual(table['productSHA256'], actual)
                self.assertEqual(table['acceptedProductByteMatch'], actual == accepted[table['width']])
                self.assertEqual(table['productIdentity'],
                                 'accepted_finite_table_bytes' if table['acceptedProductByteMatch']
                                 else 'new_unverified_table_bytes')
            self.assertNotEqual(run(output).returncode, 0)
            table = summary['tables'][0]['filename']
            supplied = folder/'supplied'
            supplied.mkdir()
            (supplied/table).write_bytes((output/table).read_bytes())
            substituted = subprocess.run([sys.executable, str(ENTRY), '--config', str(source),
                                          '--digest', str(digest), '--output', str(folder/'other'),
                                          '--allow-unverified-config', '--historical-tables', str(supplied)],
                                         capture_output=True, text=True)
            self.assertNotEqual(substituted.returncode, 0)
            self.assertIn('Historical table identity mismatch', substituted.stderr)
            config['roots'][0]['rootKey'][2] = 2
            write(config)
            bad_roster = run(folder/'bad-roster')
            self.assertNotEqual(bad_roster.returncode, 0)
            self.assertIn('Invalid fixed scientific configuration', bad_roster.stderr)
            digest.write_text('0'*64)
            bad_digest = run(folder/'bad-digest')
            self.assertNotEqual(bad_digest.returncode, 0)
            self.assertIn('configuration digest mismatch', bad_digest.stderr)


if __name__ == '__main__':
    unittest.main()
