"""Boundary tests for fresh preparation; no scientific draws are performed."""
import hashlib
import json

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.design_validation import c5_fresh_prepare as prepare
from src.design_validation import c5_fresh_population as population


class PreparationBoundaryTests(unittest.TestCase):
    def test_rejects_unpinned_manifest_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            manifest = tmp_path / 'population.json'
            manifest.write_text('{}')
            output = tmp_path / 'output'
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                prepare.prepare_population(manifest, '0' * 64, output,
                    Path(prepare.__file__).resolve().parents[2], expected_inputs={},
                    expected_sources={}, expected_entry_sha256='0' * 64, runtime={}, expected_runtime_sha256='0' * 64)
            self.assertFalse(output.exists())

    def test_stubbed_preparation_stages_ninety_roots(self):
        # All scientific assembly and admissions are stubs: this checks orchestration only.
        from src.design_validation import c5_input_assembly as assembly
        from src.design_validation import c5_public_inputs as inputs
        from src.design_validation import personal_time_inputs as personal
        from src.design_validation import personal_time_sensitivity as sensitivity
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = {name: base / (name + '.json') for name in
                     ('recipe', 'roots', 'lookup', 'latent', 'report', 'reference')}
            for path in paths.values():
                path.write_text('{}')
            width = float(1).hex()
            manifest = {'recipePath': 'recipe.json', 'rootsPath': 'roots.json',
                        'lookupPath': 'lookup.json',
                        'lookupSHA256': prepare.common._sha(paths['lookup'].read_bytes()),
                        'latentPath': 'latent.json', 'reportPaths': {width: 'report.json'}}
            manifest_path = base / 'population.json'
            manifest_path.write_bytes(prepare.common.canonical_bytes(manifest))
            paths.update({'manifest': manifest_path, 'report:' + width: paths.pop('report')})
            hashes = {k: prepare.common._sha(p.read_bytes()) for k, p in paths.items()}
            roots = [{'ordinal': i} for i in range(90)]
            rows = [{'rootKey': ['C5', i], 'config': {}, 'decision': {},
                     'reportWidth': 1.0} for i in range(90)]
            assembled = {'lookup': {'roots': rows}, 'geometryCertificates': [
                {'design': {}} for _ in range(9)]}
            runtime = {'stub': True}
            runtime_sha = prepare.common._sha(prepare.common.canonical_bytes(runtime))
            (base / 'checkout' / 'scripts').mkdir(parents=True)
            entry = base / 'checkout' / 'scripts' / 'fresh_c5_run.py'
            entry.write_bytes(b'test entry')
            entry_sha = prepare.common._sha(entry.read_bytes())
            with mock.patch.object(prepare.common, '_executing_root', return_value=base / 'checkout'), \
                 mock.patch.object(prepare.common, '_sources', return_value={}), \
                 mock.patch.object(prepare.common, 'sanitize_runtime', return_value=runtime), \
                 mock.patch.object(prepare.common, '_require_readonly_snapshot'), \
                 mock.patch.object(sensitivity, 'load_context', return_value={
                     'recipe': {'reference': 'reference.json'}, 'reference': {'recipe': {}}}), \
                 mock.patch.object(personal, 'INPUT_SHA256', {}), \
                 mock.patch.object(personal, 'selection', return_value=[]), \
                 mock.patch.object(personal, 'validate_roots', return_value=roots), \
                 mock.patch.object(assembly, 'assemble', return_value=assembled), \
                 mock.patch.object(prepare.fresh, 'admit', return_value={'identity_sha256': 'stub'}), \
                 mock.patch.object(prepare.fresh, 'admit_reference', return_value={'identity_sha256': 'stub'}), \
                 mock.patch.object(prepare.fresh, 'check_unchanged'):
                result = prepare.prepare_population(manifest_path, hashes['manifest'], base / 'out',
                    base / 'checkout', expected_inputs=hashes, expected_sources={},
                    expected_entry_sha256=entry_sha, runtime=runtime, expected_runtime_sha256=runtime_sha)
                with self.assertRaisesRegex(ValueError, 'Entry source identity'):
                    prepare.prepare_population(manifest_path, hashes['manifest'], base / 'out-bad',
                        base / 'checkout', expected_inputs=hashes, expected_sources={},
                        expected_entry_sha256='0' * 64, runtime=runtime, expected_runtime_sha256=runtime_sha)
            self.assertEqual(len(result['roots']), 90)
            self.assertFalse(result['statisticalInferenceAuthorized'])
            self.assertEqual(len(list((base / 'out').glob('root-*.json'))), 90)

    def test_stubbed_load_and_rejections(self):
        # Admissions return synthetic roots; no policy, law or random stream is evaluated.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input-manifest.json'
            keys = sorted(population.ROSTER)
            records = [{'rootKey': ['C5', *key], 'specPath': f'root-{i}.json',
                        'inputs': {}, 'producerIdentitySHA256': 'stub',
                        'referenceIdentitySHA256': 'stub'} for i, key in enumerate(keys)]
            manifest = {'mode': 'fresh-input-unverified', 'historicalAuthenticated': False,
                        'statisticalInferenceAuthorized': False, 'populationInputs': {},
                        'sources': {}, 'runtime': {}, 'roots': records}
            def admission(**kwargs):
                i = int(kwargs['spec_path'].stem.split('-')[1])
                variant, index = keys[i]
                return {'identity_sha256': 'stub', 'spec': {'rawRoot': {
                    'conditionId': 'C5', 'scheduleVariant': variant, 'mainIndex': index}}}
            def load():
                raw = prepare.common.canonical_bytes(manifest)
                path.write_bytes(raw)
                return prepare.load_prepared(path, prepare.common._sha(raw), Path(directory), runtime={})
            with mock.patch.object(prepare.common, '_executing_root', return_value=Path(directory)), \
                 mock.patch.object(prepare.common, '_sources', return_value={}), \
                 mock.patch.object(prepare.common, 'sanitize_runtime', return_value={}), \
                 mock.patch.object(prepare.fresh, 'admit', side_effect=admission), \
                 mock.patch.object(prepare.fresh, 'admit_reference', side_effect=admission):
                contexts, refs = load()
                self.assertEqual((len(contexts), len(refs)), (90, 90))
                manifest['roots'][1]['rootKey'] = manifest['roots'][0]['rootKey']
                with self.assertRaisesRegex(ValueError, 'invalid prepared root'):
                    load()
                manifest['roots'][1]['rootKey'] = ['C5', *keys[1]]
                manifest['roots'][0]['referenceIdentitySHA256'] = 'tampered'
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    load()
                with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                    prepare.load_prepared(path, '0' * 64, Path(directory), runtime={})


    def test_rejects_incomplete_manifest_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            manifest = tmp_path / 'population.json'
            raw = json.dumps({'recipePath': 'missing'}).encode()
            manifest.write_bytes(raw)
            output = tmp_path / 'output'
            with self.assertRaisesRegex(ValueError, 'complete population'):
                prepare.prepare_population(manifest, hashlib.sha256(raw).hexdigest(), output,
                    Path(prepare.__file__).resolve().parents[2], expected_inputs={},
                    expected_sources={}, expected_entry_sha256='0' * 64, runtime={}, expected_runtime_sha256='0' * 64)
            self.assertFalse(output.exists())
