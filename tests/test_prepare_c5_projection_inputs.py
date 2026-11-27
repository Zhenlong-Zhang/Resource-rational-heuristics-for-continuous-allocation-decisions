"""Mocked preparation tests; synthetic-generation integration requires registration."""
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_c5_projection_inputs as producer


class PreparationTests(unittest.TestCase):
    def test_supplied_report_rejects_symlink_fifo_and_oversize(self):
        width = float(3.0).hex()
        name = hashlib.sha256(width.encode('ascii')).hexdigest() + '.json'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            target = Path(directory) / 'target'
            target.write_bytes(b'{}')
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'regular file'):
                producer.supplied_report(width, directory)
            path.unlink()
            if hasattr(os, 'mkfifo'):
                os.mkfifo(path)
                with self.assertRaisesRegex(ValueError, 'regular file'):
                    producer.supplied_report(width, directory)
                path.unlink()
            with path.open('wb') as stream:
                stream.truncate(producer.MAX_REPORT_BYTES + 1)
            with self.assertRaisesRegex(ValueError, '16 MiB'):
                producer.supplied_report(width, directory)

    def test_supplied_report_checks_bytes_and_independent_table(self):
        width = float(3.0).hex()
        data = producer.json_bytes({'key': 'scientific-table'})
        name = hashlib.sha256(width.encode('ascii')).hexdigest() + '.json'
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(producer.ACCEPTED_TABLE_SHA, {width: producer.sha(data)}), \
                patch.object(producer.canonical_report_reference, 'verify_report_table') as verify:
            path = Path(directory) / name
            path.write_bytes(data)
            self.assertEqual(producer.supplied_report(width, directory), data)
            verify.assert_called_once_with(json.loads(data), width)
            verify.side_effect = ValueError('invalid scientific table')
            with self.assertRaisesRegex(ValueError, 'invalid scientific table'):
                producer.supplied_report(width, directory)
            path.write_bytes(data + b' ')
            with self.assertRaisesRegex(ValueError, 'byte identity mismatch'):
                producer.supplied_report(width, directory)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                producer.supplied_report(width, directory)

    def test_existing_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'existing'
            target.mkdir()
            marker = target / 'marker'
            marker.write_text('keep')
            with self.assertRaises(FileExistsError):
                producer.prepare(target)
            self.assertEqual(marker.read_text(), 'keep')

    def test_mismatch_is_recorded_and_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'checkout'
            configs = root / 'configs'
            configs.mkdir(parents=True)
            names = ['personal_time_sensitivity.json', 'reference.json', 'base.json',
                     'instance.json', 'design.json', 'identity.json',
                     'canonical_c5_latent.json', 'c5_fixed_initial_policy.json']
            for name in names:
                (configs / name).write_bytes(name.encode())
            context = {'recipe': {'reference': 'reference.json'}, 'reference': {'recipe': {
                'base': 'base.json', 'instance': 'instance.json', 'design': 'design.json',
                'root_identity': 'identity.json'}}}
            output = Path(directory) / 'output'
            lookup_sha = producer.sha((configs / 'c5_fixed_initial_policy.json').read_bytes())
            latent_sha = producer.sha((configs / 'canonical_c5_latent.json').read_bytes())
            with patch.object(producer, 'ROOT', root), \
                    patch.object(producer.personal_time_sensitivity, 'safe_output'), \
                    patch.object(producer.personal_time_sensitivity, 'load_context', return_value=context), \
                    patch.object(producer.c5_input_assembly, 'LOOKUP_BYTES_SHA256', lookup_sha), \
                    patch.object(producer.c5_input_assembly, 'LATENT_BYTES_SHA256', latent_sha), \
                    patch.object(producer.c5_public_inputs, 'load_fixed_inputs',
                                 return_value={'roots': [{'reportWidth': 3.0}]}), \
                    patch.object(producer.personal_time_inputs, 'selection', return_value=[]), \
                    patch.object(producer.personal_time_inputs, 'prepare', return_value={}), \
                    patch.object(producer.canonical_report, 'generate_report_table', return_value={'key': 'test'}), \
                    patch.object(producer.canonical_report_reference, 'verify_report_table'), \
                    patch.object(producer.c5_input_assembly, 'assemble') as assemble:
                with self.assertRaisesRegex(ValueError, 'new unverified identity'):
                    producer.prepare(output)
                assemble.assert_not_called()
                accepted = producer.sha(producer.json_bytes({'key': 'test'}))
                with patch.dict(producer.ACCEPTED_TABLE_SHA, {float(3.0).hex(): accepted}):
                    failed = Path(directory) / 'assembly-failed'
                    assemble.side_effect = ValueError('assembly refused')
                    with self.assertRaisesRegex(ValueError, 'assembly refused'):
                        producer.prepare(failed)
                    self.assertFalse((failed / 'population_manifest.json').exists())
                    completed = Path(directory) / 'completed'

                    def check_before_publication(*args):
                        self.assertFalse((completed / 'population_manifest.json').exists())

                    assemble.side_effect = check_before_publication
                    result = producer.prepare(completed)
                    manifest_bytes = (completed / 'population_manifest.json').read_bytes()
                    diagnostic = json.loads((completed / 'preparation_summary.json').read_bytes())
                    self.assertNotIn('assemblyPassed', diagnostic)
                    self.assertEqual(diagnostic['summaryScope'],
                                     'diagnostic-only-manifest-is-sole-completion-signal')
                    self.assertEqual(result['manifestSHA256'], producer.sha(manifest_bytes))
                    self.assertEqual(result['reportProducts'][float(3.0).hex()]['source'],
                                     'freshly_generated')
                    supplied_dir = Path(directory) / 'supplied'
                    supplied_dir.mkdir()
                    width = float(3.0).hex()
                    supplied_data = producer.json_bytes({'key': 'test'})
                    (supplied_dir / (hashlib.sha256(width.encode('ascii')).hexdigest() + '.json')).write_bytes(supplied_data)
                    supplied_output = Path(directory) / 'supplied-output'
                    assemble.side_effect = None
                    with patch.object(producer.canonical_report, 'generate_report_table',
                                      side_effect=AssertionError('must not generate supplied tables')):
                        supplied_result = producer.prepare(supplied_output, report_dir=supplied_dir)
                    self.assertEqual(supplied_result['reportProducts'][width]['source'],
                                     'supplied_accepted_product')
                    self.assertEqual((supplied_output / 'reports' / 'report_0.json').read_bytes(), supplied_data)
                    supplied_file = supplied_dir / (hashlib.sha256(width.encode('ascii')).hexdigest() + '.json')
                    supplied_file.write_bytes(supplied_data + b' ')
                    corrupt_output = Path(directory) / 'corrupt-output'
                    with self.assertRaisesRegex(ValueError, 'byte identity mismatch'):
                        producer.prepare(corrupt_output, report_dir=supplied_dir)
                    self.assertFalse((corrupt_output / 'population_manifest.json').exists())
                    supplied_file.unlink()
                    missing_output = Path(directory) / 'missing-output'
                    with self.assertRaises(FileNotFoundError):
                        producer.prepare(missing_output, report_dir=supplied_dir)
                    self.assertFalse((missing_output / 'population_manifest.json').exists())
                    self.assertEqual(set(json.loads(manifest_bytes)),
                                     {'recipePath', 'rootsPath', 'lookupPath', 'lookupSHA256',
                                      'latentPath', 'reportPaths'})
                    interrupted = Path(directory) / 'publication-failed'
                    assemble.side_effect = None
                    with patch.object(producer.personal_time_inputs, 'write_bytes',
                                      side_effect=OSError('publication failed')):
                        with self.assertRaisesRegex(OSError, 'publication failed'):
                            producer.prepare(interrupted)
                    self.assertFalse((interrupted / 'population_manifest.json').exists())
                    diagnostic = json.loads((interrupted / 'preparation_summary.json').read_bytes())
                    self.assertNotIn('assemblyPassed', diagnostic)
                    self.assertEqual(diagnostic['summaryScope'],
                                     'diagnostic-only-manifest-is-sole-completion-signal')
            summary = json.loads((output / 'preparation_summary.json').read_text())
            self.assertFalse(summary['acceptedProductBytesMatch'])
            self.assertNotIn('assemblyPassed', summary)
            self.assertEqual(summary['c5PolicyDraws'], 0)
            self.assertIn('synthetic', summary['nativeSyntheticAncestry'])
            self.assertEqual(next(iter(summary['reportProducts'].values()))['identity'],
                             'new_unverified_table_bytes')
            self.assertFalse((output / 'population_manifest.json').exists())

    def test_checkout_output_refused(self):
        with patch.object(producer.personal_time_sensitivity, 'safe_output'):
            with self.assertRaisesRegex(ValueError, 'outside the source checkout'):
                producer.prepare(producer.ROOT / 'generated-output')


@unittest.skipUnless(os.environ.get('C5_REPORT_PRODUCTS_INTEGRATION') == '1',
                     'opt-in scheduled verification of supplied accepted report products only')
class SuppliedProductsIntegration(unittest.TestCase):
    def test_all_real_supplied_accepted_products(self):
        directory = os.environ.get('C5_ACCEPTED_REPORT_DIR')
        self.assertTrue(directory, 'C5_ACCEPTED_REPORT_DIR must name supplied scientific products')
        self.assertEqual(len(producer.ACCEPTED_TABLE_SHA), 3)
        for width, expected in producer.ACCEPTED_TABLE_SHA.items():
            with self.subTest(width=width):
                self.assertEqual(producer.sha(producer.supplied_report(width, directory)), expected)


@unittest.skipUnless(os.environ.get('C5_PREPARE_SCHEDULED_INTEGRATION') == '1',
                     'opt-in actual-source scheduled integration only')
class ScheduledSourceIntegration(unittest.TestCase):
    def test_actual_generation_and_population_loader(self):
        from src.design_validation import c5_public_reproduction
        self.assertTrue(os.environ.get('C5_PREFIX_PATH'), 'scheduled gate requires C5_PREFIX_PATH')
        self.assertTrue(os.environ.get('C5_RUNTIME_JSON'), 'scheduled gate requires C5_RUNTIME_JSON')
        prefix = Path(os.environ['C5_PREFIX_PATH']).resolve(strict=True)
        runtime = json.loads(Path(os.environ['C5_RUNTIME_JSON']).read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'population'
            summary = producer.prepare(output, node=os.environ.get('C5_NODE', 'node'))
            manifest_bytes = (output / 'population_manifest.json').read_bytes()
            manifest = json.loads(manifest_bytes)
            self.assertEqual(set(manifest), {'recipePath', 'rootsPath', 'lookupPath',
                                             'lookupSHA256', 'latentPath', 'reportPaths'})
            self.assertEqual(summary['manifestSHA256'], producer.sha(manifest_bytes))
            self.assertTrue(summary['acceptedProductBytesMatch'])
            self.assertTrue(summary['assemblyPassed'])
            self.assertEqual(summary['c5PolicyDraws'], 0)
            for name in ('recipePath', 'rootsPath', 'lookupPath', 'latentPath'):
                self.assertTrue((output / manifest[name]).is_file())
            for path in manifest['reportPaths'].values():
                self.assertTrue((output / path).is_file())
            for path in output.rglob('*'):
                if path.is_file():
                    path.chmod(0o444)
            population = c5_public_reproduction.load_population(
                output / 'population_manifest.json', summary['manifestSHA256'], prefix,
                producer.sha(prefix.read_bytes()), producer.ROOT, runtime=runtime,
                input_mode='accepted-prefix')
            self.assertEqual(len(population['contexts']), 90)


if __name__ == '__main__':
    unittest.main()
