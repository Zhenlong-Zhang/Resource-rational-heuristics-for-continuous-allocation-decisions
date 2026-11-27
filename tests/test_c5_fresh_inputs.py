"""Admission contract tests; no policy or random streams are executed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation import c5_fresh_inputs as fresh
from src.design_validation import c5_public_reproduction as common
from src.design_validation import c5_public_streams as streams
from src.design_validation import c5_public_trajectory as producer
from src.design_validation import c5_public_trajectory_reference as receiver


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class FreshInputsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        (self.path / 'scripts').mkdir()
        (self.path / 'scripts' / 'fresh_c5_run.py').write_bytes(b'test entry')
        (self.path / 'lookup').write_bytes(b'lookup')
        (self.path / 'latent').write_bytes(b'{}')
        (self.path / 'report').write_bytes(b'report')
        self.spec = {'rawRoot': {'identity': 'root'}, 'config': {'task': {'report_width': 1.0}},
                     'initialDecision': {}, 'design': {}, 'lookupPath': 'lookup',
                     'lookupSHA256': digest(b'lookup'), 'geometryCertificate': {},
                     'latentPath': 'latent', 'reportPaths': {float(1).hex(): 'report'}}
        self.write_spec()
        self.inputs = {'spec': digest((self.path / 'spec').read_bytes()),
                       'lookup': digest(b'lookup'), 'latent': digest(b'{}'),
                       'report:' + float(1).hex(): digest(b'report')}
        self.runtime = {'identity': 'test-observation'}
        self.runtime_sha = digest(common.canonical_bytes(self.runtime))
        self.sources = {'src/example.py': 'source-digest'}

    def write_spec(self):
        (self.path / 'spec').write_bytes(json.dumps(self.spec).encode())

    def call(self, inputs=None, sources=None, runtime_sha=None):
        return fresh.admit(self.path / 'spec', self.inputs if inputs is None else inputs,
                           self.path, runtime=self.runtime,
                           expected_sources=self.sources if sources is None else sources,
                           expected_runtime_sha256=self.runtime_sha if runtime_sha is None else runtime_sha)

    def test_separate_admission_and_root_checks(self):
        class Snapshot:
            def __init__(self, spec, latent, reports):
                self.table = ((float(1).hex(), object()),)
        with patch.object(common, '_executing_root', return_value=self.path), \
             patch.object(common, '_sources', return_value=self.sources), \
             patch.object(common, 'sanitize_runtime', return_value=self.runtime), \
             patch.object(common, '_require_readonly_snapshot') as readonly, \
             patch.object(streams, 'ValidatedLatentConfig', side_effect=lambda x: object()), \
             patch.object(producer, 'AdmittedSpec', Snapshot), \
             patch.object(receiver, 'AdmittedSpec', Snapshot), \
             patch.object(producer, '_validate_root') as p_check, \
             patch.object(receiver, 'validate_spec') as r_check:
            result = self.call()
            self.assertIsNot(result['ps'], result['rs'])
            self.assertIsNot(result['p_latent'], result['r_latent'])
            self.assertFalse(result['historicalAuthenticated'])
            self.assertFalse(result['statisticalInferenceAuthorized'])
            p_check.assert_called_once()
            r_check.assert_called_once()
            self.assertGreaterEqual(readonly.call_count, 2)
            (self.path / 'report').write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError, 'changed'):
                fresh.check_unchanged(result)

    def test_missing_or_replaced_input_and_identity_are_refused(self):
        with patch.object(common, '_executing_root', return_value=self.path), \
             patch.object(common, '_sources', return_value=self.sources), \
             patch.object(common, 'sanitize_runtime', return_value=self.runtime):
            with self.assertRaises(ValueError):
                self.call(inputs={'spec': self.inputs['spec']})
            with self.assertRaisesRegex(ValueError, 'Source identity'):
                self.call(sources={})
            with self.assertRaisesRegex(ValueError, 'Runtime identity'):
                self.call(runtime_sha='0' * 64)
            (self.path / 'latent').write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'Input byte identity'):
                self.call()

    def test_reference_admission_does_not_invoke_producer(self):
        class Snapshot:
            def __init__(self, spec, latent, reports):
                self.table = ((float(1).hex(), object()),)
        with patch.object(common, '_executing_root', return_value=self.path), \
             patch.object(common, '_sources', return_value=self.sources), \
             patch.object(common, 'sanitize_runtime', return_value=self.runtime), \
             patch.object(common, '_require_readonly_snapshot'), \
             patch.object(streams, 'ValidatedLatentConfig', side_effect=lambda x: object()), \
             patch.object(receiver, 'AdmittedSpec', Snapshot), \
             patch.object(receiver, 'validate_spec') as r_check, \
             patch.object(producer, 'AdmittedSpec', side_effect=AssertionError('producer called')), \
             patch.object(producer, '_validate_root', side_effect=AssertionError('producer called')):
            result = fresh.admit_reference(
                self.path / 'spec', self.inputs, self.path, runtime=self.runtime,
                expected_sources=self.sources, expected_runtime_sha256=self.runtime_sha)
            r_check.assert_called_once()
            self.assertNotIn('ps', result)
            self.assertNotIn('p_latent', result)

    def test_duplicate_spec_field_rejected(self):
        (self.path / 'spec').write_bytes(b'{"rawRoot":{},"rawRoot":{}}')
        with patch.object(common, '_executing_root', return_value=self.path):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                self.call()

    def test_missing_file_refused(self):
        (self.path / 'report').unlink()
        with patch.object(common, '_executing_root', return_value=self.path):
            with self.assertRaises(FileNotFoundError):
                self.call()
