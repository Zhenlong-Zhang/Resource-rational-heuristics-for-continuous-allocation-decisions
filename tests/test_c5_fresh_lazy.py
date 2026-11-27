"""Stub-only contract checks for the unvalidated lazy candidate."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.design_validation import c5_fresh_lazy as lazy


class LazyInventoryTests(unittest.TestCase):
    def setUp(self):
        self.spec = {'rawRoot': {'conditionId': 'C5', 'scheduleVariant': 0, 'mainIndex': 1},
                     'lookupPath': 'lookup', 'latentPath': 'latent',
                     'lookupSHA256': lazy.common._sha(b'input'), 'reportPaths': {'w': 'report'}}
        self.spec_bytes = json.dumps(self.spec).encode()
        self.record = {'rootKey': ['C5', 0, 1], 'specPath': 'root.json',
                       'inputs': {'spec': lazy.common._sha(self.spec_bytes),
                                  'lookup': lazy.common._sha(b'input'),
                                  'latent': lazy.common._sha(b'input'),
                                  'report:w': lazy.common._sha(b'input')},
                       'producerIdentitySHA256': 'a' * 64, 'referenceIdentitySHA256': 'a' * 64}
        self.manifest = {'mode': 'fresh-input-unverified', 'historicalAuthenticated': False,
                         'statisticalInferenceAuthorized': False, 'populationInputs': {},
                         'sources': {}, 'runtime': {}, 'roots': [self.record]}

    def inventory(self):
        raw = json.dumps(self.manifest).encode()
        path = Path('/stub/input-manifest.json')
        def read(item):
            return raw if item == path else (self.spec_bytes if item.name == 'root.json' else b'input')
        with patch.object(lazy.common, '_file', side_effect=read), \
             patch.object(Path, 'resolve', lambda self, **kwargs: self), \
             patch.object(lazy.common, '_sources', return_value={}), \
             patch.object(lazy.common, 'sanitize_runtime', return_value={}):
            return lazy._inventory(path, lazy.common._sha(raw), Path('/stub/repo'), {}, {(0, 1)})

    def test_accepts_pinned_single_test_roster(self):
        manifest, entries = self.inventory()
        self.assertEqual(manifest['mode'], 'fresh-input-unverified')
        self.assertEqual(set(entries), {(0, 1)})

    def test_rejects_duplicate_root(self):
        self.manifest['roots'].append(dict(self.record))
        with self.assertRaises(ValueError):
            self.inventory()

    def test_rejects_changed_spec(self):
        self.record['inputs']['spec'] = '0' * 64
        with self.assertRaises(ValueError):
            self.inventory()

    def test_rejects_wrong_root_identity(self):
        self.record['rootKey'] = ['C5', 0, 2]
        with self.assertRaises(ValueError):
            self.inventory()

    def test_rejects_partial_roster_outside_test_domain(self):
        with self.assertRaises(ValueError):
            lazy.run_lazy('unused', 'unused', 'unused', 'unused', runtime={}, domain='fresh',
                          run_id='x', replicates=[1], expected_roster={(0, 1)})

    def test_rejects_later_replicate_for_partial_roster(self):
        with self.assertRaises(ValueError):
            lazy.run_lazy('unused', 'unused', 'unused', 'unused', runtime={}, domain='test-only',
                          run_id='x', replicates=[2], expected_roster={(0, 1)})


class LazyRunStubTests(unittest.TestCase):
    """Exercise orchestration only: no scientific streams, producer or receiver."""

    def setUp(self):
        from src.design_validation import c5_fresh_inputs, c5_fresh_streams
        from src.design_validation import c5_fresh_streams_reference
        from src.design_validation import c5_public_trajectory, c5_public_trajectory_reference

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.manifest_path = self.root / 'manifest.json'
        self.manifest_path.write_bytes(b'synthetic-manifest')
        self.digest = lazy.common._sha(self.manifest_path.read_bytes())
        self.output = self.root / 'output'
        self.entries = {
            key: {'specPath': f'{key[0]}-{key[1]}.json', 'inputs': {},
                  'producerIdentitySHA256': 'a' * 64}
            for key in lazy.population.ROSTER
        }
        self.by_name = {record['specPath']: key for key, record in self.entries.items()}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        def replace(obj, name, **kwargs):
            return self.stack.enter_context(patch.object(obj, name, **kwargs))

        replace(lazy.common, '_executing_root', return_value=self.root)
        replace(lazy.common, '_require_ignored', return_value=None)
        replace(lazy, '_inventory', return_value=({'sources': {}}, self.entries))
        replace(lazy.common, 'sanitize_runtime', return_value={})

        def admit(**kwargs):
            v, m = self.by_name[kwargs['spec_path'].name]
            spec = {'rawRoot': {'scheduleVariant': v, 'mainIndex': m},
                    'config': {}, 'initialDecision': {}, 'design': {}}
            return {'identity_sha256': 'a' * 64, 'spec_snapshot_sha256': 'b' * 64,
                    'repository_root': self.root, 'spec': spec, 'p_latent': None,
                    'p_table': None, 'ps': None, 'r_latent': None, 'r_table': None,
                    'rs': None}

        replace(c5_fresh_inputs, 'admit', side_effect=admit)
        replace(c5_fresh_inputs, 'admit_reference', side_effect=admit)
        replace(c5_fresh_inputs, 'check_unchanged', return_value=None)
        def stream(identity, context, rep):
            return SimpleNamespace(identity=identity, rep=rep)
        replace(c5_fresh_streams, 'ProducerStreams', side_effect=stream)
        replace(c5_fresh_streams_reference, 'ReferenceStreams', side_effect=stream)

        def produce(root, config, initial, latent, table, design, rep, *, spec, producer_streams):
            identity = producer_streams.identity
            return {'runIdentity': {'domain': identity.domain, 'runId': identity.run_id,
                                    'inputIdentitySHA256': identity.input_identity_sha256},
                    'event': [root['scheduleVariant'], root['mainIndex'], rep]}

        def receive(root, config, initial, latent, table, design, rep, result, *, spec,
                    reference_streams):
            identity = reference_streams.identity
            expected = {'domain': identity.domain, 'runId': identity.run_id,
                        'inputIdentitySHA256': identity.input_identity_sha256}
            if (result['runIdentity'] != expected or
                    result['event'] != [root['scheduleVariant'], root['mainIndex'], rep]):
                raise ValueError('Synthetic receiver rejected identity or event')

        self.producer = replace(c5_public_trajectory, 'produce_fresh_root', side_effect=produce)
        self.receiver = replace(c5_public_trajectory_reference, 'receive_fresh_root', side_effect=receive)

    def run_stub(self, replicates=(1,), *, run_id='stub-run', verify_only=False):
        return lazy.run_lazy(self.manifest_path, self.digest, self.output, self.root,
                             runtime={}, domain='test-only', run_id=run_id,
                             replicates=replicates, verify_only=verify_only)

    def checkpoint(self, key=(0, 1), rep=1):
        return self.output / (lazy.common.member_id(rep, *key) + '.json')

    def test_full_prefix_binds_all_90_checkpoints(self):
        result = self.run_stub()
        prefix = result['prefixManifest']
        self.assertEqual(len(prefix['checkpoints']), 90)
        self.assertTrue(prefix['population']['fullRoster'])
        self.assertFalse(prefix['statisticalAuthorization'])
        self.assertEqual({row['memberId'] for row in prefix['checkpoints']},
                         {lazy.common.member_id(1, *key) for key in self.entries})
        for row in prefix['checkpoints']:
            self.assertEqual(row['checkpointSHA256'],
                             lazy.common._sha((self.output / (row['memberId'] + '.json')).read_bytes()))
        self.assertEqual((self.output / 'fresh-prefix-manifest.json').read_bytes(),
                         lazy.common.canonical_bytes(prefix))
        self.assertEqual(self.producer.call_count, 90)
        self.assertEqual(self.receiver.call_count, 90)

    def test_missing_prefix_root_refuses_later_replicate(self):
        self.run_stub()
        self.checkpoint().unlink()
        self.producer.reset_mock()
        with self.assertRaisesRegex(ValueError, 'checkpoint missing'):
            self.run_stub((2,))
        self.producer.assert_not_called()
        self.assertFalse(self.checkpoint(rep=2).exists())

    def test_cross_run_checkpoint_is_rejected_before_receiver(self):
        self.run_stub()
        self.receiver.reset_mock()
        with self.assertRaisesRegex(ValueError, 'run mismatch'):
            self.run_stub(run_id='different-run', verify_only=True)
        self.receiver.assert_not_called()

    def test_rehashed_event_tampering_is_rejected_by_receiver(self):
        self.run_stub()
        path = self.checkpoint()
        value = json.loads(path.read_bytes())
        value['payload']['result']['event'][2] = 99
        value['payload']['resultSHA256'] = lazy.common._sha(
            lazy.common.canonical_bytes(value['payload']['result']))
        path.write_bytes(lazy.common.canonical_bytes(value))
        self.receiver.reset_mock()
        with self.assertRaisesRegex(ValueError, 'Synthetic receiver rejected'):
            self.run_stub(verify_only=True)
        self.assertEqual(self.receiver.call_count, 1)

    def test_failed_run_retains_lock_and_resume_only_produces_missing(self):
        original = self.producer.side_effect
        def fail_at_fourth(*args, **kwargs):
            if self.producer.call_count == 4:
                raise RuntimeError('synthetic interruption')
            return original(*args, **kwargs)
        self.producer.side_effect = fail_at_fourth
        with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            self.run_stub()
        marker = self.output / '.c5-population-active'
        self.assertTrue(marker.exists())
        existing = {p.name: p.read_bytes() for p in self.output.glob('rep-*.json')}
        self.assertEqual(len(existing), 3)
        with self.assertRaisesRegex(ValueError, 'Unfinished or concurrent'):
            self.run_stub()
        marker.unlink()  # Simulated manual review and explicit clearance.
        self.producer.side_effect = original
        self.producer.reset_mock()
        self.run_stub()
        self.assertEqual(self.producer.call_count, 87)
        self.assertEqual(existing, {name: (self.output / name).read_bytes() for name in existing})
        self.assertFalse(marker.exists())

    def test_verify_only_does_not_create_checkpoint_or_call_producer(self):
        self.run_stub()
        self.checkpoint().unlink()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.producer.reset_mock()
        with self.assertRaisesRegex(ValueError, 'checkpoint missing'):
            self.run_stub(verify_only=True)
        self.producer.assert_not_called()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_producer_and_receiver_refusals_propagate_without_checkpoint(self):
        for failing in ('producer', 'receiver'):
            with self.subTest(failing=failing):
                out = self.output
                self.output = self.root / failing
                target = getattr(self, failing)
                original = target.side_effect
                target.side_effect = RuntimeError(f'{failing} refused')
                try:
                    with self.assertRaisesRegex(RuntimeError, f'{failing} refused'):
                        self.run_stub()
                    self.assertFalse(self.checkpoint().exists())
                    self.assertTrue((self.output / '.c5-population-active').exists())
                finally:
                    target.side_effect = original
                    self.output = out


if __name__ == '__main__':
    unittest.main()
