"""Structural tests only: no sampling or scientific verification."""
import unittest
import tempfile
from pathlib import Path
from unittest import mock
from contextlib import ExitStack

from src.design_validation import c5_fresh_population as population
from src.design_validation import c5_public_reproduction as common
from src.design_validation.c5_fresh_domain import RunIdentity


def context(v, m):
    spec = {'rawRoot': {'scheduleVariant': v, 'mainIndex': m}}
    identity = {'root': [v, m]}
    return {'spec': spec, 'identity': identity,
            'identity_sha256': common._sha(common.canonical_bytes(identity)),
            'spec_snapshot_sha256': common._sha(common.canonical_bytes(spec))}


class PopulationStructureTests(unittest.TestCase):
    def test_exact_full_and_test_rosters(self):
        contexts = [context(v, m) for v, m in sorted(population.ROSTER)]
        full = population.population_manifest(contexts, domain='fresh', run_id='example')
        self.assertTrue(full['fullRoster'])
        self.assertFalse(full['statisticalAuthorization'])
        small = population.population_manifest(contexts[:1], domain='test-only', run_id='test',
                                               expected_roster={(0, 1)})
        self.assertFalse(small['fullRoster'])
        for bad in (contexts[:-1], contexts + contexts[:1]):
            with self.assertRaises(ValueError):
                population.population_manifest(bad, domain='fresh', run_id='example')

    def test_snapshot_change_refused(self):
        c = context(0, 1)
        c['spec']['rawRoot']['mainIndex'] = 2
        with self.assertRaises(ValueError):
            population.population_manifest([c], domain='fresh', run_id='a', expected_roster={(0, 2)})

    def test_checkpoint_run_root_and_input_binding(self):
        identity = RunIdentity('test-only', 'run', 'a' * 64)
        result = {'runIdentity': {'domain': identity.domain, 'runId': identity.run_id,
                                  'inputIdentitySHA256': identity.input_identity_sha256}}
        value = {'memberId': common.member_id(1, 0, 1), 'inputSHA256': 'a' * 64,
                 'payload': {'result': result, 'resultSHA256': common._sha(common.canonical_bytes(result))}}
        self.assertEqual(population.check_checkpoint_envelope(value, identity=identity, variant=0,
                                                              main=1, replicate=1), result)
        for other, main in ((RunIdentity('test-only', 'other', 'a' * 64), 1),
                            (RunIdentity('test-only', 'run', 'b' * 64), 1), (identity, 2)):
            with self.assertRaises(ValueError):
                population.check_checkpoint_envelope(value, identity=other, variant=0,
                                                     main=main, replicate=1)

    def test_test_only_stub_resume_and_rejection(self):
        # Explicit test-only substitutes exercise orchestration, not science.
        from src.design_validation import c5_fresh_inputs as inputs
        from src.design_validation import c5_fresh_streams as streams
        from src.design_validation import c5_fresh_streams_reference as reference
        from src.design_validation import c5_public_trajectory as producer
        from src.design_validation import c5_public_trajectory_reference as receiver

        c = context(0, 1)
        c['spec'].update(config={}, initialDecision={}, design={})
        c['spec_snapshot_sha256'] = common._sha(common.canonical_bytes(c['spec']))
        c.update(p_latent=None, p_table=None, ps=None, r_latent=None, r_table=None, rs=None,
                 repository_root=str(Path('/tmp').resolve()))
        seen = []

        def produce(*args, **kwargs):
            seen.append('produce')
            ident = kwargs['producer_streams']
            return {'runIdentity': {'domain': ident.domain, 'runId': ident.run_id,
                                    'inputIdentitySHA256': ident.input_identity_sha256}}

        def receive(*args, **kwargs):
            seen.append('replay')

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for target, name, replacement in (
                (common, '_executing_root', lambda value: Path('/tmp').resolve()),
                (common, '_require_ignored', lambda *args: None),
                (inputs, 'check_unchanged', lambda value: None),
                (inputs, 'admit_reference', lambda **kwargs: c),
                (streams, 'ProducerStreams', lambda identity, context, rep: identity),
                (reference, 'ReferenceStreams', lambda identity, context, rep: identity),
                (producer, 'produce_fresh_root', produce),
                (receiver, 'receive_fresh_root', receive),
            ):
                stack.enter_context(mock.patch.object(target, name, replacement))
            args = ([c], {(0, 1): {}}, directory, '/tmp')
            kwargs = dict(domain='test-only', run_id='fixture', replicates=[1],
                          expected_roster={(0, 1)})
            first = population.run_population(*args, **kwargs)
            self.assertIsNone(first['prefixManifest'])
            self.assertEqual(seen, ['produce', 'replay'])
            verification = population.run_population(*args, **kwargs, verify_only=True)
            self.assertTrue(verification['verificationOnly'])
            self.assertEqual(seen, ['produce', 'replay', 'replay'])
            with self.assertRaises(ValueError):
                population.run_population(*args, **{**kwargs, 'run_id': 'other'})
            self.assertTrue((Path(directory) / '.c5-population-active').exists())
            with self.assertRaises(ValueError):
                population.run_population(*args, **kwargs)

    def test_partial_roster_cannot_claim_fresh(self):
        with self.assertRaises(ValueError):
            population.run_population([context(0, 1)], {(0, 1): {}}, '/tmp/no-use', '/tmp',
                                      domain='fresh', run_id='fixture', replicates=[1],
                                      expected_roster={(0, 1)})

    def test_incomplete_prefix_is_replayed_before_later_members(self):
        # Stubs test the control flow only; the real producer/receiver need
        # a separate integration check with genuine scientific inputs.
        from src.design_validation import c5_fresh_inputs as inputs
        from src.design_validation import c5_fresh_streams as streams
        from src.design_validation import c5_fresh_streams_reference as reference
        from src.design_validation import c5_public_trajectory as producer
        from src.design_validation import c5_public_trajectory_reference as receiver

        contexts = [context(v, m) for v, m in sorted(population.ROSTER)]
        refs = {}
        for c in contexts:
            c['spec'].update(config={}, initialDecision={}, design={})
            c['spec_snapshot_sha256'] = common._sha(common.canonical_bytes(c['spec']))
            c.update(p_latent=None, p_table=None, ps=None, r_latent=None, r_table=None, rs=None,
                     repository_root=str(Path('/tmp').resolve()))
            root = c['spec']['rawRoot']
            refs[root['scheduleVariant'], root['mainIndex']] = {'root': c}
        produced, replayed = [], []

        def produce(*args, **kwargs):
            identity = kwargs['producer_streams']
            produced.append((args[0]['scheduleVariant'], args[0]['mainIndex'], args[6]))
            return {'runIdentity': {'domain': identity.domain, 'runId': identity.run_id,
                                    'inputIdentitySHA256': identity.input_identity_sha256}}

        def receive(*args, **kwargs):
            replayed.append((args[0]['scheduleVariant'], args[0]['mainIndex'], args[6]))

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for target, name, replacement in (
                (common, '_executing_root', lambda value: Path('/tmp').resolve()),
                (common, '_require_ignored', lambda *args: None),
                (inputs, 'check_unchanged', lambda value: None),
                (inputs, 'admit_reference', lambda **kwargs: kwargs['root']),
                (streams, 'ProducerStreams', lambda identity, context, rep: identity),
                (reference, 'ReferenceStreams', lambda identity, context, rep: identity),
                (producer, 'produce_fresh_root', produce),
                (receiver, 'receive_fresh_root', receive),
            ):
                stack.enter_context(mock.patch.object(target, name, replacement))
            args = (contexts, refs, directory, '/tmp')
            kwargs = dict(domain='test-only', run_id='fixture')
            population.run_population(*args, **kwargs, replicates=[1])
            self.assertEqual(len(produced), 90)
            missing = Path(directory) / (common.member_id(1, 8, 10) + '.json')
            missing.unlink()
            with self.assertRaisesRegex(ValueError, 'Complete verified rep1'):
                population.run_population(*args, **kwargs, replicates=[2])
            self.assertEqual(len(produced), 90)
