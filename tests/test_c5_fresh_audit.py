"""Synthetic software fixtures only: no actual replay or policy draws."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.design_validation import c5_fresh_audit as audit
from src.design_validation import c5_public_reproduction as common
from src.design_validation.c5_fresh_domain import RunIdentity, TypedKey


class FreshAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contexts = []
        for v in range(9):
            for m in range(1, 11):
                identity = {'root': [v, m]}
                spec = {'rawRoot': {'scheduleVariant': v, 'mainIndex': m}}
                digest = common._sha(common.canonical_bytes(identity))
                self.contexts.append({'identity': identity, 'identity_sha256': digest, 'spec': spec,
                                      'spec_snapshot_sha256': common._sha(common.canonical_bytes(spec))})
                self._write(v, m, digest)
        (self.root / 'fresh-prefix-manifest.json').write_bytes(b'{}')

    def _write(self, v, m, digest, *, rep=1, wrong_root=False, wrong_ordinal=False,
               duplicate=False, contribution=None):
        identity = RunIdentity('test-only', 'audit-test', digest)
        def event(recipient):
            key = TypedKey(identity, 'latent', v, m, rep, recipient, 0).as_dict()
            return {'key': key, 'keySHA256': common._sha(common.canonical_bytes(key))}
        a, b = event('A'), event('B')
        stages = []
        if duplicate:
            key = TypedKey(identity, 'report', v, m, rep, 'A', 0).as_dict()
            report = {'key': key, 'keySHA256': common._sha(common.canonical_bytes(key))}
            stages = [{'report': report}, {'report': report}]
        result = {'runIdentity': {'domain': 'test-only', 'runId': 'audit-test',
                                  'inputIdentitySHA256': digest},
                  'rootKey': ['C5', v, m + int(wrong_root)], 'replicateId': rep,
                  'ordinal': 90 * (rep - 1) + 10 * v + m - 1 + int(wrong_ordinal),
                  'latent': {'A': a, 'B': b},
                  'stages': stages, 'terminalStream': None,
                  'endpoint': {'status': 'exact_tie' if m == 1 else 'complete'},
                  'contribution': contribution if contribution is not None else {
                      'gap': ['0', '1'] if m == 1 else ['1', '2'],
                      'partial_low': ['0', '1']}}
        envelope = {'memberId': common.member_id(rep, v, m), 'inputSHA256': digest,
                    'payload': {'result': result, 'resultSHA256': common._sha(common.canonical_bytes(result))}}
        (self.root / (envelope['memberId'] + '.json')).write_bytes(common.canonical_bytes(envelope))

    def _run(self, side_effect=None, *, contexts=None, replicates=(1,)):
        contexts = self.contexts if contexts is None else contexts
        manifest = audit.population.population_manifest(self.contexts, domain='test-only', run_id='audit-test')
        response = {'population': manifest, 'replicates': sorted(replicates), 'verificationOnly': True}
        # This stub exercises audit bookkeeping only; it does not independently replay science.
        with patch.object(audit.population, 'run_population', side_effect=side_effect, return_value=response) as replay:
            value = audit.audit_population(contexts, {}, self.root, self.root, domain='test-only',
                                           run_id='audit-test', replicates=replicates)
            self.assertTrue(replay.call_args.kwargs['verify_only'])
            return value

    def test_synthetic_sums_and_scope(self):
        result = self._run()
        self.assertEqual(result['members'], 90)
        self.assertEqual(result['coverageDenominator'], 90)
        self.assertEqual(result['ties'], 9)
        self.assertEqual(result['eventCount'], 180)
        self.assertEqual(result['contributionSums']['gap'], ['81', '2'])
        self.assertFalse(result['statisticalAuthorization'])
        self.assertFalse(result['historicalEquivalence'])

    def test_missing_root_and_duplicate_replicate_rejected(self):
        with self.assertRaises(ValueError):
            self._run(contexts=self.contexts[:-1])
        with self.assertRaisesRegex(ValueError, 'distinct replicates'):
            self._run(replicates=[1, 1])

    def test_multiple_replicates_and_coordinate_rejected(self):
        for context in self.contexts:
            root = context['spec']['rawRoot']
            self._write(root['scheduleVariant'], root['mainIndex'], context['identity_sha256'], rep=2)
        result = self._run(replicates=[2, 1])
        self.assertEqual(result['replicates'], [1, 2])
        self.assertEqual(result['members'], 180)
        self.assertEqual(result['coverageDenominator'], 180)
        self.assertEqual(result['contributionSums']['gap'], ['81', '1'])
        self._write(0, 2, self.contexts[1]['identity_sha256'], rep=2, wrong_ordinal=True)
        with self.assertRaisesRegex(ValueError, 'coordinate mismatch'):
            self._run(replicates=[1, 2])

    def test_synthetic_large_roster_bookkeeping(self):
        # This is a bounded fixture, not a measurement of full-study resources.
        for rep in range(2, 21):
            for context in self.contexts:
                root = context['spec']['rawRoot']
                self._write(root['scheduleVariant'], root['mainIndex'], context['identity_sha256'], rep=rep)
        result = self._run(replicates=range(1, 21))
        self.assertEqual(result['members'], 1800)
        self.assertEqual(result['eventCount'], 3600)
        self.assertEqual(result['ties'], 180)
        self.assertEqual(result['contributionSums']['gap'], ['810', '1'])

    def test_cross_member_duplicate_result_rejected(self):
        first = self.root / (common.member_id(1, 0, 1) + '.json')
        second = self.root / (common.member_id(1, 0, 2) + '.json')
        first_value = common._strict_json(first.read_bytes())
        second_value = common._strict_json(second.read_bytes())
        second_value['payload']['resultSHA256'] = first_value['payload']['resultSHA256']
        second.write_bytes(common.canonical_bytes(second_value))
        # Isolate the audit's cross-member index from the envelope's own hash check.
        with patch.object(audit.population, 'check_checkpoint_envelope',
                          side_effect=lambda value, **kwargs: value['payload']['result']):
            with self.assertRaisesRegex(ValueError, 'Duplicate member digest'):
                self._run()

    def test_checkpoint_changed_after_replay_check_rejected(self):
        original = audit._snapshot
        target = self.root / (common.member_id(1, 0, 1) + '.json')
        def snapshot(path, *, keep_raw=False):
            if keep_raw and Path(path) == target:
                target.write_bytes(target.read_bytes() + b' ')
            return original(path, keep_raw=keep_raw)
        with patch.object(audit, '_snapshot', side_effect=snapshot):
            with self.assertRaisesRegex(ValueError, 'changed after verification'):
                self._run()

    def test_noncanonical_fractions_and_nonzero_tie_rejected(self):
        digest = self.contexts[1]['identity_sha256']
        for pair in (['2', '4'], ['01', '2'], ['1', '-2'], ['1', '0']):
            with self.subTest(pair=pair):
                self._write(0, 2, digest, contribution={'gap': pair, 'partial_low': ['0', '1']})
                with self.assertRaises(ValueError):
                    self._run()
        self._write(0, 2, digest)
        self._write(0, 1, self.contexts[0]['identity_sha256'],
                    contribution={'gap': ['0', '1'], 'partial_low': ['-1', '3']})
        with self.assertRaisesRegex(ValueError, 'Tie must have zero contribution'):
            self._run()

    def test_exact_signed_fraction_sum_with_ties(self):
        for context in self.contexts:
            root = context['spec']['rawRoot']
            v, m = root['scheduleVariant'], root['mainIndex']
            values = {'gap': ['0', '1'], 'partial_low': ['0', '1']}
            if m == 2:
                values = {'gap': ['1', '3'], 'partial_low': ['-2', '7']}
            elif m == 3:
                values = {'gap': ['-1', '6'], 'partial_low': ['1', '7']}
            self._write(v, m, context['identity_sha256'], contribution=values)
        result = self._run()
        self.assertEqual(result['ties'], 9)
        self.assertEqual(result['contributionSums'], {'gap': ['3', '2'], 'partial_low': ['-9', '7']})

    def test_missing_and_symlink_rejected(self):
        path = self.root / (common.member_id(1, 0, 1) + '.json')
        path.unlink()
        with self.assertRaises(ValueError):
            self._run()
        path.symlink_to(self.root / (common.member_id(1, 0, 2) + '.json'))
        with self.assertRaises(ValueError):
            self._run()

    def test_changed_during_replay_rejected(self):
        def change(*args, **kwargs):
            (self.root / 'fresh-prefix-manifest.json').write_bytes(b'{ }')
            return {'population': audit.population.population_manifest(self.contexts, domain='test-only',
                                                                        run_id='audit-test'),
                    'replicates': [1], 'verificationOnly': True}
        with self.assertRaises(ValueError):
            self._run(side_effect=change)

    def test_bad_coordinate_and_duplicate_key_rejected(self):
        digest = self.contexts[0]['identity_sha256']
        self._write(0, 1, digest, wrong_root=True)
        with self.assertRaises(ValueError):
            self._run()
        self._write(0, 1, digest, duplicate=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate or invalid event key'):
            self._run()

    def test_replay_failure_propagates(self):
        with self.assertRaisesRegex(ValueError, 'replay failed'):
            self._run(side_effect=ValueError('replay failed'))

    def _run_prepared(self, *, replicates=(1,), entries=None, side_effect=None, response=None):
        path = self.root / 'prepared.json'
        path.touch(exist_ok=True)
        if entries is None:
            entries = {(c['spec']['rawRoot']['scheduleVariant'], c['spec']['rawRoot']['mainIndex']):
                       {'producerIdentitySHA256': c['identity_sha256']} for c in self.contexts}
        manifest = audit.population.population_manifest(self.contexts, domain='test-only', run_id='audit-test')
        if response is None:
            response = {'population': manifest, 'replicates': sorted(replicates), 'verificationOnly': True}
        # Only the interface and bookkeeping are exercised; replay is stubbed.
        with patch.object(audit.common, '_executing_root', return_value=self.root), \
             patch.object(audit.lazy, '_inventory', return_value=({}, entries)) as inventory, \
             patch.object(audit.lazy, 'run_lazy', side_effect=side_effect, return_value=response) as replay:
            result = audit.audit_prepared_manifest(path, 'pinned-digest', self.root, self.root,
                                                   runtime={'test': True}, domain='test-only',
                                                   run_id='audit-test', replicates=replicates)
            inventory.assert_called_once_with(path, 'pinned-digest', self.root, {'test': True},
                                              audit.population.ROSTER)
            self.assertTrue(replay.call_args.kwargs['verify_only'])
            self.assertEqual(replay.call_args.kwargs['replicates'], list(replicates))
            return result

    def test_prepared_bookkeeping_matches_existing_api(self):
        self.assertEqual(self._run_prepared(), self._run())

    def test_prepared_rejects_incomplete_roster_and_replay_mismatch(self):
        entries = {(0, 1): {'producerIdentitySHA256': self.contexts[0]['identity_sha256']}}
        with self.assertRaisesRegex(ValueError, 'Complete population'):
            self._run_prepared(entries=entries)
        with self.assertRaisesRegex(ValueError, 'Independent replay'):
            self._run_prepared(response={'verificationOnly': False})

    def test_prepared_requires_complete_rep1_and_detects_changes(self):
        for context in self.contexts:
            root = context['spec']['rawRoot']
            self._write(root['scheduleVariant'], root['mainIndex'], context['identity_sha256'], rep=2)
        (self.root / (common.member_id(1, 0, 1) + '.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self._run_prepared(replicates=[2])
        self._write(0, 1, self.contexts[0]['identity_sha256'])
        def change(*args, **kwargs):
            (self.root / 'fresh-prefix-manifest.json').write_bytes(b'{ }')
            return {'population': audit.population.population_manifest(self.contexts, domain='test-only',
                                                                        run_id='audit-test'),
                    'replicates': [2], 'verificationOnly': True}
        with self.assertRaisesRegex(ValueError, 'changed during verification'):
            self._run_prepared(replicates=[2], side_effect=change)


if __name__ == '__main__':
    unittest.main()
