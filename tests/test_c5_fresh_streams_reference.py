"""Test-only identities and synthetic laws; no historical stream is sampled."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.design_validation import c5_fresh_streams as producer
from src.design_validation import c5_fresh_streams_reference as reference
from src.design_validation import c5_public_streams as arithmetic
from src.design_validation.c5_fresh_domain import RunIdentity


class ReferenceStreamTests(unittest.TestCase):
    def setUp(self):
        self.identity = RunIdentity('test-only', 'reader-unit-v1', 'b' * 64)
        counts = (1 << 53,) + (0,) * 99
        cdf = (1.0.hex(),) * 100
        law = object.__new__(arithmetic.ValidatedLatentConfig)
        object.__setattr__(law, 'roots', tuple(((counts, cdf), (counts, cdf)) for _ in range(90)))
        object.__setattr__(law, 'input_sha256', 'c' * 64)
        table = object.__new__(arithmetic.ValidatedReportTable)
        object.__setattr__(table, 'width_hex', float(1).hex())
        object.__setattr__(table, 'product_sha256', 'd' * 64)
        object.__setattr__(table, 'rows', tuple((counts, cdf) for _ in range(100)))
        cfg = {'candidateId': 'test-candidate', 'task': {'condition_id': 'C5', 'report_width': 1.0},
               'state': {'step': 1}, 'random': {'probabilities': [0.25, 0.75]}}
        self.context = {'identity_sha256': 'b' * 64,
                        'spec': {'rawRoot': {'scheduleVariant': 0, 'mainIndex': 1},
                                 'config': cfg, 'initialDecision': {}, 'design': {}},
                        'p_latent': law, 'r_latent': law, 'p_table': table, 'r_table': table,
                        'ps': SimpleNamespace(table=((float(1).hex(), table),)),
                        'rs': SimpleNamespace(table=((float(1).hex(), table),))}
        self.patches = [patch.object(producer.c5_fresh_inputs, 'check_unchanged'),
                        patch.object(producer.trajectory, '_validate_root'),
                        patch.object(reference.c5_public_trajectory_reference, 'validate_spec'),
                        patch.object(producer.encoding.A, 'validate'),
                        patch.object(producer.encoding, 'identity')]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.p = producer.ProducerStreams(self.identity, self.context, 1)
        self.r = reference.ReferenceStreams(self.identity, self.context, 1)
        self.member = {'seed': 7, 'modelId': 'test-candidate', 'personId': 'test-person',
                       'trialId': 'test-trial', 'conditionId': 'C5', 'variant': 0,
                       'replicateId': 1, 'mainIndex': 1}

    def test_latent_report_replay_and_tampering(self):
        latent = self.p.latent('A')
        report = self.p.report('A', 3, latent)
        self.assertEqual(self.r.verify_latent(latent, 'A'), latent)
        self.assertEqual(self.r.verify_report(report, 'A', 3, latent), report)
        for field, value in [('category', 2), ('uniformHex', 0.0.hex()), ('keySHA256', '0' * 64)]:
            altered = copy.deepcopy(latent)
            altered[field] = value
            with self.assertRaises(ValueError):
                self.r.verify_latent(altered, 'A')
        with self.assertRaises(ValueError):
            self.r.verify_report(report, 'B', 3, latent)
        with self.assertRaises(ValueError):
            self.r.verify_report(report, 'A', 4, latent)
        altered = copy.deepcopy(report)
        altered['bindings']['reportWidthHex'] = float(2).hex()
        with self.assertRaises(ValueError):
            self.r.verify_report(altered, 'A', 3, latent)

    def test_cross_domain_input_and_coordinates_refused(self):
        event = self.p.latent('A')
        for field, value in [('domain', 'fresh'), ('inputIdentitySHA256', 'a' * 64),
                             ('variant', 1), ('main', 2), ('replicate', 2),
                             ('logicalRecipient', 'B')]:
            changed = copy.deepcopy(event)
            changed['key'][field] = value
            with self.assertRaises(ValueError):
                self.r.verify_latent(changed, 'A')
        with self.assertRaises(ValueError):
            reference.ReferenceStreams(RunIdentity('test-only', 'reader-unit-v1', 'a' * 64), self.context, 1)

    def test_terminal_reconstructed_from_caller_values(self):
        snapshot = copy.deepcopy(self.context['spec']['config'])
        event = self.p.terminal(snapshot, self.member)
        self.assertEqual(self.r.verify_terminal(event, snapshot, self.member), event)
        changed = copy.deepcopy(event)
        changed['binding']['memberIdentity'] = {'str': 'forged'}
        with self.assertRaises(ValueError):
            self.r.verify_terminal(changed, snapshot, self.member)
        changed = copy.deepcopy(self.member)
        changed['personId'] = 'other-person'
        with self.assertRaises(ValueError):
            self.r.verify_terminal(event, snapshot, changed)
        changed = copy.deepcopy(snapshot)
        changed['state']['step'] = 2
        with self.assertRaises(ValueError):
            self.r.verify_terminal(event, changed, self.member)
        changed = copy.deepcopy(snapshot)
        changed['random']['probabilities'] = [0.5, 0.5]
        with self.assertRaises(ValueError):
            self.r.verify_terminal(event, changed, self.member)


if __name__ == '__main__':
    unittest.main()
