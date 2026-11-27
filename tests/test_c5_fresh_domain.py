import dataclasses
import unittest

from src.design_validation import c5_fresh_domain as fresh


class FreshDomainTests(unittest.TestCase):
    def test_round_trip_all_kinds(self):
        identity = fresh.RunIdentity('fresh', 'run_01', 'b'*64)
        for key in (fresh.TypedKey(identity, 'latent', 0, 1, 1, 'A', 0),
                    fresh.TypedKey(identity, 'report', 8, 10, 1840, 'B', 3),
                    fresh.TypedKey(identity, 'terminal', 0, 1, 1, terminal_binding_sha256='a'*64)):
            self.assertEqual(fresh.parse_key(key.as_dict(), identity), key)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            identity.run_id = 'other'

    def test_reject_bad_identity_and_cross_domain(self):
        for domain, run_id in [('historical', 'x'), ('fresh', ''), ('fresh', '../x'),
                               ('fresh', 'UPPER'), ('fresh', 'x' * 65), ('fresh', 1)]:
            with self.subTest(domain=domain, run_id=run_id), self.assertRaises(ValueError):
                fresh.RunIdentity(domain, run_id, 'b'*64)
        with self.assertRaises(ValueError):
            fresh.RunIdentity('fresh', 'run', 'bad')
        key = fresh.TypedKey(fresh.RunIdentity('fresh', 'x', 'b'*64), 'terminal', 0, 1, 1, terminal_binding_sha256='a'*64).as_dict()
        for other in (fresh.RunIdentity('test-only', 'x', 'b'*64), fresh.RunIdentity('fresh', 'y', 'b'*64), fresh.RunIdentity('fresh', 'x', 'c'*64)):
            with self.assertRaises(ValueError):
                fresh.parse_key(key, other)

    def test_reject_bad_coordinates(self):
        identity = fresh.RunIdentity('test-only', 'case', 'b'*64)
        bad = [('latent', -1, 1, 1, 'A', 0), ('latent', 9, 1, 1, 'A', 0),
               ('latent', 0, 0, 1, 'A', 0), ('latent', 0, 11, 1, 'A', 0),
               ('latent', 0, 1, 0, 'A', 0), ('latent', 0, 1, 1841, 'A', 0),
               ('latent', 0, 1, True, 'A', 0), ('latent', 0, 1, 1, 'A', 1),
               ('report', 0, 1, 1, 'C', 0), ('report', 0, 1, 1, 'A', -1), ('report', 0, 1, 1, 'A', 2**53),
               ('terminal', 0, 1, 1, 'A', 0), ('other', 0, 1, 1, None, None)]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                fresh.TypedKey(identity, *args)

    def test_historical_envelopes_are_not_fresh_keys(self):
        identity = fresh.RunIdentity('fresh', 'x', 'b'*64)
        for historical in ({'randomizationGroup': 'r14-c5-paired-projection-private-stream-v1'},
                           {'randomizationGroup': 'r14-c5-paired-projection-terminal-v1'}):
            with self.assertRaises(ValueError):
                fresh.parse_key(historical, identity)
        key = fresh.TypedKey(identity, 'terminal', 0, 1, 1, terminal_binding_sha256='a'*64).as_dict()
        self.assertNotIn('randomizationGroup', key)
        key['unexpected'] = 1
        with self.assertRaises(ValueError):
            fresh.parse_key(key, identity)
