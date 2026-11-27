"""Test-domain stream primitives; no historical random namespace is sampled."""
import copy
import hashlib
import json
import unittest

from src.design_validation import c5_fresh_streams as streams
from src.design_validation.c5_fresh_domain import RunIdentity, TypedKey


class FreshStreamTests(unittest.TestCase):
    def setUp(self):
        self.identity = RunIdentity('test-only', 'streams-boundary-v1', 'b'*64)
        self.key = TypedKey(self.identity, 'latent', 0, 1, 1, 'A', 0)

    def test_full_typed_envelope_drives_deterministic_seed(self):
        encoded = json.dumps(self.key.as_dict(), sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False).encode('ascii')
        digest, value = streams._uniform(self.key)
        self.assertEqual(digest, hashlib.sha256(encoded).hexdigest())
        self.assertEqual((digest, value), streams._uniform(self.key))
        self.assertTrue(0.0 <= value < 1.0)
        other = TypedKey(RunIdentity('test-only', 'streams-boundary-v2', 'b'*64), 'latent', 0, 1, 1, 'A', 0)
        self.assertNotEqual(digest, streams._uniform(other)[0])
        with self.assertRaises(ValueError):
            streams._uniform(self.key.as_dict())

    def test_partition_grid_and_boundaries(self):
        obj = object.__new__(streams.ProducerStreams)
        counts = (streams.arithmetic.GRID,) + (0,) * 99
        cumulative = (1.0.hex(),) * 100
        event = obj._partition(self.key, counts, cumulative, inputIdentitySHA256='test')
        self.assertEqual(event['category'], 1)
        self.assertEqual(event['key'], self.key.as_dict())
        self.assertEqual(event['identityStatus'], 'new-unverified')
        with self.assertRaises(ValueError):
            obj._partition(self.key, (0,) * 100, cumulative)
        with self.assertRaises(ValueError):
            obj._partition(self.key, counts, (0.0.hex(),) * 100)

    def test_typed_coordinates_and_terminal_binding_are_required(self):
        obj = object.__new__(streams.ProducerStreams)
        obj.identity, obj.variant, obj.main, obj.replicate = self.identity, 0, 1, 1
        for args in [('report', 'A', 2**53, None), ('latent', 'A', 1, None),
                     ('latent', 'C', 0, None), ('terminal', None, None, None)]:
            with self.assertRaises(ValueError):
                obj._key(*args)
        candidate = {'state': [1, 2], 'identity': self.identity.run_id}
        binding = streams._sha(candidate)
        key = obj._key('terminal', binding=binding)
        changed = copy.deepcopy(candidate)
        changed['state'][0] = 2
        other = obj._key('terminal', binding=streams._sha(changed))
        self.assertNotEqual(streams._uniform(key)[0], streams._uniform(other)[0])

    def test_missing_admitted_context_rejected(self):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            streams.ProducerStreams(self.identity, {}, 1)
        with self.assertRaises(ValueError):
            streams.ProducerStreams('test-only', {}, 1)


if __name__ == '__main__':
    unittest.main()
