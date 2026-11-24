"""Check the bounded-prefix rule and independent adjacent certificates."""
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from design_validation import c5_precision_planning as P
from design_validation import c5_prefix_reference as V


class BoundedPrefixTests(unittest.TestCase):
    def test_independent_certificate(self):
        radii = [F(1)] * 90
        result = P.bounded_prefix_certificate(radii, F(10), F(1))
        count = result['totalReplicates']
        self.assertTrue(V.verify(radii, F(10), F(1), count)['previousFailsLower'])
        with self.assertRaises(ValueError):
            V.verify(radii, F(10), F(1), count + 1)

    def test_reject_invalid_geometry(self):
        for radii, variance, maximum in (([F(1)]*89, F(1), F(1)),
                                         ([F(1)]*90, F(0), F(1)),
                                         ([F(1)]*90, F(1), F(2))):
            with self.assertRaises(ValueError):
                P.bounded_prefix_certificate(radii, variance, maximum)


if __name__ == '__main__':
    unittest.main()
