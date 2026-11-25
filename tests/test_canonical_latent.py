"""Test strict-CDF integer and uniform boundaries without random draws."""
import unittest
from src.design_validation import canonical_latent as L
from src.design_validation import canonical_latent_reference as R


class CanonicalLatentTests(unittest.TestCase):
    def test_empty_and_nonempty_intervals(self):
        counts = [0, 3, 0, L.GRID - 3] + [0] * 96
        self.assertEqual(L.category_from_integer(counts, 0), 2)
        self.assertEqual(L.category_from_integer(counts, 2), 2)
        self.assertEqual(L.category_from_integer(counts, 3), 4)
        self.assertEqual(L.category_from_integer(counts, L.GRID - 1), 4)
        self.assertTrue(R.verify_law(counts)['completeIntegerGrid'])

    def test_refuses_invalid_inputs(self):
        counts = [L.GRID] + [0] * 99
        for bad in (-1, L.GRID, True):
            with self.assertRaises(ValueError):
                L.category_from_integer(counts, bad)
        for bad in (1.0, -0.5, float('nan'), 2 ** -54):
            with self.assertRaises(ValueError):
                L.category_from_uniform(counts, bad)
        with self.assertRaises(ValueError):
            R.verify_law([1] * 100)


if __name__ == '__main__':
    unittest.main()
