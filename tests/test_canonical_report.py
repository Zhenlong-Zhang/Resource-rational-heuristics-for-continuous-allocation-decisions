"""Verify report matrices and independent strict-grid rejection behavior."""
import copy
import unittest
from src.design_validation.canonical_report import generate_report_table
from src.design_validation.canonical_report_reference import GRID, grid_partition, verify_report_table


class CanonicalReportTests(unittest.TestCase):
    def test_full_independent_report_and_tampering(self):
        width = float(10).hex()
        table = generate_report_table(width)
        self.assertTrue(verify_report_table(table, width))
        self.assertEqual(len(table['rows']), 100)
        self.assertTrue(all(len(row['counts']) == 100 and sum(row['counts']) == GRID for row in table['rows']))
        changed = copy.deepcopy(table)
        changed['rows'][0]['counts'][0] += 1
        with self.assertRaises(ValueError):
            verify_report_table(changed, width)
        changed = copy.deepcopy(table)
        changed['newScientificDraws'] = False
        with self.assertRaises(ValueError):
            verify_report_table(changed, width)
        with self.assertRaises(ValueError):
            verify_report_table(table, float(11).hex())

    def test_strict_grid_boundary_and_fallback(self):
        cumulative = [float(0.5).hex()] * 100
        counts, cuts = grid_partition(cumulative)
        self.assertEqual(counts[0], GRID // 2)
        self.assertEqual(counts[1:99], [0] * 98)
        self.assertEqual(counts[-1], GRID // 2)
        self.assertEqual(cuts[-1], GRID)
        with self.assertRaises(ValueError):
            grid_partition([float(0.5).hex(), float(0.25).hex()] + cumulative[2:])

    def test_invalid_width_refused(self):
        for width in ('0x0.0p+0', '-0x1.0000000000000p+0', 'inf', '1.0', 1.0):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    generate_report_table(width)
                with self.assertRaises(ValueError):
                    verify_report_table({}, width)


if __name__ == '__main__':
    unittest.main()
