"""Independent GPA algebra and matched-Gaussian tail audit; no private input dependencies."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from design_validation.model_bridge import audit_bridge, finite_kernel, gaussian_outside, reference_utility


def fixture() -> dict:
    recipes = [("C1", [50, 50], [3, 3], 30, 120, 1.2, [0]),
               ("C2", [45, 75], [2, 2], 6, 150, 1.2, [-5, 0, 5]),
               ("C3", [60, 60], [18, 18], 6, 170, .5, [0]),
               ("C4", [45, 65], [2, 2], 6, 55, .6, [-5, 0, 5]),
               ("C5", [55, 55], [8, 8], 3, 55, .3, [0])]
    support = {"support_min": 1, "support_max": 100, "support_step": 1}
    return {"common": {"latent_need": support.copy(), "observation": support.copy(),
                       "utility": {"gain_exponent": .5, "loss_exponent": .5, "loss_multiplier": 2},
                       "gpa": {"baseline": 3.5, "utility_scale": .03, "clipping": False},
                       "variation": {"scale_factors": [.95, 1, 1.05]},
                       "allocation": {"share_grid_denominator": 1000},
                       "maximum_online_assessments": 20},
            "conditions": [{"id": cid, "prior_kernel_centres": m, "prior_kernel_spreads": s,
                            "report_kernel_width": tau, "total_time": total,
                            "sample_time_cost": cost, "mean_gap_offsets": gaps}
                           for cid, m, s, tau, total, cost, gaps in recipes]}


class ModelBridgeTests(unittest.TestCase):
    def test_gaussian_reference_known_probability(self):
        self.assertAlmostEqual(gaussian_outside(0, 1, -1, 1), .31731050786291415, places=14)
        self.assertEqual(gaussian_outside(50, .1, 1, 100), 0)
        self.assertAlmostEqual(gaussian_outside(50, 3, 1, 100),
                               gaussian_outside(51, 3, 1, 100), places=14)

    def test_kernel_stability_at_remote_centre(self):
        p = finite_kernel(500, .1, list(range(1, 101)))
        self.assertTrue(all(math.isfinite(x) and x >= 0 for x in p))
        self.assertAlmostEqual(math.fsum(p), 1)
        self.assertEqual(p[-1], 1)

    def test_boundary_changes_mean(self):
        p = finite_kernel(1, 6, list(range(1, 101)))
        self.assertGreater(math.fsum((i + 1) * x for i, x in enumerate(p)), 1)

    def test_audit_full_fixture(self):
        summary, rows = audit_bridge(fixture())
        self.assertEqual(summary["template_count"], 27)
        self.assertEqual(len(rows), 54)
        self.assertAlmostEqual(summary["gpa_bounds"][0], 2.9)
        self.assertLess(summary["gpa_bounds"][1], 3.9)
        self.assertEqual(summary["gpa_identity_test_cases"], 5103)
        self.assertLess(summary["maximum_gpa_affine_identity_residual"], 1e-12)
        central_c1 = next(r for r in rows if r["condition_id"] == "C1" and r["scale"] == 1)
        self.assertGreater(central_c1["continuous_reference_report_outside_1_100"], .09)
        self.assertFalse(summary["gaussian_tails_are_exact_discrete_removed_mass"])
        self.assertFalse(summary["psychological_loss_aversion_estimated"])

    def test_scalar_gpa_examples(self):
        for x, expected in [(-25, 3.20), (-4, 3.38), (0, 3.50), (4, 3.56), (25, 3.65)]:
            self.assertAlmostEqual(3.5 + .03 * reference_utility(x, .5, 2), expected)

    def test_reject_changed_scientific_contract(self):
        spec = fixture()
        for field, value in [("clipping", True), ("utility_scale", 3)]:
            changed = copy.deepcopy(spec)
            changed["common"]["gpa"][field] = value
            with self.assertRaises(ValueError):
                audit_bridge(changed)
        with self.assertRaises(ValueError):
            gaussian_outside(0, 0, -1, 1)

    def test_no_legacy_import(self):
        code = (
            "import sys; sys.path.insert(0, 'src'); "
            "import design_validation.model_bridge; "
            "assert not any(n == 'src' or n.startswith(('src.', 'mdp.', 'experiments.')) "
            "for n in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
