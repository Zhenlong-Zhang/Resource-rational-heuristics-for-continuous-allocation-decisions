"""Verify C2 inference units, no-leakage prediction, preserved ambiguity and numerics."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from design_validation.identifiability import (C2Recipe, constant_share_counterexample, fit_predict_fold,
    heldout_losses, project_share, simulate_recovery, student_t_critical, student_t_upper,
    within_person_slopes, wilson_interval)


def fixture_recipe() -> C2Recipe:
    gaps = np.repeat([-5.0, 0.0, 5.0], 10)
    targets = np.repeat([0.367, 0.4, 0.433], 10)
    scales = np.concatenate([np.repeat([0.95, 1.0, 1.05], counts) for counts in ([4, 3, 3], [3, 4, 3], [3, 3, 4])])
    folds = np.tile(np.arange(10) % 3, 3)
    return C2Recipe(targets, gaps, scales, folds, tuple(f"fixture_{i}" for i in range(30)))


class IdentifiabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        recipe = fixture_recipe()
        rng = np.random.default_rng(417)
        orders = np.argsort(rng.random((12, 30)), axis=1)
        self.targets = recipe.targets[orders]
        self.folds = recipe.folds[orders]
        self.time = np.linspace(-0.5, 0.5, 30)

    def test_original_counterexamples_preserved(self) -> None:
        result = constant_share_counterexample()
        self.assertTrue(result["all_match_within_005"])
        self.assertTrue(result["all_positive_ES_minus_EO_distance"])
        self.assertTrue(result["overlap_example"]["matches_both_within_005"])

    def test_EO_slope_one_even_with_intercept_and_time(self) -> None:
        y = self.targets + np.arange(12)[:, None] * 0.001 + self.time * 0.15
        np.testing.assert_allclose(within_person_slopes(y, self.targets), 1, atol=1e-12)

    def test_fixed_share_drift_slope_zero(self) -> None:
        y = np.broadcast_to(0.4 + self.time * 0.1, self.targets.shape)
        np.testing.assert_allclose(within_person_slopes(y, self.targets), 0, atol=1e-12)

    def test_fold_fit_has_no_heldout_leakage(self) -> None:
        y = self.targets + 0.01 * self.time
        altered = np.where(self.folds == 0, 0.95, y)
        for is_eo in (False, True):
            one = fit_predict_fold(y, self.targets, self.folds, 0, equal_outcome=is_eo)
            two = fit_predict_fold(altered, self.targets, self.folds, 0, equal_outcome=is_eo)
            np.testing.assert_allclose(one, two, atol=1e-15)

    def test_noiseless_predictive_discrimination_equal_nuisance(self) -> None:
        for generating_eo in (False, True):
            y = (self.targets if generating_eo else np.full_like(self.targets, 0.4)) + 0.02 + 0.08 * self.time
            fixed_loss, eo_loss = heldout_losses(y, self.targets, self.folds)
            correct, incorrect = (eo_loss, fixed_loss) if generating_eo else (fixed_loss, eo_loss)
            self.assertTrue(np.all(correct < 1e-25))
            self.assertTrue(np.all(incorrect > 1e-5))

    def test_slope_rejects_singular_predictor(self) -> None:
        with self.assertRaises(ValueError):
            within_person_slopes(np.full((2, 30), 0.4), np.full((2, 30), 0.4))

    def test_student_t_against_independent_known_values(self) -> None:
        self.assertAlmostEqual(student_t_upper(1.0, 1), 0.25, places=12)
        self.assertAlmostEqual(student_t_upper(-1.0, 1), 0.75, places=12)
        self.assertAlmostEqual(student_t_upper(0.0, 79), 0.5, places=12)
        self.assertAlmostEqual(student_t_critical(0.05, 10), 1.81246112281, places=9)
        self.assertAlmostEqual(student_t_critical(0.025, 1), 12.7062047362, places=8)
        self.assertEqual(student_t_upper(math.inf, 10), 0.0)

    def test_grid_and_side_complement(self) -> None:
        values = np.array([-0.3, 0, 0.367, 0.4005, 0.5995, 1, 1.3])
        projected = project_share(values)
        self.assertTrue(np.all((projected >= 0) & (projected <= 1)))
        np.testing.assert_allclose(projected + project_share(1 - values), 1, atol=1e-15)
        self.assertEqual(float(project_share(0.4005)), 0.401)
        self.assertEqual(float(project_share(0.5995)), 0.599)

    def test_wilson_boundaries(self) -> None:
        self.assertAlmostEqual(wilson_interval(0, 100)[0], 0, places=15)
        self.assertAlmostEqual(wilson_interval(100, 100)[1], 1, places=15)
        self.assertTrue(wilson_interval(50, 100)[0] < 0.5 < wilson_interval(50, 100)[1])

    def test_reproducible_simulation_has_group_unit_and_all_cells(self) -> None:
        config = json.loads((ROOT / "configs/tutoring_identifiability_validation.json").read_text())
        result, rows = simulate_recovery(fixture_recipe(), config, "smoke")
        again, again_rows = simulate_recovery(fixture_recipe(), config, "smoke")
        self.assertEqual(result, again)
        self.assertEqual(rows, again_rows)
        self.assertEqual(len(result["cells"]), 36)
        self.assertEqual(len(rows), 36 * 3)
        self.assertTrue(all(cell["groups"] == 3 and cell["participants_total"] == 30 for cell in result["cells"]))
        self.assertTrue(all(cell["individual_predictive_recovery"] is None for cell in result["cells"] if cell["strategy"] == "learn_equal_outcome"))
        self.assertFalse(result["gates"]["stage_eligible_for_confirmation_gates"])

    def test_legacy_source_not_imported(self) -> None:
        code = ("import sys;sys.path.insert(0,'src');import design_validation.identifiability;"
                "assert 'src' not in sys.modules;"
                "assert not any(n.startswith(('src.','experiments.','mdp.')) for n in sys.modules)")
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)

    def test_exact_candidate_recipe_integration(self) -> None:
        from design_validation.bounded_task import TaskConfig
        from design_validation.identifiability import build_c2_recipe
        configs = []
        counts = [[4, 3, 3], [3, 4, 3], [3, 3, 4]]
        for si, scale in enumerate([0.95, 1.0, 1.05]):
            for gi, gap in enumerate([-5, 0, 5]):
                configs.append(TaskConfig("C2", f"C2_s{si}_g{gi}", scale, gap,
                    ((45+gap)*scale, (75-gap)*scale), (2*scale, 2*scale), 6*scale,
                    round(150000*scale), round(1200*scale), counts[si][gi]))
        with patch("design_validation.bounded_task.expand_templates", return_value=configs):
            recipe = build_c2_recipe({})
        self.assertEqual(sorted(set(recipe.targets)), [0.367, 0.4, 0.433])
        self.assertTrue(all(np.sum(recipe.gaps == gap) == 10 for gap in np.unique(recipe.gaps)))
        self.assertTrue(all(np.sum(recipe.scales == scale) == 10 for scale in np.unique(recipe.scales)))


if __name__ == "__main__":
    unittest.main()
