"""Test purpose: validate professor-facing active-search report calculations and input guards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate_active_search_report import (
    REPRODUCIBILITY_NOTEBOOK,
    REPOSITORY_URL,
    confirmation_overview,
    oracle_strata,
    report_html,
    write_confirmation_svg,
    validate_confirmation_manifests,
)


def confirmation_row(
    index: int,
    *,
    true_rate: float = 0.83,
    true_low: float = 0.81,
    closer_rate: float = 0.82,
    closer_low: float = 0.80,
) -> dict[str, object]:
    return {
        "environment": f"confirmation_{index}_active_search_six_test",
        "n_episodes": 1200,
        "mean_sample_count": 4.0,
        "mean_abs_allocation_from_equal": 0.10,
        "true_equal_outcome_rate": true_rate,
        "true_equal_outcome_one_sided_95_low": true_low,
        "closer_to_true_equal_outcome_than_equal_split_rate": closer_rate,
        "closer_to_true_equal_outcome_than_equal_split_one_sided_95_low": closer_low,
    }


class ActiveSearchReportTests(unittest.TestCase):
    def test_reproducibility_notebook_is_source_controlled(self) -> None:
        self.assertTrue(REPRODUCIBILITY_NOTEBOOK.is_file())
        self.assertIn(
            "%pip install",
            REPRODUCIBILITY_NOTEBOOK.read_text(encoding="utf-8"),
        )

    def test_confirmation_manifest_requires_independent_zero_prior_episodes(self) -> None:
        discovery = {"seed_namespace_offset": 1}
        confirmation = {
            "seed_namespace_offset": 2,
            "episodes_per_environment": 1200,
            "observation_draws": 500,
            "environments": [
                {
                    "config": {
                        "prior_sample_count_1": 0,
                        "prior_sample_count_2": 0,
                    }
                }
                for _ in range(12)
            ],
        }
        validate_confirmation_manifests(discovery, confirmation)
        confirmation["environments"][0]["config"]["prior_sample_count_1"] = 1
        with self.assertRaisesRegex(RuntimeError, "zero prior samples"):
            validate_confirmation_manifests(discovery, confirmation)

    def test_confirmation_requires_both_lower_bounds(self) -> None:
        rows = [confirmation_row(index) for index in range(1, 13)]
        rows[0]["closer_to_true_equal_outcome_than_equal_split_one_sided_95_low"] = 0.799
        result = confirmation_overview(rows)
        self.assertEqual(result["point_joint_count"], 12)
        self.assertEqual(result["confirmed_count"], 11)

    def test_confirmation_requires_active_search_behavior(self) -> None:
        rows = [confirmation_row(index) for index in range(1, 13)]
        rows[0]["mean_sample_count"] = 1.0
        rows[1]["mean_abs_allocation_from_equal"] = 0.049
        result = confirmation_overview(rows)
        self.assertEqual(result["point_joint_count"], 10)
        self.assertEqual(result["confirmed_count"], 10)

    def test_oracle_strata_are_episode_weighted(self) -> None:
        rows = [
            {
                "n_episodes": 100,
                "oracle_both_positive_rate": 0.5,
                "oracle_true_equal_outcome_rate_given_both_positive": 1.0,
                "oracle_closer_rate_given_both_positive": 1.0,
                "oracle_mixed_sign_rate": 0.5,
                "oracle_true_equal_outcome_rate_given_mixed_sign": 0.2,
                "oracle_closer_rate_given_mixed_sign": 0.4,
                "oracle_both_negative_rate": 0.0,
                "oracle_true_equal_outcome_rate_given_both_negative": "nan",
                "oracle_closer_rate_given_both_negative": "nan",
            }
        ]
        result = oracle_strata(rows)
        self.assertEqual(result["both_positive"]["episodes"], 50)
        self.assertAlmostEqual(result["both_positive"]["true_equal_outcome_rate"], 1.0)
        self.assertEqual(result["mixed_sign"]["episodes"], 50)
        self.assertAlmostEqual(result["mixed_sign"]["closer_rate"], 0.4)

    def test_confirmation_figure_is_self_contained(self) -> None:
        rows = [confirmation_row(index) for index in range(1, 13)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confirmation.svg"
            write_confirmation_svg(path, rows)
            value = path.read_text(encoding="utf-8")
        self.assertIn("Independent active-search confirmation", value)
        self.assertNotIn("/Users/", value)

    def test_report_links_to_public_reproducibility_source(self) -> None:
        summary = {
            "oracle": {
                "joint_candidates": 1,
                "strata": {
                    name: {
                        "episodes": 1,
                        "true_equal_outcome_rate": 1.0,
                        "closer_rate": 1.0,
                    }
                    for name in ("both_positive", "mixed_sign", "both_negative")
                },
            },
            "fixed_budget": {
                "useful_anchor_count": 1,
                "anchor_1_gain": 0.1,
                "anchor_1_low": 0.0,
                "anchor_1_high": 0.2,
                "anchor_3_gain": 0.1,
                "anchor_3_low": 0.0,
                "anchor_3_high": 0.2,
            },
            "discovery": {
                "candidate_count": 1,
                "sample_min": 4.0,
                "sample_max": 6.0,
                "allocation_distance": 0.1,
            },
            "confirmation": {
                "point_joint_count": 1,
                "confirmed_count": 0,
                "best_true_rate": 0.82,
                "best_true_low": 0.80,
                "best_closer_rate": 0.81,
                "best_closer_low": 0.79,
            },
            "solver": {"mismatches": 0, "positive_ci_count": 0, "confirmed_count": 0},
        }
        value = report_html(summary)
        self.assertIn(REPOSITORY_URL, value)
        self.assertNotIn("/Users/", value)


if __name__ == "__main__":
    unittest.main()
