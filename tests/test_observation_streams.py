"""Test purpose: ensure observation streams and realized utility share the same episode-specific hidden state."""

from __future__ import annotations

import math
import unittest

from scripts.check_observation_streams import run_observation_stream_check


class ObservationStreamRegressionTest(unittest.TestCase):
    def test_common_observation_streams_match_episode_true_state(self) -> None:
        summary = run_observation_stream_check(
            n_episodes=40,
            observations_per_person=100,
            seed=123,
            sigma_need=20.0,
            sigma_sample=10.0,
            min_correlation=0.95,
        )

        self.assertEqual(summary["true_state_mismatches"], 0)
        self.assertEqual(summary["first_sample_mismatches"], 0)

        corr_1 = summary["person1_stream_mean_true_need_correlation"]
        corr_2 = summary["person2_stream_mean_true_need_correlation"]
        self.assertFalse(math.isnan(corr_1))
        self.assertFalse(math.isnan(corr_2))
        self.assertGreaterEqual(corr_1, summary["min_correlation_required"])
        self.assertGreaterEqual(corr_2, summary["min_correlation_required"])


if __name__ == "__main__":
    unittest.main()
