"""Installed arithmetic, all-horizon inequality and paired-policy functionality."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from design_validation.bounded_task import TaskConfig, utility
from design_validation.budget_horizon import (BudgetTask, native_utility_table, native_realized,
                                            run_path, proportion, summarize_pairs)


def fixture(**changes):
    c = TaskConfig("C3", "functional", 1., 0., (45., 65.), (3., 3.), 6., 120000, 500, 10,
                   allocation_denominator=10, max_assessments=240)
    return replace(c, **changes)


class BudgetHorizonTests(unittest.TestCase):
    def test_integer_first_exact_kink_and_scalar_grid(self):
        c = fixture(allocation_denominator=1000)
        table = native_utility_table(114000, 1000, .5, 2.)
        self.assertEqual(table[500, 56], 0.)
        for j in (0, 1, 499, 500, 1000):
            for need in (1, 45, 57, 100):
                z = (114000*j-need*1000000)/1000000
                target = math.sqrt(z) if z >= 0 else -2*math.sqrt(-z)
                self.assertEqual(table[j, need-1], target)
        self.assertEqual(native_realized(c, .5, 114000, [57, 57]), 0.)

    def test_invalid_resource_and_positive_monotonicity(self):
        for value in (-1, 1.2, True):
            with self.assertRaises(ValueError): native_utility_table(value, 10, .5, 2.)
        for b in (0, 1, 300, 119999):
            self.assertTrue(np.all(native_utility_table(b+1, 10, .5, 2.) >= native_utility_table(b, 10, .5, 2.)))

    def test_finite_one_step_enumeration_independent_scalar(self):
        task = BudgetTask(fixture())
        decision = task.one_step_values(*task.initial_logs, 120000)
        p, posts = task.predictive(task.initial_logs[0])
        scalar = sum(prob*task.terminal_solution(logs, task.initial_logs[1], 119500)[1] for prob, logs in zip(p, posts))
        self.assertAlmostEqual(decision["ask_A_value"], scalar, 12)
        self.assertGreaterEqual(decision["ask_A_value"], task.terminal_solution(*task.initial_logs, 119500)[1]-1e-12)

    def test_perfect_information_upper_bound_and_mirror(self):
        task = BudgetTask(fixture())
        d = task.initial_values(); bound = task.perfect_information_after_first_cost(*task.initial_logs, 120000)
        self.assertGreaterEqual(bound["perfect_information_value_after_cost"], max(d["ask_A_value"], d["ask_B_value"])-1e-10)
        np.testing.assert_allclose(task.terminal_values(*task.initial_logs, 120000),
                                   task.terminal_values(task.initial_logs[1], task.initial_logs[0], 120000)[::-1], atol=1e-12)
        self.assertFalse(bound["machine_certified_arithmetic"])

    def test_physical_budget_and_count_cap_are_separate(self):
        task = BudgetTask(fixture(max_assessments=1))
        self.assertIsNone(task.one_step_values(*task.initial_logs, 120000, 1)["gain_A"])
        physical = BudgetTask(fixture(max_assessments=240))
        self.assertIsNone(physical.one_step_values(*physical.initial_logs, 499, 0)["gain_A"])

    def test_zero_budget_keeps_utility_null_share_and_targets(self):
        task = BudgetTask(fixture())
        share, value = task.terminal_solution(*task.initial_logs, 0)
        self.assertIsNone(share); self.assertTrue(math.isfinite(value))
        self.assertAlmostEqual(native_realized(task.config, None, 0, [45, 65]), -2*(math.sqrt(45)+math.sqrt(65)), 12)
        with self.assertRaisesRegex(ValueError, "Positive resource"):
            native_realized(task.config, None, 1, [45, 65])
        p = proportion([True, None, False]); self.assertEqual(p["valid_n"], 2); self.assertEqual(p["undefined_n"], 1)

    def test_explicit_shared_scenario_and_events_are_replayable(self):
        task = BudgetTask(fixture(cost_millihours=20000, max_assessments=6))
        supplied = {"true_needs": [45, 65], "potential_reports": [[45]*6, [65]*6]}
        a = run_path(task, supplied); b = run_path(BudgetTask(task.config), supplied)
        self.assertEqual(a, b)
        self.assertEqual(a["remaining_millihours"], 120000-a["sample_count"]*20000)
        self.assertLessEqual(a["sample_count"], 6)

    def test_original_profile_guard_is_not_relaxed(self):
        from tutoring_analysis.inference import validate_rows
        with self.assertRaisesRegex(ValueError, "historical capped"):
            validate_rows([{"profileId": "pilot-engagement-budget-v1", "questionLimitRule": "budget_only"}], "synthetic_design", "synthetic_validation")


if __name__ == "__main__": unittest.main()
