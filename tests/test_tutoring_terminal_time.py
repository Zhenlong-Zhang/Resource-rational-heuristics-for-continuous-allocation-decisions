"""Finite allocation targets, personal-time costs and rounding boundaries."""
import json
from pathlib import Path
import sys
import unittest
from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from design_validation.bounded_task import TaskConfig
from design_validation.budget_horizon import BudgetTask, native_utility_table
from design_validation.terminal_time import (MACROS, analysis_templates, break_even, display_audit,
    display_number, evaluate_state, exact_exhaustion_GPA, macro_durations, mirror_audit,
    one_step_tables, reference_rate, select_macro, terminal_summary, validate_settings)

class TerminalTimeTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((ROOT/"configs/tutoring_terminal_time_v01.json").read_text())
        self.c = TaskConfig("C1", "C1_s1_g0", 1., 0., (50., 50.), (3., 3.), 30., 120000, 1200, 10,
                            max_assessments=100)
        self.task = BudgetTask(self.c)

    def test_native_integer_arithmetic_and_zero_score(self):
        table = native_utility_table(120000, 1000, .5, 2.)
        self.assertEqual(float(table[500, 59]), 0.)
        row = terminal_summary(self.c, *self.task.initial_logs, 0)
        self.assertTrue(row["automatic_zero_budget"])
        self.assertLess(row["raw_grid_max_combined_GPA"], 7.)
        self.assertTrue(all(v["index"] is None for v in row["macros"].values()))
        durations = macro_durations(self.settings["assumed_profiles"]["base"], budget=0)
        self.assertTrue(all(d == 0 for d in durations.values()))

    def test_true_EO_differs_from_mean_surplus(self):
        c = replace(self.c, prior_centres=(40., 60.), prior_spreads=(2., 18.), total_millihours=130000)
        task = BudgetTask(c)
        row = terminal_summary(c, *task.initial_logs, c.total_millihours)
        self.assertNotEqual(row["macros"]["EO_expected_GPA"]["index"], row["macros"]["EO_raw_surplus"]["index"])
        self.assertLessEqual(row["macros"]["EO_expected_GPA"]["expected_GPA_gap"],
                             row["macros"]["EO_raw_surplus"]["expected_GPA_gap"]+1e-12)

    def test_cost_choice_zero_rho_and_equivalent_macros(self):
        row = terminal_summary(self.c, *self.task.initial_logs, 120000)
        durations = macro_durations(self.settings["assumed_profiles"]["base"])
        zero = select_macro(row, durations, 0.)
        self.assertAlmostEqual(zero["raw_max_objective"], row["raw_grid_max_combined_GPA"], places=12)
        chosen = select_macro(row, durations, 1.)
        self.assertEqual(chosen["selected_macro"], "ES")
        self.assertEqual(row["macros"]["ES"]["index"], row["macros"]["EO_expected_GPA"]["index"])
        self.assertLess(durations["ES"], durations["EO_expected_GPA"])
        self.assertEqual(row["macros"]["ES"]["expected_combined_GPA"],
                         terminal_summary(self.c, *self.task.initial_logs, 120000)["macros"]["ES"]["expected_combined_GPA"])

    def test_reference_and_symbolic_break_even(self):
        reference = reference_rate([self.c])
        rho0 = reference["rho0_GPA_per_personal_second"]
        self.assertGreater(rho0, 0.)
        self.assertFalse(reference["state_dependent"])
        row = terminal_summary(self.c, *self.task.initial_logs, 120000)
        times = macro_durations(self.settings["assumed_profiles"]["base"])
        thresholds = break_even(row, times, rho0)
        for pair in thresholds["pairs"]:
            if pair["rho_threshold"] is not None:
                self.assertAlmostEqual(pair["GPA_difference"]-pair["rho_threshold"]*pair["duration_difference_seconds"], 0., places=10)
        self.assertTrue(thresholds["ES_raw_preference_region"]["nonempty"])

    def test_complete_report_support_and_common_time_cancellation(self):
        data = one_step_tables(self.task, self.task.initial_logs, 120000)
        for action in ("ask_A", "ask_B"):
            self.assertEqual(data[action]["report_ids"], list(range(1, 101)))
            self.assertEqual(len(data[action]["children"]), 100)
            self.assertAlmostEqual(float(data[action]["probabilities"].sum()), 1., places=12)
        profile = self.settings["assumed_profiles"]["base"]
        for budget in (120000, 118800):
            summary = terminal_summary(self.c, *self.task.initial_logs, budget)
            a = select_macro(summary, macro_durations(profile, 0., budget), .01)
            b = select_macro(summary, macro_durations(profile, 2.65, budget), .01)
            self.assertAlmostEqual(a["selected_objective"]-b["selected_objective"], .0265, places=12)
        zero = terminal_summary(self.c, *self.task.initial_logs, 0)
        self.assertEqual(select_macro(zero, macro_durations(profile, 0., 0), .01)["selected_objective"],
                         select_macro(zero, macro_durations(profile, 2.65, 0), .01)["selected_objective"])

    def test_exhaustion_boundary_and_no_resource_deduction(self):
        burn = exact_exhaustion_GPA(self.task, self.task.initial_logs, 120000)
        self.assertAlmostEqual(burn, terminal_summary(self.c, *self.task.initial_logs, 0)["raw_grid_max_combined_GPA"])
        self.assertIsNone(exact_exhaustion_GPA(self.task, self.task.initial_logs, 119999))
        settings = {**self.settings, "rho_multipliers": [0., 100.]}
        result = evaluate_state(self.task, self.task.initial_logs, 1200, settings, reference_rate([self.c]), acquisition=True)
        self.assertEqual(result["terminal"]["budget_milliminutes"], 1200)
        self.assertEqual(result["scenarios"][0]["ask_once"]["ask_A"]["reports"], 100)
        self.assertTrue(all(v["no_search_status"] != "numerical_all_horizon_no_search" for v in result["scenarios"]
                            if v["exact_exhaustion_gain_vs_stop"] > 1e-10))

    def test_gap_display_and_orientation(self):
        c = replace(self.c, prior_centres=(49.5, 50.5))
        task = BudgetTask(c)
        row = terminal_summary(c, *task.initial_logs, 120000)
        audit = display_audit(row)
        self.assertAlmostEqual(audit["actual_mean_gap_minutes"], 1., places=10)
        self.assertEqual(audit["display_means_minutes"], [49.5, 50.5])
        self.assertTrue(audit["ES_EO_old_5pp_regions_overlap"])
        mirror = mirror_audit(task, task.initial_logs, 120000)
        self.assertLess(mirror["maximum_value_error"], 1e-10)
        self.assertEqual(mirror["mapping"]["MFN_A"]["reverse_macro"], "MFN_B")
        self.assertEqual(display_number(1.25, 1), 1.3)

    def test_invalid_inputs_and_frozen_settings(self):
        validate_settings(self.settings)
        row = terminal_summary(self.c, *self.task.initial_logs, 120000)
        times = macro_durations(self.settings["assumed_profiles"]["base"])
        for bad in (-1., float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                select_macro(row, times, bad)
        with self.assertRaises(ValueError):
            macro_durations({**self.settings["assumed_profiles"]["base"], "read_scalar": -1.})
        with self.assertRaises(ValueError):
            terminal_summary(self.c, *self.task.initial_logs, -1)
        with self.assertRaises(ValueError):
            validate_settings({**self.settings, "C1_gap_scaled": True})
        with self.assertRaises(ValueError):
            terminal_summary(self.c, np.zeros(99), self.task.initial_logs[1], 120000)


if __name__ == "__main__":
    unittest.main()
