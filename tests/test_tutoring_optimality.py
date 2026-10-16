"""Admissible finite Bellman envelopes, explicit coverage and budget semantics.

Small fixtures verify solver mechanics, not the scientific production recipes.
Independent rational exhaustive values check every partial toy envelope.
"""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from design_validation.bounded_task import TaskConfig
from design_validation.optimality import (Interval, Seed, Branch, BellmanEnvelopes,
                                         BudgetEnvelopeModel, NumericalEnvelopeError)


class SmallModel:
    """Two hidden values, rational report channel and horizon of at most three."""
    root = (0, 0, 0)
    utility_abs_bound = 8.

    def __init__(self, horizon=2):
        self.horizon = horizon

    def depth(self, state):
        return state[0]

    def posterior(self, state):
        _, yes, no = state
        a, b = Fraction(3, 4)**yes*Fraction(1, 4)**no, Fraction(1, 4)**yes*Fraction(3, 4)**no
        return a/(a+b)

    def terminal(self, state):
        p = self.posterior(state)
        cost = Fraction(state[0], 10)
        return max(-2*(1-p)-cost, -2*p-cost)

    def seed(self, state):
        stop = float(self.terminal(state))
        asks = {} if self.depth(state) >= self.horizon else {a: Interval(-8., -float(Fraction(state[0]+1, 10))) for a in ("ask_A", "ask_B")}
        return Seed(Interval(stop, stop), asks)

    def branches(self, state):
        if self.depth(state) >= self.horizon:
            return {}
        p = self.posterior(state)
        yes = Fraction(3, 4)*p+Fraction(1, 4)*(1-p)
        d, a, b = state
        result = [Branch(1, float(yes), (d+1, a+1, b)), Branch(2, float(1-yes), (d+1, a, b+1))]
        return {"ask_A": result, "ask_B": result}


def exhaustive(model, state):
    stop = model.terminal(state)
    if model.depth(state) >= model.horizon:
        return stop
    p = model.posterior(state)
    yes = Fraction(3, 4)*p+Fraction(1, 4)*(1-p)
    d, a, b = state
    return max(stop, yes*exhaustive(model, (d+1, a+1, b))+(1-yes)*exhaustive(model, (d+1, a, b+1)))


def task_config():
    return TaskConfig("C3", "toy_native", 1., 0., (45., 65.), (3., 4.), 6.,
                      120000, 40000, 10, allocation_denominator=10, max_assessments=3)


class OptimalityTests(unittest.TestCase):
    def test_every_partial_envelope_contains_independent_full_value(self):
        model = SmallModel(3)
        reference = float(exhaustive(model, model.root))
        solver = BellmanEnvelopes(model)
        prior = solver.summary()
        for _ in range(20):
            front = solver.frontier(3)
            if not front:
                break
            solver.expand(front[0][1], 100)
            row = solver.summary()
            self.assertLessEqual(row["lower"], reference)
            self.assertGreaterEqual(row["upper"], reference)
            self.assertGreaterEqual(row["lower"], prior["lower"])
            self.assertLessEqual(row["upper"], prior["upper"])
            prior = row
        self.assertLess(solver.summary()["gap"], 1e-8)

    def test_atomic_state_limit_keeps_all_branches_unexpanded(self):
        solver = BellmanEnvelopes(SmallModel())
        before = solver.summary()
        self.assertFalse(solver.expand(solver.model.root, 2))
        self.assertEqual(solver.summary(), before)
        self.assertTrue(solver.expand(solver.model.root, 3))
        self.assertEqual(solver.summary()["explicit_report_edges"], 4)
        self.assertEqual(solver.summary()["created_states"], 3)

    def test_depth_and_time_are_compute_limits_not_scientific_horizon(self):
        solver = BellmanEnvelopes(SmallModel(3))
        row = solver.run(max_created_states=100, max_depth=0, max_seconds=10)
        self.assertEqual(row["limit_reason"], "depth_or_numerical_closure")
        self.assertGreater(row["gap"], .001)
        self.assertFalse(row["whole_policy_optimality_claimed"])
        solver = BellmanEnvelopes(SmallModel(3))
        row = solver.run(max_created_states=100, max_depth=3, max_seconds=0)
        self.assertEqual(row["limit_reason"], "elapsed_time_limit")

    def test_lower_policy_witness_stops_at_every_unexpanded_node(self):
        solver = BellmanEnvelopes(SmallModel(3))
        solver.expand(solver.model.root, 100)
        for row in solver.witness_rows():
            if not row["expanded"]:
                self.assertEqual(row["witness_action"], "stop")
        self.assertEqual(solver.summary()["witness_action"], "ask_A")

    def test_zero_report_branch_is_rejected(self):
        class Broken(SmallModel):
            def branches(self, state):
                result = super().branches(state)
                result["ask_A"][0] = Branch(1, 0., (1, 1, 0))
                return result
        solver = BellmanEnvelopes(Broken())
        with self.assertRaises(NumericalEnvelopeError):
            solver.expand(solver.model.root, 100)

    def test_native_sufficient_statistics_and_report_normalizers(self):
        model = BudgetEnvelopeModel(task_config())
        parity = model.posterior_parity([(who, reports) for who in (0, 1)
                                        for reports in ([1, 100], [100, 1], [1, 2, 3], [100, 99, 98])])
        self.assertTrue(parity["all_pass"])
        a = model.recipient_logs(0, 2, 101)
        sequential = model.task.posterior(model.task.posterior(model.task.initial_logs[0], 1), 100)
        np.testing.assert_allclose(a, sequential, atol=1e-10, rtol=0)
        self.assertEqual(len(model.branches(model.root)["ask_A"]), 100)

    def test_native_last_affordable_query_and_no_artificial_cap(self):
        model = BudgetEnvelopeModel(task_config())
        last = (2, 90, 0, 0)
        self.assertEqual(model.budget(last), 40000)
        branches = model.branches(last)
        zero = branches["ask_A"][0].child
        self.assertEqual(model.budget(zero), 0)
        seed = model.seed(zero)
        self.assertFalse(seed.asks)
        self.assertIsNone(seed.details["installed_stop_share"])
        with self.assertRaisesRegex(ValueError, "artificial cap"):
            BudgetEnvelopeModel(replace(task_config(), max_assessments=1))

    def test_terminal_max_and_latent_tail_bound_are_explicit(self):
        model = BudgetEnvelopeModel(replace(task_config(), prior_spreads=(.1, .1)))
        seed = model.seed(model.root)
        self.assertGreater(sum(seed.details["latent_underflow_counts"]), 0)
        self.assertGreater(sum(seed.details["omitted_latent_mass_upper_bounds"]), 0.)
        self.assertGreaterEqual(seed.details["installed_selector_regret"], -1e-12)
        self.assertLessEqual(seed.stop.lower, seed.details["stop_grid_max"])
        self.assertGreaterEqual(seed.stop.upper, seed.details["stop_grid_max"])


if __name__ == "__main__":
    unittest.main()
