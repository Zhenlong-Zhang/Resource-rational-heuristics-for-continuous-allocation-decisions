"""Independent finite-task, oracle, randomness and import-isolation checks."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from design_validation.bounded_task import (
    BoundedTask, TaskConfig, belief_moments, expand_templates, heuristic_shares,
    initial_log_beliefs, load_design, normalize_log_weights, project_share,
    realized_utility, scarcity_band, simulate_episode,
)


def fixture(**kwargs):
    base = TaskConfig("C3", "test", 1, 0, (45, 65), (8, 8), 6, 130000, 500,
                      10, allocation_denominator=10, max_assessments=3)
    return replace(base, **kwargs)


def scalar_utility(x):
    return math.sqrt(x) if x >= 0 else -2*math.sqrt(-x)


def scalar_terminal(logA, logB, budget, denominator):
    pA, pB = np.exp(logA), np.exp(logB)
    values = []
    for j in range(denominator+1):
        a, b = budget*j/(1000*denominator), budget*(denominator-j)/(1000*denominator)
        values.append(math.fsum(pA[n-1]*scalar_utility(a-n)+pB[n-1]*scalar_utility(b-n)
                                for n in range(1, 101)))
    return np.array(values)


class BoundedTaskTests(unittest.TestCase):
    def test_import_isolation(self):
        code = ("import sys;sys.path.insert(0,"+repr(str(ROOT/"src"))+");"
                "import design_validation.bounded_task;"
                "assert 'src' not in sys.modules;"
                "assert not any(k.startswith(('src.experiments','src.mdp','src.policies')) for k in sys.modules)")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

    def test_frozen_template_counts_and_loader_rejection(self):
        source = ROOT/"configs/tutoring_conditions_baseline.json"
        data = load_design(source)
        templates = expand_templates(data)
        self.assertEqual(len(templates), 27)
        for cid in ("C1", "C2", "C3", "C4", "C5"):
            self.assertEqual(sum(t.repeat_count_per_30 for t in templates if t.condition_id == cid), 30)
        data["common"]["observation"]["assessment_does_not_teach"] = False
        with tempfile.TemporaryDirectory(prefix="candidate-loader-") as directory:
            target = Path(directory)/"unsupported.json"
            target.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_design(target)

    def test_kernel_posterior_and_stable_extreme_history(self):
        task = BoundedTask(fixture())
        self.assertTrue(np.allclose(np.exp(task.log_likelihood).sum(axis=1), 1, atol=1e-14))
        logA, _ = task.initial_logs
        observation = 7
        raw_likelihood = []
        for need in range(1, 101):
            norm = math.fsum(math.exp(-.5*((report-need)/6)**2) for report in range(1, 101))
            raw_likelihood.append(math.exp(-.5*((observation-need)/6)**2)/norm)
        reference = np.exp(logA)*np.array(raw_likelihood)
        reference /= reference.sum()
        self.assertTrue(np.allclose(np.exp(task.posterior(logA, observation)), reference, atol=1e-14))
        extreme = BoundedTask(fixture(prior_centres=(95, 95), prior_spreads=(.5, .5), report_width=.5))
        logs = extreme.initial_logs[0]
        for _ in range(20):
            logs = extreme.posterior(logs, 1)
        self.assertTrue(np.isfinite(logs).all())
        self.assertAlmostEqual(np.exp(logs).sum(), 1, places=13)

    def test_terminal_enumeration_and_exact_report_loop(self):
        task = BoundedTask(fixture())
        logs = task.initial_logs
        reference = scalar_terminal(*logs, task.config.total_millihours, 10)
        self.assertTrue(np.allclose(task.terminal_values(*logs, task.config.total_millihours), reference, atol=1e-12))
        values = task.one_step_values(*logs, task.config.total_millihours)
        for who, field in ((0, "ask_A_value"), (1, "ask_B_value")):
            probabilities, _ = task.predictive(logs[who])
            expectation = 0
            for report in range(1, 101):
                new_logs = list(logs)
                new_logs[who] = task.posterior(logs[who], report)
                expectation += probabilities[report-1]*scalar_terminal(*new_logs, 129500, 10).max()
            self.assertAlmostEqual(values[field], expectation, places=11)

    def test_oracle_bound_independent_hidden_state_loop(self):
        task = BoundedTask(fixture(prior_spreads=(2, 2), allocation_denominator=4))
        logA, logB = task.initial_logs
        B = (task.config.total_millihours-task.config.cost_millihours)/1000
        reference = 0
        for n in range(1, 101):
            for m in range(1, 101):
                maximum = max(scalar_utility(B*j/4-n)+scalar_utility(B*(4-j)/4-m) for j in range(5))
                reference += math.exp(logA[n-1]+logB[m-1])*maximum
        bound = task.perfect_information_after_first_cost(logA, logB, task.config.total_millihours)
        self.assertAlmostEqual(reference, bound["perfect_information_value_after_cost"], places=10)
        one = task.one_step_values(logA, logB, task.config.total_millihours)
        self.assertGreaterEqual(bound["upper_bound_gain"]+1e-10, max(one["gain_A"], one["gain_B"]))
        # A concrete two-request policy: query both, then give all to the lower report.
        probA, postA = task.predictive(logA)
        probB, postB = task.predictive(logB)
        later_B = (task.config.total_millihours-2*task.config.cost_millihours)/1000
        amounts = [0, later_B/2, later_B]
        expectedA = np.exp(postA) @ np.array([[scalar_utility(t-n) for n in range(1,101)] for t in amounts]).T
        expectedB = np.exp(postB) @ np.array([[scalar_utility(t-n) for n in range(1,101)] for t in amounts]).T
        after_two = 0
        for y in range(100):
            for z in range(100):
                index = 2 if y < z else 0 if y > z else 1
                after_two += probA[y]*probB[z]*(expectedA[y,index]+expectedB[z,2-index])
        self.assertLessEqual(after_two, reference+1e-10)

    def test_stream_replay_and_budget_no_truth_leak(self):
        task = BoundedTask(fixture())
        first, repeat = task.scenario(123, 0), task.scenario(123, 0)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, task.scenario(123, 1))
        result = simulate_episode(task, 123, 0)
        self.assertEqual(result["remaining_millihours"], 130000-result["sample_count"]*500)
        self.assertLessEqual(result["sample_count"], 3)
        positions = [0, 0]
        logs = list(task.initial_logs)
        for event in result["events"]:
            who = int(event["action"] == "ask_B")
            self.assertEqual(event["observation"], first["potential_reports"][who][positions[who]])
            logs[who] = task.posterior(logs[who], event["observation"])
            positions[who] += 1
        self.assertTrue(np.array_equal(logs[0], result["terminal_log_A"]))
        self.assertTrue(np.array_equal(logs[1], result["terminal_log_B"]))
        self.assertAlmostEqual(result["realized_utility"], realized_utility(task.config,
            result["final_share_A"], result["remaining_millihours"], first["true_needs"]), places=14)
        self.assertEqual(task.initial_values(), BoundedTask(task.config).initial_values())

    def test_grid_symmetry_heuristics_and_zero_budget(self):
        logs = initial_log_beliefs(fixture(prior_centres=(40, 80), prior_spreads=(2, 2)))
        forward, reverse = heuristic_shares(*logs, 150000), heuristic_shares(logs[1], logs[0], 150000)
        for key in forward:
            self.assertAlmostEqual(forward[key]+reverse[key], 1, places=12)
        self.assertAlmostEqual(forward["equal_outcome"], .367)
        self.assertEqual(project_share(.5), .5)
        self.assertTrue(all(v is None for v in heuristic_shares(*logs, 0).values()))
        task = BoundedTask(fixture())
        self.assertEqual(task.terminal_solution(*logs, 0)[0], None)
        self.assertEqual(task.one_step_values(*logs, 0)["chosen_action"], "stop")
        self.assertEqual(task.one_step_values(*logs, 500, 0)["ask_A_value"], task.terminal_solution(*logs, 0)[1])
        self.assertEqual(task.one_step_values(*logs, 130000, 3)["chosen_action"], "stop")

    def test_gpa_and_four_scarcity_bands(self):
        self.assertAlmostEqual(3.5-.06*math.sqrt(100), 2.9)
        self.assertLess(3.5+.03*math.sqrt(178.5-1), 3.9)
        expected = ["neither_individually_feasible", "only_lower_individually_feasible",
                    "either_but_not_both_feasible", "both_jointly_feasible"]
        for budget, label in zip((39, 40, 60, 100), expected):
            self.assertEqual(scarcity_band(budget, 40, 60), label)
        c = fixture()
        U = realized_utility(c, .4, 130000, [40, 65])
        gpa_total = (3.5+.03*scalar_utility(52-40))+(3.5+.03*scalar_utility(78-65))
        self.assertAlmostEqual(gpa_total, 7+.03*U)

    def test_integer_numerator_preserves_exact_need_threshold(self):
        config = fixture(total_millihours=142500, allocation_denominator=1000)
        # .4 * 142.5 = exactly 57 hours; the old ratio-first form was just below 57.
        value = realized_utility(config, .4, 142500, [57, 85])
        self.assertAlmostEqual(value, math.sqrt(.5), places=14)
        self.assertEqual((142500*400)/1000000, 57)


if __name__ == "__main__":
    unittest.main()
