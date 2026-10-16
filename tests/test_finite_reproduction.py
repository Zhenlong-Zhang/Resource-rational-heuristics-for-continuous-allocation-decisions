"""Small real-CLI producer/consumer checks, not a scientific campaign rerun."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from design_validation.bounded_task import TaskConfig, load_design
from design_validation.budget_horizon import budget_templates, paired_episode
from design_validation.finite_reproduction import (checkpoint_writer, paired_terminals,
                                                    produce_pairs, read_json)
from design_validation.optimality import BellmanEnvelopes, BudgetEnvelopeModel


class FiniteReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = read_json(ROOT/"configs/tutoring_paired_horizon.json")
        cls.design = load_design(ROOT/"configs/tutoring_conditions.json")
        cls.configs = budget_templates(cls.design)

    def command(self, stage, output, *options, success=True):
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1"}
        result = subprocess.run([sys.executable, str(ROOT/"scripts/reproduce_finite_tutoring.py"),
                                 stage, "--output", str(output), *map(str, options)],
                                cwd=output.parent, env=env, text=True, capture_output=True,
                                timeout=180)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue(read_json(output/"completion.json")["complete"])
        else:
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((output/"completion.json").exists())
        return result

    def test_real_cli_small_chain_and_untruncated_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = root/"pairs"
            self.command("pairs", pairs, "--template", "C3_s0_g0", "--episodes", 1)
            panel = read_json(pairs/"panel.json")
            self.assertEqual(panel["records"], 1)
            self.assertFalse(panel["full_panel_size_and_seed"])
            replay = list(paired_terminals(pairs, self.configs))
            self.assertEqual(len(replay), 1)
            original, task, logs, budget, state, error = replay[0]
            self.assertLessEqual(error, 1e-10)
            self.assertEqual(state[0]+state[2], original["budget_only"]["sample_count"])
            np.testing.assert_allclose(logs, original["budget_only"]["terminal_logs"], atol=1e-10, rtol=0)
            self.assertEqual(original["scenario"]["root_seed"], 2026100901)
            self.command("time-terminal", root/"time", "--pairs", pairs)
            impact = json.loads((root/"time/terminal_impact.jsonl").read_text())
            self.assertEqual(len(impact["analysis"]["scenarios"]), 18)
            self.assertTrue(all(not row["ask_once"] for row in impact["analysis"]["scenarios"]))
            self.assertEqual(impact["analysis"]["terminal"]["budget_milliminutes"], budget)
            self.command("bounds-terminal", root/"terminal", "--pairs", pairs)
            terminal = json.loads((root/"terminal/terminal_bounds.jsonl").read_text())
            self.assertEqual(terminal["state"], list(state))
            self.command("bounds-root", root/"bound", "--template", "C3_s0_g0",
                         "--max-seconds", 0, "--max-depth", 0, "--max-created-states", 1)
            bound = read_json(root/"bound/result.json")
            self.assertEqual(bound["scientific_horizon"], task.config.total_millihours//task.config.cost_millihours)
            self.assertGreater(bound["scientific_horizon"], 20)
            self.assertEqual(bound["limits"]["max_depth"], 0)
            self.assertEqual(bound["expanded_states"], 0)
            self.assertEqual(set(bound["action_bounds"]), {"stop", "ask_A", "ask_B"})
            witness = json.loads((root/"bound"/bound["witness_file"]).read_text())
            self.assertEqual(witness["witness_action"], "stop")
            checkpoint = read_json(root/"bound/checkpoint.json")
            self.assertEqual(checkpoint["witness_file"], bound["witness_file"])
            self.assertEqual(checkpoint["created_states"], bound["created_states"])
            self.assertTrue(all(v == "1" for v in read_json(root/"bound/completion.json")["numerical_thread_environment"].values()))
            self.command("horizon-initial", root/"initial", "--template", "C1_s1_g0")
            self.command("time-initial", root/"time_initial", "--template", "C3_s0_g0")
            self.command("bounds-initial", root/"initial_bound", "--template", "C1_s1_g0")

    def test_seeded_pair_is_replayable_with_shared_prefix(self):
        config = TaskConfig("C3", "small_pair", 1., 0., (45., 65.), (3., 3.), 6.,
                            120000, 20000, 10, allocation_denominator=10, max_assessments=6)
        first = paired_episode(config, 2026100901, 0)
        self.assertEqual(first, paired_episode(config, 2026100901, 0))
        a, b = first["budget_only"]["events"], first["capped_twenty"]["events"]
        self.assertEqual(a[:min(len(a), len(b))], b[:min(len(a), len(b))])
        self.assertEqual(len(first["scenario"]["potential_reports"][0]), 6)

    def test_pair_consumer_rejects_incomplete_duplicate_and_changed_inputs(self):
        config = TaskConfig("C3", "small_pair", 1., 0., (45., 65.), (3., 3.), 6.,
                            120000, 20000, 10, allocation_denominator=10, max_assessments=6)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            produce_pairs([config], self.settings, directory, 1)
            path = directory/"pairs.jsonl"
            original_text = path.read_text()
            original = json.loads(original_text)
            for text in ("", original_text+original_text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    list(paired_terminals(directory, [config]))
            for change in ("belief", "budget", "seed", "share", "episode"):
                row = deepcopy(original)
                if change == "belief":
                    row["budget_only"]["terminal_logs"][0][0] += .1
                elif change == "budget":
                    row["budget_only"]["remaining_millihours"] -= 1
                elif change == "seed":
                    row["scenario"]["root_seed"] += 1
                elif change == "share":
                    row["budget_only"]["final_share_A"] = -.1
                else:
                    row["episode_id"] += 1
                path.write_text(json.dumps(row)+"\n")
                with self.assertRaises(ValueError):
                    list(paired_terminals(directory, [config]))
            path.write_text(original_text)
            panel = read_json(directory/"panel.json")
            panel["templates"][0]["report_width"] += 1
            (directory/"panel.json").write_text(json.dumps(panel))
            with self.assertRaisesRegex(ValueError, "scientific design"):
                list(paired_terminals(directory, [config]))

    def test_cli_rejects_missing_predecessor_and_invalid_cutoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.command("time-terminal", root/"missing", success=False)
            self.command("pairs", root/"episodes", "--episodes", 0, success=False)
            self.command("bounds-root", root/"depth", "--template", "C3_s0_g0", "--max-depth", -1, success=False)
            self.command("bounds-root", root/"nan", "--template", "C3_s0_g0", "--max-seconds", "nan", success=False)
            self.command("horizon-initial", root/"unknown", "--template", "not_a_template", success=False)
            settings = read_json(ROOT/"configs/tutoring_optimality_v01.json")
            settings["profile"] = "incompatible-capped-profile"
            config_path = root/"changed.json"
            config_path.write_text(json.dumps(settings))
            self.command("bounds-root", root/"profile", "--config", config_path,
                         "--template", "C3_s0_g0", "--max-seconds", 0, success=False)
            self.assertFalse((root/"profile").exists())

    def test_checkpoint_keeps_a_complete_previous_generation_on_failed_pointer(self):
        config = TaskConfig("C3", "small_bound", 1., 0., (45., 65.), (3., 4.), 6.,
                            120000, 40000, 10, allocation_denominator=10, max_assessments=3)
        solver = BellmanEnvelopes(BudgetEnvelopeModel(config))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            save = checkpoint_writer(output)
            save(solver, solver.summary())
            first = read_json(output/"checkpoint.json")
            self.assertEqual(first["generation"], 0)
            self.assertTrue(solver.expand(solver.model.root, 201))
            with self.assertRaises(ValueError):
                save(solver, {**solver.summary(), "bad_number": float("nan")})
            self.assertEqual(read_json(output/"checkpoint.json"), first)
            self.assertEqual(len((output/first["witness_file"]).read_text().splitlines()), 1)
            save(solver, solver.summary())
            second = read_json(output/"checkpoint.json")
            self.assertEqual(second["generation"], 1)
            rows = [json.loads(line) for line in (output/second["witness_file"]).read_text().splitlines()]
            self.assertEqual(len(rows), second["created_states"])
            self.assertEqual(second["explicit_report_edges"], 200)


if __name__ == "__main__":
    unittest.main()
