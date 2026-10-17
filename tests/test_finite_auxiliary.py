"""Portable descriptive-reader fixtures and deterministic JS closure, not new science."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from design_validation.bounded_task import load_design
from design_validation.budget_horizon import budget_templates
from design_validation.finite_auxiliary import ACTIVE, check_scenarios
from design_validation.finite_auxiliary_inputs import parse, read, terminal_rows, write
from design_validation.finite_bound_tables import symmetry_bound, terminal_counts
from design_validation.finite_reproduction import _config_record
from design_validation.finite_time_tables import Group, preference_region, scenario_values
from design_validation.terminal_time import MACROS, analysis_templates, timing_profiles


class FiniteAuxiliaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = load_design(ROOT/"configs/tutoring_conditions.json")
        cls.time = read(ROOT/"configs/tutoring_terminal_time_v01.json")
        cls.bounds = read(ROOT/"configs/tutoring_optimality_v01.json")
        cls.budget = {c.template_id: _config_record(c) for c in budget_templates(cls.design)}

    def command(self, stage, output, *options, success=True, primary=False):
        script = "reproduce_finite_tutoring.py" if primary else "reproduce_finite_auxiliary.py"
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        result = subprocess.run([sys.executable, str(ROOT/"scripts"/script), stage,
                                 "--output", str(output), *map(str, options)],
                                cwd=output.parent, env=env, capture_output=True, text=True, timeout=180)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue(read(output/"completion.json")["complete"])
        else:
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((output/"completion.json").exists())
        return result

    def analysis_fixture(self, condition="C3", initial=False):
        # Algebra-only artificial menu: these rows are not claimed model outputs.
        terminal = {"macros": {name: {"index": 500, "share_A": .5, "expected_utility": 1.,
                                     "expected_combined_GPA": 7.03} for name in MACROS},
                    "automatic_zero_budget": False}
        scenarios, thresholds = [], {}
        for profile in timing_profiles(self.time, initial and condition == "C1"):
            durations = dict.fromkeys(MACROS, 1.)
            thresholds[profile] = {"ES_raw_preference_region": preference_region(terminal, durations, .001)}
            for rho in self.time["rho_multipliers"]:
                scenarios.append({"profile": profile, "rho_multiplier": rho, "selected_macro": "ES",
                    "duration_by_macro_seconds": durations, "selected_objective": 7.03-rho*.001,
                    "allocation_classes": {"500": list(MACROS)}, "objective_selection_regret": 0.,
                    "exact_exhaustion_gain_vs_stop": None, "no_search_status": "not_checked",
                    "positive_feasible_ask_once": False, "tied_macros": list(MACROS), "ask_once": {}})
        return {"terminal": terminal, "scenarios": scenarios, "break_even": thresholds,
                "global_adaptive_policy_solved": False, "personal_question_time_priced": False,
                "machine_certified_arithmetic": False, "new_random_scenarios": 0}

    def initial_fixture(self, path):
        path.mkdir()
        configs = [_config_record(c) for c in analysis_templates(self.design, self.time)]
        write(path/"settings.json", self.time)
        write(path/"templates.json", configs)
        write(path/"reference.json", {"rho0_GPA_per_personal_second": .001})
        manifest = []
        for config in configs:
            tid, cid = config["template_id"], config["condition_id"]
            analysis = self.analysis_fixture(cid, initial=True)
            row = {"template_id": tid, "condition_id": cid, "config": config, "analysis": analysis,
                   "working_C1_candidate": cid == "C1" and config["gap_offset"] == 1.,
                   "orientation_audit": {"forward": {"ES_EO_share_distance": 0.},
                       "reverse": {"ES_EO_share_distance": 0.}, "mapping": {}, "maximum_value_error": 0.}}
            write(path/(tid+".json"), row)
            manifest.append({"template_id": tid, "condition_id": cid, "file": tid+".json",
                             "scenario_rows": len(analysis["scenarios"])})
        write(path/"manifest.json", manifest)
        write(path/"completion.json", {"stage": "time-initial", "complete": True,
              "machine_certified_arithmetic": False, "initial_states": 36, "scenario_rows": 1800})

    def terminal_fixture(self, path, episodes=400):
        path.mkdir()
        panel = {"schema": "finite-tutoring-paired-panel-v1", "policy": "repeated_exact_one_step_myopic",
                 "root_seed": 2026100901, "episodes_per_template": episodes,
                 "templates": [self.budget[key] for key in ACTIVE], "records": 6*episodes}
        write(path/"settings.json", self.time)
        write(path/"templates.json", list(self.budget.values()))
        write(path/"reference.json", {"rho0_GPA_per_personal_second": .001})
        summary = {"input_panel": panel, "records": 6*episodes, "new_random_scenarios": 0,
                   "new_cost_aware_trajectory_claimed": False, "whole_policy_optimality_claimed": False}
        write(path/"summary.json", summary)
        write(path/"completion.json", {**summary, "stage": "time-terminal", "complete": True,
                                      "machine_certified_arithmetic": False})
        analysis = self.analysis_fixture()
        with (path/"terminal_impact.jsonl").open("w") as stream:
            for tid in ACTIVE:
                for episode in range(episodes):
                    row = {"template_id": tid, "episode_id": episode, "condition_id": tid[:2],
                           "old_budget_only_share_A": .25, "analysis": analysis, "posterior_parity_error": 0.}
                    stream.write(json.dumps(row)+"\n")

    def test_group_denominators_and_conditional_metrics(self):
        group = Group()
        group.add({"x": None}, {"selected": "ES"})
        group.add({"x": 4.}, {"selected": "ES"})
        metric = group.result()["metrics"]["x"]
        self.assertEqual(metric, {"defined_n": 1, "undefined_n": 1, "mean": 4., "minimum": 4., "maximum": 4.})
        analysis = self.analysis_fixture()
        values, _ = scenario_values(analysis["terminal"], analysis["scenarios"][0], .25)
        self.assertEqual(values["absolute_share_change_vs_old"], .25)
        self.assertIsNone(values["ask_A_feasible_gain"])
        with self.assertRaises(ValueError):
            group.add({"x": float("nan")}, {})

    def test_real_cli_initial_working_tables_and_manifest_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root/"initial"
            self.initial_fixture(initial)
            self.command("time-tables", root/"tables", "--initial", initial)
            all_rows = read(root/"tables/all_sensitivity_states_equal_template_summary.json")
            working = read(root/"tables/working_stimulus_equal_template_summary.json")
            self.assertEqual({r["records"] for r in all_rows if r["condition_id"] == "C1"}, {12})
            self.assertEqual({r["records"] for r in working if r["condition_id"] == "C1"}, {3})
            self.assertEqual(len(read(root/"tables/initial_state_ledger.json")), 1800)
            manifest = read(initial/"manifest.json")
            write(initial/"manifest.json", manifest[:-1]+manifest[:1])
            self.command("time-tables", root/"duplicate", "--initial", initial, success=False)
            write(initial/"manifest.json", manifest[:-1])
            self.command("time-tables", root/"missing", "--initial", initial, success=False)

    def test_real_cli_terminal_pooling_and_smoke_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.terminal_fixture(root/"terminal")
            self.command("time-tables", root/"tables", "--terminal", root/"terminal")
            summary = read(root/"tables/terminal_summary.json")
            self.assertEqual(summary["records"], 2400)
            self.assertEqual(set(summary["support_by_template"].values()), {400})
            self.assertEqual({r["records"] for r in summary["by_condition_profile_rate_equal_template_diagnostic"]}, {1200})
            self.terminal_fixture(root/"smoke", episodes=1)
            self.command("time-tables", root/"partial", "--terminal", root/"smoke", success=False)
            summary = read(root/"terminal/summary.json")
            summary["input_panel"]["policy"] = "different_policy"
            write(root/"terminal/summary.json", summary)
            self.command("time-tables", root/"wrong", "--terminal", root/"terminal", success=False)

    def test_scenarios_and_nonfinite_json_reject(self):
        row = {"condition_id": "C3", "analysis": self.analysis_fixture()}
        check_scenarios(row, self.time, False)
        row["analysis"]["scenarios"][-1] = row["analysis"]["scenarios"][0]
        with self.assertRaises(ValueError):
            check_scenarios(row, self.time, False)
        for raw in ('{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                parse(raw)

    def test_terminal_identity_coverage_rejects_missing_duplicate_and_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"terminal"
            self.terminal_fixture(directory, episodes=1)
            completion = read(directory/"completion.json")
            active = {key: self.budget[key] for key in ACTIVE}
            path = directory/"terminal_impact.jsonl"
            lines = path.read_text().splitlines()
            self.assertEqual(len(list(terminal_rows(directory, path.name, completion, active))), 6)
            extra = parse(lines[0])
            extra["episode_id"] = 1
            for changed in (lines[:-1], lines[:-1]+lines[:1], lines+[json.dumps(extra)]):
                path.write_text("\n".join(changed)+"\n")
                with self.assertRaises(ValueError):
                    list(terminal_rows(directory, path.name, completion, active))

    def test_raw_vs_symmetry_and_inverted_intersection(self):
        row = {"template": "C5_s0_g0", "condition": "C5", "lower": 5., "upper": 8., "gap": 3.,
               "action_bounds": {"stop": {"lower": 0., "upper": 1.},
                                 "ask_A": {"lower": 5., "upper": 7.}, "ask_B": {"lower": 4., "upper": 8.}},
               "initial_details": {"analytical_initial_recipient_symmetry": True},
               "machine_certified_arithmetic": False, "whole_policy_optimality_claimed": False}
        original = deepcopy(row)
        derived = symmetry_bound(row, self.budget["C5_s0_g0"], [.1, .01, .001])
        self.assertEqual((derived["raw_gap"], derived["derived_gap"]), (3., 2.))
        self.assertEqual(row, original)
        row["action_bounds"]["ask_B"]["upper"] = 4.
        with self.assertRaisesRegex(ValueError, "intersection"):
            symmetry_bound(row, self.budget["C5_s0_g0"], [.1, .01, .001])

    def test_real_unexpanded_bound_cli_to_partial_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.command("bounds-root", root/"root", "--template", "C5_s0_g0", "--max-seconds", 0,
                         "--max-created-states", 1, "--max-depth", 0, primary=True)
            self.command("bounds-tables", root/"tables", "--root", root/"root")
            summary = read(root/"tables/bound_summary.json")
            self.assertFalse(summary["coverage"]["full_initial_and_root_and_terminal_coverage"])
            self.assertEqual(len(summary["roots"]), 1)
            self.assertEqual(len(summary["C5_symmetry_derived_bounds"]), 1)
            self.command("bounds-tables", root/"duplicate", "--root", root/"root", "--root", root/"root", success=False)
            self.command("bounds-tables", root/"missing", "--root", root/"absent", success=False)

    def test_terminal_bound_status_check(self):
        row = {"template_id": "C3_s0_g0", "stop_lower": 2., "stop_upper": 2.1,
               "ask_upper": {"ask_A": 1., "ask_B": 1.5}, "machine_certified_arithmetic": False,
               "state": [0, 0, 0, 0], "log_parity_error": 0.,
               "details": {"budget_millihours": self.budget["C3_s0_g0"]["total_millihours"]},
               "status": "numerical_global_stop_at_sampled_terminal"}
        self.assertEqual(terminal_counts([row], self.budget, 1e-8, 1e-10), {"C3": {row["status"]: 1}})
        for field, value in (("status", "inconclusive_sampled_terminal"), ("ask_upper", {}),
                             ("ask_upper", {"ask_A": 1.}), ("state", [999, 999, 0, 0]),
                             ("state", [0, 1, 0, 0]), ("log_parity_error", 1e-3),
                             ("details", {"budget_millihours": 0})):
            bad = {**row, field: value}
            with self.assertRaises(ValueError):
                terminal_counts([bad], self.budget, 1e-8, 1e-10)
        config = self.budget["C3_s0_g0"]
        n = config["max_assessments"]
        row.update(state=[n, n, 0, 0], ask_upper={},
                   details={"budget_millihours": config["total_millihours"]-n*config["cost_millihours"]})
        self.assertEqual(terminal_counts([row], self.budget, 1e-8, 1e-10), {"C3": {row["status"]: 1}})

    @unittest.skipUnless(shutil.which("node"), "Node.js required for JS scientific parity")
    def test_real_engagement_cli_geometry_and_seed_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.command("source-panel", root/"panel")
            panel = read(root/"panel/source_panel.json")
            parity = read(root/"panel/source_parity.json")
            geometry = read(root/"panel/geometry.json")
            self.assertEqual(panel["profileId"], "pilot-engagement-budget-v1")
            self.assertEqual(panel["seed"], 2026100902)
            self.assertEqual(len(panel["schedules"]), 45)
            self.assertTrue(parity["all_pass"])
            self.assertEqual((parity["belief_cases"], parity["outcome_cases"]), (216, 1620))
            self.assertEqual(len(geometry["balanced_geometry"]), 6804)
            for schedule in panel["schedules"]:
                definitions = schedule["definitions"]
                self.assertEqual(sum(d["sideReversed"] for d in definitions), 5)
                self.assertTrue(all(d["capacity"] > 20 for d in definitions))
            request = {"design": self.design, "seed": 2026100902}
            again = subprocess.run([shutil.which("node"), str(ROOT/"src/design_validation/budget_source_panel.cjs")],
                                   input=json.dumps(request), capture_output=True, text=True, check=True, timeout=60)
            self.assertEqual(panel, parse(again.stdout))
            changed = deepcopy(self.design)
            changed["common"]["utility"]["loss_multiplier"] = 3
            write(root/"changed.json", changed)
            self.command("source-panel", root/"wrong_design", "--design", root/"changed.json", success=False)


if __name__ == "__main__":
    unittest.main()
