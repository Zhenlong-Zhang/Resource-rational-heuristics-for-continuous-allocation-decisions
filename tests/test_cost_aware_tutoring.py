"""Test purpose: quick cost-aware invariants and adversarial portable interfaces.

Boundary fixtures are not accepted histories. The real selected-trial CLI chain
is run separately, so these tests never repeat the scientific parity selection.
"""
import copy
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.design_validation.bounded_task import TaskConfig
from src.design_validation.budget_horizon import BudgetTask
from src.design_validation import cost_aware_policy as P
from src.design_validation import cost_aware_reference as R
from src.design_validation.cost_aware_bounds import expected_pathwise_max
from src.design_validation.cost_aware_inputs import (bits, compact, generate_source, load_context, log_bits, parse,
    payload, read, selections, source_selection, trial_name, validate_context, validate_source_scope, write)
from src.design_validation.cost_aware_reproduction import load_inputs
from src.design_validation.cost_aware_validation import Comparisons, Replay, record_rows, verified_reference

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {"conditionId": "C2", "scheduleVariant": 0, "mainIndex": 1,
             "replicateId": 1, "scenarioId": "base|rho_100"}


class TestCostAwareInterfaces(unittest.TestCase):
    def test_selected_source_request_deduplicates_roots_not_trials(self):
        selected = [SELECTION, {**SELECTION, "replicateId": 2},
                    {**SELECTION, "mainIndex": 10}, {**SELECTION, "mainIndex": 2}]
        expected = [{"conditionId": "C2", "scheduleVariant": 0, "mainIndex": main} for main in (1, 2, 10)]
        self.assertEqual(source_selection(selected), expected)
        self.assertEqual(len(selections(selected)), 4)
        context = load_context(ROOT / "configs/tutoring_cost_aware.json")
        with tempfile.TemporaryDirectory() as directory, \
             patch("src.design_validation.cost_aware_inputs.subprocess.run") as node:
            generate_source(context, Path(directory) / "source.json", "selected-node", selected=selected)
            args = node.call_args.args[0]
            self.assertEqual(args[0], "selected-node")
            self.assertEqual(Path(args[-1]), ROOT / "src/design_validation/cost_aware_source.cjs")
            self.assertEqual(parse(node.call_args.kwargs["input"]), {**context, "roots": expected})
            self.assertIn("x.construction,x.roots", args[2])

    def test_partial_source_scope_is_explicit_and_exact(self):
        selected = [{**SELECTION, "mainIndex": 10}]
        root = source_selection(selected)[0]
        source = {"schema": "finite-cost-aware-source-roots-v1", "scope": "selected_working_roots",
                  "fullPanel": False, "syntheticOnly": True, "humanObservations": False,
                  "status": "constructed_selected_roots", "rootSelection": [root],
                  "states": [{**root, "prefixLength": 0, "prefixKind": "fixed", "paid": []}],
                  "sessions": [{"conditionId": "C2", "scheduleVariant": 0, "actualPhase": "trial",
                                "completedTrialCount": 13, "completionClaim": "selected_root_prefix_only;no_complete_session_claim",
                                "exported": {"phase": "trial", "status": "incomplete", "currentTrialIndex": 13}}]}
        validate_source_scope(source, selected)
        for edit in (lambda x: x.update(fullPanel=True), lambda x: x["states"].append(copy.deepcopy(x["states"][0])),
                     lambda x: x["states"][0].update(prefixLength=1),
                     lambda x: x["sessions"][0]["exported"].update(status="complete"),
                     lambda x: x["sessions"][0].update(completedTrialCount=14)):
            changed = copy.deepcopy(source); edit(changed)
            with self.assertRaises(ValueError):
                validate_source_scope(changed, selected)

    def test_selection_preserves_exact_coordinates(self):
        selected = selections(read(ROOT / "configs/tutoring_cost_aware_selection.json"))
        self.assertEqual(selected[0], SELECTION)
        self.assertEqual(trial_name(selected[0]), "C2_v0_m1_r1_base_rho_100")
        self.assertEqual(trial_name(selected[1]), "C3_v0_m1_r1_low_rho_0")
        for bad in ([], [SELECTION, SELECTION], [{**SELECTION, "replicateId": True}],
                    [{**SELECTION, "mainIndex": 0}], [{**SELECTION, "scenarioId": "base|rho_2"}],
                    [{**SELECTION, "scenario": "extra"}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                selections(bad)

    def test_strict_json_and_binary64(self):
        for raw in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse(raw)
        self.assertNotEqual(payload(1), payload(1.))
        self.assertNotEqual(bits(-0.), bits(0.))
        with self.assertRaises(ValueError):
            Comparisons().exact(-0., 0., "signed zero")
        with self.assertRaises(ValueError):
            Comparisons().exact(True, 1, "typed identity")

    def test_recipe_changes_are_not_accepted_settings(self):
        context = load_context(ROOT / "configs/tutoring_cost_aware.json")
        for edit in (lambda x: x["settings"].update(personal_question_time_priced=True),
                     lambda x: x["settings"]["assumed_profiles"]["low"].update(EO_inference_seconds=6.),
                     lambda x: x["recipe"].update(decision_tolerance=1e-8)):
            changed = copy.deepcopy(context); edit(changed)
            with self.assertRaises(ValueError):
                validate_context(changed)

    def test_ndjson_exact_bytes_and_truncation(self):
        row = compact({"kind": "test", "value": 1.}).encode()
        sha = hashlib.sha256(row).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.ndjson"
            path.write_bytes(row+b"\n")
            self.assertEqual(list(record_rows(path, [sha])), [{"kind": "test", "value": 1.}])
            for content, hashes in ((row, [sha]), (row+b"\n", [sha, sha]),
                                    (row+b"\n"+row+b"\n", [sha]), (row+b" \n", [sha])):
                path.write_bytes(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    list(record_rows(path, hashes))

    def test_external_selection_and_fresh_outputs(self):
        context = load_context(ROOT / "configs/tutoring_cost_aware.json")
        root = {k: SELECTION[k] for k in ("conditionId", "scheduleVariant", "mainIndex")}
        inputs = {"schema": "finite-cost-aware-inputs-v1", "scope": "selected_working_trials",
                  "publicIdentity": "not_a_historical_manifest", "context": context,
                  "selection": [SELECTION], "roots": [root]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            write(path, inputs)
            self.assertEqual(load_inputs(directory, context, [SELECTION]), inputs)
            with self.assertRaises(FileExistsError):
                write(path, inputs)
            with self.assertRaises(ValueError):
                load_inputs(directory, context, [{**SELECTION, "replicateId": 2}])

    def test_cli_dispatch_without_running_science(self):
        spec = importlib.util.spec_from_file_location("cost_aware_cli", ROOT / "scripts/reproduce_cost_aware_tutoring.py")
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        with patch("src.design_validation.cost_aware_reproduction.produce") as producer:
            producer.return_value = {"selection": [SELECTION], "trials": [{"history": {"status": "completed"}}]}
            self.assertEqual(cli.main(["produce", "--selection", "selected.json", "--inputs", "inputs", "--output", "records"]), 0)
            self.assertEqual(producer.call_args.args[1:], (Path("selected.json"), Path("inputs"), Path("records")))
        with self.assertRaises(SystemExit):
            cli.main(["check", "--selection", "selected.json", "--inputs", "inputs", "--output", "checked"])


class TestCostAwareNumerics(unittest.TestCase):
    def boundary_fixture(self, child_increments):
        """Controlled menus isolate policy discontinuities; Bayes panels are real."""
        settings = load_context(ROOT / "configs/tutoring_cost_aware.json")["settings"]
        config = TaskConfig("C2", "boundary_fixture", 1., 0., (45., 75.), (2., 2.), 6., 2000, 1000, 0)
        task = BudgetTask(config); logs = list(task.initial_logs)
        selection = {**SELECTION, "scenarioId": "low|rho_0"}
        root = {"populationId": "working", **{k: selection[k] for k in ("conditionId", "scheduleVariant", "mainIndex")},
                "config": asdict(config), "initialLogBinary64": log_bits(logs), "logicalToDisplayed": ["A", "B"]}
        reference = {"rho0_GPA_per_personal_second": .01}
        templates = {budget: R.terminal(root["config"], logs, budget) for budget in (0, 1000)}
        def menu(child_logs, budget):
            value = 7.
            if budget == 0:
                slot = 0 if log_bits(child_logs)[0] != root["initialLogBinary64"][0] else 1
                value += child_increments[slot]
            result = copy.deepcopy(templates[budget])
            result["raw_grid_max_combined_GPA"] = value
            result["raw_EO_minimum_GPA_gap"] = result["EO_selected_gap_excess"] = 0.
            for row in result["macros"].values():
                row.update(expected_combined_GPA=value, expected_GPA_A=value/2, expected_GPA_B=value/2,
                           expected_utility=(value-7.)/.03, expected_GPA_gap=0., regret_vs_raw_grid_GPA_max=0.)
            return result
        with patch.object(P, "terminal_summary", side_effect=lambda config, a, b, budget: menu([a, b], budget)):
            decision = P.decision_table(root, task, logs, 1000, P.scenario(settings, selection, reference), settings)
        reader = Replay(root, selection, settings, reference)
        return decision, reader, menu, logs, settings

    @staticmethod
    def shift_menu(menu, delta):
        menu["raw_grid_max_combined_GPA"] += delta
        for row in menu["macros"].values():
            row["expected_combined_GPA"] += delta
            row["expected_GPA_A"] += delta/2
            row["expected_GPA_B"] += delta/2
            row["expected_utility"] += delta/.03

    @staticmethod
    def refreeze(decision, settings):
        """Update every recorded operand, so rejection cannot be mere corruption."""
        decision["currentSelection"] = R.select(decision["currentSummary"], R.durations(settings, "low", 1000), 0.)
        children, asks, raw = [], [], []
        for slot in range(2):
            rows = [r for block in decision["reportBlocks"][slot*10:(slot+1)*10] for r in block["rows"]]
            selected = [R.select(row["summary"], R.durations(settings, "low", 0), 0.) for row in rows]
            children.append(selected)
            probabilities = np.array([r["probability"] for r in rows], dtype=np.float64)
            asks.append(float(probabilities @ np.array([s["selected_objective"] for s in selected], dtype=np.float64)))
            raw.append(float(probabilities @ np.array([s["raw_max_objective"] for s in selected], dtype=np.float64)))
        decision.update(scenarioChildSelections=children, askSelected=asks, askRaw=raw)
        decision.update(R.policy(decision["currentSelection"]["selected_objective"], asks, 1000, 1000))
        decision["decisionReason"] = "voluntary_myopic" if decision["action"] == "stop" else "positive_selected_one_query_gain"

    def test_affordable_subtolerance_ask_stop_disagreement(self):
        decision, reader, menu, logs, settings = self.boundary_fixture((9.9e-11, 9.9e-11))
        with patch.object(reader, "summary", side_effect=menu):
            reader.decision(decision, logs, 1000)
            self.assertEqual(decision["action"], "stop")
            changed = copy.deepcopy(decision)
            self.shift_menu(changed["currentSummary"], -2e-12)
            for block in changed["reportBlocks"]:
                for row in block["rows"]:
                    self.shift_menu(row["summary"], 2e-12)
            self.refreeze(changed, settings)
            self.assertEqual(changed["action"], "ask_A")
            with self.assertRaisesRegex(ValueError, "Independent all-five myopic rule"):
                reader.decision(changed, logs, 1000)

    def test_affordable_subtolerance_ask_tie_disagreement(self):
        decision, reader, menu, logs, settings = self.boundary_fixture((5e-10, 5.99e-10))
        with patch.object(reader, "summary", side_effect=menu):
            reader.decision(decision, logs, 1000)
            self.assertEqual(decision["action"], "ask_A")
            self.assertTrue(decision["actionTie"])
            changed = copy.deepcopy(decision)
            for block in changed["reportBlocks"]:
                delta = -2e-12 if block["displayedSlot"] == 0 else 2e-12
                for row in block["rows"]:
                    self.shift_menu(row["summary"], delta)
            self.refreeze(changed, settings)
            self.assertEqual(changed["action"], "ask_B")
            self.assertFalse(changed["actionTie"])
            with self.assertRaisesRegex(ValueError, "Independent all-five myopic rule"):
                reader.decision(changed, logs, 1000)

    def test_tolerated_reference_shift_cannot_redefine_price(self):
        context = load_context(ROOT / "configs/tutoring_cost_aware.json")
        original = R.reference_record(context["design"])
        shifted = dict(original)
        shifted["rho0_GPA_per_personal_second"] += 5e-11
        independent = verified_reference(context["design"], shifted, Comparisons())
        self.assertEqual(independent, original)
        config = TaskConfig("C2", "boundary_fixture", 1., 0., (45., 75.), (2., 2.), 6., 2000, 1000, 0)
        task = BudgetTask(config); logs = list(task.initial_logs)
        root = {"populationId": "working", **{k: SELECTION[k] for k in ("conditionId", "scheduleVariant", "mainIndex")},
                "config": asdict(config), "initialLogBinary64": log_bits(logs), "logicalToDisplayed": ["A", "B"]}
        settings = context["settings"]
        decision = P.decision_table(root, task, logs, 500, P.scenario(settings, SELECTION, shifted), settings)
        reader = Replay(root, SELECTION, settings, independent)
        self.assertEqual(reader.rho, 100.*original["rho0_GPA_per_personal_second"])
        self.assertNotEqual(reader.rho, 100.*shifted["rho0_GPA_per_personal_second"])
        with self.assertRaisesRegex(ValueError, "Exact selector"):
            reader.decision(decision, logs, 500)

    def test_cdf_strict_boundary_fallback_and_no_repair(self):
        probabilities = np.zeros(100); probabilities[0] = probabilities[-1] = .5
        logs = np.zeros(100)
        for uniform, category in ((0., 1), (.5, 100)):
            self.assertEqual(P.categorical(logs, probabilities, uniform)["category"], category)
            self.assertEqual(R.categorical(probabilities, uniform)["category"], category)
        probabilities[-1] -= 5e-13
        self.assertTrue(P.categorical(logs, probabilities, 1.-1e-13)["fallback"])
        self.assertTrue(R.categorical(probabilities, 1.-1e-13)["fallback"])
        probabilities[-1] -= 1e-5
        for compute in (lambda: P.categorical(logs, probabilities, .5), lambda: R.categorical(probabilities, .5)):
            with self.assertRaises(ValueError):
                compute()

    def test_stream_key_is_exact_and_scenario_free(self):
        key = {"namespace": "r9-repeated-macro-bridge-v1", "conditionId": "C2", "scheduleVariant": 0,
               "replicateId": 1, "mainIndex": 1, "logicalRecipient": "A", "streamKind": "report", "reportIndex": 0}
        self.assertEqual(P.stream_uniform(key), R.stream_uniform(key))
        self.assertNotEqual(P.stream_uniform(key)["digest"], P.stream_uniform({**key, "reportIndex": 1})["digest"])
        for compute in (P.stream_uniform, R.stream_uniform):
            with self.assertRaises(ValueError):
                compute({**key, "scenarioId": "low|rho_0"})

    def test_pathwise_max_not_maximum_of_expectations(self):
        result = expected_pathwise_max(np.array([.5, .5]), np.array([0., 9.]), np.array([10., 10.]), 1., 5.)
        self.assertEqual(result["stronger_GPA_upper"], 7.)
        self.assertEqual(result["unsafe_max_of_expectations_NOT_a_bound"], 5.)

    def test_action_threshold_and_terminal_tie(self):
        self.assertEqual(R.policy(0., [1e-10, 0.], 1000, 1000)["action"], "stop")
        self.assertEqual(R.policy(0., [1., 1.+5e-11], 1000, 1000)["action"], "ask_A")
        self.assertEqual(R.policy(0., [1., 1.+2e-10], 1000, 1000)["action"], "ask_B")
        summary = {"macros": {m: {"expected_combined_GPA": 7., "index": 500} for m in R.MACROS}}
        times = dict.fromkeys(R.MACROS, 3.); times["PN"] = 2.
        self.assertEqual(R.select(summary, times, 0.)["selected_macro"], "PN")

    def test_zero_decision_and_independent_reader_rejections(self):
        settings = load_context(ROOT / "configs/tutoring_cost_aware.json")["settings"]
        config = TaskConfig("C2", "boundary_fixture", 1., 0., (45., 75.), (2., 2.), 6., 2000, 1000, 0)
        task = BudgetTask(config)
        logs = list(task.initial_logs)
        root = {"populationId": "working", **{k: SELECTION[k] for k in ("conditionId", "scheduleVariant", "mainIndex")},
                "config": asdict(config), "initialLogBinary64": log_bits(logs), "logicalToDisplayed": ["A", "B"]}
        reference = {"rho0_GPA_per_personal_second": .01}
        scenario = P.scenario(settings, SELECTION, reference)
        decision = P.decision_table(root, task, logs, 0, scenario, settings)
        reader = Replay(root, SELECTION, settings, reference)
        with patch.object(P, "decision_table", side_effect=AssertionError("Producer oracle used")), \
             patch.object(P, "terminal_summary", side_effect=AssertionError("Producer terminal used")):
            reader.decision(decision, logs, 0)
            self.assertEqual(decision["action"], "stop")
            self.assertIsNone(decision["currentSelection"]["selected_index"])
            self.assertEqual(set(decision["currentSelection"]["duration_by_macro_seconds"].values()), {0.})
            for edit in (lambda x: x.update(action="ask_A"), lambda x: x.update(budget=1),
                         lambda x: x.update(mapRole=BRANCH),
                         lambda x: x["currentSummary"]["macros"]["ES"].update(expected_combined_GPA=9.)):
                changed = copy.deepcopy(decision); edit(changed)
                with self.assertRaises(ValueError):
                    reader.decision(changed, logs, 0)


BRANCH = "predictive_joint_branch_v1"

if __name__ == "__main__":
    unittest.main()
