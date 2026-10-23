"""Purpose: portable dependency closure, exact scientific rosters, guards and refusal semantics.

These tests use synthetic fixtures and original fixed-input identities, never
private results or R fitting. Execute in the declared reference environment.
Actual selected R producer/RDS/GLS/CR2 parity is a separate entry gate.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import pkgutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from src import tutoring_statistics as package
from src.tutoring_statistics import reproduction as run
from src.tutoring_statistics import joint_generation as generation
from src.tutoring_statistics import mixed_tests, null_reference, power_reference
from src.tutoring_statistics import paired_reference, power_origins, effect_shift
from src.tutoring_statistics import independent_power, independent_paired
from src.tutoring_statistics import independent_cluster, independent_fresh
from src.tutoring_statistics import independent_serialization, restart_decision
from src.tutoring_statistics import cluster_inference

ROOT = Path(__file__).resolve().parents[1]


class PortableStatisticsTests(unittest.TestCase):
    def concurrent_children(self, code, output, barrier, values):
        """Release two stdlib-only children at an explicit race boundary."""
        barrier.mkdir()
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        children = []
        try:
            for index, value in enumerate(values):
                children.append(subprocess.Popen([sys.executable, "-c", code, str(output),
                    str(barrier), str(index), str(value)], cwd=ROOT, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            deadline = time.monotonic() + 20
            while not all((barrier / str(index)).exists() for index in range(2)):
                if any(child.poll() is not None for child in children) or time.monotonic() > deadline:
                    self.fail("Concurrent child did not reach the race boundary")
                time.sleep(.01)
            (barrier / "go").touch()
            results = []
            for child in children:
                stdout, stderr = child.communicate(timeout=30)
                self.assertEqual(child.returncode, 0, stderr)
                results.append(json.loads(stdout))
            return results
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.communicate()

    def test_dependency_closure(self):
        for item in pkgutil.iter_modules(package.__path__):
            if not item.name.startswith("comparison_"):
                importlib.import_module(package.__name__ + "." + item.name)

    def test_rejected_diagnostic_reducer_not_exported(self):
        self.assertFalse(hasattr(cluster_inference, "method_report"))
        self.assertFalse(hasattr(cluster_inference, "aggregate_reports"))
        self.assertEqual(cluster_inference.distribution([1., 2., None])["mean"], 1.5)

    def test_portable_cli_help_and_plan(self):
        env = os.environ.copy()
        script = ROOT / "scripts/reproduce_conditional_statistics.py"
        for args in (["--help"], ["plan"]):
            result = subprocess.run([sys.executable, str(script), *args], env=env,
                                    capture_output=True, text=True, check=True)
            self.assertTrue(result.stdout)

    def test_exact_null_roster_and_seed_separation(self):
        config = run.read(run.ROOT / "data/mixed_calibration.json")
        plan = null_reference.make_plan(config)
        self.assertEqual([c["caseIndex"] for c in plan["cases"]],
                         [1, 4, 5, 7, 9, 12, 13, 15, 17, 20, 21, 23])
        seeds = [s for block in null_reference.population(plan) for s in block["seeds"]]
        self.assertEqual(len(seeds), 24000)
        self.assertEqual(len(set(seeds)), 24000)
        self.assertEqual(seeds[0], 120271020)
        changed = copy.deepcopy(config)
        changed["seedBase"] += 1
        with self.assertRaises(ValueError):
            null_reference.make_plan(changed)

    def test_paired_identity_and_bounds(self):
        first, last = paired_reference.case_for(1), paired_reference.case_for(10000)
        self.assertEqual(first["designSeed"], 602650201)
        self.assertEqual(first["responseSeed"], 703010201)
        self.assertEqual(last["repID"], 10000)
        self.assertEqual(first["N"], 154)
        for rep in (0, 10001, True):
            with self.assertRaises(ValueError):
                paired_reference.case_for(rep)

    def test_power_roster_independent_identity_and_origins(self):
        self.assertEqual(power_reference.cells(), independent_power.expected_cell_roster())
        for ordinal in (0, 1, 2, 6, 39, 40, 39999, 40000, 79999):
            case = power_reference.case_for(ordinal, 2000)
            self.assertEqual(case, independent_power.case_for(ordinal, 2000))
            self.assertEqual(power_origins.planned_origin(case["caseID"], case["repID"]),
                             independent_power.legacy_origin(case))
        origins = [independent_power.legacy_origin(power_reference.case_for(i, 2000))
                   for i in (0, 1, 2)]
        self.assertEqual([x["kind"] if x else None for x in origins],
                         ["actual_shift", None, "original_production"])
        for ordinal in (-1, 80000, True):
            with self.assertRaises(ValueError):
                independent_power.case_for(ordinal, 2000)

    def test_original_shift_identity_and_no_redraw_transform(self):
        identity = effect_shift.identity(4, "parity")
        self.assertEqual(identity["sourceOrdinal"], 3)
        self.assertEqual(identity["case"]["designSeed"], 402620191)
        rows = [dict(participant="p", condition="C2", need_A=3., need_B=4., y=5.) for _ in range(4620)]
        actual = dict(model="H10", reference="C2", rows=rows)
        delta = dict(zip(generation.H10_IDS, [.2, .3, .4, .5]))
        shifted = effect_shift.shifted_request(actual, delta)
        self.assertEqual(actual["rows"][0]["y"], 5.)
        self.assertEqual(shifted["rows"][0]["y"], 5. - .2*3. - .3*4.)

    def test_guard_unavailable_and_tail_conventions(self):
        reference = dict(estimate=1., SE=.2, df=100., statistic=5., CI=[.6, 1.4])
        unavailable = mixed_tests.directional_result(reference, "greater", usable=False)
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertFalse(unavailable["candidateNominalReject"])
        self.assertIsNone(unavailable["p"])
        self.assertLess(mixed_tests.directional_result(reference, "greater")["p"], .05)
        self.assertGreater(mixed_tests.directional_result(reference, "less")["p"], .95)
        self.assertEqual(generation.tail_for("literal", "C1-C2:B"), "less")
        self.assertEqual(generation.tail_for("sign_consistent", "C1-C2:B"), "greater")

    def test_original_optimizer_guards(self):
        route = dict(extra=dict(theta=dict(names=["a", "b", "c"], values=[1., 0., 1.],
                    lower=[0, None, 0], lowerKinds=["finite", "-Inf", "finite"])))
        self.assertTrue(restart_decision.theta_valid(route))
        route["extra"]["theta"]["lower"][1] = 0
        self.assertFalse(restart_decision.theta_valid(route))
        self.assertEqual(restart_decision.CONTROLS[1]["maxeval"], 100000)
        self.assertEqual(len(restart_decision.SCREENS), 6)

    def test_serialization_permission_is_narrow(self):
        original = {"methods": {m: {"extra": {"theta": {"lower": [0, "-Inf", 0]}}}
                                for m in independent_serialization.METHODS}}
        canonical = copy.deepcopy(original)
        for m in independent_serialization.METHODS:
            canonical["methods"][m]["extra"]["theta"]["lower"][1] = None
        self.assertEqual(len(independent_serialization.verify_finite_delta(
            original, canonical, list(independent_serialization.METHODS))), 3)
        canonical["methods"]["baseline_replay"]["extra"]["theta"]["lower"][0] = 1
        with self.assertRaises(ValueError):
            independent_serialization.verify_finite_delta(original, canonical,
                                                         list(independent_serialization.METHODS))

    def test_missing_observations_do_not_become_zero_rates(self):
        self.assertIsNone(independent_cluster.rate(0, 0))
        report = independent_paired.reduce_events([], dict(case=paired_reference.case_for(1),
                 modules=dict(fresh=independent_fresh, math=independent_cluster)), planned=10000)
        self.assertFalse(report["complete"])
        self.assertEqual(report["missingTotal"], 10000)
        self.assertIsNone(report["oracleCoverage"]["C1-C3:A"]["coverageAll"])

    def test_atomic_resume_and_changed_artifact_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            descriptor = run.write(path, {"x": 1})
            before = path.read_bytes()
            self.assertEqual(run.write(path, {"x": 1}), descriptor)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                run.write(path, {"x": 2})
            path.write_text('{"x": 2}')
            with self.assertRaises(ValueError):
                run.validate(descriptor)

    def test_two_process_first_start_and_manifest_resume(self):
        code = r'''
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from src.tutoring_statistics import reproduction as io
output, barrier = Path(sys.argv[1]), Path(sys.argv[2])
index = int(sys.argv[3])
def no_runtime():
    raise AssertionError("Synthetic initialization must not enter a numerical runtime")
io.numerical_runtime = no_runtime
io.inventory = lambda: dict(files=[], syntheticConcurrencyFixture=True)
def fixture_native(args, root):
    native = root / "native"
    if not native.exists():
        native.mkdir()
        # Keep the first owner inside initialization while the other contends.
        time.sleep(.15)
        for name in ("resolved_design.json", "initial_panel.json", "h10_canonical.json"):
            io.write(native / name, dict(synthetic=True))
        io.write(root / "fixture_creator.json", dict(index=index))
    else:
        for name in ("resolved_design.json", "initial_panel.json", "h10_canonical.json"):
            assert io.read(native / name) == dict(synthetic=True)
    return native
io.prepare_native = fixture_native
(barrier / str(index)).touch()
deadline = time.monotonic() + 20
while not (barrier / "go").exists():
    if time.monotonic() > deadline:
        raise TimeoutError("Race barrier timed out")
    time.sleep(.01)
args = SimpleNamespace(rscript=sys.executable, reference_dir=None)
native, manifest = io.initialize_output(args, "null", output)
before = Path(manifest["path"]).read_bytes()
again_native, again_manifest = io.initialize_output(args, "null", output)
assert native == again_native and manifest == again_manifest
assert Path(manifest["path"]).read_bytes() == before
io.write(output / "synthetic_disjoint_members" / (str(index) + ".json"), dict(index=index))
print(json.dumps(dict(manifest=manifest, identity=io.pin(output / "output_identity.json"),
                     source=io.pin(output / "source_snapshot.json"))))
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            results = self.concurrent_children(code, output, root / "barrier", (0, 1))
            self.assertEqual(results[0], results[1])
            self.assertIn(run.read(output / "fixture_creator.json")["index"], (0, 1))
            self.assertEqual(sorted(p.stem for p in (output / "synthetic_disjoint_members").glob("*.json")),
                             ["0", "1"])
            self.assertFalse(list(output.rglob(".new-*")))

    def test_two_process_atomic_writes_converge_or_refuse_conflicts(self):
        code = r'''
import json
import sys
import time
from pathlib import Path
from src.tutoring_statistics import reproduction as io
output, barrier = Path(sys.argv[1]), Path(sys.argv[2])
index, value = int(sys.argv[3]), int(sys.argv[4])
link = io.os.link
def simultaneous_link(source, destination):
    (barrier / str(index)).touch()
    deadline = time.monotonic() + 20
    while not (barrier / "go").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Race barrier timed out")
        time.sleep(.01)
    return link(source, destination)
io.os.link = simultaneous_link
try:
    descriptor = io.write(output / "value.json", dict(value=value))
    result = dict(status="written", descriptor=descriptor)
except ValueError as error:
    assert "Refusing changed existing JSON" in str(error)
    result = dict(status="conflict")
print(json.dumps(result))
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            same = self.concurrent_children(code, root / "same", root / "same_barrier", (1, 1))
            self.assertEqual(same[0], same[1])
            self.assertEqual(same[0]["status"], "written")
            different = self.concurrent_children(code, root / "different", root / "different_barrier", (1, 2))
            self.assertEqual(sorted(result["status"] for result in different), ["conflict", "written"])
            self.assertIn(run.read(root / "different/value.json")["value"], (1, 2))
            self.assertFalse(list(root.rglob(".new-*")))

    def test_initialization_refuses_unowned_and_partial_outputs(self):
        args = type("Arguments", (), {"reference_dir": None})()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("foreign.json", "source_snapshot.json"):
                output = root / name.replace(".json", "")
                run.write(output / name, dict(uncommitted=True))
                with mock.patch.object(run, "prepare_native", side_effect=AssertionError("Native reached")), \
                     mock.patch.object(run, "inventory", side_effect=AssertionError("Inventory reached")):
                    with self.assertRaisesRegex(ValueError, "new output directory"):
                        run.initialize_output(args, "null", output)
                self.assertEqual(run.read(output / name), dict(uncommitted=True))
                self.assertFalse((output / "output_identity.json").exists())

    def test_event_digest_and_exact_case_admission(self):
        case = paired_reference.case_for(1)
        manifest = {"path": "synthetic", "sha256": "synthetic"}
        event = dict(ordinal=0, case=case, caseID=case["caseID"], repID=1,
                     fitKey=paired_reference.fit_key(case))
        saved = dict(schema=run.SCHEMA, manifest=manifest, event=event,
                     eventSha256=run.event_digest(event))
        self.assertEqual(run.admit_event(saved, "paired", manifest), event)
        changed = copy.deepcopy(saved)
        changed["event"]["repID"] = 2
        with self.assertRaises(ValueError):
            run.admit_event(changed, "paired", manifest)
        changed["eventSha256"] = run.event_digest(changed["event"])
        with self.assertRaises(ValueError):
            run.admit_event(changed, "paired", manifest)

    def test_cli_refuses_ambiguous_or_invalid_selection_before_science(self):
        script = ROOT / "scripts/reproduce_conditional_statistics.py"
        for args in (["--ids", "0,0"], ["--ids", "24000"], ["--full", "--ids", "0"],
                     ["--first", "0"], ["--ids", "-1"]):
            result = subprocess.run([sys.executable, str(script), "run", "--execute",
                      "--family", "null", "--output", "unused-output", *args],
                      capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_notebook_safety_and_packaged_assets(self):
        for name in ("mixed_model_null_validation", "paired_coverage_and_conditional_power"):
            notebook = json.loads((ROOT / "notebooks/statistical_methods" / (name + ".ipynb")).read_text())
            code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            self.assertIn("pip", "".join(code[0]["source"]))
            self.assertTrue(all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code))
        for name in ("mixed_fit.R", "kernel_entry.R", "source_transport.R", "panel_engine.cjs",
                     "data/native_identity.json", "data/mixed_calibration.json"):
            self.assertTrue((run.ROOT / name).is_file(), name)

    def test_protected_output_refused_before_numerical_runtime(self):
        for destination in [ROOT, ROOT.parent, *[ROOT / name / "invalid-statistics-output"
                            for name in ("src", "scripts", "configs", "tests", "notebooks")]]:
            arguments = ["runner", "run", "--execute", "--family", "null", "--ids", "0",
                         "--output", str(destination)]
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(
                    run, "numerical_runtime", side_effect=AssertionError("Runtime reached")):
                with self.assertRaisesRegex(ValueError, "immutable"):
                    run.main()
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "references"
            for destination in (reference, reference / "child", reference.parent):
                with self.assertRaises(ValueError):
                    run.validate_output(destination, reference)
            self.assertEqual(run.validate_output(Path(directory) / "generated", reference),
                             Path(directory) / "generated")

    def test_symlink_destination_cannot_bypass_protected_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "config-alias"
            alias.symlink_to(ROOT / "configs", target_is_directory=True)
            with self.assertRaises(ValueError):
                run.validate_output(alias / "invalid-statistics-output")
            self.assertFalse((alias / "invalid-statistics-output").exists())

    def test_exact_roster_status_and_optional_reference_independence(self):
        # A synthetic two-member orchestration fixture, not a scientific rerun.
        manifest = {"path": "synthetic", "sha256": "synthetic"}
        args = type("Arguments", (), {"reference_dir": None})()
        reducer = mock.Mock()
        reducer.reduce_events.return_value = {}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(run.TOTAL, paired=2), \
             mock.patch.object(run, "validate_manifest"), mock.patch.object(run, "inventory"), \
             mock.patch.object(run, "joint_modules", return_value={"reduction": None, "math": None}), \
             mock.patch.object(run, "module", return_value=reducer), \
             mock.patch.object(run, "compare_reference", side_effect=AssertionError("Reference accessed")):
            output = Path(directory)
            for ordinal in (0, 1):
                case = paired_reference.case_for(ordinal + 1)
                event = dict(ordinal=ordinal, case=case, caseID=case["caseID"], repID=ordinal+1,
                             fitKey=paired_reference.fit_key(case))
                run.write(output / "paired/events" / f"{ordinal:05d}.json", dict(
                    schema=run.SCHEMA, manifest=manifest, event=event,
                    eventSha256=run.event_digest(event), artifacts=[]))
                result = run.read(run.aggregate(args, "paired", output, manifest)["path"])
                complete = ordinal == 1
                self.assertEqual(result["fullRosterComplete"], complete)
                self.assertEqual(result["status"], "full_reproduction" if complete else "partial_entry_check")
                self.assertEqual(result["presentTotal"], ordinal+1)
                self.assertIsNone(result["referenceComparison"]["passed"])
                if not complete:
                    self.assertFalse((output / "paired/aggregate_full.json").exists())


if __name__ == "__main__":
    unittest.main()
