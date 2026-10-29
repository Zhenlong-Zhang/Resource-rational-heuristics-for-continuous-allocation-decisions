"""Test purpose: fixed study membership, original reductions and safe resume.

Synthetic fixtures exercise arithmetic and transport, not new policy evidence.
Scientific trajectory kernels have their separate independent-replay tests.
"""
import copy
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.design_validation import cost_aware_campaign as C
from src.design_validation import cost_aware_aggregation as A
from src.design_validation import cost_aware_reproduction as R
from src.design_validation.cost_aware_inputs import read, write

ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def test_complete_roster_is_exact_and_partition_independent(self):
        expected = {(s, c, v, r) for s in C.SCENARIOS for c in C.ARMS
                    for v in range(9) for r in range(64)}
        for size in (8, 19, 64):
            roster = C.plan("full", size)
            actual = [(t["scenarioId"], t["conditionId"], t["scheduleVariant"], r)
                      for t in roster["tasks"] for r in t["replicateIds"]]
            self.assertEqual(len(actual), 51840)
            self.assertEqual(set(actual), expected)
            self.assertEqual(roster["histories"], 518400)
        sample = C.task_selection(C.plan("full")["tasks"][0])
        self.assertEqual(len(sample), 80)
        self.assertEqual({r["mainIndex"] for r in sample}, set(range(1, 11)))

    def test_reduction_roster_preserves_original_endpoints(self):
        spec = read(ROOT / "configs/expected_gpa_analysis.json")
        endpoints = list(A.endpoint_specs(spec))
        q = [s for _, s in endpoints if s["endpoint"] == "Q"]
        self.assertEqual(len(q), 2)
        self.assertEqual((len(endpoints) - 2) * 2 * 18, 2052)
        self.assertEqual(next(s for _, s in endpoints if s["id"] == "H06_R_EGPA_C2")["endpoint"],
                         "R_EGPA_pp")

    def test_person_ordered_arithmetic_and_exhaustion_eligibility(self):
        manifest, scenario = "fixture", "base|rho_1"
        prefix = [manifest, "working", scenario, "C2", 0, 0]
        rows = []
        for main in range(1, 11):
            payload = {"personId": A.semantic(prefix), "populationId": "working",
                "scenarioId": scenario, "conditionId": "C2", "variant": 0,
                "mainIndex": main, "trialId": str(main), "choiceAvailable": True,
                "remainingIntegerBudget": 1000, "share": main / 20., "x0": float(main),
                "xT": float(main), "rawInitialGap": float(main), "scale": float(main % 3),
                "gapOffset": float(main % 2), "lowerMeanTie": False, "paidCount": 2,
                **{key: float(main) for key in A.MEANS}}
            rows.append(A.make_record("trial-metrics", manifest, prefix + [main], payload))
        person = A.model_person_endpoints(rows, {}, None)
        self.assertEqual(person["endpoints"]["Q"]["value"], 5.5)
        self.assertAlmostEqual(person["endpoints"]["initial_slope"]["value"], .05)
        changed = copy.deepcopy(rows)
        payload = {k: v for k, v in changed[-1].items()
                   if k not in ("schema", "manifestHash", "recordId", "semanticHash", "historicalReceipt")}
        payload.update(remainingIntegerBudget=0, choiceAvailable=False)
        changed[-1] = A.make_record("trial-metrics", manifest, prefix + [10], payload)
        endpoints = A.model_person_endpoints(changed, {}, None)["endpoints"]
        self.assertEqual(endpoints["Q"]["status"], "available")
        self.assertEqual(endpoints["R_EGPA_pp"]["status"], "scientific_ineligible")
        self.assertEqual(endpoints["available_history_R_EGPA_pp"]["observations"], 9)
        with self.assertRaises(ValueError):
            A.model_person_endpoints(rows[::-1], {}, None)

    def test_stratified_mean_and_person_mcse(self):
        store, refs = C.Store(), {}
        for variant in range(9):
            for rep in range(64):
                person = A.make_record("person", "fixture", [variant, rep], {
                    "complete": True, "endpoints": {"initial_slope": {
                        "status": "available", "value": float(variant + rep)}}})
                refs[variant, rep] = store.add(person)
        group = {"spec": {"id": "H03_initial_C2", "endpoint": "initial_slope",
                          "weights": {"C2": 1}, "tail": "greater"},
                 "route": "assigned_eligible_person", "scenarioId": "low|rho_0",
                 "populationScope": "working", "namespace": "primary",
                 "personRefs": lambda arm, v: [refs[v, r] for r in range(64)]}
        result = list(A.summarize_bridge({"manifestHash": "fixture",
                      "estimandGroups": lambda: [group]}, store, tick=lambda: None))[0]
        self.assertEqual(result["value"], 35.5)
        # Nine independent strata, each with 64 equally weighted model persons.
        self.assertAlmostEqual(result["mcSE"], math.sqrt((64 * 65 / 12) / (9 * 64)))
        self.assertEqual(result["countsByArm"]["C2"], {"assigned": 576, "complete": 576, "eligible": 576})
        group["route"] = "standardized_variant_diagnostic"
        result = list(A.summarize_bridge({"manifestHash": "fixture",
                      "estimandGroups": lambda: [group]}, store, tick=lambda: None))[0]
        self.assertIsNone(result["mcSE"])

    def test_partial_and_smoke_never_claim_full(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("smoke", "full"):
                with patch.object(C, "load_prepared", return_value=(C.plan(mode), {"context": {}})):
                    target = Path(directory) / (mode + ".json")
                    with patch.object(C, "read", return_value={"fixture": "prepared-marker"}) as prepared_read:
                        result = C.aggregate("unused", "unused", directory, target)
                    prepared_read.assert_called_once_with(Path("unused") / "prepared.json")
                    self.assertEqual(result["preparation"], {"fixture": "prepared-marker"})
                    self.assertFalse(result["fullStudyComplete"])
                    self.assertEqual(result["records"], [])
                    self.assertTrue(result["missingOrdinals"])

    def test_completion_identity_and_mutations_rejected(self):
        task = C.plan("smoke")["tasks"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write(path / "complete.json", {"schema": "cost-aware-campaign-shard-v1",
                  "task": task, "contextDigest": C.digest({}), "files": {}})
            with self.assertRaisesRegex(ValueError, "artifact roster"):
                C.verify_shard(path, task, {})
            with self.assertRaisesRegex(ValueError, "identity"):
                C.verify_shard(path, {**task, "ordinal": 7}, {})
        for size in (0, 65, True):
            with self.assertRaises(ValueError):
                C.plan("full", size)


class CampaignResumeTests(unittest.TestCase):
    """Synthetic transport only: no native generation, policy or numerical replay."""

    def prepared_fixture(self, directory):
        base = Path(directory)
        prepared, output = base / "prepared", base / "output"
        (prepared / "roots").mkdir(parents=True)
        roster = C.plan("smoke")
        task = roster["tasks"][0]
        selected = C.task_selection(task)
        roots = [{**r, "fixture": "original-root"} for r in C.source_selection(selected)]
        context = {"design": {"fixture": "design"}, "settings": {"fixture": "settings"}}
        source = {"states": copy.deepcopy(roots), "rootSelection": C.source_selection(selected),
                  "sessions": [{"conditionId": task["conditionId"],
                                "scheduleVariant": task["scheduleVariant"]}]}
        inputs = {"schema": "finite-cost-aware-inputs-v1", "scope": "selected_working_trials",
                  "context": context, "selection": selected, "roots": roots,
                  "publicIdentity": "not_a_historical_manifest"}
        identity = {"files": {"fixture.py": "original-source"},
                    "runtime": {"numpyBinary": "original-binary", "cpuFeatures": {"fixture": True}},
                    "node": "fixture-node-version"}
        write(prepared / "plan.json", roster)
        write(prepared / "roots/inputs.json", inputs)
        write(prepared / "roots/source_states.json", source)
        marker = {"schema": "cost-aware-campaign-prepared-v1", "planDigest": C.digest(roster),
                  "inputsDigest": C.digest(inputs),
                  "sourceDigest": C.file_digest(prepared / "roots/source_states.json"),
                  "scientificIdentity": identity}
        write(prepared / "prepared.json", marker)
        report = {"verification": "checked", "allSelectedNumericallyCompleted": True,
                  "inputsDigest": C.digest(inputs), "selection": selected,
                  "trials": [{"selection": s} for s in selected]}
        return SimpleNamespace(prepared=prepared, output=output, roster=roster, task=task,
                               selected=selected, roots=roots, context=context, source=source,
                               inputs=inputs, identity=identity, marker=marker, report=report,
                               args=("fixture-config", prepared, output, task, "fixture-node"))

    def producer_fixture(self, fixture, *, reader=False, marker=False):
        path = C.shard_path(fixture.output, fixture.task["ordinal"])
        for name in ("inputs", "records"):
            (path / name).mkdir(parents=True)
        write(path / "binding.json", {"task": fixture.task, "preparation": fixture.marker})
        write(path / "selection.json", fixture.selected)
        write(path / "inputs/inputs.json", fixture.inputs)
        write(path / "inputs/source_states.json", fixture.source)
        reference = {"schema": "finite-cost-aware-records-v1", "scope": "selected_working_trials",
                     "inputsDigest": C.digest(fixture.inputs), "selection": fixture.selected,
                     "roots": fixture.roots, "reference": {"fixture": "reference-rate"},
                     "trials": [], "historicalAcceptanceClaim": False}
        # Deliberately not numerical histories: only transport is under test.
        for selection in fixture.selected:
            history = {"status": "completed", "fixtureMain": selection["mainIndex"]}
            rows = [{"kind": "trial", "selection": selection, "inputsDigest": C.digest(fixture.inputs)},
                    {"kind": "history", "value": history}]
            raw = [C.compact(row).encode("utf-8") for row in rows]
            name = C.trial_name(selection) + ".ndjson"
            (path / "records" / name).write_bytes(b"".join(row + b"\n" for row in raw))
            reference["trials"].append({"selection": selection, "file": name, "history": history,
                                        "rowSha256": [hashlib.sha256(row).hexdigest() for row in raw]})
        write(path / "records/reference.json", reference)
        if reader:
            (path / "checked").mkdir()
            write(path / "checked/records_binding.json", self.checker_pins(path))
            write(path / "checked/verification.json", fixture.report)
        if marker:
            self.seal_shard(path, fixture)
        return path, reference

    def checker_pins(self, path):
        return {p.relative_to(path).as_posix(): C.file_digest(p)
                for folder in (path / "records", path / "inputs")
                for p in folder.iterdir() if p.is_file()}

    def seal_shard(self, path, fixture):
        files = sorted(p.relative_to(path).as_posix() for p in path.rglob("*")
                       if p.is_file() and p.name != "complete.json")
        write(path / "complete.json", {"schema": "cost-aware-campaign-shard-v1",
              "task": fixture.task, "contextDigest": C.digest(fixture.context),
              "files": {name: C.file_digest(path / name) for name in files}})

    @contextmanager
    def campaign_adapters(self, fixture, *, identity=None):
        with patch.object(C, "load_context", return_value=fixture.context), \
             patch.object(C, "scientific_identity", return_value=identity or fixture.identity), \
             patch.object(C, "selected_roots", return_value=fixture.roots), \
             patch.object(C, "produce", side_effect=AssertionError("Producer must not rerun")) as producer, \
             patch.object(C, "check", side_effect=AssertionError("Check wrapper must not rerun")) as wrapper, \
             patch("src.design_validation.cost_aware_validation.validate_computation",
                   side_effect=AssertionError("Numerical reader must not rerun")) as reader:
            yield SimpleNamespace(producer=producer, reader=reader, wrapper=wrapper)

    @contextmanager
    def producer_adapters(self, fixture):
        with patch.object(R, "load_context", return_value=fixture.context), \
             patch.object(R, "selected_roots", return_value=fixture.roots), \
             patch("src.design_validation.budget_horizon.budget_templates", return_value="fixture-templates"), \
             patch("src.design_validation.terminal_time.reference_rate",
                   return_value={"fixture": "reference-rate"}), \
             patch("src.design_validation.cost_aware_policy.run_model_trial",
                   side_effect=AssertionError("Policy must not rerun")) as policy:
            yield policy

    def test_prepared_source_runtime_and_node_changes_refuse_resume(self):
        for field in ("source", "runtime", "cpu", "node"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                f = self.prepared_fixture(directory)
                self.producer_fixture(f, reader=True, marker=True)
                with self.campaign_adapters(f):
                    self.assertEqual(C.load_prepared("fixture-config", f.prepared, "fixture-node"),
                                     (f.roster, f.inputs))
                changed = copy.deepcopy(f.identity)
                if field == "source":
                    changed["files"]["fixture.py"] = "different-source"
                elif field == "runtime":
                    changed["runtime"]["numpyBinary"] = "different-binary"
                elif field == "cpu":
                    changed["runtime"]["cpuFeatures"]["fixture"] = False
                else:
                    changed["node"] = "different-node-version"
                with self.campaign_adapters(f, identity=changed) as calls:
                    with self.assertRaisesRegex(ValueError, "Prepared inputs/source/runtime changed"):
                        C._run_task(f.args)
                    calls.producer.assert_not_called()
                    calls.reader.assert_not_called()

    def test_changed_prepared_inputs_source_and_marker_are_rejected(self):
        for filename in ("roots/inputs.json", "roots/source_states.json", "prepared.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                f = self.prepared_fixture(directory)
                target = f.prepared / filename
                value = read(target)
                if filename == "roots/inputs.json":
                    value["roots"][0]["fixture"] = "different-prepared-root"
                elif filename == "roots/source_states.json":
                    value["states"][0]["fixture"] = "different-source-root"
                else:
                    value["sourceDigest"] = "different-catalog"
                target.write_text(C.compact(value) + "\n")
                with self.campaign_adapters(f) as calls:
                    with self.assertRaisesRegex(ValueError, "Prepared inputs/source/runtime changed"):
                        C._run_task(f.args)
                    calls.producer.assert_not_called()
                    calls.reader.assert_not_called()
                self.assertFalse(C.shard_path(f.output, f.task["ordinal"]).exists())

    def test_completed_shard_binds_prepared_catalog_and_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.prepared_fixture(directory)
            path, reference = self.producer_fixture(f, reader=True, marker=True)
            self.assertEqual(C.verify_shard(path, f.task, f.context, f.inputs, f.marker), reference)
            changed_inputs = copy.deepcopy(f.inputs)
            changed_inputs["roots"][0]["fixture"] = "different-prepared-root"
            with self.assertRaisesRegex(ValueError, "different prepared root catalog"):
                C.verify_shard(path, f.task, f.context, changed_inputs, f.marker)
            changed_marker = copy.deepcopy(f.marker)
            changed_marker["scientificIdentity"]["runtime"]["numpyBinary"] = "other-binary"
            with self.assertRaisesRegex(ValueError, "source/runtime/preparation identity changed"):
                C.verify_shard(path, f.task, f.context, f.inputs, changed_marker)

    def test_partial_shard_with_other_preparation_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.prepared_fixture(directory)
            path, _ = self.producer_fixture(f)
            binding = read(path / "binding.json")
            binding["preparation"]["inputsDigest"] = "other-prepared-catalog"
            (path / "binding.json").write_text(C.compact(binding) + "\n")
            with self.campaign_adapters(f) as calls:
                with self.assertRaisesRegex(ValueError, "Unbound partial shard cannot resume"):
                    C._run_task(f.args)
                calls.producer.assert_not_called()
                calls.reader.assert_not_called()
            self.assertFalse((path / "complete.json").exists())

    def test_changed_partial_selection_and_inputs_are_rejected_without_work(self):
        cases = (("selection.json", "Resumed selection changed"),
                 ("inputs/inputs.json", "Resumed inputs changed"),
                 ("inputs/source_states.json", "Resumed inputs changed"))
        for filename, message in cases:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                f = self.prepared_fixture(directory)
                path, _ = self.producer_fixture(f, reader=True)
                target = path / filename
                value = read(target)
                if filename == "selection.json":
                    value[0]["replicateId"] = 1
                elif filename == "inputs/inputs.json":
                    value["roots"][0]["fixture"] = "different-shard-root"
                else:
                    value["states"][0]["fixture"] = "different-shard-source"
                target.write_text(C.compact(value) + "\n")
                before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
                with self.campaign_adapters(f) as calls:
                    with self.assertRaisesRegex(ValueError, message):
                        C._run_task(f.args)
                    calls.producer.assert_not_called()
                    calls.reader.assert_not_called()
                    calls.wrapper.assert_not_called()
                self.assertEqual(before, {p.relative_to(path): p.read_bytes()
                                          for p in path.rglob("*") if p.is_file()})
                self.assertFalse((path / "complete.json").exists())

    def test_changed_completed_artifacts_are_rejected_without_work(self):
        names = ("binding.json", "selection.json", "inputs/inputs.json",
                 "inputs/source_states.json", "records/reference.json",
                 "checked/verification.json", "checked/records_binding.json", "trial")
        for filename in names:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                f = self.prepared_fixture(directory)
                path, reference = self.producer_fixture(f, reader=True, marker=True)
                if filename == "trial":
                    filename = "records/" + reference["trials"][0]["file"]
                target = path / filename
                target.write_bytes(target.read_bytes() + b" ")
                before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
                with self.campaign_adapters(f) as calls:
                    with self.assertRaisesRegex(ValueError, "Changed completed file:") as raised:
                        C._run_task(f.args)
                    self.assertEqual(str(raised.exception), "Changed completed file: " + filename)
                    calls.producer.assert_not_called()
                    calls.reader.assert_not_called()
                    calls.wrapper.assert_not_called()
                self.assertEqual(before, {p.relative_to(path): p.read_bytes()
                                          for p in path.rglob("*") if p.is_file()})

    def test_completed_producer_missing_reader_reuses_without_producer_call(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.prepared_fixture(directory)
            path, reference = self.producer_fixture(f)
            before = {p.name: p.read_bytes() for p in (path / "records").iterdir()}

            def checked_with_preexisting_binding(*args, **kwargs):
                self.assertEqual(read(path / "checked/records_binding.json"), self.checker_pins(path))
                self.assertFalse((path / "checked/verification.json").exists())
                return f.report

            with self.campaign_adapters(f) as calls:
                calls.reader.side_effect = checked_with_preexisting_binding
                result = C._run_task(f.args)
                calls.producer.assert_not_called()
                calls.wrapper.assert_not_called()
                calls.reader.assert_called_once_with(f.inputs, path / "records",
                                                      expected_selection=f.selected, node="fixture-node")
            self.assertEqual(result["status"], "computed_and_checked")
            self.assertEqual(before, {p.name: p.read_bytes() for p in (path / "records").iterdir()})
            self.assertEqual(read(path / "checked/verification.json"), f.report)
            self.assertEqual(C.verify_shard(path, f.task, f.context, f.inputs, f.marker), reference)

    def test_completed_reader_missing_marker_reuses_without_new_check(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.prepared_fixture(directory)
            path, reference = self.producer_fixture(f, reader=True)
            before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
            self.assertFalse((path / "complete.json").exists())
            with self.campaign_adapters(f) as calls:
                result = C._run_task(f.args)
                calls.producer.assert_not_called()
                calls.reader.assert_not_called()
                calls.wrapper.assert_not_called()
            self.assertEqual(result["status"], "computed_and_checked")
            self.assertEqual(before, {p: (path / p).read_bytes() for p in before})
            self.assertEqual(C.verify_shard(path, f.task, f.context, f.inputs, f.marker), reference)
            with self.campaign_adapters(f) as calls:
                self.assertEqual(C._run_task(f.args), {"ordinal": f.task["ordinal"], "status": "reused_verified"})
                calls.producer.assert_not_called()
                calls.reader.assert_not_called()

    def test_changed_checked_records_cannot_gain_a_new_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.prepared_fixture(directory)
            path, reference = self.producer_fixture(f, reader=True)
            binding = (path / "checked/records_binding.json").read_bytes()
            trial = path / "records" / reference["trials"][0]["file"]
            trial.write_bytes(trial.read_bytes() + b'{"kind":"forged"}\n')
            with self.campaign_adapters(f) as calls:
                with self.assertRaisesRegex(ValueError, "Records changed after independent checking started"):
                    C._run_task(f.args)
                calls.producer.assert_not_called()
                calls.reader.assert_not_called()
            self.assertFalse((path / "complete.json").exists())
            self.assertEqual((path / "checked/records_binding.json").read_bytes(), binding)
            # Rehashing completion cannot retroactively bind the old reader.
            self.seal_shard(path, f)
            with self.assertRaisesRegex(ValueError, "Checker is not bound to current records"):
                C.verify_shard(path, f.task, f.context, f.inputs, f.marker)

    def test_older_reader_without_records_binding_is_not_rebound_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.prepared_fixture(directory)
            path, _ = self.producer_fixture(f, reader=True)
            (path / "checked/records_binding.json").unlink()
            with self.campaign_adapters(f) as calls:
                with self.assertRaisesRegex(ValueError, "Unbound older checker result"):
                    C._run_task(f.args)
                calls.producer.assert_not_called()
                calls.reader.assert_not_called()
            self.assertFalse((path / "complete.json").exists())
            self.assertFalse((path / "checked/records_binding.json").exists())
            self.seal_shard(path, f)
            with self.assertRaisesRegex(ValueError, "Exact shard artifact roster required"):
                C.verify_shard(path, f.task, f.context, f.inputs, f.marker)

    def test_complete_ndjson_without_reference_resumes_without_policy_call(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.prepared_fixture(directory)
            path, reference = self.producer_fixture(f)
            (path / "records/reference.json").unlink()
            before = {p.name: p.read_bytes() for p in (path / "records").iterdir()}
            with self.producer_adapters(f) as policy:
                actual = R.produce("fixture-config", path / "selection.json", path / "inputs",
                                   path / "records", resume=True)
                policy.assert_not_called()
                with self.assertRaisesRegex(ValueError, "Completed producer must be reused"):
                    R.produce("fixture-config", path / "selection.json", path / "inputs",
                              path / "records", resume=True)
                policy.assert_not_called()
            self.assertEqual(actual, reference)
            self.assertEqual(read(path / "records/reference.json"), reference)
            self.assertEqual(before, {name: (path / "records" / name).read_bytes() for name in before})

    def test_invalid_resumed_ndjson_is_preserved_and_never_calls_policy(self):
        cases = (("inputs", "Changed resumed trial inputs"),
                 ("selection", "Changed resumed trial inputs"),
                 ("newline", "Partial trial"), ("unfinished", "Incomplete trial"),
                 ("failed", "Incomplete trial"), ("trailing", "Extra records"))
        for kind, message in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                f = self.prepared_fixture(directory)
                path, reference = self.producer_fixture(f)
                (path / "records/reference.json").unlink()
                trial = path / "records" / reference["trials"][0]["file"]
                rows = [C.parse(line) for line in trial.read_text().splitlines()]
                if kind == "inputs":
                    rows[0]["inputsDigest"] = "other-inputs"
                elif kind == "selection":
                    rows[0]["selection"]["replicateId"] = 1
                elif kind == "unfinished":
                    rows.pop()
                elif kind == "failed":
                    rows[-1]["value"]["status"] = "numerical_failure"
                elif kind == "trailing":
                    rows.append({"kind": "extra"})
                raw = b"".join(C.compact(row).encode("utf-8") + b"\n" for row in rows)
                if kind == "newline":
                    raw = raw[:-1]
                trial.write_bytes(raw)
                with self.producer_adapters(f) as policy:
                    with self.assertRaisesRegex(ValueError, message):
                        R.produce("fixture-config", path / "selection.json", path / "inputs",
                                  path / "records", resume=True)
                    policy.assert_not_called()
                self.assertEqual(trial.read_bytes(), raw)
                self.assertFalse((path / "records/reference.json").exists())


if __name__ == "__main__":
    unittest.main()
