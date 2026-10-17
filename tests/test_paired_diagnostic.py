"""Test purpose: fixed-batch attribution, independent arithmetic and strict packets.

All integration inputs here are deterministic fixtures, not Gamma simulations.
The real selected-batch CLI chain is a separate reproduction check.
"""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np
from scipy.special import stdtr, stdtrit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tutoring_analysis import paired_diagnostic as producer
from src.tutoring_analysis import paired_readback as reader
from src.tutoring_analysis import paired_recipe as recipe

CONFIG = ROOT / "configs/tutoring_calibration_v01.json"
SCRIPT = ROOT / "scripts/reproduce_paired_diagnostic.py"


def deterministic_batch():
    """Exercise the full transport shape without consuming any random draws."""
    x = np.tile(np.linspace(-.2, .2, 10), (20, 1))
    slopes = np.linspace(-.3, .2, 20)
    y = np.tile((.5 + slopes[:, None] * x)[None, :, :], (250, 1, 1))
    centered = x - np.sum(x, axis=-1, keepdims=True) / 10
    weights = centered / np.sum(centered ** 2, axis=-1, keepdims=True)
    values = np.sum(weights[None, :, :] * y, axis=-1)
    raw = dict(y=y, x=x, records=np.ones_like(y, dtype=bool),
               personKeep=np.ones((250, 20), dtype=bool),
               endpointMasks={k: np.ones_like(y, dtype=bool) for k in ("q", "R", "L", "initial")},
               expectedY=np.full((20, 10), .5),
               scale=np.resize([.8, 1., 1.2], 10), gap=np.resize([-12., -12., 0., 0., 12., 12.], 10))
    initial = dict(values=values, available=np.ones((250, 20), dtype=bool),
                   observations=np.full((250, 20), 10), truth=np.zeros((250, 20)),
                   selectedTruth=np.zeros((250, 20)))
    return {"raw": {c: raw for c in ("C1", "C2", "C3", "C4", "C5")},
            "populations": {"primary": {"C4": {"initial": initial}}}}


class PairedDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.packet = Path(cls.temp.name) / "producer"
        cls.batch = deterministic_batch()
        with patch.object(producer.cal, "generate_batch", return_value=cls.batch) as generate:
            cls.summary = producer.produce(CONFIG, cls.packet, batches=(39,))
            cls.generate_calls = generate.call_args_list

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def read_fixture(self, changed=None, output=None, require_full=False):
        original_read = recipe.read_bytes

        def read(path, *args, **kwargs):
            raw = original_read(path, *args, **kwargs)
            if changed is not None and Path(path).name == changed[0]:
                return changed[1](raw)
            return raw

        with patch.object(reader.cal, "generate_batch", return_value=self.batch), \
                patch.object(reader, "read_bytes", side_effect=read):
            return reader.read_packet(CONFIG, self.packet, output, require_full)

    def test_selected_producer_nonempty_fixed_shape_and_identity(self):
        self.assertEqual(len(self.generate_calls), 1)
        self.assertEqual(self.generate_calls[0].args[1:], (39, 250))
        self.assertEqual(self.generate_calls[0].args[0]["seed"], 2026100627)
        self.assertEqual(self.generate_calls[0].args[0]["people"], {f"C{i}": 20 for i in range(1, 6)})
        self.assertEqual(self.summary["cases"], 250)
        self.assertTrue(self.summary["selectedComplete"])
        self.assertFalse(self.summary["fixed500Complete"])
        self.assertEqual(len(self.summary["batchSummaries"]), 1)
        cases = recipe.read_json(self.packet / "cases_39.json")
        self.assertEqual(len(cases), 250)
        self.assertEqual([c["replicationId"] for c in cases], list(range(9750, 10000)))
        self.assertEqual(reader.validate_summary_header(self.summary), (39,))
        with self.assertRaises(ValueError):
            self.read_fixture(require_full=True)

    def test_independent_reader_rebuilds_all_selected_science(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "reader"
            result = self.read_fixture(output=output)
            self.assertEqual(result["cases"], 250)
            self.assertEqual(result["independentStudentChecks"], 1000)
            self.assertTrue(result["selectedComplete"])
            self.assertFalse(result["fixed500Complete"])
            self.assertFalse(result["typeICalibrationAccepted"])
            self.assertFalse(result["historicalFailureResolved"])
            self.assertEqual(len(recipe.read_json(output / "independent_cases.json")), 250)
            for name in reader.DECISION_NAMES:
                self.assertEqual(result["aggregateDifferences"][name],
                                 self.summary["aggregateDifferences"]["decisionCounts"][name])

    def test_frozen_recipe_and_selection_reject_drift(self):
        original = recipe.read_json(CONFIG)
        self.assertEqual(recipe.fixed_cell(original)["dose"], 10)
        for name, value in (("seed", 1), ("dose", 9), ("correlation", .1),
                            ("trialRetention", .9), ("meanTrialSD", {"q": .1, "R": 3, "L": .1})):
            changed = copy.deepcopy(original)
            cell = next(c for c in changed["cells"] if c["id"] == "gamma_skew_null")
            cell[name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                recipe.fixed_cell(changed)
        for batches in ((), (0, 0), (39, 0), (1,), (False,), "0"):
            with self.subTest(batches=batches), self.assertRaises(ValueError):
                recipe.selected_batches(batches)

    def test_full_scope_and_original_schema_are_explicit(self):
        full = copy.deepcopy(self.summary)
        full.update(scope=recipe.scope((0, 39)), cases=500, fixed500Complete=True)
        self.assertEqual(reader.validate_summary_header(full, require_full=True), (0, 39))
        original = copy.deepcopy(full)
        original["schema"] = "historical-gamma-diagnostic-v01"
        for key in ("selectedComplete", "fixed500Complete", "referenceRuntime"):
            original.pop(key)
        self.assertEqual(reader.validate_summary_header(original, require_full=True), (0, 39))
        for key, value in (("fixed500Complete", True), ("selectedComplete", 1),
                            ("cases", 500), ("scope", recipe.scope((0, 39))),
                            ("schema", "historical-gamma-diagnostic-v01")):
            changed = copy.deepcopy(self.summary)
            changed[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                reader.validate_summary_header(changed)

    def test_missing_duplicate_and_bad_case_outputs_rejected(self):
        def change_cases(raw, mutation):
            cases = reader.strict_json(raw)
            mutation(cases)
            return json.dumps(cases).encode()

        mutations = {
            "missing": lambda c: c.pop(),
            "duplicate": lambda c: c.__setitem__(1, copy.deepcopy(c[0])),
            "wrong_id": lambda c: c[0].__setitem__("replicationId", 0),
            "duplicate_person": lambda c: c[0]["peopleIds"].__setitem__(1, "C4:00"),
            "changed_slope": lambda c: c[0]["vectorSlopes"].__setitem__(0, 20.),
            "changed_p": lambda c: c[0]["currentSameInput"].__setitem__("p", .99),
            "wrong_tail": lambda c: c[0]["currentProjection"].__setitem__("tail", "greater"),
            "extra_output": lambda c: c[0]["legacy"].__setitem__("unvalidated", 1),
            "wrong_boolean": lambda c: c[0]["decisions"].__setitem__("legacyLE", 1),
        }
        for name, mutation in mutations.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                self.read_fixture(("cases_39.json", lambda raw: change_cases(raw, mutation)))

    def test_changed_summary_and_plan_rejected(self):
        def changed_json(raw, mutate):
            obj = reader.strict_json(raw)
            mutate(obj)
            return json.dumps(obj).encode()

        mutations = [
            ("summary.json", lambda s: s["aggregateDifferences"]["decisionCounts"].__setitem__("currentLT", 999)),
            ("summary.json", lambda s: s["batchSummaries"][0].__setitem__("batch", 0)),
            ("summary.json", lambda s: s.__setitem__("typeIRepairClaim", True)),
            ("tests.json", lambda s: s.__setitem__("passed", 0)),
            ("plan.json", lambda s: s["weights"][0].__setitem__(0, 20.)),
        ]
        for name, mutate in mutations:
            with self.subTest(file=name), self.assertRaises(ValueError):
                self.read_fixture((name, lambda raw: changed_json(raw, mutate)))

    def test_changed_raw_data_and_shape_rejected(self):
        arrays = reader.transport_arrays(recipe.read_bytes(self.packet / "batch_39.npz"), np)
        for kind in ("value", "shape", "dtype", "nonfinite", "missing"):
            changed = {k: v.copy() for k, v in arrays.items()}
            if kind == "value":
                changed["y"][0, 0, 0] += 1.
            elif kind == "shape":
                changed["y"] = changed["y"][:1]
            elif kind == "dtype":
                changed["records"] = changed["records"].astype(np.int8)
            elif kind == "nonfinite":
                changed["y"][0, 0, 0] = float("nan")
            else:
                del changed["x"]
            buffer = io.BytesIO()
            np.savez(buffer, **changed)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.read_fixture(("batch_39.npz", lambda raw: buffer.getvalue()))

    def test_strict_json_npz_and_numeric_types(self):
        for parse in (reader.strict_json, recipe.strict_json):
            for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', "{"):
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    parse(raw)
        raw = recipe.read_bytes(self.packet / "batch_39.npz")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [(n, archive.read(n)) for n in archive.namelist()]
        duplicate = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(duplicate, "w") as archive:
                for name, data in members + [members[0]]:
                    archive.writestr(name, data)
        with self.assertRaises(ValueError):
            reader.transport_arrays(duplicate.getvalue(), np)
        with self.assertRaises(ValueError):
            reader.Audit().exact({"value": True}, {"value": 1}, "typed field")

    def test_projection_student_and_alpha_boundaries(self):
        x = [float(i) for i in range(10)]
        y = [2. + .3 * v for v in x]
        projection = reader.derive_projection(x, y, [True] * 10, True)
        self.assertAlmostEqual(projection["value"], .3)
        self.assertEqual(projection["residualDf"], 8)
        self.assertEqual(reader.derive_projection([0.] * 10, y, [True] * 10, True)["status"], "unavailable")
        self.assertEqual(reader.student([1.] * 20, "less", stdtr, stdtrit)["status"], "unavailable")
        values = [-2., -1., 0., 1., 2.]
        oracle = reader.student(values, "less", stdtr, stdtrit)
        self.assertEqual(oracle["p"], .5)
        self.assertEqual(oracle["df"], 4.)
        bound = {"status": "available", "p": .05, "CI": [-1., 1.]}
        flags = reader.decisions(bound, bound, bound)
        self.assertTrue(flags["legacyLE"])
        self.assertFalse(flags["currentLT"])
        self.assertFalse(flags["projectionLT"])
        self.assertTrue(all(flags[k] for k in ("legacyBoundary", "currentBoundary", "projectionBoundary")))
        self.assertEqual(producer.negative_tests()["passed"], 8)
        self.assertEqual(reader.self_checks(np, stdtr, stdtrit)["passed"], 11)

    def test_output_refusal_and_missing_packet(self):
        with tempfile.TemporaryDirectory() as folder:
            existing = Path(folder)
            with patch.object(producer.cal, "generate_batch") as generate:
                with self.assertRaises(ValueError):
                    producer.produce(CONFIG, existing, (0,))
                generate.assert_not_called()
            with self.assertRaises(ValueError):
                self.read_fixture(output=existing)
            with self.assertRaises(ValueError):
                self.read_fixture(output=self.packet / "reader")
            with self.assertRaises(ValueError):
                reader.read_packet(CONFIG, existing)
            path = existing / "nonfinite.json"
            with self.assertRaises(ValueError):
                recipe.write_json(path, {"x": float("nan")})
            self.assertFalse(path.exists())
            path.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                recipe.write_json(path, {"x": 1})
            self.assertEqual(path.read_text(), "existing")

    def test_cli_help_and_invalid_selection(self):
        for args in (["--help"], ["produce", "--help"], ["read", "--help"]):
            result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "bad"
            result = subprocess.run([sys.executable, str(SCRIPT), "produce",
                                     "--batches", "0", "0", "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
