"""Independent saved-input rejection fixtures; no fits, simulations or inference."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.tutoring_statistics import independent_saved_coverage as reader

SPEC = importlib.util.spec_from_file_location("independent_coverage_fixture_data", Path(__file__).with_name("saved_coverage_fixtures.py"))
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


class SavedCoverageReadbackTests(unittest.TestCase):
    def test_independent_moment_identity_and_saved_moment_rejections(self):
        result = reader.aggregate_values(fixture.moment_cell())
        self.assertAlmostEqual(result["candidateSE"]["RMS"]**2, 12.998)
        self.assertAlmostEqual(result["biasMonteCarloSE"]**2, .002)
        self.assertLess(result["biasNormalMCInterval95"][0], 0.)
        for field, value in (("SD", -1.), ("mean", float("nan")), ("mean", True)):
            cell = fixture.moment_cell()
            cell["contrastDiagnostics"][reader.KEY]["candidateSE"][field] = value
            with self.assertRaises(ValueError):
                reader.aggregate_values(cell)

    def test_copied_types_and_values_remain_exact(self):
        original = dict(count=1, available=True, moment=.3)
        reader.copied(deepcopy(original), original)
        for key, value in (("count", 1.), ("available", 1), ("moment", math.nextafter(.3, 1.))):
            changed = dict(original)
            changed[key] = value
            with self.assertRaises(ValueError):
                reader.copied(changed, original)
        with self.assertRaises(ValueError):
            reader.compare(dict(RMS=3.6), dict(RMS=3.5))

    def test_strict_json_and_bound_bytes_reject_corruption(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                reader.strict_json(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = reader.Readback(fixture.reader_request(root))
            pin = fixture.save(root/"input.json", {})
            with self.assertRaises(ValueError):
                ctx.bound(dict(pin, sha256="0"*64))
            ctx = reader.Readback(fixture.reader_request(root))
            ctx.bound(pin)
            fixture.save(root/"input.json", dict(changed=True))
            with self.assertRaises(ValueError):
                ctx.close()

    def test_unknown_summary_rejected_before_local_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = reader.Readback(fixture.reader_request(root))
            ctx.request.update(aggregate=dict(sha256=reader.AGGREGATE_SHA), independent=dict(sha256=reader.INDEPENDENT_SHA),
                               summaryIndex=dict(path="index.json"))
            aggregate = dict(schema="h10-power-design-aggregate-v1", namespace="h10-power-design-v1-20261020",
                method=reader.METHOD, complete=True, expectedTotal=80000, completedTotal=80000, summaries=[])
            proof = dict(schema="h10-power-design-independent-aggregate-v1", method=reader.METHOD,
                passed=True, complete=True, sourceClosure=True, expectedTotal=80000, completedTotal=80000,
                aggregate=dict(sha256=reader.AGGREGATE_SHA), manifest=dict(sha256=reader.MANIFEST_SHA))
            with mock.patch.object(ctx, "bound", side_effect=[proof, [dict(originalPath="unknown", localPath="never-read")]]):
                with self.assertRaisesRegex(ValueError, "Unknown/duplicate relocated summary"):
                    reader.read_historical(ctx, aggregate)

    def test_incomplete_duplicate_and_wrong_scope_rejected(self):
        with self.assertRaises(ValueError):
            reader.reconcile(fixture.scalar_rows()[:-1], fixture.moment_cell())
        rows = fixture.scalar_rows()
        rows[-1]["repID"] = 1
        with self.assertRaises(ValueError):
            reader.reconcile(rows, fixture.moment_cell())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = reader.Readback(fixture.reader_request(root))
            manifest = fixture.public_manifest(root)
            event = fixture.public_member(root/"unit", manifest)
            event["event"]["ordinal"] = 33
            fixture.refresh_event(event)
            with self.assertRaises(ValueError):
                reader.read_public_member(ctx, event, manifest)

    def test_actual_cli_rejects_wrong_pin_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = fixture.save(root/"aggregate.json", {})
            request = fixture.save(root/"request.json", dict(schema="saved-coverage-input-v1", mode="public_power",
                aggregate=dict(aggregate, sha256="0"*64), relocations=[], events=[]))
            produced = root/"producer.json"
            fixture.save(produced, {})
            output = root/"output.json"
            output.write_text("preserve-this-output")
            command = [sys.executable, str(fixture.ROOT/"scripts/read_saved_coverage.py"),
                       "--request", request["path"], "--producer-result", str(produced), "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Changed bound input", result.stderr)
            self.assertEqual(output.read_text(), "preserve-this-output")

    def test_independent_public_reread_rejects_producer_scalar_and_missing_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = reader.Readback(fixture.reader_request(root))
            manifest = fixture.public_manifest(root)
            event = fixture.public_member(root/"unit", manifest)
            reader.read_manifest(ctx, manifest)
            independently_read = reader.read_public_member(ctx, event, manifest)
            supplied = deepcopy(independently_read)
            supplied["candidate"]["lower"] = math.nextafter(supplied["candidate"]["lower"], 0.)
            with self.assertRaises(ValueError):
                reader.copied(supplied, independently_read)
            path = Path(event["artifacts"][4]["path"])
            candidate = json.loads(path.read_text())
            del candidate["rawCandidate"]["references"][reader.KEY]["CI"]
            event["artifacts"][4] = fixture.save(path, candidate)
            with self.assertRaises((ValueError, KeyError)):
                reader.read_public_member(reader.Readback(fixture.reader_request(root)), event, manifest)

    def test_transport_omission_and_semantic_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture.public_manifest(root)
            event = fixture.public_member(root/"unit", manifest)
            removed = deepcopy(event)
            removed["artifacts"].pop()
            with self.assertRaises(ValueError):
                reader.read_public_member(reader.Readback(fixture.reader_request(root)), removed, manifest)
            path = Path(event["artifacts"][-1]["path"])
            transport = json.loads(path.read_text())
            transport["fits"][event["event"]["fitKey"]]["exactResponseInR"] = False
            event["artifacts"][-1] = fixture.save(path, transport)
            with self.assertRaises(ValueError):
                reader.read_public_member(reader.Readback(fixture.reader_request(root)), event, manifest)

    def test_independent_partition_all_covered_and_single_miss(self):
        rows = fixture.scalar_rows()
        result = reader.record_values(rows)
        self.assertEqual(result["methods"]["candidate"]["covered"], 2000)
        self.assertIsNone(result["retentionByCoverage"]["False"]["C1"]["sampleSD"])
        rows[0]["candidate"].update(covered=False, belowTruth=True, lower=-2., upper=-1.)
        result = reader.record_values(rows)
        self.assertEqual(result["methods"]["candidate"]["belowTruth"], 1)
        self.assertEqual(result["methods"]["candidate"]["covered"], 1999)
        self.assertIsNone(result["retentionByCoverage"]["False"]["C1"]["sampleSD"])


if __name__ == "__main__":
    unittest.main()
