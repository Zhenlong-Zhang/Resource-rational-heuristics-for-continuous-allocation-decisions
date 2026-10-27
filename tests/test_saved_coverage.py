"""Fixed coverage algebra, historical refusal and public saved-layout fixtures.

These tests do not simulate data, fit a model, or establish scientific coverage.
"""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.tutoring_statistics import saved_coverage as producer

SPEC = importlib.util.spec_from_file_location("coverage_fixture_data", Path(__file__).with_name("saved_coverage_fixtures.py"))
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


class SavedCoverageTests(unittest.TestCase):
    def test_rms_includes_sample_variance_correction(self):
        result = producer.aggregate_diagnosis(fixture.moment_cell())
        self.assertAlmostEqual(result["candidateSE"]["RMS"]**2, 12.998)
        self.assertNotEqual(result["candidateSE"]["RMS"], 3.)
        self.assertAlmostEqual(result["biasMonteCarloSE"]**2, .002)

    def test_duplicate_cell_and_denominator_rejected(self):
        cell = fixture.moment_cell()
        with self.assertRaises(ValueError):
            producer.target_cell(dict(cases=[cell, deepcopy(cell)]))
        cell["T"] = 1999
        with self.assertRaises(ValueError):
            producer.target_cell(dict(cases=[cell]))

    def test_missing_and_nonfinite_moments_rejected(self):
        for field, value in (("missing", 1), ("mean", float("nan")), ("mean", True)):
            cell = fixture.moment_cell()
            cell["contrastDiagnostics"][producer.KEY]["estimate"][field] = value
            with self.assertRaises(ValueError):
                producer.aggregate_diagnosis(cell)

    def test_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input.json"
            fixture.save(path, {})
            observed = producer.Observations(directory)
            with self.assertRaises(ValueError):
                observed.bound(dict(path=str(path), sha256="0"*64))

    def test_unknown_summary_location_rejected_before_read(self):
        # Only the original proof-admission preconditions are stubbed here;
        # this is a negative index unit test, never a historical positive.
        with tempfile.TemporaryDirectory() as directory:
            observed = producer.Observations(directory)
            aggregate = dict(schema="h10-power-design-aggregate-v1", namespace="h10-power-design-v1-20261020",
                method=producer.METHOD, complete=True, expectedTotal=80000, completedTotal=80000, summaries=[])
            proof = dict(schema="h10-power-design-independent-aggregate-v1", method=producer.METHOD,
                passed=True, complete=True, sourceClosure=True, expectedTotal=80000, completedTotal=80000,
                aggregate=dict(sha256=producer.AGGREGATE_SHA), manifest=dict(sha256=producer.MANIFEST_SHA))
            request = dict(aggregate=dict(sha256=producer.AGGREGATE_SHA), independent=dict(sha256=producer.INDEPENDENT_SHA),
                           summaryIndex=dict(path="index.json"))
            with mock.patch.object(observed, "bound", side_effect=[proof, [dict(originalPath="unknown", localPath="never-read")]]):
                with self.assertRaisesRegex(ValueError, "Unknown/duplicate relocated summary"):
                    producer.historical_rows(request, aggregate, observed)
            self.assertEqual(observed.files, {})

    def test_nonhistorical_counts_and_undefined_subgroups(self):
        rows = fixture.scalar_rows()
        result = producer.record_diagnosis(rows)
        self.assertEqual(result["methods"]["candidate"]["covered"], 2000)
        self.assertEqual(result["commonMisses"], 0)
        self.assertIsNone(result["retentionByCoverage"]["False"]["C1"]["sampleSD"])
        rows[0]["candidate"].update(covered=False, belowTruth=True, lower=-2., upper=-1.)
        result = producer.record_diagnosis(rows)
        self.assertEqual(result["methods"]["candidate"]["covered"], 1999)
        self.assertIsNone(result["retentionByCoverage"]["False"]["C1"]["sampleSD"])

    def test_genuine_public_schema_fixture_and_mandatory_fit_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture.public_manifest(root)
            event = fixture.public_member(root/"unit", manifest)
            observed = producer.Observations(root)
            producer.public_manifest(manifest, observed)
            row = producer.public_member(event, manifest, observed)
            self.assertEqual(row["repID"], 1)
            self.assertEqual(row["baseline"]["lower"], -2.)
            for index in (5, 6, 7, 8):
                changed = deepcopy(event)
                del changed["artifacts"][index]
                fixture.refresh_event(changed)
                with self.assertRaises(ValueError):
                    producer.public_member(changed, manifest, producer.Observations(root))
            changed = deepcopy(event)
            changed["event"]["ordinal"] = 33
            fixture.refresh_event(changed)
            with self.assertRaises(ValueError):
                producer.public_member(changed, manifest, producer.Observations(root))

    def test_changed_raw_ci_and_source_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture.public_manifest(root)
            event = fixture.public_member(root/"unit", manifest)
            path = Path(event["artifacts"][4]["path"])
            candidate = json.loads(path.read_text())
            candidate["rawCandidate"]["references"][producer.KEY]["CI"] = [1., 2.]
            event["artifacts"][4] = fixture.save(path, candidate)
            with self.assertRaises(ValueError):
                producer.public_member(event, manifest, producer.Observations(root))
            # Rehashed outer descriptors do not license a different producer.
            manifest_value = json.loads(Path(manifest["path"]).read_text())
            snapshot = json.loads(Path(manifest_value["sources"]["path"]).read_text())
            replacement = fixture.save(root/"foreign"/"reproduction.py", {"not": "the producer"})
            snapshot["files"] = [replacement if p == manifest_value["driver"] else p for p in snapshot["files"]]
            manifest_value["driver"] = replacement
            manifest_value["sources"] = fixture.save(root/"snapshot_changed.json", snapshot)
            bad = fixture.save(root/"manifest_changed.json", manifest_value)
            with self.assertRaises(ValueError):
                producer.public_manifest(bad, producer.Observations(root))

    def test_saved_request_builder_locator_and_no_regeneration(self):
        # Locator positive is a complete filename roster of fixed placeholders,
        # NOT a positive scientific/public admission. The real builder must
        # subsequently reject their absent manifest/spec/raw chain.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root/"events"
            for rep in range(1, 2001):
                fixture.save(events/f"{40*(rep-1)+32:05d}.json", {"fixture": rep})
            pins = producer.locate_events(events, producer.Observations(root))
            self.assertEqual(len(pins), 2000)
            self.assertTrue(pins[0]["path"].endswith("00032.json"))
            self.assertTrue(pins[-1]["path"].endswith("79992.json"))
            aggregate = root/"aggregate.json"
            fixture.save(aggregate, dict(cases=[fixture.moment_cell()], schema="not-public"))
            output = root/"request.json"
            with self.assertRaises((ValueError, KeyError)):
                producer.prepare("public_power", aggregate, output, events_directory=events)
            self.assertFalse(output.exists())
            (events/"00032.json").unlink()
            with self.assertRaises(ValueError):
                producer.locate_events(events, producer.Observations(root))
            self.assertFalse((events/"00032.json").exists())

    def test_rebound_full_guard_and_metadata_rejected_by_both_consumers(self):
        from src.tutoring_statistics import independent_saved_coverage as reader

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture.public_manifest(root)
            for field, value in (("converged", False), ("singular", True), ("hessianEigenvalues", -1.), ("gradient", None)):
                event = fixture.public_member(root/field, manifest)
                event = fixture.rebound_guard(event, field, value)
                with self.assertRaisesRegex(ValueError, "original full guard"):
                    producer.public_member(event, manifest, producer.Observations(root))
                with self.assertRaisesRegex(ValueError, "original full guard"):
                    reader.read_public_member(reader.Readback(fixture.reader_request(root)), event, manifest)
            event = fixture.public_member(root/"metadata", manifest)
            path = Path(event["artifacts"][4]["path"])
            candidate = json.loads(path.read_text())
            candidate["candidateMetadata"]["requestDescriptor"] = dict(candidate["request"], sha256="0"*64)
            event["artifacts"][4] = fixture.save(path, candidate)
            for consume in (
                lambda: producer.public_member(event, manifest, producer.Observations(root)),
                lambda: reader.read_public_member(reader.Readback(fixture.reader_request(root)), event, manifest),
            ):
                with self.assertRaises(ValueError):
                    consume()

    def test_package_version_serializations_are_exact_not_version_relaxation(self):
        from src.tutoring_statistics import independent_saved_coverage as reader

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture.public_manifest(root)
            event = fixture.public_member(root/"unit", manifest)
            # Extraction emits packageVersion dots; metadata emits DESCRIPTION
            # hyphens. Both genuine spellings are present in this fixture.
            producer.public_member(event, manifest, producer.Observations(root))
            reader.read_public_member(reader.Readback(fixture.reader_request(root)), event, manifest)
            path = Path(event["artifacts"][7]["path"])
            extracted = json.loads(path.read_text())
            extracted["runtime"]["lme4"] = "1.1.35.6"
            event["artifacts"][7] = fixture.save(path, extracted)
            with self.assertRaisesRegex(ValueError, "package version"):
                producer.public_member(event, manifest, producer.Observations(root))
            with self.assertRaisesRegex(ValueError, "package version"):
                reader.read_public_member(reader.Readback(fixture.reader_request(root)), event, manifest)

    def test_complete_artificial_layout_builder_and_independent_readback(self):
        # A deterministic software corpus, not new scientific draws or models.
        # Unlike the single-member seam fixtures, this exercises unmodified
        # exact-2000 admission and both whole entry chains without mocks.
        from src.tutoring_statistics import independent_saved_coverage as reader

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture.public_manifest(root)
            for rep in range(1, 2001):
                commitment = fixture.public_member(root/"units"/str(rep), manifest, rep, (-1.)**rep)
                fixture.save(root/"events"/f"{40*(rep-1)+32:05d}.json", commitment)
            cell = fixture.moment_cell()
            cell["contrastDiagnostics"][producer.KEY] = {
                "estimate": dict(n=2000, missing=0, mean=0., SD=math.sqrt(2000/1999), min=-1., max=1.),
                **{name: dict(n=2000, missing=0, mean=value, SD=0., min=value, max=value)
                   for name, value in (("baselineSE", 1.), ("candidateSE", 1.), ("baselineDf", 10.), ("candidateDf", 10.))}}
            cell["paired"] = [dict(tailMode=mode, denominator=2000, contrasts={producer.KEY:
                dict(coverage={"00": 0, "01": 0, "10": 0, "11": 2000, "unknown": 0})}) for mode in producer.TAILS]
            aggregate = root/"aggregate.json"
            fixture.save(aggregate, dict(schema=fixture.PUBLIC, family="power", B=2000, expectedTotal=80000,
                presentTotal=2000, fullRosterComplete=False, status="partial_entry_check", independentNumericalReadback=True,
                reproductionManifest=manifest, cases=[cell]))
            request, result, proof = root/"request.json", root/"result.json", root/"proof.json"
            producer.prepare("public_power", aggregate, request, events_directory=root/"events")
            producer.diagnose(request, result)
            verified = reader.verify(request, result, proof)
            self.assertTrue(verified["passed"])
            self.assertEqual(verified["records"]["methods"]["candidate"]["covered"], 2000)
            self.assertEqual(verified["mode"], "public_power")
            altered = json.loads(result.read_text())
            altered["targetScalars"][0]["candidate"]["lower"] = math.nextafter(-2., 0.)
            fixture.save(root/"altered.json", altered)
            with self.assertRaisesRegex(ValueError, "Changed copied value"):
                reader.verify(request, root/"altered.json", root/"rejected.json")
            self.assertFalse((root/"rejected.json").exists())
            with self.assertRaises(ValueError):
                producer.prepare("public_power", aggregate, request, events_directory=root/"events")

    def test_relocation_preserves_descriptor_and_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = fixture.save(root/"moved.json", dict(value=1))
            logical = dict(local, path="original/input.json")
            observed = producer.Observations(root, [dict(originalPath=logical["path"], localPath="moved.json")])
            self.assertEqual(observed.bound(logical), dict(value=1))
            self.assertEqual(logical["path"], "original/input.json")
            fixture.save(root/"moved.json", dict(value=2))
            with self.assertRaises(ValueError):
                observed.close()
            with self.assertRaises(ValueError):
                producer.Observations(root, [dict(originalPath="same", localPath="one"),
                                             dict(originalPath="same", localPath="two")])

    def test_exact_native_roles_and_current_entry_source_closure(self):
        from src.tutoring_statistics import independent_saved_coverage as reader

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = producer.Observations(root)
            ctx = reader.Readback(fixture.reader_request(root))
            identities = producer.source_identity(observed)
            self.assertEqual(identities, ctx.sources())
            self.assertIn("scripts/diagnose_saved_coverage.py", identities)
            self.assertIn("scripts/read_saved_coverage.py", identities)
            self.assertIn("src/__init__.py", identities)
            self.assertIn("src/tutoring_statistics/__init__.py", identities)
            manifest = fixture.public_manifest(root)
            value = json.loads(Path(manifest["path"]).read_text())
            value["native"] = [value["native"][0]]*3
            bad = fixture.save(root/"changed_native_manifest.json", value)
            with self.assertRaisesRegex(ValueError, "native"):
                producer.public_manifest(bad, producer.Observations(root))
            with self.assertRaisesRegex(ValueError, "native"):
                reader.read_manifest(reader.Readback(fixture.reader_request(root)), bad)


if __name__ == "__main__":
    unittest.main()
