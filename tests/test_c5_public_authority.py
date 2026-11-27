"""Tests of trust boundaries; fixtures are not scientific authority."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from design_validation import c5_public_authority as authority


class AuthorityTests(unittest.TestCase):
    def test_adopted_inputs_use_fixed_pins_and_do_not_authorize_execution(self):
        fixed, prefix = {"roots": []}, {"payload": {}}
        with mock.patch.object(authority.inputs, "load_fixed_inputs", return_value=fixed) as load_fixed, \
                mock.patch.object(authority.inputs, "load_historical_prefix", return_value=prefix) as load_prefix:
            result = authority.load_adopted_inputs(initial_lookup="lookup", historical_prefix="prefix")
        load_fixed.assert_called_once_with("lookup", "68b2fd36ef99aefcadb78e1831a9167134b62731787e4588d149668699c6d1a0")
        load_prefix.assert_called_once_with(
            "prefix", "926d3b69ba90d107a2ad1c2a8150f68b7e8e3c7cb06490342e1d38523ccd0f73",
            repository_root=authority._ROOT, lookup=fixed)
        self.assertIs(result["initialLookup"], fixed)
        self.assertIs(result["historicalPrefix"], prefix)
        self.assertEqual(result["highPricePolicyReplay"], "not-established")
        self.assertEqual(result["runtimeEquivalence"], "not-established")
        self.assertEqual(result["historicalDispatchReconstruction"], "not-established")
        self.assertFalse(result["samplingAuthorized"])
        self.assertFalse(result["finalReductionAllowed"])

    def test_changed_lookup_refused_before_prefix_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lookup.json"
            path.write_bytes(b'{}')
            with mock.patch.object(authority.inputs, "load_historical_prefix") as load_prefix:
                with self.assertRaises(ValueError):
                    authority.load_adopted_inputs(initial_lookup=path, historical_prefix="unused")
                load_prefix.assert_not_called()

    def test_changed_prefix_and_loader_refusals_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefix.json"
            path.write_bytes(b'{}')
            with mock.patch.object(authority.inputs, "load_fixed_inputs", return_value={"roots": []}):
                with self.assertRaises(ValueError):
                    authority.load_adopted_inputs(initial_lookup="lookup", historical_prefix=path)
            with mock.patch.object(authority.inputs, "load_fixed_inputs", return_value={"roots": []}), \
                    mock.patch.object(authority.inputs, "load_historical_prefix", side_effect=ValueError("mapping")):
                with self.assertRaisesRegex(ValueError, "mapping"):
                    authority.load_adopted_inputs(initial_lookup="lookup", historical_prefix=path)

    def test_caller_cannot_override_pins_or_checkout_boundary(self):
        for keyword in ("expected_sha256", "repository_root"):
            with self.subTest(keyword=keyword), self.assertRaises(TypeError):
                authority.load_adopted_inputs(initial_lookup="lookup", historical_prefix="prefix",
                                              **{keyword: "attacker"})

    def test_absent_repository_manifest_refuses_caller_claims(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(authority, "_ROOT", Path(tmp)):
            with self.assertRaises(authority.AuthorityUnavailable):
                authority.inspect_scientific_authority(
                    initial_lookup="unused", latent_config="unused", runtime_profile="unused",
                    historical_rep1="unused", source_chain="unused", preparation_evidence="unused")

    def test_incomplete_or_invalid_manifest_refused(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(authority, "_ROOT", Path(tmp)):
            path = Path(tmp) / authority.MANIFEST_PATH
            path.parent.mkdir()
            path.write_text(json.dumps({"schema": authority.MANIFEST_SCHEMA,
                                        "target": authority.inputs.TARGET,
                                        "bindings": {"initial_lookup": "0" * 64}}))
            with self.assertRaises(authority.AuthorityUnavailable):
                authority._manifest()

    def test_changed_bytes_and_duplicate_members_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_bytes(b'{"x":1}')
            with self.assertRaises(ValueError):
                authority._bound_json(path, "0" * 64)
            with self.assertRaises(ValueError):
                authority._json(b'{"x":1,"x":2}')

    def test_path_free_profile_filter(self):
        self.assertTrue(authority._path_free({"executable": "file-0001"}))
        self.assertFalse(authority._path_free({"executable": "/private/python"}))
        self.assertFalse(authority._path_free({"path": "C:\\private"}))

    def test_geometry_delegates_to_independent_reference(self):
        expected = {"scope": "supplied_geometry_only_not_input_authority_or_final"}
        with mock.patch.object(authority.c5_final_reference, "check_supplied_geometry",
                               return_value=expected) as check:
            self.assertEqual(authority.check_supplied_geometry([], {}, {}), expected)
            check.assert_called_once_with([], {}, {})

    def test_partial_check_cannot_authorize_historical_final(self):
        with mock.patch.object(authority, "inspect_scientific_authority",
                               return_value={"finalReductionAllowed": True}):
            with self.assertRaises(authority.AuthorityUnavailable):
                authority.require_scientific_authority()


if __name__ == "__main__":
    unittest.main()
