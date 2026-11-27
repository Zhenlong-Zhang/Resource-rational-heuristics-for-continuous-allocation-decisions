"""Boundary tests use synthetic fixtures unless external fixtures are supplied.

Set C5_REQUIRE_REAL_FIXTURES=1 in the CI entry to require both external
fixture paths instead of skipping their checks.
"""

import hashlib
import json
import os

import tempfile
import unittest
from pathlib import Path

from src.design_validation import c5_public_inputs as inputs


def _write(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _prefix():
    def event(root, logical):
        key = {"streamVersion": "r10-private-need-report-v1",
               "populationId": "r10-working-five-v0-rep0-candidate-not-population-estimate",
               "seed": 20261019, "randomizationGroup": "r14-c5-paired-projection-private-stream-v1",
               "conditionId": root[0], "scheduleVariant": root[1], "replicateId": 1,
               "mainIndex": root[2], "logicalRecipient": logical, "streamKind": "latent", "reportIndex": 0}
        hashed = inputs._digest(key)
        return {"key": key, "keySHA256": hashed, "seedInteger": str(int(hashed[:32], 16)), "seedBits": 128}

    return {"schema": inputs.PREFIX_SCHEMA, "target": inputs.TARGET, "replicateId": 1,
            "rows": [{"rootKey": list(k),
                      "endpoint": {"rootKey": list(k), "status": "complete", "nonTie": True,
                                   "latentDisplayed": [1, 2], "latentLogical": {"A": 1, "B": 2},
                                   "initialBudget": 10, "remainingBudget": 10, "queryCount": 0,
                                   "reason": None, "higherNeedShare": ["1", "2"]},
                      "contribution": {"status": "complete", "reason": None,
                                       "values": {"gap": ["0", "1"], "partial_low": ["0", "1"]}},
                      "usedSeeds": [event(k, "A"), event(k, "B")]} for k in inputs.ROSTER]}



class PublicInputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def _missing_fixture(self, name):
        message = name + " not supplied"
        if os.environ.get("C5_REQUIRE_REAL_FIXTURES") == "1":
            self.fail(message)
        self.skipTest(message)

    def test_prefix_hash_and_roster(self):
        tmp_path = self.tmp_path
        repo = tmp_path / "repo"
        repo.mkdir()
        path = tmp_path / "external.json"
        data = _prefix()
        sha = _write(path, data)
        result = inputs.load_historical_prefix(path, sha, repository_root=repo)
        assert result["payload"] == data
        assert result["historicalAcceptance"] == "not-established-by-this-loader"
        assert result["finalReductionAllowed"] is False
        assert result["rootCount"] == 90
        with self.assertRaisesRegex(ValueError, "final reduction"):
            inputs.require_authenticated_prefix(result)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            inputs.load_historical_prefix(path, "0" * 64, repository_root=repo)
        data["rows"][1]["rootKey"] = data["rows"][0]["rootKey"]
        sha = _write(path, data)
        with self.assertRaises(ValueError):
            inputs.load_historical_prefix(path, sha, repository_root=repo)


    def test_prefix_rejects_embedded_provenance_and_bool_key(self):
        tmp_path = self.tmp_path
        repo = tmp_path / "repo"
        repo.mkdir()
        path = tmp_path / "external.json"
        data = _prefix()
        data["rows"][0]["originalProvenance"] = {"status": "fixture"}
        sha = _write(path, data)
        with self.assertRaises(ValueError):
            inputs.load_historical_prefix(path, sha, repository_root=repo)
        data["rows"][0]["rootKey"][1] = False
        with self.assertRaises(ValueError):
            inputs._key(data["rows"][0]["rootKey"])


    def test_fixed_export_rejects_wrong_source(self):
        tmp_path = self.tmp_path
        path = tmp_path / "wrong.json"
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            inputs.export_fixed_inputs(path)


    def test_personal_time_digest_has_no_newline(self):
        value = {"b": [1, 2], "a": "C5"}
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert inputs._digest(value) == hashlib.sha256(raw).hexdigest()
        assert inputs._digest(value) != hashlib.sha256(raw + b"\n").hexdigest()


    def test_initial_decision_requires_complete_stop_and_asks(self):
        task = {"total_millihours": 100, "cost_millihours": 5}
        ask = {"expectedIncrementalPersonalSeconds": 1, "objective": 2,
               "slot": 0, "tutoringIntegerCost": 5}
        stop = {"action": "MFN_A", "expectedCombinedGPA": 1, "expectedUtility": 1,
                "forced": False, "index": 1000, "legacySelection": {},
                "objective": 1, "reason": "terminal_menu", "seconds": 1}
        decision = {"action": "ask_A", "asks": [ask, dict(ask, slot=1)], "budget": 100,
                    "selectedObjective": 2, "stop": stop}
        inputs._decision(decision, task)
        del stop["index"]
        with self.assertRaises(ValueError):
            inputs._decision(decision, task)


    def test_prefix_exact_tie_and_full_denominator(self):
        tmp_path = self.tmp_path
        repo = tmp_path / "repo"
        repo.mkdir()
        path = tmp_path / "prefix.json"
        data = _prefix()
        endpoint = data["rows"][0]["endpoint"]
        endpoint.update(status="exact_tie", nonTie=False, latentDisplayed=[2, 2],
                        latentLogical={"A": 2, "B": 2}, reason="equal_true_needs", higherNeedShare=None)
        sha = _write(path, data)
        result = inputs.load_historical_prefix(path, sha, repository_root=repo)
        assert len(result["payload"]["rows"]) == result["rootCount"] == 90
        assert "including-exact-ties" in result["denominatorRule"]
        data["rows"][0]["contribution"]["values"]["gap"] = ["1", "2"]
        sha = _write(path, data)
        with self.assertRaisesRegex(ValueError, "tie"):
            inputs.load_historical_prefix(path, sha, repository_root=repo)
        endpoint["status"] = "statistical_unavailable"
        sha = _write(path, data)
        with self.assertRaises(ValueError):
            inputs.load_historical_prefix(path, sha, repository_root=repo)


    def test_prefix_rejects_tampered_used_seed(self):
        for mutation in ("hash", "seed", "domain", "root", "width", "missing", "duplicate"):
            with self.subTest(mutation=mutation):
                self.check_tampered_used_seed(mutation)


    def check_tampered_used_seed(self, mutation):
        tmp_path = self.tmp_path
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
        path = tmp_path / "prefix.json"
        data = _prefix()
        events = data["rows"][0]["usedSeeds"]
        event = events[0]
        if mutation == "hash":
            event["keySHA256"] = "0" * 64
        elif mutation == "seed":
            event["seedInteger"] = "0"
        elif mutation == "domain":
            event["key"]["randomizationGroup"] = "wrong"
            event["keySHA256"] = inputs._digest(event["key"])
            event["seedInteger"] = str(int(event["keySHA256"][:32], 16))
        elif mutation == "root":
            event["key"]["mainIndex"] = 2
            event["keySHA256"] = inputs._digest(event["key"])
            event["seedInteger"] = str(int(event["keySHA256"][:32], 16))
        elif mutation == "width":
            event["seedBits"] = 256
        elif mutation == "missing":
            events.pop()
        else:
            events.append(dict(event))
        sha = _write(path, data)
        with self.assertRaises(ValueError):
            inputs.load_historical_prefix(path, sha, repository_root=repo)


    def test_terminal_event_domain_and_hash(self):
        key = {"version": "r10-native-terminal-pcg64-sha256-v1",
               "randomizationGroup": "r14-c5-paired-projection-terminal-v1",
               "keyEncoding": "typed-json-floathex-v1",
               "candidate": {"dict": [["candidateId", {"str": "fixture"}]]},
               "identity": {"seed": 20261019, "conditionId": "C5", "variant": 0,
                            "replicateId": 1, "mainIndex": 1, "modelId": "fixture",
                            "personId": "fixture", "trialId": "fixture"}}
        hashed = inputs._digest(key)
        event = {"key": key, "keySHA256": hashed, "seedInteger": str(int(hashed, 16)), "seedBits": 256}
        assert inputs._event(event, ("C5", 0, 1)) == ("terminal", event["seedInteger"])
        key["randomizationGroup"] = "wrong"
        with self.assertRaisesRegex(ValueError, "domain"):
            inputs._event(event, ("C5", 0, 1))


    def test_fixed_candidate_rejects_tampered_bytes(self):
        tmp_path = self.tmp_path
        task = {"report_width": 1, "total_millihours": 100, "cost_millihours": 5}
        cfg = {"task": task}
        ask = {"expectedIncrementalPersonalSeconds": 1, "objective": 2,
               "slot": 0, "tutoringIntegerCost": 5}
        stop = {"action": "MFN_A", "expectedCombinedGPA": 1, "expectedUtility": 1,
                "forced": False, "index": 1000, "legacySelection": {},
                "objective": 1, "reason": "terminal_menu", "seconds": 1}
        decision = {"action": "ask_A", "asks": [ask, dict(ask, slot=1)], "budget": 100,
                    "selectedObjective": 2, "stop": stop}
        data = {"schema": inputs.FIXED_SCHEMA, "target": inputs.TARGET,
                "sourceInputSHA256": inputs.INPUT_SHA256, "identity": inputs.CANDIDATE_IDENTITY,
                "roots": [{"rootKey": list(root), "rootSHA256": "a" * 64,
                           "configSHA256": inputs._digest(cfg), "config": cfg,
                           "logicalToDisplayed": ["A", "B"], "reportWidth": 1,
                           "highPriceFirstDisplayNoQuestion": True, "decision": decision}
                          for root in inputs.ROSTER]}
        path = tmp_path / "candidate.json"
        original_sha = _write(path, data)
        assert inputs.load_fixed_inputs(path, original_sha)["identity"] == inputs.CANDIDATE_IDENTITY
        data["roots"][0]["logicalToDisplayed"] = ["B", "A"]
        _write(path, data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            inputs.load_fixed_inputs(path, original_sha)
        data["roots"][0]["decision"]["stop"]["index"] = 0
        _write(path, data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            inputs.load_fixed_inputs(path, original_sha)


    def test_prefix_mapping_uses_each_logical_recipient(self):
        tmp_path = self.tmp_path
        repo = tmp_path / "repo"
        repo.mkdir()
        path = tmp_path / "prefix.json"
        data = _prefix()
        lookup = {"roots": [{"rootKey": list(root), "logicalToDisplayed": ["A", "B"]}
                            for root in inputs.ROSTER]}
        sha = _write(path, data)
        result = inputs.load_historical_prefix(path, sha, repository_root=repo, lookup=lookup)
        assert result["mappingCheck"] == "caller-lookup-consistency-only"
        assert result["finalReductionAllowed"] is False
        lookup["roots"][0]["logicalToDisplayed"] = ["B", "A"]
        with self.assertRaisesRegex(ValueError, "mapping"):
            inputs.load_historical_prefix(path, sha, repository_root=repo, lookup=lookup)
        data["rows"][0]["endpoint"]["latentLogical"]["A"] = True
        with self.assertRaises(ValueError):
            inputs.load_historical_prefix(path, _write(path, data), repository_root=repo)


    def test_scientific_and_fixed_config_extra_fields_refused(self):
        data = _prefix()
        endpoint = data["rows"][0]["endpoint"]
        contribution = data["rows"][0]["contribution"]
        inputs._scientific_fields(endpoint, contribution)
        endpoint["sourcePath"] = "relative/private"
        with self.assertRaisesRegex(ValueError, "scientific field"):
            inputs._scientific_fields(endpoint, contribution)
        with self.assertRaisesRegex(ValueError, "config field"):
            inputs._fixed_config_fields({"task": {"sourcePath": "relative/private"}})
        with self.assertRaisesRegex(ValueError, "config field"):
            inputs._fixed_config_fields({"task": {"candidateId": "valid-name-wrong-location"}})
        with self.assertRaisesRegex(ValueError, "candidate scientific field"):
            inputs._typed_candidate({"dict": [["sourcePath", {"str": "relative/private"}]]})


    def test_external_fixed_source_config_fields(self):
        path = os.environ.get("C5_FIXED_SOURCE_INPUTS")
        if not path:
            self._missing_fixture("C5_FIXED_SOURCE_INPUTS")
        source = inputs._read(path, inputs.INPUT_SHA256)
        assert len(source["roots"]) == 90
        for row in source["roots"]:
            inputs._fixed_config_fields(row["config"])


    def test_external_reader_capsule_scientific_shape(self):
        path = os.environ.get("C5_PREFIX_READER_CAPSULE_FIXTURE")
        if not path:
            self._missing_fixture("C5_PREFIX_READER_CAPSULE_FIXTURE")
        with open(path, encoding="utf-8") as source:
            capsule = json.load(source)
        assert capsule["schema"] == "r14-c5-final-capsule-v1"
        assert capsule["role"] == "reader"
        for row in capsule["rows"]:
            inputs._scientific_fields(row["endpoint"], row["contribution"])
            root = inputs._key(row["rootKey"])
            for event in row["producerUsedSeeds"]:
                inputs._event(event, root)


class PrefixBoundaryUnitTests(unittest.TestCase):
    """Synthetic boundary tests; no historical admission or scientific run."""

    def test_type_sensitive_nested_values(self):
        self.assertNotEqual(inputs._digest({"nested": [True]}), inputs._digest({"nested": [1]}))
        self.assertNotEqual(inputs._digest({"nested": [1]}), inputs._digest({"nested": [1.0]}))

    def test_logical_display_mapping_and_bool(self):
        endpoint = {"latentLogical": {"A": 1, "B": 2}, "latentDisplayed": [1, 2]}
        inputs._check_mapping(endpoint, ["A", "B"])
        with self.assertRaises(ValueError):
            inputs._check_mapping(endpoint, ["B", "A"])
        endpoint["latentLogical"]["A"] = True
        with self.assertRaises(ValueError):
            inputs._check_mapping(endpoint, ["A", "B"])

    def test_scientific_fields_reject_extra_private_data(self):
        endpoint = dict.fromkeys(inputs._ENDPOINT_FIELDS)
        contribution = dict.fromkeys(inputs._CONTRIBUTION_FIELDS)
        inputs._scientific_fields(endpoint, contribution)
        endpoint["sourcePath"] = "/private/file"
        with self.assertRaises(ValueError):
            inputs._scientific_fields(endpoint, contribution)
        with self.assertRaises(ValueError):
            inputs._no_private_strings({"sourcePath": "relative/private"})
        with self.assertRaises(ValueError):
            inputs._typed_candidate({"dict": [["actor", {"str": "private"}]]})
        with self.assertRaises(ValueError):
            inputs._typed_candidate({"dict": [["sourcePath", {"str": "relative/private"}]]})
