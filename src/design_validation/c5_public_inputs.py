"""Fixed C5 initial policy and externally supplied historical-prefix boundaries.

The intended fixed policy is adopted, but a loaded lookup is only a candidate
until independently authenticated. It is not a fresh policy calculation. A prefix
digest establishes byte integrity only; historical acceptance needs external
evidence and is never inferred from a user-supplied hash.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

INPUT_SHA256 = "ba3425970e62773db1e3ab499202fc66ce64d308f77bb77e853b9abf0895e582"
TARGET = "r14-c5-fixed1840-canonical-policy-v1"
FIXED_SCHEMA = "c5-fixed-initial-policy-v1"
PREFIX_SCHEMA = "c5-supplied-historical-prefix-v1"
CANDIDATE_IDENTITY = "fixed-policy-candidate-not-authenticated"
ROSTER = [("C5", v, m) for v in range(9) for m in range(1, 11)]
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON member: " + key)
        out[key] = value
    return out


def _read(path, expected_sha256):
    if not isinstance(expected_sha256, str) or not _SHA.fullmatch(expected_sha256):
        raise ValueError("expected SHA256 must be 64 lowercase hex characters")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("input byte SHA256 mismatch")
    return json.loads(raw, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _digest(value):
    # The original personal_time_inputs digest has no trailing newline.
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


_ENDPOINT_FIELDS = {"rootKey", "status", "nonTie", "latentDisplayed", "latentLogical",
                    "initialBudget", "remainingBudget", "queryCount", "reason", "higherNeedShare"}
_CONTRIBUTION_FIELDS = {"status", "reason", "values"}
_CANDIDATE_FIELDS = frozenset({
    "schema", "candidateId", "policyId", "valuation", "task", "settings", "actionCosts", "random",
    "insufficientBudgetMode", "state", "pathReports", "rhoGPAperSecond", "description",
    "condition_id", "template_id", "scale", "report_width", "alpha", "loss_multiplier",
    "gpa_scale", "gap_offset", "gpa_baseline", "prior_centres", "prior_spreads", "total_millihours",
    "cost_millihours", "repeat_count_per_30", "allocation_denominator", "max_assessments",
    "predecessor_profile", "working_C1_total_gap_minutes", "C1_total_gaps_minutes", "C1_gap_scaled",
    "rho_multipliers", "common_execution_seconds", "illustrative_common_components", "point", "button",
    "mental_preparation", "timing_source", "timing_evidence_scope", "assumed_profiles", "low", "base",
    "high", "C1_independent_inference_seconds", "reference", "personal_question_time_priced",
    "automatic_zero_budget_duration_seconds", "display_rounding", "decision_tolerance", "classification_margin",
    "probability_tolerance", "posterior_parity_tolerance", "initial_states", "prior_terminal_records",
    "single_numerical_thread", "empirically_calibrated_cognitive_durations", "machine_certified_arithmetic",
    "read_scalar", "retrieve_rule", "add_subtract", "multiply_divide", "compare", "clip",
    "EO_inference_seconds", "OPT_inference_seconds", "terminalProfile", "commonTerminalSeconds",
    "requestCounts", "requestOverheadSeconds", "readCounts", "readOverheadSeconds", "uniform", "byReport",
    "probabilities", "counts", "overheadSeconds", "voluntaryEnabled", "streamNamespace", "budget", "logs",
    "slot", "report",
})


def _typed_candidate(value):
    if type(value) is not dict or len(value) != 1:
        raise ValueError("invalid typed candidate node")
    tag, content = next(iter(value.items()))
    if tag == "dict":
        if type(content) is not list:
            raise ValueError("invalid typed candidate dictionary")
        names = []
        for pair in content:
            if (type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str
                    or pair[0] not in _CANDIDATE_FIELDS):
                raise ValueError("unknown candidate scientific field")
            names.append(pair[0])
            _typed_candidate(pair[1])
        if names != sorted(set(names)):
            raise ValueError("invalid typed candidate field order")
    elif tag == "list":
        if type(content) is not list:
            raise ValueError("invalid typed candidate list")
        for item in content:
            _typed_candidate(item)
    elif tag == "null":
        if content is not True:
            raise ValueError("invalid typed null")
    elif tag == "bool":
        if type(content) is not bool:
            raise ValueError("invalid typed boolean")
    elif tag in ("int", "floatHex", "str"):
        if type(content) is not str:
            raise ValueError("invalid typed scalar")
        _no_private_strings(content)
    else:
        raise ValueError("unknown typed candidate node")


def _check_mapping(endpoint, mapping):
    if type(mapping) is not list or mapping not in (["A", "B"], ["B", "A"]):
        raise ValueError("invalid lookup logical/display mapping")
    logical, displayed = endpoint["latentLogical"], endpoint["latentDisplayed"]
    if any(type(logical[k]) is not int or not 1 <= logical[k] <= 100 for k in ("A", "B")):
        raise ValueError("invalid logical latent values")
    if [logical[k] for k in mapping] != displayed:
        raise ValueError("logical/display mapping mismatch")


def _scientific_fields(endpoint, contribution):
    if (type(endpoint) is not dict or set(endpoint) != _ENDPOINT_FIELDS
            or type(contribution) is not dict or set(contribution) != _CONTRIBUTION_FIELDS):
        raise ValueError("unexpected scientific field")


def _no_private_strings(value):
    if type(value) is str:
        if (value.startswith(("/", "file:", "~", "\\\\")) or "../" in value
                or re.match(r"^[A-Za-z]:[\\/]", value)):
            raise ValueError("private path in scientific input")
    elif type(value) is dict:
        for key, child in value.items():
            if not isinstance(key, str) or re.search(r"(?:actor|credential|hostname|scheduler|sourcepath|filepath)", key, re.I):
                raise ValueError("private field in scientific input")
            _no_private_strings(child)
    elif type(value) is list:
        for child in value:
            _no_private_strings(child)


_PROFILE_FIELDS = frozenset({"read_scalar", "retrieve_rule", "add_subtract", "multiply_divide", "compare",
                             "clip", "EO_inference_seconds", "OPT_inference_seconds"})
_CONFIG_PATH_FIELDS = {
    "": frozenset({"schema", "candidateId", "policyId", "valuation", "task", "settings", "actionCosts",
                    "random", "insufficientBudgetMode", "state", "pathReports"}),
    "valuation": frozenset({"rhoGPAperSecond", "description"}),
    "task": frozenset({"condition_id", "template_id", "scale", "gap_offset", "prior_centres", "prior_spreads",
                       "report_width", "total_millihours", "cost_millihours", "repeat_count_per_30",
                       "allocation_denominator", "max_assessments", "alpha", "loss_multiplier", "gpa_baseline", "gpa_scale"}),
    "settings": frozenset({"schema", "predecessor_profile", "working_C1_total_gap_minutes", "C1_total_gaps_minutes",
                           "C1_gap_scaled", "rho_multipliers", "common_execution_seconds", "assumed_profiles",
                           "C1_independent_inference_seconds", "personal_question_time_priced",
                           "automatic_zero_budget_duration_seconds", "empirically_calibrated_cognitive_durations",
                           "machine_certified_arithmetic", "initial_states"}),
    "settings.assumed_profiles": frozenset({"low", "base", "high"}),
    "settings.assumed_profiles.low": _PROFILE_FIELDS,
    "settings.assumed_profiles.base": _PROFILE_FIELDS,
    "settings.assumed_profiles.high": _PROFILE_FIELDS,
    "actionCosts": frozenset({"terminalProfile", "commonTerminalSeconds", "requestCounts", "requestOverheadSeconds",
                               "readCounts", "readOverheadSeconds"}),
    "actionCosts.terminalProfile": _PROFILE_FIELDS,
    "actionCosts.readCounts[]": frozenset({"uniform", "byReport"}),
    "actionCosts.readOverheadSeconds[]": frozenset({"uniform", "byReport"}),
    "random": frozenset({"probabilities", "counts", "overheadSeconds", "voluntaryEnabled", "streamNamespace"}),
    "state": frozenset({"budget", "logs"}),
    "pathReports[]": frozenset({"slot", "report"}),
}


def _fixed_config_fields(value, path=""):
    """Allow scientific fields only at their declared configuration paths."""
    if type(value) is dict:
        allowed = _CONFIG_PATH_FIELDS.get(path)
        if allowed is None:
            raise ValueError("unknown fixed config object")
        for key, child in value.items():
            if type(key) is not str or key not in allowed:
                raise ValueError("unknown fixed config field")
            _fixed_config_fields(child, path + "." + key if path else key)
    elif type(value) is list:
        for child in value:
            _fixed_config_fields(child, path + "[]")
    else:
        _no_private_strings(value)


def _key(value):
    if (not isinstance(value, list) or len(value) != 3 or value[0] != "C5"
            or type(value[1]) is not int or type(value[2]) is not int
            or tuple(value) not in ROSTER):
        raise ValueError("invalid C5 root key")
    return tuple(value)


def _decision(decision, task):
    if not isinstance(decision, dict) or set(decision) != {"action", "asks", "budget", "selectedObjective", "stop"}:
        raise ValueError("incomplete initial decision")
    stop = decision["stop"]
    required = {"action", "expectedCombinedGPA", "expectedUtility", "forced", "index", "legacySelection", "objective", "reason", "seconds"}
    if (not isinstance(stop, dict) or set(stop) != required or type(stop["index"]) is not int
            or not 0 <= stop["index"] <= 1000 or type(stop["forced"]) is not bool
            or decision["action"] != "ask_A" or decision["budget"] != task["total_millihours"]
            or not isinstance(stop["legacySelection"], dict)):
        raise ValueError("invalid initial stop/action/budget")
    asks = decision["asks"]
    if (not isinstance(asks, list) or len(asks) != 2
            or any(not isinstance(a, dict) or set(a) != {"expectedIncrementalPersonalSeconds", "objective", "slot", "tutoringIntegerCost"} for a in asks)
            or [a["slot"] for a in asks] != [0, 1]
            or any(a["tutoringIntegerCost"] != task["cost_millihours"] for a in asks)):
        raise ValueError("invalid initial ask candidates")
    _digest(decision)


def export_fixed_inputs(source_path):
    """Extract only scientific fields from the exact accepted input bytes."""
    source = _read(source_path, INPUT_SHA256)
    if source.get("schema") != "r14-c5-canonical-inputs-v1" or len(source.get("roots", [])) != 90:
        raise ValueError("wrong accepted input schema or roster")
    rows = []
    for item, expected in zip(source["roots"], ROSTER):
        key = _key(item["rootKey"])
        raw, cfg, low, law = item["rawRoot"], item["config"], item["low"], item["acceptedLaw"]["value"]
        definition, task = raw["definition"], cfg["task"]
        mapping = raw["logicalToDisplayed"]
        if (key != expected or raw["conditionId"] != "C5" or raw["scheduleVariant"] != key[1]
                or raw["mainIndex"] != key[2] or law["rootKey"] != list(key)
                or _digest(raw) != law["rootSHA256"] or _digest(cfg) != item["configSHA256"]
                or low["configSHA256"] != item["configSHA256"] or low["price"] != "rho_0.001"
                or mapping not in (["A", "B"], ["B", "A"]) or law["logicalToDisplayed"] != mapping
                or definition["sideReversed"] is not (mapping == ["B", "A"])
                or definition["stageIndex"] != key[2] or task["condition_id"] != "C5"
                or task["report_width"] != definition["reportWidth"]
                or task["total_millihours"] != definition["budgetMillihours"]
                or task["cost_millihours"] != definition["costMillihours"]
                or task["allocation_denominator"] != 1000 or law["budget"] != task["total_millihours"]
                or law["highPriceFirstDisplayNoQuestion"] is not True):
            raise ValueError("root/config/law binding mismatch: %r" % (key,))
        _decision(low["decision"], task)
        _fixed_config_fields(cfg)
        _no_private_strings(low["decision"])
        rows.append({"rootKey": list(key), "rootSHA256": law["rootSHA256"],
                     "configSHA256": item["configSHA256"], "config": cfg,
                     "logicalToDisplayed": mapping, "reportWidth": definition["reportWidth"],
                     "highPriceFirstDisplayNoQuestion": True, "decision": low["decision"]})
    return {"schema": FIXED_SCHEMA, "target": TARGET, "sourceInputSHA256": INPUT_SHA256,
            "identity": CANDIDATE_IDENTITY, "roots": rows}


def load_fixed_inputs(path, expected_sha256):
    """Load a byte-checked candidate, not an authenticated adopted lookup."""
    data = _read(path, expected_sha256)
    if (data.get("schema") != FIXED_SCHEMA or data.get("target") != TARGET
            or data.get("sourceInputSHA256") != INPUT_SHA256
            or data.get("identity") != CANDIDATE_IDENTITY
            or len(data.get("roots", [])) != 90):
        raise ValueError("invalid fixed lookup envelope")
    for row, expected in zip(data["roots"], ROSTER):
        cfg = row["config"]
        if (_key(row["rootKey"]) != expected or _digest(cfg) != row["configSHA256"]
                or not _SHA.fullmatch(row["rootSHA256"])
                or row["logicalToDisplayed"] not in (["A", "B"], ["B", "A"])
                or row["reportWidth"] != cfg["task"]["report_width"]
                or row["highPriceFirstDisplayNoQuestion"] is not True):
            raise ValueError("invalid fixed lookup row")
        _decision(row["decision"], cfg["task"])
        _fixed_config_fields(cfg)
        _no_private_strings(row["decision"])
    return data


def _event(event, root):
    """Validate the original usedSeeds shape; return its derived role and seed."""
    if not isinstance(event, dict) or set(event) != {"key", "keySHA256", "seedInteger", "seedBits"}:
        raise ValueError("invalid used seed record")
    key, bits = event["key"], event["seedBits"]
    if not isinstance(key, dict) or type(bits) is not int or bits not in (128, 256):
        raise ValueError("invalid used seed key or width")
    if bits == 128:
        fields = {"streamVersion", "populationId", "seed", "randomizationGroup", "conditionId",
                  "scheduleVariant", "replicateId", "mainIndex", "logicalRecipient", "streamKind", "reportIndex"}
        if (set(key) != fields or key["streamVersion"] != "r10-private-need-report-v1"
                or key["populationId"] != "r10-working-five-v0-rep0-candidate-not-population-estimate"
                or key["randomizationGroup"] != "r14-c5-paired-projection-private-stream-v1"
                or type(key["seed"]) is not int or key["seed"] != 20261019
                or key["conditionId"] != root[0]
                or type(key["scheduleVariant"]) is not int or key["scheduleVariant"] != root[1]
                or type(key["replicateId"]) is not int or key["replicateId"] != 1
                or type(key["mainIndex"]) is not int or key["mainIndex"] != root[2]
                or key["logicalRecipient"] not in ("A", "B") or key["streamKind"] not in ("latent", "report")
                or type(key["reportIndex"]) is not int or key["reportIndex"] < 0
                or (key["streamKind"] == "latent" and key["reportIndex"] != 0)):
            raise ValueError("invalid private event domain or coordinate")
        role = key["streamKind"]
    else:
        if (set(key) != {"version", "randomizationGroup", "identity", "candidate", "keyEncoding"}
                or key["version"] != "r10-native-terminal-pcg64-sha256-v1"
                or key["randomizationGroup"] != "r14-c5-paired-projection-terminal-v1"
                or key["keyEncoding"] != "typed-json-floathex-v1"
                or not isinstance(key["identity"], dict) or not isinstance(key["candidate"], dict)):
            raise ValueError("invalid terminal event domain")
        identity = key["identity"]
        _typed_candidate(key["candidate"])
        if (set(identity) != {"seed", "modelId", "personId", "trialId", "conditionId", "variant", "replicateId", "mainIndex"}
                or identity.get("conditionId") != root[0] or type(identity.get("variant")) is not int
                or identity["variant"] != root[1] or type(identity.get("mainIndex")) is not int
                or identity["mainIndex"] != root[2] or type(identity.get("replicateId")) is not int
                or identity["replicateId"] != 1 or type(identity.get("seed")) is not int
                or identity["seed"] != 20261019
                or any(not isinstance(identity.get(k), str) or not identity[k]
                       for k in ("modelId", "personId", "trialId"))):
            raise ValueError("invalid terminal event identity")
        role = "terminal"
    hashed = _digest(key)
    seed = str(int(hashed if bits == 256 else hashed[:32], 16))
    if event["keySHA256"] != hashed or type(event["seedInteger"]) is not str or event["seedInteger"] != seed:
        raise ValueError("used seed key/hash/seed mismatch")
    return role, seed


def require_authenticated_prefix(prefix):
    """Final reduction must not consume a prefix until authority admission exists."""
    raise ValueError("historical prefix authority is not established; final reduction is forbidden")


def load_historical_prefix(path, expected_sha256, *, repository_root, lookup=None):
    """Admit an ignored, supplied rep1 bundle without asserting provenance.

    Optional lookup is a caller-supplied fixed lookup object. Mapping checks
    establish consistency only, not that the lookup or prefix is authenticated.
    Scientific rows contain no source descriptors. Final admission remains blocked.
    """
    path, repo = Path(path).resolve(strict=True), Path(repository_root).resolve(strict=True)
    if path.is_relative_to(repo):
        check = subprocess.run(["git", "-C", str(repo), "check-ignore", "-q", "--", str(path)],
                               check=False, capture_output=True)
        if check.returncode != 0:
            raise ValueError("historical prefix inside checkout must be git-ignored")
    data = _read(path, expected_sha256)
    if (type(data) is not dict or set(data) != {"schema", "target", "replicateId", "rows"}
            or data.get("schema") != PREFIX_SCHEMA or data.get("target") != TARGET
            or data.get("replicateId") != 1 or type(data.get("replicateId")) is not int
            or not isinstance(data.get("rows"), list) or len(data["rows"]) != 90):
        raise ValueError("invalid supplied prefix envelope")
    if lookup is not None and (type(lookup) is not dict or type(lookup.get("roots")) is not list
                               or len(lookup["roots"]) != 90):
        raise ValueError("complete caller lookup required for mapping checks")
    seeds, hashes = {}, set()
    for position, (row, expected) in enumerate(zip(data["rows"], ROSTER)):
        if (set(row) != {"rootKey", "endpoint", "contribution", "usedSeeds"}
                or _key(row["rootKey"]) != expected
                or not isinstance(row.get("endpoint"), dict)
                or not isinstance(row.get("contribution"), dict)
                or not isinstance(row.get("usedSeeds"), list) or not row["usedSeeds"]):
            raise ValueError("incomplete supplied prefix row")
        endpoint, contribution = row["endpoint"], row["contribution"]
        _scientific_fields(endpoint, contribution)
        _no_private_strings(row)
        if (endpoint.get("rootKey") != list(expected) or endpoint.get("status") not in ("complete", "exact_tie")
                or contribution.get("status") != "complete"
                or not isinstance(contribution.get("values"), dict)
                or set(contribution["values"]) != {"gap", "partial_low"}
                or any(not isinstance(pair, list) or len(pair) != 2
                       or any(not isinstance(part, str) or not re.fullmatch(r"-?(0|[1-9][0-9]*)", part) for part in pair)
                       or pair[1].startswith("-") or pair[1] == "0"
                       for pair in contribution["values"].values())
                ):
            raise ValueError("incomplete prefix endpoint, contribution or event identities")
        if endpoint["status"] == "exact_tie":
            displayed = endpoint.get("latentDisplayed")
            if (endpoint.get("nonTie") is not False or not isinstance(displayed, list) or len(displayed) != 2
                    or displayed[0] != displayed[1] or endpoint.get("higherNeedShare") is not None
                    or any(pair[0] != "0" for pair in contribution["values"].values())):
                raise ValueError("invalid exact tie or nonzero tie contribution")
        elif endpoint.get("nonTie") is not True:
            raise ValueError("complete endpoint must be non-tie")
        if (type(endpoint.get("queryCount")) is not int or endpoint["queryCount"] < 0
                or type(endpoint.get("initialBudget")) is not int or endpoint["initialBudget"] <= 0
                or type(endpoint.get("remainingBudget")) is not int
                or not 0 < endpoint["remainingBudget"] <= endpoint["initialBudget"]
                or not isinstance(endpoint.get("latentDisplayed"), list) or len(endpoint["latentDisplayed"]) != 2
                or any(type(x) is not int or not 1 <= x <= 100 for x in endpoint["latentDisplayed"])
                or not isinstance(endpoint.get("latentLogical"), dict)
                or set(endpoint["latentLogical"]) != {"A", "B"}
                or any(type(x) is not int or not 1 <= x <= 100 for x in endpoint["latentLogical"].values())
                or sorted(endpoint["latentLogical"].values()) != sorted(endpoint["latentDisplayed"])
                or contribution.get("reason") is not None):
            raise ValueError("invalid saved endpoint support")
        if lookup is not None:
            fixed = lookup["roots"][position]
            if type(fixed) is not dict or _key(fixed.get("rootKey")) != expected:
                raise ValueError("lookup root roster mismatch")
            _check_mapping(endpoint, fixed.get("logicalToDisplayed"))
        if endpoint["status"] == "complete":
            share = endpoint.get("higherNeedShare")
            if (endpoint.get("reason") is not None or endpoint["latentDisplayed"][0] == endpoint["latentDisplayed"][1]
                    or not isinstance(share, list) or len(share) != 2
                    or any(type(x) is not str or not re.fullmatch(r"0|[1-9][0-9]*", x) for x in share)
                    or share[1] == "0" or int(share[0]) > int(share[1])):
                raise ValueError("invalid non-tie share")
        elif endpoint.get("reason") != "equal_true_needs":
            raise ValueError("invalid tie reason")
        latent = set()
        roles, counts = [], {"A": 0, "B": 0}
        for event in row["usedSeeds"]:
            role, seed = _event(event, expected)
            roles.append(role)
            if role == "report":
                key = event["key"]
                logical = key["logicalRecipient"]
                if key["reportIndex"] != counts[logical]:
                    raise ValueError("invalid recipient report frontier")
                counts[logical] += 1
            hashed = event["keySHA256"]
            if hashed in hashes or seed in seeds:
                raise ValueError("duplicate event or cross-width seed collision")
            hashes.add(hashed)
            seeds[seed] = hashed
            if role == "latent":
                latent.add(event["key"]["logicalRecipient"])
        if latent != {"A", "B"}:
            raise ValueError("missing latent recipient events")
        if (roles[:2] != ["latent", "latent"] or
                [event["key"].get("logicalRecipient") for event in row["usedSeeds"][:2]] != ["A", "B"] or
                roles[2:] not in (["report"] * endpoint["queryCount"],
                                   ["report"] * endpoint["queryCount"] + ["terminal"])):
            raise ValueError("event ordering or report count mismatch")
    return {"payload": data, "contentSHA256": expected_sha256,
            "historicalAcceptance": "not-established-by-this-loader",
            "mappingCheck": "caller-lookup-consistency-only" if lookup is not None else "not-checked",
            "finalReductionAllowed": False, "rootCount": 90,
            "denominatorRule": "all-90-roots-per-replicate-including-exact-ties"}
