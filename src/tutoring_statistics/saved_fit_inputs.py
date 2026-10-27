"""Authenticated inputs for the fixed saved-fit magnitude diagnostic.

Historical descriptors are never rewritten. Optional location maps resolve
archived bytes; newly generated public artifacts have a separate identity.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .saved_fit_magnitude import (TARGET, FIT_KEY, NAMESPACE, METHOD, AGGREGATE_SHA,
    INDEPENDENT_SHA, MANIFEST_SHA, exact, require, validate_population)

def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Observations:
    def __init__(self, locations=None):
        self.files = {}
        self.locations = dict(locations or {})
        require(all(type(key) is str and type(value) is str and Path(value).is_absolute()
                    for key, value in self.locations.items()), "Invalid relocation map")
        require(all(value not in self.locations or self.locations[value] == value
                    for key, value in self.locations.items() if key != value), "Chained relocation is ambiguous")

    def resolve(self, path):
        selected = Path(self.locations.get(str(path), str(path)))
        require(not selected.is_symlink(), "Data/source relocation cannot use a symlink")
        return selected.resolve()

    def actual(self, path, expected=None):
        """Bind executing code, runtime and outputs without data relocation."""
        path = Path(path).resolve()
        require(self.resolve(path) == path, "Cannot relocate actual execution artifact")
        return self.observe(path, expected)

    def observe(self, path, expected=None):
        path = self.resolve(path)
        require(path.is_absolute() and path.is_file(), "Absolute existing input required")
        digest = sha(path)
        require(expected is None or digest == expected, "Changed bound input: " + str(path))
        require(str(path) not in self.files or self.files[str(path)] == digest, "Input changed during run")
        self.files[str(path)] = digest
        return dict(path=str(path), sha256=digest)

    def read(self, pin):
        require(type(pin) is dict and set(pin) == {"path", "sha256"}
                and type(pin["sha256"]) is str and len(pin["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in pin["sha256"]), "Invalid exact descriptor")
        self.observe(pin["path"], pin["sha256"])
        return load(self.resolve(pin["path"]))

    def fresh(self):
        for path, digest in self.files.items():
            require(sha(path) == digest, "Input changed at closure: " + path)


def write_new(path, value):
    path = Path(path)
    require(path.is_absolute() and path.parent.is_dir() and not path.exists()
            and not path.is_symlink(), "New absolute output required")
    fd, temporary = tempfile.mkstemp(prefix=".saved-magnitude-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def actor(value):
    require(type(value) is dict and set(value) == {"jobID", "taskID", "host", "pid"}
            and type(value["jobID"]) is str and value["jobID"].isdigit()
            and type(value["taskID"]) is str and (value["taskID"] == "undefined" or value["taskID"].isdigit())
            and type(value["host"]) is str and bool(value["host"])
            and type(value["pid"]) is int and value["pid"] > 0, "Invalid historical actor")
    return dict(value)


INPUT_SCHEMA = "saved-magnitude-input-v1"
PUBLIC_SCHEMA = "conditional-statistics-reproduction-v1"


def admit_public(request, observed):
    """Admit an existing public producer member, without invoking production."""
    from . import power_reference
    exact(request["target"], TARGET, "fixed public target")
    manifest = observed.read(request["manifest"])
    require(manifest["schema"] == PUBLIC_SCHEMA and manifest["family"] == "power"
            and manifest["expectedTotal"] == 80000
            and manifest["identity"] == "new_public_reproduction", "Wrong public manifest")
    for name in ("sources", "driver", "rDriver", "environment", "nativeEntry", "nativeIdentity"):
        observed.observe(manifest[name]["path"], manifest[name]["sha256"])
    snapshot = observed.read(manifest["sources"])
    roles = {name: manifest[name] for name in
             ("driver", "rDriver", "environment", "nativeEntry", "nativeIdentity")}
    required_names = dict(driver="reproduction.py", rDriver="kernel_entry.R",
        environment="numerical_environment.json", nativeEntry="native_inputs.py", nativeIdentity="native_identity.json")
    for role, pin in roles.items():
        require(pin in snapshot["files"] and Path(pin["path"]).name == required_names[role],
                "Missing or replaced public source role")
    required_sources = {"power_reference.py", "power_origins.py", "joint_generation.py", "joint_fit.R",
        "population_fit.R", "cluster_fit.R", "independent_power.py", "independent_transport.py",
        "independent_cluster_rds.R", "source_transport.R", "mixed_fit.R", "joint_design.py"}
    names = [Path(pin["path"]).name for pin in snapshot["files"]]
    require(required_sources <= set(names), "Incomplete public scientific source inventory")
    require({Path(pin["path"]).name for pin in manifest["native"]} ==
            {"resolved_design.json", "initial_panel.json", "h10_canonical.json"}
            and len(manifest["native"]) == 3, "Changed public native input roster")
    for pin in snapshot["files"] + manifest["native"]:
        observed.observe(pin["path"], pin["sha256"])
    for name in ("executable", "Rscript"):
        pin = manifest["numericalRuntime"][name]
        observed.observe(pin["path"], pin["sha256"])
    spec = observed.read(request["spec"])
    exact(spec["case"], power_reference.case_for(32, 2000), "public original case")
    exact({key: spec[key] for key in TARGET}, TARGET, "public original target")
    require(spec["manifest"] == request["manifest"] and spec["fitKey"] == FIT_KEY
            and spec["origin"] is None, "Changed public target origin")
    commitment = observed.read(request["commitment"])
    event = commitment["event"]
    digest = hashlib.sha256(json.dumps(event, sort_keys=True, allow_nan=False,
                                      separators=(",", ":")).encode()).hexdigest()
    require(commitment["schema"] == PUBLIC_SCHEMA and commitment["manifest"] == request["manifest"]
            and commitment["eventSha256"] == digest, "Changed public event commitment")
    exact({key: event[key] for key in TARGET}, TARGET, "public committed target")
    require(event["fitKey"] == FIT_KEY and event["origin"] == "new_population_input"
            and event["technicalCompleted"] is True, "Unfinished or foreign public event")
    artifacts = commitment["artifacts"]
    for pin in artifacts:
        observed.observe(pin["path"], pin["sha256"])
    for pin in (request["spec"], request["baseline"], request["candidate"], spec["request"], spec["generation"]):
        require(pin in artifacts, "Nonmember public artifact")
    baseline = observed.read(request["baseline"])
    candidate = observed.read(request["candidate"])
    require(baseline["manifestSha256"] == request["manifest"]["sha256"]
            and candidate["manifest"] == request["manifest"]
            and candidate["baseline"] == request["baseline"]
            and baseline["request"] == candidate["request"] == spec["request"]
            and baseline["generation"] == candidate["generation"] == spec["generation"]
            and baseline["fitArtifact"] == candidate["fitArtifact"]
            and baseline["fitKey"] == candidate["fitKey"] == FIT_KEY
            and baseline["caseID"] == candidate["caseID"] == TARGET["caseID"]
            and baseline["repID"] == candidate["repID"] == 1
            and candidate["technicalCompleted"] is True and candidate["technicalError"] is None,
            "Changed public producer-to-reader binding")
    for pin in (baseline["rawArtifact"], baseline["fitArtifact"]):
        require(pin in artifacts, "Missing committed raw/model artifact")
        observed.observe(pin["path"], pin["sha256"])
    exact(observed.read(baseline["rawArtifact"]), baseline["raw"], "actual public raw")
    readback, transport = [], []
    for pin in artifacts:
        if Path(pin["path"]).suffix != ".json":
            continue
        saved = observed.read(pin)
        if type(saved) is not dict:
            continue
        if saved.get("schema") == "h10-inference-independent-rds-v1":
            readback.append(saved)
        elif saved.get("schema") == "h10-power-source-transport-v1":
            transport.append(saved)
    require(len(readback) == len(transport) == 1, "Missing/duplicate committed independent fit evidence")
    actual, source = readback[0]["fits"][FIT_KEY], transport[0]["fits"][FIT_KEY]
    require(actual["verified"] is True and actual["requestDescriptor"] == spec["request"]
            and source["request"] == spec["request"] and source["fitArtifact"] == baseline["fitArtifact"]
            and source["exactMatrixInR"] is True and source["exactResponseInR"] is True
            and source["oneULPMatrixMutationRejected"] is True
            and transport[0]["newScientificInputs"] == transport[0]["newFitInvocations"]
            == transport[0]["newCandidateCalculations"] == 0, "Changed committed exact-source evidence")
    exact(actual["design"], source["design"], "committed same-export source design")
    original, generation = observed.read(spec["request"]), observed.read(spec["generation"])
    metadata = candidate["candidateMetadata"]
    require(metadata["requestDescriptor"] == spec["request"] and metadata["frameVerified"] is True
            and metadata["frameRowCount"] == len(original["rows"])
            and metadata["target"] == "fitted_scaled_marginal_Theta" and metadata["inverseVariance"] is True
            and metadata["cluster"] == "participant" and metadata["residual"] == "y-Xbeta"
            and metadata["priorWeights"] is False, "Changed original public candidate metadata")
    require(original["model"] == "H10" and original["reference"] == "C2", "Wrong public model")
    exact(generation["case"], spec["case"], "public generation case")
    exact(generation["populationCounts"], spec["populationCounts"], "public actual K")
    # Public specs retain recruitment information in their bound generation record.
    normalized_spec = dict(spec, recruitedCounts=generation["recruitedCounts"],
                           retentionMask=generation["retentionMask"])
    validate_population(original, normalized_spec, generation)
    return dict(manifest=manifest, manifestPin=request["manifest"], spec=normalized_spec,
                baseline=baseline, candidate=candidate, request=original, generation=generation)


def open_input(path):
    """Read an immutable binding and preserve original versus transport identities."""
    input_path = Path(path).resolve()
    value = load(input_path)
    require(value.get("schema") == INPUT_SCHEMA and value.get("kind") in
            ("historical_power", "public_power"), "Wrong saved-input schema/kind")
    locations = value.get("locations", {})
    require(type(locations) is dict and all(type(k) is str and type(v) is str
            and Path(v).is_absolute() for k, v in locations.items()), "Invalid location map")
    observed = Observations(locations)
    require(observed.resolve(input_path) == input_path, "Cannot relocate the input binding itself")
    input_pin = observed.actual(input_path)
    if value["kind"] == "historical_power":
        original = observed.read(value["historicalRequest"])
        history = admit_history(original, observed)
    else:
        history = admit_public(value, observed)
    original_history = copy.deepcopy(history)
    # The R tools consume actual local files. Keep this projection distinct from
    # the immutable original descriptors rather than rewriting archived JSON.
    source_request = history["spec"]["request"]
    source_fit = history["baseline"]["fitArtifact"]
    current_request = observed.observe(source_request["path"], source_request["sha256"])
    current_fit = observed.observe(source_fit["path"], source_fit["sha256"])
    history = copy.deepcopy(history)
    history["spec"]["request"] = current_request
    history["baseline"]["fitArtifact"] = current_fit
    history["candidate"]["candidateMetadata"]["requestDescriptor"] = current_request
    observed.fresh()
    return dict(input=value, inputPin=input_pin, observed=observed, history=history,
                originalHistory=original_history,
                transportInputs=dict(request=current_request, fitArtifact=current_fit))


def prepare_input(output, *, reproduction_root=None, historical_request=None, locations=None):
    """Bind saved artifacts only; missing inputs are an error, never a fit request."""
    require((reproduction_root is None) != (historical_request is None), "Choose one saved-input source")
    observed = Observations(locations)
    if historical_request is not None:
        value = dict(schema=INPUT_SCHEMA, kind="historical_power",
                     historicalRequest=observed.observe(Path(historical_request).resolve()),
                     locations=locations or {})
        admit_history(observed.read(value["historicalRequest"]), observed)
    else:
        root = Path(reproduction_root).resolve() / "power"
        spec = observed.observe(root / "units/00032/spec.json")
        commitment = observed.observe(root / "events/00032.json")
        committed = observed.read(commitment)
        candidates = []
        for pin in committed["artifacts"]:
            if Path(pin["path"]).suffix != ".json":
                continue
            record = observed.read(pin)
            if record.get("schema") == "h10-inference-record-v1" and record.get("fitKey") == FIT_KEY:
                candidates.append((pin, record))
        require(len(candidates) == 1, "Missing/duplicate saved candidate")
        candidate_pin, candidate = candidates[0]
        value = dict(schema=INPUT_SCHEMA, kind="public_power", target=dict(TARGET),
                     manifest=observed.observe(root / "manifest.json"), spec=spec,
                     commitment=commitment, baseline=candidate["baseline"], candidate=candidate_pin,
                     locations=locations or {})
        admit_public(value, observed)
    observed.fresh()
    write_new(Path(output).resolve(), value)
    return value


def admit_history(request, observed):
    """Authenticate one accepted member; do not rehash the old 80k observation union."""
    exact(request["target"], TARGET, "first bounded target")
    require(request["aggregate"]["sha256"] == AGGREGATE_SHA
            and request["independentAggregate"]["sha256"] == INDEPENDENT_SHA, "Unaccepted historical final")
    aggregate = observed.read(request["aggregate"])
    independent = observed.read(request["independentAggregate"])
    require(aggregate["schema"] == "h10-power-design-aggregate-v1"
            and independent["schema"] == "h10-power-design-independent-aggregate-v1"
            and independent["aggregate"] == request["aggregate"]
            and aggregate["complete"] is True and independent["complete"] is True
            and independent["passed"] is True and aggregate["sourceClosure"] is True
            and independent["sourceClosure"] is True
            and aggregate["expectedTotal"] == aggregate["presentTotal"] == aggregate["completedTotal"] == 80000
            and aggregate["missingOrdinals"] == [], "Unclosed historical final")
    manifest_pin = aggregate["manifest"]
    require(manifest_pin == independent["manifest"] and manifest_pin["sha256"] == MANIFEST_SHA,
            "Changed historical manifest")
    require(request["blockProof"] in aggregate["independentBlocks"]
            and request["blockProof"] in independent["independentBlocks"], "Nonmember block proof")
    proof = observed.read(request["blockProof"])
    require(proof["schema"] == "h10-power-design-independent-block-v1"
            and proof["passed"] is True and proof["sourceClosure"] is True
            and proof["manifest"] == manifest_pin and proof["namespace"] == NAMESPACE
            and proof["method"] == METHOD and TARGET["ordinal"] in proof["ordinals"]
            and proof["summary"] in aggregate["summaries"], "Wrong accepted block")
    summary = observed.read(proof["summary"])
    require(summary["schema"] == "h10-power-design-summary-v1"
            and summary["manifest"] == manifest_pin and summary["receipt"] == proof["receipt"]
            and summary["ordinals"] == proof["ordinals"] and summary["sourceClosure"] is True
            and summary["namespace"] == NAMESPACE and summary["method"] == METHOD, "Changed accepted summary")
    matches = [row for row in summary["cases"] if row["ordinal"] == TARGET["ordinal"]]
    require(len(matches) == 1, "Missing/duplicate actual target")
    row = matches[0]
    exact({k: row[k] for k in TARGET}, TARGET, "accepted target row")
    manifest = observed.read(manifest_pin)
    require(manifest["schema"] == "h10-power-design-manifest-v1" and manifest["namespace"] == NAMESPACE
            and manifest["method"] == METHOD and manifest["B"] == 2000
            and manifest["expectedTotal"] == 80000 and manifest["sourceClosure"] is True, "Changed manifest scope")
    case = dict(manifest["cells"][32], repID=1, retentionSeed=802940201, designSeed=902940201,
                responseSeed=1002940201)
    require(case["cellIndex"] == 33 and case["caseID"] == TARGET["caseID"]
            and case["view"] == "initial_amount" and case["populationRegime"] == "recruit193_mcar08"
            and case["scenarioIndex"] == 1 and case["recruitedPerArm"] == 193
            and case["eligibilityProbability"] == .8, "Changed initial-amount population/units")
    exact(row["case"], case, "historical case")
    spec, outer = observed.read(row["spec"]), observed.read(row["record"])
    require(spec["schema"] == "h10-power-design-spec-v1" and outer["schema"] == "h10-power-design-record-v1"
            and spec["manifest"] == outer["manifest"] == manifest_pin and outer["spec"] == row["spec"]
            and spec["namespace"] == outer["namespace"] == NAMESPACE and outer["method"] == METHOD
            and outer["sourceClosure"] is True, "Changed persisted target source")
    for obj in (spec, outer):
        exact({k: obj[k] for k in TARGET}, TARGET, "persisted target")
        require(obj["fitKey"] == FIT_KEY and obj["cellIndex"] == 33, "Changed target fit identity")
    exact(spec["case"], case, "persisted case")
    require(spec["origin"] == row["origin"] == "new_population_input"
            and spec["legacySlot"] is None and spec["legacyEvidence"] is None
            and spec["kernelFitKey"] == FIT_KEY
            and outer["baseline"] == row["baseline"] and outer["candidate"] == row["candidate"],
            "Changed target origin/stages")
    baseline, candidate = observed.read(row["baseline"]), observed.read(row["candidate"])
    original, generation = observed.read(spec["request"]), observed.read(spec["generation"])
    require(baseline["manifestSha256"] == manifest_pin["sha256"]
            and baseline["batchSha256"] == row["spec"]["sha256"]
            and baseline["request"] == candidate["request"] == spec["request"]
            and baseline["generation"] == candidate["generation"] == spec["generation"]
            and candidate["baseline"] == row["baseline"]
            and candidate["manifest"] == manifest_pin and candidate["method"] == METHOD
            and candidate["fitArtifact"] == baseline["fitArtifact"]
            and candidate["fitKey"] == baseline["fitKey"] == FIT_KEY
            and candidate["caseID"] == TARGET["caseID"] and candidate["repID"] == 1,
            "Changed stage provenance")
    require(candidate["schema"] == "h10-inference-record-v1"
            and candidate["technicalCompleted"] is True and candidate["technicalError"] is None
            and outer["technicalCompleted"] is True and row["technicalCompleted"] is True,
            "Historical candidate is not technically complete")
    exact(observed.read(baseline["rawArtifact"]), baseline["raw"], "bound baseline raw")
    exact(row["diagnostics"], baseline["raw"]["diagnostics"], "accepted guard")
    exact(row["candidateMetadata"], candidate["candidateMetadata"], "accepted candidate metadata")
    require(generation["schema"] == "h10-power-design-generation-v1"
            and generation["namespace"] == NAMESPACE and generation["execution"] == spec["execution"],
            "Changed original generation writer")
    exact(generation["case"], case, "generation case")
    for key in ("populationCounts", "recruitedCounts", "retentionMask"):
        exact(generation[key], spec[key], "saved " + key)
    exact(row["populationCounts"], spec["populationCounts"], "accepted K")
    exact(row["recruitedCounts"], spec["recruitedCounts"], "accepted recruitment")
    require(original["model"] == "H10" and original["reference"] == "C2", "Changed model/reference")
    writers = {k: actor(v["execution"]) for k, v in (
        ("input", spec), ("baseline", baseline), ("candidate", candidate), ("outer", outer))}
    return dict(manifest=manifest, manifestPin=manifest_pin, row=row, spec=spec, outer=outer,
                baseline=baseline, candidate=candidate, request=original, generation=generation,
                historicalWriters=writers, proof=proof, summary=summary)
