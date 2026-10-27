"""Saved, fixed-cell coverage diagnosis. No simulation, fitting or CI reconstruction.

Historical admission uses the original final/proof digests. Public admission
uses the existing conditional-statistics producer's manifest and fitted-event
chain; a new reproduction is never labeled historical acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

CELL = "h10pd1_initial_amount_s1_l10_recruit193_mcar08"
KEY = "C1-C2:B"
N = 2000
ARMS = ("C1", "C2", "C3")
TAILS = ("literal", "sign_consistent")
IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
NAMES = ("(Intercept)", "conditionC1", "conditionC3", "need_A", "need_B",
         "conditionC1:need_A", "conditionC3:need_A", "conditionC1:need_B", "conditionC3:need_B")
METHOD = "participant_CR2_Satterthwaite_v1"
PUBLIC = "conditional-statistics-reproduction-v1"
AGGREGATE_SHA = "408003c741653dd27108202381cb09e4e0df17cb36eb0bb0fef90bcf178a4f38"
INDEPENDENT_SHA = "2c2f4f34ec6c0c45abf8c79c103d60ec470fe0b1b58fffae3b97cb180d05c66a"
MANIFEST_SHA = "0953299b5d0afadbf84e99aec706c49eee5126b16c1cf0ab40a28038f1426e69"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite or nonnumeric scalar")
    return value


def strict_json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(token):
        raise ValueError("Nonfinite JSON: " + token)

    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid)


def exact(actual, expected):
    require(type(actual) is type(expected), "Changed copied type")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), "Changed copied fields")
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, list):
        require(len(actual) == len(expected), "Changed copied length")
        for left, right in zip(actual, expected):
            exact(left, right)
    else:
        require(actual == expected, "Changed copied value")


def near(actual, expected):
    if type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "Changed numeric shape")
        for a, b in zip(actual, expected):
            near(a, b)
    elif type(expected) is float:
        require(math.isclose(number(actual), expected, rel_tol=2e-12, abs_tol=2e-14),
                "Saved moment disagreement")
    else:
        exact(actual, expected)


def descriptor(value):
    require(type(value) is dict and set(value) == {"path", "sha256"}
            and type(value["path"]) is str and bool(value["path"])
            and type(value["sha256"]) is str and len(value["sha256"]) == 64
            and set(value["sha256"]) <= set("0123456789abcdef"), "Invalid descriptor")
    return value


class Observations:
    """Relocations affect lookup only; logical descriptors remain immutable."""

    def __init__(self, root, relocations=()):
        self.root = Path(root).absolute()
        self.locations, self.files, self.logical = {}, {}, {}
        self.add_locations(relocations, self.root)

    def add_locations(self, entries, base):
        require(type(entries) in (list, tuple), "Invalid relocations")
        used = set(self.locations.values())
        for item in entries:
            require(type(item) is dict and set(item) == {"originalPath", "localPath"},
                    "Invalid relocation fields")
            original, local = item["originalPath"], item["localPath"]
            require(type(original) is str and original and type(local) is str and local,
                    "Invalid relocation path")
            local = Path(local)
            local = local if local.is_absolute() else Path(base) / local
            require(not local.is_symlink(), "Symlink relocation")
            local = str(local.resolve())
            require(original not in self.locations and local not in used, "Duplicate relocation")
            self.locations[original] = local
            used.add(local)

    def path(self, name):
        path = Path(self.locations.get(name, name))
        return path if path.is_absolute() else self.root / path

    def bytes(self, path, expected=None):
        path = Path(path).absolute()
        require(path.is_file() and not path.is_symlink(),
                "Regular nonsymlink input required: " + str(path))
        path = path.resolve()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(expected is None or digest == expected, "Changed bound input: " + str(path))
        require(str(path) not in self.files or self.files[str(path)] == digest, "Input changed during read")
        self.files[str(path)] = digest
        return content, dict(path=str(path), sha256=digest)

    def bound(self, pin, *, json_value=True):
        descriptor(pin)
        require(pin["path"] not in self.logical or self.logical[pin["path"]] == pin["sha256"],
                "Conflicting logical digest")
        self.logical[pin["path"]] = pin["sha256"]
        content, _ = self.bytes(self.path(pin["path"]), pin["sha256"])
        return strict_json(content) if json_value else None

    def read(self, path):
        content, pin = self.bytes(path)
        return strict_json(content), pin

    def close(self):
        for path, digest in list(self.files.items()):
            self.bytes(path, digest)


def write_new(path, value, observed):
    path = Path(path).absolute()
    require(path.parent.is_dir() and not path.exists() and not path.is_symlink(), "Refusing overwrite/output parent")
    path = path.parent.resolve() / path.name
    require(str(path) not in observed.files, "Output overlaps input")
    root = Path(__file__).resolve().parents[2]
    require(not any(path.is_relative_to(root / name) for name in ("src", "scripts", "configs", "tests", "notebooks")),
            "Output overlaps source")
    observed.close()
    fd, temporary = tempfile.mkstemp(prefix=".saved-coverage-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        observed.close()
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def source_identity(observed):
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    paths = {name: package / name for name in ("saved_coverage.py", "independent_saved_coverage.py")}
    for name in ("scripts/diagnose_saved_coverage.py", "scripts/read_saved_coverage.py",
                 "src/__init__.py", "src/tutoring_statistics/__init__.py"):
        paths[name] = root / name
    return {name: observed.bytes(path)[1]["sha256"] for name, path in paths.items()}


def validate_cell(cell):
    for name, expected in dict(caseID=CELL, cellIndex=33, view="initial_amount", scenarioIndex=1,
                               populationRegime="recruit193_mcar08", recruitedPerArm=193,
                               fixedEligiblePerArm=None).items():
        exact(cell[name], expected)
    require(cell["caseID"] == CELL and cell["cellIndex"] == 33
            and cell["view"] == "initial_amount" and cell["scenarioIndex"] == 1
            and cell["multiplier"] == 1. and cell["populationRegime"] == "recruit193_mcar08"
            and cell["recruitedPerArm"] == 193 and cell["eligibilityProbability"] == .8
            and cell["fixedEligiblePerArm"] is None and set(cell["delta"]) == set(IDS)
            and all(number(v) == 0 for v in cell["delta"].values()), "Changed target estimand/truth")


def validate_case(case, rep):
    validate_cell(case)
    require(type(rep) is int and 1 <= rep <= N and case["repID"] == rep, "Changed target rep")
    for name, base in (("retentionSeed", 802610200), ("designSeed", 902610200), ("responseSeed", 1002610200)):
        exact(case[name], base + 330000 + rep)
    require(case["originalCaseIndex"] == 7 and case["role"] == "new_population_control", "Changed original cell")
    parameters = dict(alpha=60., sdU=8., sigmaByArm={arm: 12. for arm in ARMS},
                      slopes={arm: dict(A=.25, B=-.25) for arm in ARMS}, deltaMagnitude=.15)
    exact(case["parameters"], parameters)
    exact(case["trueNullIDs"], {mode: list(IDS) for mode in TAILS})


def target_cell(aggregate):
    selected = [value for value in aggregate["cases"] if value["caseID"] == CELL]
    require(len(selected) == 1, "Missing/duplicate target cell")
    cell = selected[0]
    validate_cell(cell["cell"])
    require(cell["cellIndex"] == 33 and cell["complete"] is True
            and all(type(cell[k]) is int and cell[k] == N for k in ("M", "T", "present"))
            and cell["technicalMissing"] == 0 and cell["missingRepIDs"] == []
            and cell["unresolvedRepIDs"] == [] and cell["originalGuardCounts"] == {"true": N},
            "Changed target denominator/completion")
    return cell


def counts(value):
    require(type(value) is dict and set(value) == set(ARMS)
            and all(type(n) is int and 0 < n <= 193 for n in value.values()), "Invalid retained counts")
    return value


def reference(value):
    e, se, df = [number(value[key]) for key in ("estimate", "SE", "df")]
    require(type(value["CI"]) is list and len(value["CI"]) == 2, "Missing original CI endpoints")
    lo, hi = [number(v) for v in value["CI"]]
    require(se > 0 and df > 0 and lo <= hi, "Invalid saved reference")
    return dict(estimate=e, SE=se, df=df, lower=lo, upper=hi, covered=lo <= 0 <= hi,
                belowTruth=hi < 0, aboveTruth=lo > 0, z=e/se)


def guard_components(raw):
    require(raw["status"] == "fitted", "Fitted baseline required")
    diag = raw["diagnostics"]
    for field in ("converged", "singular", "usableCandidateInference"):
        require(type(diag[field]) is bool, "Invalid original guard category")
    exact(raw["coefficientNames"], list(NAMES))
    require(set(raw["coefficients"]) == set(NAMES) and type(raw["vcov"]) is list
            and len(raw["vcov"]) == 9 and all(type(r) is list and len(r) == 9 for r in raw["vcov"]),
            "Wrong original model shape")
    finite = lambda v: type(v) in (int, float) and math.isfinite(v)
    values = lambda v: v if type(v) is list else [v]
    eigen, gradient = values(diag.get("hessianEigenvalues")), diag.get("gradient")
    components = dict(converged=diag["converged"], nonsingular=not diag["singular"],
        hessianOK=bool(eigen) and all(finite(v) for v in eigen) and min(eigen) > 0,
        gradientOK=gradient is not None and all(finite(v) for v in values(gradient)),
        finiteBeta=all(finite(v) for v in raw["coefficients"].values()),
        finiteModelV=all(finite(v) for row in raw["vcov"] for v in row))
    require(all(components.values()) == diag["usableCandidateInference"] and all(components.values()),
            "Inconsistent original full guard")
    return components


def extracted_guard(evidence, raw):
    exact(evidence["coefficientNames"], list(NAMES))
    extracted = dict(status="fitted", diagnostics=evidence["diagnostics"], coefficientNames=evidence["coefficientNames"],
                     coefficients=dict(zip(NAMES, evidence["coefficients"])), vcov=evidence["fixedVcov"])
    require(len(evidence["coefficients"]) == 9, "Wrong extracted coefficient shape")
    exact(guard_components(extracted), guard_components(raw))
    for key in ("converged", "singular", "usableCandidateInference"):
        exact(evidence["diagnostics"][key], raw["diagnostics"][key])
    for key in ("gradient", "hessianEigenvalues"):
        near(evidence["diagnostics"][key], raw["diagnostics"][key])
    near(evidence["coefficients"], [raw["coefficients"][name] for name in NAMES])
    near(evidence["fixedVcov"], raw["vcov"])


def branch_check(branches, ref, *, saved_reference):
    require(type(branches) is list and len(branches) == 2
            and [b["tailMode"] for b in branches] == list(TAILS), "Changed tail roster")
    value = reference(ref)
    for branch in branches:
        test, coverage = branch["tests"][KEY], branch["coverage"][KEY]
        require(test["status"] == "available_candidate", "Unavailable fixed target")
        if saved_reference:
            exact(test["reference"], ref)
        require(number(coverage["truth"]) == 0, "Changed truth")
        for name, expected in dict(A=1, CIAvailable=1, C=int(value["covered"]), covered=value["covered"]).items():
            exact(coverage[name], expected)
    return value


def paired_check(event, reduced):
    require(len(event["paired"]) == 2 and [p["tailMode"] for p in event["paired"]] == list(TAILS),
            "Changed paired tail roster")
    left, right = reduced["baseline"], reduced["candidate"]
    expected = dict(estimate=left["estimate"], baselineSE=left["SE"], candidateSE=right["SE"],
                    baselineDf=left["df"], candidateDf=right["df"], baselineCovered=left["covered"],
                    candidateCovered=right["covered"], baselineAvailable=True, candidateAvailable=True)
    for branch in event["paired"]:
        for name, value in expected.items():
            exact(branch["contrasts"][KEY][name], value)


def historical_rows(request, aggregate, observed):
    exact(request["aggregate"]["sha256"], AGGREGATE_SHA)
    exact(request["independent"]["sha256"], INDEPENDENT_SHA)
    proof = observed.bound(request["independent"])
    require(aggregate["schema"] == "h10-power-design-aggregate-v1"
            and aggregate["namespace"] == "h10-power-design-v1-20261020"
            and aggregate["method"] == METHOD and aggregate["complete"] is True
            and aggregate["expectedTotal"] == aggregate["completedTotal"] == 80000,
            "Changed historical aggregate")
    require(proof["schema"] == "h10-power-design-independent-aggregate-v1" and proof["method"] == METHOD
            and proof["passed"] is proof["complete"] is proof["sourceClosure"] is True
            and proof["aggregate"]["sha256"] == AGGREGATE_SHA and proof["manifest"]["sha256"] == MANIFEST_SHA
            and proof["expectedTotal"] == proof["completedTotal"] == 80000, "Changed historical proof")
    index = observed.bound(request["summaryIndex"])
    require(type(index) is list and index, "Nonempty summary-location index required")
    accepted = {}
    for pin in aggregate["summaries"]:
        descriptor(pin)
        require(pin["path"] not in accepted, "Duplicate accepted summary")
        accepted[pin["path"]] = pin
    originals, locals_seen, rows = set(), set(), {}
    index_dir = observed.path(request["summaryIndex"]["path"]).parent
    for location in index:
        require(type(location) is dict and set(location) == {"originalPath", "localPath"}, "Invalid summary index")
        original, local = location["originalPath"], location["localPath"]
        require(original in accepted and original not in originals and local not in locals_seen,
                "Unknown/duplicate relocated summary")
        originals.add(original)
        locals_seen.add(local)
        # An explicit request relocation overrides the old index's local lookup,
        # but the accepted original descriptor and digest never change.
        path = observed.path(original) if original in observed.locations else observed.path(local)
        if not Path(local).is_absolute() and original not in observed.locations and local not in observed.locations:
            path = index_dir / local
        content, _ = observed.bytes(path, accepted[original]["sha256"])
        summary = strict_json(content)
        require(summary["schema"] == "h10-power-design-summary-v1" and summary["sourceClosure"] is True
                and summary["manifest"]["sha256"] == MANIFEST_SHA and summary["method"] == METHOD,
                "Changed accepted summary identity")
        for row in summary["cases"]:
            if row["caseID"] != CELL:
                continue
            rep = row["repID"]
            require(type(rep) is int and 1 <= rep <= N and rep not in rows and row["cellIndex"] == 33
                    and row["ordinal"] == 40*(rep-1)+32 and row["technicalCompleted"] is True
                    and row["technicalError"] is None and row["diagnostics"]["usableCandidateInference"] is True,
                    "Changed target summary member")
            validate_case(row["case"], rep)
            reduced = dict(repID=rep, populationCounts=counts(row["populationCounts"]))
            for method in ("baseline", "candidate"):
                branches = row[method+"Branches"]
                reduced[method] = branch_check(branches, branches[0]["tests"][KEY]["reference"], saved_reference=True)
            exact(reduced["baseline"]["estimate"], reduced["candidate"]["estimate"])
            paired_check(row, reduced)
            rows[rep] = reduced
    require(set(rows) == set(range(1, N+1)), "Incomplete target roster; do not regenerate")
    return [rows[rep] for rep in range(1, N+1)]


def public_manifest(pin, observed):
    manifest = observed.bound(pin)
    require(manifest["schema"] == PUBLIC and manifest["family"] == "power"
            and manifest["expectedTotal"] == 80000 and manifest["identity"] == "new_public_reproduction"
            and manifest["legacyPolicy"] == "original_input_seed_semantics_not_fresh_grid_redraw",
            "Changed public manifest identity")
    for name in ("driver", "rDriver", "environment", "nativeEntry", "nativeIdentity"):
        observed.bound(manifest[name], json_value=False)
    snapshot = observed.bound(manifest["sources"])
    require(set(snapshot) == {"files"} and type(snapshot["files"]) is list, "Invalid source inventory")
    names = {}
    source_root = Path(manifest["driver"]["path"]).parent
    for source in snapshot["files"]:
        descriptor(source)
        require(source["path"] not in names, "Duplicate source inventory")
        names[source["path"]] = source
        observed.bound(source, json_value=False)
        require(Path(source["path"]).is_relative_to(source_root), "Source outside producer package")
        relative = Path(source["path"]).relative_to(source_root)
        require(".." not in relative.parts, "Invalid source-relative path")
        observed.bytes(Path(__file__).resolve().parent / relative, source["sha256"])
    for name in ("driver", "rDriver", "environment", "nativeEntry", "nativeIdentity"):
        exact(names.get(manifest[name]["path"]), manifest[name])
    for role, name in dict(driver="reproduction.py", rDriver="kernel_entry.R", environment="numerical_environment.json",
                           nativeEntry="native_inputs.py", nativeIdentity="native_identity.json").items():
        require(Path(manifest[role]["path"]).name == name, "Changed source role")
    required = {"reproduction.py", "kernel_entry.R", "mixed_fit.R", "joint_fit.R", "cluster_fit.R",
                "population_fit.R", "independent_power.py", "independent_fresh.py",
                "independent_cluster_rds.R", "source_transport.R", "joint_generation.py", "power_reference.py"}
    require(required <= {Path(p).name for p in names}, "Incomplete producer source closure")
    require(len(manifest["native"]) == 3 and {Path(p["path"]).name for p in manifest["native"]}
            == {"resolved_design.json", "initial_panel.json", "h10_canonical.json"}, "Changed native input roster")
    for item in manifest["native"]:
        observed.bound(item)
    require(manifest["numericalRuntime"]["python"] == "3.10.9", "Changed producing Python version")
    for name in ("executable", "Rscript"):
        observed.bound(manifest["numericalRuntime"][name], json_value=False)
    return manifest


def public_member(commitment, manifest_pin, observed):
    require(commitment["schema"] == PUBLIC, "Unknown event schema")
    exact(commitment["manifest"], manifest_pin)
    event = commitment["event"]
    payload = json.dumps(event, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    require(hashlib.sha256(payload).hexdigest() == commitment["eventSha256"], "Changed event digest")
    rep = event["repID"]
    require(type(rep) is int and 1 <= rep <= N and event["ordinal"] == 40*(rep-1)+32
            and event["caseID"] == CELL and event["fitKey"] == f"{CELL}__r{rep:04d}"
            and event["origin"] == "new_population_input" and event["_powerCompact"] is True
            and event["technicalCompleted"] is True and event["technicalError"] is None
            and event["diagnostics"]["usableCandidateInference"] is True, "Changed public target member")
    exact(event["ordinal"], 40*(rep-1)+32)
    pins = commitment["artifacts"]
    require(type(pins) is list and len(pins) >= 9, "Missing fitted-event artifacts")
    require(len({descriptor(p)["path"] for p in pins}) == len(pins), "Duplicate artifact identity")
    # This is the actual joint_member emission order, not a replacement schema.
    spec, actual, generation, baseline, candidate = [observed.bound(p) for p in pins[:5]]
    for p in pins[5:]:
        observed.bound(p, json_value=False)
    validate_case(spec["case"], rep)
    exact(spec["manifest"], manifest_pin)
    require(spec["origin"] is None, "Changed target origin")
    for name in ("ordinal", "caseID", "repID", "fitKey"):
        exact(spec[name], event[name])
    exact(spec["request"], pins[1])
    exact(spec["generation"], pins[2])
    exact(candidate["baseline"], pins[3])
    require(baseline["schema"] == "h10-calibration-production-fit-v1"
            and candidate["schema"] == "h10-inference-record-v1" and candidate["method"] == METHOD
            and baseline["manifestSha256"] == manifest_pin["sha256"]
            and candidate["technicalCompleted"] is True and candidate["technicalError"] is None,
            "Changed fitted-record identity")
    exact(candidate["manifest"], manifest_pin)
    for name in ("newScientificInputs", "newFitInvocations"):
        exact(candidate[name], 0)
    for record in (baseline, candidate):
        for name in ("caseID", "repID", "fitKey", "request", "generation"):
            exact(record[name], spec[name])
    exact(candidate["fitArtifact"], baseline["fitArtifact"])
    for name in ("fitArtifact", "rawArtifact"):
        require(baseline[name] in pins[5:], "Missing committed raw/fit artifact")
    exact(observed.bound(baseline["rawArtifact"]), baseline["raw"])
    exact(spec["populationCounts"], counts(event["populationCounts"]))
    require(actual["model"] == "H10" and actual["reference"] == "C2", "Changed request model")
    exact(generation["case"], spec["case"])
    require(generation["schema"] == "h10-power-design-generation-v1"
            and generation["namespace"] == "h10-power-design-v1-20261020"
            and generation["responseTransform"] == "none"
            and generation["eligibility"] == "independent_whole_person_MCAR_complete_ten_not_trial_dropout",
            "Changed generation identity")
    exact(generation["populationCounts"], spec["populationCounts"])
    exact(generation["execution"], spec["execution"])
    exact(generation["recruitedCounts"], {arm: 193 for arm in ARMS})
    people = {}
    for row in actual["rows"]:
        require(set(row) == {"participant", "condition", "need_A", "need_B", "y"}
                and type(row["participant"]) is str and row["condition"] in ARMS, "Invalid request row")
        for key in ("need_A", "need_B", "y"):
            number(row[key])
        arm, n = people.get(row["participant"], (row["condition"], 0))
        require(arm == row["condition"], "Participant crosses arm")
        people[row["participant"]] = (arm, n+1)
    require(all(n == 10 for _, n in people.values()), "Incomplete participant")
    exact({arm: sum(a == arm for a, _ in people.values()) for arm in ARMS}, spec["populationCounts"])
    exact(list(people), generation["retainedParticipantIDs"])
    selected = []
    recruited = generation["kernelGeneration"]["people"]
    exact([p["participant"] for p in recruited], generation["recruitedParticipantIDs"])
    require(len({p["participant"] for p in recruited}) == 579, "Duplicate/missing recruited identity")
    for arm in ARMS:
        mask = generation["retentionMask"][arm]
        own = [p["participant"] for p in generation["kernelGeneration"]["people"] if p["condition"] == arm]
        require(len(mask) == len(own) == 193 and all(type(v) is bool for v in mask), "Invalid retention mask")
        selected.extend(p for p, keep in zip(own, mask) if keep)
    exact(selected, generation["retainedParticipantIDs"])
    exact(generation["retainedRowIndices"], [10*i+j for i, p in enumerate(recruited)
                                           if p["participant"] in set(selected) for j in range(10)])
    extras = [observed.bound(p) for p in pins[5:] if p != baseline["fitArtifact"]]
    extracts = [v for v in extras if v.get("schema") == "h10-inference-independent-rds-v1"]
    transports = [v for v in extras if v.get("schema") == "h10-power-source-transport-v1"]
    require(len(extracts) == len(transports) == 1, "Missing/duplicate fitted extraction or transport")
    evidence, transport = extracts[0]["fits"][spec["fitKey"]], transports[0]
    exact(evidence["fitKey"], spec["fitKey"])
    exact(evidence["requestDescriptor"], spec["request"])
    require(evidence["verified"] is True and evidence["frameRowCount"] == evidence["requestRowCount"] == len(actual["rows"]),
            "Unverified extracted frame")
    extracted_guard(evidence, baseline["raw"])
    exact(evidence["design"]["participant"], [r["participant"] for r in actual["rows"]])
    exact(evidence["design"]["condition"], [r["condition"] for r in actual["rows"]])
    require(set(evidence["design"]) == {"X", "y", "participant", "condition"}
            and len(evidence["design"]["X"]) == len(evidence["design"]["y"]) == len(actual["rows"])
            and all(len(row) == 9 for row in evidence["design"]["X"]), "Changed extracted design dimensions")
    for row in evidence["design"]["X"]:
        for value in row:
            number(value)
    for value in evidence["design"]["y"]:
        number(value)
    for name in ("newScientificInputs", "newFitInvocations", "newCandidateCalculations"):
        exact(transport[name], 0)
    transported = transport["fits"][spec["fitKey"]]
    exact(transported["request"], spec["request"])
    exact(transported["fitArtifact"], baseline["fitArtifact"])
    for name in ("exactMatrixInR", "exactResponseInR", "oneULPMatrixMutationRejected"):
        exact(transported[name], True)
    exact(transported["design"], evidence["design"])
    metadata = candidate["candidateMetadata"]
    require(metadata["frameVerified"] is True and metadata["frameRowCount"] == len(actual["rows"]),
            "Unverified candidate frame")
    require(metadata["priorWeights"] is False and metadata["cluster"] == "participant"
            and metadata["inverseVariance"] is True and metadata["target"] == "fitted_scaled_marginal_Theta"
            and metadata["residual"] == "y-Xbeta", "Changed candidate model interpretation")
    exact(metadata["requestDescriptor"], spec["request"])
    environment = observed.bound(observed.bound(manifest_pin)["environment"])["statistical_reference"]
    exact(metadata["runtime"]["R"], environment["R"])
    for name, version in environment["R_packages"].items():
        exact(metadata["runtime"][name], version)
    for runtime, packages in ((extracts[0]["runtime"], ("lme4", "lmerTest")), (transport["runtime"], ("jsonlite",))):
        exact(runtime["R"], environment["R"])
        for name in packages:
            expected = environment["R_packages"][name]
            alias = {("lme4", "1.1-35.5"): "1.1.35.5", ("lmerTest", "3.1-3"): "3.1.3"}.get((name, expected), expected)
            require(type(runtime[name]) is str and runtime[name] in (expected, alias), "Changed extracted package version")
    exact(candidate["rawCandidate"]["diagnostics"], baseline["raw"]["diagnostics"])
    exact(candidate["rawCandidate"]["method"], METHOD)
    guard_components(candidate["rawCandidate"])
    near(metadata["modelBasedVcov"], evidence["fixedVcov"])
    near([candidate["rawCandidate"]["coefficients"][name] for name in NAMES], evidence["coefficients"])
    reduced = dict(repID=rep, populationCounts=spec["populationCounts"])
    for method, raw in (("baseline", baseline["raw"]), ("candidate", candidate["rawCandidate"])):
        require(raw["status"] == "fitted" and raw["diagnostics"]["usableCandidateInference"] is True,
                "Unavailable fitted reference")
        reduced[method] = branch_check(event[method+"Branches"], raw["references"][KEY], saved_reference=False)
    exact(reduced["baseline"]["estimate"], reduced["candidate"]["estimate"])
    paired_check(event, reduced)
    return reduced


def admit(request, observed):
    require(request["schema"] == "saved-coverage-input-v1", "Unknown input schema")
    mode = request["mode"]
    expected = {"schema", "mode", "aggregate", "relocations"}
    expected |= {"independent", "summaryIndex"} if mode == "historical_power" else {"events"}
    require(mode in ("historical_power", "public_power") and set(request) == expected, "Unknown mode/fields")
    aggregate = observed.bound(request["aggregate"])
    cell = target_cell(aggregate)
    if mode == "historical_power":
        rows = historical_rows(request, aggregate, observed)
        for method, successes in (("baseline", 1881), ("candidate", 1879)):
            exact(cell[method]["coverage"][KEY]["counts"], dict(A=N, CIAvailable=N, C=successes))
    else:
        require(aggregate["schema"] == PUBLIC and aggregate["family"] == "power" and aggregate["B"] == N
                and aggregate["expectedTotal"] == 80000 and type(aggregate["fullRosterComplete"]) is bool
                and type(aggregate["presentTotal"]) is int and N <= aggregate["presentTotal"] <= 80000
                and aggregate["fullRosterComplete"] == (aggregate["presentTotal"] == 80000)
                and aggregate["status"] == ("full_reproduction" if aggregate["fullRosterComplete"] else "partial_entry_check")
                and aggregate["independentNumericalReadback"] is True, "Changed public aggregate scope")
        public_manifest(aggregate["reproductionManifest"], observed)
        require(type(request["events"]) is list and len(request["events"]) == N, "Incomplete target roster")
        require(len({descriptor(p)["path"] for p in request["events"]}) == N, "Duplicate event descriptor")
        own = {}
        for pin in request["events"]:
            row = public_member(observed.bound(pin), aggregate["reproductionManifest"], observed)
            require(row["repID"] not in own, "Duplicate target rep")
            own[row["repID"]] = row
        require(set(own) == set(range(1, N+1)), "Incomplete target roster")
        rows = [own[rep] for rep in range(1, N+1)]
    check_reduction(rows, cell)
    return cell, rows


def moments(values):
    for value in values:
        number(value)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sampleSD=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values) if values else None, max=max(values) if values else None)


def aggregate_diagnosis(cell):
    diag = cell["contrastDiagnostics"][KEY]
    for value in diag.values():
        require(value["n"] == N and value["missing"] == 0, "Missing target moments")
    mean, sd = number(diag["estimate"]["mean"]), number(diag["estimate"]["SD"])
    require(sd > 0, "Nonpositive empirical SD")
    mcse = sd / math.sqrt(N)
    result = dict(n=N, truth=0, biasEstimate=mean, empiricalSD=sd, biasMonteCarloSE=mcse,
                  biasNormalMCInterval95=[mean-1.959963984540054*mcse, mean+1.959963984540054*mcse],
                  intervalScope="normal_MC_approximation_for_mean_not_coverage_CI_or_causal_test")
    for name in ("baselineSE", "candidateSE"):
        m, s = number(diag[name]["mean"]), number(diag[name]["SD"])
        require(m > 0 and s >= 0, "Invalid SE moments")
        rms = math.sqrt(m*m + (N-1)/N*s*s)
        result[name] = dict(mean=m, sampleSD=s, RMS=rms, RMSOverEmpiricalSD=rms/sd)
    result.update(baselineCoverage=cell["baseline"]["coverage"][KEY],
                  candidateCoverage=cell["candidate"]["coverage"][KEY],
                  interpretation="descriptive_saved_moments_no_method_adoption_no_unique_cause")
    return result


def check_reduction(rows, cell):
    require(len(rows) == N and [r["repID"] for r in rows] == list(range(1, N+1)), "Incomplete target roster")
    for method in ("baseline", "candidate"):
        values = [r[method] for r in rows]
        count = sum(v["covered"] for v in values)
        coverage = cell[method]["coverage"][KEY]
        exact(coverage["counts"], dict(A=N, CIAvailable=N, C=count))
        exact(coverage["ciUnavailableAmongAvailable"], 0)
        for name, numerator in (("coverageAll", count), ("coverageAvailable", count), ("availabilityCompleted", N)):
            rate = coverage[name]
            exact(rate["numerator"], numerator)
            exact(rate["denominator"], N)
            near(rate["estimate"], numerator/N)
        for field, metric in (("SE", method+"SE"), ("df", method+"Df"), ("estimate", "estimate")):
            saved = cell["contrastDiagnostics"][KEY][metric]
            calculated = moments([v[field] for v in values])
            exact(saved["n"], N)
            exact(saved["missing"], 0)
            for source, target in (("mean", "mean"), ("SD", "sampleSD"), ("min", "min"), ("max", "max")):
                near(saved[source], calculated[target])
    exact([v["tailMode"] for v in cell["paired"]], list(TAILS))
    for item in cell["paired"]:
        exact(item["denominator"], N)
        table = {key: 0 for key in ("00", "01", "10", "11", "unknown")}
        for row in rows:
            table[str(int(row["baseline"]["covered"]))+str(int(row["candidate"]["covered"]))] += 1
        exact(item["contrasts"][KEY]["coverage"], table)


def record_diagnosis(rows):
    require(len(rows) == N, "Incomplete target roster")
    output = dict(n=N, methods={}, commonMisses=sum(not r["baseline"]["covered"] and not r["candidate"]["covered"] for r in rows))
    for method in ("baseline", "candidate"):
        values = [row[method] for row in rows]
        count = sum(v["covered"] for v in values)
        output["methods"][method] = dict(denominators=dict(M=N, T=N, A=N, CIAvailable=N), covered=count,
            belowTruth=sum(v["belowTruth"] for v in values), aboveTruth=sum(v["aboveTruth"] for v in values),
            estimate=moments([v["estimate"] for v in values]), standardizedError=moments([v["z"] for v in values]),
            RMSSE=math.sqrt(statistics.mean(v["SE"]**2 for v in values)))
    output["retentionByCoverage"] = {}
    for covered in (True, False):
        group = [r for r in rows if r["candidate"]["covered"] is covered]
        output["retentionByCoverage"][str(covered)] = {
            **{arm: moments([r["populationCounts"][arm] for r in group]) for arm in ARMS},
            "absoluteC1C2Imbalance": moments([abs(r["populationCounts"]["C1"]-r["populationCounts"]["C2"]) for r in group])}
    output["interpretation"] = "descriptive_association_not_retention_causality"
    return output


def diagnose(request_path, output):
    observed = Observations(Path(request_path).absolute().parent)
    request, request_pin = observed.read(request_path)
    observed.add_locations(request["relocations"], observed.root)
    sources = source_identity(observed)
    cell, rows = admit(request, observed)
    result = dict(schema="saved-coverage-diagnosis-v1", mode=request["mode"], cellID=CELL, contrast=KEY,
        input=request_pin, sources=sources, aggregateOnly=aggregate_diagnosis(cell),
        records=record_diagnosis(rows), targetScalars=rows, newFits=0, newDraws=0, RDSDeserializations=0,
        scope="complete_fixed_target_not_full_population_acceptance", sourceClosure=True)
    result["historicalReference"] = dict(aggregateSHA256=AGGREGATE_SHA, independentSHA256=INDEPENDENT_SHA,
        baselineCovered=1881, candidateCovered=1879, denominator=2000,
        authenticatedThisInvocation=request["mode"] == "historical_power",
        interpretation="reported_coverage_0.9395_applicability_warning_not_a_new_result_target_or_unique_cause")
    result["observedInputs"] = dict(observed.files)
    write_new(output, result, observed)
    return result


def locate_events(directory, observed):
    """Locate the fixed target only; omission never triggers production."""
    return [observed.bytes(Path(directory).absolute() / f"{40*(rep-1)+32:05d}.json")[1]
            for rep in range(1, N+1)]


def prepare(mode, aggregate, output, *, independent=None, summary_index=None, events_directory=None, relocations=None):
    observed = Observations(Path(output).absolute().parent)
    locations = []
    if relocations:
        entries, _ = observed.read(relocations)
        observed.add_locations(entries, Path(relocations).absolute().parent)
        locations = [dict(originalPath=k, localPath=v) for k, v in observed.locations.items()]
    aggregate_pin = observed.bytes(Path(aggregate).absolute())[1]
    request = dict(schema="saved-coverage-input-v1", mode=mode, aggregate=aggregate_pin, relocations=locations)
    if mode == "public_power":
        require(events_directory is not None, "Events directory required")
        request["events"] = locate_events(events_directory, observed)
    else:
        require(mode == "historical_power" and independent is not None and summary_index is not None, "Historical inputs required")
        request.update(independent=observed.bytes(Path(independent).absolute())[1],
                       summaryIndex=observed.bytes(Path(summary_index).absolute())[1])
    source_identity(observed)
    admit(request, observed)
    write_new(output, request, observed)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-public", "prepare-historical"):
        p = commands.add_parser(name, help="Bind existing inputs only; never generate missing data")
        p.add_argument("--aggregate", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--relocations")
        if name == "prepare-public":
            p.add_argument("--events-directory", required=True)
        else:
            p.add_argument("--independent", required=True)
            p.add_argument("--summary-index", required=True)
    p = commands.add_parser("diagnose")
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "diagnose":
        diagnose(args.request, args.output)
    else:
        prepare("public_power" if args.command == "prepare-public" else "historical_power",
                args.aggregate, args.output, independent=getattr(args, "independent", None),
                summary_index=getattr(args, "summary_index", None), events_directory=getattr(args, "events_directory", None),
                relocations=args.relocations)


if __name__ == "__main__":
    main()
