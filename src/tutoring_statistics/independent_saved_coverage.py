"""Independent saved-coverage admission and reduction, using standard library only.

This reader deliberately imports no producer, shared admission helper, model,
random generator or inference implementation. Original intervals are inputs,
not recalculated intervals. Saved RDS objects are byte-bound, not deserialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
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


def require(value, message):
    if not value:
        raise ValueError(message)


def scalar(value, name="scalar"):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid " + name)
    return value


def strict_json(data):
    def unique(items):
        result = {}
        for name, value in items:
            require(name not in result, "Duplicate JSON key")
            result[name] = value
        return result

    def reject(value):
        raise ValueError("Nonfinite JSON " + value)

    return json.loads(data, object_pairs_hook=unique, parse_constant=reject)


def copied(actual, expected, label="copied"):
    require(type(actual) is type(expected), "Changed copied type: " + label)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), "Changed copied keys: " + label)
        for name in expected:
            copied(actual[name], expected[name], label+"."+name)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), "Changed copied length: " + label)
        for i, value in enumerate(expected):
            copied(actual[i], value, label+"["+str(i)+"]")
    else:
        require(actual == expected, "Changed copied value: " + label)


def compare(actual, expected, label="calculated"):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), "Changed result fields: " + label)
        for name, value in expected.items():
            compare(actual[name], value, label+"."+name)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "Changed result list: " + label)
        for i, value in enumerate(expected):
            compare(actual[i], value, label+"["+str(i)+"]")
    elif type(expected) is float:
        require(math.isclose(scalar(actual, label), expected, rel_tol=2e-12, abs_tol=2e-14),
                "Independent numeric disagreement: " + label)
    else:
        copied(actual, expected, label)


class Readback:
    def __init__(self, request_path):
        self.root = Path(request_path).absolute().parent
        self.files, self.logical, self.locations = {}, {}, {}
        self.request, self.request_pin = self.read(request_path)
        locations = self.request["relocations"]
        require(type(locations) is list, "Relocation list required")
        seen = set()
        for item in locations:
            require(type(item) is dict and set(item) == {"originalPath", "localPath"}, "Changed relocation fields")
            old, new = item["originalPath"], item["localPath"]
            require(type(old) is str and old and type(new) is str and new, "Invalid relocation path")
            target = Path(new)
            target = target if target.is_absolute() else self.root / target
            require(not target.is_symlink(), "Symlink relocation")
            target = str(target.resolve())
            require(old not in self.locations and target not in seen, "Duplicate relocation")
            self.locations[old] = target
            seen.add(target)

    def locate(self, path):
        value = Path(self.locations.get(path, path))
        return value if value.is_absolute() else self.root / value

    def bytes(self, path, expected=None):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), "Regular nonsymlink input required")
        path = path.resolve()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(expected is None or digest == expected, "Changed bound input: " + str(path))
        require(str(path) not in self.files or self.files[str(path)] == digest, "Changed input during readback")
        self.files[str(path)] = digest
        return content, dict(path=str(path), sha256=digest)

    def read(self, path):
        content, pin = self.bytes(path)
        return strict_json(content), pin

    def bound(self, pin, *, binary=False):
        require(type(pin) is dict and set(pin) == {"path", "sha256"}
                and type(pin["path"]) is str and pin["path"]
                and type(pin["sha256"]) is str and len(pin["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in pin["sha256"]), "Invalid descriptor")
        require(pin["path"] not in self.logical or self.logical[pin["path"]] == pin["sha256"],
                "Conflicting descriptor digest")
        self.logical[pin["path"]] = pin["sha256"]
        content, _ = self.bytes(self.locate(pin["path"]), pin["sha256"])
        return None if binary else strict_json(content)

    def close(self):
        for name, digest in tuple(self.files.items()):
            self.bytes(name, digest)

    def sources(self):
        result = {}
        package = Path(__file__).resolve().parent
        for name in ("saved_coverage.py", "independent_saved_coverage.py"):
            result[name] = self.bytes(package / name)[1]["sha256"]
        for name in ("scripts/diagnose_saved_coverage.py", "scripts/read_saved_coverage.py",
                     "src/__init__.py", "src/tutoring_statistics/__init__.py"):
            result[name] = self.bytes(package.parents[1] / name)[1]["sha256"]
        return result


def cell_identity(value):
    required = dict(caseID=CELL, cellIndex=33, view="initial_amount", scenarioIndex=1,
                    populationRegime="recruit193_mcar08", recruitedPerArm=193,
                    fixedEligiblePerArm=None)
    for name, expected in required.items():
        copied(value[name], expected)
    require(scalar(value["multiplier"]) == 1 and scalar(value["eligibilityProbability"]) == .8
            and set(value["delta"]) == set(IDS) and all(scalar(x) == 0 for x in value["delta"].values()),
            "Wrong target truth/coordinates")


def case_identity(case, rep):
    cell_identity(case)
    require(type(rep) is int and 1 <= rep <= N, "Invalid target rep")
    copied(case["repID"], rep)
    for name, base in (("retentionSeed", 802940200), ("designSeed", 902940200), ("responseSeed", 1002940200)):
        copied(case[name], base+rep)
    copied(case["originalCaseIndex"], 7)
    copied(case["role"], "new_population_control")
    copied(case["parameters"], dict(alpha=60., sdU=8., sigmaByArm=dict(C1=12., C2=12., C3=12.),
        slopes=dict(C1=dict(A=.25, B=-.25), C2=dict(A=.25, B=-.25), C3=dict(A=.25, B=-.25)), deltaMagnitude=.15))
    copied(case["trueNullIDs"], {tail: list(IDS) for tail in TAILS})


def selected_cell(aggregate):
    matches = [c for c in aggregate["cases"] if c["caseID"] == CELL]
    require(len(matches) == 1, "Missing/duplicate target cell")
    cell = matches[0]
    cell_identity(cell["cell"])
    copied(cell["cellIndex"], 33)
    for field in ("M", "T", "present"):
        copied(cell[field], N)
    copied(cell["complete"], True)
    copied(cell["technicalMissing"], 0)
    copied(cell["missingRepIDs"], [])
    copied(cell["unresolvedRepIDs"], [])
    copied(cell["originalGuardCounts"], {"true": N})
    return cell


def retained_counts(value):
    require(type(value) is dict and set(value) == set(ARMS)
            and all(type(n) is int and 1 <= n <= 193 for n in value.values()), "Invalid retained counts")
    return value


def interval_values(ref):
    center, error, degrees = (scalar(ref[name], name) for name in ("estimate", "SE", "df"))
    require(type(ref["CI"]) is list and len(ref["CI"]) == 2, "Missing original CI endpoints")
    lower, upper = (scalar(x, "CI endpoint") for x in ref["CI"])
    require(error > 0 and degrees > 0 and lower <= upper, "Invalid saved interval")
    return dict(estimate=center, SE=error, df=degrees, lower=lower, upper=upper,
                covered=lower <= 0 <= upper, belowTruth=upper < 0, aboveTruth=lower > 0, z=center/error)


def check_full_guard(diag, names, coefficients, covariance):
    copied(names, list(NAMES))
    require(type(coefficients) is list and len(coefficients) == 9
            and type(covariance) is list and len(covariance) == 9
            and all(type(row) is list and len(row) == 9 for row in covariance), "Wrong guard model shape")
    for name in ("converged", "singular", "usableCandidateInference"):
        require(type(diag[name]) is bool, "Wrong guard flag type")
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
    eigen = diag.get("hessianEigenvalues")
    eigen = eigen if type(eigen) is list else [eigen]
    grad = diag.get("gradient")
    gradient = grad if type(grad) is list else [grad]
    checks = [diag["converged"], not diag["singular"], bool(eigen) and all(finite(x) for x in eigen) and min(eigen) > 0,
              grad is not None and all(finite(x) for x in gradient), all(finite(x) for x in coefficients),
              all(finite(x) for row in covariance for x in row)]
    require(all(checks) == diag["usableCandidateInference"] and all(checks), "Inconsistent original full guard")


def validate_guards(raw, candidate, extraction):
    require(set(raw["coefficients"]) == set(candidate["coefficients"]) == set(NAMES), "Wrong coefficient identity")
    beta = [raw["coefficients"][name] for name in NAMES]
    check_full_guard(raw["diagnostics"], raw["coefficientNames"], beta, raw["vcov"])
    check_full_guard(candidate["diagnostics"], candidate["coefficientNames"],
                     [candidate["coefficients"][name] for name in NAMES], candidate["vcov"])
    check_full_guard(extraction["diagnostics"], extraction["coefficientNames"], extraction["coefficients"], extraction["fixedVcov"])
    copied(candidate["diagnostics"], raw["diagnostics"])
    for key in ("converged", "singular", "usableCandidateInference"):
        copied(extraction["diagnostics"][key], raw["diagnostics"][key])
    for key in ("gradient", "hessianEigenvalues"):
        compare(extraction["diagnostics"][key], raw["diagnostics"][key])
    compare(extraction["coefficients"], beta)
    compare(extraction["fixedVcov"], raw["vcov"])
    compare([candidate["coefficients"][name] for name in NAMES], extraction["coefficients"])


def validate_branches(branches, ref, historical):
    copied([b["tailMode"] for b in branches], list(TAILS))
    calculated = interval_values(ref)
    for branch in branches:
        copied(branch["tests"][KEY]["status"], "available_candidate")
        if historical:
            copied(branch["tests"][KEY]["reference"], ref)
        event = branch["coverage"][KEY]
        require(scalar(event["truth"]) == 0, "Changed truth")
        copied(event["A"], 1)
        copied(event["CIAvailable"], 1)
        copied(event["covered"], calculated["covered"])
        copied(event["C"], int(calculated["covered"]))
    return calculated


def validate_pair(row, values):
    copied(values["baseline"]["estimate"], values["candidate"]["estimate"])
    copied([b["tailMode"] for b in row["paired"]], list(TAILS))
    for tail in row["paired"]:
        saved = tail["contrasts"][KEY]
        copied(saved["estimate"], values["baseline"]["estimate"])
        for method in ("baseline", "candidate"):
            for suffix, field in (("SE", "SE"), ("Df", "df"), ("Covered", "covered")):
                copied(saved[method+suffix], values[method][field])
            copied(saved[method+"Available"], True)


def read_historical(ctx, aggregate):
    request = ctx.request
    copied(request["aggregate"]["sha256"], AGGREGATE_SHA)
    copied(request["independent"]["sha256"], INDEPENDENT_SHA)
    proof = ctx.bound(request["independent"])
    require(aggregate["schema"] == "h10-power-design-aggregate-v1" and aggregate["method"] == METHOD
            and aggregate["namespace"] == "h10-power-design-v1-20261020" and aggregate["complete"] is True
            and aggregate["expectedTotal"] == aggregate["completedTotal"] == 80000, "Changed historical aggregate")
    require(proof["schema"] == "h10-power-design-independent-aggregate-v1" and proof["method"] == METHOD
            and proof["passed"] is True and proof["sourceClosure"] is True and proof["complete"] is True
            and proof["aggregate"]["sha256"] == AGGREGATE_SHA and proof["manifest"]["sha256"] == MANIFEST_SHA
            and proof["completedTotal"] == proof["expectedTotal"] == 80000, "Changed original acceptance")
    locations = ctx.bound(request["summaryIndex"])
    require(type(locations) is list and locations, "Nonempty relocation index required")
    accepted = {}
    for pin in aggregate["summaries"]:
        require(set(pin) == {"path", "sha256"} and pin["path"] not in accepted, "Invalid accepted summary roster")
        accepted[pin["path"]] = pin["sha256"]
    seen, local_seen, members = set(), set(), {}
    for location in locations:
        require(type(location) is dict and set(location) == {"originalPath", "localPath"}, "Invalid summary location")
        old, local = location["originalPath"], location["localPath"]
        require(old in accepted and old not in seen and local not in local_seen, "Unknown/duplicate relocated summary")
        seen.add(old)
        local_seen.add(local)
        if old in ctx.locations:
            path = ctx.locate(old)
        elif local in ctx.locations or Path(local).is_absolute():
            path = ctx.locate(local)
        else:
            path = ctx.locate(request["summaryIndex"]["path"]).parent / local
        content, _ = ctx.bytes(path, accepted[old])
        summary = strict_json(content)
        require(summary["schema"] == "h10-power-design-summary-v1" and summary["sourceClosure"] is True
                and summary["method"] == METHOD and summary["manifest"]["sha256"] == MANIFEST_SHA,
                "Changed original summary identity")
        for row in summary["cases"]:
            if row["caseID"] != CELL:
                continue
            rep = row["repID"]
            require(type(rep) is int and 1 <= rep <= N and rep not in members and row["cellIndex"] == 33
                    and row["ordinal"] == 40*(rep-1)+32 and row["technicalCompleted"] is True
                    and row["technicalError"] is None and row["diagnostics"]["usableCandidateInference"] is True,
                    "Changed target summary member")
            case_identity(row["case"], rep)
            values = dict(repID=rep, populationCounts=retained_counts(row["populationCounts"]))
            for method in ("baseline", "candidate"):
                branches = row[method+"Branches"]
                values[method] = validate_branches(branches, branches[0]["tests"][KEY]["reference"], True)
            validate_pair(row, values)
            members[rep] = values
    require(set(members) == set(range(1, N+1)), "Incomplete target roster; regeneration forbidden")
    return [members[i] for i in range(1, N+1)]


def read_manifest(ctx, pin):
    value = ctx.bound(pin)
    require(value["schema"] == PUBLIC and value["family"] == "power" and value["expectedTotal"] == 80000
            and value["identity"] == "new_public_reproduction"
            and value["legacyPolicy"] == "original_input_seed_semantics_not_fresh_grid_redraw", "Invalid public manifest")
    sources = ctx.bound(value["sources"])
    require(type(sources) is dict and set(sources) == {"files"} and type(sources["files"]) is list, "Invalid source snapshot")
    roster = {}
    source_root = Path(value["driver"]["path"]).parent
    for pin in sources["files"]:
        require(pin["path"] not in roster, "Duplicate source")
        ctx.bound(pin, binary=True)
        roster[pin["path"]] = pin
        require(Path(pin["path"]).is_relative_to(source_root), "Source outside producer package")
        relative = Path(pin["path"]).relative_to(source_root)
        require(".." not in relative.parts, "Invalid relative source")
        ctx.bytes(Path(__file__).resolve().parent / relative, pin["sha256"])
    for name in ("driver", "rDriver", "environment", "nativeEntry", "nativeIdentity"):
        ctx.bound(value[name], binary=True)
        copied(roster.get(value[name]["path"]), value[name])
    roles = dict(driver="reproduction.py", rDriver="kernel_entry.R", environment="numerical_environment.json",
                 nativeEntry="native_inputs.py", nativeIdentity="native_identity.json")
    for role, name in roles.items():
        copied(Path(value[role]["path"]).name, name)
    needed = ("reproduction.py", "kernel_entry.R", "mixed_fit.R", "joint_fit.R", "cluster_fit.R", "population_fit.R",
              "independent_power.py", "independent_fresh.py", "independent_cluster_rds.R", "source_transport.R",
              "joint_generation.py", "power_reference.py")
    require(set(needed) <= {Path(p).name for p in roster}, "Missing producer source roles")
    require(len(value["native"]) == 3 and {Path(p["path"]).name for p in value["native"]}
            == {"resolved_design.json", "initial_panel.json", "h10_canonical.json"}, "Invalid native roster")
    for pin in value["native"]:
        ctx.bound(pin)
    require(value["numericalRuntime"]["python"] == "3.10.9", "Wrong producing Python version")
    ctx.bound(value["numericalRuntime"]["executable"], binary=True)
    ctx.bound(value["numericalRuntime"]["Rscript"], binary=True)
    return value


def read_public_member(ctx, commitment, manifest):
    require(commitment["schema"] == PUBLIC, "Foreign event schema")
    copied(commitment["manifest"], manifest)
    event = commitment["event"]
    raw_event = json.dumps(event, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    copied(hashlib.sha256(raw_event).hexdigest(), commitment["eventSha256"])
    rep = event["repID"]
    require(type(rep) is int and 1 <= rep <= N, "Invalid public rep")
    copied(event["ordinal"], 40*(rep-1)+32)
    copied(event["caseID"], CELL)
    copied(event["fitKey"], f"{CELL}__r{rep:04d}")
    copied(event["origin"], "new_population_input")
    copied(event["_powerCompact"], True)
    require(event["technicalCompleted"] is True and event["technicalError"] is None
            and event["diagnostics"]["usableCandidateInference"] is True, "Incomplete/unguarded event")
    pins = commitment["artifacts"]
    require(type(pins) is list and len(pins) >= 9 and len({p["path"] for p in pins}) == len(pins),
            "Missing/duplicate fitted-event artifacts")
    # Fixed public emission roles: spec (no schema), request, generation, old, CR2.
    spec = ctx.bound(pins[0])
    request = ctx.bound(pins[1])
    generation = ctx.bound(pins[2])
    old = ctx.bound(pins[3])
    candidate = ctx.bound(pins[4])
    case_identity(spec["case"], rep)
    copied(spec["manifest"], manifest)
    copied(spec["origin"], None)
    for field in ("ordinal", "caseID", "repID", "fitKey"):
        copied(spec[field], event[field])
    copied(spec["request"], pins[1])
    copied(spec["generation"], pins[2])
    copied(candidate["baseline"], pins[3])
    require(old["schema"] == "h10-calibration-production-fit-v1" and old["manifestSha256"] == manifest["sha256"]
            and candidate["schema"] == "h10-inference-record-v1" and candidate["method"] == METHOD
            and candidate["technicalCompleted"] is True and candidate["technicalError"] is None, "Changed fitted records")
    copied(candidate["manifest"], manifest)
    copied(candidate["newScientificInputs"], 0)
    copied(candidate["newFitInvocations"], 0)
    for record in (old, candidate):
        for field in ("caseID", "repID", "fitKey", "request", "generation"):
            copied(record[field], spec[field])
    copied(candidate["fitArtifact"], old["fitArtifact"])
    require(old["rawArtifact"] in pins[5:] and old["fitArtifact"] in pins[5:], "Missing committed raw/fit")
    ctx.bound(old["fitArtifact"], binary=True)
    copied(ctx.bound(old["rawArtifact"]), old["raw"])
    copied(spec["populationCounts"], retained_counts(event["populationCounts"]))
    require(request["model"] == "H10" and request["reference"] == "C2", "Changed source request")
    copied(generation["case"], spec["case"])
    require(generation["schema"] == "h10-power-design-generation-v1"
            and generation["namespace"] == "h10-power-design-v1-20261020"
            and generation["responseTransform"] == "none"
            and generation["eligibility"] == "independent_whole_person_MCAR_complete_ten_not_trial_dropout",
            "Invalid generation identity")
    copied(generation["execution"], spec["execution"])
    copied(generation["populationCounts"], spec["populationCounts"])
    copied(generation["recruitedCounts"], {arm: 193 for arm in ARMS})
    people, ordering = {}, []
    for row in request["rows"]:
        require(type(row) is dict and set(row) == {"participant", "condition", "need_A", "need_B", "y"}
                and type(row["participant"]) is str and row["condition"] in ARMS, "Invalid request row")
        for key in ("need_A", "need_B", "y"):
            scalar(row[key])
        if row["participant"] not in people:
            people[row["participant"]] = [row["condition"], 0]
            ordering.append(row["participant"])
        require(people[row["participant"]][0] == row["condition"], "Cross-arm participant")
        people[row["participant"]][1] += 1
    require(all(x[1] == 10 for x in people.values()), "Incomplete retained participant")
    copied({arm: len([x for x in people.values() if x[0] == arm]) for arm in ARMS}, spec["populationCounts"])
    copied(ordering, generation["retainedParticipantIDs"])
    selected = []
    recruited = generation["kernelGeneration"]["people"]
    copied([p["participant"] for p in recruited], generation["recruitedParticipantIDs"])
    require(len({p["participant"] for p in recruited}) == 579, "Duplicate/missing recruited identity")
    for arm in ARMS:
        original_people = [p["participant"] for p in generation["kernelGeneration"]["people"] if p["condition"] == arm]
        mask = generation["retentionMask"][arm]
        require(len(original_people) == len(mask) == 193 and all(type(k) is bool for k in mask), "Invalid whole-person mask")
        selected += [person for person, keep in zip(original_people, mask) if keep]
    copied(selected, ordering)
    copied(generation["retainedRowIndices"], [10*i+j for i, p in enumerate(recruited)
                                            if p["participant"] in set(selected) for j in range(10)])
    extra = [ctx.bound(p) for p in pins[5:] if p != old["fitArtifact"]]
    extracted = [v for v in extra if v.get("schema") == "h10-inference-independent-rds-v1"]
    transported = [v for v in extra if v.get("schema") == "h10-power-source-transport-v1"]
    require(len(extracted) == len(transported) == 1, "Missing/duplicate fitted extraction or transport")
    evidence = extracted[0]["fits"][spec["fitKey"]]
    copied(evidence["fitKey"], spec["fitKey"])
    copied(evidence["requestDescriptor"], spec["request"])
    require(evidence["verified"] is True and evidence["frameRowCount"] == evidence["requestRowCount"] == len(request["rows"]),
            "Unverified extracted frame")
    validate_guards(old["raw"], candidate["rawCandidate"], evidence)
    copied(evidence["design"]["participant"], [r["participant"] for r in request["rows"]])
    copied(evidence["design"]["condition"], [r["condition"] for r in request["rows"]])
    require(set(evidence["design"]) == {"X", "y", "participant", "condition"}
            and len(evidence["design"]["X"]) == len(evidence["design"]["y"]) == len(request["rows"])
            and all(len(row) == 9 for row in evidence["design"]["X"]), "Invalid extracted design shape")
    for row in evidence["design"]["X"]:
        for value in row:
            scalar(value)
    for value in evidence["design"]["y"]:
        scalar(value)
    transport = transported[0]
    for field in ("newScientificInputs", "newFitInvocations", "newCandidateCalculations"):
        copied(transport[field], 0)
    fit = transport["fits"][spec["fitKey"]]
    copied(fit["request"], spec["request"])
    copied(fit["fitArtifact"], old["fitArtifact"])
    copied(fit["design"], evidence["design"])
    for field in ("exactMatrixInR", "exactResponseInR", "oneULPMatrixMutationRejected"):
        copied(fit[field], True)
    metadata = candidate["candidateMetadata"]
    require(metadata["frameVerified"] is True and metadata["frameRowCount"] == len(request["rows"]),
            "Unverified candidate frame")
    require(metadata["priorWeights"] is False and metadata["cluster"] == "participant"
            and metadata["inverseVariance"] is True and metadata["target"] == "fitted_scaled_marginal_Theta"
            and metadata["residual"] == "y-Xbeta", "Changed candidate covariance interpretation")
    copied(metadata["requestDescriptor"], spec["request"])
    environment = ctx.bound(ctx.bound(manifest)["environment"])["statistical_reference"]
    copied(metadata["runtime"]["R"], environment["R"])
    for package, version in environment["R_packages"].items():
        copied(metadata["runtime"][package], version)
    for runtime, names in ((extracted[0]["runtime"], ("lme4", "lmerTest")), (transport["runtime"], ("jsonlite",))):
        copied(runtime["R"], environment["R"])
        for name in names:
            expected = environment["R_packages"][name]
            spellings = {("lme4", "1.1-35.5"): ("1.1-35.5", "1.1.35.5"),
                         ("lmerTest", "3.1-3"): ("3.1-3", "3.1.3")}.get((name, expected), (expected,))
            require(type(runtime[name]) is str and runtime[name] in spellings, "Changed extracted package version")
    copied(candidate["rawCandidate"]["diagnostics"], old["raw"]["diagnostics"])
    copied(candidate["rawCandidate"]["method"], METHOD)
    compare(metadata["modelBasedVcov"], evidence["fixedVcov"])
    result = dict(repID=rep, populationCounts=spec["populationCounts"])
    for method, raw in (("baseline", old["raw"]), ("candidate", candidate["rawCandidate"])):
        require(raw["status"] == "fitted" and raw["diagnostics"]["usableCandidateInference"] is True,
                "Unavailable raw reference")
        result[method] = validate_branches(event[method+"Branches"], raw["references"][KEY], False)
    validate_pair(event, result)
    return result


def admitted_rows(ctx):
    request = ctx.request
    require(request["schema"] == "saved-coverage-input-v1" and request["mode"] in ("public_power", "historical_power"),
            "Unknown saved-input schema/mode")
    fields = {"schema", "mode", "aggregate", "relocations"}
    fields.update(("independent", "summaryIndex") if request["mode"] == "historical_power" else ("events",))
    require(set(request) == fields, "Changed request fields")
    aggregate = ctx.bound(request["aggregate"])
    cell = selected_cell(aggregate)
    if request["mode"] == "historical_power":
        rows = read_historical(ctx, aggregate)
        for method, count in (("baseline", 1881), ("candidate", 1879)):
            copied(cell[method]["coverage"][KEY]["counts"], dict(A=N, CIAvailable=N, C=count))
    else:
        require(aggregate["schema"] == PUBLIC and aggregate["family"] == "power" and aggregate["B"] == N
                and aggregate["expectedTotal"] == 80000 and aggregate["independentNumericalReadback"] is True
                and type(aggregate["presentTotal"]) is int and N <= aggregate["presentTotal"] <= 80000
                and type(aggregate["fullRosterComplete"]) is bool
                and aggregate["fullRosterComplete"] == (aggregate["presentTotal"] == 80000)
                and aggregate["status"] == ("full_reproduction" if aggregate["fullRosterComplete"] else "partial_entry_check"),
                "Changed public aggregate scope")
        read_manifest(ctx, aggregate["reproductionManifest"])
        require(type(request["events"]) is list and len(request["events"]) == N
                and len({p["path"] for p in request["events"]}) == N, "Incomplete/duplicate target descriptor roster")
        own = {}
        for pin in request["events"]:
            row = read_public_member(ctx, ctx.bound(pin), aggregate["reproductionManifest"])
            require(row["repID"] not in own, "Duplicate target rep")
            own[row["repID"]] = row
        require(set(own) == set(range(1, N+1)), "Incomplete target roster")
        rows = [own[i] for i in range(1, N+1)]
    reconcile(rows, cell)
    return rows, cell


def moments(values):
    for value in values:
        scalar(value)
    n = len(values)
    center = math.fsum(values)/n if n else None
    spread = math.sqrt(math.fsum((v-center)**2 for v in values)/(n-1)) if n > 1 else None
    return dict(n=n, mean=center, sampleSD=spread, min=min(values) if n else None, max=max(values) if n else None)


def reconcile(rows, cell):
    copied([row["repID"] for row in rows], list(range(1, N+1)))
    for method in ("baseline", "candidate"):
        coverage = cell[method]["coverage"][KEY]
        covered = sum(row[method]["covered"] for row in rows)
        copied(coverage["counts"], dict(A=N, CIAvailable=N, C=covered))
        copied(coverage["ciUnavailableAmongAvailable"], 0)
        for field, count in (("coverageAll", covered), ("coverageAvailable", covered), ("availabilityCompleted", N)):
            copied(coverage[field]["numerator"], count)
            copied(coverage[field]["denominator"], N)
            compare(coverage[field]["estimate"], count/N)
        for column, metric in (("estimate", "estimate"), ("SE", method+"SE"), ("df", method+"Df")):
            calculated = moments([row[method][column] for row in rows])
            saved = cell["contrastDiagnostics"][KEY][metric]
            copied(saved["n"], N)
            copied(saved["missing"], 0)
            for old, new in (("mean", "mean"), ("SD", "sampleSD"), ("min", "min"), ("max", "max")):
                compare(saved[old], calculated[new])
    transition = {k: 0 for k in ("00", "01", "10", "11", "unknown")}
    for row in rows:
        transition[str(int(row["baseline"]["covered"]))+str(int(row["candidate"]["covered"]))] += 1
    copied([p["tailMode"] for p in cell["paired"]], list(TAILS))
    for branch in cell["paired"]:
        copied(branch["denominator"], N)
        copied(branch["contrasts"][KEY]["coverage"], transition)


def aggregate_values(cell):
    diagnostics = cell["contrastDiagnostics"][KEY]
    for item in diagnostics.values():
        copied(item["n"], N)
        copied(item["missing"], 0)
    center = scalar(diagnostics["estimate"]["mean"])
    spread = scalar(diagnostics["estimate"]["SD"])
    require(spread > 0, "Nonpositive empirical SD")
    mcse = spread/math.sqrt(N)
    radius = NormalDist().inv_cdf(.975)*mcse
    result = dict(n=N, truth=0, biasEstimate=center, empiricalSD=spread, biasMonteCarloSE=mcse,
        biasNormalMCInterval95=[center-radius, center+radius],
        intervalScope="normal_MC_approximation_for_mean_not_coverage_CI_or_causal_test")
    for name in ("baselineSE", "candidateSE"):
        mean, sd = scalar(diagnostics[name]["mean"]), scalar(diagnostics[name]["SD"])
        require(mean > 0 and sd >= 0, "Invalid saved SE moments")
        rms = math.hypot(mean, math.sqrt((N-1)/N)*sd)
        result[name] = dict(mean=mean, sampleSD=sd, RMS=rms, RMSOverEmpiricalSD=rms/spread)
    result.update(baselineCoverage=cell["baseline"]["coverage"][KEY],
        candidateCoverage=cell["candidate"]["coverage"][KEY],
        interpretation="descriptive_saved_moments_no_method_adoption_no_unique_cause")
    return result


def record_values(rows):
    require(len(rows) == N, "Incomplete target roster")
    output = dict(n=N, methods={}, commonMisses=sum(not r["baseline"]["covered"] and not r["candidate"]["covered"] for r in rows))
    for method in ("baseline", "candidate"):
        values = [r[method] for r in rows]
        count, below, above = (sum(v[key] for v in values) for key in ("covered", "belowTruth", "aboveTruth"))
        require(count+below+above == N, "Invalid coverage partition")
        output["methods"][method] = dict(denominators=dict(M=N, T=N, A=N, CIAvailable=N), covered=count,
            belowTruth=below, aboveTruth=above, estimate=moments([v["estimate"] for v in values]),
            standardizedError=moments([v["z"] for v in values]), RMSSE=math.sqrt(math.fsum(v["SE"]**2 for v in values)/N))
    output["retentionByCoverage"] = {}
    for flag in (True, False):
        group = [r["populationCounts"] for r in rows if r["candidate"]["covered"] is flag]
        description = {arm: moments([p[arm] for p in group]) for arm in ARMS}
        description["absoluteC1C2Imbalance"] = moments([abs(p["C1"]-p["C2"]) for p in group])
        output["retentionByCoverage"][str(flag)] = description
    output["interpretation"] = "descriptive_association_not_retention_causality"
    return output


def write_new(path, value, ctx):
    path = Path(path).absolute()
    require(path.parent.is_dir() and not path.exists() and not path.is_symlink(), "Refusing overwrite/output parent")
    path = path.parent.resolve() / path.name
    require(str(path) not in ctx.files, "Output overlaps input")
    checkout = Path(__file__).resolve().parents[2]
    require(not any(path.is_relative_to(checkout/name) for name in ("src", "scripts", "configs", "tests", "notebooks")),
            "Output overlaps source")
    ctx.close()
    handle, temporary = tempfile.mkstemp(prefix=".coverage-independent-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        ctx.close()
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def verify(request_path, producer_result, output):
    ctx = Readback(request_path)
    sources = ctx.sources()
    rows, cell = admitted_rows(ctx)
    inputs = dict(ctx.files)
    produced, pin = ctx.read(producer_result)
    copied(produced["schema"], "saved-coverage-diagnosis-v1")
    copied(produced["mode"], ctx.request["mode"])
    copied(produced["cellID"], CELL)
    copied(produced["contrast"], KEY)
    copied(produced["scope"], "complete_fixed_target_not_full_population_acceptance")
    copied(produced["input"], ctx.request_pin)
    copied(produced["sources"], sources)
    copied(produced["observedInputs"], inputs)
    copied(produced["sourceClosure"], True)
    historical = dict(aggregateSHA256=AGGREGATE_SHA, independentSHA256=INDEPENDENT_SHA,
        baselineCovered=1881, candidateCovered=1879, denominator=2000,
        authenticatedThisInvocation=ctx.request["mode"] == "historical_power",
        interpretation="reported_coverage_0.9395_applicability_warning_not_a_new_result_target_or_unique_cause")
    copied(produced["historicalReference"], historical)
    for name in ("newFits", "newDraws", "RDSDeserializations"):
        copied(produced[name], 0)
    copied(produced["targetScalars"], rows, "source target scalars")
    aggregate = aggregate_values(cell)
    records = record_values(rows)
    compare(produced["aggregateOnly"], aggregate)
    for field in ("n", "truth", "biasEstimate", "empiricalSD", "baselineCoverage", "candidateCoverage"):
        copied(produced["aggregateOnly"][field], aggregate[field])
    for field in ("baselineSE", "candidateSE"):
        for key in ("mean", "sampleSD"):
            copied(produced["aggregateOnly"][field][key], aggregate[field][key])
    compare(produced["records"], records)
    result = dict(schema="saved-coverage-independent-v1", mode=ctx.request["mode"], cellID=CELL, contrast=KEY,
        input=ctx.request_pin, producerResult=pin, sources=sources, aggregateOnly=aggregate, records=records,
        targetRecordsChecked=N, passed=True, sourceClosure=True, observedInputs=dict(ctx.files),
        scope="complete_fixed_target_not_full_population_acceptance", newFits=0, newDraws=0, RDSDeserializations=0,
        scientificInterpretation="descriptive_only_no_method_adoption_or_unique_cause")
    result["historicalReference"] = historical
    write_new(output, result, ctx)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--producer-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.request, args.producer_result, args.output)


if __name__ == "__main__":
    main()
