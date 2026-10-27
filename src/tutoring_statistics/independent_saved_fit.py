"""Independent package-based verification of a bound single-fit diagnostic.

Numerical contrasts are derived by the R companion, not the producer. Saved
historical inputs and new public reproductions retain distinct identities.
"""
import hashlib
import json
import math
from pathlib import Path

from .saved_fit_inputs import Observations, load, write_new, INPUT_SCHEMA
from .saved_fit_runtime import ROOT, sources, runtime, run_r

CELL = "h10pd1_initial_amount_s1_l10_recruit193_mcar08"
TARGET = dict(caseID=CELL, repID=1, ordinal=32)
KEY = CELL + "__r0001"
IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
LINEAR_IDS = tuple(key + ":" + kind for key in IDS for kind in ("D", "S"))
NAMES = ("(Intercept)", "conditionC1", "conditionC3", "need_A", "need_B",
         "conditionC1:need_A", "conditionC3:need_A", "conditionC1:need_B", "conditionC3:need_B")
AGG_SHA = "408003c741653dd27108202381cb09e4e0df17cb36eb0bb0fef90bcf178a4f38"
INDEP_SHA = "2c2f4f34ec6c0c45abf8c79c103d60ec470fe0b1b58fffae3b97cb180d05c66a"
MANIFEST_SHA = "0953299b5d0afadbf84e99aec706c49eee5126b16c1cf0ab40a28038f1426e69"
METHOD = "h10_magnitude_DS_IUT_Bonferroni2_Holm4_candidate_v1"

def require(value, message):
    if not value:
        raise ValueError(message)


def decode(content):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid)


def exact(actual, expected, name="identity"):
    require(type(actual) is type(expected), "Changed exact type: " + name)
    if type(expected) is dict:
        require(set(actual) == set(expected), "Changed exact fields: " + name)
        for key in expected:
            exact(actual[key], expected[key], name + "." + key)
    elif type(expected) is list:
        require(len(actual) == len(expected), "Changed exact length: " + name)
        for index, item in enumerate(expected):
            exact(actual[index], item, name + "[" + str(index) + "]")
    else:
        require(actual == expected, "Changed exact value: " + name)


def near(actual, expected, name="number"):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), "Changed fields: " + name)
        for key in expected:
            near(actual[key], expected[key], name + "." + key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "Changed shape: " + name)
        for index, item in enumerate(expected):
            near(actual[index], item, name + "[" + str(index) + "]")
    elif type(expected) in (int, float) and type(expected) is not bool:
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=2e-8, abs_tol=2e-9), "Numeric disagreement: " + name)
    else:
        exact(actual, expected, name)



def population(rows):
    people = {}
    require(type(rows) is list and rows, "Missing actual rows")
    for row in rows:
        require(type(row) is dict and set(row) == {"participant", "condition", "need_A", "need_B", "y"}
                and type(row["participant"]) is str and row["participant"]
                and row["condition"] in ("C1", "C2", "C3"), "Invalid actual row")
        for name in ("need_A", "need_B", "y"):
            require(type(row[name]) in (int, float) and math.isfinite(row[name]), "Invalid row scalar")
        person = people.setdefault(row["participant"], [row["condition"], 0])
        require(person[0] == row["condition"], "Participant crosses arm")
        person[1] += 1
    require(all(count == 10 for _, count in people.values()), "Participant must have ten rows")
    counts = {arm: sum(value[0] == arm for value in people.values()) for arm in ("C1", "C2", "C3")}
    require(all(0 < value <= 193 for value in counts.values()), "Invalid random-K population")
    return counts


def full_guard(raw):
    diagnostics = raw["diagnostics"]
    require(raw["status"] == "fitted" and all(type(diagnostics[k]) is bool for k in
            ("converged", "singular", "usableCandidateInference")), "Invalid original guard")
    sequence = lambda value: value if type(value) is list else [value]
    finite = lambda value: type(value) in (int, float) and math.isfinite(value)
    eigen = sequence(diagnostics.get("hessianEigenvalues"))
    gradient = diagnostics.get("gradient")
    result = dict(converged=diagnostics["converged"], singular=diagnostics["singular"],
        hessianOK=bool(eigen) and all(finite(v) for v in eigen) and min(eigen) > 0,
        gradientOK=gradient is not None and all(finite(v) for v in sequence(gradient)),
        finiteBeta=all(finite(v) for v in raw["coefficients"].values()),
        finiteV=all(finite(v) for row in raw["vcov"] for v in row))
    result["usableCandidateInference"] = (result["converged"] and not result["singular"] and
        result["hessianOK"] and result["gradientOK"] and result["finiteBeta"] and result["finiteV"])
    exact(diagnostics["usableCandidateInference"], result["usableCandidateInference"], "full original guard")
    return result


def admit_history(request, observed):
    exact(request["target"], TARGET, "fixed target")
    require(request["aggregate"]["sha256"] == AGG_SHA and request["independentAggregate"]["sha256"] == INDEP_SHA,
            "Changed historical anchors")
    aggregate = observed.read(request["aggregate"]); final = observed.read(request["independentAggregate"])
    require(aggregate["schema"] == "h10-power-design-aggregate-v1" and aggregate["complete"] is True
            and aggregate["expectedTotal"] == aggregate["completedTotal"] == 80000
            and final["schema"] == "h10-power-design-independent-aggregate-v1"
            and final["passed"] is True and final["sourceClosure"] is True and final["complete"] is True
            and final["aggregate"]["sha256"] == AGG_SHA and final["manifest"]["sha256"] == MANIFEST_SHA,
            "Changed accepted final scope")
    proof_pin = request["blockProof"]
    require(proof_pin in aggregate["independentBlocks"] and proof_pin in final["independentBlocks"],
            "Block not accepted by both finals")
    proof = observed.read(proof_pin)
    require(proof["schema"] == "h10-power-design-independent-block-v1" and proof["passed"] is True
            and proof["sourceClosure"] is True and proof["manifest"] == final["manifest"]
            and 32 in proof["ordinals"] and proof["summary"] in aggregate["summaries"], "Wrong accepted block")
    summary = observed.read(proof["summary"])
    require(summary["schema"] == "h10-power-design-summary-v1" and summary["sourceClosure"] is True
            and summary["manifest"] == final["manifest"], "Wrong accepted summary")
    rows = [row for row in summary["cases"] if row["ordinal"] == 32]
    require(len(rows) == 1, "Missing/duplicate target row")
    row = rows[0]
    for name, value in TARGET.items():
        exact(row[name], value, "target row " + name)
    require(row["fitKey"] == KEY and row["technicalCompleted"] is True, "Wrong target completion")
    outer = observed.read(row["record"]); spec = observed.read(row["spec"])
    require(outer["schema"] == "h10-power-design-record-v1" and outer["sourceClosure"] is True
            and outer["manifest"] == final["manifest"] and outer["spec"] == row["spec"]
            and outer["baseline"] == row["baseline"] and outer["candidate"] == row["candidate"]
            and outer["ordinal"] == 32 and outer["fitKey"] == KEY and outer["technicalCompleted"] is True,
            "Wrong original outer stage")
    require(spec["schema"] == "h10-power-design-spec-v1" and spec["manifest"] == final["manifest"]
            and spec["ordinal"] == 32 and spec["caseID"] == CELL and spec["repID"] == 1
            and spec["fitKey"] == KEY and spec["case"] == row["case"], "Wrong original spec")
    case = spec["case"]
    require(case["view"] == "initial_amount" and case["populationRegime"] == "recruit193_mcar08"
            and case["recruitedPerArm"] == 193 and case["eligibilityProbability"] == .8
            and case["scenarioIndex"] == 1 and all(v == 0 for v in case["delta"].values()), "Changed original law")
    manifest = observed.read(final["manifest"])
    require(manifest["schema"] == "h10-power-design-manifest-v1" and manifest["sourceClosure"] is True
            and manifest["expectedTotal"] == 80000 and manifest["B"] == 2000, "Changed original manifest")
    for name, value in manifest["cells"][32].items():
        exact(case[name], value, "manifest target law " + name)
    for name, value in dict(repID=1, retentionSeed=802940201, designSeed=902940201, responseSeed=1002940201).items():
        exact(case[name], value, "original stream " + name)
    baseline = observed.read(row["baseline"]); candidate = observed.read(row["candidate"])
    require(baseline["fitKey"] == KEY and baseline["request"] == spec["request"]
            and baseline["raw"]["status"] == "fitted" and candidate["fitKey"] == KEY
            and candidate["request"] == spec["request"] and candidate["baseline"] == row["baseline"]
            and candidate["technicalCompleted"] is True, "Changed actual stage linkage")
    original = observed.read(spec["request"]); generation = observed.read(spec["generation"])
    require(original["model"] == "H10" and original["reference"] == "C2"
            and generation["case"] == case and generation["execution"] == spec["execution"], "Changed saved input origin")
    actual_k = population(original["rows"])
    exact(spec["populationCounts"], actual_k, "saved actual K")
    raw_artifact = observed.read(baseline["rawArtifact"])
    exact(raw_artifact, baseline["raw"], "actual raw artifact")
    observed.pin(baseline["fitArtifact"]["path"], baseline["fitArtifact"]["sha256"])
    require(candidate["fitArtifact"] == baseline["fitArtifact"], "Candidate points to another RDS")
    for value in (spec["execution"], baseline["execution"], candidate["execution"], proof["execution"]):
        require(type(value) is dict and type(value.get("jobID")) is str and value["jobID"].isdigit()
                and type(value.get("pid")) is int and value["pid"] > 0, "Missing original writer")
    historical = dict(manifest=final["manifest"], aggregate=request["aggregate"], independentAggregate=request["independentAggregate"],
        blockProof=proof_pin, summary=proof["summary"], outer=row["record"],
        spec=row["spec"], baseline=row["baseline"], candidate=row["candidate"], request=spec["request"],
        generation=spec["generation"], rawArtifact=baseline["rawArtifact"], fitArtifact=baseline["fitArtifact"],
        writers={"input": spec["execution"], "baseline": baseline["execution"],
                 "candidate": candidate["execution"], "outer": outer["execution"]})
    return dict(manifest=manifest, historical=historical, baseline=baseline, candidate=candidate,
                actualK=actual_k, spec=spec, row=row)


def independent_composition(references):
    """Separate arithmetic and Holm loop, using only independently obtained R p values."""
    require(set(references) == set(LINEAR_IDS), "Changed eight-reference roster")
    tests = {}
    for key in IDS:
        d, s = references[key + ":D"], references[key + ":S"]
        statuses = (d["status"], s["status"])
        require(all(value in ("available", "statistical_unavailable", "technical_unresolved") for value in statuses),
                "Unknown reference status")
        status = ("technical_unresolved" if "technical_unresolved" in statuses else
                  "statistical_unavailable" if "statistical_unavailable" in statuses else "available")
        if status == "available":
            inputs = dict(D_less=d["lessP"], D_greater=d["greaterP"], S_less=s["lessP"], S_greater=s["greaterP"])
            require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in inputs.values()), "Invalid p value")
            routes = dict(reference_positive=max(inputs["D_less"], inputs["S_greater"]),
                          reference_negative=max(inputs["D_greater"], inputs["S_less"]))
            p = min(1., 2 * min(routes.values()))
        else:
            inputs = routes = p = None
        tests[key] = dict(status=status, inputP=inputs, orientationP=routes, pMagnitude=p,
                         nominalReject=p < .05 if status == "available" else False if status == "statistical_unavailable" else None)
    unresolved = any(v["status"] == "technical_unresolved" for v in tests.values())
    if not unresolved:
        ordered = sorted(IDS, key=lambda key: tests[key]["pMagnitude"] if tests[key]["status"] == "available" else 1.)
        running = 0.
        for position, key in enumerate(ordered):
            value = tests[key]; padded = value["pMagnitude"] if value["status"] == "available" else 1.
            running = max(running, min(1., (4-position)*padded))
            value.update(holmPaddingP=padded, holmP=running if value["status"] == "available" else None,
                         holmReject=value["status"] == "available" and running < .05)
    else:
        for value in tests.values():
            value.update(holmPaddingP=None, holmP=None, holmReject=None)
    counts = {status: sum(v["status"] == status for v in tests.values())
              for status in ("available", "statistical_unavailable", "technical_unresolved")}
    return dict(tests=tests, anyMagnitudeRejects=None if unresolved else any(v["holmReject"] for v in tests.values()),
        status="technical_unresolved" if unresolved else "complete", holmComputed=not unresolved,
        completeFamilyAvailable=counts["available"] == 4,
        counts=dict(planned=4, technicalCompleted=4-counts["technical_unresolved"], available=counts["available"],
                    statisticalUnavailable=counts["statistical_unavailable"], technicalUnresolved=counts["technical_unresolved"]))


def compare_method(actual, independent, name):
    require(set(actual["references"]) == set(LINEAR_IDS) and set(independent["references"]) == set(LINEAR_IDS),
            "Missing/extra linear reference")
    near(actual["covariance"], independent["covariance"], name + " full covariance")
    near(actual["contrastCovariance"], independent["contrastCovariance"], name + " eight covariance")
    for key in LINEAR_IDS:
        old, new = actual["references"][key], independent["references"][key]
        exact(old["linearID"], key, "linear ID"); exact(new["linearID"], key, "R linear ID")
        exact(old["coefficientNames"], list(NAMES), "coefficient names")
        exact(new["coefficientNames"], list(NAMES), "R coefficient names")
        # JSON int/float representations of literal L coefficients are equivalent;
        # bool and any numerical perturbation remain forbidden.
        require(type(old["L"]) is list and type(new["L"]) is list and len(old["L"]) == len(new["L"]) == 9
                and all(type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) and a == b
                        for a, b in zip(old["L"], new["L"])), "Changed exact linear vector")
        exact(old["status"], new["status"], name + " reference status")
        for field in ("estimate", "SE", "df", "statistic", "lessP", "greaterP", "twoSidedP", "CI"):
            near(old[field], new[field], name + " " + key + " " + field)
    calculated = independent_composition(independent["references"])
    composition = actual["composition"]
    require(composition["methodID"] == METHOD and composition["familySize"] == 4 and composition["alpha"] == .05,
            "Changed scientific composition family")
    for field in ("status", "holmComputed", "anyMagnitudeRejects", "completeFamilyAvailable", "counts"):
        exact(composition[field], calculated[field], "family " + field)
    require(set(composition["tests"]) == set(IDS), "Changed scientific comparison roster")
    for key in IDS:
        for field, value in calculated["tests"][key].items():
            if field in ("status", "holmReject", "nominalReject"):
                exact(composition["tests"][key][field], value, key + " " + field)
            else:
                near(composition["tests"][key][field], value, key + " " + field)
    return calculated



class Observed(Observations):
    def pin(self, path, expected=None, *, runtime=False):
        return self.observe(path, expected)


def admit_public_input(binding, observed):
    """Re-read committed artifacts; do not use producer output as source truth."""
    from . import power_reference
    manifest = observed.read(binding["manifest"])
    require(manifest["schema"] == "conditional-statistics-reproduction-v1"
            and manifest["family"] == "power" and manifest["identity"] == "new_public_reproduction"
            and manifest["expectedTotal"] == 80000, "Wrong independent public manifest")
    for key in ("sources", "driver", "rDriver", "environment", "nativeEntry", "nativeIdentity"):
        observed.pin(manifest[key]["path"], manifest[key]["sha256"])
    inventory = observed.read(manifest["sources"])["files"]
    for key, name in dict(driver="reproduction.py", rDriver="kernel_entry.R",
        environment="numerical_environment.json", nativeEntry="native_inputs.py", nativeIdentity="native_identity.json").items():
        require(manifest[key] in inventory and Path(manifest[key]["path"]).name == name,
                "Independent public source role is missing or replaced")
    require({"power_reference.py", "power_origins.py", "joint_generation.py", "joint_fit.R",
        "population_fit.R", "cluster_fit.R", "independent_power.py", "independent_transport.py",
        "independent_cluster_rds.R", "source_transport.R", "mixed_fit.R", "joint_design.py"}
        <= {Path(pin["path"]).name for pin in inventory}, "Incomplete independent public inventory")
    require(len(manifest["native"]) == 3 and {Path(pin["path"]).name for pin in manifest["native"]}
            == {"resolved_design.json", "initial_panel.json", "h10_canonical.json"}, "Wrong independent native roster")
    for pin in inventory + manifest["native"]:
        observed.pin(pin["path"], pin["sha256"])
    for pin in (manifest["numericalRuntime"]["executable"], manifest["numericalRuntime"]["Rscript"]):
        observed.pin(pin["path"], pin["sha256"])
    spec, commitment = observed.read(binding["spec"]), observed.read(binding["commitment"])
    event = commitment["event"]
    require(commitment["schema"] == manifest["schema"] and commitment["manifest"] == binding["manifest"]
            and commitment["eventSha256"] == hashlib.sha256(json.dumps(event, sort_keys=True,
                allow_nan=False, separators=(",", ":")).encode()).hexdigest(), "Changed independent event")
    exact(binding["target"], TARGET, "public selected target")
    exact({k: spec[k] for k in TARGET}, TARGET, "public spec target")
    exact({k: event[k] for k in TARGET}, TARGET, "public event target")
    exact(spec["case"], power_reference.case_for(32, 2000), "public case and seeds")
    require(spec["manifest"] == binding["manifest"] and spec["origin"] is None
            and spec["fitKey"] == event["fitKey"] == KEY
            and event["origin"] == "new_population_input" and event["technicalCompleted"] is True,
            "Changed public member identity")
    pins = commitment["artifacts"]
    for pin in pins:
        observed.pin(pin["path"], pin["sha256"])
    for pin in (binding["spec"], binding["baseline"], binding["candidate"], spec["request"], spec["generation"]):
        require(pin in pins, "Independent artifact is not a committed member")
    baseline, candidate = observed.read(binding["baseline"]), observed.read(binding["candidate"])
    require(baseline["manifestSha256"] == binding["manifest"]["sha256"]
            and candidate["manifest"] == binding["manifest"] and candidate["baseline"] == binding["baseline"]
            and baseline["fitArtifact"] == candidate["fitArtifact"]
            and baseline["request"] == candidate["request"] == spec["request"]
            and baseline["generation"] == candidate["generation"] == spec["generation"]
            and baseline["fitKey"] == candidate["fitKey"] == KEY
            and baseline["caseID"] == candidate["caseID"] == CELL
            and baseline["repID"] == candidate["repID"] == 1
            and candidate["technicalCompleted"] is True and candidate["technicalError"] is None,
            "Independent saved-stage binding differs")
    for pin in (baseline["rawArtifact"], baseline["fitArtifact"]):
        require(pin in pins, "Independent raw/model artifact is not committed")
        observed.pin(pin["path"], pin["sha256"])
    exact(observed.read(baseline["rawArtifact"]), baseline["raw"], "independent actual raw")
    evidence_by_schema = {}
    for pin in pins:
        if Path(pin["path"]).suffix == ".json":
            artifact = observed.read(pin)
            if type(artifact) is dict and artifact.get("schema") in (
                    "h10-inference-independent-rds-v1", "h10-power-source-transport-v1"):
                require(artifact["schema"] not in evidence_by_schema, "Duplicate saved independent evidence")
                evidence_by_schema[artifact["schema"]] = artifact
    require(set(evidence_by_schema) == {"h10-inference-independent-rds-v1", "h10-power-source-transport-v1"},
            "Missing saved independent source evidence")
    extraction = evidence_by_schema["h10-inference-independent-rds-v1"]["fits"][KEY]
    transport = evidence_by_schema["h10-power-source-transport-v1"]
    source = transport["fits"][KEY]
    require(extraction["verified"] is True and extraction["requestDescriptor"] == spec["request"]
            and source["request"] == spec["request"] and source["fitArtifact"] == baseline["fitArtifact"]
            and source["exactMatrixInR"] is True and source["exactResponseInR"] is True
            and source["oneULPMatrixMutationRejected"] is True
            and transport["newScientificInputs"] == transport["newFitInvocations"]
            == transport["newCandidateCalculations"] == 0, "Independent committed source evidence differs")
    exact(extraction["design"], source["design"], "independent saved source design")
    original, generation = observed.read(spec["request"]), observed.read(spec["generation"])
    metadata = candidate["candidateMetadata"]
    require(metadata["requestDescriptor"] == spec["request"] and metadata["frameVerified"] is True
            and metadata["frameRowCount"] == len(original["rows"])
            and metadata["target"] == "fitted_scaled_marginal_Theta" and metadata["inverseVariance"] is True
            and metadata["cluster"] == "participant" and metadata["residual"] == "y-Xbeta"
            and metadata["priorWeights"] is False, "Changed independent original candidate metadata")
    require(original["model"] == "H10" and original["reference"] == "C2", "Wrong saved model")
    exact(generation["case"], spec["case"], "independent generation")
    actual_k = population(original["rows"])
    exact(spec["populationCounts"], actual_k, "independent actual counts")
    exact(generation["populationCounts"], actual_k, "independent generation counts")
    exact(generation["recruitedCounts"], {arm: 193 for arm in actual_k}, "independent recruitment")
    for arm, count in actual_k.items():
        mask = generation["retentionMask"][arm]
        require(len(mask) == 193 and all(type(x) is bool for x in mask) and sum(mask) == count,
                "Changed independent retention mask")
    exact(generation["retainedParticipantIDs"], list(dict.fromkeys(row["participant"] for row in original["rows"])),
          "independent retained row order")
    return dict(spec=spec, baseline=baseline, candidate=candidate, actualK=actual_k)


def verify(input_path, producer_path, output_path, rscript):
    binding_path = Path(input_path).resolve()
    binding = load(binding_path)
    require(binding.get("schema") == INPUT_SCHEMA and binding.get("kind") in
            ("historical_power", "public_power"), "Wrong scientific request")
    observed = Observed(binding.get("locations", {}))
    require(observed.resolve(binding_path) == binding_path, "Cannot relocate the input binding itself")
    require(observed.resolve(Path(producer_path).resolve()) == Path(producer_path).resolve(),
            "Cannot relocate the producer output under review")
    input_pin = observed.actual(binding_path)
    produced_pin = observed.actual(Path(producer_path).resolve())
    produced = observed.read(produced_pin)
    if binding["kind"] == "historical_power":
        ctx = admit_history(observed.read(binding["historicalRequest"]), observed)
    else:
        ctx = admit_public_input(binding, observed)
    current_sources = sources(observed)
    require(produced["schema"] == "saved-magnitude-result-v1" and produced["input"] == input_pin
            and produced["kind"] == binding["kind"] and produced["target"] == TARGET
            and produced["fitKey"] == KEY and produced["sources"] == current_sources
            and produced["scientificAdoption"] is False and produced["constituentValidityEstablished"] is False
            and produced["independentEightLVerified"] is False and produced["sourceClosure"] is True
            and produced["status"] == "complete" and produced["technicalErrors"] == []
            and produced["computation"] == dict(newFits=0, newDraws=0,
                rdsReadToolInvocations=2, completedEightLMethods=2)
            and produced["actualRdsInspected"] is True and produced["sameRExactAndSameExportVerified"] is True,
            "Changed or incomplete saved-fit result")
    for path, digest in produced["observedInputs"].items():
        observed.actual(path, digest)
    exact(produced["actualK"], ctx["actualK"], "producer actual K")
    exact(produced["linearIDs"], list(LINEAR_IDS), "producer linear order")
    exact(produced["coefficientNames"], list(NAMES), "producer coefficient order")
    pins = {name: observed.pin(pin["path"], pin["sha256"]) for name, pin in
            (("request", ctx["spec"]["request"]), ("fitArtifact", ctx["baseline"]["fitArtifact"]))}
    exact(produced["transportInputs"], pins, "actual transported sources")
    output = Path(output_path).resolve()
    require(output.parent.is_dir() and not output.exists(), "Use a new independent output")
    environment = runtime(rscript, observed, output)
    exact(environment["runtimeFiles"], produced["runtime"]["runtimeFiles"], "current numerical runtime bytes")
    independent, artifacts = run_r(environment["Rscript"], ROOT / "independent_saved_fit.R",
        dict(schema="h10-magnitude-independent-r-request-v1", target=TARGET, actualK=ctx["actualK"],
             **pins, packageVersions=environment["packageVersions"], runtimeFiles=environment["runtimeFiles"],
             sources=current_sources), output.with_suffix(".r.json"), observed)
    require(independent["schema"] == "h10-magnitude-independent-r-v1" and independent["target"] == TARGET
            and independent["sourceClosure"] is True and independent["frameVerified"] is True
            and independent["exactSourceDesignInR"] is True
            and independent["newFits"] == independent["newDraws"] == 0 and independent["RDSReads"] == 1
            and independent["request"] == pins["request"] and independent["fitArtifact"] == pins["fitArtifact"],
            "Incomplete independent package extraction")
    exact(independent["coefficientNames"], list(NAMES), "R coefficient order")
    exact(independent["linearIDs"], list(LINEAR_IDS), "R linear order")
    exact(independent["actualK"], ctx["actualK"], "R actual K")
    guard = full_guard(ctx["baseline"]["raw"])
    exact(independent["guard"], guard, "independent full original guard")
    require(guard["usableCandidateInference"] is True, "Fixed target original guard is not usable")
    exact(produced["originalGuard"], dict(converged=guard["converged"], nonsingular=not guard["singular"],
          hessianOK=guard["hessianOK"], gradientOK=guard["gradientOK"], finiteBeta=guard["finiteBeta"],
          finiteModelV=guard["finiteV"]), "producer full guard")
    near(produced["coefficients"], independent["coefficients"], "independent beta")
    near(independent["coefficients"], [ctx["baseline"]["raw"]["coefficients"][name] for name in NAMES], "saved beta")
    for name in ("residualVariance", "randomCovariance"):
        near(independent[name], ctx["baseline"]["raw"][name], "saved " + name)
    checked = {}
    for method in ("baseline", "candidate"):
        checked[method] = compare_method(produced["methods"][method], independent["methods"][method], method)
        raw = ctx[method]["raw" if method == "baseline" else "rawCandidate"]
        near(independent["methods"][method]["covariance"], raw["vcov"], "saved full V " + method)
        for key in IDS:
            reference = independent["methods"][method]["references"][key + ":D"]
            for field in ("estimate", "SE", "df", "statistic", "CI"):
                near(reference[field], raw["references"][key][field], "saved D " + method + " " + key + " " + field)
            p_key = "twoSidedP" if method == "baseline" else "packageTwoSidedP"
            near(reference["twoSidedP"], raw["references"][key][p_key], "saved package p")
    for path, digest in independent["observedInputs"].items():
        observed.pin(path, digest)
    observed.fresh()
    result = dict(schema="saved-magnitude-independent-v1", input=input_pin, producer=produced_pin,
        target=TARGET, kind=binding["kind"], actualK=ctx["actualK"], independentR=artifacts,
        checked=checked, sources=current_sources, runtime=environment, observedInputs=observed.files,
        passed=True, sourceClosure=True, numericalEightLAgreement=True, constituentValidityEstablished=False,
        scientificAdoption=False, newFits=0, newDraws=0, RDSReads=1)
    write_new(output, result)
    return result
