"""Portable source-calling conditional calibration, coverage and power.

Plan is read-only. Numerical work requires an explicit selector and --execute.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import fcntl
import gzip
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT
_SOURCE_BASELINE = None
TOTAL = {"null": 24000, "paired": 10000, "power": 80000}
SCHEMA = "conditional-statistics-reproduction-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_output(output, reference_dir=None):
    """Reject source/configuration/reference overlap before numerical imports or writes."""
    output = Path(output).resolve()
    protected = [ROOT]
    checkout = next((p for p in ROOT.parents if (p / "pyproject.toml").is_file()), None)
    if checkout is not None:
        protected.extend((checkout / name).resolve()
                         for name in ("src", "scripts", "configs", "tests", "notebooks"))
    if reference_dir is not None:
        protected.append(Path(reference_dir).resolve())
    require(all(not output.is_relative_to(path) and not path.is_relative_to(output)
                for path in protected), "Output overlaps immutable source/configuration/reference")
    return output


def read(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate(descriptor):
    require(pin(descriptor["path"]) == descriptor, "Changed committed source/input: " + descriptor["path"])


def write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, "Refusing changed existing JSON: " + str(path))
        return pin(path)
    fd, temporary = tempfile.mkstemp(prefix=".new-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(read(path) == value, "Refusing changed existing JSON: " + str(path))
    finally:
        os.unlink(temporary)
    return pin(path)


def inventory():
    global _SOURCE_BASELINE
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and
                   p.suffix in {".py", ".R", ".cjs", ".json"} and "__pycache__" not in p.parts)
    value = dict(files=[pin(p) for p in paths])
    if _SOURCE_BASELINE is None:
        _SOURCE_BASELINE = value
    require(value == _SOURCE_BASELINE, "Changed public source/input during invocation")
    return value


@lru_cache(maxsize=None)
def module(relative):
    return importlib.import_module(__package__ + "." + Path(relative).stem)


def numerical_runtime():
    require(sys.version_info[:3] == (3, 10, 9), "Use reference Python 3.10.9")
    import numpy
    import scipy
    require(numpy.__version__ == "1.23.5" and scipy.__version__ == "1.10.0",
            "Use NumPy 1.23.5 and SciPy 1.10.0")


def run_r(args, request, receipt, *, script=None):
    require(not receipt.exists(), "Existing unaccepted R receipt; choose a new attempt")
    with receipt.with_suffix(".log").open("x") as log:
        subprocess.run([args.rscript, "--vanilla", str(script or ROOT / "kernel_entry.R"),
                        str(request), str(receipt)], cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)
    return read(receipt)


@lru_cache(maxsize=1)
def source_pins():
    return inventory()["files"]


def base_r_request(family):
    return dict(schema="round10-statistical-relocated-r-request-v1", family=family,
                sources=source_pins(), packages=read(ROOT / "data/numerical_environment.json")["statistical_reference"]["R_packages"])


@lru_cache(maxsize=1)
def null_plan():
    from . import null_reference as ni
    return ni.make_plan(read(SOURCE / "data/mixed_calibration.json"))


def prepare_native(args, output):
    from .native_inputs import generate
    target = output / "native"
    commitment = output / "native_commit.json"
    if commitment.exists():
        for descriptor in read(commitment)["artifacts"]:
            validate(descriptor)
        return target
    require(not target.exists(), "Partial native preparation exists; inspect it before retrying")
    generate(target, node=args.node)
    from .joint_design import adapt_normalized_records, prepare_population
    canonical = adapt_normalized_records(read(target / "native_joint_normalized.json"))
    require(canonical["rows"] == read(ROOT / "data/joint_covariates.json")["rows"],
            "Generated joint-X rows differ from accepted exact inputs")
    population = prepare_population(canonical)
    require(len(population["eligibleRows"]) == 270 and len(population["eligibleParticipants"]) == 27,
            "Native joint-X population differs")
    write(target / "h10_canonical.json", canonical)
    initial = read(target / "initial_panel.json")
    require(len(initial["rows"]) == 450 and initial["rows"] == read(ROOT / "data/initial_covariates.json")["rows"],
            "Generated initial rows differ from accepted exact inputs")
    write(commitment, dict(schema=SCHEMA, role="exact_native_covariate_inputs",
                           artifacts=[pin(p) for p in sorted(target.glob("*.json"))]))
    return target


def family_manifest(args, family, output, native):
    path = output / family / "manifest.json"
    value = dict(schema=SCHEMA, family=family, expectedTotal=TOTAL[family],
        sources=pin(output / "source_snapshot.json"), driver=pin(Path(__file__)),
        rDriver=pin(ROOT / "kernel_entry.R"), environment=pin(ROOT / "data/numerical_environment.json"),
        nativeEntry=pin(ROOT / "native_inputs.py"), nativeIdentity=pin(ROOT / "data/native_identity.json"),
        native=[pin(native / name) for name in ("resolved_design.json", "initial_panel.json", "h10_canonical.json")],
        numericalRuntime=dict(python=sys.version.split()[0], executable=pin(sys.executable), Rscript=pin(args.rscript)),
        identity="new_public_reproduction",
        legacyPolicy="original_input_seed_semantics_not_fresh_grid_redraw")
    descriptor = write(path, value)
    validate_manifest(descriptor)
    return descriptor


def validate_manifest(descriptor):
    validate(descriptor)
    value = read(descriptor["path"])
    for key in ("sources","driver","rDriver","environment","nativeEntry","nativeIdentity"):
        validate(value[key])
    for item in value["native"]:
        validate(item)
    for key in ("executable","Rscript"):
        validate(value["numericalRuntime"][key])


def initialize_output(args, family, output):
    """Serialize first-start ownership, native preparation and family publication."""
    output = validate_output(output, getattr(args, "reference_dir", None))
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "native_preparation.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require((output / "output_identity.json").exists() or
                all(path == lock_path for path in output.iterdir()),
                "Use a new output directory or this pipeline's existing reproduction output")
        write(output / "source_snapshot.json", inventory())
        write(output / "output_identity.json", dict(schema=SCHEMA,
            sourceInventory=pin(output / "source_snapshot.json"),
            driver=pin(Path(__file__)), rDriver=pin(ROOT / "kernel_entry.R")))
        native = prepare_native(args, output)
        manifest = family_manifest(args, family, output, native)
    return native, manifest


def power_precision(output, native, manifest):
    """Original all-count precision criterion, bound to this public reproduction."""
    from . import power_reference as power
    directory = output / "power" / "precision"
    request_path, result_path = directory / "request.json", directory / "producer.json"
    request = dict(schema="h10-power-design-precision-request-v1", precision=power.PRECISION,
                   reproductionManifest=manifest)
    write(request_path, request)
    if (directory / "independent.json").exists():
        proof = read(directory / "independent.json")
        validate(proof["producer"])
        require(proof["selectedB"] == 2000 and proof["allIntegerCountsChecked"] is True,"Invalid resumed precision proof")
        return
    if not result_path.exists():
        def relocated_context(value):
            require(value == request,"Changed relocated precision request")
            validate(manifest)
            inventory()
            native_pin = pin(native / "h10_canonical.json")
            observed = {str(ROOT / x["path"]): x["sha256"] for x in inventory()["files"]}
            observed.update({manifest["path"]: manifest["sha256"],native_pin["path"]:native_pin["sha256"]})
            return SOURCE,observed,native_pin
        power.precision(str(request_path),str(result_path),source_context=relocated_context)
    result = read(result_path)
    require(result["selectedB"] == 2000 and result["precision"] == power.PRECISION,"Changed original B selection")
    independent = module("independent_power.py")
    evaluations, selected = independent.precision_table()
    require(selected == result["selectedB"],"Independent B selection differs")
    module("independent_family.py").near(result["evaluations"],evaluations,"all-k pointwise CP precision",atol=2e-9,rtol=2e-8)
    write(directory / "independent.json",dict(schema=SCHEMA,producer=pin(result_path),selectedB=selected,
        allIntegerCountsChecked=True,independentBetaInversion=True,newScientificInputs=0,newFits=0))


def compare_reference(family, result, reference_dir):
    """Compare calculated report quantities; references never produce a result."""
    import csv
    import math
    fmt = module("results_format.py")
    if family == "null":
        tables = {"mixed_null_calibration.csv": fmt.null_rows(result)}
    elif family == "paired":
        context = dict(caseID=result["case"]["caseID"])
        coverage = fmt.coverage_rows(result["case"],context)
        for contrast,value in result["oracleCoverage"].items():
            row = dict(context,method="known_Omega_GLS",contrast=contrast)
            row.update(fmt.rate_fields("coverage_all",value["coverageAll"]))
            row.update(fmt.rate_fields("coverage_available",value["coverageApplicable"]))
            coverage.append(row)
        comparisons = [dict(target=target,**value["counts"],coverage_difference=value["coverageDifference"],
            simultaneous_lower95=value["mcSimultaneousDifferenceInterval"][0],
            simultaneous_upper95=value["mcSimultaneousDifferenceInterval"][1],mcnemar_p=value["mcnemarP"],
            holm2_adjusted_p=value["holmAdjustedP"],holm2_reject=value["holmReject"],
            practical_loss_margin=result["primaryPaired"]["practicalDifference"]["margin"])
            for target,value in result["primaryPaired"]["tests"].items()]
        tables = {"paired_coverage.csv":coverage,"paired_family_events.csv":fmt.family_rows(result["case"],context),
                  "paired_comparisons.csv":comparisons}
    else:
        families,coverages = [],[]
        for case in result["cases"]:
            context = fmt.power_context(case)
            families.extend(fmt.family_rows(case,context))
            coverages.extend(fmt.coverage_rows(case,context))
        for row in families:
            matches = [g for g in result["descriptiveGrid"]["effectGrid"] if g["method"] == row["method"] and
                       g["tailMode"] == row["tail"] and any(p["caseID"] == row["caseID"] for p in g["points"])]
            row["effect_grid_status"] = matches[0]["gridStatus"] if matches else "not_effect_grid"
            row["first_supported_multiplier"] = matches[0]["firstPointwiseSupportedMultiplier"] if matches else None
        tables = {"power_coverage.csv":coverages,"power_family_events.csv":families}
    comparisons = []
    for name,rows in tables.items():
        with (reference_dir / name).open() as stream:
            expected = list(csv.DictReader(stream))
        require(len(rows) == len(expected),"Reference table row count differs: "+name)
        for number,(actual,reference) in enumerate(zip(rows,expected),1):
            require(set(actual) == set(reference),"Reference table columns differ: "+name)
            for key,value in actual.items():
                text = "" if value is None else str(value)
                if text == reference[key]:
                    continue
                discrete = (key.startswith("count_") or key.endswith(("_numerator","_denominator")) or
                    key in {"caseIndex","cellIndex","scenarioIndex","nParticipants","planned","available","unavailable",
                            "rejected","covered","recruitedPerArm","fixedEligiblePerArm","a","b","c","d",
                            "M","T","A","C","first_supported_multiplier"})
                require(not discrete,f"Exact discrete reference differs: {name} row{number} {key}")
                try:
                    a,b = float(text),float(reference[key])
                except ValueError:
                    raise ValueError(f"Reference differs: {name} row{number} {key}")
                require(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=2e-9,rel_tol=2e-8),
                        f"Reference numerical difference: {name} row{number} {key}")
        comparisons.append(dict(table=name,rows=len(rows),passed=True,reference=pin(reference_dir / name)))
    return dict(passed=True,tables=comparisons,countsAndIdentities="exact",floatingTolerance=dict(atol=2e-9,rtol=2e-8))


def null_member(args, ordinal, output, native, manifest, *, prepare_only=False):
    from . import null_reference as ni
    from .mixed_tests import directional_result
    plan = null_plan()
    case, rep = plan["cases"][ordinal // 2000], ordinal % 2000 + 1
    root = output / "null"
    unit = root / "units" / f"{ordinal:05d}"
    unit.mkdir(parents=True, exist_ok=True)
    producer = dict(schema="mixed-independent-null-producer-request-v1", namespace=ni.NAMESPACE,
        evidenceRole=ni.ROLE, mode="produce", plan=plan, case=case, repIDs=[rep],
        shardIndex=ordinal+1, shardID=f"portable_{ordinal:05d}", outputDir=str(root),
        paths=dict(generator=str(SOURCE / "null_generation.R"),
                   diagnostics=str(SOURCE / "restart_diagnostics.R"),
                   backend=str(SOURCE / "mixed_fit.R"),
                   design=str(native / "initial_panel.json")),
        designSha256=pin(native / "initial_panel.json")["sha256"],
        newBinding=dict(schema="round10-null-reproduction-binding-v1", reproductionManifest=manifest,
                        frozenDependencies={x["path"]: x["sha256"] for x in source_pins()}))
    producer_path = unit / "producer_request.json"
    write(producer_path, producer)
    request = base_r_request("null")
    request.update(producerRequest=str(producer_path), definitionFiles=[producer["paths"]["generator"],
        producer["paths"]["diagnostics"], str(SOURCE / "null_fit.R")])
    request_path = unit / "request.json"
    write(request_path, request)
    if prepare_only:
        return producer
    receipt = unit / "receipt.json"
    marker = read(unit / "batch.json") if (unit / "batch.json").exists() else None
    if marker:
        validate(marker["receipt"])
    elif not receipt.exists():
        run_r(args, request_path, receipt)
    diagnostic_path = root / "diagnostics" / case["caseID"] / f"rep_{rep:06d}.json.gz"
    original = read(diagnostic_path)
    require(original["case"] == case and original["repID"] == rep and
            original["seed"] == plan["seedBase"]+case["caseIndex"]*10000+rep, "Changed null science identity")
    extraction_path, extraction_result = unit / "independent_request.json", unit / "independent_rds.json"
    rows = [dict(key=method, methodID=method, path=str(root / route["fitArtifact"]["path"]),
                 actualRequestPath=str(root / original["actualRequestPath"]))
            for method, route in original["methods"].items() if route["status"] == "fitted"]
    if marker:
        validate(marker["independent"])
        extraction_result = Path(marker["independent"]["path"])
        prefix = str(ordinal)+":"
        facts = {row["methodID"]: row for row in read(extraction_result)["rows"] if row["key"].startswith(prefix)}
    else:
        write(extraction_path, dict(schema="mixed-null-repair-independent-rds-request-v1", rows=rows))
        if not extraction_result.exists():
            run_r(args, extraction_path, extraction_result, script=SOURCE / "independent_null_rds.R")
        facts = {row["key"]: row for row in read(extraction_result)["rows"]}
    independent = module("independent_serialization.py")
    canonical, changes = independent.canonical_view(original, facts, root / original["actualRequestPath"])
    # Only independently RDS-proven numeric -Inf lower positions become null.
    write(unit / "canonical_diagnostic.json", canonical)
    write(unit / "serialization_proof.json", dict(changes=changes, newFits=0, newScientificInputs=0))
    technical = ni.technical_completion(canonical)
    decision, error = ni.resolve_fresh(canonical, technical)
    require(technical["complete"] and error is None, "Unresolved null calculation; retain artifacts and do not redraw")
    base = canonical["methods"][ni.frozen.ROUTES[0]]
    inference = directional_result(base["raw"]["references"]["delta"], "less" if case["H"] == "H5" else "greater",
                                  usable=base["raw"]["diagnostics"]["usableCandidateInference"] is True) if base["status"] == "fitted" else None
    baseline = dict(status=inference["status"] if inference else "unavailable",
        reason=inference.get("reason") if inference else base.get("error"),
        p=inference.get("p") if inference else None, reject=inference.get("candidateNominalReject") is True if inference else False,
        **{key: inference.get(key) if inference else None for key in ("estimate", "SE", "df")})
    # The independent historical mathematical helper's input schema is a
    # projection convention. Its parity flags here are proved by actual RDS/raw
    # comparison above, not an assertion of a historical job or archive receipt.
    kernel_record = dict(schema="mixed-convergence-diagnostic-record-v1", oldCase=case,
        request=canonical["request"], methods=canonical["methods"], scientificAdoption=False,
        historicalInferenceChanged=False,
        baselineParity=dict(passed=ni.frozen.ROUTES[0] in facts,
            nontrivialFiniteReference=bool(inference and inference.get("SE", 0) and inference.get("df", 0))))
    decision_reader = module("independent_restart.py")
    decision_reader.verify_decision(kernel_record, decision)
    record = dict(caseID=case["caseID"], repID=rep, case=case, baselineDecision=baseline,
                  baselineInference=inference if baseline["status"] == "available_candidate" else None,
                  decision=decision, technicalCompletion=technical)
    compact = ni.compact(record, canonical)
    return dict(compact, ordinal=ordinal), [pin(diagnostic_path), pin(extraction_result),
        pin(unit / "canonical_diagnostic.json"), pin(root / original["actualRequestPath"])] + [pin(row["path"]) for row in rows]


@lru_cache(maxsize=1)
def joint_modules():
    return dict(math=module("independent_cluster.py"),
                accepted=module("independent_family.py"),
                generation=module("independent_generation.py"),
                reduction=module("independent_fresh.py"))


def original_shift_source(case, library, generation_reader, output):
    """Reuse each original fresh-S1 input across its four deterministic shifts."""
    from . import joint_generation as m
    directory = output / "original_shift_sources" / f"{case['caseID']}__r{case['repID']:04d}"
    directory.mkdir(parents=True,exist_ok=True)
    with (directory / "source.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        commitment = directory / "commit.json"
        if commitment.exists():
            value = read(commitment)
            require(value["case"] == case and value["sourceInventory"] == pin(output / "source_snapshot.json") and
                    value["nativeCanonical"] == pin(output / "native/h10_canonical.json"),
                    "Changed original shift-source identity")
            validate(value["request"]); validate(value["generation"])
            return read(value["request"]["path"]),read(value["generation"]["path"]),dict(
                commitment=pin(commitment),request=value["request"],generation=value["generation"],nativeCanonical=value["nativeCanonical"])
        require(not (directory / "request.json").exists() and not (directory / "generation.json").exists(),
                "Uncommitted original source exists; inspect instead of silently regenerating")
        actual,generation = m.generate(case,library)
        independent,_ = generation_reader.regenerate(case,library)
        require(actual == independent,"Independent original fresh source differs")
        request_pin = write(directory / "request.json",actual)
        generation_pin = write(directory / "generation.json",generation)
        write(commitment,dict(schema=SCHEMA,case=case,request=request_pin,generation=generation_pin,
            sourceInventory=pin(output / "source_snapshot.json"),independentInputReplay=True,
            nativeCanonical=pin(output / "native/h10_canonical.json"),
            execution=dict(pid=os.getpid())))
        return actual,generation,dict(commitment=pin(commitment),request=request_pin,generation=generation_pin,
            nativeCanonical=pin(output / "native/h10_canonical.json"))


def joint_input(family, ordinal, library, generation_reader, output):
    from . import joint_generation as m, paired_reference as paired
    from . import power_reference as power, power_origins as legacy
    from . import effect_shift as shift, fresh_reference as fresh
    if family == "paired":
        case = paired.case_for(ordinal+1)
        actual, generation = m.generate(case, library)
        independent_actual, _ = generation_reader.regenerate(case, library)
        require(actual == independent_actual, "Independent paired input generation differs")
        return case, paired.fit_key(case), actual, generation, None, {arm: 154 for arm in m.ARMS}
    case = power.case_for(ordinal, 2000)
    origin = legacy.planned_origin(case["caseID"], case["repID"])
    require(origin == module("independent_power.py").legacy_origin(case),
            "Independent legacy origin/seed mapping differs")
    if origin is None:
        actual, generation = power.generate(case, library)
        counts = generation["populationCounts"]
    elif origin["kind"] == "original_production":
        old = copy.deepcopy(m.measurement_plan()["cases"][origin["caseIndex"]-1])
        old.update(repID=origin["repID"], designSeed=origin["designSeed"], responseSeed=origin["responseSeed"])
        require(old["caseID"] == origin["caseID"] and old["delta"] == case["delta"] and old["parameters"] == case["parameters"],
                "Legacy baseline target differs")
        actual, generation = m.generate(old, library)
        independent_actual, _ = generation_reader.regenerate(old, library)
        require(actual == independent_actual, "Independent legacy baseline generation differs")
        counts = {arm: 154 for arm in m.ARMS}
    else:
        require(origin["kind"] == "actual_shift", "Unknown historical origin")
        identity = shift.identity(origin["ordinal"], "parity")
        old = fresh.ordinal_case(identity["sourceOrdinal"])
        require(old["designSeed"] == origin["designSeed"] and old["responseSeed"] == origin["responseSeed"] and
                identity["caseID"] == origin["caseID"] and identity["case"]["delta"] == case["delta"], "Legacy shift identity changed")
        base, source_generation, source_inputs = original_shift_source(old,library,generation_reader,output)
        actual = shift.shifted_request(base, case["delta"])
        generation = dict(schema="round10-original-shift-reproduction-v1", origin=origin,
                          sourceGeneration=source_generation, sourceInputs=source_inputs,responseTransform="deterministic_X_d_addition",
                          newRandomDrawsForShift=0, newIndependentScientificInputsForShift=0)
        counts = {arm: 154 for arm in m.ARMS}
    # Actual historical labels/streams are retained. The outer grid coordinates
    # identify consumption, never rename the generated participants or seed.
    return case, power.fit_key(case), actual, generation, origin, counts


@lru_cache(maxsize=2)
def joint_library(native):
    from . import joint_generation as m
    return m.schedule_library(read(native / "h10_canonical.json"))


def joint_member(args, family, ordinal, output, native, manifest, *, prepare_only=False):
    from . import joint_generation as m, cluster_inference as hi
    from . import power_reference as power, paired_reference as paired
    modules = joint_modules()
    reader = module("independent_power.py")
    library = joint_library(native)
    root, unit = output / family, output / family / "units" / f"{ordinal:05d}"
    unit.mkdir(parents=True, exist_ok=True)
    # Never repeat RNG after a committed input exists; authenticate its exact
    # scientific identity and original seed map before reusing it.
    spec_path = unit / "spec.json"
    if spec_path.exists():
        spec = read(spec_path)
        require(spec["ordinal"] == ordinal and spec["manifest"] == manifest, "Changed resumed spec")
        for key in ("request", "generation"):
            validate(spec[key])
        actual = read(spec["request"]["path"])
    else:
        case, key, actual, generation, origin, counts = joint_input(family, ordinal, library, modules["generation"],output)
        request_pin, generation_pin = write(unit / "actual_request.json", actual), write(unit / "generation.json", generation)
        spec = dict(ordinal=ordinal, caseID=case["caseID"], repID=case["repID"], fitKey=key, case=case,
            request=request_pin, generation=generation_pin, populationCounts=counts,
            origin=origin, manifest=manifest, support=power.support(actual, counts))
        if family == "power" and origin is None:
            spec["execution"] = generation["execution"]
        write(spec_path, spec)
    case, key = spec["case"], spec["fitKey"]
    expected = paired.case_for(ordinal+1) if family == "paired" else power.case_for(ordinal, 2000)
    require(case == expected, "Resumed scientific case differs")
    if family == "power":
        from . import power_origins as legacy
        require(spec["origin"] == legacy.planned_origin(case["caseID"],case["repID"]) == reader.legacy_origin(case),
                "Resumed independent legacy origin/seed map differs")
    input_proof = unit / "input_identity.json"
    if input_proof.exists():
        require(read(input_proof)["spec"] == pin(spec_path),"Changed independently admitted input")
    elif family == "power" and spec["origin"] is None:
        class GenerationObservation:
            @staticmethod
            def read(descriptor):
                validate(descriptor)
                return read(descriptor["path"])
        independent_actual, _ = reader.replay_new(spec, dict(modules=modules, library=library, observed=GenerationObservation()))
        require(actual == independent_actual, "Independent power input generation differs")
        write(input_proof,dict(schema=SCHEMA,spec=pin(spec_path),independentGeneration=True))
    else:
        write(input_proof,dict(schema=SCHEMA,spec=pin(spec_path),independentGeneration=True))
    reader.support_equal(spec["support"], reader.expected_support(actual["rows"]), actual["rows"])
    if prepare_only:
        return spec
    run_request = base_r_request("h10")
    run_request.update(scientificFamily=family,outputDir=str(root / "numerical"), manifest=manifest, batch=pin(spec_path), fits=[spec],
        backend=pin(SOURCE / "mixed_fit.R"),
        baselineKernel=str(SOURCE / "joint_fit.R"),
        candidateKernel=str(SOURCE / "cluster_fit.R"),
        powerKernel=str(SOURCE / "population_fit.R"))
    request_path, receipt_path = unit / "request.json", unit / "receipt.json"
    marker = read(unit / "batch.json") if (unit / "batch.json").exists() else None
    if marker:
        validate(marker["receipt"])
        receipt = read(marker["receipt"]["path"])
    else:
        write(request_path, run_request)
        receipt = read(receipt_path) if receipt_path.exists() else run_r(args, request_path, receipt_path)
    entries = [row for row in receipt["records"] if row["fitKey"] == key]
    require(len(entries) == 1,"Missing or duplicate batch member")
    entry = entries[0]
    for name in ("baseline", "candidate"):
        validate(entry[name])
    base, candidate = read(entry["baseline"]["path"]), read(entry["candidate"]["path"])
    require(candidate["baseline"] == entry["baseline"] and candidate["request"] == base["request"] == spec["request"] and
            candidate["generation"] == base["generation"] == spec["generation"] and
            candidate["fitArtifact"] == base["fitArtifact"] and base["manifestSha256"] == manifest["sha256"] and
            candidate["manifest"] == manifest and base["fitKey"] == candidate["fitKey"] == key and
            base["caseID"] == candidate["caseID"] == case["caseID"] and base["repID"] == candidate["repID"] == case["repID"],
            "Broken producer-to-reader binding")
    evidence_path, extraction_request = unit / "independent_rds.json", unit / "independent_request.json"
    raw, independent_raw = base["raw"], candidate["rawCandidate"]
    artifacts = [pin(spec_path), spec["request"], spec["generation"], entry["baseline"], entry["candidate"]]
    if spec["origin"] and spec["origin"]["kind"] == "actual_shift":
        for descriptor in read(spec["generation"]["path"])["sourceInputs"].values():
            validate(descriptor)
            artifacts.append(descriptor)
    for name in ("fitArtifact", "rawArtifact"):
        if base.get(name):
            validate(base[name]); artifacts.append(base[name])
    if base.get("rawArtifact"):
        require(read(base["rawArtifact"]["path"]) == raw,"Original emitted raw artifact differs")
    if (candidate.get("candidateMetadata") or {}).get("calculationSkipped"):
        require(family == "power" and raw["status"] == "fitted" and raw["diagnostics"]["usableCandidateInference"] is False
                and candidate["rawCandidate"] == dict(status="unavailable",reason="original_guard_unavailable",diagnostics=raw["diagnostics"])
                and candidate["candidateMetadata"]["CR2CovarianceComputed"] is False,"Changed original-guard skip")
    if base.get("fitArtifact"):
        if marker:
            validate(marker["independent"])
            evidence_path = Path(marker["independent"]["path"])
        else:
            write(extraction_request, dict(schema="h10-inference-independent-rds-request-v1",
                fits=[dict(fitKey=key, requestPath=spec["request"]["path"], rdsPath=base["fitArtifact"]["path"])]))
            if not evidence_path.exists():
                run_r(args, extraction_request, evidence_path, script=SOURCE / "independent_cluster_rds.R")
        evidence = read(evidence_path)["fits"][key]
        artifacts.append(pin(evidence_path))
        if marker:
            validate(marker["transport"])
            transport_path = Path(marker["transport"]["path"])
        else:
            transport_path = source_transport(args,[spec],[entry],unit)
        transported = read(transport_path)
        require(transported["schema"] == "h10-power-source-transport-v1" and
                transported["newScientificInputs"] == transported["newFitInvocations"] == transported["newCandidateCalculations"] == 0,
                "Changed source transport identity")
        source = transported["fits"][key]
        require(source["request"] == spec["request"] and source["fitArtifact"] == base["fitArtifact"] and
                source["exactMatrixInR"] is True and source["exactResponseInR"] is True and source["oneULPMatrixMutationRejected"] is True,
                "Missing exact pre-serialization request/RDS check")
        artifacts.append(pin(transport_path))
        fit_reader = module("independent_transport.py")
        transport_key = spec["request"]["path"],spec["request"]["sha256"]
        fit_reader.SOURCE_TRANSPORT[transport_key] = source["design"]
        try:
            if raw["status"] == "fitted":
                facts, L = fit_reader.independent_fit(raw, evidence, spec["request"], actual["rows"], modules)
                if candidate["technicalCompleted"] and independent_raw["status"] == "fitted":
                    modules["accepted"].near(independent_raw["diagnostics"], raw["diagnostics"], "unchanged original guard")
                    independent_raw = fit_reader.independent_candidate(independent_raw, evidence, facts, L, modules)
            else:
                require(evidence["verified"] is True and evidence["requestDescriptor"] == spec["request"], "Unverified failed-fit frame")
        finally:
            # Only the current exact-transport readback needs this matrix.
            # The imported helper is cached, while population-sized designs are not.
            fit_reader.SOURCE_TRANSPORT.pop(transport_key,None)
    else:
        require(raw["status"] == "unavailable" and raw["reason"] in ("backend_fit_error", "design_support_unavailable"), "Unclassified baseline absence")
    require(candidate["technicalCompleted"] is True and candidate["technicalError"] is None,
            "Unresolved candidate computation; retain artifacts and do not redraw")
    baseline_branches = [m.branch_result(case, mode, raw) for mode in m.TAILS]
    candidate_branches = [m.branch_result(case, mode, candidate["rawCandidate"]) for mode in m.TAILS]
    checked_branches = [modules["generation"].branch_outcome(case, mode, independent_raw, modules["accepted"]) for mode in m.TAILS]
    # The independent branch helper is checked below against exact production
    # events; scientific unavailability and Holm padding remain unchanged.
    for production, independent in zip(candidate_branches, checked_branches):
        for field in ("tailMode","trueNullIDs","falseNullIDs","completeFamilyAvailable","counts","coverage","rates","fwerApplicability"):
            modules["accepted"].near(production[field],independent[field],"independent candidate branch "+field)
        for contrast in m.H10_IDS:
            for field in ("status","p","candidateNominalReject","holmP","holmReject"):
                modules["accepted"].near(production["tests"][contrast][field],independent["tests"][contrast][field],"independent Holm event "+field)
    row = dict(ordinal=ordinal, caseID=case["caseID"], repID=case["repID"], fitKey=key, case=case,
        populationCounts=spec["populationCounts"], origin="legacy_import" if spec["origin"] else "new_population_input",
        technicalCompleted=True, technicalError=None, diagnostics=raw.get("diagnostics"),
        candidateMetadata=candidate["candidateMetadata"], baselineBranches=baseline_branches,
        candidateBranches=candidate_branches, paired=hi.paired_result(baseline_branches,candidate_branches))
    if family == "paired":
        oracle = module("oracle_gls.py")
        independent_oracle = module("independent_gls.py")
        source = paired.oracle_input(actual, oracle)
        calculated = oracle.known_gls(source, case)
        p = case["parameters"]
        checked = independent_oracle.gls_oracle(source, sd_u=p["sdU"], residual_sds=p["sigmaByArm"],
            truths=case["delta"], parameters=p)
        # Both original oracles expose the same reference fields and independent
        # normal-equation versus QR calculations; compare the declared result.
        modules["accepted"].near(calculated, checked, "independent known-Omega GLS")
        row.update(cr2TechnicalCompleted=True, cr2TechnicalError=None, oracleTechnicalCompleted=True,
            oracleTechnicalError=None, oracleApplicable=True, oracleResult=calculated,
            pairedOracle=paired.pair_events(candidate_branches,dict(technicalCompleted=True,applicable=True,result=calculated)))
        compact = module("independent_paired.py").compact_events(row, modules["reduction"])
    else:
        row.update(cellIndex=case["cellIndex"])
        compact = reader.compact_scalar(row, modules["reduction"])
    return compact, artifacts


def source_transport(args, specifications, entries, directory):
    request_path,output_path = directory / "source_request.json",directory / "source_transport.json"
    fits = []
    for spec,entry in zip(specifications,entries):
        baseline = read(entry["baseline"]["path"])
        if baseline.get("fitArtifact"):
            fits.append(dict(fitKey=spec["fitKey"],request=spec["request"],fitArtifact=baseline["fitArtifact"]))
    write(request_path,dict(schema="h10-power-source-transport-request-v1",fits=fits))
    if not output_path.exists():
        run_r(args,request_path,output_path,script=SOURCE / "source_transport.R")
    return output_path


def prime_batch(args, family, ordinals, output, native, manifest):
    """One producer and one independent R process per bounded disjoint block."""
    missing = [o for o in ordinals if not (output / family / "events" / f"{o:05d}.json").exists() and
               not (output / family / "units" / f"{o:05d}" / "batch.json").exists()]
    if not missing:
        return
    token = hashlib.sha256(json.dumps(missing).encode()).hexdigest()[:16]
    directory = output / family / "batches" / token
    directory.mkdir(parents=True,exist_ok=True)
    request_path,receipt_path = directory / "request.json",directory / "receipt.json"
    independent_request,independent_path = directory / "independent_request.json",directory / "independent.json"
    if family == "null":
        require(len({o//2000 for o in missing}) == 1 and len(missing) <= 100,"Null batch must be one original case,at most100 IDs")
        producers = [null_member(args,o,output,native,manifest,prepare_only=True) for o in missing]
        producer = copy.deepcopy(producers[0])
        producer["repIDs"] = [value["repIDs"][0] for value in producers]
        combined = directory / "producer_request.json"
        write(combined,producer)
        request = base_r_request("null")
        request.update(producerRequest=str(combined),definitionFiles=[producer["paths"]["generator"],
            producer["paths"]["diagnostics"],str(SOURCE / "null_fit.R")])
        write(request_path,request)
        if not receipt_path.exists():
            run_r(args,request_path,receipt_path)
        rows = []
        for ordinal,individual in zip(missing,producers):
            rep = individual["repIDs"][0]
            diagnostic = read(output / "null" / "diagnostics" / individual["case"]["caseID"] / f"rep_{rep:06d}.json.gz")
            for method,route in diagnostic["methods"].items():
                if route["status"] == "fitted":
                    rows.append(dict(key=f"{ordinal}:{method}",methodID=method,
                        path=str(output / "null" / route["fitArtifact"]["path"]),
                        actualRequestPath=str(output / "null" / diagnostic["actualRequestPath"])))
        write(independent_request,dict(schema="mixed-null-repair-independent-rds-request-v1",rows=rows))
        if not independent_path.exists():
            run_r(args,independent_request,independent_path,script=SOURCE / "independent_null_rds.R")
    else:
        fits = [joint_member(args,family,o,output,native,manifest,prepare_only=True) for o in missing]
        batch_pin = write(directory / "selection.json",dict(manifest=manifest,ordinals=missing,specifications=[
            pin(output / family / "units" / f"{o:05d}" / "spec.json") for o in missing]))
        request = base_r_request("h10")
        request.update(scientificFamily=family,outputDir=str(output / family / "numerical"),manifest=manifest,
            batch=batch_pin,fits=fits,backend=pin(SOURCE / "mixed_fit.R"),
            baselineKernel=str(SOURCE / "joint_fit.R"),
            candidateKernel=str(SOURCE / "cluster_fit.R"),
            powerKernel=str(SOURCE / "population_fit.R"))
        write(request_path,request)
        receipt = read(receipt_path) if receipt_path.exists() else run_r(args,request_path,receipt_path)
        rows = []
        for spec,entry in zip(fits,receipt["records"]):
            require(entry["fitKey"] == spec["fitKey"],"R batch roster order changed")
            validate(entry["baseline"])
            baseline = read(entry["baseline"]["path"])
            if baseline.get("fitArtifact"):
                rows.append(dict(fitKey=spec["fitKey"],requestPath=spec["request"]["path"],rdsPath=baseline["fitArtifact"]["path"]))
        write(independent_request,dict(schema="h10-inference-independent-rds-request-v1",fits=rows))
        if not independent_path.exists():
            run_r(args,independent_request,independent_path,script=SOURCE / "independent_cluster_rds.R")
        transport_path = source_transport(args,fits,receipt["records"],directory)
    for ordinal in missing:
        marker = dict(schema=SCHEMA,manifest=manifest,ordinal=ordinal,receipt=pin(receipt_path),
                      independent=pin(independent_path),selection=missing)
        if family != "null":
            marker["transport"] = pin(transport_path)
        write(output / family / "units" / f"{ordinal:05d}" / "batch.json",marker)


def event_digest(event):
    return hashlib.sha256(json.dumps(event,sort_keys=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()


def admit_event(commitment, family, manifest):
    require(commitment["schema"] == SCHEMA and commitment["manifest"] == manifest,"Foreign event commitment")
    event = commitment["event"]
    require(commitment["eventSha256"] == event_digest(event),"Committed event payload changed")
    ordinal = event["ordinal"]
    require(type(ordinal) is int and 0 <= ordinal < TOTAL[family],"Event ordinal out of range")
    if family == "null":
        from . import null_reference as ni
        cases = null_plan()["cases"]
        case = cases[ordinal//2000]
        require(event["caseID"] == case["caseID"] and event["repID"] == ordinal%2000+1 and
                event["truth"] == ni.summaries.truth(case),"Null case/rep/ordinal/declared truth differs")
    elif family == "paired":
        from . import paired_reference as paired
        case = paired.case_for(ordinal+1)
        require(event["case"] == case and event["caseID"] == case["caseID"] and event["repID"] == ordinal+1 and
                event["fitKey"] == paired.fit_key(case),"Paired exact scientific ID differs")
    else:
        from . import power_reference as power
        reader = module("independent_power.py")
        case = power.case_for(ordinal,2000)
        require(event["caseID"] == case["caseID"] and event["repID"] == case["repID"] and
                event["fitKey"] == power.fit_key(case) and event["origin"] ==
                ("legacy_import" if reader.legacy_origin(case) else "new_population_input"),"Power exact scientific ID/origin differs")
    return event


def aggregate(args, family, output, manifest):
    validate_manifest(manifest)
    directory = output / family
    events = []
    for path in sorted((directory / "events").glob("*.json")):
        commitment = read(path)
        event = admit_event(commitment,family,manifest)
        for descriptor in commitment["artifacts"]:
            validate(descriptor)
        require(path.stem == f"{event['ordinal']:05d}","Event filename/ordinal differs")
        events.append(event)
    require(events and len({row["ordinal"] for row in events}) == len(events), "Empty or duplicate event roster")
    complete = {row["ordinal"] for row in events} == set(range(TOTAL[family]))
    if family == "null":
        from . import null_reference as ni
        plan = null_plan()
        reader = module("independent_null.py")
        reports = []
        for case in plan["cases"]:
            own = {r["repID"]: r for r in events if r["caseID"] == case["caseID"]}
            report = ni.case_summary(case, list(range(1,2001)), own)
            roster = [dict(repID=rep, oldDecision=own[rep]["baselineDecision"] if rep in own else
                dict(status="unavailable",reject=False), oldInference=own[rep]["baselineInference"] if rep in own else None)
                for rep in range(1,2001)]
            transformed = {rep: dict(row,oldDecision=row["baselineDecision"],oldInference=row["baselineInference"]) for rep,row in own.items()}
            independent = reader.case_report(dict(case=case,records=roster),transformed)
            for name in ("available","unavailable","rejected","covered"):
                require(report["candidate"][name] == independent["candidate"][name], "Independent null count disagreement")
            for key in ("rejectionAll","rejectionAmongAvailable","coverageAmongAvailable"):
                reader.same(report["candidate"]["rates"][key],independent["candidate"]["rates"][key],tolerance=1e-10)
            reports.append(report)
        result = dict(cases=reports)
    elif family == "paired":
        from . import paired_reference as paired
        reducer = module("independent_paired.py")
        modules = joint_modules()
        result = reducer.reduce_events(events,dict(case=paired.case_for(1),modules=dict(fresh=modules["reduction"],math=modules["math"])),planned=10000)
    else:
        from . import power_reference as power
        reader = module("independent_power.py")
        modules = joint_modules()
        cases, seen = reader.independent_reduction(events,dict(manifest=dict(cells=power.cells(),B=2000),modules=modules))
        result = dict(cases=cases,descriptiveGrid=reader.grid_description(cases),B=2000)
        require(seen == {r["ordinal"] for r in events}, "Independent aggregate roster disagreement")
    result.update(schema=SCHEMA,family=family,reproductionManifest=manifest,expectedTotal=TOTAL[family],
        presentTotal=len(events),fullRosterComplete=complete,status="full_reproduction" if complete else "partial_entry_check",
        independentNumericalReadback=True,comparison="optional_external_reference_only")
    result["referenceComparison"] = compare_reference(family,result,args.reference_dir) if complete and args.reference_dir else dict(passed=None,
        reason="optional_reference_not_requested" if complete else "partial_selector_cannot_establish_full_reference_agreement")
    # Multiple disjoint submissions may append events; aggregate filenames are
    # immutable by admitted roster, rather than overwriting a previous result.
    token = hashlib.sha256(json.dumps(sorted(r["ordinal"] for r in events)).encode()).hexdigest()[:16]
    validate_manifest(manifest)
    inventory()
    descriptor = write(directory / ("aggregate_full.json" if complete else f"aggregate_partial_{token}.json"),result)
    print(json.dumps(dict(aggregate=descriptor,fullRosterComplete=complete,present=len(events),expected=TOTAL[family])))
    return descriptor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("plan","run","aggregate"),default="plan")
    parser.add_argument("--family",choices=tuple(TOTAL),default="null")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--ids",help="Explicit zero-based ordinal list, e.g. 0,4,32")
    parser.add_argument("--first",type=int)
    parser.add_argument("--count",type=int)
    parser.add_argument("--full",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--batch-size",type=int,default=20,help="At most100; bounded block amortizes R startup and source admission")
    parser.add_argument("--no-aggregate",action="store_true",help="For disjoint array blocks; reduce once after all blocks finish")
    parser.add_argument("--rscript",default=os.environ.get("TUTORING_RSCRIPT","Rscript"))
    parser.add_argument("--node",default="node")
    parser.add_argument("--reference-dir",type=Path,help="Optional external comparison tables; never a computation input")
    args = parser.parse_args()
    inventory()
    if args.action == "plan":
        require(not args.execute,"Plan never executes scientific code")
        print(json.dumps(dict(schema=SCHEMA,families=TOTAL,defaultFull=False,
            ordinals=dict(null="case-major 12 selected original indices x rep1..2000",
                paired="rep1..10000 minus one",power="rep-major,40 original cells,rep1..2000"),
            sourceInventory=str(ROOT),runtime=str(ROOT / "data/numerical_environment.json")),indent=2))
        return
    require(args.execute and args.output is not None,"Actual work requires --execute and --output")
    ids = None
    if args.action == "run":
        choices = sum((bool(args.full),args.ids is not None,args.first is not None or args.count is not None))
        require(choices == 1,"Choose exactly one of --full, --ids, or --first/--count")
        if args.full:
            ids = list(range(TOTAL[args.family]))
        elif args.ids is not None:
            ids = [int(x) for x in args.ids.split(",")]
        else:
            require(args.first is not None and args.count is not None and args.count > 0,"Explicit positive --count required")
            ids = list(range(args.first,args.first+args.count))
        require(ids and ids == sorted(set(ids)) and all(0 <= x < TOTAL[args.family] for x in ids),"Duplicate, unordered or out-of-range roster")
        require(1 <= args.batch_size <= 100,"Batch size must be1..100")
    output = validate_output(args.output, args.reference_dir)
    numerical_runtime()
    import shutil
    require(shutil.which("sha256sum") is not None,
            "GNU coreutils sha256sum is required for exact artifact binding")
    executable = shutil.which(args.rscript)
    require(executable is not None,"Rscript executable unavailable")
    args.rscript = str(Path(executable).resolve())
    native, manifest = initialize_output(args, args.family, output)
    if args.family == "power":
        with (output / "precision.lock").open("a+") as precision_lock:
            fcntl.flock(precision_lock,fcntl.LOCK_EX)
            power_precision(output,native,manifest)
    if args.action == "aggregate":
        aggregate(args,args.family,output,manifest)
        return
    # Real execution and source identity are pinned separately from historical
    # scientific RNG identity. No environment variable is manufactured here.
    blocks = []
    for ordinal in ids:
        if not blocks or len(blocks[-1]) >= args.batch_size or (args.family == "null" and blocks[-1][-1]//2000 != ordinal//2000):
            blocks.append([])
        blocks[-1].append(ordinal)
    for block in blocks:
        with ExitStack() as stack:
            for ordinal in block:
                lock_path = output / args.family / "locks" / f"{ordinal:05d}.lock"
                lock_path.parent.mkdir(parents=True,exist_ok=True)
                lock = stack.enter_context(lock_path.open("a+"))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            prime_batch(args,args.family,block,output,native,manifest)
            for ordinal in block:
                event_path = output / args.family / "events" / f"{ordinal:05d}.json"
                if event_path.exists():
                    committed = read(event_path)
                    event = admit_event(committed,args.family,manifest)
                    require(event["ordinal"] == ordinal,"Foreign resumed event")
                    for descriptor in committed["artifacts"]:
                        validate(descriptor)
                    continue
                if args.family == "null":
                    event,artifacts = null_member(args,ordinal,output,native,manifest)
                else:
                    event,artifacts = joint_member(args,args.family,ordinal,output,native,manifest)
                write(event_path,dict(schema=SCHEMA,manifest=manifest,event=event,eventSha256=event_digest(event),artifacts=artifacts,
                    execution=dict(pid=os.getpid())))
    inventory()
    validate_manifest(manifest)
    if not args.no_aggregate:
        aggregate(args,args.family,output,manifest)


if __name__ == "__main__":
    main()
