"""Portable reproduction of the closed absolute-slope magnitude study.

Plans never generate inputs. Execution retains fixed case/seed identities and
requires an explicit selector. Saved-power analysis reuses committed fits from
the conditional-statistics workflow instead of fitting them again.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from . import reproduction as io
from .saved_fit_inputs import Observations

ROOT = Path(__file__).resolve().parent
SCHEMA = "magnitude-study-reproduction-v1"
INPUT_SCHEMA = "magnitude-study-fit-input-v1"
MODES = ("fresh_validation", "saved_power_core")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plan(mode):
    from .magnitude_design import manifest
    return dict(schema=SCHEMA, identity="new_public_reproduction", **manifest(mode),
                methodAdoptedForHumanData=False, establishesHumanPower=False,
                scope="fixed_finite_validation_and_conditional_power")


def select_ids(manifest, ids=None, start=None, count=None, all_ids=False):
    roster = manifest["expectedOrdinals"]
    require(sum((ids is not None, start is not None, bool(all_ids))) == 1,
            "Choose exactly one of --ids, --start/--count, or --all")
    if ids is not None:
        require(count is None, "--count requires --start")
        selected = ids
    elif all_ids:
        require(count is None, "--count requires --start")
        selected = roster
    else:
        require(type(start) is int and start >= 0 and type(count) is int and count > 0
                and start + count <= len(roster), "Invalid roster-position slice")
        selected = roster[start:start+count]
    require(type(selected) is list and selected and all(type(v) is int for v in selected)
            and selected == sorted(set(selected)) and set(selected) <= set(roster),
            "IDs must be nonempty, unique, ordered members of the fixed roster")
    return selected


def source_files():
    return io.inventory()["files"]


def initialize(output, mode, args):
    require((mode == "saved_power_core") == bool(args.power_directory),
            "Saved power requires --power-directory; fresh validation must not use it")
    output = io.validate_output(output, getattr(args, "power_directory", None))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "initialize.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = output / "manifest.json"
        if marker.exists():
            value = io.read(marker)
            require(value["plan"] == plan(mode), "Changed resumed scientific plan")
            io.validate(value["sources"])
            require(io.read(value["sources"]["path"])["files"] == source_files(),
                    "Changed resumed source inventory")
            require(value["powerDirectory"] == (str(Path(args.power_directory).resolve())
                    if args.power_directory else None), "Changed saved-power input directory")
            return io.pin(marker)
        require({p.name for p in output.iterdir()} == {"initialize.lock"},
                "Use a new output directory; preserve incomplete preparations")
        inventory = io.write(output / "sources.json", dict(files=source_files()))
        observed = Observations()
        from .saved_fit_runtime import runtime
        environment = runtime(args.rscript, observed, output / "environment.json")
        native = io.prepare_native(args, output)
        native_pin = io.pin(native / "h10_canonical.json")
        value = dict(schema=SCHEMA, plan=plan(mode), sources=inventory,
                     driver=io.pin(Path(__file__)), native=native_pin,
                     environment=environment, runtimeInputs=dict(observed.files),
                     powerDirectory=str(Path(args.power_directory).resolve()) if args.power_directory else None)
        return io.write(marker, value)


def observe_manifest(manifest_pin, observed):
    manifest = observed.read(manifest_pin)
    require(manifest["schema"] == SCHEMA and manifest["plan"] == plan(manifest["plan"]["mode"]),
            "Changed scientific manifest")
    require(observed.read(manifest["sources"])["files"] == source_files(), "Changed source inventory")
    for descriptor in io.read(manifest["sources"]["path"])["files"]:
        observed.observe(descriptor["path"], descriptor["sha256"])
    for path, digest in manifest["runtimeInputs"].items():
        observed.observe(path, digest)
    observed.read(manifest["native"])
    return manifest


def fresh_spec(manifest, manifest_pin, ordinal, directory):
    from . import magnitude_design as design, power_reference as generator
    from . import joint_generation, independent_generation, independent_power
    expected = design.case_for(manifest["plan"], ordinal)
    path = directory / "spec.json"
    if path.exists():
        spec = io.read(path)
        require(spec["case"] == expected and spec["manifest"] == manifest_pin
                and spec["ordinal"] == ordinal, "Changed resumed input identity")
        io.validate(spec["request"]); io.validate(spec["generation"])
        proof = io.read(directory / "generation_proof.json")
        require(proof == dict(spec=io.pin(path), manifest=manifest_pin,
                             request=spec["request"], generation=spec["generation"],
                             independentReplay=True), "Missing or changed independent generation proof")
        return spec, io.pin(path)
    require(not (directory / "actual_request.json").exists()
            and not (directory / "generation.json").exists(),
            "Incomplete input remains; do not silently redraw")
    library = joint_generation.schedule_library(io.read(manifest["native"]["path"]))
    actual, generation = generator.generate(expected, library)
    request_pin = io.write(directory / "actual_request.json", actual)
    generation_pin = io.write(directory / "generation.json", generation)
    spec = dict(ordinal=ordinal, caseID=expected["caseID"], repID=expected["repID"],
                fitKey=f"{expected['caseID']}__r{expected['repID']:04d}", case=expected,
                request=request_pin, generation=generation_pin,
                populationCounts=generation["populationCounts"], execution=generation["execution"],
                retentionMask=generation["retentionMask"], recruitedCounts=generation["recruitedCounts"],
                support=generator.support(actual, generation["populationCounts"]), manifest=manifest_pin)
    class Reader:
        @staticmethod
        def read(descriptor):
            io.validate(descriptor)
            return io.read(descriptor["path"])
    rebuilt, independent = independent_power.replay_new(spec, dict(
        modules=dict(generation=independent_generation), library=library, observed=Reader()))
    require(actual == rebuilt, "Independent generated input differs")
    independent_power.exact(generation, independent, "independent generation")
    spec_pin = io.write(path, spec)
    io.write(directory / "generation_proof.json", dict(spec=spec_pin, manifest=manifest_pin,
             request=request_pin, generation=generation_pin, independentReplay=True))
    return spec, spec_pin


def fit_fresh(manifest, manifest_pin, spec, spec_pin, directory):
    request_path = directory / "fit_request.json"
    receipt_path = directory / "fit_receipt.json"
    request = io.base_r_request("h10")
    request.update(scientificFamily="power", outputDir=str(directory / "numerical"),
                   manifest=manifest_pin, batch=spec_pin, fits=[spec],
                   backend=io.pin(ROOT / "mixed_fit.R"), baselineKernel=str(ROOT / "joint_fit.R"),
                   candidateKernel=str(ROOT / "cluster_fit.R"), powerKernel=str(ROOT / "population_fit.R"))
    io.write(request_path, request)
    if receipt_path.exists():
        receipt = io.read(receipt_path)
    else:
        require(not receipt_path.with_suffix(".log").exists()
                and not (directory / "numerical").exists(),
                "Uncommitted fit attempt remains; preserve it instead of silently refitting")
        receipt = io.run_r(SimpleNamespace(rscript=manifest["environment"]["Rscript"]),
                           request_path, receipt_path)
    require(receipt["manifest"] == manifest_pin and len(receipt["records"]) == 1,
            "Changed fit receipt")
    entry = receipt["records"][0]
    require(entry["fitKey"] == spec["fitKey"], "Changed fit key")
    return entry["baseline"], entry["candidate"], io.pin(receipt_path)


def saved_input(manifest, ordinal, observed):
    from .magnitude_design import case_for
    source = Path(manifest["powerDirectory"])
    public_manifest = observed.actual(source / "power/manifest.json")
    io.validate_manifest(public_manifest)
    original_manifest = observed.read(public_manifest)
    require(original_manifest["family"] == "power" and original_manifest["expectedTotal"] == 80000,
            "Expected a public conditional-power reproduction")
    unit = source / "power/units" / f"{ordinal:05d}"
    commitment_pin = observed.actual(source / "power/events" / f"{ordinal:05d}.json")
    commitment = observed.read(commitment_pin)
    event = io.admit_event(commitment, "power", public_manifest)
    for descriptor in commitment["artifacts"]:
        observed.observe(descriptor["path"], descriptor["sha256"])
    spec_pin = observed.actual(unit / "spec.json")
    spec = observed.read(spec_pin)
    require(spec_pin in commitment["artifacts"] and spec["case"] == case_for(manifest["plan"], ordinal)
            and spec["manifest"] == public_manifest and spec["origin"] is None,
            "Changed saved case, origin or spec binding")
    key = spec["fitKey"]
    baseline_pin = observed.actual(source / "power/numerical/fits" / key / "record.json")
    candidate_pin = observed.actual(source / "power/numerical/records" /
                                     f"rep_{spec['repID']:04d}" / (spec["caseID"] + ".json"))
    require(baseline_pin in commitment["artifacts"] and candidate_pin in commitment["artifacts"]
            and event["fitKey"] == key, "Saved outputs are not committed input artifacts")
    return spec, spec_pin, baseline_pin, candidate_pin, commitment_pin


def analyze_input(binding, output, rscript, *, environment=None):
    """Read an existing fit, without fitting or generating an observation."""
    from . import magnitude_design as design, saved_fit_magnitude as core
    from . import magnitude_composition as composition, independent_cluster as kernel
    from . import independent_saved_fit as reader
    from .saved_fit_runtime import runtime, run_r
    from .magnitude_reduction import compact_event
    output = io.validate_output(output)
    require(not output.exists(), "Use a new output file")
    observed = Observations()
    input_pin = observed.actual(binding)
    request = observed.read(input_pin)
    require(request["schema"] == INPUT_SCHEMA and request["mode"] in MODES,
            "Wrong magnitude input schema")
    expected = design.case_for(plan(request["mode"]), request["ordinal"])
    spec = observed.read(request["spec"])
    baseline = observed.read(request["baseline"])
    candidate = observed.read(request["candidate"]) if request.get("candidate") else None
    require(spec["case"] == expected and spec["ordinal"] == request["ordinal"]
            and spec["caseID"] == expected["caseID"] and spec["repID"] == expected["repID"]
            and spec["fitKey"] == f"{expected['caseID']}__r{expected['repID']:04d}", "Wrong fit identity")
    require(baseline["request"] == spec["request"] and baseline["generation"] == spec["generation"]
            and baseline["fitKey"] == spec["fitKey"], "Broken baseline fit linkage")
    if candidate is not None:
        require(candidate["request"] == spec["request"] and candidate["generation"] == spec["generation"]
                and candidate["baseline"] == request["baseline"] and candidate["fitKey"] == spec["fitKey"]
                and candidate["fitArtifact"] == baseline["fitArtifact"], "Broken candidate fit linkage")
    original, generation = observed.read(spec["request"]), observed.read(spec["generation"])
    require(generation["case"] == expected and original["model"] == "H10"
            and original["reference"] == "C2", "Different case/seed or fitted model")
    # The older public power spec keeps the retention mask in its bound
    # generation object. Preserve that file while exposing the same fields to
    # the shared population validator.
    spec = dict(spec)
    for field in ("retentionMask", "recruitedCounts"):
        require(field not in spec or spec[field] == generation[field], "Conflicting retention metadata")
        spec.setdefault(field, generation[field])
    require(observed.read(baseline["rawArtifact"]) == baseline["raw"], "Changed emitted fitted record")
    observed.observe(baseline["fitArtifact"]["path"], baseline["fitArtifact"]["sha256"])
    history = dict(spec=spec, baseline=baseline, candidate=candidate, request=original, generation=generation)
    counts = core.validate_population(original, spec, generation)
    guard = core.guard_components(baseline["raw"])
    environment = environment or runtime(rscript, observed, output)
    for file, digest in environment["runtimeFiles"].items():
        observed.observe(file, digest)
    sources = [observed.actual(p["path"]) for p in source_files()]
    artifacts, errors = {}, []
    if all(guard.values()):
        key = spec["fitKey"]
        extracted, artifacts["extraction"] = run_r(environment["Rscript"], ROOT / "independent_cluster_rds.R",
            dict(schema="h10-inference-independent-rds-request-v1", fits=[dict(fitKey=key,
                 requestPath=spec["request"]["path"], rdsPath=baseline["fitArtifact"]["path"])]),
            output.with_suffix(".extraction.json"), observed, runtime_binding=environment)
        pins = dict(request=spec["request"], fitArtifact=baseline["fitArtifact"])
        transported, artifacts["transport"] = run_r(environment["Rscript"], ROOT / "source_transport.R",
            dict(schema="h10-power-source-transport-request-v1", fits=[dict(fitKey=key, **pins)]),
            output.with_suffix(".transport.json"), observed, runtime_binding=environment)
        actual = extracted["fits"][key]
        core.validate_extraction(actual, transported["fits"][key], history, fit_key=key)
        methods, errors = core.calculate_methods(actual, kernel, composition)
        saved_methods = [("baseline", baseline["raw"])]
        if candidate is not None:
            saved_methods.append(("candidate", candidate["rawCandidate"]))
        for name, raw in saved_methods:
            if methods[name]["status"] == "complete":
                core.check_old_differences(methods[name]["references"], raw, name + " saved D")
    else:
        methods = {name: core.unavailable_method("statistical_unavailable", "accepted_original_guard_false", composition)
                   for name in ("baseline", "candidate")}
    independent, artifacts["independent"] = run_r(environment["Rscript"], ROOT / "independent_saved_fit.R",
        dict(schema="h10-magnitude-independent-r-request-v1", target=dict(ordinal=request["ordinal"],
             caseID=spec["caseID"], repID=spec["repID"]), request=spec["request"],
             fitArtifact=baseline["fitArtifact"], actualK=counts, sources=sources,
             packageVersions=environment["packageVersions"], runtimeFiles=environment["runtimeFiles"]),
        output.with_suffix(".independent.json"), observed, runtime_binding=environment)
    require(independent["sourceClosure"] is True and independent["frameVerified"] is True
            and independent["exactSourceDesignInR"] is True and independent["newFits"] == independent["newDraws"] == 0
            and independent["guard"]["usableCandidateInference"] == all(guard.values()),
            "Independent model/frame/guard verification failed")
    checked = {}
    for name in ("baseline", "candidate"):
        if methods[name]["status"] != "technical_unresolved":
            checked[name] = reader.compare_method(methods[name], independent["methods"][name], name)
    require(not errors and len(checked) == 2, "Technical calculation unresolved; preserve outputs, do not count as nonrejection")
    record = dict(ordinal=request["ordinal"], caseID=spec["caseID"], repID=spec["repID"],
                  populationCounts=counts, truth=design.truth(expected), methods=methods)
    event = compact_event(record)
    from . import independent_magnitude_reduction as scalar_reader
    independent_event = scalar_reader.compact_event(record,
        {name: independent["methods"][name]["references"] for name in checked}, checked)
    scalar_reader.compare_report(event, independent_event, "independent scalar event")
    observed.fresh()
    result = dict(schema=SCHEMA, mode=request["mode"], input=input_pin, record=record,
                  event=event, independentEvent=independent_event,
                  independentEightLVerified=True, artifacts=artifacts,
                  environment=environment, observedInputs=dict(observed.files), sourceClosure=True,
                  newFits=0, newDraws=0, methodAdoptedForHumanData=False,
                  scope="supplied_fit_arithmetic_only", generationLawVerified=False,
                  campaignEligible=False)
    io.write(output, result)
    return result


def member(payload):
    manifest_pin, ordinal = payload
    observed = Observations()
    manifest = observe_manifest(manifest_pin, observed)
    output = Path(manifest_pin["path"]).parent
    directory = output / "units" / f"{ordinal:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "unit.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        final = directory / "commit.json"
        if final.exists():
            return admit_commit(manifest_pin, io.pin(final), expected_ordinal=ordinal)["event"]
        # Never silently rerun a partially completed numerical extraction.
        require(not any(directory.glob("magnitude.*")), "Uncommitted numerical output; preserve and inspect before retry")
        if manifest["plan"]["mode"] == "fresh_validation":
            spec, spec_pin = fresh_spec(manifest, manifest_pin, ordinal, directory)
            base, candidate, origin = fit_fresh(manifest, manifest_pin, spec, spec_pin, directory)
            generation_proof = io.pin(directory / "generation_proof.json")
        else:
            spec, spec_pin, base, candidate, origin = saved_input(manifest, ordinal, observed)
            generation_proof = None
        binding = dict(schema=INPUT_SCHEMA, mode=manifest["plan"]["mode"], ordinal=ordinal,
                       spec=spec_pin, baseline=base, candidate=candidate)
        binding_pin = io.write(directory / "input.json", binding)
        result_path = directory / "magnitude.json"
        result = analyze_input(binding_pin["path"], result_path, manifest["environment"]["Rscript"],
                               environment=manifest["environment"])
        observed.fresh()
        io.write(final, dict(schema=SCHEMA, manifest=manifest_pin, ordinal=ordinal,
                             result=io.pin(result_path), origin=origin,
                             generationProof=generation_proof, input=binding_pin,
                             event=result["event"], sourceClosure=True))
        return result["event"]


def validate_numerical_join(independent, record, spec, baseline, original, generation):
    """Join readback to the admitted fit, not just to copied numerical arrays."""
    from .saved_fit_magnitude import validate_population, guard_components
    counts = validate_population(original, spec, generation)
    require(record["populationCounts"] == counts and independent["actualK"] == counts,
            "Independent population differs from admitted input")
    require(independent["schema"] == "h10-magnitude-independent-r-v1"
            and independent["target"] == {key: record[key] for key in ("ordinal", "caseID", "repID")}
            and independent["request"] == spec["request"]
            and independent["fitArtifact"] == baseline["fitArtifact"]
            and independent["sourceClosure"] is True and independent["frameVerified"] is True
            and independent["exactSourceDesignInR"] is True
            and independent["newFits"] == independent["newDraws"] == 0
            and independent["guard"]["usableCandidateInference"] == all(guard_components(baseline["raw"]).values()),
            "Foreign or incomplete independent fit evidence")


def admit_commit(manifest_pin, commit_pin, *, expected_ordinal=None):
    observed = Observations()
    manifest = observe_manifest(manifest_pin, observed)
    commitment = observed.read(commit_pin)
    require(commitment["schema"] == SCHEMA and commitment["manifest"] == manifest_pin
            and commitment["sourceClosure"] is True, "Wrong committed result")
    ordinal = commitment["ordinal"]
    canonical = Path(manifest_pin["path"]).parent / "units" / f"{ordinal:05d}" / "commit.json"
    require(Path(commit_pin["path"]).resolve() == canonical.resolve()
            and (expected_ordinal is None or ordinal == expected_ordinal), "Wrong resumed unit identity/location")
    result = observed.read(commitment["result"])
    require(result["schema"] == SCHEMA and result["sourceClosure"] is True
            and result["independentEightLVerified"] is True and result["input"] == commitment["input"]
            and result["mode"] == manifest["plan"]["mode"]
            and result["event"] == commitment["event"] and result["event"]["ordinal"] == ordinal,
            "Changed independently checked event")
    for file, digest in result["observedInputs"].items():
        observed.observe(file, digest)
    origin = observed.read(commitment["origin"])
    binding = observed.read(commitment["input"])
    spec = observed.read(binding["spec"])
    require(binding["mode"] == result["mode"] and binding["ordinal"] == ordinal
            and binding["schema"] == INPUT_SCHEMA, "Changed result input identity")
    if result["mode"] == "fresh_validation":
        proof = observed.read(commitment["generationProof"])
        require(proof == dict(spec=binding["spec"], manifest=manifest_pin,
                request=spec["request"], generation=spec["generation"], independentReplay=True),
                "Changed generation proof")
        require(origin["manifest"] == manifest_pin and len(origin["records"]) == 1
                and origin["records"][0]["fitKey"] == spec["fitKey"]
                and origin["records"][0]["baseline"] == binding["baseline"]
                and origin["records"][0]["candidate"] == binding["candidate"], "Changed original fit receipt")
    else:
        _, sp, bp, cp, op = saved_input(manifest, ordinal, observed)
        require((sp, bp, cp, op) == (binding["spec"], binding["baseline"], binding["candidate"], commitment["origin"]),
                "Changed upstream saved-power admission")
    from .magnitude_design import case_for, truth
    expected = case_for(manifest["plan"], commitment["ordinal"])
    require(spec["case"] == expected and spec["ordinal"] == ordinal
            and spec["caseID"] == expected["caseID"] and spec["repID"] == expected["repID"]
            and spec["fitKey"] == f"{expected['caseID']}__r{expected['repID']:04d}"
            and result["record"]["truth"] == truth(expected)
            and result["event"]["caseID"] == expected["caseID"]
            and result["event"]["repID"] == expected["repID"], "Changed expected event identity")
    from .magnitude_reduction import compact_event
    from . import independent_magnitude_reduction as scalar_reader, independent_saved_fit as numerical_reader
    independent = observed.read(result["artifacts"]["independent"]["output"])
    for bundle in result["artifacts"].values():
        for descriptor in bundle.values():
            observed.observe(descriptor["path"], descriptor["sha256"])
    base, candidate = observed.read(binding["baseline"]), observed.read(binding["candidate"])
    original, generation = observed.read(spec["request"]), observed.read(spec["generation"])
    require(base["request"] == candidate["request"] == spec["request"]
            and base["generation"] == candidate["generation"] == spec["generation"]
            and base["fitKey"] == candidate["fitKey"] == spec["fitKey"]
            and candidate["baseline"] == binding["baseline"]
            and candidate["fitArtifact"] == base["fitArtifact"]
            and generation["case"] == expected, "Changed admitted fit linkage")
    population_spec = dict(spec)
    for field in ("retentionMask", "recruitedCounts"):
        require(field not in spec or spec[field] == generation[field], "Conflicting retained population")
        population_spec.setdefault(field, generation[field])
    validate_numerical_join(independent, result["record"], population_spec, base, original, generation)
    checked = {name: numerical_reader.compare_method(result["record"]["methods"][name],
               independent["methods"][name], name) for name in ("baseline", "candidate")}
    scalar_reader.compare_report(compact_event(result["record"]), result["event"], "stored scalar event")
    independent_event = scalar_reader.compact_event(result["record"],
        {name: independent["methods"][name]["references"] for name in checked}, checked)
    scalar_reader.compare_report(result["event"], independent_event, "rechecked scalar event")
    scalar_reader.compare_report(result["independentEvent"], independent_event, "stored independent event")
    observed.fresh()
    return commitment


def aggregate(manifest_pin, destination):
    from . import magnitude_reduction as producer, independent_magnitude_reduction as reader
    output = Path(manifest_pin["path"]).parent
    manifest = io.read(manifest_pin["path"])
    rows = [admit_commit(manifest_pin, io.pin(p))["event"] for p in sorted((output / "units").glob("*/commit.json"))]
    scientific = manifest["plan"]
    produced = producer.reduce_rows(scientific, rows)
    checked = reader.reduce_rows(dict(manifest=scientific, expectedSet=set(scientific["expectedOrdinals"])), rows)
    reader.compare_report(produced, checked, "independent magnitude aggregate")
    complete = [r["ordinal"] for r in rows] == scientific["expectedOrdinals"]
    value = dict(schema=SCHEMA, manifest=manifest_pin, results=produced,
                 completedTotal=len(rows), plannedTotal=len(scientific["expectedOrdinals"]), complete=complete,
                 independentReductionAgreement=True, methodAdoptedForHumanData=False,
                 establishes80PercentMDE=False, establishesHumanSampleSize=False)
    io.write(Path(destination).resolve(), value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ids", type=Path, help="JSON list of fixed scientific ordinals")
    parser.add_argument("--start", type=int, help="Zero-based position in the selected mode's ordered roster")
    parser.add_argument("--count", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--power-directory", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--node", default="node")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--aggregate", type=Path, help="New summary destination; requires --execute")
    parser.add_argument("--verify-input", type=Path, help="Check supplied-fit arithmetic only, not generation-law/campaign admission; no draws/fits")
    args = parser.parse_args()
    require(type(args.workers) is int and args.workers > 0, "Workers must be positive")
    if args.verify_input:
        require(args.execute and args.output and not args.aggregate and not args.mode,
                "--verify-input requires --execute --output FILE and no --mode/--aggregate")
        require(not any((args.ids, args.start is not None, args.count is not None, args.all, args.power_directory)),
                "Supplied-fit checking does not accept campaign selectors")
        args.output = io.validate_output(args.output)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        analyze_input(args.verify_input, args.output, args.rscript)
        return
    require(args.mode is not None, "--mode is required")
    scientific = plan(args.mode)
    if not args.execute:
        print(json.dumps(scientific, indent=2))
        return
    require(args.output is not None, "--execute requires --output")
    require(not args.aggregate or (args.output / "manifest.json").exists(), "Aggregation requires an existing manifest")
    if args.aggregate:
        require(not any((args.ids, args.start is not None, args.count is not None, args.all)),
                "Aggregation does not accept an execution selector")
    else:
        ids = select_ids(scientific, io.read(args.ids) if args.ids else None, args.start, args.count, args.all)
    # Each worker handles an independent fitted input; BLAS does not multiply
    # concurrency beyond the explicit worker count.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    io.numerical_runtime()
    args.rscript = shutil.which(args.rscript)
    require(args.rscript is not None, "Rscript is unavailable")
    manifest_pin = initialize(args.output, args.mode, args)
    if args.aggregate:
        aggregate(manifest_pin, args.aggregate)
        return
    payloads = [(manifest_pin, ordinal) for ordinal in ids]
    if args.workers == 1:
        for completed, payload in enumerate(payloads, 1):
            member(payload)
            print(f"{completed}/{len(ids)} independently checked inputs", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(ids))) as pool:
            for completed, _ in enumerate(pool.map(member, payloads), 1):
                print(f"{completed}/{len(ids)} independently checked inputs", flush=True)


if __name__ == "__main__":
    main()
