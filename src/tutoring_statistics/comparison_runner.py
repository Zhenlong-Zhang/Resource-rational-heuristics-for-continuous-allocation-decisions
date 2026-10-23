"""Source-calling historical statistical comparisons; plan is read-only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

from . import comparison_design as design
from . import reproduction as io

SCHEMA = "statistical-comparison-reproduction-v1"
ROOT = Path(__file__).resolve().parent


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def output_root(path):
    path = Path(path).resolve()
    repository = ROOT.parents[1]
    protected = [ROOT, repository / "src", repository / "configs", repository / "scripts",
                 repository / "notebooks", repository / "tests"]
    design.require(path != repository and all(p != path and p not in path.parents and path not in p.parents
                                            for p in protected), "Output overlaps protected source")
    return path


def namespace(family):
    return "fresh" if family == "fresh" else "production"


def context(args, output):
    io.numerical_runtime()
    resolved = shutil.which(args.rscript)
    design.require(resolved is not None, "Rscript executable not found")
    args.rscript = str(Path(resolved).resolve())
    source = io.inventory()
    if args.stage in ("check", "aggregate"):
        design.require((output / "source_snapshot.json").exists() and (output / "native_commit.json").exists() and
                       (output / "comparison_cache" / namespace(args.family) / "manifest.json").exists(),
                       "Readback/reduction requires an existing prepared reproduction; no input generation")
    io.write(output / "source_snapshot.json", source)
    native = io.prepare_native(args, output)
    ns = namespace(args.family)
    cache = output / "comparison_cache" / ns
    manifest = io.write(cache / "manifest.json", dict(schema=SCHEMA, namespace=ns,
        sources=io.pin(output / "source_snapshot.json"), native=io.pin(native / "h10_canonical.json"),
        environment=io.pin(ROOT / "data/numerical_environment.json"),
        python=io.pin(sys.executable), Rscript=io.pin(args.rscript),
        identity="new_public_reproduction_original_population",
        baselineReuse="same_original_fit_shared_between_baseline_and_saved_fit_diagnostic"))
    return cache, native, manifest


def validate_manifest(manifest):
    io.validate(manifest)
    value = io.read(manifest["path"])
    for name in ("sources", "native", "environment", "python", "Rscript"):
        io.validate(value[name])
    design.require(io.read(value["sources"]["path"]) == io.inventory(), "Changed comparison source closure")


def prepare_member(family, ordinal, cache, native, manifest):
    from . import joint_generation as generator, independent_generation as reader
    case = design.case_for(family, ordinal)
    design.require(case == design.independent_case(family, ordinal), "Independent case/seed identity differs")
    key = design.fit_key(family, case)
    directory = cache / "inputs" / key
    target = directory / "spec.json"
    with locked(directory):
        if target.exists():
            spec = io.read(target)
            design.require(spec["case"] == case and spec["manifest"] == manifest and spec["fitKey"] == key,
                           "Changed resumed input identity")
            for name in ("request", "generation"):
                io.validate(spec[name])
            design.require(spec["caseID"] == case["caseID"] and spec["repID"] == case["repID"] and
                           spec["independentGeneration"] is True, "Changed resumed generation admission")
            return spec, io.pin(target)
        design.require(not (directory / "actual_request.json").exists() and not (directory / "generation.json").exists(),
                       "Uncommitted input exists; inspect it rather than silently regenerating")
        canonical = io.read(native / "h10_canonical.json")
        actual, generation = generator.generate(case, generator.schedule_library(canonical))
        replay, replay_generation = reader.regenerate(case, reader.design_library(canonical))
        design.require(actual == replay and generation == replay_generation, "Independent original input replay differs")
        support = generator.support(actual)
        support_equal(support, reader.expected_support(replay["rows"]), replay["rows"])
        spec = dict(schema=SCHEMA, case=case, caseID=case["caseID"], repID=case["repID"], fitKey=key,
            manifest=manifest, support=support, independentGeneration=True,
            request=io.write(directory / "actual_request.json", actual),
            generation=io.write(directory / "generation.json", generation))
        return spec, io.write(target, spec)


def support_equal(saved, expected, rows):
    """Accepted fixed-population diagnostic representation rule; no rank relaxation."""
    from .independent_power import exact
    design.require(isinstance(saved, dict) and isinstance(expected, dict) and
                   set(saved) == set(expected) and "scaledSingularValues" in saved, "Changed support container")
    exact({k: v for k, v in saved.items() if k != "scaledSingularValues"},
          {k: v for k, v in expected.items() if k != "scaledSingularValues"}, "exact support categories")
    design.require(isinstance(rows, list) and len(rows) == 4620 and expected["columns"] == 9,
                   "Changed actual support dimensions")
    stored, reconstructed = saved["scaledSingularValues"], expected["scaledSingularValues"]
    for vector in (stored, reconstructed):
        design.require(isinstance(vector, list) and len(vector) == 9 and
            all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in vector) and
            all(a >= b for a, b in zip(vector, vector[1:])), "Invalid finite descending support diagnostic")
    tau = max(len(rows), 9) * 2.**-52 * reconstructed[0]
    design.require(tau > 0 and math.isfinite(tau), "Unresolved zero support scale")
    design.require(all(abs(a - b) <= tau for a, b in zip(stored, reconstructed)),
                   "Support difference exceeds original rank resolution")
    stored_tau = max(len(rows), 9) * 2.**-52 * stored[0]
    design.require(all(abs(value - tau) > tau + abs(stored_tau - tau) for value in reconstructed),
                   "Technical unresolved: support rank boundary is not stable")
    design.require(type(expected["rank"]) is int and expected["rank"] == saved["rank"] ==
                   sum(v > tau for v in reconstructed) == sum(v > stored_tau for v in stored),
                   "Changed original rank decision")


def run_stage(args, cache, manifest, spec, spec_pin, stage):
    directory = cache / "stages" / spec["fitKey"]
    with locked(directory):
        receipt_path = directory / (stage + "_receipt.json")
        request_path = directory / (stage + "_request.json")
        request = dict(schema="statistical-comparison-r-request-v1", stage=stage,
            sources=io.inventory()["files"], manifest=manifest, spec=spec_pin, fitKey=spec["fitKey"],
            packages=io.read(ROOT / "data/numerical_environment.json")["statistical_reference"]["R_packages"],
            backend=io.pin(ROOT / "mixed_fit.R"), baselineDir=str(cache / "baseline"), candidateDir=str(cache / "candidate"))
        io.write(request_path, request)
        # An interrupted attempt is never silently restarted. Committed receipts
        # can resume, but a partial RDS/raw write requires deliberate inspection.
        if not receipt_path.exists():
            design.require(not receipt_path.with_suffix(".log").exists(), "Interrupted R stage; inspect the retained attempt")
            io.run_r(args, request_path, receipt_path, script=ROOT / "comparison_entry.R")
        receipt = io.read(receipt_path)
        design.require(receipt["schema"] == "statistical-comparison-r-receipt-v1" and
                       receipt["stage"] == stage and receipt["manifest"] == manifest and receipt["spec"] == spec_pin and
                       receipt["fitKey"] == spec["fitKey"] and receipt["sourceClosure"] is True,
                       "Changed comparison receipt")
        io.validate(receipt["baseline"])
        if stage == "inference":
            io.validate(receipt["candidate"])
            design.require(receipt["newFitInvocations"] == 0, "Saved-fit inference must not refit")
        return receipt, io.pin(receipt_path)


def check_member(args, family, ordinal, cache, manifest, spec, spec_pin):
    """Independent actual-RDS, exact native-frame, df/CR2, and decision reader."""
    from . import joint_generation as generator, cluster_inference as candidate_math
    from . import independent_transport as transport, independent_cluster as math_reader
    from . import independent_family as accepted, independent_generation as event_reader
    directory = cache / "checks" / spec["fitKey"]
    directory.mkdir(parents=True, exist_ok=True)
    baseline_path = cache / "baseline" / "fits" / spec["fitKey"] / "record.json"
    design.require(baseline_path.exists(), "Missing baseline fit; check never launches a fit")
    baseline_pin = io.pin(baseline_path)
    baseline = io.read(baseline_path)
    design.require(baseline["request"] == spec["request"] and baseline["generation"] == spec["generation"] and
                   baseline["manifestSha256"] == manifest["sha256"] and baseline["batchSha256"] == spec_pin["sha256"] and
                   baseline["caseID"] == spec["caseID"] and baseline["repID"] == spec["repID"] and
                   baseline["fitKey"] == spec["fitKey"], "Broken baseline binding")
    raw = baseline["raw"]
    artifacts = [spec_pin, spec["request"], spec["generation"], baseline_pin]
    for name in ("rawArtifact", "fitArtifact"):
        if baseline.get(name):
            io.validate(baseline[name]); artifacts.append(baseline[name])
    if baseline.get("rawArtifact"):
        design.require(io.read(baseline["rawArtifact"]["path"]) == raw, "Changed emitted raw result")
    modules = dict(math=math_reader, accepted=accepted)
    evidence = facts = L = None
    if baseline.get("fitArtifact"):
        extraction = directory / "independent_request.json"
        evidence_path = directory / "independent_rds.json"
        io.write(extraction, dict(schema="h10-inference-independent-rds-request-v1", fits=[dict(
            fitKey=spec["fitKey"], requestPath=spec["request"]["path"], rdsPath=baseline["fitArtifact"]["path"])]))
        if not evidence_path.exists():
            io.run_r(args, extraction, evidence_path, script=ROOT / "independent_cluster_rds.R")
        evidence = io.read(evidence_path)["fits"][spec["fitKey"]]
        exact_path = io.source_transport(args, [spec], [dict(baseline=baseline_pin)], directory)
        transported = io.read(exact_path)
        exact = transported["fits"][spec["fitKey"]]
        design.require(transported["newScientificInputs"] == transported["newFitInvocations"] ==
                       transported["newCandidateCalculations"] == 0 and exact["request"] == spec["request"] and
                       exact["fitArtifact"] == baseline["fitArtifact"] and exact["exactMatrixInR"] is True and
                       exact["exactResponseInR"] is True and exact["oneULPMatrixMutationRejected"] is True,
                       "Missing exact same-R input admission")
        artifacts.extend([io.pin(evidence_path), io.pin(exact_path)])
        key = spec["request"]["path"], spec["request"]["sha256"]
        transport.SOURCE_TRANSPORT[key] = exact["design"]
        try:
            if raw["status"] == "fitted":
                facts, L = transport.independent_fit(raw, evidence, spec["request"],
                    io.read(spec["request"]["path"])["rows"], modules)
                design.require(facts["participants"] == 462 and facts["rows"] == 4620 and
                               facts["counts"] == {a: 154 for a in generator.ARMS},
                               "Changed fixed154 population")
            else:
                design.require(evidence["verified"] is True and evidence["requestDescriptor"] == spec["request"],
                               "Failed-fit frame was not verified")
        finally:
            transport.SOURCE_TRANSPORT.pop(key, None)
    else:
        design.require(raw["status"] == "unavailable" and raw["reason"] in
                       ("backend_fit_error", "design_support_unavailable"), "Unclassified missing RDS")
    case = spec["case"]
    baseline_branches = [generator.branch_result(case, mode, raw) for mode in generator.TAILS]
    checked_baseline = [event_reader.branch_outcome(case, mode, raw, accepted) for mode in generator.TAILS]
    check_branches(baseline_branches, checked_baseline)
    failure = raw.get("reason")
    if raw["status"] == "fitted" and raw["diagnostics"]["usableCandidateInference"] is False:
        failure = "fit_diagnostics_not_acceptable"
    row = dict(ordinal=ordinal, case=case, caseID=case["caseID"], repID=case["repID"], fitKey=spec["fitKey"],
        technicalCompleted=True, technicalError=None, baselineBranches=baseline_branches,
        status=raw["status"], diagnostics=raw.get("diagnostics"), failureReason=failure)
    if family != "baseline":
        candidate_path = cache / "candidate" / "records" / f"rep_{case['repID']:04d}" / (case["caseID"] + ".json")
        design.require(candidate_path.exists(), "Missing saved-fit inference; check never computes CR2 in R")
        record = io.read(candidate_path)
        design.require(record["manifest"] == manifest and record["baseline"] == baseline_pin and
                       record["request"] == spec["request"] and record["generation"] == spec["generation"] and
                       record["fitArtifact"] == baseline["fitArtifact"] and record["fitKey"] == spec["fitKey"] and
                       record["caseID"] == case["caseID"] and record["repID"] == case["repID"] and
                       record["newFitInvocations"] == record["newScientificInputs"] == 0,
                       "Broken saved-fit candidate binding")
        artifacts.append(io.pin(candidate_path))
        technical = record["technicalCompleted"]
        candidate_raw = record["rawCandidate"]
        candidate_branches = checked_candidate = None
        if technical:
            if candidate_raw["status"] == "fitted":
                design.require(evidence is not None and facts is not None, "Candidate without fitted baseline")
                accepted.near(candidate_raw["diagnostics"], raw["diagnostics"], "unchanged original guard")
                independent_raw = transport.independent_candidate(candidate_raw, evidence, facts, L, modules)
                accepted.near(record["candidateMetadata"]["designDiagnostics"],
                              math_reader.descriptive_diagnostics(evidence, case), "fixed-W known-DGP diagnostic")
            else:
                design.require(candidate_raw == raw and raw["status"] == "unavailable", "Changed unavailable inference")
                independent_raw = candidate_raw
            candidate_branches = [generator.branch_result(case, mode, candidate_raw) for mode in generator.TAILS]
            checked_candidate = [event_reader.branch_outcome(case, mode, independent_raw, accepted) for mode in generator.TAILS]
            check_branches(candidate_branches, checked_candidate)
        paired = candidate_math.paired_result(baseline_branches, candidate_branches)
        accepted.near(paired, math_reader.paired_events(checked_baseline, checked_candidate), "independent paired transitions")
        row.update(technicalCompleted=technical, technicalError=record["technicalError"],
            candidateBranches=candidate_branches, candidateMetadata=record["candidateMetadata"], paired=paired)
    design.validate_row(family, row)
    for descriptor in artifacts:
        io.validate(descriptor)
    return row, artifacts


def check_branches(produced, independent):
    from .independent_family import near
    from .joint_generation import H10_IDS
    for a, b in zip(produced, independent):
        for field in ("tailMode", "trueNullIDs", "falseNullIDs", "completeFamilyAvailable", "counts", "coverage", "rates", "fwerApplicability"):
            near(a[field], b[field], "independent branch " + field)
        for key in H10_IDS:
            for field in ("status", "p", "candidateNominalReject", "holmP", "holmReject"):
                near(a["tests"][key][field], b["tests"][key][field], "independent decision " + field)


def event_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def admit_event(path, family, manifest):
    commit = io.read(path)
    design.require(commit["schema"] == SCHEMA and commit["family"] == family and commit["manifest"] == manifest and
                   commit["eventDigest"] == event_digest(commit["event"]) and commit["independentReadback"] is True,
                   "Changed event commitment")
    design.validate_row(family, commit["event"])
    design.require(path.name == f"{commit['event']['ordinal']:05d}.json", "Cross-ID event filename")
    for artifact in commit["artifacts"]:
        io.validate(artifact)
    return commit["event"]


def member(args, output, cache, native, manifest, ordinal):
    target = output / args.family / "events" / f"{ordinal:05d}.json"
    with locked(cache / "members" / design.fit_key(args.family, design.case_for(args.family, ordinal))):
        if target.exists():
            admit_event(target, args.family, manifest)
            return
        if args.stage == "check":
            key = design.fit_key(args.family, design.case_for(args.family, ordinal))
            design.require((cache / "inputs" / key / "spec.json").exists(),
                           "Missing committed input; independent check never generates inputs")
        spec, spec_pin = prepare_member(args.family, ordinal, cache, native, manifest)
        if args.stage == "prepare":
            return
        if args.stage != "check":
            run_stage(args, cache, manifest, spec, spec_pin, "baseline")
            if args.family != "baseline":
                run_stage(args, cache, manifest, spec, spec_pin, "inference")
        row, artifacts = check_member(args, args.family, ordinal, cache, manifest, spec, spec_pin)
        validate_manifest(manifest)
        io.write(target, dict(schema=SCHEMA, family=args.family, manifest=manifest,
            event=row, eventDigest=event_digest(row), artifacts=artifacts, independentReadback=True))


def aggregate(args, output, manifest):
    from .comparison_reduction import reduce_rows
    directory = output / args.family / "events"
    paths = sorted(directory.glob("*.json"))
    result = reduce_rows(args.family, (admit_event(p, args.family, manifest) for p in paths), require_full=args.full)
    digest = hashlib.sha256("".join(io.pin(p)["sha256"] for p in paths).encode()).hexdigest()
    result.update(manifest=manifest, eventSetDigest=digest)
    validate_manifest(manifest)
    name = "aggregate_full.json" if result["complete"] else "aggregate_partial_" + digest[:16] + ".json"
    target = output / args.family / name
    io.write(target, result)
    return dict(output=str(target), complete=result["complete"], completedTotal=result["completedTotal"],
                expectedTotal=result["expectedTotal"], technicalUnresolved=result["technicalUnresolved"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(design.TOTAL), required=True)
    parser.add_argument("--stage", choices=("plan", "prepare", "run", "check", "aggregate", "all"), default="plan")
    parser.add_argument("--output", type=Path, default=Path("results/statistical_comparisons"))
    parser.add_argument("--full", action="store_true", help="Exact original roster, never a new campaign definition")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--node", default="node")
    args = parser.parse_args(argv)
    ordinals = design.selection(args.family, full=args.full, start=args.start, limit=args.limit)
    if args.stage == "plan":
        print(json.dumps(dict(family=args.family, expectedTotal=design.TOTAL[args.family], selected=len(ordinals),
            firstOrdinal=ordinals.start, lastOrdinal=ordinals.stop - 1, scope=design.SCOPE[args.family],
            draws=0, fits=0, full=args.full, cacheNamespace=namespace(args.family))))
        return
    design.require(args.execute, "Numerical work requires --execute")
    output = output_root(args.output)
    with locked(output / ".preparation"):
        cache, native, manifest = context(args, output)
    validate_manifest(manifest)
    if args.stage != "aggregate":
        for ordinal in ordinals:
            member(args, output, cache, native, manifest, ordinal)
    if args.stage in ("all", "aggregate"):
        print(json.dumps(aggregate(args, output, manifest), sort_keys=True))
    else:
        print(json.dumps(dict(family=args.family, stage=args.stage, selected=len(ordinals), output=str(output))))


if __name__ == "__main__":
    main()
