"""Portable joint, oracle and interval comparisons on public input fits.

The upstream magnitude runner creates or reuses the matching fixed input law.
This runner never fits a model or redraws responses. New calculations require
an explicit selector and --execute; committed members are admitted on resume.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path

from . import magnitude_design as design, magnitude_reproduction as upstream
from . import reproduction as io
from .saved_fit_inputs import Observations
from .saved_fit_magnitude import require, near

ROOT = Path(__file__).resolve().parent
SCHEMA = "closed-statistics-reproduction-v1"
INPUT_MODES = dict(joint_saved="saved_power_core", joint_fresh="fresh_joint_validation",
                   oracle_s1="saved_coverage_s1", coverage="fresh_coverage_validation")


def plan(mode):
    require(mode in INPUT_MODES, "Unknown closed result family")
    return dict(schema=SCHEMA, analysis=mode, inputPlan=upstream.plan(INPUT_MODES[mode]),
                methodAdoptedForHumanData=False, humanPowerEstablished=False,
                independentNegativeCorrelationSolver=False, independentKRAlgorithm=False)


def initialize(args):
    output = io.validate_output(args.output, args.input)
    output.mkdir(parents=True, exist_ok=True)
    input_pin = io.pin(Path(args.input) / "manifest.json")
    observed = Observations()
    parent = upstream.observe_manifest(input_pin, observed)
    require(parent["plan"] == plan(args.mode)["inputPlan"], "Wrong original input law")
    with (output / "initialize.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = output / "manifest.json"
        if marker.exists():
            saved = io.read(marker)
            require(saved["plan"] == plan(args.mode) and saved["inputManifest"] == input_pin
                    and saved["sources"] == upstream.source_files(), "Changed resumed source, plan or input")
            return io.pin(marker)
        require({p.name for p in output.iterdir()} == {"initialize.lock"}, "Use a new output directory")
        environment = None
        if args.mode == "coverage":
            from .saved_fit_runtime import runtime
            environment = runtime(args.rscript, observed, output / "environment.json",
                extra_packages=io.read(ROOT / "data/interval_environment.json")["additional_R_packages"])
        observed.fresh()
        return io.write(marker, dict(schema=SCHEMA, plan=plan(args.mode), inputManifest=input_pin,
            sources=upstream.source_files(), environment=environment, observedInputs=dict(observed.files)))


def context(manifest_pin, observed):
    manifest = observed.read(manifest_pin)
    mode = manifest["plan"]["analysis"]
    require(manifest["schema"] == SCHEMA and manifest["plan"] == plan(mode)
            and manifest["sources"] == upstream.source_files(), "Changed closed manifest")
    for descriptor in manifest["sources"]:
        observed.observe(descriptor["path"], descriptor["sha256"])
    for path, digest in manifest["observedInputs"].items():
        observed.observe(path, digest)
    parent = upstream.observe_manifest(manifest["inputManifest"], observed)
    require(parent["plan"] == manifest["plan"]["inputPlan"], "Changed upstream law")
    return manifest


def original_member(manifest, ordinal, observed):
    pin = manifest["inputManifest"]
    directory = Path(pin["path"]).parent / "units" / f"{ordinal:05d}"
    commitment_pin = observed.actual(directory / "commit.json")
    commitment = upstream.admit_commit(pin, commitment_pin, expected_ordinal=ordinal)
    result = observed.read(commitment["result"])
    binding = observed.read(commitment["input"])
    spec = observed.read(binding["spec"])
    actual = observed.read(spec["request"])
    independent = observed.read(result["artifacts"]["independent"]["output"])
    return commitment_pin, result, binding, spec, actual, independent


def joint_event(record, original_event, methods):
    from .magnitude_design import IDS
    result = {key: copy.deepcopy(record[key]) for key in
              ("ordinal", "caseID", "repID", "populationCounts", "truth")}
    result.update(methods={}, diagnostics={})
    for name, method in methods.items():
        done = method["technicalCompleted"]
        nulls = [key for key in IDS if record["truth"][key]["trueNull"]]
        alts = [key for key in IDS if not record["truth"][key]["trueNull"]]
        events = dict(R=method["anyRejects"],
            F=any(method["tests"][k]["rejected"] for k in nulls) if done and nulls else None,
            D=any(method["tests"][k]["rejected"] for k in alts) if done and alts else None)
        result["methods"][name] = dict(technicalCompleted=done, status=method["status"],
            availableCount=method["availableCount"], events=events, original=original_event["methods"][name],
            tests={key: {field: method["tests"][key][field] for field in
                   ("status", "pJoint", "holmP", "rejected")} for key in IDS})
        result["diagnostics"][name] = {key: dict(trueD=record["truth"][key]["D"],
            trueS=record["truth"][key]["S"], D=record["methods"][name]["references"][key+":D"],
            S=record["methods"][name]["references"][key+":S"], jointTest=method["tests"][key]) for key in IDS}
    return result


def joint_analysis(result, independent, *, saved=False):
    from .joint_magnitude import compose_saved_method
    from .independent_joint_magnitude import independent_method
    from .sign_congruence import gaussian_comparison
    record = result["record"]
    produced, checked = {}, {}
    counts = dict(analyticNonnegativeComparisons=0, negativeCorrelationComparisons=[], covarianceMappingChecks=0)
    for name in (("candidate",) if saved else ("baseline", "candidate")):
        old = record["methods"][name]
        basis = "CR2" if name == "candidate" else "model_based_diagnostic"
        produced[name] = compose_saved_method(old, covariance_source=basis)
        other = dict(independent["methods"][name], status=old["status"]) if old["status"] == "complete" else old
        checked[name] = independent_method(other, basis, gaussian_comparison, counts)
        near(produced[name], checked[name], "independent joint method")
    event = joint_event(record, result["event"], produced)
    independent_record = copy.deepcopy(record)
    for name in ("baseline", "candidate"):
        if record["methods"][name]["status"] == "complete":
            independent_record["methods"][name]["references"] = independent["methods"][name]["references"]
    other = joint_event(independent_record, result["independentEvent"], checked)
    from .independent_magnitude_reduction import compare_report
    compare_report(event, other, "independent joint event")
    return dict(event=event, independentEvent=other, methods=produced, independentMethods=checked,
                independentChecks=counts, independentNegativeCorrelationSolver=False)


def oracle_analysis(mode, spec, actual):
    from .known_covariance import known_gls
    from .independent_known_covariance import reconstruct
    label = "s5_l2_rep1" if mode == "coverage" and spec["case"]["caseIndex"] == 2 else "s1_rep1"
    producer = known_gls(label, actual, spec["case"])
    independent = reconstruct(label, actual, spec["case"])
    for key in ("coefficientNames", "beta", "covariance", "actualK", "rowCount", "clusterCount", "knownParameters"):
        near(producer[key], independent[key], "independent oracle " + key)
    for key, ref in producer["references"].items():
        near(ref, {field: independent["references"][key][field] for field in ref}, "independent oracle target")
    return producer, independent


def oracle_event(result, oracle, checked):
    from . import known_covariance_reduction as producer, independent_known_covariance_reduction as reader
    from .known_covariance import comparisons
    record = result["record"]
    scalar = dict(repID=record["repID"], populationCounts=record["populationCounts"])
    for name in ("baseline", "candidate"):
        ref = record["methods"][name]["references"]["C1-C2:B:D"]
        require(ref["status"] == "available", "Saved oracle diagnostic requires original available target")
        scalar[name] = dict(estimate=ref["estimate"], SE=ref["SE"], df=ref["df"],
            lower=ref["CI"][0], upper=ref["CI"][1], covered=ref["CI"][0] <= 0 <= ref["CI"][1],
            z=ref["estimate"]/ref["SE"])
    event = producer.compact_event(dict(repID=record["repID"], scalar=scalar, oracle=oracle,
        comparisons=comparisons(oracle, record["methods"])))
    independent = reader.compact_event(record["repID"], checked, scalar)
    near(event, independent, "independent known-covariance event")
    return dict(event=event, independentEvent=independent, oracle=oracle, independentOracle=checked)


def linear_interval(ref, truth):
    if ref["status"] != "available":
        return dict(status=ref["status"], reason=ref["reason"], truth=truth,
                    covered=None, coverageNumericallyUnresolved=False)
    ambiguous = any(math.isclose(truth, end, abs_tol=2e-9, rel_tol=2e-8) for end in ref["CI"])
    return dict({k: ref[k] for k in ("L", "status", "estimate", "SE", "df", "CI")},
                truth=truth, reason=None, covered=None if ambiguous else ref["CI"][0] <= truth <= ref["CI"][1],
                coverageNumericallyUnresolved=ambiguous)


def coverage_analysis(manifest, result, binding, spec, independent, oracle, checked, directory, observed):
    from .joint_interval_design import COVERAGE_TARGETS
    from .saved_fit_runtime import run_r
    targets = COVERAGE_TARGETS[spec["case"]["caseIndex"]]
    base = observed.read(binding["baseline"])
    runtime = manifest["environment"]
    response, artifacts = run_r(runtime["Rscript"], ROOT / "interval_fit.R",
        dict(schema="closed-interval-r-request-v1", target={key: result["record"][key] for key in
             ("ordinal", "caseID", "repID")}, caseIndex=spec["case"]["caseIndex"], references=targets,
             request=spec["request"], fitArtifact=base["fitArtifact"], actualK=spec["populationCounts"],
             sources=manifest["sources"], packageVersions=runtime["packageVersions"], runtimeFiles=runtime["runtimeFiles"]),
        directory / "interval.json", observed, runtime_binding=runtime)
    require(response["sourceClosure"] is True and response["frameVerified"] is True
            and response["exactSourceDesignInR"] is True and response["newFits"] == response["newDraws"] == 0
            and response["request"] == spec["request"] and response["fitArtifact"] == base["fitArtifact"]
            and response["actualK"] == spec["populationCounts"]
            and response["guard"]["usableCandidateInference"] == independent["guard"]["usableCandidateInference"],
            "Unclosed independent KR source/frame/guard")
    for path, digest in response["observedInputs"].items():
        observed.observe(path, digest)
    near(response["producer"], response["independent"], "public KR inversion")
    events = []
    for other, kr, known in ((False, response["producer"], oracle), (True, response["independent"], checked)):
        event = {key: result["record"][key] for key in ("ordinal", "caseID", "repID")}
        event.update(caseIndex=spec["case"]["caseIndex"], actualK=spec["populationCounts"], methods={})
        for name in ("baseline", "candidate"):
            source = result["record"]["methods"][name]
            refs = independent["methods"][name]["references"] if other and source["status"] == "complete" else source["references"]
            event["methods"][name] = dict(status=source["status"], references={key: linear_interval(refs[key], v["truth"])
                                                                           for key, v in targets.items()})
        event["methods"]["KR"] = dict(status="complete" if all(v["status"] == "available" for v in kr.values())
                                      else "statistical_unavailable", references=kr)
        event["oracle"] = dict(status="complete", reason=None,
                               **{key: known[key] for key in ("beta", "covariance", "references", "actualK")})
        # The independent QR adds a receiving-tolerance diagnostic, not science.
        event["oracle"]["references"] = {k: {f: v[f] for f in oracle["references"][k]} for k, v in known["references"].items()}
        events.append(event)
    from .independent_interval_reduction import audit_event
    for event in events:
        audit_event(event)
    near(events[0], events[1], "independent interval event")
    return dict(event=events[0], independentEvent=events[1], artifacts=artifacts,
                independentKRAlgorithm=False, sharedPackageCore=True)


def member(manifest_pin, ordinal):
    observed = Observations()
    manifest = context(manifest_pin, observed)
    require(type(ordinal) is int and ordinal in manifest["plan"]["inputPlan"]["expectedOrdinals"], "Unplanned member")
    directory = Path(manifest_pin["path"]).parent / "units" / f"{ordinal:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "unit.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = directory / "commit.json"
        if marker.exists():
            return admit(manifest_pin, io.pin(marker))
        require({p.name for p in directory.iterdir()} == {"unit.lock"}, "Uncommitted analysis; preserve instead of recomputing")
        parent, result, binding, spec, actual, independent = original_member(manifest, ordinal, observed)
        mode = manifest["plan"]["analysis"]
        if mode.startswith("joint_"):
            analysis = joint_analysis(result, independent, saved=mode == "joint_saved")
        else:
            oracle, checked = oracle_analysis(mode, spec, actual)
            analysis = (oracle_event(result, oracle, checked) if mode == "oracle_s1" else
                        coverage_analysis(manifest, result, binding, spec, independent, oracle, checked, directory, observed))
        observed.fresh()
        result_pin = io.write(directory / "result.json", dict(schema=SCHEMA, manifest=manifest_pin,
            ordinal=ordinal, parent=parent, **analysis, observedInputs=dict(observed.files), sourceClosure=True,
            newFits=0, newDraws=0, methodAdoptedForHumanData=False))
        io.write(marker, dict(schema=SCHEMA, manifest=manifest_pin, ordinal=ordinal, parent=parent, result=result_pin))
        return analysis


def admit(manifest_pin, commit_pin):
    observed = Observations()
    manifest = context(manifest_pin, observed)
    commit = observed.read(commit_pin)
    require(commit["schema"] == SCHEMA and commit["manifest"] == manifest_pin, "Wrong analysis commitment")
    ordinal = commit["ordinal"]
    expected = design.case_for(manifest["plan"]["inputPlan"], ordinal)
    require(Path(commit_pin["path"]) == Path(manifest_pin["path"]).parent / "units" / f"{ordinal:05d}" / "commit.json",
            "Relabeled analysis commitment path")
    parent, _, _, _, _, _ = original_member(manifest, ordinal, observed)
    require(commit["parent"] == parent, "Changed original input commitment")
    result = observed.read(commit["result"])
    require(result["schema"] == SCHEMA and result["manifest"] == manifest_pin and result["ordinal"] == ordinal
            and result["parent"] == parent and result["sourceClosure"] is True
            and result["newFits"] == result["newDraws"] == 0, "Unclosed analysis result")
    for path, digest in result["observedInputs"].items():
        observed.observe(path, digest)
    for key in ("event", "independentEvent"):
        event = result[key]
        require(event["ordinal"] == ordinal and event["repID"] == expected["repID"], "Changed event identity")
        if "caseID" in event:
            require(event["caseID"] == expected["caseID"], "Changed case identity")
    from .independent_magnitude_reduction import compare_report
    compare_report(result["event"], result["independentEvent"], "admitted independent event")
    observed.fresh()
    return result


def aggregate(manifest_pin, destination):
    from . import independent_magnitude_reduction as scalar_reader
    observed = Observations()
    manifest = context(manifest_pin, observed)
    root = Path(manifest_pin["path"]).parent
    reports = [admit(manifest_pin, observed.actual(p)) for p in sorted((root / "units").glob("*/commit.json"))]
    rows, independent_rows = ([r[key] for r in reports] for key in ("event", "independentEvent"))
    scientific = manifest["plan"]["inputPlan"]
    ids = [row["ordinal"] for row in rows]
    require(ids == sorted(set(ids)) and set(ids) <= set(scientific["expectedOrdinals"]), "Repeated/unplanned final union")
    mode = manifest["plan"]["analysis"]
    if mode == "joint_saved":
        from . import saved_joint_reduction, independent_saved_joint_reduction
        def saved_events(values):
            return [dict({k: row[k] for k in ("ordinal", "caseID", "repID", "truth")},
                         **row["methods"]["candidate"]) for row in values]
        produced = saved_joint_reduction.reduce_events(scientific, saved_events(rows))
        checked = independent_saved_joint_reduction.reduce_events(scientific, saved_events(independent_rows))
    elif mode == "joint_fresh":
        from . import joint_reduction, independent_joint_reduction
        ctx = dict(manifest=scientific, modules=dict(audit=design), originalReader=scalar_reader)
        produced = joint_reduction.reduce_events(ctx, rows)
        checked = independent_joint_reduction.reduce_events(ctx, independent_rows)
    elif mode == "oracle_s1":
        from . import known_covariance_reduction, independent_known_covariance_reduction
        produced = known_covariance_reduction.reduce_events(rows)
        checked = independent_known_covariance_reduction.reduce_events(independent_rows)
    else:
        from . import interval_reduction, independent_interval_reduction
        for row in rows + independent_rows:
            independent_interval_reduction.audit_event(row)
        produced = interval_reduction.reduce_rows(rows)
        checked = independent_interval_reduction.reduce_independently(independent_rows)
    scalar_reader.compare_report(produced, checked, "closed independent reduction")
    observed.fresh()
    destination = io.validate_output(destination, Path(manifest["inputManifest"]["path"]).parent)
    result = dict(schema=SCHEMA, analysis=mode, manifest=manifest_pin, results=produced,
        independentResults=checked, independentReductionAgreement=True,
        present=len(ids), expectedTotal=scientific["expectedTotal"], complete=ids == scientific["expectedOrdinals"],
        missingOrdinals=sorted(set(scientific["expectedOrdinals"])-set(ids)),
        methodAdoptedForHumanData=False, humanPowerEstablished=False,
        independentNegativeCorrelationSolver=False, independentKRAlgorithm=False,
        pointwiseMonteCarlo=True, simultaneousMonteCarloGuarantee=False,
        inputCommitments=[r["parent"] for r in reports], observedInputs=dict(observed.files))
    io.write(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(INPUT_MODES), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Matching public magnitude-study output root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ids", type=Path, help="JSON list of exact original ordinals")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--aggregate", type=Path, help="Aggregate only; do not combine with selectors")
    args = parser.parse_args(argv)
    scientific = plan(args.mode)
    if args.aggregate:
        require(not args.all and args.ids is None and args.start is None and args.count is None,
                "Aggregate must not select new calculation")
        ids = None
    else:
        ids = upstream.select_ids(scientific["inputPlan"], io.read(args.ids) if args.ids else None,
                                  args.start, args.count, args.all)
    print(json.dumps(dict(plan=scientific, selectedOrdinals=ids, execute=args.execute)))
    if not args.execute:
        return
    io.numerical_runtime()
    manifest_pin = initialize(args)
    if args.aggregate:
        aggregate(manifest_pin, args.aggregate)
    else:
        for ordinal in ids:
            member(manifest_pin, ordinal)


if __name__ == "__main__":
    main()
