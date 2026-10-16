"""Portable orchestration of the fixed statistical generators and summaries.

Full runs preserve the original batch size and seeds. Smoke is batch zero, not
additional independent evidence. Fixture runs use another RNG shape and are
explicitly not a prefix of the scientific run.
"""
import csv
import json
import math
from pathlib import Path

from . import robustness, identifiability_robustness as adaptive
from .bounded_inference import BOUNDS, GEOMETRY, source_fixture_rows, fit_bounded
from .empirical_bernstein import cohort_identification
from .inference import CORE, CONDITIONS, POLICIES, fit, synthetic_example_rows
from .planning import planning_report


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows, fields):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fresh_output(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; never overwrite a prior run")
    output.mkdir(parents=True, exist_ok=True)
    return output


def cohort_examples(config, domain):
    """The fixed 135 finite-enrolled-cohort examples, not confidence intervals."""
    results=[]
    for recipe in config["cohortIdentification"]["effectRecipes"]:
        p=config["effectRecipes"][recipe]
        for count in config["cohortIdentification"]["retainedFullPersonCounts"]:
            for identifier,_,endpoint,weights,_ in CORE:
                bounds={c:domain["numericBounds"] if endpoint=="slope_initial" else BOUNDS[endpoint] for c in weights}
                values=[]
                for c in weights:
                    mu=p["beta"][("C2","C3","C4").index(c)] if endpoint=="slope_initial" else p[adaptive.KEYS[endpoint]][CONDITIONS.index(c)]
                    for i in range(count):
                        value=mu
                        if endpoint=="anyQuestion":
                            lower=math.floor(mu*10)/10;upper=math.ceil(mu*10)/10
                            value=lower if i%2==0 else upper
                        values.append({"participantId":f"hypothetical-{c}-{i}","conditionId":c,"value":value,
                            "endpointKind":"slope" if endpoint=="slope_initial" else "mean","observedTrialCount":10})
                result=cohort_identification(values,{c:80 for c in weights},bounds,weights,config["cohortIdentification"]["target"])
                results.append({"id":identifier,"recipe":recipe,"observedFullPeopleEachArm":count,
                    "hypotheticalEnrolledEachArm":80,"exampleProvenance":"hypothetical_bound_domain_assumptions_not_engine_observed_people",
                    "QObservedFractionsUseNativeTenths":True,**result})
    return results


def historical_failures(data):
    """Retain every false historical criterion, including supporting windows."""
    failures=[]
    for s in data["cells"]:
        cell=s["cell"]
        for population,p in s["populations"].items():
            def fail(route,criterion,metric,threshold):
                failures.append({"cell":cell["id"],"classification":cell["nominalStatus"],"population":population,"route":route,
                    "criterion":criterion,"numerator":metric["numerator"],"denominator":metric["denominator"],"rate":metric["rate"],
                    "lower":metric["interval"][0],"upper":metric["interval"][1],"threshold":threshold})
            for t in p["tests"]:
                if t["coverageCriterionPassed"] is False:fail(t["id"],"pointwise_coverage_lower",t["coverage"],data["criteria"]["coverageLower"])
                if t["rawErrorCriterionPassed"] is False:fail(t["id"],"raw_directional_error_upper",t["rawFalseReject"],data["criteria"]["ordinaryErrorUpper"])
                if t["designatedLowTailCriterionPassed"] is False:fail(t["id"],"alpha_over_nine_tail_upper",t["firstGlobalThresholdReject"],.05/9+data["criteria"]["lowTailErrorAllowance"])
            for policy,r in p["policies"].items():
                if r["globalErrorCriterionPassed"] is False:fail(policy,"global_true_null_FWER_upper",r["trueNullFWER"],data["criteria"]["ordinaryErrorUpper"])
                for family,f in r["families"].items():
                    if f["criterionPassed"] is False:fail(policy+"/"+family,"family_true_null_FWER_upper",f["trueNullFWER"],data["criteria"]["ordinaryErrorUpper"])
    return failures


def write_historical_failures(output, data):
    failures = historical_failures(data)
    write_json(Path(output)/"criteria_readback.json", {
        "status": "MIXED_CALIBRATION_not_blanket_PASS" if failures else "no_failed_criteria_in_this_run",
        "attempts": data["attempts"], "failureRecords": failures,
        "scope": "fixed generators only; not human validity or selection repair"})
    write_csv(Path(output)/"all_criterion_failures.csv", failures,
              ["cell", "classification", "population", "route", "criterion", "numerator",
               "denominator", "rate", "lower", "upper", "threshold"])


def run_inference(output, root):
    output = fresh_output(output)
    rows = synthetic_example_rows(6)
    write_json(output/"synthetic_rows.json", rows)
    for policy in POLICIES:
        write_json(output/(policy+".json"), fit(rows, policy=policy))
    finite = source_fixture_rows(root=root)
    write_json(output/"finite_fixture_rows.json", finite)
    write_json(output/"finite_bounded_fit.json", fit_bounded(finite, "synthetic_design", "synthetic_validation", GEOMETRY))
    return {"scope": "deterministic functionality, not calibration", "rows": len(rows)}


def run_planning(config, output):
    output = fresh_output(output)
    result = planning_report(config)
    write_json(output/"planning.json", result)
    return {"scenarios": len(result["scenarios"]), "finalNSelected": False}


def run_calibration(kind, config, output, mode="smoke", design=None, node="node"):
    if kind not in ("bounded", "variance-adaptive") or mode not in ("fixture", "smoke", "full"):
        raise ValueError("Explicit supported analysis and mode required")
    output = fresh_output(output)
    module = robustness if kind == "bounded" else adaptive
    context = None
    if kind == "variance-adaptive":
        if design is None:
            raise ValueError("A public scientific design is required")
        context = adaptive.prepare_context(config, design, node)
        cells = context["cells"]
        write_json(output/"prepared_context.json", context)
        write_json(output/"source_slope_domain.json", context["domain"])
        from design_validation.c1_identifiability import audit_candidates
        write_json(output/"C1_states.json", audit_candidates(design, config["c1"]))
        write_json(output/"cohort_identification.json", cohort_examples(config, context["domain"]))
    else:
        cells = robustness.resolved_cells(config)
    size = 3 if mode == "fixture" else config["batchSize"]
    count = config["replicationsPerCell"]//size if mode == "full" else 1
    write_json(output/"settings.json", {"analysis": kind, "mode": mode, "config": config,
        "batchSize": size, "batchesPerCell": count,
        "fixtureIsAlternateSizedNotScientificPrefix": mode == "fixture",
        "smokeOverlapsFullBatchZero": mode == "smoke", "primaryReplacement": False})
    (output/"batches").mkdir()
    summaries = []
    for cell in cells:
        aggregates = []
        for index in range(count):
            if context is None:
                batch = module.generate_batch(cell, index, size)
                statistics = module.core_statistics(batch["groups"])
            else:
                batch = module.generate_batch(cell, index, context, size)
                statistics = module.core_statistics(batch, context)
            parities = []
            if index in (0, count-1):
                replicas = (0, size-1) if mode == "fixture" else config["scalarParity"]["replicaIndices"]
                for replica in sorted(set(replicas)):
                    for policy in config["policies"]:
                        if context is None:
                            parity = module.scalar_parity(cell, batch, statistics, policy, replica, config["alpha"])
                        else:
                            parity = module.scalar_parity(cell, batch, statistics, context, policy, replica, config)
                        parities.append(parity)
            aggregate = module.aggregate_batch(cell, batch, statistics, config)
            write_json(output/"batches"/f"{cell['id']}_{index:03d}.json",
                       {"aggregate": aggregate, "parities": parities})
            aggregates.append(aggregate)
        summaries.append(module.combine_batches(cell, aggregates, config))
    failures = [failure for cell in summaries for failure in cell["criteriaFailures"]]
    result = {"analysis": kind, "mode": mode, "attempts": sum(c["attempts"] for c in summaries),
        "cells": summaries, "criteriaFailures": failures, "primaryReplacement": False,
        "humanEvidence": False, "originalPopulationSelectionRepaired": False,
        "status": "MIXED" if failures else "no_failed_criteria_in_declared_scope"}
    write_json(output/"summary.json", result)
    write_csv(output/"criteria_failures.csv", failures, ["cell", "id", "criterion", "value", "threshold"])
    routes = []
    for cell in summaries:
        methods = cell["methods"] if context is not None else {"Hoeffding": cell}
        for method, own in methods.items():
            routes.extend({"cell": cell["id"], "method": method, **row} for row in own["routes"])
    write_csv(output/"route_summary.csv", routes,
              ["cell", "method", "id", "attempts", "availableAttempts", "unavailableAttempts",
               "meanEstimate", "meanWidth", "meanIntervalWidth", "selectedRawTruth",
               "originalRawTruth", "selectedTruth", "originalTruth"])
    return {"analysis": kind, "mode": mode, "attempts": result["attempts"], "status": result["status"]}
