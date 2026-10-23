"""Count-assignment reducers with independent aggregate comparisons.

Missing work is not a statistical nonrejection. A saved-fit diagnostic retains
both methods on the same technically completed subset, including original
guard-unavailable outcomes. No inference is made from a small selection.
"""
from __future__ import annotations

from collections import Counter

from . import comparison_design as design
from . import fresh_reference as producer
from . import independent_cluster as independent
from . import independent_fresh as reader
from . import independent_family
from . import joint_generation as generation

SCHEMA = "statistical-comparison-count-assignment-v1"


def method_report(rows, name, case, planned):
    """Accumulate events only; M/T are assigned by the accepted fresh reducer."""
    state = producer.empty_state(case)
    state["completed"] = {row["repID"] for row in rows}
    target = state["methods"]["baseline"]
    for row in rows:
        for j, branch in enumerate(row[name]):
            target["branches"][j]["counts"].update({k: branch["counts"][k] for k in producer.EVENTS})
            n = sum(t["status"] == "available_candidate" for t in branch["tests"].values())
            target["branches"][j]["partial"] += int(0 < n < 4)
        for key in generation.H10_IDS:
            target["coverage"][key].update({k: row[name][0]["coverage"][key][k]
                                           for k in ("A", "CIAvailable", "C")})
    calculated = producer.method_report(state, "baseline", planned)
    checked = reader.method_report(rows, name, case, independent, planned)
    independent_family.near(calculated, checked, "independent count-assignment aggregate")
    return calculated


def case_report(family, case, rows):
    planned = design.planned(family, case)
    completed = [r for r in rows if r["technicalCompleted"]]
    T = len(completed)
    present = {r["repID"] for r in rows}
    report = dict(caseID=case["caseID"], M=planned, T=T, complete=T == planned,
        present=len(rows), technicalUnresolved=len(rows) - T, technicalMissing=planned - T,
        missingRepIDs=[r for r in range(1, planned + 1) if r not in present],
        unresolvedRepIDs=sorted(r["repID"] for r in rows if not r["technicalCompleted"]),
        baseline=method_report(completed, "baselineBranches", case, planned),
        technicalFailureReasons=dict(Counter(r["technicalError"]["phase"] for r in rows if not r["technicalCompleted"])),
        originalGuardCounts=dict(Counter(str((r["diagnostics"] or {}).get("usableCandidateInference")).lower() for r in rows)))
    if family == "baseline":
        report["diagnostics"] = {label: dict(count=n, rate=independent.rate(n, T)) for label, n in
            Counter(["status:" + r["status"] for r in completed] +
                [k + ":" + (str((r["diagnostics"] or {}).get(k)).lower()
                 if type((r["diagnostics"] or {}).get(k)) is bool else "unknown")
                 for r in completed for k in ("converged", "singular", "usableCandidateInference")]).items()}
        report["failureReasons"] = dict(Counter(r["failureReason"] or "none" for r in completed))
        # These rates belong to the original baseline presentation, not a new estimator.
        for branch in report["baseline"]["branches"]:
            for field, label in (("partialFamilies", "partialFamilyRate"), ("zeroFamilies", "zeroFamilyRate")):
                branch[label] = independent.rate(branch[field], T)
        for value in report["baseline"]["coverage"].values():
            value["availabilityAll"] = independent.rate(value["counts"]["A"], planned) if T == planned else None
        return report
    report["candidate"] = method_report(completed, "candidateBranches", case, planned)
    report["paired"] = []
    for j, mode in enumerate(generation.TAILS):
        report["paired"].append(dict(tailMode=mode, denominator=T,
            transitions={k: independent.event_table([(r["paired"][j]["transitions"][k]["baseline"],
                r["paired"][j]["transitions"][k]["candidate"]) for r in completed]) for k in producer.TRANSITIONS},
            contrasts={k: {label: independent.event_table([(r["paired"][j]["contrasts"][k][old],
                r["paired"][j]["contrasts"][k][new]) for r in completed])
                for label, old, new in (("coverage", "baselineCovered", "candidateCovered"),
                    ("availability", "baselineAvailable", "candidateAvailable"), ("rejection", "baselineReject", "candidateReject"))}
                for k in generation.H10_IDS}))
    report["contrastDiagnostics"] = {}
    for key in generation.H10_IDS:
        values = {metric: independent.distribution([r["paired"][0]["contrasts"][key][metric] for r in completed])
                  for metric in producer.METRICS}
        values["oracleSE"] = independent.distribution([
            (((r["candidateMetadata"] or {}).get("designDiagnostics") or {}).get("oracle") or {}).get("contrastSE", {}).get(key)
            for r in completed])
        if family == "diagnostic":
            values["planningMDE"] = {method: {label: independent.distribution([
                ((r["paired"][0]["contrasts"][key][method] or {}).get(label) or {}).get("effectMagnitude") for r in completed])
                for label in ("alpha_0.05", "alpha_0.0125")} for method in ("baselinePlanningMDE", "candidatePlanningMDE")}
        report["contrastDiagnostics"][key] = values
    if family == "fresh":
        report["practicalTarget"] = producer.practical_target(report)
        checked = reader.reduce_events(case, [reader.compact_events(r) for r in rows], independent, planned)
    else:
        checked = independent.aggregate_events(dict(cases=[case]), [dict(cases=rows)])[0]
    independent_family.near(report, checked, "independent paired aggregate")
    if family == "diagnostic":
        report["fixedWorkingWeightSERatios"] = oracle_ratios(completed)
        report["coverageNetChanges"] = {k: dict(
            baselineCovered=report["baseline"]["coverage"][k]["counts"]["C"],
            candidateCovered=report["candidate"]["coverage"][k]["counts"]["C"],
            net=report["paired"][0]["contrasts"][k]["coverage"]["01"] -
                report["paired"][0]["contrasts"][k]["coverage"]["10"],
            denominator=T) for k in generation.H10_IDS}
    return report


def oracle_ratios(rows):
    """Mean of per-fit SE ratios, not ratio of mean SEs; fixed estimated W."""
    from .cluster_inference import ratio
    return {key: {label: independent.distribution([ratio(r["paired"][0]["contrasts"][key][field],
        (((r["candidateMetadata"] or {}).get("designDiagnostics") or {}).get("oracle") or {}).get("contrastSE", {}).get(key))
        for r in rows]) for label, field in (("baselineOverOracleSE", "baselineSE"), ("CR2OverOracleSE", "candidateSE"))}
        for key in generation.H10_IDS}


def reduce_rows(family, rows, *, require_full=False):
    groups = {c["caseID"]: [] for c in design.templates(family)}
    seen = set()
    for row in rows:
        design.validate_row(family, row)
        design.require(row["ordinal"] not in seen, "Duplicate scientific comparison ID")
        seen.add(row["ordinal"])
        groups[row["caseID"]].append(row)
    cases = [case_report(family, case, sorted(groups[case["caseID"]], key=lambda r: r["repID"]))
             for case in design.templates(family)]
    T = sum(c["T"] for c in cases)
    complete = seen == set(range(design.TOTAL[family])) and T == design.TOTAL[family]
    design.require(not require_full or complete, "Full aggregate requires every original technically completed ID")
    result = dict(schema=SCHEMA, family=family, scope=design.SCOPE[family], complete=complete,
        expectedTotal=design.TOTAL[family], presentTotal=len(seen), completedTotal=T,
        absentTotal=design.TOTAL[family] - len(seen), technicalUnresolved=len(seen) - T,
        technicalMissing=design.TOTAL[family] - T, cases=cases,
        independentReduction=True, externalReferenceComparison="not_requested", scientificAdoption=False)
    if family == "diagnostic":
        result["negativeCoverageChanges"] = [dict(caseID=c["caseID"], contrast=k, **v)
            for c in cases for k, v in c["coverageNetChanges"].items() if v["net"] < 0]
        result["oracleScope"] = "known_DGP_covariance_with_estimated_working_W_held_fixed_not_candidate_inference"
    return result
