"""Scalar event and report calculations for fixed magnitude comparisons.

Callers admit source-bound records and exact unions before claiming completion.
"""
from __future__ import annotations

import math

from .magnitude_design import ARMS, IDS, LINEAR_IDS, case_for, require, truth


def compact_event(record):
    """Only independently checkable scalar events go to final reduction."""
    methods = {}
    for name in ("baseline", "candidate"):
        source = record["methods"][name]; comp = source["composition"]
        completed = comp["status"] == "complete"
        tests = {}
        for key in IDS:
            test = comp["tests"][key]
            tests[key] = dict(available=test["status"] == "available", trueNull=record["truth"][key]["trueNull"],
                pMagnitude=test["pMagnitude"], holmP=test["holmP"], rejected=test["holmReject"])
        nulls = [k for k in IDS if tests[k]["trueNull"]]
        alternatives = [k for k in IDS if not tests[k]["trueNull"]]
        references = {}
        for key in LINEAR_IDS:
            ref = source["references"][key]; target = record["truth"][key[:-2]][key[-1]]
            available = ref["status"] == "available"
            references[key] = dict(available=available, truth=target, estimate=ref["estimate"], SE=ref["SE"], df=ref["df"],
                covered=(ref["CI"][0] <= target <= ref["CI"][1]) if available else None,
                lessP=ref["lessP"], greaterP=ref["greaterP"],
                lessTrueNull=target >= 0, greaterTrueNull=target <= 0)
        methods[name] = dict(status=source["status"], technicalCompleted=completed,
            availableCount=sum(v["available"] for v in tests.values()), allAvailable=all(v["available"] for v in tests.values()),
            R=any(t["rejected"] for t in tests.values()) if completed else None,
            F=any(tests[k]["rejected"] for k in nulls) if completed and nulls else None,
            D=any(tests[k]["rejected"] for k in alternatives) if completed and alternatives else None,
            tests=tests, references=references)
    return dict(ordinal=record["ordinal"], caseID=record["caseID"], repID=record["repID"],
                populationCounts=record["populationCounts"], truth=record["truth"], methods=methods)


def measurement(rows, planned):
    present = len(rows)
    counts = {name: sum(r["methods"][name]["technicalCompleted"] for r in rows) for name in ("baseline", "candidate")}
    return dict(M=planned, present=present, candidateT=counts["candidate"], baselineT=counts["baseline"],
        jointT=sum(all(r["methods"][n]["technicalCompleted"] for n in counts) for r in rows),
        missing=planned-present, candidateTechnicalUnresolved=present-counts["candidate"],
        baselineTechnicalUnresolved=present-counts["baseline"], complete=counts["candidate"] == planned,
        bothMethodsComplete=all(v == planned for v in counts.values()))


def interval(k, n):
    from scipy.stats import beta
    require(type(k) is int and type(n) is int and 0 <= k <= n, "Invalid binomial counts")
    if not n:
        return None
    return [0. if not k else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k))]


def rate(k, completed, planned):
    require(0 <= k <= completed <= planned, "Invalid method denominator")
    return dict(count=k, M=planned, T=completed, fullRate=k/planned if completed == planned and planned else None,
                fullCI=interval(k, planned) if completed == planned else None,
                completedRate=k/completed if completed else None, completedCI=interval(k, completed),
                unknownEventBounds=[k/planned, (k+planned-completed)/planned] if planned else None,
                complete=completed == planned)


def distribution(values):
    import statistics
    valid = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in valid), "Nonfinite compact diagnostic")
    return dict(n=len(valid), missing=len(values)-len(valid), mean=statistics.mean(valid) if valid else None,
                sd=statistics.stdev(valid) if len(valid)>1 else None, minimum=min(valid) if valid else None,
                maximum=max(valid) if valid else None)


def reduce_method(rows, name, planned):
    entries = [r["methods"][name] for r in rows]
    complete = [r for r in entries if r["technicalCompleted"]]
    available = [r for r in complete if r["allAvailable"]]
    T, A4 = len(complete), len(available)
    output = dict(M=planned, T=T, technicalUnresolved=len(entries)-T, missing=planned-len(entries), A4=A4,
                  statisticalUnavailableFamilies=sum(r["availableCount"] < 4 for r in complete), family={}, contrasts={}, linearReferences={})
    for key in ("R", "F", "D"):
        applicable = all(r[key] is not None for r in complete) if complete else None
        # Applicability comes from fixed truth, even for an empty prefix.
        if rows:
            truth_mask = rows[0]["truth"]
            applicable = key == "R" or (any(v["trueNull"] for v in truth_mask.values()) if key == "F" else any(not v["trueNull"] for v in truth_mask.values()))
        output["family"][key] = (dict(all=rate(sum(r[key] is True for r in complete), T, planned),
            allAvailable=dict(count=sum(r[key] is True for r in available), denominator=A4,
                rate=sum(r[key] is True for r in available)/A4 if A4 else None,
                CI=interval(sum(r[key] is True for r in available), A4))) if applicable else None)
    for key in IDS:
        tests = [v["tests"][key] for v in complete]; a = [v for v in tests if v["available"]]
        output["contrasts"][key] = dict(rejection=rate(sum(v["rejected"] is True for v in tests), T, planned),
            available=len(a), availableRejection=rate(sum(v["rejected"] is True for v in a), len(a), len(a)))
    for key in LINEAR_IDS:
        refs = [v["references"][key] for v in complete]; a = [v for v in refs if v["available"]]
        tails = {}
        for tail in ("less", "greater"):
            is_null = rows[0]["truth"][key[:-2]][key[-1]] >= 0 if tail == "less" and rows else (
                rows[0]["truth"][key[:-2]][key[-1]] <= 0 if rows else None)
            tails[tail] = dict(trueNull=is_null, rejectionByThreshold=[dict(threshold=t,
                all=rate(sum(v[tail+"P"] < t for v in a), T, planned),
                available=rate(sum(v[tail+"P"] < t for v in a), len(a), len(a)))
                for t in (.05, .025, .0125, .05/6, .00625)])
        output["linearReferences"][key] = dict(available=len(a),
            coverageAll=rate(sum(v["covered"] is True for v in refs), T, planned),
            coverageAvailable=rate(sum(v["covered"] is True for v in a), len(a), len(a)), tails=tails,
            estimate=distribution([v["estimate"] for v in refs]), SE=distribution([v["SE"] for v in refs]),
            df=distribution([v["df"] for v in refs]))
    return output


def reduce_rows(manifest, rows):
    wanted = set(manifest["expectedOrdinals"])
    require([r["ordinal"] for r in rows] == sorted({r["ordinal"] for r in rows}) and set(r["ordinal"] for r in rows) <= wanted,
            "Repeated/unplanned compact evidence")
    results = []
    for case in manifest["cases"]:
        selected = [r for r in rows if r["caseID"] == case["caseID"]]
        expected_truth = truth(case)
        require(all(r["truth"] == expected_truth and r["caseID"] == case_for(manifest, r["ordinal"])["caseID"]
                    and r["repID"] == case_for(manifest, r["ordinal"])["repID"] for r in selected), "Changed compact identity/truth")
        # An empty prefix still has fixed truth/applicability. Pass it separately
        # by creating no artificial event or fabricated technical completion.
        methods = {m: reduce_method(selected, m, manifest["B"]) for m in ("baseline", "candidate")}
        for method in methods.values():
            for key in ("R", "F", "D"):
                applicable = key == "R" or (any(v["trueNull"] for v in expected_truth.values()) if key == "F" else any(not v["trueNull"] for v in expected_truth.values()))
                if not selected and applicable:
                    method["family"][key] = dict(all=rate(0, 0, manifest["B"]), allAvailable=dict(count=0, denominator=0, rate=None, CI=None))
            if not selected:
                for key in LINEAR_IDS:
                    target = expected_truth[key[:-2]][key[-1]]
                    method["linearReferences"][key]["tails"]["less"]["trueNull"] = target >= 0
                    method["linearReferences"][key]["tails"]["greater"]["trueNull"] = target <= 0
        paired = [r for r in selected if all(r["methods"][m]["technicalCompleted"] for m in ("baseline", "candidate"))]
        transitions = {key: sum(("1" if r["methods"]["baseline"]["R"] else "0")+("1" if r["methods"]["candidate"]["R"] else "0") == key for r in paired) for key in ("00","01","10","11")}
        results.append(dict(caseID=case["caseID"], case=case, truth=expected_truth,
            measurement=measurement(selected, manifest["B"]), methods=methods,
            paired=dict(jointCompleted=len(paired), familyRejectionTransitions=transitions),
            populationCounts={arm: distribution([r["populationCounts"][arm] for r in selected]) for arm in ARMS}))
    return results
