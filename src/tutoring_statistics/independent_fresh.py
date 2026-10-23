"""independent fresh: accepted numerical definitions, portable source projection."""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from copy import deepcopy
from statistics import NormalDist
import copy
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tempfile

import numpy as np
import scipy
from scipy import stats, special, optimize
from scipy.stats import beta as beta_distribution, binom, nct, t as t_distribution
from scipy.optimize import brentq
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, stdtrit



NAMESPACE = "h10-cr2-validation-v1-20261019"

METHOD = "participant_CR2_Satterthwaite_v1"

IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")

MODES = ("literal", "sign_consistent")

INDICES = (1, 2, 4, 7, 8, 9, 10, 13, 14, 16, 19, 20, 22)

REPS = 2000

TOTAL = 26000

METRICS = ("estimate", "baselineSE", "candidateSE", "baselineDf", "candidateDf", "SERatio", "dfRatio", "CIWidthRatio",
           "assumedEffectMagnitudeOverBaselineSE", "assumedEffectMagnitudeOverCandidateSE")

TRANSITIONS = ("availability", "falseRejection", "anyRejection", "alternativeRejection")

def validate_ordinals(values):
    require(isinstance(values, list) and bool(values) and
            all(type(value) is int and 0 <= value < TOTAL for value in values) and
            values == sorted(set(values)), "Changed fresh ordinal selection")
    return values

def require(condition, message):
    if not condition:
        raise ValueError(message)

def case_for_ordinal(ordinal, cases):
    validate_ordinals([ordinal])
    case = copy.deepcopy(cases[ordinal % len(INDICES)])
    rep = ordinal//len(INDICES)+1
    case.update(repID=rep, designSeed=402610190+10000*case["scenarioIndex"]+rep,
                responseSeed=502610190+10000*case["caseIndex"]+rep)
    return case

def fit_key(case):
    return f"{case['caseID']}__r{case['repID']:04d}"

def method_report(rows, name, case, math_kernel, planned=REPS):
    """Assign M/T explicitly; statistical unavailability stays inside common T."""
    require(type(planned) is int and planned > 0 and len(rows) <= planned and
            all(row["technicalCompleted"] is True for row in rows), "Invalid common technical subset")
    T = len(rows)
    branches = []
    for j, mode in enumerate(MODES):
        counts = dict.fromkeys(("A4", "F", "F4", "R", "R4", "D", "D4"), 0)
        partial = 0
        for row in rows:
            branch = row[name][j]
            require(branch["tailMode"] == mode and branch["counts"]["M"] ==
                    branch["counts"]["T"] == 1 and branch["counts"]["technicalMissing"] == 0,
                    "Changed per-input branch denominator")
            for key in counts:
                value = branch["counts"][key]
                require(type(value) is int and value in (0, 1), "Invalid scientific event count")
                counts[key] += value
            available = sum(test["status"] == "available_candidate" for test in branch["tests"].values())
            require(len(branch["tests"]) == 4 and bool(branch["counts"]["A4"]) is (available == 4),
                    "Changed complete-family definition")
            partial += int(0 < available < 4)
        counts.update(M=planned, T=T, technicalMissing=planned-T)
        nulls = case["trueNullIDs"][mode]
        rate = math_kernel.rate
        branches.append(dict(tailMode=mode, counts=counts, trueNullIDs=nulls,
            fwerApplicability="applicable" if nulls else "not_applicable_no_true_null",
            rates=dict(availabilityAll=rate(counts["A4"], planned) if T == planned else None,
                availabilityCompleted=rate(counts["A4"], T),
                falseRejectionAll=rate(counts["F"], planned) if T == planned and nulls else None,
                falseRejectionCompleted=rate(counts["F"], T) if nulls else None,
                falseRejectionAvailable=rate(counts["F4"], counts["A4"]) if nulls else None,
                anyRejectionAll=rate(counts["R"], planned) if T == planned else None,
                anyRejectionAvailable=rate(counts["R4"], counts["A4"]),
                alternativeRejectionAll=rate(counts["D"], planned) if T == planned and len(nulls) < 4 else None,
                alternativeRejectionAvailable=rate(counts["D4"], counts["A4"]) if len(nulls) < 4 else None),
            partialFamilies=partial, zeroFamilies=T-counts["A4"]-partial))
    coverage = {}
    for key in IDS:
        totals = {field: sum(row[name][0]["coverage"][key][field] for row in rows)
                  for field in ("A", "CIAvailable", "C")}
        require(0 <= totals["C"] <= totals["CIAvailable"] <= totals["A"] <= T,
                "Changed contrast coverage denominator")
        coverage[key] = dict(counts=totals,
            coverageAll=math_kernel.rate(totals["C"], planned) if T == planned else None,
            coverageAvailable=math_kernel.rate(totals["C"], totals["A"]),
            availabilityCompleted=math_kernel.rate(totals["A"], T),
            ciUnavailableAmongAvailable=totals["A"]-totals["CIAvailable"])
    return dict(branches=branches, coverage=coverage)

def practical_assessment(method, T, planned=REPS):
    """Keep the accepted practical bound separate from nominal adverse results."""
    assessments, nominal = [], []
    for branch in method["branches"]:
        conditions = {"technicalComplete": T == planned}
        for key in ("falseRejectionAll", "falseRejectionAvailable"):
            value = branch["rates"][key]
            if branch["trueNullIDs"]:
                conditions[key] = value is not None and value["interval95"][1] <= .065
                if value is not None and value["interval95"][0] > .05:
                    nominal.append(dict(tailMode=branch["tailMode"], target=key,
                                        reference=.05, adverse="interval_entirely_above_nominal", rate=value))
        for contrast, proof in method["coverage"].items():
            for key in ("coverageAll", "coverageAvailable"):
                value = proof[key]
                conditions[contrast+":"+key] = value is not None and value["interval95"][0] >= .935
                if value is not None and value["interval95"][1] < .95:
                    item = dict(contrast=contrast, target=key, reference=.95,
                                adverse="interval_entirely_below_nominal", rate=value)
                    if item not in nominal:
                        nominal.append(item)
        assessments.append(dict(tailMode=branch["tailMode"], conditions=conditions,
                                established=all(conditions.values())))
    return dict(criterion="prespecified_1.5pp_practical_diagnostic_not_exact_nominal_or_simultaneous",
                branches=assessments, nominalAdverseIntervals=nominal)

def compact_events(row):
    """Keep only event/scalar evidence needed by the independent final reduction."""
    result = {key: row[key] for key in ("ordinal", "caseID", "repID", "fitKey", "technicalCompleted", "technicalError")}
    result["diagnostics"] = dict(usableCandidateInference=(row["diagnostics"] or {}).get("usableCandidateInference"))
    if not row["technicalCompleted"]:
        require(row["candidateBranches"] is None and isinstance(row["technicalError"], dict),
                "Technical unknown became a scientific nonrejection")
        return result
    for name in ("baselineBranches", "candidateBranches"):
        result[name] = [dict(tailMode=value["tailMode"], counts=value["counts"], coverage=value["coverage"],
                            tests={key: dict(status=test["status"]) for key, test in value["tests"].items()})
                        for value in row[name]]
    result["paired"] = [dict(tailMode=value["tailMode"], transitions=value["transitions"],
        contrasts={key: {field: values[field] for field in METRICS +
            ("baselineCovered", "candidateCovered", "baselineAvailable", "candidateAvailable", "baselineReject", "candidateReject")}
            for key, values in value["contrasts"].items()}) for value in row["paired"]]
    result["oracleSE"] = (((row["candidateMetadata"] or {}).get("designDiagnostics") or {}).get("oracle") or {}).get("contrastSE", {})
    return result

def practical_target(report):
    results = []
    for branch in report["candidate"]["branches"]:
        limits, adverse = {}, []
        for label in ("falseRejectionAll", "falseRejectionAvailable"):
            value = branch["rates"][label]
            upper = None if value is None else value["interval95"][1]
            limits[label] = dict(bound=upper, limit=.065, established=upper is not None and upper <= .065)
            if value is not None and value["interval95"][0] > .05:
                adverse.append(label)
        for key in IDS:
            for name in ("coverageAll", "coverageAvailable"):
                value = report["candidate"]["coverage"][key][name]
                lower = None if value is None else value["interval95"][0]
                label = key+"/"+name
                limits[label] = dict(bound=lower, limit=.935, established=lower is not None and lower >= .935)
                if value is not None and value["interval95"][1] < .95:
                    adverse.append(label)
        results.append(dict(tailMode=branch["tailMode"], technicalComplete=report["complete"], limits=limits,
            established=report["complete"] and all(value["established"] for value in limits.values()),
            nominalAdverseIntervals=adverse,
            interpretation="prespecified_practical_pointwise_diagnostic_not_exact_nominal_or_simultaneous"))
    return results

def reduce_events(case, rows, math_kernel, planned=REPS):
    """Independent scalar/event reduction, never the rejected historical reducer."""
    require(len({row["repID"] for row in rows}) == len(rows) and
            all(type(row["repID"]) is int and 1 <= row["repID"] <= planned and
                row["caseID"] == case["caseID"] and type(row["technicalCompleted"]) is bool for row in rows),
            "Duplicate, wrong-case or out-of-range scientific ID")
    completed = [row for row in rows if row["technicalCompleted"]]
    T = len(completed)
    paired = []
    for j, mode in enumerate(MODES):
        transitions = {name: math_kernel.event_table([(row["paired"][j]["transitions"][name]["baseline"],
            row["paired"][j]["transitions"][name]["candidate"]) for row in completed]) for name in TRANSITIONS}
        contrasts = {key: {label: math_kernel.event_table([(row["paired"][j]["contrasts"][key][old],
            row["paired"][j]["contrasts"][key][new]) for row in completed])
            for label, old, new in (("coverage", "baselineCovered", "candidateCovered"),
                                   ("availability", "baselineAvailable", "candidateAvailable"),
                                   ("rejection", "baselineReject", "candidateReject"))} for key in IDS}
        paired.append(dict(tailMode=mode, denominator=T, transitions=transitions, contrasts=contrasts))
    diagnostics = {key: {name: math_kernel.distribution([row["paired"][0]["contrasts"][key][name] for row in completed])
        for name in METRICS} for key in IDS}
    for key in IDS:
        diagnostics[key]["oracleSE"] = math_kernel.distribution([row["oracleSE"].get(key) for row in completed])
    errors, guards = {}, {}
    for row in rows:
        if not row["technicalCompleted"]:
            phase = row["technicalError"]["phase"]
            errors[phase] = errors.get(phase, 0)+1
        label = str((row["diagnostics"] or {}).get("usableCandidateInference")).lower()
        guards[label] = guards.get(label, 0)+1
    report = dict(caseID=case["caseID"], M=planned, T=T, complete=T == planned, present=len(rows),
        technicalUnresolved=len(rows)-T, technicalMissing=planned-T,
        missingRepIDs=[rep for rep in range(1, planned+1) if rep not in {row["repID"] for row in rows}],
        unresolvedRepIDs=sorted(row["repID"] for row in rows if not row["technicalCompleted"]),
        baseline=method_report(completed, "baselineBranches", case, math_kernel, planned),
        candidate=method_report(completed, "candidateBranches", case, math_kernel, planned), paired=paired,
        contrastDiagnostics=diagnostics, technicalFailureReasons=errors, originalGuardCounts=guards)
    report["practicalTarget"] = practical_target(report)
    return report

def disjoint_union(scopes):
    seen = set()
    for values in scopes:
        selected = validate_ordinals(values)
        require(not seen.intersection(selected), "Overlapping independent scientific IDs")
        seen.update(selected)
    return seen
