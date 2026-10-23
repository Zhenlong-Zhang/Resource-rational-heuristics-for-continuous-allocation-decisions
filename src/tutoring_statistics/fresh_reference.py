"""fresh reference: accepted numerical definitions, portable source projection."""
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

from . import joint_generation as m
from . import cluster_inference as hi
require = hi.require

NAMESPACE = "h10-cr2-validation-v1-20261019"

SCOPE = "fresh_conditional_calibration"

METHOD = hi.METHOD

INDICES = (1, 2, 4, 7, 8, 9, 10, 13, 14, 16, 19, 20, 22)

B = 2000

TOTAL = 26000

RULE = dict(order="rep_major_caseIndex", firstOrdinal=0, lastOrdinal=25999,
            casesPerRep=13, firstRep=1, lastRep=2000)

EVENTS = ("A4", "F", "F4", "R", "R4", "D", "D4")

TRANSITIONS = ("availability", "falseRejection", "anyRejection", "alternativeRejection")

METRICS = ("estimate", "baselineSE", "candidateSE", "baselineDf", "candidateDf", "SERatio", "dfRatio", "CIWidthRatio",
           "assumedEffectMagnitudeOverBaselineSE", "assumedEffectMagnitudeOverCandidateSE")

def cases_for(rep):
    require(type(rep) is int and 1 <= rep <= B, "Replicate outside fresh roster")
    result = []
    for case in m.measurement_plan()["cases"]:
        if case["caseIndex"] in INDICES:
            case.update(repID=rep, designSeed=402610190+10000*case["scenarioIndex"]+rep,
                        responseSeed=502610190+10000*case["caseIndex"]+rep)
            result.append(case)
    return result

def ordinal_case(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < TOTAL, "Invalid fresh ordinal")
    return cases_for(ordinal // 13 + 1)[ordinal % 13]

def fit_key(case):
    return f"{case['caseID']}__r{case['repID']:04d}"

def empty_state(case):
    return dict(case=case, present=set(), completed=set(), errors=Counter(), guards=Counter(),
        methods={method: dict(branches=[dict(counts=Counter(), partial=0) for _ in m.TAILS],
            coverage={key: Counter() for key in m.H10_IDS}) for method in ("baseline", "candidate")},
        paired=[dict(transitions={name: Counter() for name in TRANSITIONS},
            contrasts={key: {name: Counter() for name in ("coverage", "availability", "rejection")}
                       for key in m.H10_IDS}) for _ in m.TAILS],
        diagnostics={key: {name: [] for name in METRICS+("oracleSE",)} for key in m.H10_IDS})

def transition_key(a, b):
    return "unknown" if type(a) is not bool or type(b) is not bool else f"{int(a)}{int(b)}"

def add_row(state, row):
    rep = row["repID"]
    require(rep not in state["present"], "Overlapping accepted scientific IDs")
    expected = ordinal_case(row["ordinal"])
    require(row["case"] == expected and expected["caseID"] == state["case"]["caseID"] and
            row["caseID"] == expected["caseID"] and rep == expected["repID"] and row["fitKey"] == fit_key(expected),
            "Changed summary scientific identity")
    state["present"].add(rep)
    state["guards"][str((row["diagnostics"] or {}).get("usableCandidateInference")).lower()] += 1
    require(type(row["technicalCompleted"]) is bool, "Unknown technical completion")
    if not row["technicalCompleted"]:
        require(row["candidateBranches"] is None and isinstance(row["technicalError"], dict), "Technical error became scientific result")
        state["errors"][row["technicalError"]["phase"]] += 1
        return
    require(row["technicalError"] is None, "Completed input retains technical error")
    state["completed"].add(rep)
    for method in ("baseline", "candidate"):
        branches = row[f"{method}Branches"]
        require(len(branches) == 2, "Missing tail branch")
        for j, mode in enumerate(m.TAILS):
            branch = branches[j]; counts = branch["counts"]
            require(branch["tailMode"] == mode and branch["trueNullIDs"] == expected["trueNullIDs"][mode]
                    and counts["M"] == counts["T"] == 1 and counts["technicalMissing"] == 0
                    and all(type(counts[k]) is int and counts[k] in (0, 1) for k in EVENTS), "Changed per-input counts")
            available = sum(test["status"] == "available_candidate" for test in branch["tests"].values())
            require(set(branch["tests"]) == set(m.H10_IDS) and bool(counts["A4"]) == (available == 4), "Changed full-family availability")
            state["methods"][method]["branches"][j]["counts"].update({k: counts[k] for k in EVENTS})
            state["methods"][method]["branches"][j]["partial"] += int(0 < available < 4)
        for key in m.H10_IDS:
            values = branches[0]["coverage"][key]
            require(values["truth"] == expected["delta"][key] and
                    all(type(values[k]) is int and values[k] in (0, 1) for k in ("A", "CIAvailable", "C")) and
                    values["C"] <= values["CIAvailable"] <= values["A"], "Changed pointwise coverage")
            state["methods"][method]["coverage"][key].update({k: values[k] for k in ("A", "CIAvailable", "C")})
    for j, paired in enumerate(row["paired"]):
        require(paired["tailMode"] == m.TAILS[j], "Changed paired tail")
        for name in TRANSITIONS:
            value = paired["transitions"][name]
            state["paired"][j]["transitions"][name][transition_key(value["baseline"], value["candidate"])] += 1
        for key in m.H10_IDS:
            values = paired["contrasts"][key]
            for name, a, b in (("coverage", "baselineCovered", "candidateCovered"),
                               ("availability", "baselineAvailable", "candidateAvailable"),
                               ("rejection", "baselineReject", "candidateReject")):
                state["paired"][j]["contrasts"][key][name][transition_key(values[a], values[b])] += 1
    for key in m.H10_IDS:
        for name in METRICS:
            state["diagnostics"][key][name].append(row["paired"][0]["contrasts"][key][name])
        state["diagnostics"][key]["oracleSE"].append(
            (((row["candidateMetadata"] or {}).get("designDiagnostics") or {}).get("oracle") or {}).get("contrastSE", {}).get(key))

def method_report(state, method, planned=B):
    """New reducer epoch: M/T are assignments, never Counter additions."""
    require(type(planned) is int and planned >= len(state["completed"]), "Invalid planned denominator")
    T = len(state["completed"]); template = state["case"]; branches = []
    for j, mode in enumerate(m.TAILS):
        source = state["methods"][method]["branches"][j]
        counts = {k: source["counts"][k] for k in EVENTS}
        counts["M"] = planned; counts["T"] = T; counts["technicalMissing"] = planned-T
        require(all(0 <= counts[k] <= T for k in EVENTS) and
                all(counts[k] <= counts["A4"] for k in ("F4", "R4", "D4")), "Invalid aggregate event count")
        nulls = template["trueNullIDs"][mode]; alternatives = len(nulls) < 4
        branches.append(dict(tailMode=mode, counts=counts, trueNullIDs=nulls,
            fwerApplicability="applicable" if nulls else "not_applicable_no_true_null",
            rates=dict(availabilityAll=hi.rate(counts["A4"], planned) if T == planned else None,
                availabilityCompleted=hi.rate(counts["A4"], T),
                falseRejectionAll=hi.rate(counts["F"], planned) if T == planned and nulls else None,
                falseRejectionCompleted=hi.rate(counts["F"], T) if nulls else None,
                falseRejectionAvailable=hi.rate(counts["F4"], counts["A4"]) if nulls else None,
                anyRejectionAll=hi.rate(counts["R"], planned) if T == planned else None,
                anyRejectionAvailable=hi.rate(counts["R4"], counts["A4"]),
                alternativeRejectionAll=hi.rate(counts["D"], planned) if T == planned and alternatives else None,
                alternativeRejectionAvailable=hi.rate(counts["D4"], counts["A4"]) if alternatives else None),
            partialFamilies=source["partial"], zeroFamilies=T-counts["A4"]-source["partial"]))
    coverage = {}
    for key in m.H10_IDS:
        counts = {k: state["methods"][method]["coverage"][key][k] for k in ("A", "CIAvailable", "C")}
        require(0 <= counts["C"] <= counts["CIAvailable"] <= counts["A"] <= T, "Invalid coverage denominator")
        coverage[key] = dict(counts=counts, coverageAll=hi.rate(counts["C"], planned) if T == planned else None,
            coverageAvailable=hi.rate(counts["C"], counts["A"]), availabilityCompleted=hi.rate(counts["A"], T),
            ciUnavailableAmongAvailable=counts["A"]-counts["CIAvailable"])
    return dict(branches=branches, coverage=coverage)

def practical_target(report):
    result = []
    for branch in report["candidate"]["branches"]:
        limits = {}; adverse = []
        for name in ("falseRejectionAll", "falseRejectionAvailable"):
            rate = branch["rates"][name]; upper = None if rate is None else rate["interval95"][1]
            limits[name] = dict(bound=upper, limit=.065, established=upper is not None and upper <= .065)
            if rate is not None and rate["interval95"][0] > .05:
                adverse.append(name)
        for key, coverage in report["candidate"]["coverage"].items():
            for name in ("coverageAll", "coverageAvailable"):
                rate = coverage[name]; lower = None if rate is None else rate["interval95"][0]
                label = f"{key}/{name}"
                limits[label] = dict(bound=lower, limit=.935, established=lower is not None and lower >= .935)
                if rate is not None and rate["interval95"][1] < .95:
                    adverse.append(label)
        result.append(dict(tailMode=branch["tailMode"], technicalComplete=report["complete"], limits=limits,
            established=report["complete"] and all(x["established"] for x in limits.values()),
            nominalAdverseIntervals=adverse,
            interpretation="prespecified_practical_pointwise_diagnostic_not_exact_nominal_or_simultaneous"))
    return result

def finalize_state(state):
    T = len(state["completed"])
    fixed_table = lambda values: {k: values[k] for k in ("00", "01", "10", "11", "unknown")}
    paired = [dict(tailMode=mode, denominator=T,
        transitions={name: fixed_table(values) for name, values in state["paired"][j]["transitions"].items()},
        contrasts={key: {name: fixed_table(values) for name, values in fields.items()}
                   for key, fields in state["paired"][j]["contrasts"].items()}) for j, mode in enumerate(m.TAILS)]
    report = dict(caseID=state["case"]["caseID"], M=B, T=T, complete=T == B,
        present=len(state["present"]), technicalUnresolved=len(state["present"])-T, technicalMissing=B-T,
        missingRepIDs=[r for r in range(1, B+1) if r not in state["present"]],
        unresolvedRepIDs=sorted(state["present"]-state["completed"]),
        baseline=method_report(state, "baseline"), candidate=method_report(state, "candidate"), paired=paired,
        contrastDiagnostics={key: {name: hi.distribution(values) for name, values in fields.items()}
                             for key, fields in state["diagnostics"].items()},
        technicalFailureReasons=dict(state["errors"]), originalGuardCounts=dict(state["guards"]))
    report["practicalTarget"] = practical_target(report)
    return report
