"""restart summary: accepted numerical definitions, portable source projection."""
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

from . import restart_decision as frozen
require = frozen.require
finite = frozen.finite

ROLE = "matched_validation"

METHOD = "strict_restart_port_check_v1"

PAIRS = ("baseline_strict", "baseline_PORT", "strict_PORT")

NUMBER_FIELDS = ("objectiveDifference", "deltaSERelativeDifference", "deltaDFRelativeDifference",
                 "thetaRelativeMaxDifference", "residualVarianceRelativeDifference")

def truth(case):
    if case["responseTransform"] == "gaussian_exact":
        return dict(known=True, beta=case["beta"], interpretation="gaussian_conditional_fixed_coefficient")
    require(case["responseTransform"] == "bounded_grid", "Unknown response transform")
    return dict(known=case["beta"] == 0, beta=0 if case["beta"] == 0 else None,
                interpretation="bounded_null_symmetry" if case["beta"] == 0 else "bounded_nonzero_observed_slope_unknown")

def rate(numerator, denominator):
    require(type(numerator) is int and type(denominator) is int and 0 <= numerator <= denominator, "Invalid rate counts")
    if denominator == 0:
        return dict(numerator=0, denominator=0, estimate=None, interval95=None)
    low = 0. if numerator == 0 else float(stats.beta.ppf(.025, numerator, denominator-numerator+1))
    high = 1. if numerator == denominator else float(stats.beta.ppf(.975, numerator+1, denominator-numerator))
    return dict(numerator=numerator, denominator=denominator, estimate=numerator/denominator, interval95=[low, high])

def describe(numbers):
    valid = sorted(x for x in numbers if finite(x))
    n = len(valid)
    return dict(finite=n, unknown=len(numbers)-n, mean=math.fsum(valid)/n if n else None,
                median=valid[(n-1)//2] if n else None,
                q10=valid[(n-1)//10] if n else None, q90=valid[(9*(n-1))//10] if n else None,
                minimum=valid[0] if n else None, maximum=valid[-1] if n else None)

def bool_counts(values):
    return {"true": sum(x is True for x in values), "false": sum(x is False for x in values),
            "unknown": sum(x is not True and x is not False for x in values)}

def covers(inference, target):
    if not target["known"] or inference is None:
        return None
    ci = inference.get("CI")
    require(isinstance(ci, list) and len(ci) == 2 and all(finite(x) for x in ci) and ci[0] <= ci[1], "Invalid available coverage interval")
    return ci[0] <= target["beta"] <= ci[1]

def compact(record):
    decision = record["decision"]
    diagnostics = record["diagnosticRecord"]
    directions, warnings = {}, {}
    for method in frozen.ROUTES:
        value = frozen.at(diagnostics, "methods", method, "raw", "coefficients", "delta")
        directions[method] = (1 if value > 0 else -1 if value < 0 else 0) if finite(value) else None
        warnings[method] = len(frozen.values(frozen.at(diagnostics, "methods", method, "raw", "diagnostics", "warnings")))
    return dict(caseID=record["caseID"], repID=record["repID"], oldDecision=record["oldDecision"],
                oldInference=record["oldInference"], truth=record["truth"], decision=decision,
                technicalCompletion=record["technicalCompletion"], routeDirections=directions, ordinaryWarningCounts=warnings)

def paired_table(rows, old_value, new_value, labels):
    keys = [a+"_to_"+b for a in labels for b in labels]
    ids = {key: [] for key in keys}
    for row in rows:
        key = labels[int(bool(old_value(row)))]+"_to_"+labels[int(bool(new_value(row)))]
        ids[key].append(row["repID"])
    return dict(denominator=len(rows), table={key: len(ids[key]) for key in keys}, ids=ids)

def change_ids(ids, states):
    return dict(changedIDs=[i for i, v in zip(ids, states) if v is True],
                unchangedIDs=[i for i, v in zip(ids, states) if v is False],
                unknownIDs=[i for i, v in zip(ids, states) if v is not True and v is not False])

def case_summary(shard, records):
    """Summarize compact rows; old matched decisions always retain all M IDs."""
    expected = [item["repID"] for item in shard["records"]]
    require(len(set(expected)) == len(expected) and set(records) <= set(expected), "Unexpected/duplicate summary IDs")
    m = len(expected); present = [records[i] for i in expected if i in records]
    completed = [r for r in present if r["technicalCompletion"]["complete"] is True]
    t = len(completed); available = [r for r in completed if r["decision"]["status"] == "available_candidate"]
    a = len(available); rejected = sum(r["decision"]["candidateNominalReject"] is True for r in completed)
    old_a = sum(item["oldDecision"]["status"] == "available_candidate" for item in shard["records"])
    old_r = sum(item["oldDecision"]["reject"] is True for item in shard["records"])
    target = truth(shard["case"])
    old_c = sum(covers(item["oldInference"], target) is True for item in shard["records"]) if target["known"] else None
    covered = sum(covers(row["decision"]["inference"], target) is True for row in available) if target["known"] else None
    both = [row for row in available if row["oldDecision"]["status"] == "available_candidate"]
    paired = dict(
        availability=paired_table(completed, lambda r: r["oldDecision"]["status"] == "available_candidate",
                                  lambda r: r["decision"]["status"] == "available_candidate", ("unavailable", "available")),
        rejection=paired_table(completed, lambda r: r["oldDecision"]["reject"], lambda r: r["decision"]["candidateNominalReject"], ("no_reject", "reject")),
        rejectionBothAvailable=paired_table(both, lambda r: r["oldDecision"]["reject"], lambda r: r["decision"]["candidateNominalReject"], ("no_reject", "reject")))
    original = dict(available=old_a, rejected=old_r, covered=old_c,
                    rates=dict(availabilityAll=rate(old_a, m), rejectionAll=rate(old_r, m), rejectionAmongAvailable=rate(old_r, old_a),
                               coverageAmongAvailable=rate(old_c, old_a) if target["known"] else None,
                               coverageAndAvailableAll=rate(old_c, m) if target["known"] else None))
    candidate = dict(available=a, unavailable=t-a, rejected=rejected, covered=covered,
        rates=dict(availabilityAll=rate(a, m) if t == m else None, availabilityCompleted=rate(a, t),
                   observedAvailabilityLowerBound=dict(numerator=a, denominator=m, estimate=a/m,
                                                      interpretation="observed_lower_bound_until_technical_completion"),
                   rejectionAll=rate(rejected, m) if t == m else None, rejectionCompleted=rate(rejected, t),
                   rejectionAmongAvailable=rate(rejected, a),
                   coverageAmongAvailable=rate(covered, a) if target["known"] else None,
                   coverageAndAvailableAll=rate(covered, m) if target["known"] and t == m else None))
    observed_reasons = Counter(reason for row in present for reason in row["decision"]["reasons"])
    completed_reasons = Counter(reason for row in completed for reason in row["decision"]["reasons"])
    primary = Counter(row["decision"]["reasons"][0] for row in completed if row["decision"]["reasons"])
    tech_reasons = Counter(reason for row in present for reason in row["technicalCompletion"]["reasons"])
    rows = [records.get(rep) for rep in expected]
    route_summaries = {}
    for method in frozen.ROUTES:
        facts = [frozen.at(row, "decision", "routeChecks", method) or {} for row in rows]
        route_summaries[method] = dict(statusCounts=dict(Counter(f.get("status", "missing") for f in facts)),
            guardCounts=bool_counts([f.get("guardUsable") for f in facts]),
            validReferenceCounts=bool_counts([f.get("validReference") for f in facts]),
            richardsonStatusCounts=dict(Counter(f.get("richardsonStatus", "missing") for f in facts)),
            ordinaryWarningCount=sum(frozen.at(row, "ordinaryWarningCounts", method) or 0 for row in rows))
    comparisons = {}
    for pair, first, second in (("baseline_strict", frozen.ROUTES[0], frozen.ROUTES[1]),
                                 ("baseline_PORT", frozen.ROUTES[0], frozen.ROUTES[2]),
                                 ("strict_PORT", frozen.ROUTES[1], frozen.ROUTES[2])):
        values = [frozen.at(row, "decision", "comparisons", pair) or {} for row in rows]
        directions = []
        for row in rows:
            x, y = frozen.at(row, "routeDirections", first), frozen.at(row, "routeDirections", second)
            directions.append(x != y if x is not None and y is not None else None)
        comparisons[pair] = dict(
            screens={name: bool_counts([frozen.at(v, "screens", name) for v in values]) for name in frozen.SCREENS},
            continuous={name: describe([v.get(name) for v in values]) for name in NUMBER_FIELDS},
            fixedCoefficientDifferences={name: describe([frozen.at(v, "fixedCoefficientDifferences", name) for v in values])
                                         for name in ("(Intercept)", "delta")},
            directionChanges=change_ids(expected, directions),
            pThresholdCrossings=change_ids(expected, [v.get("diagnosticPThresholdCrossing") for v in values]))
        for matrix_name in ("fixedVcovDifference", "randomCovarianceDifference"):
            cells = {}
            for i in range(2):
                for j in range(2):
                    numbers = []
                    for value in values:
                        array = value.get(matrix_name)
                        numbers.append(array[i][j] if isinstance(array, list) and len(array) == 2
                                       and isinstance(array[i], list) and len(array[i]) == 2 else None)
                    cells[f"{i}_{j}"] = describe(numbers)
            comparisons[pair][matrix_name] = cells
    nonworsening = {route: dict(counts=bool_counts([frozen.at(row, "decision", "nonworsening", route, "passes") for row in rows]),
                               objectiveDifference=describe([frozen.at(row, "decision", "nonworsening", route, "objectiveDifference") for row in rows]))
                    for route in ("strict", "PORT")}
    return dict(case=shard["case"], planned=m, recordPresent=len(present), technicalCompleted=t,
                technicalUnresolved=m-t, presentTechnicalUnresolved=len(present)-t, missing=m-len(present), complete=t == m,
                missingIDs=[i for i in expected if i not in records],
                technicalUnresolvedIDs=[i for i in expected if i not in records or records[i]["technicalCompletion"]["complete"] is not True],
                technicalReasons=dict(tech_reasons), original=original, candidate=candidate, paired=paired, truth=target,
                coverageReason=None if target["known"] else "bounded_nonzero_observed_slope_unknown",
                candidateReasonsCompleted=dict(completed_reasons), candidatePrimaryReasonsCompleted=dict(primary),
                candidateReasonsObserved=dict(observed_reasons),
                diagnostics=dict(scope="all_observed_records_including_technical_unresolved", routes=route_summaries,
                                 comparisons=comparisons, nonworsening=nonworsening))
