"""independent paired: accepted numerical definitions, portable source projection."""
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



NAMESPACE = "h10-cr2-prospective-paired-v1-20261020"

LABEL_NAMESPACE = "h10-gaussian-reference-v1-20261019"

METHOD = "participant_CR2_Satterthwaite_v1"

CASE_ID = "h10m1_terminal_amount_s2_n154"

IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")

TARGETS = IDS[2:]

MODES = ("literal", "sign_consistent")

SCOPE = "prospective_paired_coverage_validation"

ORACLE = "known_DGP_Omega_GLS_oracle_v1"

CASE_INDEX, SCENARIO_INDEX, TOTAL = 20, 2, 10000

def validate_rep_ids(values):
    require(type(values) is list and bool(values) and
            all(type(value) is int and 1 <= value <= TOTAL for value in values) and
            values == sorted(set(values)), "Changed fixed prospective rep roster")
    return values

def identity(rep, template):
    validate_rep_ids([rep])
    require(template["caseID"] == CASE_ID and template["caseIndex"] == CASE_INDEX
            and template["scenarioIndex"] == SCENARIO_INDEX, "Changed prospective case")
    result = copy.deepcopy(template)
    result.update(repID=rep, designSeed=602610200+20000*SCENARIO_INDEX+rep,
                  responseSeed=702610200+20000*CASE_INDEX+rep)
    return result

def fit_key(rep):
    validate_rep_ids([rep])
    return f"{CASE_ID}__r{rep:05d}"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def exact(actual, expected, label="copied evidence"):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), "Changed "+label+" keys")
        for key in expected:
            exact(actual[key], expected[key], label+"."+key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "Changed "+label+" length")
        for i, item in enumerate(expected):
            exact(actual[i], item, label+f"[{i}]")
    else:
        require(type(actual) is type(expected) and actual == expected, "Changed "+label)

def near(actual, expected, label="independent numbers", *, atol=2e-9, rtol=2e-8):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), "Changed "+label+" keys")
        for key in expected:
            near(actual[key], expected[key], label+"."+key, atol=atol, rtol=rtol)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "Changed "+label+" length")
        for i, item in enumerate(expected):
            near(actual[i], item, label+f"[{i}]", atol=atol, rtol=rtol)
    elif finite(expected):
        require(finite(actual) and (type(expected) is not int or type(actual) is int)
                and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), "Changed "+label)
    else:
        exact(actual, expected, label)

def interval(k, n, confidence=.95):
    """Central Clopper-Pearson via incomplete-beta inversion, not producer ppf."""
    require(type(k) is int and type(n) is int and 0 <= k <= n and n > 0 and
            finite(confidence) and 0 < confidence < 1, "Invalid CP counts/confidence")
    alpha = (1-confidence)/2
    def inverse(a, b, probability):
        low, high = 0., 1.
        for _ in range(64):
            middle = (low+high)/2
            if float(special.betainc(a, b, middle)) < probability:
                low = middle
            else:
                high = middle
        return (low+high)/2
    return [0. if k == 0 else inverse(k, n-k+1, alpha),
            1. if k == n else inverse(k+1, n-k, 1-alpha)]

def holm_two(pvalues):
    require(len(pvalues) == 2 and all(finite(v) and 0 <= v <= 1 for v in pvalues),
            "Changed fixed two-target MC family")
    order = sorted(range(2), key=lambda i: (pvalues[i], i))
    adjusted, previous = [None, None], 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (2-rank)*pvalues[index]))
        adjusted[index] = previous
    return adjusted

def mcnemar_greater(b, c):
    require(type(b) is int and type(c) is int and b >= 0 and c >= 0, "Invalid discordant counts")
    d = b+c
    return 1. if d == 0 or b == 0 else float(special.bdtrc(b-1, d, .5))

def paired_delta_interval(b, c, n):
    require(type(b) is int and type(c) is int and b >= 0 and c >= 0 and b+c <= n,
            "Invalid paired cell partition")
    lb, ub = interval(b, n, .9875)
    lc, uc = interval(c, n, .9875)
    return dict(bInterval9875=[lb, ub], cInterval9875=[lc, uc], deltaIntervalSimultaneous95=[lc-ub, uc-lb])

def oracle_cells(cr2_covered, oracle_covered):
    require(type(cr2_covered) is bool and type(oracle_covered) is bool, "Unknown coverage became paired miss")
    return dict(a=int(cr2_covered and oracle_covered), b=int(not cr2_covered and oracle_covered),
                c=int(cr2_covered and not oracle_covered), d=int(not cr2_covered and not oracle_covered))

def paired_statistics(cells, denominator, *, complete):
    require(type(denominator) is int and denominator >= 0 and type(complete) is bool and
            type(cells) is dict and set(cells) == {"a", "b", "c", "d"} and
            all(type(value) is int and value >= 0 for value in cells.values()) and
            sum(cells.values()) == denominator, "Invalid exact paired partition/denominator")
    delta = (cells["c"]-cells["b"])/denominator if denominator else None
    p = mcnemar_greater(cells["b"], cells["c"]) if denominator else None
    bounds = paired_delta_interval(cells["b"], cells["c"], denominator) if denominator else None
    # The computable completed subset remains descriptive when planned
    # technical/oracle prerequisites do not hold for every planned input.
    return dict(cells=dict(cells), denominator=denominator, delta=delta, p=p, intervals=bounds,
                primaryReleased=bool(complete and denominator > 0),
                oracleNominalAnchor=bool(complete and denominator > 0))

def measurement(rows):
    total = len(rows)
    joint = sum(row["technicalCompleted"] for row in rows)
    return dict(M=total, T=joint, C=sum(row["cr2TechnicalCompleted"] for row in rows),
        O=sum(row["oracleTechnicalCompleted"] for row in rows),
        oracleApplicable=sum(row["oracleApplicable"] for row in rows),
        technicalMissing=total-joint, complete=joint == total)

def pair_events(branches, oracle):
    if branches is None or not oracle["technicalCompleted"] or not oracle["applicable"]:
        return None
    values = {}
    for key in TARGETS:
        candidate = bool(branches[0]["coverage"][key]["C"])
        reference = oracle["result"]["references"][key]["covered"]
        values[key] = dict(candidateCovered=candidate, oracleCovered=reference,
                           jointAvailable=bool(branches[0]["coverage"][key]["A"] and
                                               branches[0]["coverage"][key]["CIAvailable"]),
                           **oracle_cells(candidate, reference))
    return values

def compact_events(row, fresh):
    original = dict(row, technicalCompleted=row["cr2TechnicalCompleted"], technicalError=row["cr2TechnicalError"])
    result = fresh.compact_events(original)
    result.update(case=row["case"], technicalCompleted=row["technicalCompleted"],
        cr2TechnicalCompleted=row["cr2TechnicalCompleted"], cr2TechnicalError=row["cr2TechnicalError"],
        oracleTechnicalCompleted=row["oracleTechnicalCompleted"], oracleTechnicalError=row["oracleTechnicalError"],
        oracleApplicable=row["oracleApplicable"], pairedOracle=row["pairedOracle"],
        oracleResult=None if row["oracleResult"] is None else dict(method=ORACLE,
            references=row["oracleResult"]["references"]))
    return result

def paired_report(tables, completed, planned=TOTAL):
    require(type(completed) is int and 0 <= completed <= planned and type(planned) is int and planned > 0,
            "Invalid primary planned/completed denominator")
    complete, tests, pvalues = completed == planned, {}, []
    for key in TARGETS:
        counts = {name: tables[key].get(name, 0) for name in "abcd"}
        check = paired_statistics(counts, completed, complete=complete)
        bounds = check["intervals"]
        tests[key] = dict(counts=dict(M=planned, T=completed, **counts), discordant=counts["b"]+counts["c"],
            coverageDifference=check["delta"] if complete else None,
            completedCoverageDifference=check["delta"],
            mcSimultaneousDifferenceInterval=None if bounds is None else bounds["deltaIntervalSimultaneous95"],
            intervalDenominator=completed, intervalPopulation="full_planned" if complete else "completed_subset_only",
            nominalOracleAnchorApplicable=complete, cellIntervalLevel=.9875,
            intervalScope="two_targets_Bonferroni_CP_difference_MC_only",
            mcnemarP=check["p"] if complete else None, completedDiagnosticP=check["p"],
            holmAdjustedP=None, holmReject=None)
        pvalues.append(check["p"])
    if complete:
        for key, adjusted in zip(TARGETS, holm_two(pvalues)):
            tests[key].update(holmAdjustedP=adjusted, holmReject=adjusted <= .05)
    intervals = [tests[key]["mcSimultaneousDifferenceInterval"] for key in TARGETS]
    classification = "inconclusive"
    if complete:
        if all(value[0] >= -.015 for value in intervals):
            classification = "supported"
        elif any(value[1] < -.015 for value in intervals):
            classification = "adverse"
    return dict(primaryFamily=list(TARGETS), alpha=.05, alternative="CR2_coverage_less_than_oracle",
        complete=complete, tests=tests,
        practicalDifference=dict(margin=-.015, classification=classification,
            technicalAndOracleComplete=complete, simultaneousMCTargets=list(TARGETS)),
        interpretation="fixed_prospective_MC_family_not_human_power_or_new_scientific_contrast")

def available_paired_report(tables, planned=TOTAL):
    tests = {}
    for key in TARGETS:
        counts = {name: tables[key].get(name, 0) for name in "abcd"}
        n = sum(counts.values())
        check = paired_statistics(counts, n, complete=False)
        tests[key] = dict(counts=dict(M=planned, jointAvailable=n, **counts), coverageDifference=check["delta"],
            mcDifferenceInterval=None if check["intervals"] is None else check["intervals"]["deltaIntervalSimultaneous95"],
            intervalDenominator=n, intervalPopulation="joint_available_only", nominalOracleAnchorApplicable=False,
            formalPrimaryTest=False)
    return dict(tests=tests, interpretation="descriptive_selection_conditional_no_nominal_oracle_anchor")

def reduce_events(events, ctx, planned=TOTAL):
    """Independent event reduction. No producer import or RDS/global raw scan."""
    require(type(planned) is int and planned > 0 and len(events) <= planned and
            len({row["repID"] for row in events}) == len(events), "Duplicate prospective event identity")
    cr2_rows, joint_rows, oracle_rows = [], [], []
    tables = {key: dict.fromkeys("abcd", 0) for key in TARGETS}
    available = {key: dict.fromkeys("abcd", 0) for key in TARGETS}
    oracle_errors = {}
    for row in events:
        rep = row["repID"]
        validate_rep_ids([rep])
        require(rep <= planned and row["caseID"] == CASE_ID and row["ordinal"] == rep-1 and row["fitKey"] == fit_key(rep),
                "Changed scientific event identity")
        exact(row["case"], identity(rep, ctx["case"]), "exact event case/seed/truth")
        flags = ("technicalCompleted", "cr2TechnicalCompleted", "oracleTechnicalCompleted", "oracleApplicable")
        require(all(type(row[key]) is bool for key in flags) and
                row["technicalCompleted"] is (row["cr2TechnicalCompleted"] and row["oracleTechnicalCompleted"]) and
                row["oracleApplicable"] is row["oracleTechnicalCompleted"], "Changed separate technical event semantics")
        cr2 = dict(row, technicalCompleted=row["cr2TechnicalCompleted"], technicalError=row["cr2TechnicalError"])
        cr2_rows.append(cr2)
        if row["oracleTechnicalCompleted"]:
            require(row["oracleTechnicalError"] is None and row["oracleResult"]["method"] == ORACLE,
                    "Oracle unknown counted as result")
            for key in IDS:
                ref = row["oracleResult"]["references"][key]
                require(type(ref["covered"]) is bool, "Unknown oracle coverage counted as miss")
                exact(ref["truth"], row["case"]["delta"][key], "oracle event truth")
            oracle_rows.append(row)
        else:
            require(row["oracleResult"] is None and type(row["oracleTechnicalError"]) is dict,
                    "Oracle error became scientific noncoverage")
            phase = row["oracleTechnicalError"]["phase"]
            oracle_errors[phase] = oracle_errors.get(phase, 0)+1
        if not row["cr2TechnicalCompleted"]:
            require(type(row["cr2TechnicalError"]) is dict and "candidateBranches" not in row,
                    "Unknown candidate became ordinary nonrejection")
        else:
            require(row["cr2TechnicalError"] is None, "Completed candidate retains technical error")
        if row["technicalCompleted"]:
            joint_rows.append(row)
            expected = pair_events(row["candidateBranches"], dict(technicalCompleted=True,
                applicable=True, result=row["oracleResult"]))
            exact(row["pairedOracle"], expected, "independent paired coverage cells")
            for key in TARGETS:
                for cell in "abcd":
                    tables[key][cell] += expected[key][cell]
                    if expected[key]["jointAvailable"]:
                        available[key][cell] += expected[key][cell]
        else:
            require(row["pairedOracle"] is None, "Unknown joint event became a paired miss")
    modules = ctx["modules"]
    case_report = modules["fresh"].reduce_events(ctx["case"], cr2_rows, modules["math"], planned=planned)
    total_oracle = len(oracle_rows)
    oracle_report = {}
    for key in IDS:
        refs = [row["oracleResult"]["references"][key] for row in oracle_rows]
        C = sum(ref["covered"] for ref in refs)
        oracle_report[key] = dict(counts=dict(M=planned, O=total_oracle, C=C),
            coverageAll=modules["math"].rate(C, planned) if total_oracle == planned else None,
            coverageApplicable=modules["math"].rate(C, total_oracle),
            diagnostics={name: modules["math"].distribution([ref[name] for ref in refs])
                         for name in ("estimate", "SE", "error", "halfWidth")})
    secondary = {}
    for label, counts, total in (
            ("candidate", {key: case_report["candidate"]["coverage"][key]["counts"]["C"] for key in IDS}, case_report["T"]),
            ("oracle", {key: oracle_report[key]["counts"]["C"] for key in IDS}, total_oracle)):
        secondary[label] = {key: dict(C=counts[key], M=planned, T=total,
            nominal95UndercoverageP=float(special.bdtr(counts[key], planned, .95)) if total == planned else None,
            interpretation="secondary_absolute_nominal_not_selected_against_primary") for key in IDS}
    present = {row["repID"] for row in events}
    joint = len(joint_rows)
    return dict(expectedTotal=planned, presentTotal=len(events), completedTotal=joint,
        cr2CompletedTotal=case_report["T"], oracleCompletedTotal=total_oracle, oracleApplicableTotal=total_oracle,
        technicalUnresolved=len(events)-joint, missingTotal=planned-len(events), complete=joint == planned,
        missingRepIDs=sorted(set(range(1, planned+1))-present),
        case=case_report, oracleCoverage=oracle_report, oracleTechnicalFailures=oracle_errors,
        primaryPaired=paired_report(tables, joint, planned), pairedAvailable=available_paired_report(available, planned),
        secondaryAbsolute=secondary)
