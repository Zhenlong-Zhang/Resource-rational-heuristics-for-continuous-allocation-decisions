"""paired reference: accepted numerical definitions, portable source projection."""
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
from . import fresh_reference as f
require = hi.require

NAMESPACE = "h10-cr2-prospective-paired-v1-20261020"

SCOPE = "prospective_paired_coverage_validation"

METHOD = hi.METHOD

ORACLE = "known_DGP_Omega_GLS_oracle_v1"

CASE_ID = "h10m1_terminal_amount_s2_n154"

B = 10000

TARGETS = ("C1-C3:A", "C1-C3:B")

RULE = dict(firstRep=1, lastRep=B, firstOrdinal=0, lastOrdinal=B-1,
            order="repID_ascending", casesPerRep=1)

def case_for(rep):
    require(type(rep) is int and 1 <= rep <= B, "Replicate outside fixed paired roster")
    case = copy.deepcopy(m.measurement_plan()["cases"][19])
    require(case["caseID"] == CASE_ID and case["caseIndex"] == 20 and case["scenarioIndex"] == 2,
            "Changed accepted case template")
    case.update(repID=rep, designSeed=602610200+20000*2+rep,
                responseSeed=702610200+20000*20+rep)
    return case

def fit_key(case):
    return f"{case['caseID']}__r{case['repID']:05d}"

def rep_ids(value):
    require(isinstance(value, list) and value and
            all(type(x) is int and 1 <= x <= B for x in value) and
            value == sorted(set(value)), "Invalid duplicate or unordered paired repIDs")
    return value

def oracle_input(actual, kernel):
    rows = actual["rows"]
    names = list(kernel.NAMES)
    X = [[1., float(r["condition"] == "C1"), float(r["condition"] == "C3"),
          r["need_A"], r["need_B"],
          r["need_A"] if r["condition"] == "C1" else 0.,
          r["need_A"] if r["condition"] == "C3" else 0.,
          r["need_B"] if r["condition"] == "C1" else 0.,
          r["need_B"] if r["condition"] == "C3" else 0.] for r in rows]
    L = kernel.contrast_matrix(names)
    return dict(coefficientNames=names, design=dict(X=X, y=[r["y"] for r in rows],
        participant=[r["participant"] for r in rows], condition=[r["condition"] for r in rows]),
        references={key: dict(L=L[i].tolist()) for i, key in enumerate(m.H10_IDS)})

def pair_events(candidate_branches, oracle):
    if candidate_branches is None or not oracle["technicalCompleted"] or not oracle["applicable"]:
        return None
    result = {}
    for key in TARGETS:
        candidate = bool(candidate_branches[0]["coverage"][key]["C"])
        reference = oracle["result"]["references"][key]["covered"]
        require(type(reference) is bool, "Unknown oracle coverage")
        result[key] = dict(candidateCovered=candidate, oracleCovered=reference,
            jointAvailable=bool(candidate_branches[0]["coverage"][key]["A"] and
                                candidate_branches[0]["coverage"][key]["CIAvailable"]),
            a=int(candidate and reference), b=int(not candidate and reference),
            c=int(candidate and not reference), d=int(not candidate and not reference))
    return result

def measurements(rows):
    total = len(rows); joint = sum(r["technicalCompleted"] for r in rows)
    return dict(M=total, T=joint, C=sum(r["cr2TechnicalCompleted"] for r in rows),
        O=sum(r["oracleTechnicalCompleted"] for r in rows),
        oracleApplicable=sum(r["oracleApplicable"] for r in rows),
        technicalMissing=total-joint, complete=joint == total)

def cp(k, n, level):
    require(type(k) is int and type(n) is int and 0 <= k <= n and 0 < level < 1, "Invalid CP counts")
    if n == 0:
        return None
    alpha = 1-level
    return [0. if k == 0 else float(beta_distribution.ppf(alpha/2, k, n-k+1)),
            1. if k == n else float(beta_distribution.ppf(1-alpha/2, k+1, n-k))]

def paired_report(tables, completed, planned=B):
    require(type(completed) is int and 0 <= completed <= planned, "Invalid paired denominator")
    complete = completed == planned; result = {}; ps = {}
    for key in TARGETS:
        counts = {name: int(tables[key][name]) for name in ("a", "b", "c", "d")}
        require(min(counts.values()) >= 0 and sum(counts.values()) == completed, "Invalid paired cells")
        b, c = counts["b"], counts["c"]; discordant = b+c
        p = float(binom.sf(b-1, discordant, .5)) if discordant else 1.
        # These are MC intervals on joint-cell probabilities, never widened
        # scientific coefficient CIs. Four 98.75% CP intervals give >=95%
        # simultaneous coverage for the two target coverage differences.
        bi, ci = cp(b, completed, .9875), cp(c, completed, .9875)
        delta_interval = None if bi is None else [ci[0]-bi[1], ci[1]-bi[0]]
        result[key] = dict(counts=dict(M=planned, T=completed, **counts), discordant=discordant,
            coverageDifference=None if not complete else (c-b)/planned,
            completedCoverageDifference=None if completed == 0 else (c-b)/completed,
            mcSimultaneousDifferenceInterval=delta_interval,
            intervalDenominator=completed,
            intervalPopulation="full_planned" if complete else "completed_subset_only",
            nominalOracleAnchorApplicable=complete,
            cellIntervalLevel=.9875, intervalScope="two_targets_Bonferroni_CP_difference_MC_only",
            mcnemarP=p if complete else None, completedDiagnosticP=p if completed else None,
            holmAdjustedP=None, holmReject=None)
        if complete:
            ps[key] = p
    if complete:
        running = 0.
        for rank, key in enumerate(sorted(TARGETS, key=lambda k: (ps[k], TARGETS.index(k)))):
            running = max(running, (2-rank)*ps[key])
            result[key]["holmAdjustedP"] = min(1., running)
            result[key]["holmReject"] = min(1., running) <= .05
    intervals = [result[key]["mcSimultaneousDifferenceInterval"] for key in TARGETS]
    practical = "inconclusive"
    if complete:
        if all(ci[0] >= -.015 for ci in intervals):
            practical = "supported"
        elif any(ci[1] < -.015 for ci in intervals):
            practical = "adverse"
    return dict(primaryFamily=list(TARGETS), alpha=.05, alternative="CR2_coverage_less_than_oracle",
        complete=complete, tests=result,
        practicalDifference=dict(margin=-.015, classification=practical,
            technicalAndOracleComplete=complete, simultaneousMCTargets=list(TARGETS)),
        interpretation="fixed_prospective_MC_family_not_human_power_or_new_scientific_contrast")

def available_paired_report(tables):
    tests = {}
    for key in TARGETS:
        counts = {name: int(tables[key][name]) for name in ("a", "b", "c", "d")}
        n = sum(counts.values()); bi, ci = cp(counts["b"], n, .9875), cp(counts["c"], n, .9875)
        tests[key] = dict(counts=dict(M=B, jointAvailable=n, **counts),
            coverageDifference=(counts["c"]-counts["b"])/n if n else None,
            mcDifferenceInterval=None if n == 0 else [ci[0]-bi[1], ci[1]-bi[0]],
            intervalDenominator=n, intervalPopulation="joint_available_only",
            nominalOracleAnchorApplicable=False, formalPrimaryTest=False)
    return dict(tests=tests, interpretation="descriptive_selection_conditional_no_nominal_oracle_anchor")
