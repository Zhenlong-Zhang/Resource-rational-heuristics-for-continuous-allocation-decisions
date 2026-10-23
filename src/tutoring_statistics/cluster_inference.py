"""cluster inference: accepted numerical definitions, portable source projection."""
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
import statistics
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

METHOD = "participant_CR2_Satterthwaite_v1"

SCOPE = "fixed_existing_fit_development"

def require(value, message):
    if not value:
        raise ValueError(message)

def cases_for(rep):
    require(type(rep) is int and 1 <= rep <= 20, "Replicate outside fixed development subset")
    cases = copy.deepcopy(m.measurement_plan()["cases"])
    for case in cases:
        case.update(repID=rep, designSeed=202610190+10000*case["scenarioIndex"]+rep,
                    responseSeed=302610190+10000*case["caseIndex"]+rep)
    return cases

def fit_key(case):
    return case["caseID"] if case["repID"] == 1 else f"{case['caseID']}__r{case['repID']:04d}"

def finite(x):
    return type(x) in (int, float) and math.isfinite(x)

def ratio(a, b):
    return a/b if finite(a) and finite(b) and b > 0 else None

def planning_mde(reference):
    se, df = reference.get("SE"), reference.get("df")
    if not finite(se) or not finite(df) or se <= 0 or df <= 0:
        return None
    result = {}
    for label, alpha in (("alpha_0.05", .05), ("alpha_0.0125", .0125)):
        critical = float(t_distribution.ppf(1-alpha, df))
        try:
            ncp = float(brentq(lambda x: float(nct.sf(critical, df, x))-.8, 0., 1000., xtol=1e-10, rtol=1e-12))
            require(finite(ncp), "Undefined noncentral-t planning approximation")
            result[label] = dict(alpha=alpha, targetPower=.8, noncentrality=ncp, effectMagnitude=se*ncp)
        except (ValueError, ArithmeticError) as exc:
            result[label] = dict(alpha=alpha, targetPower=.8, noncentrality=None, effectMagnitude=None,
                                 diagnosticError=str(exc))
    return result

def paired_result(baseline, candidate):
    results = []
    for old in baseline:
        new = None if candidate is None else candidate[len(results)]
        transitions = {}
        for label, field in (("availability", "A4"), ("falseRejection", "F"), ("anyRejection", "R"), ("alternativeRejection", "D")):
            transitions[label] = dict(baseline=bool(old["counts"][field]), candidate=None if new is None else bool(new["counts"][field]))
        contrasts = {}
        for key in m.H10_IDS:
            a = old["tests"][key]; b = None if new is None else new["tests"][key]
            x, y = a.get("reference") or {}, {} if b is None else b.get("reference") or {}
            def width(z):
                ci = z.get("CI")
                return ci[1]-ci[0] if isinstance(ci, list) and len(ci) == 2 and all(finite(v) for v in ci) and ci[1] >= ci[0] else None
            contrasts[key] = dict(estimate=x.get("estimate"), baselineSE=x.get("SE"), candidateSE=y.get("SE"),
                baselineDf=x.get("df"), candidateDf=y.get("df"), SERatio=ratio(y.get("SE"), x.get("SE")),
                dfRatio=ratio(y.get("df"), x.get("df")), CIWidthRatio=ratio(width(y), width(x)),
                baselineCovered=old["coverage"][key]["covered"], candidateCovered=None if new is None else new["coverage"][key]["covered"],
                baselineAvailable=a["status"] == "available_candidate",
                candidateAvailable=None if b is None else b["status"] == "available_candidate",
                baselineReject=a["holmReject"], candidateReject=None if b is None else b["holmReject"],
                assumedEffectMagnitudeOverBaselineSE=ratio(abs(old["coverage"][key]["truth"]), x.get("SE")),
                assumedEffectMagnitudeOverCandidateSE=ratio(abs(old["coverage"][key]["truth"]), y.get("SE")),
                baselinePlanningMDE=planning_mde(x), candidatePlanningMDE=planning_mde(y))
        results.append(dict(tailMode=old["tailMode"], transitions=transitions, contrasts=contrasts))
    return results

def rate(k, n):
    if n == 0:
        return None
    require(0 <= k <= n, "Invalid binomial count")
    return dict(numerator=k, denominator=n, estimate=k/n, interval95=[0. if k == 0 else float(beta_distribution.ppf(.025, k, n-k+1)),
        1. if k == n else float(beta_distribution.ppf(.975, k+1, n-k))], intervalMethod="pointwise_Clopper_Pearson")

def distribution(values):
    available = sorted(v for v in values if finite(v))
    if not available:
        return dict(n=0, missing=len(values), mean=None, SD=None, min=None, median=None, max=None)
    return dict(n=len(available), missing=len(values)-len(available), mean=statistics.mean(available),
        SD=statistics.stdev(available) if len(available)>1 else None, min=available[0],
        median=available[(len(available)-1)//2], max=available[-1])

def table(pairs):
    counts = Counter("unknown" if type(a) is not bool or type(b) is not bool else f"{int(a)}{int(b)}" for a, b in pairs)
    return {k: counts[k] for k in ("00", "01", "10", "11", "unknown")}
