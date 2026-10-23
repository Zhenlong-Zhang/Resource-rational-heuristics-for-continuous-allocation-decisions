"""independent family: accepted numerical definitions, portable source projection."""
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



IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")

FIT_KEYS = ("initial_amount", "initial_percent", "terminal_amount", "terminal_percent")

PROFILE = "pilot-r9-design-v1"

FLOW = "r10-prepractice-v1"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def near(actual, expected, label="numeric", *, atol=2e-9, rtol=2e-8):
    if expected is None or isinstance(expected, (bool, str)):
        require(type(actual) is type(expected) and actual == expected, f"Changed {label}")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Changed {label} fields")
        for key, value in expected.items():
            near(actual[key], value, f"{label}.{key}", atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Changed {label} shape")
        for i, value in enumerate(expected):
            near(actual[i], value, f"{label}[{i}]", atol=atol, rtol=rtol)
    else:
        require(finite(actual) and finite(expected) and
                math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), f"Changed {label}")

def directed(reference, tail, usable):
    require(tail in ("less", "greater"), "Unknown directed tail")
    values = [reference.get(key) for key in ("estimate", "SE", "df", "statistic")]
    reason = None
    if not all(finite(x) for x in values):
        reason = "nonfinite_coefficient_inference"
    else:
        estimate, se, df, statistic = values
        if se <= 0 or df <= 0 or not math.isclose(estimate / se, statistic, abs_tol=1e-8, rel_tol=1e-8):
            reason = "invalid_coefficient_inference"
    diagnostic = None
    if reason is None:
        diagnostic = float(stats.t.cdf(statistic, df) if tail == "less" else stats.t.sf(statistic, df))
        if not usable:
            reason = "fit_diagnostics_not_acceptable"
    p = diagnostic if reason is None else None
    result = dict(tail=tail, status="available_candidate" if reason is None else "unavailable",
                  reason=reason, p=p, candidateNominalReject=p is not None and p < .05)
    if diagnostic is not None:
        result["diagnosticP"] = diagnostic
    return result

def family(tests):
    require(set(tests) == set(IDS), "Exactly four fixed comparisons required")
    ps = [tests[key]["p"] if tests[key]["status"] == "available_candidate" else 1. for key in IDS]
    require(all(finite(p) and 0 <= p <= 1 for p in ps), "Invalid family p")
    order = sorted(range(4), key=lambda i: ps[i])
    adjusted, previous = [1.] * 4, 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (4 - rank) * ps[i]))
        adjusted[i] = previous
    output = {}
    for i, key in enumerate(IDS):
        available = tests[key]["status"] == "available_candidate"
        output[key] = dict(tests[key], holmP=adjusted[i] if available else None, holmPaddingP=ps[i],
                           holmReject=available and adjusted[i] < .05)
    return dict(familySize=4, tests=output,
                anyCandidateContrastRejects=any(value["holmReject"] for value in output.values()),
                completeFamilyAvailable=all(value["status"] == "available_candidate" for value in output.values()))
