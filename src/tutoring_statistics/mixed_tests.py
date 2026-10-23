"""mixed tests: accepted numerical definitions, portable source projection."""
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



H10_IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")

SCOPE = "initial_raw_need_percent_allocation_candidate_not_adopted"

def _unavailable(reason, **more):
    return {"status": "unavailable", "reason": reason, "p": None,
            "candidateNominalReject": False, **more}

def directional_result(reference, tail, usable=True):
    """Recompute directed t inference from package coefficient/contrast output."""
    if tail not in ("greater", "less"):
        raise ValueError("Unsupported tail")
    out = dict(reference, tail=tail, candidateNominalReject=False)
    numbers = [reference.get(k) for k in ("estimate", "SE", "df", "statistic")]
    if any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in numbers):
        return {**out, **_unavailable("nonfinite_coefficient_inference")}
    estimate, se, df, statistic = numbers
    if se <= 0 or df <= 0 or not math.isclose(estimate/se, statistic, rel_tol=1e-8, abs_tol=1e-8):
        return {**out, **_unavailable("invalid_coefficient_inference")}
    p = float(stats.t.sf(statistic, df) if tail == "greater" else stats.t.cdf(statistic, df))
    out["diagnosticP"] = p
    if not usable:
        return {**out, **_unavailable("fit_diagnostics_not_acceptable")}
    out.update(status="available_candidate", reason=None, p=p,
               candidateNominalReject=p < .05)
    return out

def holm_four(tests):
    """Keep the four-item family, including unavailable members as p=1 padding."""
    if set(tests) != set(H10_IDS):
        raise ValueError("H10 family must contain exactly the four planned contrasts")
    ps = []
    for key in H10_IDS:
        result = tests[key]
        p = result.get("p")
        available = result.get("status") == "available_candidate"
        if available and (not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("Invalid available p-value")
        if not available and p is not None:
            raise ValueError("Unavailable contrast must retain null raw p")
        ps.append(float(p) if available else 1.)
    order = sorted(range(4), key=lambda i: ps[i])
    adjusted, previous = [1.]*4, 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (4-rank)*ps[i]))
        adjusted[i] = previous
    output = {}
    for i, key in enumerate(H10_IDS):
        available = tests[key].get("status") == "available_candidate"
        output[key] = dict(tests[key], holmP=adjusted[i] if available else None,
                          holmPaddingP=ps[i], holmReject=available and adjusted[i] < .05)
    return {"familySize": 4, "tests": output,
            "anyCandidateContrastRejects": any(t["holmReject"] for t in output.values()),
            "allFourCandidateContrastsReject": all(t["holmReject"] for t in output.values()),
            "claim": "signed_slope_differences_do_not_alone_establish_absolute_attenuation"}
