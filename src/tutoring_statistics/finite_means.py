"""Ordered finite-belief means used by the native design adapter."""
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

def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

def _logs(values):
    if not isinstance(values, (list, tuple)) or len(values) != 100 or not all(finite(v) for v in values):
        raise ValueError("Exactly 100 finite log weights required")
    return list(values)

def finite_mean(logs):
    logs = _logs(logs)
    peak = max(logs)
    weights = [math.exp(x - peak) for x in logs]
    return math.fsum((i + 1) * w for i, w in enumerate(weights)) / math.fsum(weights)

def canonical_mean(values):
    """One specified binary64 map for scalar, vector and finite truth endpoints.

    Sum in main-index order using separate NumPy add operations, then divide.
    The last axis is rounds. This is not data-dependent recentering.
    """
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] == 0:
        raise ValueError("Mean needs a nonempty last axis")
    total = np.zeros(a.shape[:-1], dtype=np.float64)
    for i in range(a.shape[-1]):
        total = np.add(total, a[..., i])
    return np.divide(total, np.float64(a.shape[-1]))
