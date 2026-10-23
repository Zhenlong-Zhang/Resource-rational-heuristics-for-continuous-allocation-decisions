"""independent gls: accepted numerical definitions, portable source projection."""
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



CONTRASTS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")

CASE_INDICES = (1, 2, 4, 7, 8, 9, 10, 13, 14, 16, 19, 20, 22)

SCOPE = "saved_input_coverage_diagnostic_only"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def contrast_matrix(names):
    require(isinstance(names, list) and len(names) == len(set(names)) == 9,
            "Changed fixed coefficient names")
    expected = {"(Intercept)", "conditionC1", "conditionC3", "need_A", "need_B",
                "conditionC1:need_A", "conditionC3:need_A",
                "conditionC1:need_B", "conditionC3:need_B"}
    require(set(names) == expected, "Changed C2-reference fixed model")
    result = np.zeros((4, 9), dtype=np.float64)
    for i, key in enumerate(CONTRASTS):
        other, need = key.split(":")
        result[i, names.index("conditionC1:need_"+need)] = 1.0
        if other == "C1-C3":
            result[i, names.index("conditionC3:need_"+need)] = -1.0
    return result

def gls_oracle(extracted, *, sd_u, residual_sds, truths, parameters=None):
    """Known-Omega GLS through block whitening and QR, with no model refit."""
    require(finite(sd_u) and sd_u >= 0 and set(residual_sds) == {"C1", "C2", "C3"}
            and all(finite(v) and v > 0 for v in residual_sds.values()),
            "Invalid known DGP covariance")
    require(set(truths) == set(CONTRASTS) and all(finite(v) for v in truths.values()),
            "Changed oracle contrast truth")
    design = extracted["design"]
    x = np.asarray(design["X"], dtype=np.float64)
    y = np.asarray(design["y"], dtype=np.float64)
    people, arms = design["participant"], design["condition"]
    require(x.shape == (4620, 9) and y.shape == (4620,) and
            len(people) == len(arms) == 4620 and np.isfinite(x).all() and np.isfinite(y).all(),
            "Changed actual oracle frame")
    require(all(isinstance(v, str) and v for v in people) and set(arms) == {"C1", "C2", "C3"},
            "Changed actual cluster/arm labels")
    names = extracted["coefficientNames"]
    contrasts = contrast_matrix(names)
    a, b = x[:, names.index("need_A")], x[:, names.index("need_B")]
    encoded = {"(Intercept)": np.ones(4620), "need_A": a, "need_B": b}
    for arm in ("C1", "C3"):
        indicator = np.asarray([c == arm for c in arms], dtype=np.float64)
        encoded["condition"+arm] = indicator
        encoded["condition"+arm+":need_A"] = indicator*a
        encoded["condition"+arm+":need_B"] = indicator*b
    require(all(np.array_equal(x[:, j], encoded[name]) for j, name in enumerate(names)),
            "Changed actual fixed-model columns")
    for i, key in enumerate(CONTRASTS):
        require(np.array_equal(np.asarray(extracted["references"][key]["L"]), contrasts[i]),
                "Changed persisted contrast vector")
    clusters = {}
    for row, person in enumerate(people):
        clusters.setdefault(person, []).append(row)
    require(len(clusters) == 462, "Changed distinct participant count")
    wx, wy = np.empty_like(x), np.empty_like(y)
    lower_by_arm = {}
    for arm, sigma in residual_sds.items():
        covariance = np.full((10, 10), sd_u*sd_u, dtype=np.float64)
        covariance[np.diag_indices(10)] += sigma*sigma
        lower_by_arm[arm] = np.linalg.cholesky(covariance)
    arm_counts = dict.fromkeys(("C1", "C2", "C3"), 0)
    for indices in clusters.values():
        require(len(indices) == 10 and len({arms[i] for i in indices}) == 1,
                "Changed participant rows or nested condition")
        arm = arms[indices[0]]
        arm_counts[arm] += 1
        whitened = solve_triangular(lower_by_arm[arm], np.column_stack((x[indices], y[indices])),
                                    lower=True, check_finite=False)
        wx[indices], wy[indices] = whitened[:, :9], whitened[:, 9]
    require(set(arm_counts.values()) == {154}, "Changed analyzable arm counts")
    q, r = np.linalg.qr(wx, mode="reduced")
    require(np.isfinite(r).all() and np.all(np.diag(r) != 0), "Singular known-Omega design")
    beta = np.linalg.solve(r, q.T @ wy)
    inverse_r = np.linalg.solve(r, np.eye(9))
    covariance = inverse_r @ inverse_r.T
    contrast_covariance = contrasts @ covariance @ contrasts.T
    require(np.isfinite(beta).all() and np.isfinite(covariance).all()
            and np.all(np.diag(contrast_covariance) > 0), "Invalid known-Omega GLS result")
    critical = NormalDist().inv_cdf(.975)
    references = {}
    for i, key in enumerate(CONTRASTS):
        estimate = float(contrasts[i] @ beta)
        se = math.sqrt(float(contrast_covariance[i, i]))
        interval = [estimate-critical*se, estimate+critical*se]
        error = estimate-truths[key]
        references[key] = dict(L=contrasts[i].tolist(), estimate=estimate, SE=se,
            truth=truths[key], error=error, reference="known_covariance_normal", critical95=critical,
            CI=interval, halfWidth=critical*se, standardizedError=error/se,
            covered=interval[0] <= truths[key] <= interval[1],
            lowerMiss=truths[key] < interval[0], upperMiss=truths[key] > interval[1])
    return dict(method="known_DGP_Omega_GLS_oracle_v1", coefficientNames=names,
        beta=beta.tolist(), covariance=covariance.tolist(), contrastCovariance=contrast_covariance.tolist(),
        references=references, knownParameters=parameters, participantCount=len(clusters), rowCount=len(y),
        interpretation="truth_informed_reference_not_human_candidate")
