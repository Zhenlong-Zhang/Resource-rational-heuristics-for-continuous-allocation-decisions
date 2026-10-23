"""oracle gls: accepted numerical definitions, portable source projection."""
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

ARMS = ("C1", "C2", "C3")

NAMES = ("(Intercept)", "conditionC1", "conditionC3", "need_A", "need_B",
         "conditionC1:need_A", "conditionC3:need_A",
         "conditionC1:need_B", "conditionC3:need_B")

CASE_INDICES = (1, 2, 4, 7, 8, 9, 10, 13, 14, 16, 19, 20, 22)

FRESH_NAMESPACE = "h10-cr2-validation-v1-20261019"

METHOD = "participant_CR2_Satterthwaite_v1"

SCOPE = "saved_input_coverage_diagnostic_only"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def contrast_matrix(names):
    require(tuple(names) == NAMES, "Changed nine named fixed coefficients")
    L = np.zeros((4, 9))
    L[0, 5] = L[1, 7] = L[2, 5] = L[3, 7] = 1.
    L[2, 6] = L[3, 8] = -1.
    return L

def block_information(x, response, random_sd, residual_sd):
    require(x.ndim == 2 and x.shape[0] == 10 and response.shape == (10,) and
            np.isfinite(x).all() and np.isfinite(response).all() and
            finite(random_sd) and random_sd >= 0 and finite(residual_sd) and residual_sd > 0,
            "Invalid known covariance block")
    coefficient = random_sd**2/(residual_sd**2 + 10*random_sd**2)
    sx, sy = x.sum(axis=0), float(response.sum())
    return ((x.T @ x - coefficient*np.outer(sx, sx))/(residual_sd**2),
            (x.T @ response - coefficient*sx*sy)/(residual_sd**2))

def known_gls(extracted, case):
    X = np.asarray(extracted["design"]["X"], dtype=float)
    y = np.asarray(extracted["design"]["y"], dtype=float)
    people, arms = extracted["design"]["participant"], extracted["design"]["condition"]
    require(X.shape == (4620, 9) and y.shape == (4620,) and len(people) == len(arms) == 4620 and
            np.isfinite(X).all() and np.isfinite(y).all(), "Invalid saved actual design")
    L = contrast_matrix(extracted["coefficientNames"])
    positions = {}
    for i, person in enumerate(people):
        require(isinstance(person, str) and person and arms[i] in ARMS, "Invalid saved cluster identity")
        positions.setdefault(person, []).append(i)
    require(len(positions) == 462 and all(len(ix) == 10 for ix in positions.values()) and
            all(len({p for p, c in zip(people, arms) if c == arm}) == 154 for arm in ARMS),
            "Changed fixed participant population")
    # Actual X must encode the declared condition-by-two-needs mean model.
    a, b = X[:, 3], X[:, 4]
    c1, c3 = np.asarray([c == "C1" for c in arms]), np.asarray([c == "C3" for c in arms])
    expected = np.column_stack((np.ones(4620), c1, c3, a, b, c1*a, c3*a, c1*b, c3*b))
    require(np.array_equal(X, expected), "Changed actual fixed model columns")
    tau = case["parameters"]["sdU"]
    require(finite(tau) and tau >= 0, "Invalid known random-intercept SD")
    information, score = np.zeros((9, 9)), np.zeros(9)
    for indices in positions.values():
        arm = arms[indices[0]]
        require(all(arms[i] == arm for i in indices), "Participant crosses condition")
        sigma = case["parameters"]["sigmaByArm"][arm]
        require(finite(sigma) and sigma > 0, "Invalid known residual SD")
        x, response = X[indices], y[indices]
        block_matrix, block_score = block_information(x, response, tau, sigma)
        information += block_matrix
        score += block_score
    np.linalg.cholesky(information)
    beta = np.linalg.solve(information, score)
    covariance = np.linalg.solve(information, np.eye(9))
    C = L @ covariance @ L.T
    require(np.isfinite(beta).all() and np.isfinite(C).all() and np.all(np.diag(C) > 0),
            "Invalid known-covariance oracle")
    critical = float(stats.norm.ppf(.975))
    refs = {}
    for i, key in enumerate(IDS):
        require(np.array_equal(np.asarray(extracted["references"][key]["L"], dtype=float), L[i]),
                "Changed actual contrast vector")
        estimate, se = float(L[i] @ beta), float(math.sqrt(C[i, i]))
        truth = case["delta"][key]
        error = estimate-truth
        ci = [estimate-critical*se, estimate+critical*se]
        refs[key] = dict(L=L[i].tolist(), truth=truth, estimate=estimate, error=error, SE=se,
                         reference="known_covariance_normal", critical95=critical, CI=ci,
                         halfWidth=critical*se, standardizedError=error/se,
                         covered=ci[0] <= truth <= ci[1], lowerMiss=truth < ci[0], upperMiss=truth > ci[1])
    return dict(method="known_DGP_Omega_GLS_oracle_v1", coefficientNames=list(NAMES), beta=beta.tolist(),
                covariance=covariance.tolist(), contrastCovariance=C.tolist(), references=refs,
                knownParameters=case["parameters"], rowCount=4620, participantCount=462,
                interpretation="truth_informed_reference_not_human_candidate")
