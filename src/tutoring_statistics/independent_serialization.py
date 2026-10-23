"""independent serialization: accepted numerical definitions, portable source projection."""
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



EPOCH = "finite_numeric_json_v4"

METHODS = ("baseline_replay", "strict_nlopt_restart", "alternate_PORT")

ROLE = "independent_null_calibration"

NAMESPACE = "r10_refined_independent_null_v1_20261019"

METHOD = "strict_restart_port_check_v1"

class RepairIntegrityError(ValueError):
    """A source/representation contradiction, never a scientific nonrejection."""

def check(condition, message):
    if not condition:
        raise RepairIntegrityError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def read(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            RepairIntegrityError("Non-JSON numeric value: " + value)))

def same(actual, expected, *, tolerance=0., trail="root"):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), "Different keys: " + trail)
        for key in expected:
            same(actual[key], expected[key], tolerance=tolerance, trail=trail + "." + key)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), "Different list: " + trail)
        for index, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, tolerance=tolerance, trail=trail + f"[{index}]")
    elif type(actual) in (int, float) and type(expected) in (int, float):
        check(math.isfinite(actual) and math.isfinite(expected)
              and math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance),
              "Different numeric value: " + trail)
    else:
        check(type(actual) is type(expected) and actual == expected, "Different value/type: " + trail)

def leaf_differences(original, canonical, path=()):
    """Exact JSON-semantic differences, including changed containers and keys."""
    if isinstance(original, dict) and isinstance(canonical, dict):
        check(set(original) == set(canonical), "Repair changed object keys")
        return [item for key in original for item in leaf_differences(original[key], canonical[key], path + (key,))]
    if isinstance(original, list) and isinstance(canonical, list):
        check(len(original) == len(canonical), "Repair changed array length")
        return [item for index, (a, b) in enumerate(zip(original, canonical))
                for item in leaf_differences(a, b, path + (index,))]
    if type(original) is type(canonical) and original == canonical:
        return []
    if type(original) in (int, float) and type(canonical) in (int, float) and original == canonical:
        return []
    return [dict(path=list(path), before=original, after=canonical)]

def verify_finite_delta(original, canonical, methods):
    expected = [dict(path=["methods", method, "extra", "theta", "lower", 1], before="-Inf", after=None)
                for method in METHODS if method in methods]
    actual = leaf_differences(original, canonical)
    same(sorted(actual, key=lambda row: str(row["path"])), sorted(expected, key=lambda row: str(row["path"])))
    return expected

def validate_rds_science(route, fact, actual_request_path, row_count):
    """Check the old scientific fields against the fit, without replacing any."""
    check(route["status"] == "fitted" and fact["methodID"] == route["methodID"], "RDS route identity")
    theta = route["extra"]["theta"]
    same(theta["lower"], [0, "-Inf", 0]); same(theta["lowerKinds"], ["finite", "-Inf", "finite"])
    same(fact["theta"]["lower"], [0, None, 0])
    same(fact["theta"]["lowerKinds"], ["finite", "-Inf", "finite"])
    same(fact["theta"]["lowerFinite"], [True, False, True])
    same(fact["theta"]["lowerNegativeInfinite"], [False, True, False])
    for key in ("names", "values"):
        same(theta[key], fact["theta"][key], tolerance=1e-10, trail="theta." + key)
    raw = route["raw"]; extracted = fact["raw"]
    for key in ("formula", "REML", "optimizer", "coefficientNames", "coefficients", "vcov",
                "randomCovariance", "residualVariance", "references"):
        same(raw[key], extracted[key], tolerance=1e-9, trail="RDS.raw." + key)
    for key, value in extracted["diagnostics"].items():
        same(raw["diagnostics"][key], value, tolerance=1e-7, trail="RDS.guard." + key)
    same(route["extra"]["optimizerCode"], fact["optimizerCode"])
    same(route["extra"]["controls"], fact["controls"])
    same(route["extra"]["finiteDifference"]["Hessian"], fact["Hessian"], tolerance=1e-9)
    for key in ("remlCriterion", "logLik"):
        same(route["extra"]["objective"][key], fact[key], tolerance=1e-9, trail="RDS." + key)
    identity = fact["inputCheck"]
    check(identity["passed"] is True and identity["participantOrderEqual"] is True
          and identity["actualRequestPath"] == str(actual_request_path) and identity["rowCount"] == row_count,
          "RDS/actual-request input identity")
    same([row["field"] for row in identity["numericColumns"]], ["y", "delta", "trial_c"])
    check(all(type(row["maxAbsoluteError"]) in (int, float) and math.isfinite(row["maxAbsoluteError"])
              and 0 <= row["maxAbsoluteError"] <= 1e-9 for row in identity["numericColumns"]),
          "RDS numeric input disagreement")

def canonical_view(original, rds_facts, actual_request_path):
    """Only RDS-proven numeric lower positions become canonical JSON null."""
    canonical = deepcopy(original)
    changed = []
    for method in METHODS:
        route = original["methods"].get(method)
        if route is None or route["status"] != "fitted":
            continue
        check(method in rds_facts, "Missing actual RDS extraction: " + method)
        validate_rds_science(route, rds_facts[method], actual_request_path, len(original["request"]["rows"]))
        canonical["methods"][method]["extra"]["theta"]["lower"][1] = None
        changed.append(method)
    return canonical, verify_finite_delta(original, canonical, changed)
