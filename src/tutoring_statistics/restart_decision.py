"""restart decision: accepted numerical definitions, portable source projection."""
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



METHOD = "strict_restart_port_check_v1"

ROUTES = ("baseline_replay", "strict_nlopt_restart", "alternate_PORT")

SCREENS = ("commonObjective", "fixedCoefficients", "deltaSE", "deltaDF", "theta", "residualVariance")

CONTROLS = (
    {"xtol_abs": 1e-8, "ftol_abs": 1e-8, "maxeval": 10000},
    {"xtol_abs": 1e-10, "ftol_abs": 1e-10, "maxeval": 100000},
    {"eval.max": 20000, "iter.max": 10000, "rel.tol": 1e-10, "x.tol": 1e-10},
)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def values(x):
    return x if isinstance(x, list) else ([] if x is None else [x])

def at(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj

def difference(a, b):
    return b-a if finite(a) and finite(b) else None

def relative(a, b, anchor):
    d = difference(a, b)
    return abs(d)/anchor if d is not None and finite(anchor) and anchor > 0 else None

def matrix(value, size):
    if not isinstance(value, list) or len(value) != size:
        return None
    if not all(isinstance(row, list) and len(row) == size and all(finite(v) for v in row) for row in value):
        return None
    return np.asarray(value, dtype=float)

def baseline_scales(value, names):
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(row, list) and len(row) == 2 for row in value):
        return {name: None for name in names}
    return {name: math.sqrt(value[i][i]) if finite(value[i][i]) and value[i][i] > 0 else None
            for i, name in enumerate(names)}

def theta_valid(route):
    theta = at(route, "extra", "theta") or {}
    vals, names = theta.get("values"), theta.get("names")
    return (isinstance(vals, list) and len(vals) == 3 and all(finite(v) for v in vals)
            and vals[0] >= 0 and vals[2] >= 0 and isinstance(names, list)
            and len(names) == 3 and len(set(names)) == 3 and all(isinstance(n, str) for n in names)
            and theta.get("lower") == [0, None, 0]
            and theta.get("lowerKinds") == ["finite", "-Inf", "finite"])

def route_check(route, method, tail):
    status = at(route, "status") or "missing"
    result = dict(status=status, guardUsable=False, validReference=False, diagnosticP=None,
                  richardsonStatus=at(route, "richardson", "status") or "missing")
    if route is None:
        return result
    require(route.get("methodID") == method, "Route identity mismatch")
    require(status in ("fitted", "error", "dependency_unavailable"), "Invalid route status")
    if status != "fitted":
        require(bool(route.get("error")), "Unreported route failure")
        return result
    raw, extra = route.get("raw") or {}, route.get("extra") or {}
    require(raw.get("status") == "fitted" and raw.get("REML") is True
            and re.sub(r"\s+", "", raw.get("formula", "")) == "y~delta+(1+trial_c|participant)",
            "Changed candidate model")
    require(raw.get("optimizer") == ("nlminbwrap" if method == ROUTES[2] else "nloptwrap"),
            "Changed route optimizer")
    controls = extra.get("controls") or {}
    expected_controls = CONTROLS[ROUTES.index(method)]
    # lme4 stores nloptwrap's observational print_level=0 in optinfo. It is
    # present in accepted artifacts; no other extra optimizer knob is allowed.
    extras = set(controls)-set(expected_controls)
    allowed_extras = not extras or (method != ROUTES[2] and extras == {"print_level"} and controls["print_level"] == 0)
    require(allowed_extras and all(controls.get(key) == expected for key, expected in expected_controls.items()),
            "Changed route controls")
    names, beta = raw.get("coefficientNames"), raw.get("coefficients") or {}
    require(isinstance(names, list) and len(names) == 2 and set(names) == {"(Intercept)", "delta"}
            and set(beta) == set(names), "Changed coefficient names")
    covariance = matrix(raw.get("vcov"), 2)
    diag = raw.get("diagnostics") or {}
    codes, eig, grad = values(extra.get("optimizerCode")), values(diag.get("hessianEigenvalues")), values(diag.get("gradient"))
    # This is the original guard, with an observed numeric optimizer status.
    # Richardson and ordinary warning text deliberately do not enter this gate.
    converged = bool(codes) and all(finite(c) and c == 0 for c in codes) and not values(diag.get("convergenceMessages"))
    singular = not theta_valid(route) or any(extra["theta"]["values"][i] < 1e-4 for i in (0, 2))
    usable = (converged and not singular and bool(eig) and all(finite(x) and x > 0 for x in eig)
              and bool(grad) and all(finite(x) for x in grad) and all(finite(x) for x in beta.values())
              and covariance is not None and diag.get("usableCandidateInference") is True)
    ref = at(raw, "references", "delta") or {}
    valid = all(finite(ref.get(k)) for k in ("estimate", "SE", "df", "statistic"))
    valid = valid and ref["SE"] > 0 and ref["df"] > 0
    if valid:
        j = names.index("delta")
        valid = (covariance is not None and covariance[j, j] > 0 and finite(beta.get("delta"))
                 and math.isclose(ref["estimate"], beta["delta"], rel_tol=1e-8, abs_tol=1e-8)
                 and math.isclose(ref["SE"], math.sqrt(covariance[j, j]), rel_tol=1e-8, abs_tol=1e-8)
                 and math.isclose(ref["statistic"], ref["estimate"]/ref["SE"], rel_tol=1e-8, abs_tol=1e-8))
    if valid:
        ci = ref.get("CI")
        critical = float(stats.t.ppf(.975, ref["df"]))
        expected = [ref["estimate"]-critical*ref["SE"], ref["estimate"]+critical*ref["SE"]]
        valid = (isinstance(ci, list) and len(ci) == 2 and all(finite(x) for x in ci)
                 and all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7) for a, b in zip(ci, expected)))
    if valid:
        from .mixed_tests import directional_result
        directed = directional_result(ref, tail, usable=True)
        valid = directed["status"] == "available_candidate"
        result["diagnosticP"] = directed.get("p")
    result.update(guardUsable=bool(usable), validReference=bool(valid))
    return result

def compare(first, second, baseline, first_check, second_check):
    out = dict(objectiveDifference=None, fixedCoefficientDifferences=None, baselineFixedSE=None,
               deltaSERelativeDifference=None, deltaDFRelativeDifference=None,
               thetaRelativeMaxDifference=None, residualVarianceRelativeDifference=None,
               fixedVcovDifference=None, randomCovarianceDifference=None,
               screens={key: None for key in SCREENS}, allDiagnosticScreens=None,
               diagnosticPThresholdCrossing=None)
    if any(at(r, "status") != "fitted" for r in (first, second, baseline)):
        return dict(out, status="unknown_route_failure")
    a, b, base = first["raw"], second["raw"], baseline["raw"]
    names = base["coefficientNames"]
    scales = baseline_scales(base.get("vcov"), names)
    fixed = {name: difference(a["coefficients"].get(name), b["coefficients"].get(name)) for name in names}
    objective = difference(at(first, "extra", "objective", "commonAtTheta"), at(second, "extra", "objective", "commonAtTheta"))
    se = relative(at(a, "references", "delta", "SE"), at(b, "references", "delta", "SE"), at(base, "references", "delta", "SE"))
    df = relative(at(a, "references", "delta", "df"), at(b, "references", "delta", "df"), at(base, "references", "delta", "df"))
    residual = relative(a.get("residualVariance"), b.get("residualVariance"), base.get("residualVariance"))
    theta = None
    if all(theta_valid(r) for r in (first, second, baseline)):
        av, bv, anchors = (r["extra"]["theta"]["values"] for r in (first, second, baseline))
        theta = max(abs(y-x) for x, y in zip(av, bv))/max(1., max(abs(x) for x in anchors))
    screens = dict(commonObjective=abs(objective) <= 1e-4 if objective is not None else None,
                   fixedCoefficients=all(abs(fixed[n]) <= .01*scales[n] for n in names) if all(v is not None for v in (*fixed.values(), *scales.values())) else None,
                   deltaSE=se <= .001 if se is not None else None,
                   deltaDF=df <= .01 if df is not None else None,
                   theta=theta <= .001 if theta is not None else None,
                   residualVariance=residual <= .001 if residual is not None else None)
    out.update(status="compared", objectiveDifference=objective, fixedCoefficientDifferences=fixed,
               baselineFixedSE=scales, deltaSERelativeDifference=se, deltaDFRelativeDifference=df,
               thetaRelativeMaxDifference=theta, residualVarianceRelativeDifference=residual,
               screens=screens, allDiagnosticScreens=all(screens.values()) if all(v is not None for v in screens.values()) else None)
    for key, field in (("fixedVcovDifference", "vcov"), ("randomCovarianceDifference", "randomCovariance")):
        ma, mb = matrix(a.get(field), 2), matrix(b.get(field), 2)
        out[key] = (mb-ma).tolist() if ma is not None and mb is not None else None
    ps = [first_check["diagnosticP"], second_check["diagnosticP"]]
    if all(finite(p) for p in ps):
        out["diagnosticPThresholdCrossing"] = (ps[0] < .05) != (ps[1] < .05)
    return out

def resolve(record):
    """Resolve one bound diagnostic record without selecting on p or warnings."""
    hypothesis = at(record, "oldCase", "H")
    require(hypothesis in ("H3", "H4", "H5"), "Unsupported hypothesis")
    require(at(record, "request", "model") == hypothesis, "Request/case hypothesis mismatch")
    tail = "less" if hypothesis == "H5" else "greater"
    methods = record.get("methods") or {}
    require(set(methods) <= set(ROUTES), "Unexpected route")
    routes = [methods.get(method) for method in ROUTES]
    checks = {method: route_check(route, method, tail) for method, route in zip(ROUTES, routes)}
    fitted = [r for r in routes if at(r, "status") == "fitted"]
    if fitted:
        for route in fitted[1:]:
            require(at(route, "raw", "coefficientNames") == at(fitted[0], "raw", "coefficientNames"), "Coefficient order mismatch")
            if at(route, "extra", "theta", "names") is not None and at(fitted[0], "extra", "theta", "names") is not None:
                require(at(route, "extra", "theta", "names") == at(fitted[0], "extra", "theta", "names"), "Theta order mismatch")
    base, strict, port = routes
    pairs = (("baseline_strict", 0, 1), ("baseline_PORT", 0, 2), ("strict_PORT", 1, 2))
    comparisons = {name: compare(routes[i], routes[j], base, checks[ROUTES[i]], checks[ROUTES[j]]) for name, i, j in pairs}
    reasons = []
    anchor = (at(base, "status") == "fitted" and theta_valid(base)
              and finite(at(base, "extra", "objective", "commonAtTheta")))
    anchor = anchor and all(finite(x) and x > 0 for x in (
        at(base, "raw", "references", "delta", "SE"), at(base, "raw", "references", "delta", "df"),
        at(base, "raw", "residualVariance")))
    scales = baseline_scales(at(base, "raw", "vcov"), ["(Intercept)", "delta"])
    anchor = anchor and all(v is not None for v in scales.values())
    if not anchor:
        reasons.append("baseline_anchor_invalid_or_missing")
    for method in ROUTES[1:]:
        check = checks[method]
        if check["status"] != "fitted":
            reasons.append(method+":"+check["status"])
        else:
            if not check["guardUsable"]:
                reasons.append(method+":original_guard_failed")
            if not check["validReference"]:
                reasons.append(method+":invalid_reference")
    for key, passed in comparisons["strict_PORT"]["screens"].items():
        if passed is not True:
            reasons.append("strict_PORT:"+key+(":unknown" if passed is None else ":failed"))
    nonworsening = {}
    for label, route in (("strict", strict), ("PORT", port)):
        diff = (difference(at(base, "extra", "objective", "commonAtTheta"), at(route, "extra", "objective", "commonAtTheta"))
                if at(base, "status") == at(route, "status") == "fitted" else None)
        passed = diff <= 1e-4 if diff is not None else None
        nonworsening[label] = dict(objectiveDifference=diff, passes=passed)
        if passed is not True:
            reasons.append(label+":objective_nonworsening"+(":unknown" if passed is None else ":failed"))
    inference = None
    if not reasons:
        from .mixed_tests import directional_result
        inference = directional_result(strict["raw"]["references"]["delta"], tail, usable=True)
        require(inference["status"] == "available_candidate", "Valid strict reference unexpectedly unavailable")
    return dict(status="unavailable" if reasons else "available_candidate", reasons=reasons,
                officialMethod=ROUTES[1], hypothesis=hypothesis, tail=tail, routeChecks=checks,
                comparisons=comparisons, nonworsening=nonworsening, inference=inference,
                p=inference["p"] if inference else None,
                candidateNominalReject=inference["candidateNominalReject"] if inference else False)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda s: (_ for _ in ()).throw(ValueError("Nonstandard JSON number: "+s)))

def beneath(root, relative):
    root, path = Path(root).resolve(), (Path(root)/relative).resolve()
    require(path.is_relative_to(root) and path != root, "Path escapes bound root")
    return path

def write_new(path, payload):
    path = Path(path)
    if path.exists():
        require(load(path) == payload, "Existing output differs: "+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".refined-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)

def observed_file(path, expected, observed):
    path = Path(path).resolve()
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid expected digest")
    digest = sha(path)
    require(digest == expected, "Changed bound file: "+str(path))
    require(str(path) not in observed or observed[str(path)] == digest, "Contradictory bound file")
    observed[str(path)] = digest
    return path
