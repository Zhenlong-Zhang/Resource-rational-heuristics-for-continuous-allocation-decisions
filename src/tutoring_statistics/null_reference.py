"""null reference: accepted numerical definitions, portable source projection."""
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

from . import restart_decision as frozen
from . import restart_summary as summaries
from .mixed_tests import directional_result
require, at = frozen.require, frozen.at

ROLE = "independent_null_calibration"

NAMESPACE = "r10_refined_independent_null_v1_20261019"

METHOD = frozen.METHOD

INDICES = (1, 4, 5, 7, 9, 12, 13, 15, 17, 20, 21, 23)

def make_plan(config):
    require(config["seedBase"] == 20261019 and config["seedStride"] == 10000
            and [c["caseIndex"] for c in config["cases"]] == list(range(1, 25)), "Changed original plan")
    cases = [c for c in config["cases"] if c["caseIndex"] in INDICES]
    require(tuple(c["caseIndex"] for c in cases) == INDICES and len({c["caseID"] for c in cases}) == 12
            and all(c["beta"] == 0 and c["nReplicates"] == 2000 for c in cases), "Changed null population")
    plan = dict(config, seedBase=120261019, cases=cases)
    new = [plan["seedBase"]+c["caseIndex"]*10000+i for c in cases for i in range(1, 2001)]
    old = {config["seedBase"]+c["caseIndex"]*10000+i for c in config["cases"] for i in range(1, c["nReplicates"]+1)}
    require(len(new) == len(set(new)) == 24000 and not old.intersection(new), "Fresh seed collision")
    return plan

def population(plan):
    result = []
    for case in plan["cases"]:
        for start in range(1, 2001, 100):
            index = len(result)+1; ids = list(range(start, start+100))
            result.append(dict(shardIndex=index, shardID=f"shard_{index:04d}", caseID=case["caseID"],
                caseIndex=case["caseIndex"], repIDs=ids,
                seeds=[plan["seedBase"]+case["caseIndex"]*plan["seedStride"]+rep for rep in ids]))
    return result

def technical_completion(record):
    reasons, failures = [], []
    methods = record.get("methods") or {}
    for method in frozen.ROUTES:
        route = methods.get(method)
        if not isinstance(route, dict):
            reasons.append("missing_route:"+method); continue
        if route.get("status") != "fitted":
            known = route.get("methodFailure") is True and route.get("errorPhase") == "fit"
            if method == frozen.ROUTES[0]:
                known = known and at(route, "raw", "status") == "unavailable" and at(route, "raw", "reason") == "backend_fit_error"
            if method == frozen.ROUTES[1] and route.get("status") == "dependency_unavailable":
                base = methods.get(frozen.ROUTES[0]) or {}
                known = (route.get("methodFailure") is True and route.get("errorPhase") == "dependency"
                         and at(base, "raw", "reason") == "backend_fit_error" and base.get("methodFailure") is True)
            (failures if known else reasons).append(method if known else "technical_route_error:"+method+":"+str(route.get("errorPhase")))
            continue
        require(route.get("methodFailure") is False and route.get("errorPhase") is None, "Contradictory fitted phase")
        if not isinstance(route.get("fitArtifact"), dict): reasons.append("missing_fit_artifact:"+method)
        required = ("raw.status", "raw.formula", "raw.REML", "raw.optimizer", "raw.coefficientNames", "raw.coefficients",
            "raw.vcov", "raw.randomCovariance", "raw.residualVariance", "raw.diagnostics", "raw.references.delta",
            "extra.theta", "extra.objective.commonAtTheta", "extra.controls", "extra.optimizerCode",
            "extra.finiteDifference", "extra.optinfo", "extra.matrixX", "extra.matrixZ",
            "raw.diagnostics.converged", "raw.diagnostics.singular", "raw.diagnostics.gradient",
            "raw.diagnostics.hessianEigenvalues", "raw.diagnostics.convergenceMessages", "raw.diagnostics.usableCandidateInference",
            "raw.references.delta.estimate", "raw.references.delta.SE", "raw.references.delta.df",
            "raw.references.delta.statistic", "raw.references.delta.CI", "extra.theta.names", "extra.theta.values",
            "extra.theta.lower", "extra.theta.lowerKinds")
        containers = {"raw.diagnostics", "raw.references.delta", "extra.theta", "extra.finiteDifference",
                      "extra.optinfo", "extra.matrixX", "extra.matrixZ"}
        for path in required:
            current = route; found = True
            for part in path.split("."):
                if not isinstance(current, dict) or part not in current:
                    found = False; break
                current = current[part]
            if not found or (path in containers and not isinstance(current, dict)):
                reasons.append("missing_core:"+method+":"+path)
    return dict(complete=not reasons, reasons=reasons, identifiedMethodFailures=failures)

def resolve_fresh(diagnostic, technical):
    projection = dict(oldCase=diagnostic["case"], request=diagnostic["request"], methods=diagnostic["methods"])
    # These are source/model contradictions, not technical missing-core exits.
    for method, route in diagnostic["methods"].items():
        if route.get("status") != "fitted": continue
        raw, extra = route.get("raw") or {}, route.get("extra") or {}
        if "status" in raw: require(raw["status"] == "fitted", "Contradictory fitted backend status")
        if "formula" in raw:
            require(isinstance(raw["formula"], str) and "".join(raw["formula"].split()) == "y~delta+(1+trial_c|participant)", "Changed candidate model")
        if "REML" in raw: require(raw["REML"] is True, "Changed candidate REML")
        if "optimizer" in raw:
            require(raw["optimizer"] == ("nlminbwrap" if method == frozen.ROUTES[2] else "nloptwrap"), "Changed route optimizer")
        if "coefficientNames" in raw:
            require(isinstance(raw["coefficientNames"], list) and len(raw["coefficientNames"]) == 2
                    and set(raw["coefficientNames"]) == {"(Intercept)", "delta"}, "Changed coefficient names")
        if "coefficients" in raw:
            require(isinstance(raw["coefficients"], dict) and set(raw["coefficients"]) == {"(Intercept)", "delta"}, "Changed coefficient names")
        if "controls" in extra:
            controls = extra["controls"] or {}; expected = frozen.CONTROLS[frozen.ROUTES.index(method)]
            extras = set(controls)-set(expected)
            require((not extras or (method != frozen.ROUTES[2] and extras == {"print_level"} and controls["print_level"] == 0))
                    and all(controls.get(k) == v for k, v in expected.items()), "Changed route controls")
    try:
        return frozen.resolve(projection), None
    except (KeyError, TypeError, ValueError) as error:
        if technical["complete"]: raise
        technical["complete"] = False
        technical["reasons"].append("candidate_resolution_unresolved")
        return None, repr(error)

def compact(record, diagnostic):
    directions, warnings = {}, {}
    for method in frozen.ROUTES:
        value = at(diagnostic, "methods", method, "raw", "coefficients", "delta")
        directions[method] = (1 if value > 0 else -1 if value < 0 else 0) if frozen.finite(value) else None
        warnings[method] = len(frozen.values(at(diagnostic, "methods", method, "raw", "diagnostics", "warnings")))
    return dict(caseID=record["caseID"], repID=record["repID"], baselineDecision=record["baselineDecision"],
        baselineInference=record["baselineInference"], truth=summaries.truth(record["case"]), decision=record["decision"],
        technicalCompletion=record["technicalCompletion"], routeDirections=directions, ordinaryWarningCounts=warnings)

def case_summary(case, ids, records):
    # Old names exist only inside this numerical helper's argument convention.
    transformed, roster = {}, []
    for rep in ids:
        row = records.get(rep); complete = row is not None and row["technicalCompletion"]["complete"] is True
        baseline = row["baselineDecision"] if complete else dict(status="technical_unresolved", reject=False)
        inference = row["baselineInference"] if complete else None
        roster.append(dict(repID=rep, oldDecision=baseline, oldInference=inference))
        if row is not None:
            decision = row["decision"] or dict(reasons=[], routeChecks={}, comparisons={}, nonworsening={})
            transformed[rep] = dict(row, oldDecision=baseline, oldInference=inference, decision=decision)
    report = summaries.case_summary(dict(case=case, records=roster), transformed)
    base = report.pop("original"); t, m = report["technicalCompleted"], report["planned"]
    base["unavailable"] = t-base["available"]
    base["rates"].update(availabilityCompleted=summaries.rate(base["available"], t),
                         rejectionCompleted=summaries.rate(base["rejected"], t))
    if t < m:
        for key in ("availabilityAll", "rejectionAll", "coverageAndAvailableAll"): base["rates"][key] = None
    report.update(baseline=base, baselineObservationScope="same_technically_complete_pairs")
    return report
