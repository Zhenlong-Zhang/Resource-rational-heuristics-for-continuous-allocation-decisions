"""independent restart: accepted numerical definitions, portable source projection."""
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



METHODS = ("baseline_replay","strict_nlopt_restart","alternate_PORT")

METHOD_ID = "strict_restart_port_check_v1"

ROLE = "development_regression"

class IntegrityError(ValueError):
    """A source/protocol contradiction, never ordinary scientific unavailability."""

def check(condition, message):
    if not condition:
        raise IntegrityError(message)

def number(x):
    return type(x) in (int,float) and math.isfinite(x)

def vector(x):
    return [] if x is None else (x if isinstance(x,list) else [x])

def same(a,b, *, tolerance=1e-10, trail="root"):
    if isinstance(b,dict):
        check(isinstance(a,dict) and set(a)==set(b),"Keys differ: "+trail)
        for key in b:
            same(a[key],b[key],tolerance=tolerance,trail=trail+"."+key)
    elif isinstance(b,list):
        check(isinstance(a,list) and len(a)==len(b),"List differs: "+trail)
        for i,(x,y) in enumerate(zip(a,b)):
            same(x,y,tolerance=tolerance,trail=f"{trail}[{i}]")
    elif number(b):
        check(number(a) and math.isclose(a,b,rel_tol=tolerance,abs_tol=tolerance),"Number differs: "+trail)
    else:
        check(type(a) is type(b) and a==b,"Value differs: "+trail)

def admit_history(record):
    check(record["schema"]=="mixed-convergence-diagnostic-record-v1","Diagnostic schema")
    check(record["scientificAdoption"] is False and record["historicalInferenceChanged"] is False,"Historical scope flags")
    parity=record.get("baselineParity") or {}
    check(parity.get("passed") is True
          and parity.get("nontrivialFiniteReference") is True,
          "Historical baseline parity is unresolved")
    check(set(record["methods"])<=set(METHODS),"Unexpected route")
    check(record["request"]["model"] in ("H3","H4","H5"),"Unexpected hypothesis")
    check(record["request"]["model"]==record["oldCase"]["H"],"Request/hypothesis mismatch")

def route_facts(route,method,tail):
    if route is None:
        return dict(status="missing",guardUsable=False,validReference=False,diagnosticP=None,richardsonStatus="missing")
    check(route["methodID"]==method,"Route method identity")
    rich=route.get("richardson")
    rich_status=(rich.get("status") or "missing") if isinstance(rich,dict) else "missing"
    if route["status"]!="fitted":
        check(route["status"] in ("error","dependency_unavailable") and route.get("error"),"Undocumented route failure")
        return dict(status=route["status"],guardUsable=False,validReference=False,diagnosticP=None,richardsonStatus=rich_status)
    raw,extra=route["raw"],route["extra"]
    check(raw["status"]=="fitted" and raw["REML"] is True
          and re.sub(r"\s+","",raw["formula"])=="y~delta+(1+trial_c|participant)","Changed formula/REML")
    check(raw["optimizer"]==("nlminbwrap" if method=="alternate_PORT" else "nloptwrap"),"Changed optimizer")
    names=raw["coefficientNames"]
    check(len(names)==2 and set(names)=={"(Intercept)","delta"} and set(raw["coefficients"])==set(names),"Coefficient names")
    v=np.asarray(raw["vcov"],dtype=float)
    check(v.shape==(2,2),"Coefficient covariance dimensions")
    theta=extra.get("theta") or {}
    theta_ok=(isinstance(theta.get("values"),list) and len(theta["values"])==3
              and all(number(x) for x in theta["values"]) and theta["values"][0]>=0 and theta["values"][2]>=0
              and isinstance(theta.get("names"),list) and len(theta["names"])==3
              and len(set(theta["names"]))==3 and all(isinstance(n,str) for n in theta["names"])
              and theta.get("lower")==[0,None,0] and theta.get("lowerKinds")==["finite","-Inf","finite"])
    diag=raw["diagnostics"]
    codes=vector(extra["optimizerCode"])
    check(all(number(c) for c in codes),"Nonnumeric optimizer code")
    converged=bool(codes) and all(c==0 for c in codes) and not vector(diag["convergenceMessages"])
    singular=any(theta["values"][i]<1e-4 for i in (0,2)) if theta_ok else None
    g=vector(diag["gradient"]); eig=vector(diag["hessianEigenvalues"])
    hessian_ok=bool(eig) and all(number(e) for e in eig) and min(eig)>0
    h=extra.get("finiteDifference",{}).get("Hessian")
    if h is not None:
        matrix=np.asarray(h,dtype=float)
        if matrix.shape==(3,3) and np.isfinite(matrix).all():
            check(np.allclose(matrix,matrix.T,rtol=1e-10,atol=1e-10),"Asymmetric core Hessian")
            same(sorted(eig),np.linalg.eigvalsh(matrix).tolist(),tolerance=1e-7)
    original_converged=all(c==0 for c in codes) and not vector(diag["convergenceMessages"])
    usable=bool(original_converged and singular is False and hessian_ok and g and all(number(z) for z in g)
                and all(number(z) for z in raw["coefficients"].values()) and np.isfinite(v).all())
    same(diag["converged"],bool(original_converged))
    if theta_ok:
        same(diag["singular"],singular)
        same(diag["usableCandidateInference"],usable)
    reference=raw["references"]["delta"]
    valid=all(number(reference.get(k)) for k in ("estimate","SE","df","statistic"))
    valid=bool(valid and reference["SE"]>0 and reference["df"]>0
               and math.isclose(reference["estimate"]/reference["SE"],reference["statistic"],rel_tol=1e-8,abs_tol=1e-8))
    if valid:
        j=names.index("delta")
        valid=bool(np.isfinite(v).all() and number(v[j,j].item()) and v[j,j]>0
                   and math.isclose(reference["estimate"],raw["coefficients"]["delta"],rel_tol=1e-8,abs_tol=1e-8)
                   and math.isclose(reference["SE"],math.sqrt(v[j,j]),rel_tol=1e-8,abs_tol=1e-8))
    ci=reference.get("CI")
    valid=bool(valid and isinstance(ci,list) and len(ci)==2 and all(number(x) for x in ci) and ci[0]<=ci[1])
    p=None
    if valid:
        critical=float(stats.t.ppf(.975,reference["df"]))
        expected=[reference["estimate"]-critical*reference["SE"],reference["estimate"]+critical*reference["SE"]]
        valid=all(math.isclose(x,y,rel_tol=1e-7,abs_tol=1e-7) for x,y in zip(ci,expected))
        if valid:
            p=float(stats.t.sf(reference["statistic"],reference["df"]) if tail=="greater" else stats.t.cdf(reference["statistic"],reference["df"]))
    controls=(dict(xtol_abs=1e-8,ftol_abs=1e-8,maxeval=10000) if method==METHODS[0] else
              dict(xtol_abs=1e-10,ftol_abs=1e-10,maxeval=100000) if method==METHODS[1] else
              {"eval.max":20000,"iter.max":10000,"rel.tol":1e-10,"x.tol":1e-10})
    for key,value in controls.items():
        same(extra["controls"][key],value,tolerance=0)
    allowed=set(controls)|({"print_level"} if method!=METHODS[2] else set())
    check(set(extra["controls"])<=allowed,"Unexpected optimizer controls")
    if "print_level" in extra["controls"]:
        same(extra["controls"]["print_level"],0,tolerance=0)
    return dict(status="fitted",guardUsable=bool(usable and converged),validReference=valid,
                diagnosticP=p,richardsonStatus=rich_status)

def numeric_comparison(first,second,baseline):
    screen_names=("commonObjective","fixedCoefficients","deltaSE","deltaDF","theta","residualVariance")
    if any(r is None or r["status"]!="fitted" for r in (first,second,baseline)):
        return dict(status="unknown_route_failure",objectiveDifference=None,fixedCoefficientDifferences=None,baselineFixedSE=None,
                    deltaSERelativeDifference=None,deltaDFRelativeDifference=None,thetaRelativeMaxDifference=None,
                    residualVarianceRelativeDifference=None,fixedVcovDifference=None,randomCovarianceDifference=None,
                    screens={name:None for name in screen_names},allDiagnosticScreens=None)
    a,b,base=(r["raw"] for r in (first,second,baseline))
    ae,be,bas=(r["extra"] for r in (first,second,baseline))
    same(a["coefficientNames"],base["coefficientNames"],tolerance=0)
    same(b["coefficientNames"],base["coefficientNames"],tolerance=0)
    theta_dicts=[e.get("theta") or {} for e in (ae,be,bas)]
    for t in theta_dicts[:2]:
        if t.get("names") is not None and theta_dicts[2].get("names") is not None:
            same(t["names"],theta_dicts[2]["names"],tolerance=0)
    difference=lambda x,y:y-x if number(x) and number(y) else None
    names=base["coefficientNames"]
    covariance=np.asarray(base["vcov"],dtype=float)
    scales={name:math.sqrt(covariance[i,i]) if np.isfinite(covariance[i,i]) and covariance[i,i]>0 else None for i,name in enumerate(names)}
    coefficients={name:difference(a["coefficients"][name],b["coefficients"][name]) for name in names}
    objective=difference(ae["objective"]["commonAtTheta"],be["objective"]["commonAtTheta"])
    def relative(x,y,denominator):
        d=difference(x,y)
        return abs(d)/denominator if d is not None and number(denominator) and denominator>0 else None
    se=relative(a["references"]["delta"]["SE"],b["references"]["delta"]["SE"],base["references"]["delta"]["SE"])
    df=relative(a["references"]["delta"]["df"],b["references"]["delta"]["df"],base["references"]["delta"]["df"])
    t_vectors=[t.get("values") for t in theta_dicts]
    theta=(max(abs(y-x) for x,y in zip(t_vectors[0],t_vectors[1]))/max(1,max(abs(x) for x in t_vectors[2]))) if all(isinstance(v,list) and len(v)==3 and all(number(x) for x in v) and v[0]>=0 and v[2]>=0 for v in t_vectors) and all(t.get("lower")==[0,None,0] and t.get("lowerKinds")==["finite","-Inf","finite"] and isinstance(t.get("names"),list) and len(t["names"])==3 and len(set(t["names"]))==3 for t in theta_dicts) else None
    residual=relative(a["residualVariance"],b["residualVariance"],base["residualVariance"])
    screens=dict(commonObjective=abs(objective)<=1e-4 if objective is not None else None,
        fixedCoefficients=all(abs(coefficients[k])<=.01*scales[k] for k in names) if all(number(coefficients[k]) and number(scales[k]) for k in names) else None,
        deltaSE=se<=.001 if se is not None else None,deltaDF=df<=.01 if df is not None else None,
        theta=theta<=.001 if theta is not None else None,residualVariance=residual<=.001 if residual is not None else None)
    def matrix_difference(key):
        x,y=np.asarray(a[key],dtype=float),np.asarray(b[key],dtype=float)
        check(x.shape==y.shape,"Covariance shapes differ")
        return (y-x).tolist() if np.isfinite(x).all() and np.isfinite(y).all() else None
    return dict(status="compared",objectiveDifference=objective,fixedCoefficientDifferences=coefficients,baselineFixedSE=scales,
        deltaSERelativeDifference=se,deltaDFRelativeDifference=df,thetaRelativeMaxDifference=theta,
        residualVarianceRelativeDifference=residual,fixedVcovDifference=matrix_difference("vcov"),
        randomCovarianceDifference=matrix_difference("randomCovariance"),screens=screens,
        allDiagnosticScreens=all(screens.values()) if all(x is not None for x in screens.values()) else None)

def derive(record):
    """New gates are reconstructed from raw routes, independently of production."""
    admit_history(record)
    hypothesis=record["request"]["model"]
    tail="less" if hypothesis=="H5" else "greater"
    routes=record["methods"]
    facts={m:route_facts(routes.get(m),m,tail) for m in METHODS}
    comparisons={}
    for key,a,b in (("baseline_strict",METHODS[0],METHODS[1]),("baseline_PORT",METHODS[0],METHODS[2]),("strict_PORT",METHODS[1],METHODS[2])):
        value=numeric_comparison(routes.get(a),routes.get(b),routes.get(METHODS[0]))
        pa,pb=facts[a]["diagnosticP"],facts[b]["diagnosticP"]
        value["diagnosticPThresholdCrossing"]=(pa<.05)!=(pb<.05) if value["status"]=="compared" and pa is not None and pb is not None else None
        comparisons[key]=value
    base=routes.get(METHODS[0])
    anchors=False
    if base is not None and base["status"]=="fitted":
        raw,extra=base["raw"],base["extra"]
        theta=extra.get("theta") or {}
        v=np.asarray(raw["vcov"],dtype=float)
        ref=raw["references"]["delta"]
        anchors=bool(v.shape==(2,2) and all(np.isfinite(v[i,i]) and v[i,i]>0 for i in range(2))
            and isinstance(theta.get("values"),list) and len(theta["values"])==3 and all(number(x) for x in theta["values"])
            and isinstance(theta.get("names"),list) and len(theta["names"])==3 and len(set(theta["names"]))==3
            and theta.get("lower")==[0,None,0] and theta.get("lowerKinds")==["finite","-Inf","finite"]
            and all(number(x) and x>0 for x in (ref["SE"],ref["df"],raw["residualVariance"]))
            and number(extra["objective"]["commonAtTheta"])
            and all(number(x) and (low is None or x>=low) for x,low in zip(theta["values"],theta["lower"])))
    nonworsening={}
    for key,method in (("strict",METHODS[1]),("PORT",METHODS[2])):
        d=None
        if base is not None and routes.get(method) is not None and base["status"]==routes[method]["status"]=="fitted":
            x=base["extra"]["objective"]["commonAtTheta"]
            y=routes[method]["extra"]["objective"]["commonAtTheta"]
            if number(x) and number(y):
                d=y-x
        nonworsening[key]=dict(objectiveDifference=d,passes=d<=1e-4 if d is not None else None)
    eligible=bool(anchors and all(facts[m]["guardUsable"] and facts[m]["validReference"] for m in METHODS[1:])
                  and comparisons["strict_PORT"]["allDiagnosticScreens"] is True
                  and all(v["passes"] is True for v in nonworsening.values()))
    inference=None
    if eligible:
        inference=dict(routes[METHODS[1]]["raw"]["references"]["delta"],tail=tail,diagnosticP=facts[METHODS[1]]["diagnosticP"],
                       status="available_candidate",reason=None,p=facts[METHODS[1]]["diagnosticP"],
                       candidateNominalReject=facts[METHODS[1]]["diagnosticP"]<.05)
    reasons=[]
    if not anchors:
        reasons.append("baseline_anchor_invalid_or_missing")
    for method in METHODS[1:]:
        fact=facts[method]
        if fact["status"]!="fitted":
            reasons.append(method+":"+fact["status"])
        else:
            if not fact["guardUsable"]:
                reasons.append(method+":original_guard_failed")
            if not fact["validReference"]:
                reasons.append(method+":invalid_reference")
    for name,passed in comparisons["strict_PORT"]["screens"].items():
        if passed is not True:
            reasons.append("strict_PORT:"+name+(":unknown" if passed is None else ":failed"))
    for name,nonworse in nonworsening.items():
        if nonworse["passes"] is not True:
            reasons.append(name+":objective_nonworsening"+(":unknown" if nonworse["passes"] is None else ":failed"))
    check(eligible == (not reasons),"Independent reason coverage inconsistency")
    return dict(status="available_candidate" if eligible else "unavailable",officialMethod=METHODS[1],hypothesis=hypothesis,tail=tail,
                reasons=reasons,routeChecks=facts,comparisons=comparisons,nonworsening=nonworsening,inference=inference,
                p=inference["p"] if inference is not None else None,
                candidateNominalReject=inference["candidateNominalReject"] if inference is not None else False),anchors

def verify_decision(record,actual):
    expected,_=derive(record)
    same(actual,expected,tolerance=1e-9)
    return expected
