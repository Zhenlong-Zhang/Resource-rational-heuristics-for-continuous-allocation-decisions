"""independent null: accepted numerical definitions, portable source projection."""
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



ROLE = "matched_validation"

METHOD = "strict_restart_port_check_v1"

METHODS = ("baseline_replay","strict_nlopt_restart","alternate_PORT")

class ValidationIntegrityError(ValueError):
    """A contradictory identity/schema, never an ordinary unavailable fit."""

def require(condition, message):
    if not condition:
        raise ValidationIntegrityError(message)

def same(actual, expected, *, tolerance=0., trail="root"):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),"Mismatched keys: "+trail)
        for key,value in expected.items(): same(actual[key],value,tolerance=tolerance,trail=trail+"."+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),"Mismatched list: "+trail)
        for index,(a,b) in enumerate(zip(actual,expected)): same(a,b,tolerance=tolerance,trail=f"{trail}[{index}]")
    elif type(expected) in (int,float):
        require(type(actual) in (int,float) and math.isfinite(expected) and math.isfinite(actual)
                and math.isclose(actual,expected,rel_tol=tolerance,abs_tol=tolerance),"Mismatched number: "+trail)
    else:
        require(type(actual) is type(expected) and actual==expected,"Mismatched value: "+trail)

def declared_truth(case):
    """Observed slope truth exists for exact Gaussian and symmetric null only."""
    require(case["responseTransform"] in ("gaussian_exact","bounded_grid"), "Unknown response law")
    beta = case["beta"]
    require(type(beta) in (int,float) and math.isfinite(beta) and case["fixedTrialTrend"]==0,
            "Changed declared scientific case")
    if case["responseTransform"]=="gaussian_exact":
        return beta
    if beta==0 and case["intercept"]==50:
        return 0
    return None

def proportion(numerator, denominator):
    require(type(numerator) is int and type(denominator) is int and 0<=numerator<=denominator,
            "Invalid binomial counts")
    if denominator==0:
        return dict(numerator=0,denominator=0,estimate=None,interval95=None)
    lower = 0. if numerator==0 else float(stats.beta.ppf(.025,numerator,denominator-numerator+1))
    upper = 1. if numerator==denominator else float(stats.beta.ppf(.975,numerator+1,denominator-numerator))
    return dict(numerator=numerator,denominator=denominator,estimate=numerator/denominator,interval95=[lower,upper])

def covered(reference, truth):
    interval = reference["CI"]
    require(isinstance(interval,list) and len(interval)==2
            and all(type(x) in (int,float) and math.isfinite(x) for x in interval) and interval[0]<=interval[1],
            "Invalid available confidence interval")
    return interval[0]<=truth<=interval[1]

def paired_case(case, selected, observations):
    """Count original planned inputs separately from technically resolved pairs.

    `selected` is the source-validated immutable per-case manifest population.
    `observations` contains independently admitted new records keyed by repID.
    Missing and technical-unresolved inputs never enter a no-rejection cell.
    """
    ids = [row["repID"] for row in selected]
    require(len(ids)==len(set(ids)) and set(observations)<=set(ids), "Unexpected/duplicate paired identity")
    truth = declared_truth(case)
    counts = Counter(planned=len(ids),present=len(observations),completed=0,available=0,rejected=0,covered=0,
                     oldAvailable=0,oldRejected=0,oldCovered=0)
    availability = [[0,0],[0,0]]
    rejection = [[0,0],[0,0]]
    jointly_available = [[0,0],[0,0]]
    changed = {key:[] for key in ("newlyAvailable","lostAvailable","newlyRejected","lostRejection")}
    technical_reasons, candidate_reasons = Counter(),Counter()
    completed_ids, unresolved_ids, missing_ids = [],[],[]
    for source in selected:
        rep = source["repID"]
        old = source["oldDecision"]
        old_available = old["status"]=="available_candidate"
        require(old["status"] in ("available_candidate","unavailable") and type(old["reject"]) is bool,
                "Unknown original decision status")
        require(not old["reject"] or old_available,"Original unavailable decision cannot reject")
        counts["oldAvailable"] += old_available
        counts["oldRejected"] += old["reject"]
        if old_available and truth is not None:
            counts["oldCovered"] += covered(source["oldInference"],truth)
        observation = observations.get(rep)
        if observation is None:
            missing_ids.append(rep)
            continue
        technical = observation["technicalCompletion"]
        if technical["complete"] is not True:
            unresolved_ids.append(rep); technical_reasons.update(technical["reasons"])
            continue
        decision = observation["decision"]
        available = decision["status"]=="available_candidate"
        require(decision["status"] in ("available_candidate","unavailable")
                and type(decision["candidateNominalReject"]) is bool,"Unknown candidate decision status")
        reject = decision["candidateNominalReject"]
        require(not reject or available,"Unavailable candidate cannot reject")
        if not available:
            require(decision["p"] is None and decision["inference"] is None,"Unavailable candidate exposes formal inference")
        counts["completed"] += 1; completed_ids.append(rep)
        counts["available"] += available; counts["rejected"] += reject
        candidate_reasons.update(decision["reasons"])
        if available and truth is not None:
            counts["covered"] += covered(decision["inference"],truth)
        availability[int(old_available)][int(available)] += 1
        rejection[int(old["reject"])][int(reject)] += 1
        if old_available and available:
            jointly_available[int(old["reject"])][int(reject)] += 1
        if available and not old_available: changed["newlyAvailable"].append(rep)
        if old_available and not available: changed["lostAvailable"].append(rep)
        if reject and not old["reject"]: changed["newlyRejected"].append(rep)
        if old["reject"] and not reject: changed["lostRejection"].append(rep)
    m,t,a,r,c = (counts[key] for key in ("planned","completed","available","rejected","covered"))
    a0,r0,c0 = (counts[key] for key in ("oldAvailable","oldRejected","oldCovered"))
    complete = t==m
    return dict(caseID=case["caseID"],counts=dict(counts),complete=complete,
        missing=len(missing_ids),technicalUnresolved=len(unresolved_ids),unresolvedIncludingMissing=m-t,
        missingIDs=missing_ids,technicalUnresolvedIDs=unresolved_ids,completedIDs=completed_ids,
        truth=dict(known=truth is not None,beta=truth),
        candidate=dict(availability=proportion(a,m) if complete else None,
            availabilityAmongCompleted=proportion(a,t),availabilityObservedLowerBound=a/m if m else None,
            rejectionAllPlanned=proportion(r,m) if complete else None,
            rejectionAmongCompleted=proportion(r,t),rejectionAmongAvailable=proportion(r,a),
            coverageAmongAvailable=proportion(c,a) if truth is not None else None,
            coverageAndAvailableAllPlanned=proportion(c,m) if complete and truth is not None else None),
        original=dict(availability=proportion(a0,m),rejectionAllPlanned=proportion(r0,m),
            rejectionAmongAvailable=proportion(r0,a0),coverageAmongAvailable=proportion(c0,a0) if truth is not None else None,
            coverageAndAvailableAllPlanned=proportion(c0,m) if truth is not None else None),
        pairedAvailability=dict(denominator=t,rows="original unavailable/available",columns="candidate unavailable/available",counts=availability),
        pairedRejection=dict(denominator=t,rows="original no-reject/reject",columns="candidate no-reject/reject",counts=rejection),
        pairedRejectionJointlyAvailable=dict(denominator=sum(map(sum,jointly_available)),
            rows="original no-reject/reject",columns="candidate no-reject/reject",counts=jointly_available),
        changedIDs=changed,technicalReasons=dict(technical_reasons),candidateReasons=dict(candidate_reasons))

def at(value, *keys):
    for key in keys:
        if not isinstance(value,dict) or key not in value: return None
        value = value[key]
    return value

def as_list(value):
    return value if isinstance(value,list) else ([] if value is None else [value])

def finite(value):
    return type(value) in (int,float) and math.isfinite(value)

def technical_completion(record):
    reasons, identified = [],[]
    if not (at(record,"baselineParity","passed") is True and at(record,"baselineParity","nontrivialFiniteReference") is True):
        reasons.append("baseline_parity_unresolved")
    required = ("raw.status","raw.formula","raw.REML","raw.optimizer","raw.coefficientNames","raw.coefficients",
        "raw.vcov","raw.randomCovariance","raw.residualVariance","raw.diagnostics","raw.references.delta",
        "extra.theta","extra.objective.commonAtTheta","extra.controls","extra.optimizerCode","extra.finiteDifference",
        "extra.optinfo","extra.matrixX","extra.matrixZ","raw.diagnostics.converged","raw.diagnostics.singular",
        "raw.diagnostics.gradient","raw.diagnostics.hessianEigenvalues","raw.diagnostics.convergenceMessages",
        "raw.diagnostics.usableCandidateInference","raw.references.delta.estimate","raw.references.delta.SE",
        "raw.references.delta.df","raw.references.delta.statistic","raw.references.delta.CI","extra.theta.names",
        "extra.theta.values","extra.theta.lower","extra.theta.lowerKinds")
    containers = {"raw.diagnostics","raw.references.delta","extra.theta","extra.finiteDifference",
                  "extra.optinfo","extra.matrixX","extra.matrixZ"}
    for method in METHODS:
        route = at(record,"methods",method)
        if route is None:
            reasons.append("missing_route:"+method); continue
        if route.get("status")!="fitted":
            if at(route,"raw","status")=="unavailable" and at(route,"raw","reason")=="backend_fit_error":
                identified.append(method)
            else: reasons.append("unclassified_route_error:"+method)
            continue
        if not isinstance(route.get("fitArtifact"),dict): reasons.append("missing_fit_artifact:"+method)
        for path in required:
            value,found = route,True
            for key in path.split("."):
                if not isinstance(value,dict) or key not in value:
                    found = False; break
                value = value[key]
            if not found or (path in containers and not isinstance(value,dict)):
                reasons.append("missing_core:"+method+":"+path)
    return dict(complete=not reasons,reasons=reasons,identifiedMethodFailures=identified)

def compact_record(record):
    if "diagnosticRecord" not in record:
        # Outside a requested partial scope, only the producer's compact
        # summary is reconciled; no numerical/source acceptance is claimed.
        return record
    directions,warnings = {},{}
    for method in METHODS:
        beta = at(record,"diagnosticRecord","methods",method,"raw","coefficients","delta")
        directions[method] = (int(beta>0)-int(beta<0)) if finite(beta) else None
        warnings[method] = len(as_list(at(record,"diagnosticRecord","methods",method,"raw","diagnostics","warnings")))
    return {**{key:record[key] for key in ("caseID","repID","oldDecision","oldInference","truth","decision","technicalCompletion")},
            "routeDirections":directions,"ordinaryWarningCounts":warnings}

def describe(values):
    numbers = sorted(value for value in values if finite(value)); n = len(numbers)
    quantile = lambda q: numbers[math.floor(q*(n-1))] if n else None
    return dict(finite=n,unknown=len(values)-n,mean=math.fsum(numbers)/n if n else None,
                median=quantile(.5),q10=quantile(.1),q90=quantile(.9),
                minimum=numbers[0] if n else None,maximum=numbers[-1] if n else None)

def boolean_counts(values):
    return dict(true=sum(value is True for value in values),false=sum(value is False for value in values),
                unknown=sum(value is not True and value is not False for value in values))

def change_ids(ids,values):
    return dict(changedIDs=[rep for rep,value in zip(ids,values) if value is True],
                unchangedIDs=[rep for rep,value in zip(ids,values) if value is False],
                unknownIDs=[rep for rep,value in zip(ids,values) if value is not True and value is not False])

def case_report(shard,observations):
    """Project independently counted pairs and raw-derived diagnostics to schema48."""
    selected = shard["records"]; independent = paired_case(shard["case"],selected,observations)
    ids = [row["repID"] for row in selected]; rows = [observations.get(rep) for rep in ids]
    present = [row for row in rows if row is not None]
    completed = [row for row in present if row["technicalCompletion"]["complete"] is True]
    both = [row for row in completed if row["oldDecision"]["status"]==row["decision"]["status"]=="available_candidate"]
    def table(population,old_key,new_key,labels):
        buckets = {a+"_to_"+b:[] for a in labels for b in labels}
        for row in population:
            a,b = bool(old_key(row)),bool(new_key(row))
            buckets[labels[int(a)]+"_to_"+labels[int(b)]].append(row["repID"])
        return dict(denominator=len(population),table={key:len(value) for key,value in buckets.items()},ids=buckets)
    paired = dict(availability=table(completed,lambda r:r["oldDecision"]["status"]=="available_candidate",
        lambda r:r["decision"]["status"]=="available_candidate",("unavailable","available")),
        rejection=table(completed,lambda r:r["oldDecision"]["reject"],lambda r:r["decision"]["candidateNominalReject"],("no_reject","reject")),
        rejectionBothAvailable=table(both,lambda r:r["oldDecision"]["reject"],lambda r:r["decision"]["candidateNominalReject"],("no_reject","reject")))
    compact = [compact_record(row) if row is not None else None for row in rows]
    screens = ("commonObjective","fixedCoefficients","deltaSE","deltaDF","theta","residualVariance")
    route_reports = {}
    for method in METHODS:
        facts = [at(row,"decision","routeChecks",method) or {} for row in compact]
        route_reports[method] = dict(statusCounts=dict(Counter(fact.get("status","missing") for fact in facts)),
            guardCounts=boolean_counts([fact.get("guardUsable") for fact in facts]),
            validReferenceCounts=boolean_counts([fact.get("validReference") for fact in facts]),
            richardsonStatusCounts=dict(Counter(fact.get("richardsonStatus","missing") for fact in facts)),
            ordinaryWarningCount=sum(at(row,"ordinaryWarningCounts",method) or 0 for row in compact))
    comparisons = {}
    for pair,a,b in (("baseline_strict",METHODS[0],METHODS[1]),("baseline_PORT",METHODS[0],METHODS[2]),("strict_PORT",METHODS[1],METHODS[2])):
        values = [at(row,"decision","comparisons",pair) or {} for row in compact]
        directions = []
        for row in compact:
            x,y = at(row,"routeDirections",a),at(row,"routeDirections",b)
            directions.append(x!=y if x is not None and y is not None else None)
        report = dict(screens={key:boolean_counts([at(value,"screens",key) for value in values]) for key in screens},
            continuous={key:describe([value.get(key) for value in values]) for key in ("objectiveDifference","deltaSERelativeDifference",
                "deltaDFRelativeDifference","thetaRelativeMaxDifference","residualVarianceRelativeDifference")},
            fixedCoefficientDifferences={name:describe([at(value,"fixedCoefficientDifferences",name) for value in values]) for name in ("(Intercept)","delta")},
            directionChanges=change_ids(ids,directions),pThresholdCrossings=change_ids(ids,[value.get("diagnosticPThresholdCrossing") for value in values]))
        for field in ("fixedVcovDifference","randomCovarianceDifference"):
            report[field] = {}
            for i in range(2):
                for j in range(2):
                    numbers = []
                    for value in values:
                        matrix = value.get(field)
                        numbers.append(matrix[i][j] if isinstance(matrix,list) and len(matrix)==2
                            and isinstance(matrix[i],list) and len(matrix[i])==2 else None)
                    report[field][f"{i}_{j}"] = describe(numbers)
        comparisons[pair] = report
    nonworsening = {method:dict(counts=boolean_counts([at(row,"decision","nonworsening",method,"passes") for row in compact]),
        objectiveDifference=describe([at(row,"decision","nonworsening",method,"objectiveDifference") for row in compact])) for method in ("strict","PORT")}
    c = independent["counts"]; original,candidate = independent["original"],independent["candidate"]
    known = independent["truth"]["known"]
    target = dict(known=known,beta=independent["truth"]["beta"],interpretation=("gaussian_conditional_fixed_coefficient"
        if shard["case"]["responseTransform"]=="gaussian_exact" else "bounded_null_symmetry" if known else "bounded_nonzero_observed_slope_unknown"))
    return dict(case=shard["case"],planned=c["planned"],recordPresent=c["present"],technicalCompleted=c["completed"],
        technicalUnresolved=c["planned"]-c["completed"],presentTechnicalUnresolved=independent["technicalUnresolved"],
        missing=independent["missing"],complete=independent["complete"],missingIDs=independent["missingIDs"],
        technicalUnresolvedIDs=[rep for rep in ids if rep not in observations or observations[rep]["technicalCompletion"]["complete"] is not True],
        technicalReasons=independent["technicalReasons"],truth=target,coverageReason=None if known else "bounded_nonzero_observed_slope_unknown",
        original=dict(available=c["oldAvailable"],rejected=c["oldRejected"],covered=c["oldCovered"] if known else None,
            rates=dict(availabilityAll=original["availability"],rejectionAll=original["rejectionAllPlanned"],
                rejectionAmongAvailable=original["rejectionAmongAvailable"],coverageAmongAvailable=original["coverageAmongAvailable"],
                coverageAndAvailableAll=original["coverageAndAvailableAllPlanned"])),
        candidate=dict(available=c["available"],unavailable=c["completed"]-c["available"],rejected=c["rejected"],covered=c["covered"] if known else None,
            rates=dict(availabilityAll=candidate["availability"],availabilityCompleted=candidate["availabilityAmongCompleted"],
                observedAvailabilityLowerBound=dict(numerator=c["available"],denominator=c["planned"],estimate=c["available"]/c["planned"],
                    interpretation="observed_lower_bound_until_technical_completion"),
                rejectionAll=candidate["rejectionAllPlanned"],rejectionCompleted=candidate["rejectionAmongCompleted"],
                rejectionAmongAvailable=candidate["rejectionAmongAvailable"],coverageAmongAvailable=candidate["coverageAmongAvailable"],
                coverageAndAvailableAll=candidate["coverageAndAvailableAllPlanned"])),paired=paired,
        candidateReasonsCompleted=independent["candidateReasons"],
        candidatePrimaryReasonsCompleted=dict(Counter(row["decision"]["reasons"][0] for row in completed if row["decision"]["reasons"])),
        candidateReasonsObserved=dict(Counter(reason for row in present for reason in row["decision"]["reasons"])),
        diagnostics=dict(scope="all_observed_records_including_technical_unresolved",routes=route_reports,comparisons=comparisons,nonworsening=nonworsening))
