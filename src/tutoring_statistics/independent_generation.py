"""independent generation: accepted numerical definitions, portable source projection."""
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



NAMESPACE = "h10-gaussian-reference-v1-20261019"

IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")

ARMS = ("C1", "C2", "C3")

VIEWS = ("initial_percent", "initial_amount", "terminal_percent", "terminal_amount")

MODES = ("literal", "sign_consistent")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def null_membership(delta, mode):
    require(set(delta) == set(IDS) and all(finite(delta[key]) for key in IDS) and mode in MODES,
            "Invalid contrast truth/tail")
    return [
        key for key in IDS if (delta[key] <= 0 if mode == "sign_consistent" and key.endswith(":B") else delta[key] >= 0)]

def expected_plan():
    names = ("common_slope_null", "partial_boundary", "direction_boundary", "heterogeneous_noise_null",
             "sign_consistent_alternative", "literal_alternative")
    cases = []
    for vi, view in enumerate(VIEWS):
        percentage = view.endswith("percent")
        g_a, g_b, d, base_sigma = (.2, -.2, .1, 10.) if percentage else (.25, -.25, .15, 12.)
        vectors = ((0., 0., 0., 0.), (0., d, -d, 0.), (d, 0., 0., -d),
                   (0., 0., 0., 0.), (-d, d, -d, d), (-d, -d, -d, -d))
        for si in range(1, 7):
            delta = dict(zip(IDS, vectors[si-1]))
            parameters = dict(alpha=50. if percentage else 60., sdU=6. if percentage else 8.,
                sigmaByArm={arm: base_sigma * ((2., 1., 1.5)[ai] if si == 4 else 1.) for ai, arm in enumerate(ARMS)},
                slopes={"C1": dict(A=g_a, B=g_b),
                        "C2": dict(A=g_a-delta[IDS[0]], B=g_b-delta[IDS[1]]),
                        "C3": dict(A=g_a-delta[IDS[2]], B=g_b-delta[IDS[3]])}, deltaMagnitude=d)
            index = 6*vi + si
            cases.append(dict(caseID=f"h10m1_{view}_s{si}_n154", caseIndex=index, view=view,
                scenarioID=names[si-1], scenarioIndex=si, N=154, repID=1,
                designSeed=202610190+10000*si+1, responseSeed=302610190+10000*index+1,
                parameters=parameters, delta=delta,
                trueNullIDs={mode: null_membership(delta, mode) for mode in MODES},
                futureProposedB=2000 if si <= 4 else 1000,
                dgpClass="heteroscedastic_stress" if si == 4 else "conditional_gaussian_model_reference"))
    return dict(namespace=NAMESPACE, measurementReplicatesPerCase=1, NPerArm=154,
        views=list(VIEWS), scenarios=list(names), cases=cases, futureProposedFits=40000,
        futureProductionAuthorized=False, scope="design_and_target_N_measurement_not_calibration_or_power_estimation")

def design_library(canonical):
    rows = canonical["rows"]
    require(len(rows) == 270, "Changed accepted native design size")
    library = {}
    for arm in ARMS:
        library[arm] = []
        for variant in range(9):
            participant = f"h10-alternatives-{arm}-v{variant}"
            own = sorted((r for r in rows if r["participant"] == participant), key=lambda x: x["mainIndex"])
            require(len(own) == 10 and [r["mainIndex"] for r in own] == list(range(1, 11)), "Changed joint native schedule")
            require(all(r["condition"] == arm and r["session"] == participant + "-session" and
                        r["profileId"] == "pilot-r9-design-v1" and r["flowVersion"] == "r10-prepractice-v1" and
                        type(r["remainingBudgetMs"]) is int and r["remainingBudgetMs"] > 0 and
                        finite(r["share"]) and r["share"] == r["shareIndex"]/1000 for r in own),
                    "Changed accepted design identity/qualification")
            require(all(finite(r[k]) and 1 <= r[k] <= 100 for r in own for k in
                        ("meanAInitial", "meanBInitial", "meanATerminal", "meanBTerminal")), "Invalid exact design covariates")
            library[arm].append(own)
    require({r["participant"] for r in rows} == {r[0]["participant"] for group in library.values() for r in group},
            "Extra native person/source")
    return library

def regenerate(case, library):
    """Replay the declared input-law primitives independently of the producer."""
    require(np.__version__ == "1.23.5", "Wrong frozen PCG64 implementation")
    n = case["N"]
    indices = np.random.Generator(np.random.PCG64(case["designSeed"])).integers(0, 9, size=(3, n), dtype=np.int64)
    response = np.random.Generator(np.random.PCG64(case["responseSeed"]))
    zi = response.standard_normal(3*n)
    zij = response.standard_normal((3*n, 10))
    people, rows = [], []
    p = case["parameters"]
    suffix = "Initial" if case["view"].startswith("initial_") else "Terminal"
    for ai, arm in enumerate(ARMS):
        for pi in range(n):
            person = f"{NAMESPACE}:{case['caseID']}:r{case['repID']}:{arm}:p{pi+1:03d}"
            i = ai*n + pi
            schedule = library[arm][int(indices[ai, pi])]
            u = p["sdU"] * float(zi[i])
            residuals = [p["sigmaByArm"][arm] * float(value) for value in zij[i]]
            people.append(dict(participant=person, condition=arm, personIndex=pi+1,
                scheduleIndex=int(indices[ai, pi]), sourceParticipant=schedule[0]["participant"], sourceSession=schedule[0]["session"],
                sourceMainIndices=[r["mainIndex"] for r in schedule], interceptZ=float(zi[i]), residualZ=zij[i].tolist(),
                randomIntercept=u, residuals=residuals))
            for ti in range(10):
                a, b = schedule[ti]["meanA" + suffix], schedule[ti]["meanB" + suffix]
                eta = p["alpha"] + p["slopes"][arm]["A"] * a + p["slopes"][arm]["B"] * b
                rows.append(dict(participant=person, condition=arm, need_A=a, need_B=b, y=eta+u+residuals[ti]))
    generation = dict(schema="h10-calibration-measure-generation-v1", namespace=NAMESPACE, case=case,
        rng=dict(numpyVersion=np.__version__, bitGenerator="PCG64", designSeed=case["designSeed"], responseSeed=case["responseSeed"],
            designDraw="integers(0,9,size=(3,N),dtype=int64)",
            responseDrawOrder=["standard_normal(3*N)", "standard_normal((3*N,10))"]),
        people=people, responseTransform="none", clippedCount=0, roundedCount=0,
        sourceRole="accepted_native_joint_X_only", responseRole="new_conditional_gaussian_reference",
        rowOrder="C1,C2,C3; person 1..N; source mainIndex 1..10")
    return dict(model="H10", reference="C2", rows=rows), generation

def expected_support(rows):
    X = np.asarray([[value for arm in ARMS for value in
                    ((1., row["need_A"], row["need_B"]) if row["condition"] == arm else (0., 0., 0.))]
                    for row in rows], dtype=float)
    scale = np.linalg.norm(X, axis=0)
    sv = np.linalg.svd(X / np.where(scale > 0, scale, 1.), compute_uv=False)
    rank = int(np.sum(sv > max(X.shape)*np.finfo(float).eps*sv[0]))
    arms = {arm: dict(participants=len({r["participant"] for r in rows if r["condition"] == arm}),
                     rows=sum(r["condition"] == arm for r in rows)) for arm in ARMS}
    usable = rank == 9 and all(x["participants"] == 154 and x["rows"] == 1540 for x in arms.values())
    return dict(arms=arms, rank=rank, columns=9, scaledSingularValues=sv.tolist(), fitEligible=usable,
                reason=None if usable else "design_support_unavailable")

def branch_outcome(case, mode, raw, accepted):
    nulls = null_membership(case["delta"], mode)
    require(case["trueNullIDs"][mode] == nulls, "Changed true-null membership")
    tests = {}
    for key in IDS:
        tail = "greater" if mode == "sign_consistent" and key.endswith(":B") else "less"
        if raw["status"] == "fitted":
            ref = raw["references"][key]
            tests[key] = dict(ref, **accepted.directed(ref, tail, raw["diagnostics"]["usableCandidateInference"] is True), reference=ref)
        else:
            tests[key] = dict(tail=tail, status="unavailable", reason=raw["reason"], p=None,
                              candidateNominalReject=False, reference=None)
    family = accepted.family(tests)
    a4 = family["completeFamilyAvailable"]
    F = any(family["tests"][key]["holmReject"] for key in nulls)
    D = any(family["tests"][key]["holmReject"] for key in IDS if key not in nulls)
    R = family["anyCandidateContrastRejects"]
    coverage = {}
    for key in IDS:
        test = family["tests"][key]
        available = test["status"] == "available_candidate"
        ci = test["reference"].get("CI") if test["reference"] is not None else None
        ci_valid = available and isinstance(ci, list) and len(ci) == 2 and all(finite(x) for x in ci) and ci[0] <= ci[1]
        covered = ci[0] <= case["delta"][key] <= ci[1] if ci_valid else None
        c = int(covered is True)
        coverage[key] = dict(A=int(available), CIAvailable=int(ci_valid), C=c, covered=covered, truth=case["delta"][key],
            coverageAvailable=c if available else None, coverageAll=c, availabilityAll=int(available),
            interpretation="conditional_mean_slope_difference")
    return dict(tailMode=mode, trueNullIDs=nulls, falseNullIDs=[key for key in IDS if key not in nulls], **family,
        counts=dict(M=1, T=1, A4=int(a4), F=int(F), F4=int(a4 and F), R=int(R), R4=int(a4 and R),
                    D=int(D), D4=int(a4 and D), technicalMissing=0),
        fwerApplicability="applicable" if nulls else "not_applicable_no_true_null",
        rates=dict(availabilityAll=int(a4), falseRejectionAll=int(F) if nulls else None,
            falseRejectionAvailable=int(F) if a4 and nulls else None, anyRejectionAll=int(R),
            anyRejectionAvailable=int(R) if a4 else None, alternativeRejectionAll=int(D) if len(nulls) < 4 else None,
            alternativeRejectionAvailable=int(D) if a4 and len(nulls) < 4 else None), coverage=coverage)

def validate_actual_fit(raw, evidence, accepted):
    near = accepted.near
    require("".join(raw["formula"].split()) == evidence["formula"] and raw["REML"] is True and
            raw["optimizer"] == evidence["optimizer"] and raw["method"] == "lmerTest_Satterthwaite_candidate", "Changed actual model")
    near(evidence["modelFrame"], evidence["requestRows"], "RDS exact source frame", atol=0, rtol=0)
    near(raw["coefficientNames"], evidence["coefficientNames"], "actual coefficient names")
    near([raw["coefficients"][key] for key in evidence["coefficientNames"]], evidence["coefficients"], "actual beta")
    near(raw["vcov"], evidence["fixedVcov"], "complete actual covariance")
    near(raw["randomCovariance"], evidence["randomCovariance"], "actual random covariance")
    near(raw["residualVariance"], evidence["residualVariance"], "actual residual variance")
    for key in ("converged", "singular", "gradient", "hessianEigenvalues", "usableCandidateInference"):
        near(raw["diagnostics"][key], evidence["diagnostics"][key], "original guard " + key)
    near(raw["diagnostics"]["convergenceMessages"], evidence["diagnostics"]["optimizerMessages"], "original optimizer messages")
    require(set(raw["conditionSlopes"]) == set(evidence["conditionSlopes"]) and set(raw["references"]) == set(IDS), "Changed slopes/contrast scope")
    for key, slope in evidence["conditionSlopes"].items():
        for field in ("L", "estimate", "SE"):
            near(raw["conditionSlopes"][key][field], slope[field], "actual condition slope")
    for key in IDS:
        ref, proof = raw["references"][key], evidence["references"][key]
        for field in ("estimate", "SE", "df", "statistic", "L", "coefficientNames"):
            near(ref[field], proof[field], "independent actual contrast")
        near(proof["packageEstimate"], proof["estimate"], "independent package estimate")
        near(proof["packageSE"], proof["SE"], "independent package SE")
        if all(finite(proof[k]) for k in ("estimate", "SE", "df", "statistic")) and proof["SE"] > 0 and proof["df"] > 0:
            critical = float(stats.t.ppf(.975, proof["df"]))
            ci = [proof["estimate"]-critical*proof["SE"], proof["estimate"]+critical*proof["SE"]]
            near(ref["CI"], ci, "independent pointwise CI")
            near(proof["packageConfidenceInterval"], ci, "independent named R interval")
            near(ref["twoSidedP"], float(2*stats.t.sf(abs(proof["statistic"]), proof["df"])), "independent two-sided p")
