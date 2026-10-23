"""independent cluster: accepted numerical definitions, portable source projection."""
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

MODES = ("literal", "sign_consistent")

METHOD = "participant_CR2_Satterthwaite_v1"

NAMES = ("(Intercept)", "conditionC1", "conditionC3", "need_A", "need_B",
         "conditionC1:need_A", "conditionC3:need_A",
         "conditionC1:need_B", "conditionC3:need_B")

def require(value, message):
    if not value:
        raise ValueError(message)

def positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            "Invalid positive " + label)
    return float(value)

def symmetric_inverse_sqrt(value):
    """A positive-definite inverse square root, without silent eigenvalue clipping."""
    value = np.asarray(value, dtype=float)
    require(value.ndim == 2 and value.shape[0] == value.shape[1] and
            np.isfinite(value).all() and np.allclose(value, value.T, atol=1e-10, rtol=1e-10),
            "Invalid symmetric CR2 adjustment matrix")
    eigenvalues, vectors = np.linalg.eigh((value + value.T) / 2)
    require(np.isfinite(eigenvalues).all() and eigenvalues.min() > 0,
            "Nonpositive CR2 adjustment matrix")
    return (vectors * (1 / np.sqrt(eigenvalues))) @ vectors.T

def contrasts(names):
    require(tuple(names) == NAMES, "Changed condition/reference coefficient design")
    result = np.zeros((4, len(names)))
    result[0, 5] = result[1, 7] = result[2, 5] = result[3, 7] = 1
    result[2, 6] = result[3, 8] = -1
    return result

def cr2_reconstruct(X, y, participant, beta, theta, model_covariance, sigma_squared,
                    L, *, expected_cluster_count=462, expected_rows_per_cluster=10):
    """Rebuild CR2 and contrast df from actual model components, not package internals."""
    X, y, beta = np.asarray(X, float), np.asarray(y, float), np.asarray(beta, float)
    L, model_covariance = np.asarray(L, float), np.asarray(model_covariance, float)
    require(X.ndim == 2 and y.shape == (len(X),) and beta.shape == (X.shape[1],) and
            L.ndim == 2 and L.shape[1] == X.shape[1] and
            model_covariance.shape == (X.shape[1], X.shape[1]) and
            len(participant) == len(X), "Invalid independent model component dimensions")
    require(all(np.isfinite(v).all() for v in (X, y, beta, L, model_covariance)),
            "Nonfinite independent model components")
    positive(sigma_squared, "residual variance")
    require(type(theta) in (int, float) and math.isfinite(theta) and theta >= 0,
            "Invalid fitted relative random-intercept SD")
    require(all(isinstance(v, str) and v for v in participant), "Invalid participant identity")
    positions = {}
    for i, person in enumerate(participant):
        positions.setdefault(person, []).append(i)
    require(len(positions) == expected_cluster_count and
            all(len(v) == expected_rows_per_cluster for v in positions.values()),
            "Changed participant clustering")
    parts = []
    information = np.zeros_like(model_covariance)
    for indices in positions.values():
        indices = np.asarray(indices)
        x = X[indices]
        target = np.eye(len(indices)) + float(theta)**2 * np.ones((len(indices), len(indices)))
        weights = np.linalg.inv(target)
        information += x.T @ weights @ x
        parts.append((x, y[indices] - x @ beta, target, weights))
    M = np.linalg.inv(information)
    require(np.allclose(M, model_covariance / sigma_squared, atol=2e-9, rtol=2e-7),
            "Fitted bread does not match actual X/working covariance")
    chol_M_lower = np.linalg.cholesky(M)
    scores, g_squared, h_rows, leverage = [], [], [], []
    normal_score = np.zeros(X.shape[1])
    for x, residual, target, weights in parts:
        D = np.linalg.cholesky(target).T
        B = target - x @ M @ x.T
        A = D.T @ symmetric_inverse_sqrt(D @ B @ D.T) @ D
        component = M @ x.T @ weights @ A
        normal_score += x.T @ weights @ residual
        white_x = np.linalg.solve(D.T, x)
        leverage.append(float(np.trace(white_x @ M @ white_x.T)))
        scores.append(component @ residual)
        q = L @ component
        g_squared.append(np.sum((q @ D.T)**2, axis=1))
        h_rows.append(q @ x @ chol_M_lower)
    score = np.asarray(scores)
    covariance = score.T @ score
    d = np.asarray(g_squared)
    h = np.asarray(h_rows)
    dfs, traces, denominators = [], [], []
    for j in range(len(L)):
        Hj = h[:, j, :]
        norm2 = np.sum(Hj**2, axis=1)
        trace = float(np.sum(d[:, j]) - np.sum(norm2))
        denominator = float(np.sum(d[:, j]**2) - 2*np.sum(d[:, j]*norm2) +
                            np.sum((Hj.T @ Hj)**2))
        positive(trace, "Satterthwaite trace")
        positive(denominator, "Satterthwaite squared norm")
        dfs.append(trace**2 / denominator)
        traces.append(trace)
        denominators.append(denominator)
    C = L @ covariance @ L.T
    require(np.isfinite(covariance).all() and np.all(np.diag(C) > 0),
            "Invalid independently reconstructed CR2 covariance")
    return dict(covariance=covariance.tolist(), contrastCovariance=C.tolist(),
                estimates=(L @ beta).tolist(), SE=np.sqrt(np.diag(C)).tolist(),
                df=dfs, dfTrace=traces, dfSquaredNorm=denominators,
                participantCount=len(parts), rowCount=len(X),
                normalEquationScore=normal_score.tolist(), participantLeverageTrace=leverage)

def holm(values):
    require(len(values) == 4 and all(type(v) in (int, float) and
            math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid four-item Holm input")
    order = sorted(range(4), key=lambda i: (values[i], i))
    adjusted, largest = [None] * 4, 0.0
    for rank, index in enumerate(order):
        largest = max(largest, min(1.0, (4-rank)*values[index]))
        adjusted[index] = largest
    return adjusted

def references(math_result, coefficient_names, L):
    output = {}
    for j, key in enumerate(IDS):
        estimate, se, df = (math_result[name][j] for name in ("estimates", "SE", "df"))
        positive(se, "contrast SE")
        positive(df, "contrast df")
        t = estimate / se
        critical = float(stats.t.ppf(.975, df))
        output[key] = dict(estimate=estimate, SE=se, df=df, statistic=t,
                           CI=[estimate-critical*se, estimate+critical*se],
                           twoSidedP=float(2*stats.t.sf(abs(t), df)),
                           L=list(L[j]), coefficientNames=list(coefficient_names))
    return output

def planning_mde(SE, df, alpha):
    """Single-contrast fixed-SE/df noncentral-t planning approximation, not family power."""
    positive(SE, "planning SE")
    positive(df, "planning df")
    require(alpha in (.05, .0125), "Unplanned MDE alpha")
    critical = float(stats.t.ppf(1-alpha, df))
    def difference(noncentrality):
        return float(stats.nct.sf(critical, df, noncentrality)) - .8
    upper = 1.
    while difference(upper) < 0:
        upper *= 2
        require(upper <= 1024, "Undefined fixed-SE/df MDE approximation")
    noncentrality = float(optimize.brentq(difference, 0., upper, xtol=1e-12))
    return dict(alpha=alpha, targetPower=.8, SE=SE, df=df, absoluteEffect=SE*noncentrality,
                interpretation="single_contrast_fixed_SE_df_noncentral_t_approximation_not_family_power")

def baseline_degrees_of_freedom(model_covariance, covariance_jacobians,
                                variance_parameter_covariance, L):
    """Independently apply original Satterthwaite delta method to saved derivatives."""
    V, J, Q, L = (np.asarray(x, float) for x in
                  (model_covariance, covariance_jacobians, variance_parameter_covariance, L))
    require(J.ndim == 3 and J.shape[1:] == V.shape and Q.shape == (len(J), len(J)) and
            all(np.isfinite(x).all() for x in (V, J, Q, L)),
            "Invalid saved original Satterthwaite components")
    result = []
    for vector in L:
        variance = float(vector @ V @ vector)
        gradient = np.asarray([vector @ j @ vector for j in J])
        denominator = float(gradient @ Q @ gradient)
        positive(variance, "original contrast variance")
        positive(denominator, "original Satterthwaite denominator")
        result.append(2*variance**2 / denominator)
    return result

def expected_design(rows):
    result = []
    people, condition, y = [], [], []
    for row in rows:
        require(set(row) == {"participant", "condition", "need_A", "need_B", "y"} and
                row["condition"] in ("C1", "C2", "C3"), "Changed actual model request")
        a, b, arm = row["need_A"], row["need_B"], row["condition"]
        c1, c3 = int(arm == "C1"), int(arm == "C3")
        result.append([1., c1, c3, a, b, c1*a, c3*a, c1*b, c3*b])
        people.append(row["participant"])
        condition.append(arm)
        y.append(row["y"])
    require(len(people) == 4620 and len(set(people)) == 462 and
            all(len({p for p, c in zip(people, condition) if c == arm}) == 154 for arm in ("C1", "C2", "C3")),
            "Changed actual participant population")
    require(all(len({c for p, c in zip(people, condition) if p == person}) == 1 for person in set(people)),
            "Participant crosses condition")
    return dict(X=result, y=y, participant=people, condition=condition)

def check_candidate(original, candidate, evidence, request_pin, case, production, kernel, accepted):
    """Independent original-fit and new covariance/reference verification."""
    raw = original["raw"]
    production.validate_actual_fit(raw, evidence, accepted, request_pin, kernel)
    L = contrasts(evidence["coefficientNames"])
    near = accepted.near
    for key, vector in zip(IDS, L):
        near(evidence["references"][key]["L"], vector.tolist(), "four original signed L", atol=0, rtol=0)
    saved = evidence["originalSatterthwaite"]
    original_df = baseline_degrees_of_freedom(evidence["fixedVcov"], saved["covarianceJacobians"],
                                             saved["varianceParameterCovariance"], L)
    near([raw["references"][key]["df"] for key in IDS], original_df,
         "independently reconstructed original df")
    facts = cr2_reconstruct(evidence["design"]["X"], evidence["design"]["y"],
                            evidence["design"]["participant"], evidence["coefficients"],
                            evidence["theta"], evidence["fixedVcov"], evidence["residualVariance"], L)
    refs = references(facts, evidence["coefficientNames"], L)
    expected_refs = {}
    for key, ref in refs.items():
        expected_refs[key] = {k: v for k, v in ref.items() if k != "twoSidedP"}
        expected_refs[key].update(packageTwoSidedP=ref["twoSidedP"], reconstructedEstimate=ref["estimate"],
                                  reconstructedSE=ref["SE"])
    expected_raw = dict(status="fitted", formula=raw["formula"], coefficients=raw["coefficients"],
                        coefficientNames=raw["coefficientNames"], vcov=facts["covariance"],
                        diagnostics=raw["diagnostics"], references=expected_refs, method=METHOD)
    near(candidate, expected_raw, "independent candidate covariance/references/guard")
    differences = []
    def collect(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
                collect(a[key], b[key])
        elif isinstance(a, list) and isinstance(b, list):
            for x, y in zip(a, b):
                collect(x, y)
        elif type(a) in (int, float) and type(b) in (int, float):
            differences.append((abs(a-b), abs(a-b)/max(1., abs(b))))
    collect(candidate, expected_raw)
    return expected_raw, dict(originalDf=original_df, CR2=facts,
        candidateNumericComparisons=len(differences),
        candidateMaximumAbsoluteError=max((v[0] for v in differences), default=0.),
        candidateMaximumScaledError=max((v[1] for v in differences), default=0.))

def descriptive_diagnostics(evidence, case):
    """Known-truth fixed-working-weight diagnostic, separate from candidate inference."""
    design = evidence["design"]
    X, y, beta = (np.asarray(v, float) for v in
                  (design["X"], design["y"], evidence["coefficients"]))
    people, arms, theta = design["participant"], design["condition"], evidence["theta"]
    residual = y-X @ beta
    groups = {}
    for i, person in enumerate(people):
        groups.setdefault(person, []).append(i)
    A = X[:, [0, 3, 4]]
    by_arm, information = {}, np.zeros((9, 9))
    meat = np.zeros((9, 9))
    for arm in ("C1", "C2", "C3"):
        index = [i for i, value in enumerate(arms) if value == arm]
        a, b, r = X[index, 3], X[index, 4], residual[index]
        by_arm[arm] = dict(rows=len(index), participants=len({people[i] for i in index}),
            needARange=[float(a.min()), float(a.max())], needBRange=[float(b.min()), float(b.max())],
            needCorrelation=float(np.corrcoef(a, b)[0, 1]), marginalResidualMean=float(np.mean(r)),
            marginalResidualMeanSquare=float(np.mean(r**2)), marginalResidualRMS=float(np.sqrt(np.mean(r**2))),
            designGram=np.zeros((3, 3)))
    for index in groups.values():
        n, arm = len(index), arms[index[0]]
        W = np.eye(n)-np.ones((n, n))*theta**2/(1+n*theta**2)
        x, a = X[index], A[index]
        information += x.T @ W @ x
        true_cov = np.eye(n)*case["parameters"]["sigmaByArm"][arm]**2 + \
            np.ones((n, n))*case["parameters"]["sdU"]**2
        meat += x.T @ W @ true_cov @ W @ x
        by_arm[arm]["designGram"] += a.T @ W @ a
    for values in by_arm.values():
        gram = values["designGram"]
        for field, j, other in (("needAAdjustedInformation", 1, [0, 2]),
                                ("needBAdjustedInformation", 2, [0, 1])):
            values[field] = float(gram[j, j] - gram[j, other] @
                                  np.linalg.solve(gram[np.ix_(other, other)], gram[other, j]))
        values["designGram"] = gram.tolist()
    bread = np.linalg.inv(information)
    oracle = bread @ meat @ bread
    L = contrasts(evidence["coefficientNames"])
    return dict(clusterCount=len(groups), rowCount=len(X), sigmaSquared=evidence["residualVariance"],
        theta=theta, randomInterceptVariance=evidence["residualVariance"]*theta**2, ICC=theta**2/(1+theta**2),
        byCondition=by_arm,
        designInformationInterpretation="weighted_need_information_adjusted_for_intercept_and_other_need",
        residualInterpretation="marginal_y_minus_Xbeta_no_degrees_of_freedom_bias_correction",
        oracle=dict(interpretation="known_DGP_covariance_with_estimated_working_W_held_fixed_not_candidate_inference",
                    coefficientVcov=oracle.tolist(),
                    contrastSE=dict(zip(IDS, np.sqrt(np.diag(L @ oracle @ L.T)).tolist()))))

def paired_events(baseline, candidate):
    result = []
    for j, old in enumerate(baseline):
        new = None if candidate is None else candidate[j]
        transitions = {name: dict(baseline=bool(old["counts"][field]),
                                  candidate=None if new is None else bool(new["counts"][field]))
                       for name, field in (("availability", "A4"), ("falseRejection", "F"),
                                           ("anyRejection", "R"), ("alternativeRejection", "D"))}
        comparisons = {}
        for key in IDS:
            a, b = old["tests"][key], None if new is None else new["tests"][key]
            x, y = a.get("reference") or {}, {} if b is None else b.get("reference") or {}
            def quotient(numerator, denominator):
                return numerator/denominator if type(numerator) in (float, int) and type(denominator) in (float, int) and \
                    math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0 else None
            def width(ref):
                ci = ref.get("CI")
                return ci[1]-ci[0] if isinstance(ci, list) and len(ci) == 2 and \
                    all(type(v) in (int, float) and math.isfinite(v) for v in ci) and ci[1] >= ci[0] else None
            def mde(ref):
                if not all(type(ref.get(k)) in (int, float) and math.isfinite(ref[k]) and ref[k] > 0 for k in ("SE", "df")):
                    return None
                result = {}
                for alpha in (.05, .0125):
                    value = planning_mde(ref["SE"], ref["df"], alpha)
                    result["alpha_" + str(alpha)] = dict(alpha=alpha, targetPower=.8,
                        noncentrality=value["absoluteEffect"]/ref["SE"], effectMagnitude=value["absoluteEffect"])
                return result
            comparisons[key] = dict(estimate=x.get("estimate"), baselineSE=x.get("SE"), candidateSE=y.get("SE"),
                baselineDf=x.get("df"), candidateDf=y.get("df"), SERatio=quotient(y.get("SE"), x.get("SE")),
                dfRatio=quotient(y.get("df"), x.get("df")), CIWidthRatio=quotient(width(y), width(x)),
                baselineCovered=old["coverage"][key]["covered"],
                candidateCovered=None if new is None else new["coverage"][key]["covered"],
                baselineAvailable=a["status"] == "available_candidate",
                candidateAvailable=None if b is None else b["status"] == "available_candidate",
                baselineReject=a["holmReject"], candidateReject=None if b is None else b["holmReject"],
                assumedEffectMagnitudeOverBaselineSE=quotient(abs(old["coverage"][key]["truth"]), x.get("SE")),
                assumedEffectMagnitudeOverCandidateSE=quotient(abs(old["coverage"][key]["truth"]), y.get("SE")),
                baselinePlanningMDE=mde(x), candidatePlanningMDE=mde(y))
        result.append(dict(tailMode=MODES[j], transitions=transitions, contrasts=comparisons))
    return result

def rate(k, n):
    require(type(k) is int and type(n) is int and 0 <= k <= n, "Invalid binomial denominator")
    if n == 0:
        return None
    interval = [0. if k == 0 else float(stats.beta.ppf(.025, k, n-k+1)),
                1. if k == n else float(stats.beta.ppf(.975, k+1, n-k))]
    return dict(numerator=k, denominator=n, estimate=k/n, interval95=interval,
                intervalMethod="pointwise_Clopper_Pearson")

def event_table(pairs):
    result = dict.fromkeys(("00", "01", "10", "11", "unknown"), 0)
    for a, b in pairs:
        key = f"{int(a)}{int(b)}" if type(a) is bool and type(b) is bool else "unknown"
        result[key] += 1
    return result

def distribution(values):
    x = sorted(v for v in values if type(v) in (int, float) and math.isfinite(v))
    if not x:
        return dict(n=0, missing=len(values), mean=None, SD=None, min=None, median=None, max=None)
    return dict(n=len(x), missing=len(values)-len(x), mean=math.fsum(x)/len(x),
        SD=None if len(x) == 1 else float(np.std(x, ddof=1)), min=x[0], median=x[(len(x)-1)//2], max=x[-1])

def method_report(rows, name, case):
    T = len(rows)
    result = []
    for j, mode in enumerate(MODES):
        counts = dict.fromkeys(("M", "T", "technicalMissing", "A4", "F", "F4", "R", "R4", "D", "D4"), 0)
        partial = 0
        for row in rows:
            branch = row[name][j]
            for key, value in branch["counts"].items():
                counts[key] = counts.get(key, 0)+value
            available = sum(t["status"] == "available_candidate" for t in branch["tests"].values())
            partial += int(0 < available < 4)
        counts.update(M=20, T=T, technicalMissing=20-T)
        nulls = case["trueNullIDs"][mode]
        result.append(dict(tailMode=mode, counts=counts, trueNullIDs=nulls,
            fwerApplicability="applicable" if nulls else "not_applicable_no_true_null",
            rates=dict(availabilityAll=rate(counts["A4"], 20) if T == 20 else None,
                availabilityCompleted=rate(counts["A4"], T),
                falseRejectionAll=rate(counts["F"], 20) if T == 20 and nulls else None,
                falseRejectionCompleted=rate(counts["F"], T) if nulls else None,
                falseRejectionAvailable=rate(counts["F4"], counts["A4"]) if nulls else None,
                anyRejectionAll=rate(counts["R"], 20) if T == 20 else None,
                anyRejectionAvailable=rate(counts["R4"], counts["A4"]),
                alternativeRejectionAll=rate(counts["D"], 20) if T == 20 and len(nulls) < 4 else None,
                alternativeRejectionAvailable=rate(counts["D4"], counts["A4"]) if len(nulls) < 4 else None),
            partialFamilies=partial, zeroFamilies=T-counts["A4"]-partial))
    coverage = {}
    for key in IDS:
        counts = {field: sum(row[name][0]["coverage"][key][field] for row in rows)
                  for field in ("A", "CIAvailable", "C")}
        coverage[key] = dict(counts=counts, coverageAll=rate(counts["C"], 20) if T == 20 else None,
                            coverageAvailable=rate(counts["C"], counts["A"]),
                            availabilityCompleted=rate(counts["A"], T),
                            ciUnavailableAmongAvailable=counts["A"]-counts["CIAvailable"])
    return dict(branches=result, coverage=coverage)

def aggregate_events(manifest, batches):
    output = []
    for case in manifest["cases"]:
        rows = [row for batch in batches for row in batch["cases"] if row["caseID"] == case["caseID"]]
        require(len({row["repID"] for row in rows}) == len(rows), "Duplicate independent scientific ID")
        complete = [row for row in rows if row["technicalCompleted"]]
        T = len(complete)
        pairs, diagnostics = [], {}
        for j, mode in enumerate(MODES):
            transitions = {key: event_table([(row["paired"][j]["transitions"][key]["baseline"],
                                              row["paired"][j]["transitions"][key]["candidate"]) for row in complete])
                           for key in ("availability", "falseRejection", "anyRejection", "alternativeRejection")}
            values = {key: {label: event_table([(row["paired"][j]["contrasts"][key][old],
                                                row["paired"][j]["contrasts"][key][new]) for row in complete])
                           for label, old, new in (("coverage", "baselineCovered", "candidateCovered"),
                                                  ("rejection", "baselineReject", "candidateReject"),
                                                  ("availability", "baselineAvailable", "candidateAvailable"))} for key in IDS}
            pairs.append(dict(tailMode=mode, denominator=T, transitions=transitions, contrasts=values))
        for key in IDS:
            diagnostics[key] = {name: distribution([row["paired"][0]["contrasts"][key][name] for row in complete])
                               for name in ("estimate", "baselineSE", "candidateSE", "baselineDf", "candidateDf", "SERatio", "dfRatio", "CIWidthRatio",
                                            "assumedEffectMagnitudeOverBaselineSE", "assumedEffectMagnitudeOverCandidateSE")}
            diagnostics[key]["oracleSE"] = distribution([
                ((row["candidateMetadata"] or {}).get("designDiagnostics") or {}).get("oracle", {}).get("contrastSE", {}).get(key)
                for row in complete])
            diagnostics[key]["planningMDE"] = {method: {label: distribution([
                ((row["paired"][0]["contrasts"][key][method] or {}).get(label) or {}).get("effectMagnitude")
                for row in complete]) for label in ("alpha_0.05", "alpha_0.0125")}
                for method in ("baselinePlanningMDE", "candidatePlanningMDE")}
        errors, guards = {}, {}
        for row in rows:
            if not row["technicalCompleted"]:
                phase = row["technicalError"]["phase"]
                errors[phase] = errors.get(phase, 0)+1
            label = str((row["diagnostics"] or {}).get("usableCandidateInference")).lower()
            guards[label] = guards.get(label, 0)+1
        output.append(dict(caseID=case["caseID"], M=20, T=T, complete=T == 20, present=len(rows),
            technicalUnresolved=len(rows)-T, technicalMissing=20-T,
            missingRepIDs=[r for r in range(1, 21) if r not in {row["repID"] for row in rows}],
            unresolvedRepIDs=[row["repID"] for row in rows if not row["technicalCompleted"]],
            baseline=method_report(complete, "baselineBranches", case),
            candidate=method_report(complete, "candidateBranches", case), paired=pairs,
            contrastDiagnostics=diagnostics, technicalFailureReasons=errors, originalGuardCounts=guards))
    return output
