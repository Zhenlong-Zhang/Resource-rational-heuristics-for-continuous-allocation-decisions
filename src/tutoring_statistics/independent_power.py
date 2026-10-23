"""independent power: accepted numerical definitions, portable source projection."""
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



NAMESPACE = "h10-power-design-v1-20261020"

METHOD = "participant_CR2_Satterthwaite_v1"

IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")

ARMS = ("C1", "C2", "C3")

MODES = ("literal", "sign_consistent")

def require(value, message):
    if not value:
        raise ValueError(message)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def exact(actual, expected, label="identity"):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), label + " keys")
        for key in expected:
            exact(actual[key], expected[key], label + "." + key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected):
            exact(actual[i], value, label + f"[{i}]")
    else:
        require(type(actual) is type(expected) and actual == expected, "Changed " + label)

def validate_ordinals(values, total):
    require(type(total) is int and total > 0 and type(values) is list and bool(values)
            and all(type(value) is int and 0 <= value < total for value in values)
            and values == sorted(set(values)), "Invalid fixed power ordinal selection")
    return values

def expected_cell_roster():
    """Independently enumerate the accepted finite scientific coordinates."""
    coordinates = []
    for view in ("initial_amount", "terminal_amount"):
        for scenario in (5, 6):
            for multiplier in (.5, 1., 2., 4.):
                for population in ("fixed154", "recruit193_mcar08"):
                    coordinates.append((view, scenario, multiplier, population))
    for view in ("initial_amount", "terminal_amount"):
        for scenario in (1, 2):
            coordinates.append((view, scenario, 1., "recruit193_mcar08"))
    for view in ("initial_amount", "terminal_amount"):
        for scenario in (5, 6):
            coordinates.append((view, scenario, 1., "fixed193"))
    result = []
    for index, (view, scenario, multiplier, population) in enumerate(coordinates, 1):
        delta, parameters, nulls = delta_and_parameters(scenario, multiplier)
        count = 154 if population == "fixed154" else 193
        result.append(dict(cellIndex=index, caseID=f"h10pd1_{view}_s{scenario}_l{int(multiplier * 10):02d}_{population}",
                           view=view, scenarioIndex=scenario, multiplier=multiplier,
                           populationRegime=population, recruitedPerArm=count,
                           eligibilityProbability=.8 if population == "recruit193_mcar08" else 1.,
                           fixedEligiblePerArm=None if population == "recruit193_mcar08" else count,
                           originalCaseIndex=(6 if view == "initial_amount" else 18) + scenario,
                           role="power" if index <= 32 else "new_population_control" if index <= 36 else "effective_N_endpoint",
                           parameters=parameters, delta=delta, trueNullIDs=nulls))
    return result

def delta_and_parameters(scenario, multiplier):
    require(type(scenario) is int and scenario in (1, 2, 5, 6)
            and finite(multiplier) and multiplier in (.5, 1., 2., 4.), "Changed finite effect coordinates")
    vectors = {1: (0., 0., 0., 0.), 2: (0., .15, -.15, 0.),
               5: (-.15, .15, -.15, .15), 6: (-.15, -.15, -.15, -.15)}
    delta = {key: value * multiplier for key, value in zip(IDS, vectors[scenario])}
    parameters = dict(alpha=60., sdU=8., sigmaByArm={arm: 12. for arm in ARMS},
        slopes={"C1": dict(A=.25, B=-.25),
                "C2": dict(A=.25-delta[IDS[0]], B=-.25-delta[IDS[1]]),
                "C3": dict(A=.25-delta[IDS[2]], B=-.25-delta[IDS[3]])}, deltaMagnitude=.15)
    nulls = {mode: [key for key in IDS if
                    (delta[key] <= 0 if mode == "sign_consistent" and key.endswith(":B") else delta[key] >= 0)]
             for mode in MODES}
    return delta, parameters, nulls

def new_seeds(cell_index, rep):
    require(type(cell_index) is int and 1 <= cell_index <= 40 and type(rep) is int and 1 <= rep <= 2000,
            "Invalid new scientific seed identity")
    return {role: base + 10000 * cell_index + rep for role, base in
            (("retentionSeed", 802610200), ("designSeed", 902610200), ("responseSeed", 1002610200))}

def case_for(ordinal, B):
    require(type(B) is int and B in (200, 1000, 2000), "Unselected finite B")
    validate_ordinals([ordinal], 40 * B)
    case = expected_cell_roster()[ordinal % 40]
    case.update(repID=ordinal // 40 + 1, **new_seeds(ordinal % 40 + 1, ordinal // 40 + 1))
    return case

def fit_key(case):
    return f"{case['caseID']}__r{case['repID']:04d}"

def legacy_origin(case):
    if case["populationRegime"] != "fixed154":
        return None
    rep, scenario, view = case["repID"], case["scenarioIndex"], case["view"]
    if case["multiplier"] == 1. and rep <= 1000:
        name = f"h10m1_{view}_s{scenario}_n154"
        return dict(kind="original_production", namespace="h10-gaussian-reference-v1-20261019", caseID=name,
            caseIndex=case["originalCaseIndex"], repID=rep, fitKey=name if rep == 1 else f"{name}__r{rep:04d}",
            recordOrigin="accepted_measurement" if rep == 1 else "production",
            designSeed=202610190+10000*scenario+rep, responseSeed=302610190+10000*case["originalCaseIndex"]+rep)
    if case["multiplier"] in (.5, 4.) and rep <= 2:
        vi = 1 if view == "initial_amount" else 3
        ordinal = (rep-1)*16+vi*4+(scenario-5)*2+int(case["multiplier"] == 4.)
        name = f"h10p_{view}_s{scenario}_l{int(case['multiplier']*10):02d}_n154"
        return dict(kind="actual_shift", namespace="h10-cr2-shift-parity-v1-20261019", ordinal=ordinal,
            caseID=name, caseIndex=case["originalCaseIndex"], repID=rep, fitKey=f"{name}__r{rep:04d}",
            sourceOrdinal=(rep-1)*13+(3 if vi == 1 else 10), designSeed=402620190+rep,
            responseSeed=502610190+10000*(7 if vi == 1 else 19)+rep)
    return None

def precision_specification():
    return dict(BMenu=[200, 1000, 2000], confidence=.95, maximumHalfWidth=.025,
        criterion="maximum_over_all_integer_success_counts", selection="smallest_feasible_menu_value_else_null",
        interpretation="pointwise_MC_precision_not_simultaneous_or_human_power")

def population_counts(rows):
    """Check whole-person clusters and the actual nine-column C2 reference X."""
    require(type(rows) is list, "Rows must be a list")
    people, arms, matrix, responses = {}, {}, [], []
    for row in rows:
        require(type(row) is dict and set(row) == {"participant", "condition", "need_A", "need_B", "y"}
                and row["condition"] in ARMS and type(row["participant"]) is str and row["participant"],
                "Changed actual request row")
        require(all(finite(row[key]) for key in ("need_A", "need_B", "y")), "Nonfinite request row")
        person, arm = row["participant"], row["condition"]
        if person in people:
            require(people[person]["condition"] == arm, "Participant crosses condition")
            people[person]["rows"] += 1
        else:
            people[person] = dict(condition=arm, rows=1)
        a, b = row["need_A"], row["need_B"]
        c1, c3 = int(arm == "C1"), int(arm == "C3")
        matrix.append([1., c1, c3, a, b, c1 * a, c3 * a, c1 * b, c3 * b])
        responses.append(row["y"])
    require(all(person["rows"] == 10 for person in people.values()), "Incomplete person retained")
    for arm in ARMS:
        arms[arm] = sum(person["condition"] == arm for person in people.values())
    return dict(counts=arms, rows=len(rows), participants=len(people),
                X=matrix, y=responses, participant=[row["participant"] for row in rows],
                condition=[row["condition"] for row in rows])

def masked_population(request, people, mask):
    """Apply a declared person-level mask without deleting inconvenient trials."""
    require(type(mask) is dict and set(mask) == set(ARMS), "Wrong retention arms")
    selected = set()
    for arm in ARMS:
        own = [person for person in people if person["condition"] == arm]
        require(type(mask[arm]) is list and len(mask[arm]) == len(own)
                and all(type(value) is bool for value in mask[arm]), "Wrong whole-person mask")
        selected.update(person["participant"] for person, keep in zip(own, mask[arm]) if keep)
    result = dict(model="H10", reference="C2",
                  rows=[row for row in request["rows"] if row["participant"] in selected])
    facts = population_counts(result["rows"])
    require(facts["participants"] == len(selected), "Mask has unknown participants")
    return result, facts["counts"]

def expected_support(rows):
    facts = population_counts(rows)
    # The rank calculation matches the accepted scaled-SVD definition, while
    # its population checks use the actual retained people rather than154.
    X = np.asarray([[value for arm in ARMS for value in
                     ((1., row["need_A"], row["need_B"]) if row["condition"] == arm else (0., 0., 0.))]
                    for row in rows], float).reshape((-1, 9))
    scale = np.linalg.norm(X, axis=0)
    values = np.linalg.svd(X / np.where(scale > 0, scale, 1.), compute_uv=False) if len(rows) else np.zeros(9)
    tolerance = float(max(X.shape) * np.finfo(float).eps * values[0])
    rank = int(np.sum(values > tolerance))
    arms = {arm: dict(participants=facts["counts"][arm], rows=10 * facts["counts"][arm]) for arm in ARMS}
    eligible = rank == 9 and all(value["participants"] > 0 for value in arms.values())
    return dict(arms=arms, rank=rank, columns=9, scaledSingularValues=values.tolist(), rankTolerance=tolerance,
                fitEligible=eligible, reason=None if eligible else "design_support_unavailable")

def support_equal(saved, expected, rows):
    require(type(saved) is dict and set(saved) == set(expected), "Changed support schema")
    fields = set(expected) - {"scaledSingularValues", "rankTolerance"}
    exact({key: saved[key] for key in fields}, {key: expected[key] for key in fields}, "support categories")
    a, b = saved["scaledSingularValues"], expected["scaledSingularValues"]
    require(len(a) == len(b) == 9 and all(finite(value) and value >= 0 for value in a+b)
            and all(left >= right for left, right in zip(a, a[1:])), "Invalid support singular values")
    if not rows:
        exact(a, [0.] * 9, "empty population support")
        exact(saved["rankTolerance"], 0., "empty population tolerance")
        return
    tau = max(len(rows), 9) * 2.**-52 * b[0]
    stored_tau = max(len(rows), 9) * 2.**-52 * a[0]
    require(tau > 0 and all(abs(left-right) <= tau for left, right in zip(a, b)),
            "Support difference exceeds original scaled-SVD rank resolution")
    exact(saved["rankTolerance"], stored_tau, "stored original rank threshold")
    require(sum(value > stored_tau for value in a) == sum(value > tau for value in b) == expected["rank"],
            "Original rank decision differs")
    if expected["fitEligible"]:
        require(all(abs(value-tau) > tau+abs(stored_tau-tau) for value in b),
                "Technical unresolved: eligible rank boundary unstable")
    # A missing-arm/deficient realization remains typed unavailability. This
    # does not declare a deficient design eligible or promote any inference.

def cp_interval(k, n):
    """Independent central95% beta inversion through bisection of betainc."""
    require(type(k) is int and type(n) is int and n > 0 and 0 <= k <= n, "Invalid MC counts")

    def invert(a, b, probability):
        lo, hi = 0., 1.
        for _ in range(64):
            middle = (lo + hi) / 2
            if float(special.betainc(a, b, middle)) < probability:
                lo = middle
            else:
                hi = middle
        return (lo + hi) / 2

    return [0. if k == 0 else invert(k, n - k + 1, .025),
            1. if k == n else invert(k + 1, n - k, .975)]

def rate(k, n):
    require(type(k) is int and type(n) is int and 0 <= k <= n, "Invalid exact event denominator")
    return None if not n else dict(numerator=k, denominator=n, estimate=k / n,
                                   interval95=cp_interval(k, n), intervalMethod="pointwise_Clopper_Pearson")

def precision_table():
    """Only called on a compute node; not evaluated in local preparation tests."""
    values = []
    for B in (200, 1000, 2000):
        intervals = []
        for k in range(B + 1):
            lower, upper = cp_interval(k, B)
            intervals.append(dict(k=k, lower=lower, upper=upper, halfWidth=(upper-lower)/2))
        widths = [row["halfWidth"] for row in intervals]
        maximum = max(widths)
        values.append(dict(B=B, maximumHalfWidth=maximum,
            maximizingCounts=[k for k, width in enumerate(widths) if width == maximum],
            qualifies=maximum <= .025, evaluatedCount=B+1, intervals=intervals))
    chosen = next((value["B"] for value in values if value["qualifies"]), None)
    return values, chosen

def independent_fit(raw, evidence, request_pin, rows, modules):
    """Authenticate actual variable-count RDS and the original model references."""
    math_kernel, accepted = modules["math"], modules["accepted"]
    facts = population_counts(rows)
    require(evidence.get("verified") is True and evidence["REML"] is True
            and evidence["formula"] == "y~condition*need_A+condition*need_B+(1|participant)"
            and evidence["requestDescriptor"] == request_pin
            and evidence["frameRowCount"] == evidence["requestRowCount"] == len(rows)
            and evidence["verifiedFields"] == ["participant", "condition", "need_A", "need_B", "y"],
            "Wrong actual RDS admission")
    near = accepted.near
    near(evidence["design"], {key: facts[key] for key in ("X", "y", "participant", "condition")},
         "actual retained model frame", atol=0, rtol=0)
    require(evidence["optimizer"] == raw["optimizer"] == "nloptwrap"
            and raw["REML"] is True and raw["method"] == "lmerTest_Satterthwaite_candidate"
            and "".join(raw["formula"].split()) == evidence["formula"], "Changed original model")
    near(raw["coefficientNames"], evidence["coefficientNames"], "original coefficient names")
    near([raw["coefficients"][key] for key in evidence["coefficientNames"]], evidence["coefficients"], "original beta")
    for raw_name, proof_name in (("vcov", "fixedVcov"), ("randomCovariance", "randomCovariance"),
                                 ("residualVariance", "residualVariance")):
        near(raw[raw_name], evidence[proof_name], "actual original " + raw_name)
    for key in ("converged", "singular", "gradient", "hessianEigenvalues", "usableCandidateInference"):
        near(raw["diagnostics"][key], evidence["diagnostics"][key], "original guard " + key)
    near(raw["diagnostics"]["convergenceMessages"], evidence["diagnostics"]["optimizerMessages"], "optimizer messages")
    L = math_kernel.contrasts(evidence["coefficientNames"])
    saved = evidence["originalSatterthwaite"]
    usable = raw["diagnostics"]["usableCandidateInference"]
    dfs = math_kernel.baseline_degrees_of_freedom(evidence["fixedVcov"], saved["covarianceJacobians"],
                saved["varianceParameterCovariance"], L) if usable else None
    for j, key in enumerate(IDS):
        ref, source = raw["references"][key], evidence["references"][key]
        near(source["L"], L[j].tolist(), "four original signed contrasts", atol=0, rtol=0)
        for field in ("L", "coefficientNames", "estimate", "SE", "df", "statistic"):
            near(ref[field], source[field], "saved original reference " + field)
        if dfs is not None:
            near(ref["df"], float(dfs[j]), "independent original Satterthwaite df")
        for name, source_name in (("CI", "packageConfidenceInterval"), ("twoSidedP", "packageTwoSidedP")):
            near(ref[name], source[source_name], "actual original " + name)
    return facts, L

def independent_candidate(raw, evidence, facts, L, modules):
    kernel, accepted = modules["math"], modules["accepted"]
    result = kernel.cr2_reconstruct(evidence["design"]["X"], evidence["design"]["y"],
        evidence["design"]["participant"], evidence["coefficients"], evidence["theta"],
        evidence["fixedVcov"], evidence["residualVariance"], L,
        expected_cluster_count=facts["participants"], expected_rows_per_cluster=10)
    refs = kernel.references(result, evidence["coefficientNames"], L)
    accepted.near(raw["vcov"], result["covariance"], "independent CR2 covariance")
    require(raw["method"] == METHOD, "Wrong candidate method")
    for key in IDS:
        for field in ("estimate", "SE", "df", "statistic", "CI", "L", "coefficientNames"):
            accepted.near(raw["references"][key][field], refs[key][field], "independent CR2 " + field)
        accepted.near(raw["references"][key]["packageTwoSidedP"], refs[key]["twoSidedP"], "independent CR2 p")
    expected_refs = {key: {**{field: value for field, value in ref.items() if field != "twoSidedP"},
                          "packageTwoSidedP": ref["twoSidedP"],
                          "reconstructedEstimate": ref["estimate"], "reconstructedSE": ref["SE"]}
                     for key, ref in refs.items()}
    return dict(raw, vcov=result["covariance"], references=expected_refs)

def replay_new(spec, ctx):
    case, reader = spec["case"], ctx["modules"]["generation"]
    kernel_case = dict(case, N=case["recruitedPerArm"])
    actual, kernel_generation = reader.regenerate(kernel_case, ctx["library"])
    n = case["recruitedPerArm"]
    if case["populationRegime"] == "recruit193_mcar08":
        uniforms = np.random.Generator(np.random.PCG64(case["retentionSeed"])).random((3, n))
        masks, values = (uniforms < .8).tolist(), uniforms.tolist()
    else:
        masks, values = [[True] * n for _ in ARMS], None
    mask = dict(zip(ARMS, masks))
    retained, counts = masked_population(actual, kernel_generation["people"], mask)
    selected_people = {row["participant"] for row in retained["rows"]}
    generation = dict(schema="h10-power-design-generation-v1", namespace=NAMESPACE, case=case,
        generator="frozen_measure_generate_all_recruited_then_whole_person_mask", kernelGeneration=kernel_generation,
        populationCounts=counts, recruitedCounts={arm: n for arm in ARMS}, retentionMask=mask,
        retentionUniforms=values, retentionRNG=dict(bitGenerator="PCG64", seed=case["retentionSeed"],
            draw="random((3,recruitedPerArm))" if values is not None else "none_fixed_eligible"),
        recruitedParticipantIDs=[person["participant"] for person in kernel_generation["people"]],
        retainedParticipantIDs=[person["participant"] for person in kernel_generation["people"]
                                if person["participant"] in selected_people],
        retainedRowIndices=[i for i, row in enumerate(actual["rows"])
                            if row["participant"] in selected_people],
        eligibility="independent_whole_person_MCAR_complete_ten_not_trial_dropout", responseTransform="none")
    saved = ctx["observed"].read(spec["generation"])
    exact(saved["execution"], spec["execution"], "original new-input writer")
    generation["execution"] = saved["execution"]
    return retained, generation

def compact_scalar(row, historical):
    compact = historical.compact_events(row)
    compact.update(_powerCompact=True, populationCounts=row["populationCounts"], origin=row["origin"])
    if row["technicalCompleted"]:
        for method in ("baseline", "candidate"):
            for j in range(2):
                for key in IDS:
                    test = row[method+"Branches"][j]["tests"][key]
                    compact[method+"Branches"][j]["tests"][key].update(
                        holmReject=test["holmReject"], candidateNominalReject=test["candidateNominalReject"])
    return compact

def independent_reduction(rows, ctx):
    """Use accepted independent scalar reduction; add only the new finite axes."""
    modules, manifest = ctx["modules"], ctx["manifest"]
    historical, mathematics = modules["reduction"], modules["math"]
    class Math:
        rate = staticmethod(rate)
        event_table = staticmethod(mathematics.event_table)
        distribution = staticmethod(mathematics.distribution)

    by_cell = {cell["caseID"]: [] for cell in manifest["cells"]}
    seen = set()
    for row in rows:
        ordinal = row["ordinal"]
        expected = case_for(ordinal, manifest["B"])
        require(ordinal not in seen and (row.get("_powerCompact") is True or row["case"] == expected)
                and row["caseID"] == expected["caseID"]
                and row["repID"] == expected["repID"] and row["fitKey"] == fit_key(expected),
                "Duplicate or relabeled final member")
        seen.add(ordinal)
        by_cell[expected["caseID"]].append(row if row.get("_powerCompact") is True else compact_scalar(row, historical))
    reports = []
    for cell in manifest["cells"]:
        own = by_cell[cell["caseID"]]
        report = historical.reduce_events(cell, own, Math, planned=manifest["B"])
        report.pop("technicalUnresolved")
        diagnostics = report.pop("practicalTarget")
        report["pointwiseCalibrationDiagnostic"] = dict(
            applicability="prespecified_random_population_controls" if cell["role"] == "new_population_control" else "not_applicable_power_or_N_endpoint",
            reports=diagnostics if cell["role"] == "new_population_control" else None,
            interpretation="descriptive_pointwise_control_limits_not_method_adoption_or_power_gate")
        report.update(cellIndex=cell["cellIndex"], cell=cell)
        for name in ("baseline", "candidate"):
            values = []
            completed = [row for row in own if row["technicalCompleted"] is True]
            for j, mode in enumerate(MODES):
                comparisons = {}
                for key in IDS:
                    tests = [row[name+"Branches"][j]["tests"][key] for row in completed]
                    available = sum(test["status"] == "available_candidate" for test in tests)
                    significant = sum(test["holmReject"] for test in tests)
                    nominal = sum(test["candidateNominalReject"] for test in tests)
                    comparisons[key] = dict(counts=dict(A=available, H=significant, nominal=nominal),
                        holmAll=rate(significant, manifest["B"]) if report["complete"] else None,
                        holmCompleted=rate(significant, len(completed)), holmAvailable=rate(significant, available),
                        nominalAll=rate(nominal, manifest["B"]) if report["complete"] else None)
                values.append(dict(tailMode=mode, contrasts=comparisons))
            report[name]["contrastRejection"] = values
        report["population"] = {arm: mathematics.distribution([row["populationCounts"][arm] for row in own]) for arm in ARMS}
        origins = {}
        for row in own:
            origins[row["origin"]] = origins.get(row["origin"], 0) + 1
        report["sourceOrigins"] = origins
        reports.append(report)
    return reports, seen

def grid_description(reports):
    effect_grid = []
    for view in ("initial_amount", "terminal_amount"):
        for scenario in (5, 6):
            for population in ("fixed154", "recruit193_mcar08"):
                own = sorted([report for report in reports if report["cell"]["view"] == view
                    and report["cell"]["scenarioIndex"] == scenario and report["cell"]["populationRegime"] == population],
                    key=lambda report: report["cell"]["multiplier"])
                require([report["cell"]["multiplier"] for report in own] == [.5, 1., 2., 4.], "Changed finite effect grid")
                for method in ("baseline", "candidate"):
                    for j, tail in enumerate(MODES):
                        points = []
                        for report in own:
                            rates = report[method]["branches"][j]["rates"]
                            discovery = rates["alternativeRejectionAll"]
                            if discovery is None:
                                status = "incomplete_or_not_applicable"
                            elif discovery["interval95"][0] >= .8:
                                status = "pointwise_supported"
                            elif discovery["interval95"][1] < .8:
                                status = "pointwise_below"
                            else:
                                status = "uncertain"
                            points.append(dict(caseID=report["caseID"], multiplier=report["cell"]["multiplier"],
                                anyRejection=rates["anyRejectionAll"], correctAlternativeDiscovery=discovery, status=status))
                        supported = [point["multiplier"] for point in points if point["status"] == "pointwise_supported"]
                        label = ("descriptive_tested_grid_threshold" if supported else
                                 "right_censored_not_established_within_tested_grid" if all(point["status"] == "pointwise_below" for point in points)
                                 else "uncertain_or_incomplete")
                        effect_grid.append(dict(view=view, scenarioIndex=scenario, populationRegime=population,
                            method=method, tailMode=tail, points=points,
                            firstPointwiseSupportedMultiplier=min(supported) if supported else None, gridStatus=label,
                            interpretation="correct_alternative_discovery_grid_not_continuous_MDE_or_simultaneous_selection"))
    endpoints = []
    for view in ("initial_amount", "terminal_amount"):
        for scenario in (5, 6):
            names = {}
            for population in ("fixed154", "fixed193"):
                matches = [report for report in reports if report["cell"]["view"] == view and
                           report["cell"]["scenarioIndex"] == scenario and report["cell"]["populationRegime"] == population
                           and report["cell"]["multiplier"] == 1.]
                require(len(matches) == 1, "Changed two-point N comparison")
                names[population] = matches[0]["caseID"]
            endpoints.append(dict(view=view, scenarioIndex=scenario, **names,
                interpretation="two_effective_N_endpoints_not_a_minimum_N_or_recruitment_guarantee"))
    return dict(effectGrid=effect_grid, effectiveNEndpoints=endpoints, pointwiseOnly=True,
        monotonicityAssumed=False, target=.8, humanPowerClaim=False, historicalShiftReuseGateOverridden=False)
