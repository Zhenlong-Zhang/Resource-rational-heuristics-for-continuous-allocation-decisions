"""power reference: accepted numerical definitions, portable source projection."""
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

SCOPE = "conditional_power_population_and_effect_sensitivity"

ARMS = ("C1", "C2", "C3")

TAILS = ("literal", "sign_consistent")

CONTRASTS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")

PRECISION = {"BMenu": [200, 1000, 2000], "confidence": .95,
             "maximumHalfWidth": .025, "criterion": "maximum_over_all_integer_success_counts",
             "selection": "smallest_feasible_menu_value_else_null",
             "interpretation": "pointwise_MC_precision_not_simultaneous_or_human_power"}

SEEDS = {"retention": 802610200, "design": 902610200, "response": 1002610200}

STRIDE = 10000

_KERNELS = None

def require(condition, message):
    if not condition:
        raise ValueError(message)

def kernels():
    global _KERNELS
    require(sys.version_info[:2] == (3, 10), "Changed Python runtime")
    if _KERNELS is None:
        from . import joint_generation as m
        from . import cluster_inference as hi
        from . import fresh_reference as f
        import numpy as np
        import scipy
        require(np.__version__ == "1.23.5" and scipy.__version__ == "1.10.0", "Changed numerical runtime")
        _KERNELS = (m, hi, f, np)
    return _KERNELS

def read(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)

def pin(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file(), "Missing absolute bound file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}

def observe(descriptor, observed):
    require(isinstance(descriptor, dict) and set(descriptor) == {"path", "sha256"}
            and Path(descriptor["path"]).is_absolute(), "Changed bound file")
    if descriptor["path"] in observed:
        require(observed[descriptor["path"]] == descriptor["sha256"], "Conflicting source pins")
        return descriptor
    require(pin(descriptor["path"]) == descriptor, "Changed bound file")
    require(observed.setdefault(descriptor["path"], descriptor["sha256"]) == descriptor["sha256"],
            "Conflicting source pins")
    return descriptor

def closure(observed):
    for path, digest in observed.items():
        require(pin(path)["sha256"] == digest, "Source changed at final closure")
    return True

def write(value, path):
    path = Path(path)
    require(path.is_absolute() and not path.exists() and not path.is_symlink(), "Refusing overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".uncommitted-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return pin(path)

def execution():
    return dict(pid=os.getpid())

@lru_cache(maxsize=1)
def _cell_templates():
    roster = []
    for view in ("initial_amount", "terminal_amount"):
        for scenario in (5, 6):
            for multiplier in (.5, 1., 2., 4.):
                for population in ("fixed154", "recruit193_mcar08"):
                    roster.append((view, scenario, multiplier, population, "power"))
    roster += [(v, s, 1., "recruit193_mcar08", "new_population_control")
               for v in ("initial_amount", "terminal_amount") for s in (1, 2)]
    roster += [(v, s, 1., "fixed193", "effective_N_endpoint")
               for v in ("initial_amount", "terminal_amount") for s in (5, 6)]
    result = []
    vectors = {1: (0, 0, 0, 0), 2: (0, 1, -1, 0), 5: (-1, 1, -1, 1), 6: (-1, -1, -1, -1)}
    for index, (view, scenario, multiplier, population, role) in enumerate(roster, 1):
        delta = dict(zip(CONTRASTS, [sign * .15 * multiplier for sign in vectors[scenario]]))
        nulls = {mode: [key for key, value in delta.items() if
                 (value <= 0 if mode == "sign_consistent" and key.endswith(":B") else value >= 0)]
                 for mode in TAILS}
        count = 154 if population == "fixed154" else 193
        result.append(dict(cellIndex=index, caseID=f"h10pd1_{view}_s{scenario}_l{int(multiplier*10):02d}_{population}",
            view=view, scenarioIndex=scenario, multiplier=multiplier, role=role,
            originalCaseIndex=(6 if view == "initial_amount" else 18)+scenario,
            populationRegime=population, recruitedPerArm=count,
            eligibilityProbability=.8 if population == "recruit193_mcar08" else 1.,
            fixedEligiblePerArm=None if population == "recruit193_mcar08" else count,
            parameters=dict(alpha=60., sdU=8., sigmaByArm={a: 12. for a in ARMS}, deltaMagnitude=.15,
                slopes={"C1": {"A": .25, "B": -.25},
                        "C2": {"A": .25-delta[CONTRASTS[0]], "B": -.25-delta[CONTRASTS[1]]},
                        "C3": {"A": .25-delta[CONTRASTS[2]], "B": -.25-delta[CONTRASTS[3]]}}),
            delta=delta, trueNullIDs=nulls))
    return result

def cells():
    return copy.deepcopy(_cell_templates())

def case_for(ordinal, B):
    require(type(B) is int and B in PRECISION["BMenu"] and type(ordinal) is int and 0 <= ordinal < 40*B,
            "Invalid fixed scientific ordinal")
    case = copy.deepcopy(_cell_templates()[ordinal % 40])
    rep = ordinal//40+1
    case.update(repID=rep, **{role+"Seed": base+STRIDE*case["cellIndex"]+rep for role, base in SEEDS.items()})
    return case

def fit_key(case):
    return f"{case['caseID']}__r{case['repID']:04d}"

def ordinals(value, B):
    require(isinstance(value, list) and value and all(type(v) is int and 0 <= v < 40*B for v in value)
            and value == sorted(set(value)), "Invalid duplicate or unordered ordinals")
    return value

def precision(request_path, output_path, *, source_context):
    request = read(request_path)
    require(request.get("schema") == "h10-power-design-precision-request-v1" and request["precision"] == PRECISION,
            "Changed precision specification")
    _, observed, _ = source_context(request)
    from scipy.stats import beta, binom
    np = kernels()[3]
    evidence = pin(Path(request_path).resolve()); observe(evidence, observed)
    evaluations = []
    for B in PRECISION["BMenu"]:
        k = np.arange(B+1)
        lower = np.zeros(B+1); upper = np.ones(B+1)
        lower[1:] = beta.ppf(.025, k[1:], B-k[1:]+1)
        upper[:-1] = beta.ppf(.975, k[:-1]+1, B-k[:-1])
        widths = (upper-lower)/2
        require(np.all(np.isfinite(widths)) and np.all((widths >= 0) & (widths <= .5)), "Invalid CP intervals")
        maximum = float(np.max(widths))
        evaluations.append(dict(B=B, maximumHalfWidth=maximum, maximizingCounts=np.flatnonzero(widths == maximum).tolist(),
            qualifies=maximum <= PRECISION["maximumHalfWidth"], evaluatedCount=B+1,
            intervals=[dict(k=int(i), lower=float(lower[i]), upper=float(upper[i]), halfWidth=float(widths[i])) for i in k]))
    selected = next((e["B"] for e in evaluations if e["qualifies"]), None)
    assurance = float(binom.sf(153, 193, .8))
    return write(dict(schema="h10-power-design-precision-v1", request=evidence, namespace=NAMESPACE,
        precision=PRECISION, evaluations=evaluations, selectedB=selected,
        retentionCountAssurance=dict(recruitedPerArm=193, probability=.8, requiredEligible=154,
            singleArm=assurance, threeIndependentArms=assurance**3, fiveIndependentArms=assurance**5,
            interpretation="independent_whole_person_binomial_count_assumption_not_power"),
        newScientificInputs=0, newFitInvocations=0, execution=execution(), observedInputs=observed,
        sourceClosure=closure(observed)), Path(output_path).resolve())

def generate(case, library):
    m, _, _, np = kernels()
    kernel_case = copy.deepcopy(case); kernel_case["N"] = case["recruitedPerArm"]
    n = case["recruitedPerArm"]
    if case["populationRegime"] == "recruit193_mcar08":
        uniforms = np.random.Generator(np.random.PCG64(case["retentionSeed"])).random((3, n))
        mask = (uniforms < .8).tolist(); uniform_values = uniforms.tolist()
    else:
        mask = [[True]*n for _ in ARMS]; uniform_values = None
    actual, old = m.generate(kernel_case, library)
    retained = {p["participant"] for i, arm in enumerate(ARMS) for p, keep in
                zip([v for v in old["people"] if v["condition"] == arm], mask[i]) if keep}
    counts = {arm: sum(mask[i]) for i, arm in enumerate(ARMS)}
    rows = [r for r in actual["rows"] if r["participant"] in retained]
    generation = dict(schema="h10-power-design-generation-v1", namespace=NAMESPACE, case=case,
        generator="frozen_measure_generate_all_recruited_then_whole_person_mask", kernelGeneration=old,
        populationCounts=counts, recruitedCounts={a: n for a in ARMS}, retentionMask=dict(zip(ARMS, mask)),
        retentionUniforms=uniform_values, retentionRNG=dict(bitGenerator="PCG64", seed=case["retentionSeed"],
            draw="random((3,recruitedPerArm))" if uniform_values is not None else "none_fixed_eligible"),
        recruitedParticipantIDs=[p["participant"] for p in old["people"]],
        retainedParticipantIDs=[p["participant"] for p in old["people"] if p["participant"] in retained],
        retainedRowIndices=[i for i, row in enumerate(actual["rows"]) if row["participant"] in retained],
        eligibility="independent_whole_person_MCAR_complete_ten_not_trial_dropout", responseTransform="none",
        execution=execution())
    return dict(model="H10", reference="C2", rows=rows), generation

def support(actual, counts):
    np = kernels()[3]
    rows = actual["rows"]
    X = np.array([[v for arm in ARMS for v in ((1., r["need_A"], r["need_B"])
                  if r["condition"] == arm else (0., 0., 0.))] for r in rows], dtype=float).reshape((-1, 9))
    norms = np.linalg.norm(X, axis=0)
    singular = np.linalg.svd(X/np.where(norms > 0, norms, 1.), compute_uv=False) if len(rows) else np.zeros(9)
    singular = np.pad(singular, (0, 9-len(singular)))
    tau = max(X.shape)*np.finfo(float).eps*singular[0]
    rank = int(np.sum(singular > tau))
    arms = {a: {"participants": len({r["participant"] for r in rows if r["condition"] == a}),
                "rows": sum(r["condition"] == a for r in rows)} for a in ARMS}
    require(all(arms[a] == {"participants": counts[a], "rows": 10*counts[a]} for a in ARMS), "Changed actual retained population")
    people = {}
    for row in rows:
        person = people.setdefault(row["participant"], [row["condition"], 0])
        require(person[0] == row["condition"], "Participant crosses condition")
        person[1] += 1
    require(all(v[1] == 10 for v in people.values()), "Incomplete person in actual population")
    eligible = rank == 9 and all(counts[a] > 0 for a in ARMS)
    return dict(arms=arms, rank=rank, columns=9, scaledSingularValues=singular.tolist(),
                rankTolerance=float(tau), fitEligible=eligible, reason=None if eligible else "design_support_unavailable")

def reduce_rows(rows, manifest):
    """Use original branch events; only scientific roster/denominator admission is new."""
    _, hi, f, _ = kernels()
    states = {c["caseID"]: f.empty_state(c) for c in manifest["cells"]}
    for state in states.values():
        state["population"] = {a: [] for a in ARMS}
        state["origins"] = Counter()
        for method in ("baseline", "candidate"):
            state["methods"][method]["pointwise"] = [{key: Counter() for key in CONTRASTS} for _ in TAILS]
    for row in rows:
        expected = case_for(row["ordinal"], manifest["B"]); state = states[expected["caseID"]]; rep = expected["repID"]
        require(row["case"] == expected and row["caseID"] == expected["caseID"] and row["repID"] == rep
                and rep not in state["present"], "Duplicate or changed aggregate identity")
        state["present"].add(rep)
        state["origins"][row["origin"]] += 1
        for arm in ARMS:
            state["population"][arm].append(row["populationCounts"][arm])
        state["guards"][str((row["diagnostics"] or {}).get("usableCandidateInference")).lower()] += 1
        if row["technicalCompleted"] is False:
            require(row["candidateBranches"] is None and isinstance(row["technicalError"], dict), "Technical unknown became nonrejection")
            state["errors"][row["technicalError"]["phase"]] += 1
            continue
        require(row["technicalCompleted"] is True and row["technicalError"] is None, "Unknown technical classification")
        state["completed"].add(rep)
        for method in ("baseline", "candidate"):
            branches = row[method+"Branches"]
            require(len(branches) == 2, "Missing tail branch")
            for j, mode in enumerate(TAILS):
                branch = branches[j]; counts = branch["counts"]
                require(branch["tailMode"] == mode and branch["trueNullIDs"] == expected["trueNullIDs"][mode]
                        and counts["M"] == counts["T"] == 1 and counts["technicalMissing"] == 0
                        and all(type(counts[k]) is int and counts[k] in (0, 1) for k in f.EVENTS), "Changed event semantics")
                available = sum(v["status"] == "available_candidate" for v in branch["tests"].values())
                require(set(branch["tests"]) == set(CONTRASTS) and bool(counts["A4"]) == (available == 4), "Changed family")
                state["methods"][method]["branches"][j]["counts"].update({k: counts[k] for k in f.EVENTS})
                state["methods"][method]["branches"][j]["partial"] += int(0 < available < 4)
                for key in CONTRASTS:
                    test = branch["tests"][key]
                    a = test["status"] == "available_candidate"
                    state["methods"][method]["pointwise"][j][key].update(
                        A=int(a), H=int(test["holmReject"]), nominal=int(test["candidateNominalReject"]))
            for key in CONTRASTS:
                coverage = branches[0]["coverage"][key]
                require(coverage["truth"] == expected["delta"][key], "Changed truth")
                state["methods"][method]["coverage"][key].update({k: coverage[k] for k in ("A", "CIAvailable", "C")})
        for j, paired in enumerate(row["paired"]):
            require(paired["tailMode"] == TAILS[j], "Changed paired tail")
            for name in f.TRANSITIONS:
                value = paired["transitions"][name]
                state["paired"][j]["transitions"][name][f.transition_key(value["baseline"], value["candidate"])] += 1
            for key in CONTRASTS:
                values = paired["contrasts"][key]
                for name, a, b in (("coverage", "baselineCovered", "candidateCovered"),
                                  ("availability", "baselineAvailable", "candidateAvailable"),
                                  ("rejection", "baselineReject", "candidateReject")):
                    state["paired"][j]["contrasts"][key][name][f.transition_key(values[a], values[b])] += 1
        for key in CONTRASTS:
            for name in f.METRICS:
                state["diagnostics"][key][name].append(row["paired"][0]["contrasts"][key][name])
            state["diagnostics"][key]["oracleSE"].append(
                (((row["candidateMetadata"] or {}).get("designDiagnostics") or {}).get("oracle") or {}).get("contrastSE", {}).get(key))
    reports = []
    for cell in manifest["cells"]:
        state = states[cell["caseID"]]; T = len(state["completed"]); B = manifest["B"]
        report = dict(caseID=cell["caseID"], cellIndex=cell["cellIndex"], cell=cell, M=B, T=T,
            present=len(state["present"]), complete=T == B, technicalMissing=B-T,
            missingRepIDs=[r for r in range(1, B+1) if r not in state["present"]],
            unresolvedRepIDs=sorted(state["present"]-state["completed"]),
            baseline=f.method_report(state, "baseline", planned=B), candidate=f.method_report(state, "candidate", planned=B),
            originalGuardCounts=dict(state["guards"]), technicalFailureReasons=dict(state["errors"]))
        for method in ("baseline", "candidate"):
            report[method]["contrastRejection"] = []
            for j, mode in enumerate(TAILS):
                values = {}
                for key in CONTRASTS:
                    c = state["methods"][method]["pointwise"][j][key]
                    values[key] = dict(counts={name: c[name] for name in ("A", "H", "nominal")},
                        holmAll=hi.rate(c["H"], B) if T == B else None,
                        holmCompleted=hi.rate(c["H"], T), holmAvailable=hi.rate(c["H"], c["A"]),
                        nominalAll=hi.rate(c["nominal"], B) if T == B else None)
                report[method]["contrastRejection"].append(dict(tailMode=mode, contrasts=values))
        table = lambda values: {k: values[k] for k in ("00", "01", "10", "11", "unknown")}
        report["paired"] = [dict(tailMode=mode, denominator=T,
            transitions={name: table(value) for name, value in state["paired"][j]["transitions"].items()},
            contrasts={key: {name: table(value) for name, value in fields.items()}
                       for key, fields in state["paired"][j]["contrasts"].items()}) for j, mode in enumerate(TAILS)]
        report["population"] = {a: hi.distribution(state["population"][a]) for a in ARMS}
        report["sourceOrigins"] = dict(state["origins"])
        report["contrastDiagnostics"] = {key: {name: hi.distribution(values) for name, values in fields.items()}
                                          for key, fields in state["diagnostics"].items()}
        # Preserve pointwise calibration diagnostics as descriptive boundaries;
        # they do not adopt a method or gate the fixed power roster.
        report["pointwiseCalibrationDiagnostic"] = dict(
            applicability="prespecified_random_population_controls" if cell["role"] == "new_population_control" else "not_applicable_power_or_N_endpoint",
            reports=f.practical_target(report) if cell["role"] == "new_population_control" else None,
            interpretation="descriptive_pointwise_control_limits_not_method_adoption_or_power_gate")
        reports.append(report)
    return reports

def grid_description(reports):
    """Finite pointwise descriptions only: no interpolation, extension or selection guarantee."""
    groups = []
    for view in ("initial_amount", "terminal_amount"):
        for scenario in (5, 6):
            for population in ("fixed154", "recruit193_mcar08"):
                selected = sorted((r for r in reports if r["cell"]["view"] == view and
                    r["cell"]["scenarioIndex"] == scenario and r["cell"]["populationRegime"] == population),
                    key=lambda r: r["cell"]["multiplier"])
                for method in ("baseline", "candidate"):
                    for j, tail in enumerate(TAILS):
                        points = []
                        for row in selected:
                            rates = row[method]["branches"][j]["rates"]
                            d = rates["alternativeRejectionAll"]
                            status = "incomplete_or_not_applicable" if d is None else (
                                "pointwise_supported" if d["interval95"][0] >= .8 else
                                "pointwise_below" if d["interval95"][1] < .8 else "uncertain")
                            points.append(dict(caseID=row["caseID"], multiplier=row["cell"]["multiplier"],
                                anyRejection=rates["anyRejectionAll"], correctAlternativeDiscovery=d, status=status))
                        supported = [p["multiplier"] for p in points if p["status"] == "pointwise_supported"]
                        groups.append(dict(view=view, scenarioIndex=scenario, populationRegime=population,
                            method=method, tailMode=tail, points=points,
                            firstPointwiseSupportedMultiplier=min(supported) if supported else None,
                            gridStatus="descriptive_tested_grid_threshold" if supported else
                            "right_censored_not_established_within_tested_grid" if all(p["status"] == "pointwise_below" for p in points)
                            else "uncertain_or_incomplete",
                            interpretation="correct_alternative_discovery_grid_not_continuous_MDE_or_simultaneous_selection"))
    endpoints = [dict(view=v, scenarioIndex=s, fixed154=next(r["caseID"] for r in reports if
        r["cell"]["view"] == v and r["cell"]["scenarioIndex"] == s and r["cell"]["populationRegime"] == "fixed154" and r["cell"]["multiplier"] == 1.),
        fixed193=next(r["caseID"] for r in reports if r["cell"]["view"] == v and r["cell"]["scenarioIndex"] == s and r["cell"]["populationRegime"] == "fixed193"),
        interpretation="two_effective_N_endpoints_not_a_minimum_N_or_recruitment_guarantee")
        for v in ("initial_amount", "terminal_amount") for s in (5, 6)]
    return dict(effectGrid=groups, effectiveNEndpoints=endpoints, pointwiseOnly=True, monotonicityAssumed=False,
                target=.8, humanPowerClaim=False, historicalShiftReuseGateOverridden=False)
