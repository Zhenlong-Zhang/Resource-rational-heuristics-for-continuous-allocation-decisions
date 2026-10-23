"""joint generation: accepted numerical definitions, portable source projection."""
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

from .mixed_tests import H10_IDS, directional_result, holm_four

NAMESPACE = "h10-gaussian-reference-v1-20261019"

VIEWS = ("initial_percent", "initial_amount", "terminal_percent", "terminal_amount")

SCENARIOS = ("common_slope_null", "partial_boundary", "direction_boundary", "heterogeneous_noise_null",
             "sign_consistent_alternative", "literal_alternative")

ARMS = ("C1", "C2", "C3")

TAILS = ("literal", "sign_consistent")

def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def pin(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Absolute path required")
    return dict(path=str(path), sha256=sha256(path))

def check_pin(value, observed):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or pin(value["path"]) != value:
        raise ValueError("Changed bound file")
    if observed.setdefault(value["path"], value["sha256"]) != value["sha256"]:
        raise ValueError("Contradictory file pins")
    return Path(value["path"])

def closure(observed):
    for path, digest in observed.items():
        if sha256(path) != digest:
            raise ValueError(f"Source changed at closure: {path}")
    return True

def write_json(value, path, *, identical_ok=False):
    path = Path(path)
    if path.exists():
        if identical_ok and read_json(path) == value:
            return pin(path)
        raise ValueError(f"Refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".uncommitted-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return pin(path)

def tail_for(mode, contrast):
    return "greater" if mode == "sign_consistent" and contrast.endswith(":B") else "less"

def null_ids(delta, mode):
    return [key for key in H10_IDS
            if (delta[key] <= 0 if tail_for(mode, key) == "greater" else delta[key] >= 0)]

def measurement_plan():
    cases = []
    for vi, view in enumerate(VIEWS):
        percent = view.endswith("percent")
        d = .10 if percent else .15
        vectors = ((0., 0., 0., 0.), (0., d, -d, 0.), (d, 0., 0., -d),
                   (0., 0., 0., 0.), (-d, d, -d, d), (-d, -d, -d, -d))
        for si, (scenario, values) in enumerate(zip(SCENARIOS, vectors), 1):
            index = 6 * vi + si
            a, b = (.20, -.20) if percent else (.25, -.25)
            sigma = 10. if percent else 12.
            delta = dict(zip(H10_IDS, values))
            parameters = dict(alpha=50. if percent else 60., sdU=6. if percent else 8.,
                sigmaByArm=dict(zip(ARMS, [2*sigma, sigma, 1.5*sigma] if si == 4 else [sigma]*3)),
                slopes={"C1": dict(A=a, B=b), "C2": dict(A=a-values[0], B=b-values[1]),
                        "C3": dict(A=a-values[2], B=b-values[3])}, deltaMagnitude=d)
            cases.append(dict(caseID=f"h10m1_{view}_s{si}_n154", caseIndex=index, view=view,
                scenarioID=scenario, scenarioIndex=si, N=154, repID=1,
                designSeed=202610190+10000*si+1, responseSeed=302610190+10000*index+1,
                parameters=parameters, delta=delta, trueNullIDs={t: null_ids(delta, t) for t in TAILS},
                futureProposedB=2000 if si <= 4 else 1000,
                dgpClass="heteroscedastic_stress" if si == 4 else "conditional_gaussian_model_reference"))
    return dict(namespace=NAMESPACE, measurementReplicatesPerCase=1, NPerArm=154,
                views=list(VIEWS), scenarios=list(SCENARIOS), cases=cases,
                futureProposedFits=40000, futureProductionAuthorized=False,
                scope="design_and_target_N_measurement_not_calibration_or_power_estimation")

def schedule_library(canonical):
    rows = canonical["rows"]
    if len(rows) != 270 or {r["condition"] for r in rows} != set(ARMS):
        raise ValueError("Wrong accepted native design population")
    library = {}
    for arm in ARMS:
        schedules = []
        for variant in range(9):
            participant = f"h10-alternatives-{arm}-v{variant}"
            own = sorted((r for r in rows if r["participant"] == participant), key=lambda r: r["mainIndex"])
            if len(own) != 10 or [r["mainIndex"] for r in own] != list(range(1, 11)) or any(r["condition"] != arm for r in own):
                raise ValueError("Changed whole-person native schedule")
            for row in own:
                for key in ("meanAInitial", "meanBInitial", "meanATerminal", "meanBTerminal"):
                    if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                        raise ValueError("Nonfinite exact design covariate")
            schedules.append(own)
        library[arm] = schedules
    return library

def generate(case, library):
    if np.__version__ != "1.23.5":
        raise ValueError("Expected NumPy 1.23.5 for the frozen PCG64 stream")
    n = case["N"]
    design_rng = np.random.Generator(np.random.PCG64(case["designSeed"]))
    indices = design_rng.integers(0, 9, size=(3, n), dtype=np.int64)
    response_rng = np.random.Generator(np.random.PCG64(case["responseSeed"]))
    intercept_z = response_rng.standard_normal(3*n)
    residual_z = response_rng.standard_normal((3*n, 10))
    parameters, suffix = case["parameters"], case["view"].split("_")[0].title()
    people, rows = [], []
    for ai, arm in enumerate(ARMS):
        for pi in range(n):
            ordinal = ai*n + pi
            source = library[arm][int(indices[ai, pi])]
            participant = f"{NAMESPACE}:{case['caseID']}:r{case['repID']}:{arm}:p{pi+1:03d}"
            u = parameters["sdU"] * float(intercept_z[ordinal])
            eps = [parameters["sigmaByArm"][arm] * float(z) for z in residual_z[ordinal]]
            people.append(dict(participant=participant, condition=arm, personIndex=pi+1,
                scheduleIndex=int(indices[ai, pi]), sourceParticipant=source[0]["participant"],
                sourceSession=source[0]["session"], sourceMainIndices=[r["mainIndex"] for r in source],
                interceptZ=float(intercept_z[ordinal]), residualZ=residual_z[ordinal].tolist(),
                randomIntercept=u, residuals=eps))
            for ti, original in enumerate(source):
                a, b = original[f"meanA{suffix}"], original[f"meanB{suffix}"]
                mean = parameters["alpha"] + parameters["slopes"][arm]["A"] * a + parameters["slopes"][arm]["B"] * b
                y = mean + u + eps[ti]
                rows.append(dict(participant=participant, condition=arm, need_A=a, need_B=b, y=y))
    generation = dict(schema="h10-calibration-measure-generation-v1", namespace=NAMESPACE, case=case,
        rng=dict(numpyVersion=np.__version__, bitGenerator="PCG64", designSeed=case["designSeed"],
                 responseSeed=case["responseSeed"], designDraw="integers(0,9,size=(3,N),dtype=int64)",
                 responseDrawOrder=["standard_normal(3*N)", "standard_normal((3*N,10))"]),
        people=people, responseTransform="none", clippedCount=0, roundedCount=0,
        sourceRole="accepted_native_joint_X_only", responseRole="new_conditional_gaussian_reference",
        rowOrder="C1,C2,C3; person 1..N; source mainIndex 1..10")
    return dict(model="H10", reference="C2", rows=rows), generation

def support(request):
    rows = request["rows"]
    matrix = np.array([[v for arm in ARMS for v in
        ((1., r["need_A"], r["need_B"]) if r["condition"] == arm else (0., 0., 0.))] for r in rows])
    norms = np.linalg.norm(matrix, axis=0)
    singular = np.linalg.svd(matrix / np.where(norms > 0, norms, 1.), compute_uv=False)
    rank = int(np.sum(singular > max(matrix.shape) * np.finfo(float).eps * singular[0]))
    arms = {arm: dict(participants=len({r["participant"] for r in rows if r["condition"] == arm}),
                     rows=sum(r["condition"] == arm for r in rows)) for arm in ARMS}
    eligible = rank == 9 and all(a["participants"] == 154 and a["rows"] == 1540 for a in arms.values())
    return dict(arms=arms, rank=rank, columns=9, scaledSingularValues=singular.tolist(),
                fitEligible=eligible, reason=None if eligible else "design_support_unavailable")

def branch_result(case, mode, raw):
    tests = {}
    for key in H10_IDS:
        tail = tail_for(mode, key)
        if raw["status"] == "fitted":
            reference = raw["references"][key]
            tests[key] = dict(directional_result(reference, tail, raw["diagnostics"]["usableCandidateInference"] is True), reference=reference)
        else:
            tests[key] = dict(tail=tail, status="unavailable", reason=raw["reason"], p=None,
                              candidateNominalReject=False, reference=None)
    family = holm_four(tests)
    family.pop("allFourCandidateContrastsReject")
    family.pop("claim")
    nulls = case["trueNullIDs"][mode]
    a4 = all(t["status"] == "available_candidate" for t in tests.values())
    false_rejection = any(family["tests"][key]["holmReject"] for key in nulls)
    alt_rejection = any(family["tests"][key]["holmReject"] for key in H10_IDS if key not in nulls)
    coverage = {}
    for key, test in tests.items():
        ci = test["reference"].get("CI") if test["reference"] else None
        valid = (test["status"] == "available_candidate" and isinstance(ci, list) and len(ci) == 2
                 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in ci) and ci[0] <= ci[1])
        covered = (ci[0] <= case["delta"][key] <= ci[1]) if valid else None
        available = int(test["status"] == "available_candidate")
        coverage[key] = dict(A=available, CIAvailable=int(valid), C=int(covered is True),
            covered=(ci[0] <= case["delta"][key] <= ci[1]) if valid else None,
            truth=case["delta"][key], coverageAvailable=int(covered is True) if available else None,
            coverageAll=int(covered is True), availabilityAll=available,
            interpretation="conditional_mean_slope_difference")
    counts = dict(M=1, T=1, A4=int(a4), F=int(false_rejection), F4=int(a4 and false_rejection),
                  R=int(family["anyCandidateContrastRejects"]), R4=int(a4 and family["anyCandidateContrastRejects"]),
                  D=int(alt_rejection), D4=int(a4 and alt_rejection), technicalMissing=0)
    return dict(tailMode=mode, trueNullIDs=nulls, falseNullIDs=[k for k in H10_IDS if k not in nulls],
        **family, completeFamilyAvailable=a4, counts=counts,
        fwerApplicability="applicable" if nulls else "not_applicable_no_true_null",
        rates=dict(availabilityAll=int(a4), falseRejectionAll=int(false_rejection) if nulls else None,
                   falseRejectionAvailable=int(false_rejection) if a4 and nulls else None,
                   anyRejectionAll=int(family["anyCandidateContrastRejects"]),
                   anyRejectionAvailable=int(family["anyCandidateContrastRejects"]) if a4 else None,
                   alternativeRejectionAll=int(alt_rejection) if len(nulls) < 4 else None,
                   alternativeRejectionAvailable=int(alt_rejection) if a4 and len(nulls) < 4 else None), coverage=coverage)
