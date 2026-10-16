"""Prospective source contexts and native binary projection tables.

Only deterministic preparation is defined here initially. Statistical draws are
separately gated; prior model, recipe, raw solver and primary fit are read-only.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np

from design_validation.bounded_task import belief_moments, expand_templates, initial_log_beliefs
from .empirical_bernstein import (ATOL, RTOL, METHOD, attest_initial_rows, source_slope_domain,
                                 _schedule_reason)
from .inference import CONDITIONS, CORE, POLICIES, participant_slope

CELL_IDS = ("eb_null80", "eb_null320", "eb_null1280", "eb_weak80", "eb_weak320", "eb_weak1280",
    "eb_mixed320", "eb_skew_null320", "eb_composite80", "eb_unequal_mcar_learning320",
    "eb_mnar_selected320", "eb_constant_boundary80")
KEYS = {"anyQuestion": "Q", "EOResemblance": "R", "lowerNeedDirectional": "L", "slope_initial": "slope"}
PROSPECTIVE_CONFIG_OBJECT_SHA256 = "6de0337c4b4c64be99dda8b6034b9d4928a75f4f5cfcde4b7c7054284d21826f"


def validate_config(config):
    # Seal ALL literal constants, including hardcoded generator/projection formulas.
    # Whitespace/key order in the JSON file is immaterial; meaningful edits require
    # a prospective new config/source gate, not ignored fields in a bound receipt.
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != PROSPECTIVE_CONFIG_OBJECT_SHA256:
        raise ValueError("Frozen literal config object changed; obtain a prospective new gate")
    if config.get("schema") != "tutoring-identifiability-variance-adaptive-v1" or config.get("method") != METHOD or config.get("primaryReplacement") is not False or config.get("automaticCandidateAdoption") is not False:
        raise ValueError("Wrong explicit optional-method scope")
    if config.get("policies") != list(POLICIES) or config.get("alpha") != .05 or config.get("confidence") != .95:
        raise ValueError("Frozen policies/tails changed")
    if config.get("batchSize") != 250 or config.get("replicationsPerCell") != 10000 or config.get("attempts") != 120000:
        raise ValueError("Frozen execution counts changed")
    if [r.get("id") for r in config["cells"]] != list(CELL_IDS) or [r.get("seed") for r in config["cells"]] != list(range(720071, 720083)):
        raise ValueError("Frozen cell/seed order changed")
    if config["sourceAttestation"]["absoluteTolerance"] != ATOL or config["sourceAttestation"]["relativeTolerance"] != RTOL:
        raise ValueError("Source attestation predicate changed")
    if config["projectionSchedules"]["quadraturePoints"] != 6 or config["projectionSchedules"]["allocationPatternCount"] != 1024:
        raise ValueError("Frozen projection table/integration changed")
    cells = []
    for own in config["cells"]:
        cell = {**config["defaults"], **own}
        n = cell["people"]
        cell["people"] = {c: n for c in CONDITIONS} if isinstance(n, int) and not isinstance(n, bool) else dict(n)
        if set(cell["people"]) != set(CONDITIONS) or any(not isinstance(v, int) or isinstance(v, bool) or v < 2 for v in cell["people"].values()):
            raise ValueError("Positive fixed person counts >=2 required")
        if cell["generator"] not in ("uniform", "shared", "skew", "constant"):
            raise ValueError("Unknown contracted generator")
        for key in ("participantRetention", "recordRetention", "endpointRetention"):
            v = cell[key]
            if isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1: raise ValueError("Invalid retention")
        cell["parameters"] = config["effectRecipes"][cell["effects"]]
        cells.append(cell)
    return cells


def source_call(mode, design, spec, node="node"):
    helper = Path(__file__).with_name("source_projection_panel.cjs")
    process = subprocess.run([node, str(helper)], input=json.dumps({"mode": mode, "design": design, "spec": spec}),
                             text=True, capture_output=True, check=False)
    if process.returncode: raise ValueError("Source engine helper failed: "+process.stdout[:500]+process.stderr[:500])
    data = json.loads(process.stdout)
    if not data.get("ok"): raise ValueError("Source engine helper rejected request: "+str(data.get("error")))
    return data["result"]


def schedule_rows(definitions, design, cid, identity):
    configs = {c.template_id: c for c in expand_templates(design)}
    rows = []
    for definition in definitions:
        config = configs[definition["templateId"]]
        means = [belief_moments(log)[0] for log in initial_log_beliefs(config)]
        if definition["sideReversed"]: means.reverse()
        gap = means[0]-means[1]; budget = config.total_millihours/1000
        rows.append({"participantId": identity, "sessionId": identity+"-session", "conditionId": cid,
            "trialId": definition["trialId"], "stage": "main", "stageIndex": definition["stageIndex"],
            "scheduleIndex": definition["scheduleIndex"], "sideReversed": definition["sideReversed"],
            "templateId": definition["templateId"], "scale": config.scale, "gapOffset": config.gap_offset,
            "initialBudget": budget, "meanAInitial": means[0], "meanBInitial": means[1],
            "initialNeedGapRaw": gap, "initialNeedGap": gap, "initialTie": False,
            "xInitial": gap/budget, "shareA": 0., "analysisEligible": True})
    return rows


def coefficient_table(rows, domain):
    x = np.asarray([r["xInitial"] for r in rows]); centered = x-x.mean(); sxx = float(centered@centered)
    patterns = ((np.arange(1024)[:, None] >> np.arange(10)) & 1).astype(float)
    raw = np.empty(1024)
    for i, shares in enumerate(patterns):
        own = [{**r, "shareA": float(shares[j])} for j, r in enumerate(rows)]
        fitted = participant_slope(own, "xInitial")
        if fitted["status"] != "available": raise ValueError("Source corner projection unexpectedly unsupported")
        raw[i] = fitted["value"]
    algebra = patterns@centered/sxx
    discrepancy = float(np.max(np.abs(raw-algebra)))
    if discrepancy > 1e-10 or np.any(raw < domain["numericBounds"][0]) or np.any(raw > domain["numericBounds"][1]):
        raise ValueError("Deterministic raw corner/domain parity failure")
    return {"x": x.tolist(), "order": ((np.arange(1, 11)-5.5)/9).tolist(), "Sxx": sxx,
        "rawCoefficients": raw.tolist(), "maximumRawAlgebraDifference": discrepancy,
        "rawCoefficientRange": [float(raw.min()), float(raw.max())], "patterns": 1024,
        "zeroPatternCoefficient": float(raw[0]), "allPatternCoefficient": float(raw[-1]),
        "oldSolverModified": False, "universalMachineCertificate": False}


def projection_truth(table, beta, intercept, learning, selected=False, constant=False):
    if constant:
        return {"ideal": 0., "raw": table["zeroPatternCoefficient"], "difference": table["zeroPatternCoefficient"],
                "probabilityRange": [0., 0.], "normalizationError": 0., "constantNativeShare": 0}
    x, order = np.asarray(table["x"]), np.asarray(table["order"])
    left, right = (.75, 1.) if selected else (0., 1.)
    nodes, quadrature = np.polynomial.legendre.leggauss(6)
    nodes = (left+right)/2+(right-left)*nodes/2
    # Uniform expectation on the selected interval: transformed weights / interval width.
    quadrature = quadrature/2
    extremes = np.asarray([intercept+(beta+.4*(u-.5))*x+learning*order for u in (left, right)])
    if np.min(extremes) < 0 or np.max(extremes) > 1: raise ValueError("Affine source-share generator leaves probability domain before draws")
    patterns = ((np.arange(1024)[:, None] >> np.arange(10)) & 1).astype(bool)
    raw, norm_error = 0., 0.
    for u, weight in zip(nodes, quadrature):
        probability = intercept+(beta+.4*(u-.5))*x+learning*order
        law = np.prod(np.where(patterns, probability, 1-probability), axis=1)
        norm_error = max(norm_error, abs(math.fsum(law)-1))
        raw += float(weight)*float(law@np.asarray(table["rawCoefficients"]))
    centered = x-x.mean()
    ideal = beta+.4*((left+right)/2-.5)+learning*float(centered@order/table["Sxx"])
    if norm_error > 1e-12 or not math.isclose(raw, ideal, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("Raw table integration/ideal projection truth parity failed before draws")
    return {"ideal": ideal, "raw": raw, "difference": raw-ideal,
        "probabilityRange": [float(extremes.min()), float(extremes.max())], "normalizationError": norm_error,
        "integration": "6point_Gauss_Legendre_degree10_exact_real_approximate_binary64", "rawTruthSnapped": False}


def cell_truths(cell, tables):
    p = cell["parameters"]; original, selected = {}, {}; projections = {}
    for i, cid in enumerate(CONDITIONS):
        original[cid] = {k: float(p[k][i]) for k in ("Q", "R", "L")}
        selected[cid] = dict(original[cid])
        if cell["selection"]:
            selected[cid] = {"Q": max(0., min(p["Q"][i]-.75, .25))/.25,
                "R": p["R"][i]+15., "L": p["L"][i]+.225}
        if cid in tables:
            beta = p["beta"][("C2", "C3", "C4").index(cid)]
            h = .2 if cell["generator"] == "skew" else .5
            own = projection_truth(tables[cid], beta, h, cell["learning"], False, cell["generator"] == "constant")
            retained = projection_truth(tables[cid], beta, h, cell["learning"], bool(cell["selection"]), cell["generator"] == "constant")
            original[cid]["slope"] = own["raw"]; selected[cid]["slope"] = retained["raw"]
            projections[cid] = {"original": own, "selected": retained}
    result = {"original": original, "selected": selected, "projections": projections}
    for population in ("original", "selected"):
        raw_values, ideal_values = [], []
        for _, _, endpoint, weights, _ in CORE:
            key = KEYS[endpoint]
            raw_values.append(math.fsum(w*result[population][cid][key] for cid, w in weights.items()))
            ideal_values.append(math.fsum(w*(projections[cid][population]["ideal"] if key == "slope" else result[population][cid][key]) for cid, w in weights.items()))
        signed_raw = [v if CORE[j][4] == "greater" else -v for j, v in enumerate(raw_values)]
        signed_ideal = [v if CORE[j][4] == "greater" else -v for j, v in enumerate(ideal_values)]
        result[population+"Core"] = {"rawTruth": raw_values, "idealTruth": ideal_values,
            "rawSignedTruth": signed_raw, "idealSignedTruth": signed_ideal,
            "rawNullRoster": [CORE[j][0] for j, v in enumerate(signed_raw) if v <= 0],
            "idealNullRoster": [CORE[j][0] for j, v in enumerate(signed_ideal) if v <= 0],
            "rawAlternativeRoster": [CORE[j][0] for j, v in enumerate(signed_raw) if v > 0],
            "nearZeroRawAlternativeIsNotExactNull": [CORE[j][0] for j, v in enumerate(signed_raw) if 0 < v < 1e-10]}
    return result


def prepare_context(config, design, node="node"):
    cells = validate_config(config); domain = source_slope_domain(design)
    schedules = source_call("schedules", design, config["projectionSchedules"], node)
    tables, rows = {}, {}
    for own in schedules:
        cid = own["conditionId"]
        main = schedule_rows(own["definitions"], design, cid, own["options"]["participantId"])
        main = attest_initial_rows(main, domain)
        reason = _schedule_reason(main)
        if reason: raise ValueError("Frozen source schedule is illegal: "+reason)
        rows[cid] = main; tables[cid] = coefficient_table(main, domain)
    for cell in cells:
        cell["truths"] = cell_truths(cell, tables)
        # Validate artificial mean generators against known domains, not extrema from draws.
        for i, cid in enumerate(CONDITIONS):
            r, l = cell["parameters"]["R"][i], cell["parameters"]["L"][i]
            if cell["generator"] == "constant": continue
            extremes = (r-30/11, r+300/11, l-.3/11, l+3/11) if cell["generator"] == "skew" else (r-20, r+20, l-.3, l+.3)
            if extremes[0] < -50 or extremes[1] > 50 or extremes[2] < -.5 or extremes[3] > .5:
                raise ValueError("Abstract mean generator leaves known source range before draws")
    return {"schema": "variance-adaptive-deterministic-context-v1", "domain": domain,
        "scheduleDefinitions": schedules, "scheduleRows": rows, "projectionTables": tables,
        "cells": cells, "statisticalCalibrationAttempts": 0, "modelObservationDraws": 0,
        "fixedSourceScheduleSeededRNGUsed": True, "cornerCoefficients": 3072,
        "scope": "source schedules and analytic pattern laws; not stochastic or human evidence"}


def generate_batch(cell, batch_index, context, batch_size=250):
    """Independent persons/arms, fixed pattern table estimator, never task policies."""
    if not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 0 or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Invalid deterministic batch identity/size")
    rng = np.random.default_rng(np.random.SeedSequence([cell["seed"], batch_index]))
    groups, raw = {}, {}
    for arm_index, cid in enumerate(CONDITIONS):
        n = cell["people"][cid]; shape = (batch_size, n)
        mu = cell["truths"]["original"][cid]; selected = cell["truths"]["selected"][cid]
        kind = cell["generator"]
        slope_u = rng.random(shape)
        if kind == "shared":
            u = slope_u; q = (u < mu["Q"]).astype(float)
            r, l = mu["R"]+40*(u-.5), mu["L"]+.6*(u-.5)
        elif kind == "skew":
            u = rng.beta(.5, 5., shape); q = rng.binomial(10, mu["Q"], shape)/10
            r, l = mu["R"]+30*(u-1/11), mu["L"]+.3*(rng.beta(.5, 5., shape)-1/11)
        elif kind == "constant":
            u = rng.random(shape)
            q, r, l = (np.full(shape, mu[k]) for k in ("Q", "R", "L"))
        else:
            u = rng.random(shape); q = rng.binomial(10, mu["Q"], shape)/10
            r, l = mu["R"]+40*(u-.5), mu["L"]+.6*(rng.random(shape)-.5)
        record_mask = rng.random((*shape, 10)) < cell["recordRetention"]
        person_keep = (rng.random(shape) < cell["participantRetention"]) & np.all(record_mask, axis=-1)
        if cell["selection"]: person_keep &= slope_u > .75
        groups[cid] = {}
        for key, values in (("Q", q), ("R", r), ("L", l)):
            available = person_keep & (rng.random(shape) < cell["endpointRetention"])
            groups[cid][key] = {"values": np.where(available, values, np.nan),
                "truth": np.broadcast_to(mu[key], shape), "selectedTruth": np.broadcast_to(selected[key], shape),
                "available": available}
        own = {"recordCounts": np.sum(record_mask, axis=-1), "includedPeople": person_keep,
               "slopeU": slope_u, "recordMask": record_mask}
        if cid in context["projectionTables"]:
            table = context["projectionTables"][cid]
            x, order = np.asarray(table["x"]), np.asarray(table["order"])
            beta = cell["parameters"]["beta"][("C2", "C3", "C4").index(cid)]
            if kind == "constant": pattern = np.zeros(shape, dtype=np.int16)
            else:
                intercept = .2 if kind == "skew" else .5
                probabilities = intercept+(beta+.4*(slope_u[..., None]-.5))*x+cell["learning"]*order
                if np.any(probabilities < 0) or np.any(probabilities > 1): raise ValueError("Preverified source-share probability domain violated")
                allocations = rng.random((*shape, 10)) < probabilities
                pattern = np.sum(allocations*(1 << np.arange(10)), axis=-1).astype(np.int16)
            share_mask = rng.random((*shape, 10)) < cell["endpointRetention"]
            slope_available = person_keep & np.all(share_mask, axis=-1)
            coefficients = np.asarray(table["rawCoefficients"])[pattern]
            lo, hi = context["domain"]["numericBounds"]
            if np.any(coefficients < lo) or np.any(coefficients > hi):
                raise ValueError("Supported raw projection outside domain: HALT, never select by outcome")
            groups[cid]["slope"] = {"values": np.where(slope_available, coefficients, np.nan),
                "truth": np.broadcast_to(mu["slope"], shape), "selectedTruth": np.broadcast_to(selected["slope"], shape),
                "available": slope_available}
            own.update(patternIndices=pattern, shareMask=share_mask, slopeAvailable=slope_available)
        raw[cid] = own
    return {"cell": cell["id"], "batchIndex": batch_index, "attempts": batch_size,
            "groups": groups, "raw": raw, "samplingUnit": "independent_person_each_arm"}


def _vector_moments(group):
    values = group["values"]; valid = ~np.isnan(values)
    if values.ndim != 2 or np.any(np.isinf(values)) or not np.array_equal(valid, group["available"]):
        raise ValueError("Invalid vector values/support masks")
    n = valid.sum(axis=1)
    mean = np.sum(np.where(valid, values, 0.), axis=1)/np.maximum(n, 1)
    variance = np.sum(np.where(valid, values-mean[:, None], 0.)**2, axis=1)/np.maximum(n-1, 1)
    return n, mean, variance


def vector_contrast(groups, key, weights, bounds, side, method="empirical_Bernstein", confidence=.95):
    batch = next(iter(groups.values()))["Q"]["values"].shape[0]
    estimate, A, B, V = (np.zeros(batch) for _ in range(4))
    estimable = np.ones(batch, bool); available = np.ones(batch, bool); counts = {}
    for cid, w in weights.items():
        group = groups[cid][key]; lo, hi = bounds[cid]
        values = group["values"]; observed = group["available"]
        if np.any(observed & ((values < lo) | (values > hi))): raise ValueError("Vector person endpoint outside fixed source range")
        n, mean, variance = _vector_moments(group); counts[cid] = n
        estimable &= n > 0; available &= n >= (2 if method == "empirical_Bernstein" else 1)
        estimate += w*mean
        A += abs(w)*np.sqrt(2*variance/np.maximum(n, 1))
        B += abs(w)*7*(hi-lo)/(3*np.maximum(n-1, 1))
        V += w*w*(hi-lo)**2/np.maximum(n, 1)
    legal_low = math.fsum(w*bounds[c][0 if w > 0 else 1] for c, w in weights.items())
    legal_high = math.fsum(w*bounds[c][1 if w > 0 else 0] for c, w in weights.items())
    signed = estimate if side == "greater" else -estimate
    if method == "empirical_Bernstein":
        positive = np.maximum(signed, 0.)
        # all-zero root has A=0 and positive=0; keep exact null p=1 without 0/0.
        root = np.divide(2*positive, A+np.hypot(A, 2*np.sqrt(B*positive)), out=np.zeros(batch), where=(positive>0)&(A>0))
        # Mirror scalar constant branch; B*t may underflow for a legal tiny range.
        constants = (A == 0) & (positive > 0)
        root[constants] = np.sqrt(positive[constants]/B[constants])
        logp = np.where(signed > 0, np.minimum(0., math.log(2*len(weights))-root**2), 0.)
        log_ci = math.log(4*len(weights))-math.log1p(-confidence)
        radius = A*math.sqrt(log_ci)+B*log_ci
    elif method == "Hoeffding":
        logp = -2*np.maximum(signed, 0.)**2/V
        radius = np.sqrt(V*(math.log(2)-math.log1p(-confidence))/2)
    else: raise ValueError("Unknown explicitly selected vector method")
    p = np.exp(logp); raw_low, raw_high = estimate-radius, estimate+radius
    result = {"estimate": estimate, "p": p, "logp": logp, "radius": radius,
        "rawLow": raw_low, "rawHigh": raw_high, "low": np.maximum(legal_low, raw_low),
        "high": np.minimum(legal_high, raw_high), "A": A, "B": B, "V": V}
    if any(np.any(available & ~np.isfinite(v)) for v in result.values()):
        raise ValueError("Unexpected supported numeric vector failure")
    return {**{k: np.where(estimable if k == "estimate" else available, v, np.nan) for k, v in result.items()},
            "available": available, "counts": counts}


def core_statistics(batch, context):
    from .bounded_inference import BOUNDS
    from .calibration import vector_contrast as student_vector
    groups = batch["groups"]; results = {"empirical_Bernstein": [], "Hoeffding": [], "StudentWelch": []}
    counts = {}
    for j, (_, _, endpoint, weights, side) in enumerate(CORE):
        ranges = {c: context["domain"]["numericBounds"] if endpoint == "slope_initial" else BOUNDS[endpoint] for c in weights}
        for method in ("empirical_Bernstein", "Hoeffding"):
            out = vector_contrast(groups, KEYS[endpoint], weights, ranges, side, method)
            results[method].append(out); counts[j] = out["counts"]
        results["StudentWelch"].append(student_vector(groups, KEYS[endpoint], weights, side))
    return {**{method: {k: np.column_stack([r[k] for r in own]) for k in own[0] if k != "counts"} for method, own in results.items()},
            "counts": counts}


def scalar_parity(cell, batch, statistics, context, policy, index, config):
    from .bounded_inference import BOUNDS, bounded_contrast
    from .empirical_bernstein import empirical_bernstein_contrast
    from .inference import contrast, multiplicity
    from .calibration import vector_holm
    tolerance = config["scalarParity"]
    atol, rtol = tolerance["absoluteTolerance"], tolerance["relativeTolerance"]
    core = {method: [] for method in ("empirical_Bernstein", "Hoeffding", "StudentWelch")}
    for j, (identifier, family, endpoint, weights, side) in enumerate(CORE):
        values = []
        for cid in weights:
            for person, value in enumerate(batch["groups"][cid][KEYS[endpoint]]["values"][index]):
                values.append({"participantId":f"fresh-vector-{cid}-{person}", "conditionId":cid,
                    "status":"available" if np.isfinite(value) else "unavailable", "value":float(value) if np.isfinite(value) else None})
        bounds = {c:context["domain"]["numericBounds"] if endpoint == "slope_initial" else BOUNDS[endpoint] for c in weights}
        scalar = {"empirical_Bernstein":empirical_bernstein_contrast(values, weights, bounds, side),
            "Hoeffding":bounded_contrast(values, weights, bounds, side), "StudentWelch":contrast(values, weights, side)}
        for method, result in scalar.items():
            core[method].append({**result, "id":identifier, "family":family})
            for key, vector_key in (("estimate","estimate"),("pValue","p")):
                actual = statistics[method][vector_key][index,j]; expected = result[key]
                if expected is None:
                    if np.isfinite(actual): raise AssertionError("Unavailable scalar parity mismatch")
                elif not np.isclose(expected, actual, atol=atol, rtol=rtol): raise AssertionError(f"{method}/{identifier}/{key} parity mismatch")
            if result["interval"] is not None and not np.allclose(result["interval"], [statistics[method]["low"][index,j], statistics[method]["high"][index,j]],atol=atol,rtol=rtol):
                raise AssertionError("Scalar interval parity mismatch")
        for scalar_key, vector_key in (("varianceComponentA","A"),("rangeComponentB","B"),("logPValue","logp"),("pointwiseRadius","radius")):
            expected = scalar["empirical_Bernstein"][scalar_key]; actual = statistics["empirical_Bernstein"][vector_key][index,j]
            if expected is not None and not np.isclose(expected,actual,atol=atol,rtol=rtol): raise AssertionError("EB component scalar parity mismatch")
    for method, own in core.items():
        for means_only in (False, True):
            mask = statistics[method]["available"].copy()
            rows = own
            if means_only:
                mask[:,2:5]=False
                rows=[{**r,"status":"unavailable","pValue":None} if 2<=j<=4 else r for j,r in enumerate(own)]
            corrected = multiplicity(rows, policy, config["alpha"])
            adjusted, reject = vector_holm(statistics[method]["p"],mask,policy,config["alpha"])
            for j,r in enumerate(corrected["tests"]):
                if bool(reject[index,j]) != r["reject"]: raise AssertionError("Scalar correction rejection parity mismatch")
                if r["adjustedP"] is not None and not np.isclose(r["adjustedP"],adjusted[index,j],atol=atol,rtol=rtol): raise AssertionError("Scalar adjusted-p parity mismatch")
    checked_people = 0
    for cid in ("C2","C3","C4"):
        table=context["projectionTables"][cid]; eligible=np.flatnonzero(batch["raw"][cid]["slopeAvailable"][index])
        if not len(eligible): continue
        for person in sorted(set((int(eligible[0]),int(eligible[len(eligible)//2]),int(eligible[-1])))):
            pattern=int(batch["raw"][cid]["patternIndices"][index,person]); shares=((pattern>>np.arange(10))&1)
            rows=[{**r,"shareA":float(shares[k])} for k,r in enumerate(context["scheduleRows"][cid])]
            raw=participant_slope(rows,"xInitial")["value"]
            x=np.asarray(table["x"]); algebra=float((x-x.mean())@shares/table["Sxx"])
            if not np.isclose(raw,table["rawCoefficients"][pattern],atol=atol,rtol=rtol) or not np.isclose(raw,algebra,atol=atol,rtol=rtol):
                raise AssertionError("Raw lookup/person projection parity mismatch")
            checked_people+=1
    return {"cell":cell["id"],"batch":batch["batchIndex"],"replica":index,"policy":policy,
            "status":"PASS","methods":3,"fixedNineRoutes":9,"matchedSixChecked":True,"oldRawPersonChecks":checked_people}


def _truth_masks(cell, population, representation):
    own = cell["truths"][population+"Core"]
    values = np.asarray(own[representation+"SignedTruth"])
    return values <= 0, values > 0


def aggregate_batch(cell, batch, statistics, config):
    from .calibration import vector_holm
    methods = {}; n = batch["attempts"]
    selected = np.asarray(cell["truths"]["selectedCore"]["rawTruth"])
    original = np.asarray(cell["truths"]["originalCore"]["rawTruth"])
    mean_indices = np.asarray([0,1,5,6,7,8]); families = sorted({r[1] for r in CORE})
    for method in ("empirical_Bernstein", "Hoeffding", "StudentWelch"):
        own = statistics[method]; available = own["available"]
        routes=[]
        for j, (identifier,family,endpoint,_,_) in enumerate(CORE):
            valid=available[:,j]; estimate=own["estimate"][:,j]
            routes.append({"id":identifier,"family":family,"endpoint":endpoint,"available":int(valid.sum()),
                "rawReject":int(np.sum(valid & (own["p"][:,j]<=config["alpha"]))),
                "smallTailReject":int(np.sum(valid & (own["p"][:,j]<=config["smallTailAlpha"]))),
                "selectedCoverage":int(np.sum(valid & (own["low"][:,j]<=selected[j]) & (own["high"][:,j]>=selected[j]))),
                "originalCoverage":int(np.sum(valid & (own["low"][:,j]<=original[j]) & (own["high"][:,j]>=original[j]))),
                "estimateSum":float(np.sum(np.where(valid,estimate,0.))),
                "widthSum":float(np.sum(np.where(valid,own["high"][:,j]-own["low"][:,j],0.))),
                "requiredGroupCountSums":{cid:int(v.sum()) for cid,v in statistics["counts"][j].items()}})
        policies={}
        for policy in config["policies"]:
            scopes={}
            for scope in ("fixed_nine","matched_six_means_nine_slots"):
                mask=available.copy()
                if scope!="fixed_nine":mask[:,2:5]=False
                _,reject=vector_holm(own["p"],mask,policy,config["alpha"])
                events={"anyFixedNineReject":int(np.any(reject,axis=1).sum()),
                    "allFixedNineReject":int(np.all(reject,axis=1).sum()),
                    "allSixMeanReject":int(np.all(reject[:,mean_indices],axis=1).sum()),
                    "routeRejects":{CORE[j][0]:int(reject[:,j].sum()) for j in range(9)}}
                for population in ("selected","original"):
                    for representation in ("raw","ideal"):
                        null,alternative=_truth_masks(cell,population,representation)
                        if scope!="fixed_nine":null[2:5]=False;alternative[2:5]=False
                        tag=population+"_"+representation
                        family_rosters={family:[CORE[j][0] for j in range(9) if null[j] and CORE[j][1]==family] for family in families}
                        events[tag]={"nullRoster":[CORE[j][0] for j in range(9) if null[j]],
                            "alternativeRoster":[CORE[j][0] for j in range(9) if alternative[j]],
                            "familyNullRosters":family_rosters,
                            "falseEvent":int(np.any(reject[:,null],axis=1).sum()) if null.any() else None,
                            "anyAlternative":int(np.any(reject[:,alternative],axis=1).sum()) if alternative.any() else None,
                            "allAlternatives":int(np.all(reject[:,alternative],axis=1).sum()) if alternative.any() else None,
                            "familyFalseEvents":{family:int(np.any(reject[:,null & np.asarray([r[1]==family for r in CORE])],axis=1).sum()) if family_rosters[family] else None for family in families}}
                intended = np.ones(9,bool) if cell["effects"]=="weak" else np.asarray([j in (0,5,6) for j in range(9)]) if cell["effects"]=="mixed" else _truth_masks(cell,"selected","ideal")[1]
                if scope!="fixed_nine":intended[2:5]=False
                events["prespecifiedIntendedAlternatives"]={"roster":[CORE[j][0] for j,v in enumerate(intended) if v],
                    "anyDiscovery":int(np.any(reject[:,intended],axis=1).sum()) if intended.any() else None,
                    "allDiscovery":int(np.all(reject[:,intended],axis=1).sum()) if intended.any() else None}
                scopes[scope]=events
            policies[policy]=scopes
        methods[method]={"routes":routes,"policies":policies}
    paired={}
    eb=statistics["empirical_Bernstein"]
    for comparator in ("Hoeffding","StudentWelch"):
        own=statistics[comparator]; rows=[]
        for j in mean_indices:
            common=eb["available"][:,j]&own["available"][:,j]
            rows.append({"id":CORE[j][0],"commonAvailable":int(common.sum()),
                "intervalWidthDifferenceSum":float(np.sum(np.where(common,(eb["high"][:,j]-eb["low"][:,j])-(own["high"][:,j]-own["low"][:,j]),0.))),
                "rawPValueDifferenceSum":float(np.sum(np.where(common,eb["p"][:,j]-own["p"][:,j],0.)))})
        paired[comparator]=rows
    return {"cell":cell["id"],"batch":batch["batchIndex"],"attempts":n,"methods":methods,
        "pairedMeanComparisons":paired,"unexpectedNumericFailures":0,
        "personSupport":{c:{"generated":int(r["includedPeople"].size),"completeRecords":int((r["recordCounts"]==10).sum()),
            "includedCompletePeople":int(r["includedPeople"].sum()),"includedFullShareEndpointPeople":int(r["slopeAvailable"].sum()) if "slopeAvailable" in r else None} for c,r in batch["raw"].items()}}


def combine_batches(cell,batches,config):
    from .calibration import binomial_interval
    if not batches or len({b["batch"] for b in batches})!=len(batches) or any(b["cell"]!=cell["id"] for b in batches):
        raise ValueError("Missing/duplicate/conflicting batch identities")
    n=sum(b["attempts"] for b in batches); methods={};failures=[]
    for method in ("empirical_Bernstein","Hoeffding","StudentWelch"):
        routes=[]
        for j,definition in enumerate(CORE):
            first=batches[0]["methods"][method]["routes"][j]
            total={k:sum(b["methods"][method]["routes"][j][k] for b in batches) for k in ("available","rawReject","smallTailReject","selectedCoverage","originalCoverage","estimateSum","widthSum")}
            count=total["available"]
            truth=cell["truths"]["selectedCore"]["rawTruth"][j];original=cell["truths"]["originalCore"]["rawTruth"][j]
            r={"id":definition[0],"family":definition[1],"endpoint":definition[2],"attempts":n,"availableAttempts":count,
                "unavailableAttempts":n-count,"selectedRawTruth":truth,"originalRawTruth":original,
                "idealSelectedTruth":cell["truths"]["selectedCore"]["idealTruth"][j],
                "rawRejection":binomial_interval(total["rawReject"],n),"smallTailRejection":binomial_interval(total["smallTailReject"],n),
                "rawRejectionConditionalAvailable":binomial_interval(total["rawReject"],count),
                "selectedPointwiseCoverageConditionalAvailable":binomial_interval(total["selectedCoverage"],count),
                "selectedAvailabilityAndCoverage":binomial_interval(total["selectedCoverage"],n),
                "originalPointwiseCoverageDiagnostic":binomial_interval(total["originalCoverage"],count),
                "meanEstimate":total["estimateSum"]/count if count else None,"meanWidth":total["widthSum"]/count if count else None,
                "meanRequiredGroupCounts":{c:sum(b["methods"][method]["routes"][j]["requiredGroupCountSums"][c] for b in batches)/n for c in first["requiredGroupCountSums"]}}
            r["selectedBias"]=r["meanEstimate"]-truth if count else None
            r["originalBiasDiagnostic"]=r["meanEstimate"]-original if count else None
            if method=="empirical_Bernstein":
                signed=cell["truths"]["selectedCore"]["rawSignedTruth"][j]
                if signed<=0:
                    for name,limit in (("rawRejection",config["criteria"]["rawNullUpper"]),("smallTailRejection",config["criteria"]["smallTailNullUpper"])):
                        if r[name]["interval"][1]>limit:failures.append({"cell":cell["id"],"id":r["id"],"criterion":name,"value":r[name]["interval"][1],"threshold":limit})
                ci=r["selectedPointwiseCoverageConditionalAvailable"]["interval"]
                if ci is None or ci[0]<config["criteria"]["selectedCoverageLower"]:
                    failures.append({"cell":cell["id"],"id":r["id"],"criterion":"selectedPointwiseCoverageConditionalAvailable","value":ci[0] if ci else None,"threshold":config["criteria"]["selectedCoverageLower"]})
            routes.append(r)
        policies={}
        for policy in config["policies"]:
            scopes={}
            for scope in ("fixed_nine","matched_six_means_nine_slots"):
                first=batches[0]["methods"][method]["policies"][policy][scope]
                event={"globalAnyGuarantee":policy!="within_family_holm","roster":"nine_slots_fixed_even_in_mean_only_diagnostic"}
                for key in ("anyFixedNineReject","allFixedNineReject","allSixMeanReject"):
                    event[key]=binomial_interval(sum(b["methods"][method]["policies"][policy][scope][key] for b in batches),n)
                event["routeRejects"]={identifier:binomial_interval(sum(b["methods"][method]["policies"][policy][scope]["routeRejects"][identifier] for b in batches),n) for identifier,*_ in CORE}
                for population in ("selected","original"):
                    for representation in ("raw","ideal"):
                        tag=population+"_"+representation;origin=first[tag]
                        counts=lambda key:sum(b["methods"][method]["policies"][policy][scope][tag][key] for b in batches)
                        event[tag]={"nullRoster":origin["nullRoster"],"nullCount":len(origin["nullRoster"]),
                            "alternativeRoster":origin["alternativeRoster"],"alternativeCount":len(origin["alternativeRoster"]),
                            "familyNullRosters":origin["familyNullRosters"],
                            **{key:binomial_interval(counts(key),n) if origin[key] is not None else {"rate":None,"interval":None,"reason":"empty_null_subset_not_null_calibration" if key=="falseEvent" else "empty_alternative_subset_not_power","denominator":n,"status":"unavailable"} for key in ("falseEvent","anyAlternative","allAlternatives")}}
                        event[tag]["familyFalseEvents"]={f:binomial_interval(sum(b["methods"][method]["policies"][policy][scope][tag]["familyFalseEvents"][f] for b in batches),n) if origin["familyFalseEvents"][f] is not None else {"rate":None,"interval":None,"reason":"empty_family_null_subset_not_null_calibration","denominator":n,"status":"unavailable"} for f in origin["familyFalseEvents"]}
                own=first["prespecifiedIntendedAlternatives"]
                event["prespecifiedIntendedAlternatives"]={"roster":own["roster"],**{key:binomial_interval(sum(b["methods"][method]["policies"][policy][scope]["prespecifiedIntendedAlternatives"][key] for b in batches),n) if own[key] is not None else {"rate":None,"interval":None,"reason":"empty_prespecified_subset_not_power","denominator":n} for key in ("anyDiscovery","allDiscovery")}}
                if method=="empirical_Bernstein" and scope=="fixed_nine":
                    raw=event["selected_raw"]
                    if policy!="within_family_holm" and raw["falseEvent"]["interval"] is not None and raw["falseEvent"]["interval"][1]>config["criteria"]["globalTrueNullFWERUpper"]:
                        failures.append({"cell":cell["id"],"id":policy,"criterion":"selectedRawTrueNullFWER","value":raw["falseEvent"]["interval"][1],"threshold":config["criteria"]["globalTrueNullFWERUpper"]})
                    if policy=="within_family_holm":
                        for f,value in raw["familyFalseEvents"].items():
                            if value["interval"] is not None and value["interval"][1]>config["criteria"]["globalTrueNullFWERUpper"]:
                                failures.append({"cell":cell["id"],"id":policy+"/"+f,"criterion":"selectedRawWithinFamilyFWER","value":value["interval"][1],"threshold":config["criteria"]["globalTrueNullFWERUpper"]})
                scopes[scope]=event
            policies[policy]=scopes
        methods[method]={"routes":routes,"policies":policies,"comparisonOnly":method!="empirical_Bernstein"}
    paired={}
    for method in ("Hoeffding","StudentWelch"):
        rows=[]
        for j,first in enumerate(batches[0]["pairedMeanComparisons"][method]):
            count=sum(b["pairedMeanComparisons"][method][j]["commonAvailable"] for b in batches)
            rows.append({"id":first["id"],"commonAvailableAttempts":count,
                "meanWidthDifference":sum(b["pairedMeanComparisons"][method][j]["intervalWidthDifferenceSum"] for b in batches)/count if count else None,
                "meanRawPValueDifference":sum(b["pairedMeanComparisons"][method][j]["rawPValueDifferenceSum"] for b in batches)/count if count else None,
                "comparison":"matched_draws_person_target_and_common_support_not_cross_round"})
        paired[method]=rows
    support={cid:{key:sum(b["personSupport"][cid][key] for b in batches) if first_value is not None else None for key,first_value in batches[0]["personSupport"][cid].items()} for cid in CONDITIONS}
    return {"id":cell["id"],"seed":cell["seed"],"label":cell["label"],"attempts":n,"batches":len(batches),
        "analyticTruthsAndNullRosters":cell["truths"],"methods":methods,"pairedMeanComparisons":paired,
        "criteriaFailures":failures,"empiricalCriteriaPass":not failures,"unexpectedNumericFailures":sum(b["unexpectedNumericFailures"] for b in batches),"personSupportCensus":support,
        "originalMNARPopulationRepaired":False,"humanEvidence":False,"primaryReplacement":False}
