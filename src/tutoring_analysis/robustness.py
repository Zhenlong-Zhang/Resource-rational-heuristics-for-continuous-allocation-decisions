"""Frozen artificial bounded-person vector calibration; NOT finite task policies.

Uses the original production correction and read-only matched Student/Welch means.
Never treats old QA or these vectors as human observations or slope calibration.
"""
from __future__ import annotations

import math
import numpy as np

from .bounded_inference import (BOUNDS, METHOD, PERSON_R_ENVELOPE, REPRESENTATION,
                               TRIAL_R_ENVELOPE, UNIT_ROUNDOFF, bounded_contrast)
from .inference import CONDITIONS, CORE, FAMILIES, POLICIES, contrast, multiplicity
from .calibration import binomial_interval, vector_contrast, vector_holm

KEYS = {"anyQuestion": "Q", "EOResemblance": "R", "lowerNeedDirectional": "L"}
SUPPORTED = np.asarray([i for i, x in enumerate(CORE) if x[2] in KEYS])
CELL_IDS = ("bounded_null", "bounded_signed", "bounded_opposite", "bounded_mixed",
            "bounded_composite", "bounded_skew_null", "bounded_unequal_null",
            "bounded_shared_null", "bounded_mcar_null", "bounded_mnar_selected")


def resolved_cells(config):
    if config.get("schema") != "tutoring-bounded-robustness-v1" or config.get("method") != METHOD or config.get("primaryReplacement") is not False:
        raise ValueError("Wrong explicit bounded sensitivity configuration")
    if config.get("purpose") != "synthetic_bounded_person_vector_calibration_not_finite_policy_or_human_evidence":
        raise ValueError("Wrong synthetic calibration purpose")
    if config.get("policies") != list(POLICIES) or config.get("alpha") != .05 or config.get("confidence") != .95:
        raise ValueError("Prospective policies/alpha/confidence changed")
    if config.get("batchSize") != 250 or config.get("replicationsPerCell") != 10000 or config.get("attempts") != 100000:
        raise ValueError("Frozen count/batch contract changed")
    if config.get("bounds") != {"Q": [0, 1], "R": [-50, 50], "L": [-.5, .5]}:
        raise ValueError("Universal source bounds must not use observed extrema")
    expected_numeric = {"idealRBounds": [-50, 50], "binary64UnitRoundoff": UNIT_ROUNDOFF,
        "rawTrialRAbsoluteEnvelope": TRIAL_R_ENVELOPE, "rawPersonRAbsoluteEnvelope": PERSON_R_ENVELOPE,
        "rawTrialRBounds": [-50-TRIAL_R_ENVELOPE, 50+TRIAL_R_ENVELOPE],
        "rawPersonRBounds": list(BOUNDS["EOResemblance"]), "rawValuesClamped": False,
        "boundsFittedToObservedExtrema": False, "theorem": "exact_real_probability_bound",
        "implementedInference": "approximate_binary64_not_machine_certified"}
    if config.get("numericRepresentation") != expected_numeric: raise ValueError("Prospective numeric representation changed")
    if config.get("scalarParity") != {"batchPositions": ["first", "last"], "replicaIndices": [0, 17, 249],
        "absoluteTolerance": 1e-10, "relativeTolerance": 1e-9, "includeMatchedStudentWelch": True}:
        raise ValueError("Frozen parity settings/tolerances changed")
    if [c.get("id") for c in config["cells"]] != list(CELL_IDS): raise ValueError("Fixed cell roster/order changed")
    out = []
    for i, own in enumerate(config["cells"]):
        c = {**config["defaults"], **own, "replications": config["replicationsPerCell"]}
        if c["seed"] != 710071+i or isinstance(c["seed"], bool): raise ValueError("Frozen seed changed")
        if c["generator"] not in ("independent_uniform", "shared_uniform", "beta_skew"):
            raise ValueError("Unknown bounded generator")
        if set(c["people"]) != set(CONDITIONS) or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in c["people"].values()):
            raise ValueError("Positive fixed independent-group counts required")
        if c["requiredRecordCount"] != 10 or c["samplingUnit"] != "independent_person_each_arm" or c["expectedSupportedMeans"] != 6 or c["unsupportedSlopes"] != 3:
            raise ValueError("Eligibility/sampling/support scope changed")
        for k in ("participantRetention", "recordRetention", "endpointRetention"):
            if isinstance(c[k], bool) or not math.isfinite(c[k]) or not 0 <= c[k] <= 1: raise ValueError("Invalid retention probability")
        if c["effects"] not in config["effectRecipes"]: raise ValueError("Unknown frozen effect recipe")
        c["parameters"] = config["effectRecipes"][c["effects"]]
        c["analyticTruths"] = truths(c)
        # Bounds are universal, not the generator's narrower observed range.
        for cid in CONDITIONS:
            p, r, l = [c["analyticTruths"]["original"][cid][k] for k in ("Q", "R", "L")]
            if not 0 <= p <= 1: raise ValueError("Invalid Bernoulli probability")
            rlo, rhi, llo, lhi = (r-20, r+20, l-.3, l+.3)
            if c["generator"] == "beta_skew": rlo, rhi, llo, lhi = r-30/11, r+300/11, l-.3/11, l+3/11
            if rlo < -50 or rhi > 50 or llo < -.5 or lhi > .5: raise ValueError("Generator leaves source domains")
        out.append(c)
    return out


def truths(cell):
    original = {cid: {key: float(cell["parameters"][key].get(cid, 0.)) for key in ("Q", "R", "L")} for cid in CONDITIONS}
    selected = {cid: dict(v) for cid, v in original.items()}
    if cell.get("selection"):
        if cell["selection"] != "own_uniform_above_0.75" or cell["generator"] != "shared_uniform":
            raise ValueError("Unsupported selected target")
        for cid, v in original.items():
            selected[cid] = {"Q": max(0., min(v["Q"]-.75, .25))/.25,
                             "R": v["R"]+40*(.875-.5), "L": v["L"]+.6*(.875-.5)}
    return {"original": original, "selected": selected,
            "selectionTargetNotOriginalRepair": bool(cell.get("selection"))}


def generate_batch(cell, batch_index, batch_size=250):
    if not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 0 or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Invalid deterministic batch identity/shape")
    rng = np.random.default_rng(np.random.SeedSequence([cell["seed"], batch_index]))
    groups, raw = {}, {}
    for cid in CONDITIONS:
        n = cell["people"][cid]; shape = (batch_size, n); mu = cell["analyticTruths"]["original"][cid]
        kind = cell["generator"]
        if kind == "shared_uniform":
            u = rng.random(shape); q = (u < mu["Q"]).astype(float)
            r, l = mu["R"]+40*(u-.5), mu["L"]+.6*(u-.5)
        elif kind == "beta_skew":
            u = rng.beta(.5, 5., shape); q = rng.binomial(10, mu["Q"], shape)/10
            r = mu["R"]+30*(u-1/11); l = mu["L"]+.3*(rng.beta(.5, 5., shape)-1/11)
        else:
            u = rng.random(shape); q = rng.binomial(10, mu["Q"], shape)/10
            r = mu["R"]+40*(u-.5); l = mu["L"]+.6*(rng.random(shape)-.5)
        person_keep = rng.random(shape) < cell["participantRetention"]
        record_mask = rng.random((*shape, 10)) < cell["recordRetention"]
        person_keep &= np.all(record_mask, axis=-1)
        if cell.get("selection"): person_keep &= u > .75
        groups[cid] = {}
        for key, values in (("Q", q), ("R", r), ("L", l)):
            valid = person_keep & (rng.random(shape) < cell["endpointRetention"])
            groups[cid][key] = {"values": np.where(valid, values, np.nan),
                               "truth": np.full(shape, mu[key]),
                               "selectedTruth": np.full(shape, cell["analyticTruths"]["selected"][cid][key]),
                               "available": valid}
        raw[cid] = {"recordCounts": np.sum(record_mask, axis=-1), "includedPeople": person_keep, "ownU": u}
    return {"groups": groups, "raw": raw, "attempts": batch_size, "batchIndex": batch_index}


def vector_bounded_contrast(groups, endpoint, weights, side="greater", confidence=.95):
    key = KEYS[endpoint]; lo, hi = BOUNDS[endpoint]
    batch = next(iter(groups.values()))[key]["values"].shape[0]
    estimate, V = np.zeros(batch), np.zeros(batch)
    available = np.ones(batch, bool); counts = {}; original, selected = 0., 0.
    for cid, w in weights.items():
        g = groups[cid][key]; v = np.asarray(g["values"], float)
        if v.ndim != 2 or v.shape[0] != batch or np.any(np.isinf(v)):
            raise ValueError("Invalid bounded person vector shape/nonfinite value")
        valid = ~np.isnan(v)
        if np.any(valid & ((v < lo) | (v > hi))): raise ValueError("Bounded vector leaves known source range")
        if not np.array_equal(valid, g["available"]): raise ValueError("Person support/value mask mismatch")
        n = valid.sum(axis=1); counts[cid] = n
        means = np.sum(np.where(valid, v, 0.), axis=1)/np.maximum(n, 1)
        estimate += w*means; V += w*w*(hi-lo)**2/np.maximum(n, 1); available &= n > 0
        # Frozen truths are constant within a cell/condition, not estimated responses.
        original += w*float(g["truth"][0, 0]); selected += w*float(g["selectedTruth"][0, 0])
    signed = estimate if side == "greater" else -estimate
    logp = -2*np.maximum(signed, 0.)**2/V
    radius = np.sqrt(V*(math.log(2.)-math.log1p(-confidence))/2)
    legal_low = sum(w*(lo if w > 0 else hi) for w in weights.values())
    legal_high = sum(w*(hi if w > 0 else lo) for w in weights.values())
    out = {"estimate": estimate, "V": V, "p": np.exp(logp), "logp": logp, "radius": radius,
           "rawLow": estimate-radius, "rawHigh": estimate+radius,
           "low": np.maximum(legal_low, estimate-radius), "high": np.minimum(legal_high, estimate+radius)}
    if any(not np.all(np.isfinite(x)) for x in out.values()): raise ValueError("Unexpected nonfinite supported bounded arithmetic")
    return {**{k: np.where(available, v, np.nan) for k, v in out.items()}, "available": available,
            "counts": counts, "originalTruth": float(original), "selectedTruth": float(selected)}


def core_statistics(groups):
    batch = groups["C1"]["Q"]["values"].shape[0]
    stats = {k: np.full((batch, 9), np.nan) for k in ("estimate", "V", "p", "logp", "radius", "rawLow", "rawHigh", "low", "high")}
    stats["available"] = np.zeros((batch, 9), bool)
    stats["originalTruth"] = np.full(9, np.nan); stats["selectedTruth"] = np.full(9, np.nan); stats["counts"] = {}
    matched = {k: np.full((batch, 9), np.nan) for k in ("estimate", "p", "low", "high", "se", "df", "statistic")}
    matched["available"] = np.zeros((batch, 9), bool)
    for j, (_, _, endpoint, weights, side) in enumerate(CORE):
        if endpoint not in KEYS: continue
        r = vector_bounded_contrast(groups, endpoint, weights, side)
        for k in stats:
            if k == "counts": stats[k][j] = r[k]
            elif k in ("originalTruth", "selectedTruth"): stats[k][j] = r[k]
            else: stats[k][:, j] = r[k]
        old = vector_contrast(groups, KEYS[endpoint], weights, side)
        for k in matched: matched[k][:, j] = old[k]
    return {"bounded": stats, "matchedStudentWelch": matched}


def scalar_parity(cell, batch, statistics, policy, index, alpha=.05):
    core, matched_core, checked = [], [], []
    for j, (identifier, family, endpoint, weights, side) in enumerate(CORE):
        if endpoint not in KEYS:
            r = {"status": "unavailable", "pValue": None}; old = dict(r)
        else:
            values = []
            for cid in weights:
                for p, value in enumerate(batch["groups"][cid][KEYS[endpoint]]["values"][index]):
                    values.append({"participantId": f"fresh-vector-{cid}-{p}", "conditionId": cid,
                                   "status": "available" if np.isfinite(value) else "unavailable",
                                   "value": float(value) if np.isfinite(value) else None})
            r = bounded_contrast(values, weights, {cid: BOUNDS[endpoint] for cid in weights}, side)
            old = contrast(values, weights, side)
            for source, fields in ((r, (("estimate", "estimate"), ("rangeProxyV", "V"), ("pValue", "p"), ("logPValue", "logp"), ("pointwiseRadius", "radius"))),
                                   (old, (("estimate", "estimate"), ("pValue", "p"), ("standardError", "se"), ("df", "df"), ("statistic", "statistic")))):
                actual = statistics["bounded" if source is r else "matchedStudentWelch"]
                for sk, vk in fields:
                    a, b = source[sk], actual[vk][index, j]
                    if a is None:
                        if np.isfinite(b): raise AssertionError(f"Unexpected supported {identifier}/{sk}")
                    elif not np.isclose(a, b, atol=1e-10, rtol=1e-9): raise AssertionError(f"Scalar arithmetic mismatch {identifier}/{sk}")
                if source["interval"] is not None and not np.allclose(source["interval"], [actual["low"][index, j], actual["high"][index, j]], atol=1e-10, rtol=1e-9):
                    raise AssertionError(f"Scalar interval mismatch {identifier}")
            checked.append({"id": identifier, "boundedStatus": r["status"], "matchedStatus": old["status"]})
        core.append({**r, "id": identifier, "family": family}); matched_core.append({**old, "id": identifier, "family": family})
    for name, rows in (("bounded", core), ("matchedStudentWelch", matched_core)):
        scalar = multiplicity(rows, policy, alpha)
        s = statistics[name]; adjusted, reject = vector_holm(s["p"], s["available"], policy, alpha)
        for j, row in enumerate(scalar["tests"]):
            if row["reject"] != bool(reject[index, j]): raise AssertionError("Scalar correction rejection mismatch")
            if row["adjustedP"] is not None and not np.isclose(row["adjustedP"], adjusted[index, j], atol=1e-10, rtol=1e-9):
                raise AssertionError("Scalar correction adjusted-p mismatch")
    return {"cell": cell["id"], "batch": batch["batchIndex"], "replica": index, "policy": policy,
            "status": "PASS", "supportedMeanComparisons": checked, "slopesMethodUnsupported": 3}


def aggregate_batch(cell, batch, statistics, config):
    n = batch["attempts"]; s = statistics["bounded"]; t = statistics["matchedStudentWelch"]
    routes = []
    for j, (identifier, family, endpoint, _, _) in enumerate(CORE):
        available = s["available"][:, j]; matched_available = t["available"][:, j]
        mu, selected = s["originalTruth"][j], s["selectedTruth"][j]
        routes.append({"id": identifier, "family": family, "endpoint": endpoint,
            "methodUnsupported": endpoint not in KEYS, "attempts": n, "available": int(available.sum()),
            "originalTruth": float(mu) if np.isfinite(mu) else None, "selectedTruth": float(selected) if np.isfinite(selected) else None,
            "rawReject": int(np.sum(available & (s["p"][:, j] <= config["alpha"]))),
            "smallTailReject": int(np.sum(available & (s["p"][:, j] <= config["alpha"]/9))),
            "originalCoverage": int(np.sum(available & (s["low"][:, j] <= mu) & (s["high"][:, j] >= mu))),
            "selectedCoverage": int(np.sum(available & (s["low"][:, j] <= selected) & (s["high"][:, j] >= selected))),
            "estimateSum": float(np.nansum(s["estimate"][:, j])), "radiusSum": float(np.nansum(s["radius"][:, j])),
            "untruncatedWidthSum": float(np.nansum(s["rawHigh"][:, j]-s["rawLow"][:, j])),
            "widthSum": float(np.nansum(s["high"][:, j]-s["low"][:, j])),
            "requiredGroupCountSums": {cid: int(v.sum()) for cid, v in s["counts"].get(j, {}).items()},
            "matchedAvailable": int(matched_available.sum()),
            "matchedRawReject": int(np.sum(matched_available & (t["p"][:, j] <= config["alpha"]))),
            "matchedWidthSum": float(np.nansum(t["high"][:, j]-t["low"][:, j])),
            "matchedSelectedCoverage": int(np.sum(matched_available & (t["low"][:, j] <= selected) & (t["high"][:, j] >= selected)))})
    original_null = np.isfinite(s["originalTruth"]) & (s["originalTruth"] <= 0)
    selected_null = np.isfinite(s["selectedTruth"]) & (s["selectedTruth"] <= 0)
    selected_alt = np.isfinite(s["selectedTruth"]) & (s["selectedTruth"] > 0)
    policies = {}
    for policy in config["policies"]:
        _, reject = vector_holm(s["p"], s["available"], policy, config["alpha"])
        _, old_reject = vector_holm(t["p"], t["available"], policy, config["alpha"])
        policies[policy] = {"attempts": n, "anyFixedNineReject": int(np.any(reject, axis=1).sum()),
            "allFixedNineReject": int(np.all(reject, axis=1).sum()),
            "allSixSupportedReject": int(np.all(reject[:, SUPPORTED], axis=1).sum()),
            "selectedTrueNullFalseEvent": int(np.any(reject[:, selected_null], axis=1).sum()),
            "originalNullFalseEventDiagnostic": int(np.any(reject[:, original_null], axis=1).sum()),
            "anyTrueAlternativeDiscovery": int(np.any(reject[:, selected_alt], axis=1).sum()) if selected_alt.any() else None,
            "allTrueAlternativeDiscovery": int(np.all(reject[:, selected_alt], axis=1).sum()) if selected_alt.any() else None,
            "matchedAnyFixedNineReject": int(np.any(old_reject, axis=1).sum()),
            "matchedAllSixSupportedReject": int(np.all(old_reject[:, SUPPORTED], axis=1).sum()),
            "routeRejects": {CORE[j][0]: int(reject[:, j].sum()) for j in range(9)},
            "familyFalseEvents": {family: int(np.any(reject[:, selected_null & np.asarray([r[1] == family for r in CORE])], axis=1).sum())
                if family != "responsiveness" else None for family in FAMILIES}}
    return {"cell": cell["id"], "batch": batch["batchIndex"], "attempts": n, "routes": routes, "policies": policies,
            "unexpectedNumericFailures": 0, "personRecordSupport": {cid: {"included": int(r["includedPeople"].sum()),
                "completeRecords": int((r["recordCounts"] == 10).sum()), "generated": int(r["recordCounts"].size)} for cid, r in batch["raw"].items()}}


def combine_batches(cell, batches, config):
    if not batches or len({b["batch"] for b in batches}) != len(batches) or any(b["cell"] != cell["id"] for b in batches):
        raise ValueError("Duplicate/conflicting batch identities")
    n = sum(b["attempts"] for b in batches); results, failures = [], []
    for j, definition in enumerate(CORE):
        first = batches[0]["routes"][j]
        total = {k: sum(b["routes"][j][k] for b in batches) for k in ("available", "rawReject", "smallTailReject", "originalCoverage", "selectedCoverage", "estimateSum", "radiusSum", "widthSum", "untruncatedWidthSum", "matchedAvailable", "matchedRawReject", "matchedWidthSum", "matchedSelectedCoverage")}
        count = total["available"]; old_count = total["matchedAvailable"]
        r = {"id": definition[0], "family": definition[1], "endpoint": definition[2], "attempts": n,
             "methodUnsupported": first["methodUnsupported"], "availableAttempts": count, "unavailableAttempts": n-count,
             "originalTruth": first["originalTruth"], "selectedTruth": first["selectedTruth"],
             "rawRejection": binomial_interval(total["rawReject"], n), "smallTailRejection": binomial_interval(total["smallTailReject"], n),
             "selectedPointwiseCoverage": binomial_interval(total["selectedCoverage"], count),
             "originalPointwiseCoverage": binomial_interval(total["originalCoverage"], count),
             "meanEstimate": total["estimateSum"]/count if count else None,
             "meanRadius": total["radiusSum"]/count if count else None,
             "meanIntervalWidth": total["widthSum"]/count if count else None,
             "meanUntruncatedWidth": total["untruncatedWidthSum"]/count if count else None,
             "meanRequiredGroupCounts": {cid: sum(b["routes"][j]["requiredGroupCountSums"][cid] for b in batches)/n for cid in first["requiredGroupCountSums"]},
             "matchedStudentWelch": {"comparisonOnly": True, "availableAttempts": old_count, "unavailableAttempts": n-old_count,
                 "rawRejection": binomial_interval(total["matchedRawReject"], n),
                 "selectedCoverage": binomial_interval(total["matchedSelectedCoverage"], old_count),
                 "meanWidth": total["matchedWidthSum"]/old_count if old_count else None}}
        r["biasSelected"] = r["meanEstimate"]-r["selectedTruth"] if count else None
        r["biasOriginalDiagnostic"] = r["meanEstimate"]-r["originalTruth"] if count else None
        if r["methodUnsupported"]:
            for k in ("rawRejection", "smallTailRejection", "selectedPointwiseCoverage", "originalPointwiseCoverage"):
                r[k] = {"rate": None, "interval": None, "numerator": None, "denominator": n, "reason": "method_unsupported_not_inferential_evidence"}
        if not r["methodUnsupported"]:
            if r["selectedTruth"] <= 0:
                for metric, threshold in (("rawRejection", config["criteria"]["rawNullErrorUpper"]), ("smallTailRejection", config["criteria"]["smallTailErrorUpper"])):
                    if r[metric]["interval"][1] > threshold: failures.append({"cell": cell["id"], "id": r["id"], "criterion": metric, "value": r[metric]["interval"][1], "threshold": threshold})
            coverage = r["selectedPointwiseCoverage"]["interval"]
            if coverage is None or coverage[0] < config["criteria"]["pointwiseCoverageLower"]:
                failures.append({"cell": cell["id"], "id": r["id"], "criterion": "selectedPointwiseCoverage", "value": coverage[0] if coverage else None, "threshold": config["criteria"]["pointwiseCoverageLower"]})
        results.append(r)
    policies = {}
    for policy in config["policies"]:
        first = batches[0]["policies"][policy]; own = {"globalAnyGuarantee": policy != "within_family_holm", "responsivenessFamilyStatus": "method_unsupported_no_inferential_evidence"}
        for key in ("anyFixedNineReject", "allFixedNineReject", "allSixSupportedReject", "selectedTrueNullFalseEvent", "originalNullFalseEventDiagnostic", "anyTrueAlternativeDiscovery", "allTrueAlternativeDiscovery", "matchedAnyFixedNineReject", "matchedAllSixSupportedReject"):
            own[key] = binomial_interval(sum(b["policies"][policy][key] for b in batches), n) if first[key] is not None else {"rate": None, "reason": "no_true_alternative_subset", "numerator": None, "denominator": n}
        own["familyFalseEvents"] = {family: binomial_interval(sum(b["policies"][policy]["familyFalseEvents"][family] for b in batches), n) if family != "responsiveness" else {"rate": None, "reason": "method_unsupported_family", "numerator": None, "denominator": n} for family in FAMILIES}
        own["routeRejects"] = {identifier: binomial_interval(sum(b["policies"][policy]["routeRejects"][identifier] for b in batches), n) for identifier, *_ in CORE}
        if policy != "within_family_holm" and own["selectedTrueNullFalseEvent"]["interval"][1] > config["criteria"]["globalFwerUpper"]:
            failures.append({"cell": cell["id"], "id": policy, "criterion": "selectedTrueNullFWER", "value": own["selectedTrueNullFalseEvent"]["interval"][1], "threshold": config["criteria"]["globalFwerUpper"]})
        threshold = config["criteria"]["budgetFamilyFwerUpper"] if policy == "four_family_alpha_budget" else config["criteria"]["familyFwerUpper"]
        for family, result in own["familyFalseEvents"].items():
            if family != "responsiveness" and result["interval"][1] > threshold:
                failures.append({"cell": cell["id"], "id": policy+"/"+family, "criterion": "selectedTrueNullFamilyFWER", "value": result["interval"][1], "threshold": threshold})
        policies[policy] = own
    return {"id": cell["id"], "label": cell["label"], "seed": cell["seed"], "attempts": n, "batches": len(batches),
            "analyticTruths": cell["analyticTruths"], "routes": results, "policies": policies,
            "unexpectedNumericFailures": sum(b["unexpectedNumericFailures"] for b in batches),
            "criteriaFailures": failures, "empiricalCriteriaPass": not failures,
            "originalMNARTargetDiagnosticOnly": not cell["selectedTargetIsOriginal"],
            "generatorScope": cell["geometry"], "humanEvidence": False, "primaryReplacement": False}
