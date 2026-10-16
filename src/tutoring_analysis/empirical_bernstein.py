"""Explicit MP11 sensitivity with source-known domains, never a primary fallback.

The mean of full-endpoint initial projections is a narrower target than the old
observed-subset projection. Binary64 inference is approximate, not certified.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from numbers import Real
import math

import numpy as np

from design_validation.bounded_task import belief_moments, expand_templates, initial_log_beliefs
from .bounded_inference import BOUNDS, GEOMETRY, REPRESENTATION, validate_geometry
from .inference import (CONDITIONS, CORE, POLICIES, _boolean, _number, multiplicity,
                        participant_slope, participant_values)

METHOD = "independent_person_empirical_Bernstein_MP11"
CONTEXT = "source_legal_complete_ten_initial_share_endpoints"
ATOL = RTOL = 1e-10
MEAN_ENDPOINTS = ("anyQuestion", "EOResemblance", "lowerNeedDirectional")


def _real(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _inputs(values, weights, bounds, side, null, confidence):
    if side not in ("greater", "less") or not _real(null) or not _real(confidence) or not 0 < confidence < 1:
        raise ValueError("Invalid finite non-Boolean side/null/confidence")
    if not weights or set(weights) != set(bounds): raise ValueError("Fixed required weights/bounds disagree")
    for cid, weight in weights.items():
        if cid not in CONDITIONS or not _real(weight) or weight == 0: raise ValueError("Invalid required arm weight")
        interval = bounds[cid]
        if not isinstance(interval, (list, tuple, np.ndarray)) or len(interval) != 2 or any(not _real(x) for x in interval) or interval[0] >= interval[1]:
            raise ValueError("Nonzero source-known finite ranges required")
    groups, seen = {cid: [] for cid in weights}, set()
    for row in values:
        p, cid = row.get("participantId"), row.get("conditionId")
        if not isinstance(p, str) or not p.strip() or p != p.strip() or cid not in CONDITIONS:
            raise ValueError("Invalid independent-person identity")
        if p in seen: raise ValueError("Repeated independent-person identity")
        seen.add(p)
        status, value = row.get("status"), row.get("value")
        if status not in ("available", "unavailable"): raise ValueError("Invalid person availability")
        if status == "unavailable":
            if value is not None: raise ValueError("Unavailable person supplies outcome")
            continue
        if not _real(value): raise ValueError("Invalid person outcome")
        if cid in groups:
            if not bounds[cid][0] <= value <= bounds[cid][1]:
                raise ValueError("Supported person outcome outside declared domain: halt, never exclude")
            groups[cid].append(float(value))
    return groups


def inverse_tail(estimate, null, side, A, B, arms):
    """Stable inversion of A sqrt(log(2k/p)) + B log(2k/p)."""
    if any(not _real(x) for x in (estimate, null, A, B)) or A < 0 or B <= 0 or not isinstance(arms, int) or isinstance(arms, bool) or arms < 1 or side not in ("greater", "less"):
        raise ValueError("Invalid supported inverse components")
    signed = estimate-null if side == "greater" else null-estimate
    if not math.isfinite(signed): raise ValueError("Nonfinite signed deviation")
    if signed <= 0: return {"pValue": 1., "logPValue": 0., "pValueUnderflow": False, "signedDeviation": signed, "inverseLogRadius": 0.}
    # Avoid B*t underflow when the observed variance component is zero.
    try:
        root = math.sqrt(signed/B) if A == 0 else 2*signed/(A+math.hypot(A, 2*math.sqrt(B*signed)))
        z = root*root
        logp = min(0., math.log(2*arms)-z)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Nonfinite empirical-Bernstein inverse") from exc
    if not all(math.isfinite(x) for x in (root, z, logp)):
        raise ValueError("Nonfinite empirical-Bernstein inverse")
    p = math.exp(logp)
    return {"pValue": p, "logPValue": logp, "pValueUnderflow": p == 0. and logp < 0.,
            "signedDeviation": signed, "inverseLogRadius": z}


def empirical_bernstein_contrast(values, weights, bounds, side="greater", null=0., confidence=.95):
    groups = _inputs(values, weights, bounds, side, null, confidence)
    summary = {cid: {"participants": len(v), "mean": float(np.mean(v)) if v else None,
                    "variance": float(np.var(v, ddof=1)) if len(v) > 1 else None,
                    "knownBounds": list(bounds[cid])} for cid, v in groups.items()}
    estimate = math.fsum(weights[c]*summary[c]["mean"] for c in weights) if all(groups.values()) else None
    result = {"method": METHOD, "status": "unavailable", "reason": None, "weights": dict(weights),
        "side": side, "null": float(null), "confidence": float(confidence), "groups": summary,
        "estimate": estimate, "pValue": None, "logPValue": None, "pValueUnderflow": False,
        "interval": None, "untruncatedInterval": None, "pointwiseRadius": None,
        "varianceComponentA": None, "rangeComponentB": None, "signedDeviation": None,
        "inverseLogRadius": None, "legalContrastBounds": None,
        "oneSidedArmBudget": "delta/requiredArms", "twoSidedArmTailBudget": "delta/(2*requiredArms)",
        "independentUnit": "participant", "knownBoundsEstimatedFromSample": False,
        "intervalSimultaneous": False, "numericCertification": "approximate_binary64_not_machine_certified",
        "target": "mean of included independent persons' expectations; selected context, not MNAR original repair"}
    if any(len(v) < 2 for v in groups.values()):
        return {**result, "reason": "required_arm_fewer_than_two_people_no_fallback"}
    try:
        A = math.fsum(abs(weights[c])*math.sqrt(2*summary[c]["variance"]/len(groups[c])) for c in weights)
        B = math.fsum(abs(weights[c])*7*(bounds[c][1]-bounds[c][0])/(3*(len(groups[c])-1)) for c in weights)
        low = math.fsum(weights[c]*bounds[c][0 if weights[c] > 0 else 1] for c in weights)
        high = math.fsum(weights[c]*bounds[c][1 if weights[c] > 0 else 0] for c in weights)
        radius = A*math.sqrt(math.log(4*len(weights))-math.log1p(-confidence))+B*(math.log(4*len(weights))-math.log1p(-confidence))
        raw = [estimate-radius, estimate+radius]
    except (OverflowError, ValueError) as exc: raise ValueError("Nonfinite supported EB arithmetic") from exc
    if not all(math.isfinite(x) for x in (estimate, A, B, low, high, radius, *raw)) or B <= 0:
        raise ValueError("Nonfinite supported EB arithmetic")
    return {**result, **inverse_tail(estimate, null, side, A, B, len(weights)),
            "status": "available", "reason": None, "varianceComponentA": A, "rangeComponentB": B,
            "legalContrastBounds": [low, high], "pointwiseRadius": radius,
            "untruncatedInterval": raw, "interval": [max(low, raw[0]), min(high, raw[1])]}


def source_slope_domain(design):
    rows = []
    for config in expand_templates(design):
        if config.condition_id not in ("C2", "C3", "C4"): continue
        means = [belief_moments(p)[0] for p in initial_log_beliefs(config)]
        for reversed_side in (False, True):
            displayed = means[::-1] if reversed_side else means
            x = (displayed[0]-displayed[1])/(config.total_millihours/1000)
            tolerance = ATOL+RTOL*abs(x)
            rows.append({"conditionId": config.condition_id, "templateId": config.template_id,
                "scale": config.scale, "gapOffset": config.gap_offset, "sideReversed": reversed_side,
                "initialBudget": config.total_millihours/1000,
                "meanAInitial": displayed[0], "meanBInitial": displayed[1], "xInitial": x,
                "attestationTolerance": tolerance, "safeMagnitude": abs(x)-tolerance})
    a, safe = min(abs(r["xInitial"]) for r in rows), min(r["safeMagnitude"] for r in rows)
    if safe <= 0 or not math.isfinite(safe): raise ValueError("Source-global positive predictor bound unavailable")
    ideal = 1/(2*safe); bound = math.ceil(ideal)+1
    return {"schema": "source-full-endpoint-slope-domain-v1", "context": CONTEXT,
        "canonicalMinimumMagnitude": a, "acceptedInputSafeMagnitude": safe,
        "SxxLowerBound": 10*safe**2, "idealSlopeAbsoluteBound": ideal,
        "numericSlopeAbsoluteBound": bound, "numericBounds": [-float(bound), float(bound)],
        "rows": rows, "sourceTemplateOrientationCount": len(rows), "atol": ATOL, "rtol": RTOL,
        "proof": "5 predictors >= aSafe and 5 <= -aSafe; Sxx>=10*aSafe^2; Syy<=10/4",
        "numericSlackRule": "ceil(idealBound)+1", "rawSolverModified": False,
        "machineCertified": False, "outOfDomainSupportedCoefficient": "raise_not_exclude"}


def _near(actual, expected, name):
    if actual is None or not math.isfinite(actual) or abs(actual-expected) > ATOL+RTOL*abs(expected):
        raise ValueError(f"Initial source attestation mismatch: {name}")


def attest_initial_rows(rows, domain):
    templates = {(r["conditionId"], r["templateId"], r["sideReversed"]): r for r in domain["rows"]}
    checked = []
    for original in rows:
        row = dict(original)
        if row["stage"] != "main" or row["conditionId"] not in ("C2", "C3", "C4"):
            checked.append(row); continue
        if "sideReversed" not in row: raise ValueError("Missing explicit displayed orientation")
        row["sideReversed"] = _boolean(row["sideReversed"], "sideReversed")
        key = (row["conditionId"], row.get("templateId"), row["sideReversed"])
        if key not in templates: raise ValueError("Illegal source main template")
        expected = templates[key]
        for name in ("scale", "gapOffset", "initialBudget", "meanAInitial", "meanBInitial"):
            if name not in row: raise ValueError(f"Missing initial source field {name}")
            row[name] = _number(row[name], name)
            # Categorical recipe values and resource millihours are exact.
            if name in ("scale", "gapOffset", "initialBudget"):
                if row[name] != expected[name]: raise ValueError(f"Exact source recipe mismatch {name}")
            else: _near(row[name], expected[name], name)
        schedule_index = _number(row.get("scheduleIndex"), "scheduleIndex")
        if schedule_index != row["stageIndex"]+3: raise ValueError("Illegal main schedule index")
        row["scheduleIndex"] = int(schedule_index)
        for name in ("initialNeedGapRaw", "initialNeedGap"):
            if name not in row: raise ValueError(f"Missing {name}")
            row[name] = _number(row[name], name)
            _near(row[name], expected["meanAInitial"]-expected["meanBInitial"], name)
        if "initialTie" not in row or _boolean(row["initialTie"], "initialTie"):
            raise ValueError("Source initial predictor is not a tie")
        if row["xInitial"] is not None:
            _near(row["xInitial"], expected["xInitial"], "xInitial")
            if (row["xInitial"] > 0) != row["sideReversed"]:
                raise ValueError("Initial sign/orientation disagrees")
        if row.get("scheduleVariant") not in (None, ""):
            variant = _number(row["scheduleVariant"], "scheduleVariant")
            if variant != int(variant) or not 0 <= variant <= 8: raise ValueError("Invalid external variant")
            row["scheduleVariant"] = int(variant)
        checked.append(row)
    return checked


def _schedule_reason(main):
    if len(main) != 10 or {r["stageIndex"] for r in main} != set(range(1, 11)):
        return "incomplete_ten_main_records"
    ordered = sorted(main, key=lambda r: r["stageIndex"])
    if sum(r["sideReversed"] for r in ordered) != 5:
        return "source_schedule_not_balanced_five_per_orientation"
    if any(len({r["scale"] for r in ordered[i:i+4]}) == 1 for i in range(7)):
        return "source_schedule_four_consecutive_same_scale"
    cid = ordered[0]["conditionId"]
    counts = Counter(r["templateId"] for r in ordered)
    if cid in ("C2", "C4"):
        expected = {f"{cid}_s{si}_g{gi}" for si in range(3) for gi in range(3)}
        if set(counts) != expected or sorted(counts.values()) != [1]*8+[2]:
            return "source_schedule_illegal_nine_cell_plus_one_multiset"
        extra = next(k for k, v in counts.items() if v == 2)
        variant = 3*int(extra.split("_s")[1][0])+int(extra.split("_g")[1])
        variants = {variant}
    else:
        if set(counts) != {f"C3_s{i}_g0" for i in range(3)} or sorted(counts.values()) != [3, 3, 4]:
            return "source_schedule_illegal_three_scale_multiset"
        extra = next(k for k, v in counts.items() if v == 4)
        variants = {int(extra.split("_s")[1][0])+3*i for i in range(3)}
    supplied = {r["scheduleVariant"] for r in ordered if r.get("scheduleVariant") not in (None, "")}
    if len(supplied) > 1 or supplied and not supplied <= variants:
        return "external_variant_disagrees_with_source_multiset"
    return None


def initial_projection_values(rows, domain, predictor="xInitial", model="unadjusted", context=CONTEXT):
    people = defaultdict(list)
    for r in rows: people[r["participantId"]].append(r)
    out = []
    for p, own in sorted(people.items()):
        main = [r for r in own if r["analysisEligible"]]
        base = {"participantId": p, "sessionId": own[0]["sessionId"], "conditionId": own[0]["conditionId"],
            "endpoint": "slope_initial", "context": context, "mainTrials": len(main),
            "status": "unavailable", "value": None, "observations": sum(r["shareA"] is not None and r["xInitial"] is not None for r in main),
            "eligibilityChangedFromPrimary": True, "numericKnownBounds": domain["numericBounds"]}
        reason = None
        if predictor != "xInitial" or model != "unadjusted" or context != CONTEXT:
            reason = "method_unsupported_partial_terminal_or_adjusted_context"
        elif base["conditionId"] in ("C1", "C5"):
            reason = "retained_design_initial_predictor_structurally_constant"
        else:
            reason = _schedule_reason(main)
            if reason is None and base["observations"] != 10:
                reason = "method_unsupported_missing_initial_or_share_endpoint"
        base["completeTen"] = len(main) == 10 and {r["stageIndex"] for r in main} == set(range(1, 11))
        base["endpointMissingWithinCompleteHistory"] = base["completeTen"] and base["observations"] != 10
        if reason:
            out.append({**base, "reason": reason}); continue
        result = participant_slope(sorted(main, key=lambda r: r["stageIndex"]), "xInitial")
        if result["status"] == "available" and not domain["numericBounds"][0] <= result["value"] <= domain["numericBounds"][1]:
            raise ValueError("Supported raw projection outside declared range: HALT, no outcome-selected exclusion")
        out.append({**base, **result})
    return out


def fit_variance_adaptive(rows, design, origin, purpose, geometry=GEOMETRY, policy="global_holm", context=CONTEXT):
    if policy not in POLICIES: raise ValueError("Unknown fixed correction")
    checked = validate_geometry(rows, origin, purpose, geometry)
    domain = source_slope_domain(design)
    checked = attest_initial_rows(checked, domain)
    people, tests = [], []
    initial = initial_projection_values(checked, domain, context=context)
    for identifier, family, endpoint, weights, side in CORE:
        values = initial if endpoint == "slope_initial" else participant_values(checked, endpoint)
        ranges = {c: domain["numericBounds"] if endpoint == "slope_initial" else BOUNDS[endpoint] for c in weights}
        result = empirical_bernstein_contrast(values, weights, ranges, side)
        tests.append({**result, "id": identifier, "family": family, "endpoint": endpoint,
                      "eligibility": CONTEXT if endpoint == "slope_initial" else "complete_ten_records_observed_endpoints"})
        for v in values: people.append({**v, "testId": identifier})
    return {"schema": "tutoring-variance-adaptive-fit-v1", "method": METHOD,
        "origin": origin, "purpose": purpose, "primaryReplacement": False,
        "domain": domain, "numericRepresentation": REPRESENTATION, "participantEstimates": people,
        "correction": multiplicity(tests, policy), "context": context,
        "warning": "Optional sensitivity; source declaration is not authentication or approval. Different slope eligibility; no original-population MNAR repair."}


def cohort_identification(observed, enrolled, bounds, weights, potential_target):
    """Finite-cohort identification, not a sampling interval or point imputation."""
    if potential_target != "finite_enrolled_cohort_defined_full_potential_endpoint":
        raise ValueError("Explicit defined full-potential finite-cohort target required")
    if not weights or set(enrolled) != set(weights) or set(bounds) != set(weights):
        raise ValueError("Required enrollment/weights/bounds disagree")
    _inputs([], weights, bounds, "greater", 0., .95)
    for N in enrolled.values():
        if not isinstance(N, int) or isinstance(N, bool) or N < 0:
            raise ValueError("ALL required enrollment counts must be non-Boolean nonnegative integers")
    groups = {c: [] for c in weights}; seen = set(); kinds = set()
    for row in observed:
        p, c = row.get("participantId"), row.get("conditionId")
        if not isinstance(p, str) or not p.strip() or p != p.strip() or p in seen or c not in groups:
            raise ValueError("Invalid/repeated observed cohort identity")
        seen.add(p)
        kind = row.get("endpointKind", "mean")
        if kind not in ("mean", "slope"): raise ValueError("Explicit endpointKind must be mean or slope")
        kinds.add(kind)
        if len(kinds) > 1: raise ValueError("A finite-cohort target cannot mix mean and slope endpoints")
        trials = row.get("observedTrialCount", 10)
        value = row.get("value")
        if not isinstance(trials, int) or isinstance(trials, bool) or not 0 <= trials <= 10:
            raise ValueError("Invalid observed potential endpoint/count")
        if trials == 0:
            if value is not None: raise ValueError("Zero observed endpoints cannot supply an imputed value")
            groups[c].append({"interval": list(bounds[c]), "observedTrials": 0, "kind": kind})
            continue
        if not _real(value): raise ValueError("Observed endpoints need a finite value")
        if kind == "slope" and trials != 10:
            raise ValueError("A partial projection is not an observed full-potential slope")
        if not bounds[c][0] <= value <= bounds[c][1]: raise ValueError("Observed cohort endpoint outside domain")
        lo, hi = bounds[c]
        groups[c].append({"interval": [(trials*value+(10-trials)*lo)/10, (trials*value+(10-trials)*hi)/10],
                          "observedTrials": trials, "kind": kind})
    arm_results = {}
    for c, N in enrolled.items():
        if N < len(groups[c]): raise ValueError("Enrollment count smaller than identified people")
        if N == 0:
            return {"status": "unavailable", "reason": "required_arm_zero_enrollment_no_defined_mean", "interval": None,
                    "confidenceInterval": False, "target": potential_target}
        lo, hi = bounds[c]; M = len(groups[c])
        interval = [(math.fsum(v["interval"][0] for v in groups[c])+(N-M)*lo)/N,
                    (math.fsum(v["interval"][1] for v in groups[c])+(N-M)*hi)/N]
        if not all(math.isfinite(x) for x in interval): raise ValueError("Nonfinite cohort interval")
        arm_results[c] = {"enrolled": N, "identifiedPeopleWithDeclaredEndpointCounts": M,
            "fullyObservedPeople": sum(v["observedTrials"] == 10 for v in groups[c]),
            "partiallyObservedPeople": sum(0 < v["observedTrials"] < 10 for v in groups[c]),
            "identifiedPeopleWithZeroObservedEndpoints": sum(v["observedTrials"] == 0 for v in groups[c]),
            "whollyUnobservedPeople": N-M, "observedTrialEndpointCount": sum(v["observedTrials"] for v in groups[c]),
            "potentialTrialEndpointCount": N*10, "interval": interval, "knownBounds": list(bounds[c])}
    low = math.fsum(weights[c]*arm_results[c]["interval"][0 if weights[c] > 0 else 1] for c in weights)
    high = math.fsum(weights[c]*arm_results[c]["interval"][1 if weights[c] > 0 else 0] for c in weights)
    return {"status": "available", "reason": None, "target": potential_target, "interval": [low, high],
            "arms": arm_results, "weights": weights, "endpointKind": next(iter(kinds)) if kinds else "externally_defined_no_observed_endpoints",
            "confidenceInterval": False, "superpopulationRepair": False,
            "zeroCompatible": low <= 0 <= high, "positiveCompatible": high > 0, "negativeCompatible": low < 0,
            "assumption": "external enrollment and full-potential endpoint definition; declarations not authentication"}
