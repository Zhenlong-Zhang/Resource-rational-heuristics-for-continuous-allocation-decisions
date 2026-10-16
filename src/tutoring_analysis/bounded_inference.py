"""Optional known-bound independent-person means, never a Student/Welch fallback.

Hoeffding (1963), Theorem 2; fixed nine slots retain three unsupported slopes.
Declared metric provenance and geometry are not human/source authentication.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from numbers import Real
from pathlib import Path

import numpy as np

from design_validation.bounded_task import project_share
from .inference import (CONDITIONS, CORE, POLICIES, _number, multiplicity,
                        participant_values, validate_rows)

METHOD = "independent_person_Hoeffding_known_bounds"
GEOMETRY = "finite_pilot_metrics_declared"
FIXTURE_NAMESPACE = "fresh_bounded_source_fixture_710060"
UNIT_ROUNDOFF = 2.**-53
TRIAL_R_ENVELOPE = 512*UNIT_ROUNDOFF
PERSON_R_ENVELOPE = 1024*UNIT_ROUNDOFF
IDEAL_BOUNDS = {"anyQuestion": (0., 1.), "EOResemblance": (-50., 50.),
                "lowerNeedDirectional": (-.5, .5)}
BOUNDS = {**IDEAL_BOUNDS, "EOResemblance": (-50.-PERSON_R_ENVELOPE, 50.+PERSON_R_ENVELOPE)}
REPRESENTATION = {"binary64UnitRoundoff": UNIT_ROUNDOFF,
    "rawTrialRAbsoluteEnvelope": TRIAL_R_ENVELOPE, "rawPersonRAbsoluteEnvelope": PERSON_R_ENVELOPE,
    "idealEndpointBounds": IDEAL_BOUNDS, "numericPersonBounds": BOUNDS,
    "rawValuesClamped": False, "boundsFittedToObservedExtrema": False,
    "proofScope": "known raw domain envelopes; subsequent inference is approximate binary64, not certified interval arithmetic"}
EXTRA = ("remainingBudget", "meanATerminal", "meanBTerminal",
         "target_equal_split", "target_equal_outcome")
TOL = 1e-9


def validate_geometry(rows, origin, purpose, geometry):
    if geometry != GEOMETRY:
        raise ValueError("Explicit prevalidated finite-pilot metric geometry declaration required")
    checked = validate_rows(rows, origin, purpose)
    for row in checked:
        for key in EXTRA:
            if key not in row: raise ValueError(f"Missing source geometry field {key}")
            row[key] = _number(row[key], key)
        budget, a, b = (row[k] for k in EXTRA[:3])
        if budget is None or budget < 0 or a is None or b is None or not (1 <= a <= 100 and 1 <= b <= 100):
            raise ValueError("Invalid finite source budget/terminal mean domain")
        tied = abs(a-b) <= TOL*max(1., a, b)
        if row["terminalTie"] != tied: raise ValueError("Terminal tie disagrees with source means")
        share, es, eo = row["shareA"], row["target_equal_split"], row["target_equal_outcome"]
        if budget == 0:
            if any(row[k] is not None for k in ("shareA", "target_equal_split", "target_equal_outcome",
                    "distancePP_equal_split", "distancePP_equal_outcome", "EOResemblance", "lowerNeedDirectional")):
                raise ValueError("Exhausted source budget must have no share/targets/choice endpoints")
            continue
        if es != .5 or eo is None or not 0 <= eo <= 1:
            raise ValueError("Invalid source ES/EO target")
        expected_eo = project_share(.5+(a-b)/(2*budget))
        if abs(eo-expected_eo) > TOL: raise ValueError("EO target disagrees with shared finite/grid projection")
        if share is None:
            if any(row[k] is not None for k in ("distancePP_equal_split", "distancePP_equal_outcome", "EOResemblance", "lowerNeedDirectional")):
                raise ValueError("Missing share cannot have observed choice endpoints")
            continue
        if abs(share*1000-round(share*1000)) > TOL: raise ValueError("Share is outside canonical 1001-point grid")
        expected = {"distancePP_equal_split": 100*abs(share-.5),
                    "distancePP_equal_outcome": 100*abs(share-eo),
                    "EOResemblance": 100*(abs(share-.5)-abs(share-eo)),
                    "lowerNeedDirectional": 0. if tied else share-.5 if a < b else .5-share}
        for key, value in expected.items():
            actual = row[key]
            if actual is not None and abs(actual-value) > TOL:
                raise ValueError(f"{key} disagrees with source share/target/orientation")
        trial_bounds = {**IDEAL_BOUNDS, "EOResemblance": (-50.-TRIAL_R_ENVELOPE, 50.+TRIAL_R_ENVELOPE),
                        "distancePP_equal_split": (0., 50.), "distancePP_equal_outcome": (0., 100.)}
        for key, (lo, hi) in trial_bounds.items():
            if row[key] is not None and not lo <= row[key] <= hi: raise ValueError(f"{key} outside fixed source bounds")
    return checked


def bounded_contrast(values, weights, bounds, side="greater", null=0., confidence=.95):
    """Conservative one-sided tail and pointwise CI with known, not sampled bounds."""
    def real(v): return isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)
    if side not in ("greater", "less") or not real(confidence) or not 0 < confidence < 1 or not real(null):
        raise ValueError("Invalid fixed side/null/confidence")
    if not weights or set(bounds) != set(weights): raise ValueError("Required weights and bounds must match")
    for cid, weight in weights.items():
        if cid not in CONDITIONS or not real(weight) or weight == 0:
            raise ValueError("Invalid condition weight")
        domain = bounds[cid]
        if not isinstance(domain, (tuple, list, np.ndarray)) or len(domain) != 2 or any(not real(v) for v in domain) or domain[0] >= domain[1]:
            raise ValueError("Known nonzero finite ranges required")
    groups = {cid: [] for cid in weights}
    seen = set()
    for row in values:
        p, cid = row.get("participantId"), row.get("conditionId")
        if not isinstance(p, str) or not p.strip() or p != p.strip() or cid not in CONDITIONS:
            raise ValueError("Invalid independent-person identity")
        if p in seen: raise ValueError("Repeated independent-person identity")
        seen.add(p)
        if row.get("status") not in ("available", "unavailable"): raise ValueError("Invalid person status")
        value = row.get("value")
        if row["status"] == "unavailable":
            if value is not None: raise ValueError("Unavailable person cannot supply endpoint")
            continue
        if not real(value): raise ValueError("Invalid person endpoint")
        if cid in groups:
            if not bounds[cid][0] <= value <= bounds[cid][1]: raise ValueError("Person endpoint outside known domain")
            groups[cid].append(float(value))
    summary = {cid: {"participants": len(v), "mean": float(np.mean(v)) if v else None,
                       "observedVarianceDiagnostic": float(np.var(v, ddof=1)) if len(v) > 1 else None,
                       "knownBounds": list(bounds[cid])} for cid, v in groups.items()}
    base = {"method": METHOD, "weights": weights, "side": side, "null": null,
            "confidence": confidence, "groups": summary, "status": "unavailable", "reason": None,
            "estimate": None, "rangeProxyV": None, "pValue": None, "logPValue": None,
            "pValueUnderflow": False, "interval": None, "untruncatedInterval": None,
            "pointwiseRadius": None, "legalContrastBounds": None,
            "independentUnit": "participant", "weighting": "equal participant within condition; fixed condition weights",
            "boundsEstimatedFromSample": False, "intervalSimultaneous": False}
    if any(not v for v in groups.values()): return {**base, "reason": "empty_required_condition"}
    try:
        estimate = math.fsum(weights[c]*summary[c]["mean"] for c in weights)
        V = math.fsum(weights[c]**2*(bounds[c][1]-bounds[c][0])**2/len(groups[c]) for c in weights)
        low = math.fsum(weights[c]*bounds[c][0 if weights[c] > 0 else 1] for c in weights)
        high = math.fsum(weights[c]*bounds[c][1 if weights[c] > 0 else 0] for c in weights)
    except (OverflowError, ValueError) as exc: raise ValueError("Nonfinite known-bound contrast arithmetic") from exc
    if not all(math.isfinite(x) for x in (estimate, V, low, high)) or V <= 0:
        raise ValueError("Nonfinite known-bound contrast arithmetic")
    signed = estimate-null if side == "greater" else null-estimate
    log_p = -2*(signed/math.sqrt(V))**2 if signed > 0 else 0.
    radius = math.sqrt(V*(math.log(2.)-math.log1p(-confidence))/2)
    raw = [estimate-radius, estimate+radius]
    if not all(math.isfinite(x) for x in (log_p, radius, *raw)): raise ValueError("Nonfinite tail/interval arithmetic")
    probability = math.exp(log_p)
    return {**base, "status": "available", "estimate": estimate, "rangeProxyV": V,
            "pValue": probability, "logPValue": log_p,
            "pValueUnderflow": bool(probability == 0. and math.isfinite(log_p)),
            "interval": [max(low, raw[0]), min(high, raw[1])], "untruncatedInterval": raw,
            "pointwiseRadius": radius, "legalContrastBounds": [low, high]}


def fit_bounded(rows, origin, purpose, geometry, policy="global_holm", alpha=.05):
    checked = validate_geometry(rows, origin, purpose, geometry)
    cache, people, tests = {}, [], []
    for identifier, family, endpoint, weights, side in CORE:
        if endpoint not in BOUNDS:
            result = {"method": METHOD, "status": "unavailable", "reason": "method_unsupported_initial_projection",
                      "estimate": None, "pValue": None, "logPValue": None, "pValueUnderflow": False,
                      "interval": None, "untruncatedInterval": None, "rangeProxyV": None,
                      "pointwiseRadius": None, "groups": {}, "weights": weights, "side": side}
        else:
            if endpoint not in cache:
                cache[endpoint] = participant_values(checked, endpoint)
                people.extend(cache[endpoint])
            result = bounded_contrast(cache[endpoint], weights, {c: BOUNDS[endpoint] for c in weights}, side)
        tests.append({**result, "id": identifier, "family": family, "endpoint": endpoint, "role": "optional_sensitivity_core_slot",
                      "eligibility": "complete_ten", "window": "all", "primaryReplacement": False,
                      "confirmatoryRescueAllowed": False})
    correction = multiplicity(tests, policy, alpha)
    return {"schema": "tutoring-bounded-mean-inference-v1", "method": METHOD, "origin": origin, "purpose": purpose,
            "metricGeometryDeclaration": geometry, "sourceAuthenticated": False,
            "numericalRepresentation": REPRESENTATION,
            "humanEvidence": False if origin == "synthetic_design" else "declared_not_authenticated",
            "primaryReplacement": False, "statisticalCalibrationPerformed": False,
            "primaryEligibility": "complete_ten_unique_main_indices_1_to_10_not_endpoint_complete",
            "populationTarget": "included_person_observed_endpoint_expectations_under_declared_selection_rule",
            "inputRows": len(checked), "practiceRowsExcluded": sum(r["stage"] == "practice" for r in checked),
            "participants": len({r["participantId"] for r in checked}), "participantEstimates": people,
            "tests": correction["tests"], "multiplicity": {k: v for k, v in correction.items() if k != "tests"},
            "allSixSupportedSuccess": all(t["reject"] for t in correction["tests"] if t["endpoint"] in BOUNDS),
            "limitations": ["Known-range independent-person concentration can be conservative, even with n=1/zero observed variance.",
                "Three initial C2/C3/C4 projections are method-unsupported, not design-constant; all-nine success is impossible.",
                "Declared source geometry/provenance does not authenticate enrollment, replay or human readiness.",
                "Complete record history is not complete endpoint observation; selection changes targets and is not repaired.",
                "No slope, supporting-window, terminal-association or original Student/Welch failure rescue."]}


def source_fixture_rows(people_per_condition=2, case="full", root=None):
    """Fresh deterministic finite-source geometry, NOT recorded QA or policy calibration."""
    from design_validation.bounded_task import BoundedTask, expand_templates, realized_utility
    import json
    from .finite_metrics import metrics
    if not isinstance(people_per_condition, int) or isinstance(people_per_condition, bool) or people_per_condition < 1:
        raise ValueError("Positive fixture person count required")
    if case not in ("full", "complete_nine_choice_endpoints", "partial_history", "exhausted_budget", "tie_and_orientation"):
        raise ValueError("Unknown fixed source fixture case")
    root = Path(root or Path(__file__).resolve().parents[2])
    design = json.loads((root/"configs/tutoring_conditions.json").read_text())
    templates = expand_templates(design); rows = []
    for cid in CONDITIONS:
        cs = [c for c in templates if c.condition_id == cid]
        for p in range(people_per_condition):
            for index in range(1, 12):
                main = index <= 10; stage, ix = ("main", index) if main else ("practice", 1)
                c = cs[(index-1) % len(cs)]; task = BoundedTask(c)
                reverse = bool((index+p) % 2); logs = [x.copy() for x in task.initial_logs]
                if reverse: logs.reverse()
                initial = [x.copy() for x in logs]; q = int((index+p) % 3 == 0)
                if q: logs[0] = task.posterior(logs[0], 20+(index*7+p*3) % 65)
                budget = c.total_millihours-q*c.cost_millihours
                if case == "exhausted_budget" and index == 1: budget = 0
                if case == "tie_and_orientation" and index == 1: logs[1] = logs[0].copy()
                share_index = (0, 250, 500, 750, 1000)[(index+p-1) % 5]
                share = None if not budget else share_index/1000
                total = realized_utility(c, share, budget, [40, 60])
                t = {"participantId": f"{FIXTURE_NAMESPACE}-{cid}-{p}", "sessionId": f"{FIXTURE_NAMESPACE}-session-{cid}-{p}",
                     "conditionId": cid, "trialId": f"{stage}-{ix:02d}", "committedNotAdvanced": False,
                     "initialLogs": [x.tolist() for x in initial], "terminalLogs": [x.tolist() for x in logs],
                     "paid": [{"student": 0}] if q else [], "events": [],
                     "definition": {"stage": stage, "stageIndex": ix, "scheduleIndex": index-1,
                         "templateId": c.template_id, "scale": c.scale, "gap": c.gap_offset,
                         "sideReversed": reverse, "priorCentres": c.prior_centres, "priorSpreads": c.prior_spreads,
                         "reportWidth": c.report_width, "budgetMillihours": c.total_millihours, "costMillihours": c.cost_millihours},
                     "result": {"remainingMillihours": budget, "shareIndex": share_index,
                         "assessmentTimeMillihours": q*c.cost_millihours, "stopReason": "source_fixture",
                         "totalUtility": total, "totalGPA": 7+c.gpa_scale*total}}
                r = metrics(t); r.update(dataRole="synthetic_design", analysisEligible=main, exclusionReasons="",
                    fixturePurpose="fresh_finite_geometry_functionality_not_replay_or_policy_calibration")
                if case == "complete_nine_choice_endpoints" and index == 1:
                    for key in ("shareA", "EOResemblance", "lowerNeedDirectional", "distancePP_equal_split", "distancePP_equal_outcome"): r[key] = None
                if case != "partial_history" or not main or p != 0 or index <= 4: rows.append(r)
    return rows
