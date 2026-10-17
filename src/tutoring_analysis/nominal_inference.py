"""Person-level nominal inference for the finite expected-GPA analysis.

Every designated test is marginal alpha .05. Supporting analyses cannot rescue
an unavailable or unfavorable primary. No familywise/independence claim is made.
"""
from __future__ import annotations

import math
import struct
from collections import defaultdict

from .expected_outcomes import (ROW_SCHEMA, PROFILE, ANALYSIS_ID,
                                allocation_metrics, digest, finite, integer)

FROZEN_CLAIMS = (
    ("H01_Q_C3_minus_C2", "Q", {"C3": 1, "C2": -1}, "greater"),
    ("H02_Q_C5_minus_C4", "Q", {"C5": 1, "C4": -1}, "greater"),
    ("H03_initial_C2", "initial_slope", {"C2": 1}, "greater"),
    ("H04_initial_C3", "initial_slope", {"C3": 1}, "greater"),
    ("H05_initial_C4", "initial_slope", {"C4": 1}, "less"),
    ("H06_R_EGPA_C2", "R_EGPA_pp", {"C2": 1}, "greater"),
    ("H07_R_EGPA_C3", "R_EGPA_pp", {"C3": 1}, "greater"),
    ("H08_lower_C4", "L_lower_mean_share", {"C4": 1}, "greater"),
    ("H09_lower_C5", "L_lower_mean_share", {"C5": 1}, "greater"),
)
MEAN_ENDPOINTS = ("Q", "R_EGPA_pp", "R_surplus_pp", "D_ES_pp", "D_EGPA_pp", "D_surplus_pp", "L_lower_mean_share")
SLOPES = {"initial_slope": "x0", "terminal_slope": "xT", "raw_initial_slope": "rawInitialGap"}


def validate_spec(spec):
    if (spec.get("schema") != "person-level-nominal-spec-v1"
            or spec.get("analysisId") != ANALYSIS_ID or spec.get("profileId") != PROFILE
            or spec.get("questionLimitRule") != "budget_only"
            or spec.get("primaryNull") != 0 or spec.get("alpha") != .05
            or spec.get("confidence") != .95):
        raise ValueError("Unsupported finite-model inference semantics")
    actual = tuple((c["id"], c["endpoint"], c["weights"], c["tail"]) for c in spec["claims"])
    if actual != FROZEN_CLAIMS or not all(c.get("primary") is True for c in spec["claims"]):
        raise ValueError("Fixed primary roster/tail changed")
    if spec.get("diagnostics") != [{"id": "D10_R_EGPA_C1", "endpoint": "R_EGPA_pp",
            "weights": {"C1": 1}, "tail": "less", "primary": False}]:
        raise ValueError("D10 remains a separate conditional diagnostic")
    if spec.get("support") != {"minimumPairs": 3, "minimumLevels": 2, "SxxStrictlyAbove": 1e-12,
                               "minimumPeoplePerArm": 2, "completeMainIndices": list(range(1, 11))}:
        raise ValueError("Fixed person/support population changed")
    return spec


def _unsupported(reason, **details):
    return {"status": "unavailable", "value": None, "reason": reason, **details}


def canonical_mean(values):
    """One specified binary64 map for scalar, vector and finite truth endpoints.

    Sum in main-index order using separate NumPy add operations, then divide.
    The last axis is rounds. This is not data-dependent recentering.
    """
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] == 0:
        raise ValueError("Mean needs a nonempty last axis")
    total = np.zeros(a.shape[:-1], dtype=np.float64)
    for i in range(a.shape[-1]):
        total = np.add(total, a[..., i])
    return np.divide(total, np.float64(a.shape[-1]))


def canonical_projection(weights, responses):
    """Fixed linear projection plan; exactly the same ordered arithmetic map."""
    import numpy as np
    w, y = np.asarray(weights, dtype=np.float64), np.asarray(responses, dtype=np.float64)
    if w.ndim < 1 or y.ndim < 1 or not y.shape[-1] or w.shape[-1] != y.shape[-1]:
        raise ValueError("Projection plan/response shape mismatch")
    value = np.zeros(np.broadcast_shapes(w.shape[:-1], y.shape[:-1]), dtype=np.float64)
    for i in range(y.shape[-1]):
        product = np.multiply(w[..., i], y[..., i])
        value = np.add(value, product)
    return value


def _projection_plan(rows, predictor, adjusted=None):
    import numpy as np
    pairs = [(r, r.get(predictor), r.get("share")) for r in rows]
    if any(not finite(x) or not finite(y) for _, x, y in pairs):
        return _unsupported("undefined_endpoint_pair", observations=len(rows))
    x = np.asarray([p[1] for p in pairs], dtype=float)
    y = np.asarray([p[2] for p in pairs], dtype=float)
    n, levels = len(x), len(set(x.tolist()))
    xc = x - x.mean() if n else x
    sxx = float(xc @ xc)
    span = float(x.max() - x.min()) if n else None
    detail = {"observations": n, "predictorLevels": levels, "predictorSpan": span, "Sxx": sxx}
    if n < 3 or levels < 2 or sxx <= 1e-12:
        return _unsupported("insufficient_predictor_support", **detail)
    columns = [np.ones(n), xc]
    names = ["intercept", predictor]
    if adjusted:
        names += ["mainIndex"]
        columns.append(np.asarray([r["mainIndex"] for r in rows], dtype=float))
        if adjusted == "order_scale_gap":
            names += ["scale", "gapOffset"]
            if any(not finite(r.get(k)) for r in rows for k in ("scale", "gapOffset")):
                return _unsupported("missing_named_covariate", **detail)
            columns.extend(np.asarray([r[k] for r in rows], dtype=float) for k in ("scale", "gapOffset"))
    matrix = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(matrix))
    df = n - matrix.shape[1]
    detail.update({"rank": rank, "namedTerms": names, "residualDf": df})
    if rank != matrix.shape[1] or df <= 0:
        return _unsupported("full_named_model_rank_or_df_failure", **detail)
    # Freeze the linear map once, independently of responses. Scalar metrics,
    # lookup truth enumeration and MC all apply these weights in the same order.
    weights = np.linalg.lstsq(matrix, np.eye(n), rcond=None)[0][1]
    return {"status": "available", "reason": None, "weights": weights, **detail}


def _slope(rows, predictor, adjusted=None):
    plan = _projection_plan(rows, predictor, adjusted)
    if plan["status"] != "available":
        return plan
    weights = plan.pop("weights")
    frozen = rows[0].get("initialProjectionPlan") if len(rows) == 10 and predictor == "x0" and adjusted is None else None
    if frozen is not None:
        if any(r.get("initialProjectionPlan") != frozen for r in rows):
            return _unsupported("mixed_frozen_projection_plans")
        payload = {k: v for k, v in frozen.items() if k != "planHash"}
        expected_x = [struct.pack(">d", float(r["x0"])).hex() for r in rows]
        if (frozen.get("schema") != "frozen-projection-plan-v1" or frozen.get("planHash") != digest(payload)
                or frozen.get("predictorBinary64") != expected_x
                or frozen.get("operationOrder") != "ordered_float64_multiply_then_add_mainIndex_1_to_10"):
            return _unsupported("unbound_frozen_projection_plan")
        try:
            bits = frozen["weightBinary64"]
            if not isinstance(bits, list) or len(bits) != 10 or any(not isinstance(b, str) or len(b) != 16 for b in bits):
                raise ValueError("Ten binary64 weights required")
            weights = [struct.unpack(">d", bytes.fromhex(b))[0] for b in bits]
            if not all(math.isfinite(w) for w in weights):
                raise ValueError("Nonfinite frozen weight")
        except (KeyError, TypeError, ValueError, struct.error):
            return _unsupported("invalid_frozen_projection_plan")
    value = float(canonical_projection(weights, [r["share"] for r in rows]))
    if not math.isfinite(value):
        return {**plan, "status": "unavailable", "value": None, "reason": "nonfinite_projection"}
    return {**plan, "value": value, "allocationChangeOverSpan": value * plan["predictorSpan"]}


def _row_reason(row):
    if row.get("metric_schema") != ROW_SCHEMA or row.get("rowValidated") is not True:
        return "unvalidated_metric_row"
    if row.get("profileId") != PROFILE or row.get("analysisId") != ANALYSIS_ID:
        return "wrong_profile_or_analysis"
    if row.get("humanObservations") is not False or row.get("origin") not in ("synthetic_design", "synthetic_validation"):
        return "unreleased_provenance"
    try:
        integer(row["mainIndex"], "main index", 1, 10)
        b0 = integer(row["initialIntegerBudget"], "initial budget", 1)
        cost = integer(row["paidIntegerCost"], "cost", 1)
        count = integer(row["paidCount"], "count")
        bt = integer(row["remainingIntegerBudget"], "remaining budget")
        if count > b0 // cost or bt != b0 - count * cost or row.get("Q") != int(count > 0) or not row.get("paidHistoryValid"):
            return "invalid_paid_budget_identity"
        if any(not isinstance(row.get(k), (list, tuple)) or len(row[k]) != 2 or not all(finite(v) for v in row[k])
               for k in ("initialMeans", "terminalMeans")):
            return "invalid_finite_means"
        expected = allocation_metrics(row, row.get("shareIndex"))
        for key in ("share", "D_ES_pp", "D_EGPA_pp", "R_EGPA_pp", "D_surplus_pp", "R_surplus_pp", "L_lower_mean_share", "lowerMeanTie"):
            if row.get(key) != expected[key]:
                return "metric_native_geometry_mismatch"
        if not finite(row.get("x0")) or row.get("x0") != (row["initialMeans"][0] - row["initialMeans"][1]) / (b0 / 1000):
            return "initial_finite_predictor_mismatch"
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return "invalid_metric_fields"
    return None


def participant_endpoints(rows, endpoint_spec):
    """One record/person; all ten are checked before any endpoint filtering."""
    validate_spec(endpoint_spec)
    groups = defaultdict(list)
    for row in rows:
        person = row.get("participantId")
        if not isinstance(person, str) or not person:
            raise ValueError("Missing participant identity")
        groups[person].append(row)
    people = []
    for person, all_rows in sorted(groups.items()):
        reasons = []
        session_values, arm_values = [r.get("sessionId") for r in all_rows], [r.get("conditionId") for r in all_rows]
        sessions = {v for v in session_values if isinstance(v, str) and v}
        arms = {v for v in arm_values if isinstance(v, str)}
        if (len(sessions) != 1 or any(not isinstance(v, str) or not v for v in session_values)
                or len(arms) != 1 or any(not isinstance(v, str) for v in arm_values)
                or not arms <= {"C1", "C2", "C3", "C4", "C5"}):
            reasons.append("conflicting_person_session_or_arm")
        indices = [r.get("mainIndex") for r in all_rows]
        valid_indices = [i for i in indices if isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= 10]
        if len(valid_indices) != len(indices):
            reasons.append("invalid_main_index")
        if len(set(valid_indices)) != len(valid_indices):
            reasons.append("duplicate_main_index")
        reasons.extend(filter(None, (_row_reason(r) for r in all_rows)))
        present = sorted((r for r in all_rows if r.get("recordPresent") is True),
                         key=lambda r: r["mainIndex"] if isinstance(r.get("mainIndex"), int) and not isinstance(r["mainIndex"], bool) else 0)
        complete = len(valid_indices) == len(indices) and [r["mainIndex"] for r in present] == list(range(1, 11))
        complete_choice = complete and not reasons and all(r.get("choiceAvailable") and r.get("endpointPresent") is True
                                           and r["remainingIntegerBudget"] > 0 for r in present)
        endpoints = {}
        for endpoint in MEAN_ENDPOINTS:
            eligible = complete if endpoint == "Q" else complete_choice
            values = [r.get(endpoint) for r in present]
            if reasons:
                endpoints[endpoint] = _unsupported("quarantined_person")
            elif not eligible or any(not finite(v) for v in values):
                endpoints[endpoint] = _unsupported("incomplete_ten_paid" if endpoint == "Q" else "incomplete_ten_positive_defined_choices")
            else:
                endpoints[endpoint] = {"status": "available", "value": float(canonical_mean(values)), "reason": None, "observations": 10}
        for endpoint, predictor in SLOPES.items():
            endpoints[endpoint] = (_unsupported("quarantined_person") if reasons else
                                   _slope(present, predictor) if complete_choice else
                                   _unsupported("incomplete_ten_positive_defined_choices"))
        valid_choice = [r for r in present if r.get("choiceAvailable") and r.get("endpointPresent") is True]
        tie_free = [r for r in valid_choice if not r.get("lowerMeanTie")]
        endpoints["tie_free_L"] = (_unsupported("quarantined_or_incomplete_choices") if reasons or not complete_choice else
                                     {"status": "available", "value": float(canonical_mean([r["L_lower_mean_share"] for r in tie_free])), "reason": None, "observations": len(tie_free)} if tie_free else
                                     _unsupported("tied_only_person"))
        for kind in ("order", "order_scale_gap"):
            endpoints[kind + "_adjusted_initial"] = (_slope(present, "x0", kind) if complete_choice and not reasons
                                                        else _unsupported("quarantined_or_incomplete_choices"))
        for endpoint in MEAN_ENDPOINTS:
            subset = present if endpoint == "Q" else valid_choice
            values = [r[endpoint] for r in subset if finite(r.get(endpoint))]
            endpoints["available_history_" + endpoint] = (_unsupported("quarantined_or_empty_history") if reasons or not values else
                                                           {"status": "available", "value": float(canonical_mean(values)), "reason": None, "observations": len(values), "changedPopulation": True})
        for endpoint, predictor in SLOPES.items():
            endpoints["available_history_" + endpoint] = _unsupported("quarantined_person") if reasons else _slope(valid_choice, predictor)
        halves = {}
        for label, index_set in (("first", set(range(1, 6))), ("second", set(range(6, 11)))):
            half = [r for r in present if isinstance(r.get("mainIndex"), int) and r["mainIndex"] in index_set]
            halves[label] = {"recordCount": len(half), "Q": None if not half or reasons else math.fsum(r["Q"] for r in half) / len(half),
                             "paidCount": None if not half or reasons else math.fsum(r["paidCount"] for r in half) / len(half),
                             "quarantined": bool(reasons)}
        people.append({"participantId": person, "sessionId": next(iter(sessions)) if len(sessions) == 1 else None,
                       "conditionId": next(iter(arms)) if len(arms) == 1 else None, "endpoints": endpoints,
                       "eligibility": {"recordedRows": len(all_rows), "presentRows": len(present),
                                       "completePaidTen": complete and not reasons, "completeChoiceTen": complete_choice and not reasons,
                                       "quarantineReasons": sorted(set(reasons))}, "descriptiveHalves": halves})
    return people


def test_contrast(person_endpoints, test_spec):
    """Student for one arm; fixed-weight independent Welch otherwise."""
    from scipy.stats import t
    weights = test_spec["weights"]
    if not weights or any(not finite(w) or w == 0 for w in weights.values()):
        raise ValueError("All frozen required weights must be nonzero finite")
    tail = test_spec["tail"]
    if tail not in ("greater", "less", "two-sided") or test_spec.get("null", 0) != 0:
        raise ValueError("Unreleased tail/null")
    alpha, confidence = test_spec.get("alpha", .05), test_spec.get("confidence", .95)
    if alpha != .05 or confidence != .95:
        raise ValueError("Frozen alpha/confidence changed")
    endpoint = test_spec["endpoint"]
    if test_spec.get("primary") is True and (test_spec["id"], endpoint, weights, tail) not in FROZEN_CLAIMS:
        raise ValueError("Unknown/changed designated candidate test")
    arms, seen = defaultdict(list), set()
    for person in person_endpoints:
        pid = person["participantId"]
        if pid in seen:
            raise ValueError("Duplicate independent person")
        seen.add(pid)
        value = person["endpoints"].get(endpoint, {})
        if person["conditionId"] in weights and value.get("status") == "available":
            if not finite(value.get("value")):
                raise ValueError("Available endpoint is not finite")
            arms[person["conditionId"]].append(value["value"])
    counts = {arm: len(arms[arm]) for arm in weights}
    result = {"testId": test_spec["id"], "endpointId": endpoint, "weights": dict(weights), "tail": tail,
              "null": 0, "alpha": alpha, "confidence": confidence, "primary": test_spec.get("primary", False),
              "requiredArmCounts": counts, "effect": None, "SE": None, "df": None, "statistic": None,
              "p": None, "CI": None, "candidateNominalReject": False, "status": "unavailable",
              "reason": None, "method": "Student" if len(weights) == 1 else "fixed_weight_Welch_Satterthwaite"}
    if any(n < 2 for n in counts.values()):
        result["reason"] = "fewer_than_two_eligible_people_in_required_arm"
        return result
    means = {arm: math.fsum(arms[arm]) / counts[arm] for arm in weights}
    variance_terms = {}
    for arm, weight in weights.items():
        variance = math.fsum((x - means[arm]) ** 2 for x in arms[arm]) / (counts[arm] - 1)
        variance_terms[arm] = weight ** 2 * variance / counts[arm]
    effect = math.fsum(weights[a] * means[a] for a in weights)
    se2 = math.fsum(variance_terms.values())
    result.update({"effect": effect, "armMeans": means})
    if not finite(se2) or se2 <= 0:
        result["reason"] = "zero_or_nonfinite_aggregate_SE"
        return result
    denominator = math.fsum(v ** 2 / (counts[a] - 1) for a, v in variance_terms.items())
    df, se = se2 ** 2 / denominator, math.sqrt(se2)
    statistic = effect / se
    p = float(t.sf(statistic, df) if tail == "greater" else t.cdf(statistic, df) if tail == "less" else 2 * t.sf(abs(statistic), df))
    radius = float(t.ppf(.975, df)) * se
    if not all(finite(v) for v in (effect, se, df, statistic, p, radius)):
        result["reason"] = "nonfinite_test_uncertainty"
        return result
    result.update({"SE": se, "df": df, "statistic": statistic, "p": p, "CI": [effect - radius, effect + radius],
                   "candidateNominalReject": p < alpha, "status": "available", "reason": None})
    return result


def analyze(rows, analysis_spec):
    """Analyze supplied validated synthetic rows, without changing the population."""
    validate_spec(analysis_spec)
    rows = list(rows)
    for row in rows:
        if row.get("syntheticRecipeId") is not None or row.get("nativeProfileClaim") is False:
            raise ValueError("Conditional controls cannot enter the working population")
        if row.get("humanObservations") is not False:
            raise ValueError("This interface has no human-session validation adapter")
    people = participant_endpoints(rows, analysis_spec)
    tests = []
    for namespace, specs in (("primary", analysis_spec["claims"]),
                             ("conditional_diagnostic", analysis_spec["diagnostics"]),
                             ("supporting_descriptive", analysis_spec.get("supporting", []))):
        for spec in specs:
            result = test_contrast(people, spec)
            result.update({"namespace": namespace,
                           "personPopulation": analysis_spec["population"]})
            tests.append(result)
    sensitivity = []
    for endpoint in ("tie_free_L", "order_adjusted_initial", "order_scale_gap_adjusted_initial"):
        for arm in ("C1", "C2", "C3", "C4", "C5"):
            sensitivity.append(test_contrast(people, {"id": "S_" + endpoint + "_" + arm,
                "endpoint": endpoint, "weights": {arm: 1}, "tail": "two-sided"}))
    for claim in analysis_spec["claims"]:
        alt = {**claim, "id": "S_available_" + claim["id"],
               "endpoint": "available_history_" + claim["endpoint"], "primary": False}
        sensitivity.append(test_contrast(people, alt))
    for result in sensitivity:
        result.update({"namespace": "supporting_descriptive", "noPrimaryRescue": True})
    return {"schema": "person-level-nominal-result-v1", "analysisId": ANALYSIS_ID,
            "analysisSpecHash": digest(analysis_spec), "humanObservations": False,
            "tests": tests, "supportingSensitivities": sensitivity, "people": people,
            "globalErrorGuarantee": None,
            "scope": "fixed_person_level_statistics_not_full_calibration_or_human_validation"}
