"""Person-level and stratified reductions for cost-aware tutoring histories.

Numerical reductions preserve the original ordered person arithmetic, eligibility
and equal-variant Monte Carlo estimands. Transport identities are new public
identities, not historical execution receipts. This module makes no human
inference or policy-optimality claim.
"""
import hashlib
import math
import numpy as np

from .cost_aware_inputs import (bits, compact, unbits, log_bits, require as need)
from ..tutoring_analysis.expected_outcomes import (
    allocation_metrics, binary_payload, digest, finite_mean as _accepted_finite_mean)
from ..tutoring_analysis.nominal_inference import canonical_mean, canonical_projection

MEANS = ("Q", "R_EGPA_pp", "R_surplus_pp", "D_ES_pp", "D_EGPA_pp", "D_surplus_pp", "L_lower_mean_share")
SLOPES = {"initial_slope": "x0", "terminal_slope": "xT", "raw_initial_slope": "rawInitialGap"}
PERFORMANCE = ("expectedUtility", "expectedCombinedGPA", "expectedObjective", "realizedUtility",
               "realizedCombinedGPA", "realizedObjective", "paidCount", "terminalSeconds")


def finite_mean(logs):
    """Preserve the accepted list arithmetic at the native ndarray boundary."""
    if isinstance(logs, np.ndarray):
        need(type(logs) is np.ndarray and logs.ndim == 1 and logs.shape == (100,)
             and logs.dtype == np.dtype(np.float64) and logs.dtype.isnative,
             "Exactly100 native float64 ndarray log weights required")
        logs = logs.tolist()
    return _accepted_finite_mean(logs)


def semantic(value):
    return hashlib.sha256(compact(binary_payload(value)).encode("utf-8")).hexdigest()


def decode_logs(rows):
    need(type(rows) is list and len(rows) == 2 and all(len(row) == 100 for row in rows),
         "Two complete log-belief vectors required")
    return [np.asarray([unbits(x) for x in row], dtype=np.float64) for row in rows]


def reference(record):
    validate_record(record)
    return {"recordId": record["recordId"], "semanticHash": record["semanticHash"]}


def make_record(kind, manifest_hash, identity, payload):
    need(kind in ("root", "history", "decision", "trial-metrics", "person", "estimand", "performance"),
         "Unknown reduction record")
    reserved = {"schema", "manifestHash", "recordId", "semanticHash", "historicalReceipt"}
    need(not reserved.intersection(payload), "Payload overwrites identity")
    record = {"schema": "finite-cost-aware-reduction-" + kind + "-v1",
              "manifestHash": manifest_hash, "recordId": semantic(identity),
              "historicalReceipt": False, **payload}
    record["semanticHash"] = semantic(record)
    return record


def validate_record(record):
    need(type(record) is dict and record.get("historicalReceipt") is False
         and record.get("schema", "").startswith("finite-cost-aware-reduction-"),
         "Public reduction record required")
    need(record.get("semanticHash") == semantic({k: v for k, v in record.items() if k != "semanticHash"}),
         "Reduction record integrity")
    return record


def endpoint_specs(analysis):
    for namespace, key in (("primary", "claims"), ("diagnostic", "diagnostics"), ("supporting", "supporting")):
        for item in analysis[key]: yield namespace, item
    for endpoint in ("tie_free_L", "order_adjusted_initial", "order_scale_gap_adjusted_initial"):
        for arm in ("C1", "C2", "C3", "C4", "C5"):
            yield "supporting", {"id": "S_"+endpoint+"_"+arm, "endpoint": endpoint, "weights": {arm: 1}, "tail": "two-sided"}
    for claim in analysis["claims"]:
        yield "supporting", {**claim, "id": "S_available_"+claim["id"], "endpoint": "available_history_"+claim["endpoint"], "primary": False}


def initial_plan(rows, predictor="x0", adjusted=None):
    """Inspected centered OLS/SVD rules; responses never construct a plan."""
    x = np.asarray([r.get(predictor) for r in rows], dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all(): return None, "undefined_endpoint_pair"
    xc, n = x-x.mean(), len(x)
    detail = {"observations": n, "predictorLevels": len(set(x.tolist())), "predictorSpan": float(x.max()-x.min()), "Sxx": float(xc @ xc)}
    if n < 3 or detail["predictorLevels"] < 2 or detail["Sxx"] <= 1e-12: return None, "insufficient_predictor_support"
    columns, names = [np.ones(n), xc], ["intercept", predictor]
    if adjusted:
        columns.append(np.asarray([r["mainIndex"] for r in rows], dtype=float)); names.append("mainIndex")
        if adjusted == "order_scale_gap":
            names.extend(("scale", "gapOffset"))
            columns.extend(np.asarray([r[k] for r in rows], dtype=float) for k in ("scale", "gapOffset"))
    matrix = np.column_stack(columns)
    detail.update(rank=int(np.linalg.matrix_rank(matrix)), namedTerms=names, residualDf=n-matrix.shape[1])
    if detail["rank"] != matrix.shape[1] or detail["residualDf"] <= 0: return None, "full_named_model_rank_or_df_failure"
    weights = np.linalg.lstsq(matrix, np.eye(n), rcond=None)[0][1]
    payload = {"schema": "r9-frozen-projection-plan-v1", "predictorBinary64": [bits(float(v)) for v in x],
               "weightBinary64": [bits(float(v)) for v in weights],
               "operationOrder": "ordered_float64_multiply_then_add_mainIndex_1_to_10", "diagnostics": {"status": "available", "reason": None, **detail}}
    return {**payload, "planHash": digest(payload)}, None

def model_trial_metrics(history, root, final_summary):
    validate_record(history); need(history["status"] == "completed" and history["root"] == reference(root), "Closed bound model history required")
    budget = history["finalBudget"]; c = root["config"]
    need(budget == c["total_millihours"]-c["cost_millihours"]*history["paidCount"], "Integer paid identity")
    means = [finite_mean(x) for x in decode_logs(history["finalLogBinary64"])]
    macros = final_summary["macros"]
    row = {"trialId": history["trialId"], "personId": history["personId"], "populationId": root["populationId"], "scenarioId": history["scenarioId"], "conditionId": root["conditionId"], "variant": root["variant"], "mainIndex": root["mainIndex"], "initialIntegerBudget": c["total_millihours"], "paidIntegerCost": c["cost_millihours"], "remainingIntegerBudget": budget, "paidCount": history["paidCount"], "Q": int(history["paidCount"] > 0), "initialMeans": root["initialMeans"], "terminalMeans": means, "rawInitialGap": root["initialMeans"][0]-root["initialMeans"][1], "x0": root["x0"], "xT": None if budget == 0 else (means[0]-means[1])/(budget/1000), "jEGPA": macros["EO_expected_GPA"]["index"], "jSurplus": macros["EO_raw_surplus"]["index"], "jAL": macros["AL"]["index"], "jML": macros["ML"]["index"], "gapOffset": root["gapOffset"], "scale": c["scale"], "initialProjectionPlan": root["initialProjectionPlan"]}
    row.update(allocation_metrics(row, history["shareIndex"]))
    return make_record("trial-metrics", root["manifestHash"], [history["trialId"], "metrics", history["semanticHash"]], row)


def _endpoint(value=None, reason=None, observations=0, plan=None, status=None, diagnostics=None):
    return {"status": status or ("available" if value is not None else "scientific_ineligible"), "value": value, "reason": reason, "observations": observations, "planHash": None if plan is None else plan["planHash"], "diagnostics": diagnostics or {}}


def model_person_endpoints(rows, analysis_spec, initial_projection_plan):
    need(len(rows) == 10 and [r["mainIndex"] for r in rows] == list(range(1, 11)), "Ten unique ordered model metrics")
    for row in rows: validate_record(row)
    need(len({(r["personId"], r["populationId"], r["scenarioId"], r["conditionId"], r["variant"]) for r in rows}) == 1, "One model person/population")
    positive = [r for r in rows if r["choiceAvailable"] and r["remainingIntegerBudget"] > 0]
    complete_choice = len(positive) == 10
    endpoints = {}
    for endpoint in MEANS:
        eligible = endpoint == "Q" or complete_choice
        endpoints[endpoint] = _endpoint(float(canonical_mean([r[endpoint] for r in rows])), observations=10) if eligible else _endpoint(reason="incomplete_ten_positive_defined_choices")
        subset = rows if endpoint == "Q" else positive
        endpoints["available_history_"+endpoint] = _endpoint(float(canonical_mean([r[endpoint] for r in subset])), observations=len(subset)) if subset else _endpoint(reason="empty_positive_choice_history")
    def slope(subset, predictor, adjusted=None, frozen=None):
        plan, why = initial_plan(subset, predictor, adjusted)
        if plan is None: return _endpoint(reason=why, observations=len(subset), status="support_unavailable")
        if frozen is not None:
            need(frozen["predictorBinary64"] == [bits(r[predictor]) for r in subset] and frozen["planHash"] == digest({k: v for k, v in frozen.items() if k != "planHash"}), "Frozen native projection identity")
            plan = frozen
        weights = [unbits(x) for x in plan["weightBinary64"]]
        return _endpoint(float(canonical_projection(weights, [r["share"] for r in subset])), observations=len(subset), plan=plan, diagnostics=plan["diagnostics"])
    for endpoint, predictor in SLOPES.items():
        endpoints[endpoint] = slope(rows, predictor, frozen=initial_projection_plan if predictor == "x0" else None) if complete_choice else _endpoint(reason="incomplete_ten_positive_defined_choices")
        endpoints["available_history_"+endpoint] = slope(positive, predictor) if positive else _endpoint(reason="empty_positive_choice_history")
    for kind in ("order", "order_scale_gap"):
        endpoints[kind+"_adjusted_initial"] = slope(rows, "x0", kind) if complete_choice else _endpoint(reason="incomplete_ten_positive_defined_choices")
    tie_free = [r for r in positive if not r["lowerMeanTie"]]
    endpoints["tie_free_L"] = (_endpoint(float(canonical_mean([r["L_lower_mean_share"] for r in tie_free])), observations=len(tie_free)) if complete_choice and tie_free else _endpoint(reason="tied_only_person" if complete_choice else "incomplete_ten_positive_defined_choices"))
    halves = {name: {"recordCount": 5, "Q": math.fsum(r["Q"] for r in subset)/5, "paidCount": math.fsum(r["paidCount"] for r in subset)/5, "quarantined": False} for name, subset in (("first", rows[:5]), ("second", rows[5:]))}
    first = rows[0]
    person_id = first["personId"]
    prefix = [first["manifestHash"], first["populationId"], first["scenarioId"], first["conditionId"], first["variant"]]
    matches = [rep for rep in range(64) if semantic(prefix+[rep]) == person_id]
    need(len(matches) == 1, "Person ID must resolve to exactly one declared replicate")
    return make_record("person", first["manifestHash"], prefix+matches, {"personId": person_id, "conditionId": first["conditionId"], "populationId": first["populationId"], "scenarioId": first["scenarioId"], "variant": first["variant"], "replicateId": matches[0], "historyIds": [r["trialId"] for r in rows], "metricIds": [r["recordId"] for r in rows], "complete": True, "computationalReasons": [], "endpoints": endpoints, "eligibility": {k: {"status": v["status"], "reason": v["reason"]} for k, v in endpoints.items()}, "descriptiveHalves": halves})


def summarize_bridge(manifest, store, *, tick):
    """Orchestration supplies streamed groups, never an unbound directory scan."""
    for group in manifest["estimandGroups"]():
        tick(); spec = group["spec"]; weights = {a: float(w) for a, w in spec["weights"].items()}
        by_arm, counts, variant_counts, fractions, variances = {}, {}, {}, {}, {}
        incomplete, unsupported, zero, numerical = False, False, False, False
        for arm in weights:
            arm_incomplete, arm_unsupported, arm_numerical = False, False, False
            strata = []; variant_counts[arm] = {}; assigned = complete = eligible = 0
            if arm in group.get("absentArms", []):
                unsupported = True
                by_arm[arm] = {"status": "unavailable", "value": None, "mcSE": None, "reason": "scenario_or_control_not_declared_for_required_arm"}
                counts[arm] = {"assigned": 0, "complete": 0, "eligible": 0}
                variant_counts[arm] = {str(v): {"assigned": 0, "complete": 0, "eligible": 0} for v in range(9)}
                fractions[arm] = {"pooled": None, "byVariant": {str(v): None for v in range(9)}, "reason": "scenario_or_control_not_declared_for_required_arm"}
                continue
            if group["route"] == "deterministic_root_Q":
                means = []
                for variant in range(9):
                    outcomes = []; observed = 0
                    for ref in group["rootDecisionRefs"](arm, variant):
                        assigned += 1
                        if ref is None: arm_incomplete = True; continue
                        decision = store.read(ref)
                        need(decision["rootState"] and decision["scenarioId"] == group["scenarioId"], "Root-Q bound table identity")
                        if decision["numericalStatus"] != "valid": arm_numerical = True; continue
                        observed += 1; complete += 1; eligible += 1
                        outcomes.append(int(decision["action"] in ("ask_A", "ask_B")))
                        del decision
                    variant_counts[arm][str(variant)] = {"assigned": 10, "complete": observed, "eligible": observed}
                    means.append(float(canonical_mean(outcomes)) if len(outcomes) == 10 else None)
                need(assigned == 90, "All90 actual roots required for root-Q")
                mu = None if any(x is None for x in means) else math.fsum(means)/9
                counts[arm] = {"assigned": assigned, "complete": complete, "eligible": eligible}
                why = "unresolved_numerical_root" if arm_numerical else "computation_unresolved" if arm_incomplete else None
                by_arm[arm] = {"status": "unresolved" if why else "available", "value": mu, "mcSE": None, "reason": why}
                fractions[arm] = {"pooled": None if why else 1.0, "byVariant": {str(v): None if variant_counts[arm][str(v)]["complete"] != 10 else 1.0 for v in range(9)}, "reason": why}
                incomplete |= arm_incomplete; numerical |= arm_numerical
                continue
            for variant in range(9):
                tick(); values = []; observed = 0; all_values = []
                for ref in group["personRefs"](arm, variant):
                    assigned += 1
                    if ref is None: arm_incomplete = True; all_values.append(None); continue
                    person = store.read(ref)
                    if not person["complete"]: arm_incomplete = True; all_values.append(None); continue
                    complete += 1; observed += 1; endpoint = person["endpoints"][spec["endpoint"]]
                    value = endpoint["value"] if endpoint["status"] == "available" else None
                    if endpoint["status"] == "computation_unresolved": arm_incomplete = True
                    if endpoint["status"] == "support_unavailable": arm_unsupported = True
                    if value is not None: values.append(value); eligible += 1
                    all_values.append(value)
                variant_counts[arm][str(variant)] = {"assigned": 64, "complete": observed, "eligible": len(values)}
                need(len(all_values) == 64, "Full declared64 assigned slots per variant")
                strata.append((values, all_values))
            counts[arm] = {"assigned": assigned, "complete": complete, "eligible": eligible}
            flat = [x for values, _ in strata for x in values]
            mu = math.fsum(flat)/eligible if eligible else None
            zero |= not eligible
            p = eligible/assigned
            se = None
            if not arm_incomplete and eligible >= 2:
                variance = 0.
                for _, all_values in strata:
                    z = [0. if x is None else x-mu for x in all_values]; zm = math.fsum(z)/64
                    variance += (1/9)**2*(math.fsum((x-zm)**2 for x in z)/63)/64
                variance /= p*p; se = math.sqrt(variance); variances[arm] = variance
            route = group["route"]
            if route == "standardized_variant_diagnostic":
                mu = math.fsum(math.fsum(values)/len(values) for values, _ in strata)/9 if all(values for values, _ in strata) else None
                se = None
            by_arm[arm] = {"status": "available" if mu is not None and not arm_incomplete else "unresolved" if arm_incomplete else "unavailable", "value": None if arm_incomplete else mu, "mcSE": None if arm_incomplete else se, "reason": "computation_unresolved" if arm_incomplete else None if mu is not None else "unsupported_endpoint" if arm_unsupported else "no_eligible_people"}
            fractions[arm] = {"pooled": None if arm_incomplete else p, "byVariant": {str(v): None if variant_counts[arm][str(v)]["complete"] != 64 else len(strata[v][0])/64 for v in range(9)}, "reason": "computation_unresolved" if arm_incomplete else None}
            incomplete |= arm_incomplete; unsupported |= arm_unsupported
        value = None if incomplete or numerical or any(v["value"] is None for v in by_arm.values()) else math.fsum(weights[a]*by_arm[a]["value"] for a in weights)
        se = math.sqrt(math.fsum(weights[a]**2*variances[a] for a in weights)) if value is not None and group["route"] == "assigned_eligible_person" and set(variances) == set(weights) else None
        status = "UNRESOLVED_NUMERICAL" if numerical else "UNRESOLVED_COMPUTATION" if incomplete else "UNAVAILABLE_SUPPORT" if value is None and unsupported else "UNAVAILABLE_NO_ELIGIBLE" if value is None or zero else "DETERMINISTIC_NUMERICAL_POLICY_MAP" if group["route"] == "deterministic_root_Q" else "MODEL_MC_ESTIMATE"
        identity = [manifest["manifestHash"], group["populationScope"], group["scenarioId"], spec["id"], group["route"]]
        yield make_record("estimand", manifest["manifestHash"], identity, {"testId": spec["id"], "namespace": group["namespace"], "endpoint": spec["endpoint"], "tail": spec["tail"], "weights": weights, "scenarioId": group["scenarioId"], "populationScope": group["populationScope"], "route": group["route"], "status": status, "value": value, "byArm": by_arm, "countsByArm": counts, "countsByVariant": variant_counts, "eligibilityFractions": fractions, "mcSE": se, "uncertaintyReason": None if se is not None else "deterministic_root_Q" if group["route"] == "deterministic_root_Q" else "not_defined_for_this_route_or_support", "reason": None if value is not None else status.lower(), "directionRelation": None if value is None else "positive" if value > 0 else "negative" if value < 0 else "zero", "noPrimaryRescue": True})
    for group in manifest.get("performanceGroups", lambda: ())():
        tick(); strata = {key: [] for key in PERFORMANCE}; counts = {}; complete_people = complete_trials = 0
        unfinished, numerical = False, False
        for variant in range(9):
            variant_trials = variant_people = 0
            for rep in range(64):
                histories = []
                for ref in group["historyRefs"](variant, rep):
                    if ref is None: unfinished = True; continue
                    history = store.read(ref)
                    if history["status"] != "completed":
                        numerical |= history["status"] == "numerical_failure"; unfinished = True; continue
                    histories.append(history); variant_trials += 1
                if len(histories) != 10: continue
                variant_people += 1
                for key in PERFORMANCE:
                    values = [h["realized"][{"realizedUtility": "combinedUtility", "realizedCombinedGPA": "combinedGPA", "realizedObjective": "realizedObjective"}[key]] if key.startswith("realized") else h[key] for h in histories]
                    if len(strata[key]) <= variant: strata[key].append([])
                    strata[key][variant].append(float(canonical_mean(values)))
            counts[str(variant)] = {"assignedTrials": 640, "completedTrials": variant_trials, "assignedPersons": 64, "completedPersons": variant_people}
            complete_people += variant_people; complete_trials += variant_trials
            for key in PERFORMANCE:
                if len(strata[key]) <= variant: strata[key].append([])
        values = {}
        for key in PERFORMANCE:
            if unfinished:
                values[key] = {"value": None, "mcSE": None, "uncertaintyReason": "computationally_unfinished_required_history"}; continue
            need(all(len(row) == 64 for row in strata[key]), "Full fixed-stratum all-assigned performance")
            mean = math.fsum(x for row in strata[key] for x in row)/576
            variance = math.fsum((1/9)**2*(math.fsum((x-math.fsum(row)/64)**2 for x in row)/63)/64 for row in strata[key])
            values[key] = {"value": mean, "mcSE": math.sqrt(variance), "uncertaintyReason": None}
        status = "UNRESOLVED_NUMERICAL" if numerical else "UNRESOLVED_COMPUTATION" if unfinished else "MODEL_MC_ESTIMATE"
        identity = [manifest["manifestHash"], group["populationId"], group["scenarioId"], group["conditionId"], "performance"]
        yield make_record("performance", manifest["manifestHash"], identity, {"populationId": group["populationId"], "scenarioId": group["scenarioId"], "conditionId": group["conditionId"], "assignedTrials": 5760, "completedTrials": complete_trials, "assignedPersons": 576, "completedPersons": complete_people, "countsByVariant": counts, "status": status, "values": values, "reason": None if not unfinished else "computationally_unfinished_required_history"})
