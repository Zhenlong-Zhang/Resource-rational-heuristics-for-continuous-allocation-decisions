"""Repeated selected-macro decisions and raw single-trial scientific witnesses.

Questioning consumes tutoring resource, not personal time. Terminal time prices
the objective only. No person-level inference or aggregate reduction is provided.
"""
from collections import OrderedDict
import hashlib
import numpy as np

from .bounded_task import TaskConfig, normalize_log_weights
from .budget_horizon import BudgetTask, native_realized
from .terminal_time import macro_durations, select_macro, terminal_summary, validate_settings
from .cost_aware_bounds import native_latent_matrices, expected_pathwise_max
from .cost_aware_inputs import (NAMESPACE, KEY_FIELDS, bits, unbits, log_bits, compact,
                                require, integer, selections, root_key, identical)
from dataclasses import asdict

BRANCH = "predictive_joint_branch_v1"
OBSERVED = "observed_report_posterior_v1"
SAMPLER = "ordered_float64_cdf_strict_greater_final100_fallback_v1"


class SummaryCache:
    """A trial-local cache; stored operands are not rounded or canonicalized."""
    def __init__(self, config):
        self.config, self.rows = config, OrderedDict()

    def summary(self, logs, budget):
        key = (budget, tuple(tuple(row) for row in log_bits(logs)))
        if key not in self.rows:
            self.rows[key] = terminal_summary(self.config, *logs, budget)
            if len(self.rows) > 64:
                self.rows.popitem(last=False)
        self.rows.move_to_end(key)
        return self.rows[key]


def scenario(settings, selection, reference):
    selections([selection])
    name, rate = selection["scenarioId"].split("|rho_")
    return {"scenarioId": selection["scenarioId"], "profile": settings["assumed_profiles"][name],
            "rhoMultiplier": float(rate), "rho": float(float(rate)*reference["rho0_GPA_per_personal_second"])}


def decision_table(root, task, logs, budget, scenario, settings, *, cache=None):
    """Evaluate both complete report panels with the original full-vector dots."""
    integer(budget, 0, task.config.total_millihours)
    require(identical(asdict(task.config), root["config"]), "Exact root/task configuration")
    logs = [np.array(x, dtype=np.float64, copy=True) for x in logs]
    cache = cache or SummaryCache(task.config)
    encoded = log_bits(logs)
    initial = encoded == root["initialLogBinary64"] and budget == task.config.total_millihours
    summary = cache.summary(logs, budget)
    rho, profile = scenario["rho"], scenario["profile"]
    durations = macro_durations(profile, settings["common_execution_seconds"], budget)
    selected = select_macro(summary, durations, rho)
    result = {"scenarioId": scenario["scenarioId"], "budget": budget, "rootState": initial,
              "mapRole": "initial" if initial else OBSERVED, "orderedLogBinary64": encoded,
              "currentSummary": summary, "currentSelection": selected, "reportBlocks": [],
              "scenarioChildSelections": [], "probabilityResiduals": [], "askSelected": None,
              "askRaw": None, "bestGain": None, "chosenGain": None, "action": "stop",
              "decisionReason": "automatic_zero" if budget == 0 else "structural_unaffordable",
              "actionTie": None, "nearThreshold": None, "bounds": None,
              "boundsReason": None if initial else "root_annotation_only",
              "numericalStatus": "valid", "numericalReasons": []}
    cost = task.config.cost_millihours
    if budget >= cost:
        asks, raw, selections_by_slot = [], [], []
        child_times = macro_durations(profile, settings["common_execution_seconds"], budget-cost)
        for slot in range(2):
            probabilities, posteriors = task.predictive(logs[slot])
            require(probabilities.shape == (100,) and np.isfinite(probabilities).all()
                    and np.all(probabilities >= 0), "Complete nonnegative report vector")
            residual = float(probabilities.sum())-1.
            require(abs(residual) <= 1e-12, "Report probability mass residual")
            children = []
            for start in range(0, 100, 10):
                rows = []
                for index in range(start, start+10):
                    post = posteriors[index]
                    actual = task.posterior(logs[slot], index+1)
                    parity = float(np.max(abs(post-actual)))
                    require(np.isfinite(post).all() and parity <= 1e-10, "Branch/observed posterior parity")
                    child_logs = [post, logs[1]] if slot == 0 else [logs[0], post]
                    child = cache.summary(child_logs, budget-cost)
                    rows.append({"report": index+1, "probability": float(probabilities[index]),
                                 "probabilityBinary64": bits(float(probabilities[index])),
                                 "reportProbabilityUnderflow": bool(probabilities[index] == 0),
                                 "branchLogBinary64": [bits(float(x)) for x in post],
                                 "actualUpdateLogBinary64": [bits(float(x)) for x in actual],
                                 "posteriorParityMaximum": parity, "summary": child,
                                 "latentUnderflowCount": int(np.count_nonzero(np.exp(normalize_log_weights(post)) == 0))})
                    children.append(select_macro(child, child_times, rho))
                result["reportBlocks"].append({"mapId": BRANCH, "displayedSlot": slot,
                    "childBudget": budget-cost, "firstReport": start+1, "lastReport": start+10,
                    "reportProbabilityUnderflowCount": sum(r["reportProbabilityUnderflow"] for r in rows), "rows": rows})
            asks.append(float(probabilities @ np.array([x["selected_objective"] for x in children], dtype=np.float64)))
            raw.append(float(probabilities @ np.array([x["raw_max_objective"] for x in children], dtype=np.float64)))
            selections_by_slot.append(children)
            result["probabilityResiduals"].append(residual)
        best = max(asks)-selected["selected_objective"]
        who = 0 if asks[0] >= max(asks)-1e-10 else 1
        result.update(askSelected=asks, askRaw=raw, scenarioChildSelections=selections_by_slot,
                      bestGain=best, chosenGain=asks[who]-selected["selected_objective"] if best > 1e-10 else 0.,
                      action=("ask_A" if who == 0 else "ask_B") if best > 1e-10 else "stop",
                      decisionReason="positive_selected_one_query_gain" if best > 1e-10 else "voluntary_myopic",
                      actionTie=abs(asks[0]-asks[1]) <= 1e-10, nearThreshold=abs(best-1e-10) <= 1e-8)
    if initial:
        upper, why = None, "structural_unaffordable"
        if budget >= cost:
            zero, pi = native_latent_matrices(root["config"], budget-cost)
            weights = np.outer(np.exp(normalize_log_weights(logs[0])), np.exp(normalize_log_weights(logs[1])))
            tau = min(macro_durations(profile, settings["common_execution_seconds"], 1).values())
            upper = expected_pathwise_max(weights, zero, pi, rho, tau)["stronger_GPA_upper"]
            why = None
        burn = cache.summary(logs, 0)["raw_grid_max_combined_GPA"] if budget % cost == 0 else None
        contradiction = upper is not None and max(result["askSelected"]+result["askRaw"]) > upper+1e-8
        result["bounds"] = {"selectedStopObjective": selected["selected_objective"],
            "rawMenuStopMaximum": selected["raw_max_objective"],
            "rawGridCombinedGPAMaximum": summary["raw_grid_max_combined_GPA"],
            "continuationUpper": upper, "continuationReason": why,
            "macroClassRootUpper": max(selected["raw_max_objective"], upper) if upper is not None else selected["raw_max_objective"],
            "unrestrictedGridRootUpper": max(summary["raw_grid_max_combined_GPA"], upper) if upper is not None else summary["raw_grid_max_combined_GPA"],
            "oneQueryFloatingWitnessBySlot": result["askSelected"], "burnZeroWitness": burn,
            "burnReason": None if burn is not None else "positive_residual_prevents_exact_zero",
            "macroClassScope": "ten_macro_stopping_only",
            "unrestrictedScope": "native_grid_nonnegative_stop_penalty_dropped", "classificationMargin": 1e-8,
            "classifications": {"selectedStopAboveContinuation": None if upper is None else selected["selected_objective"] > upper+1e-8,
                "oneQueryAboveStopBySlot": None if result["askSelected"] is None else [v > selected["selected_objective"]+1e-8 for v in result["askSelected"]],
                "burnAboveStop": None if burn is None else burn > selected["selected_objective"]+1e-8,
                "witnessUpperContradiction": contradiction}}
        if contradiction:
            result.update(action=None, numericalStatus="unresolved", numericalReasons=["floating_witness_upper_contradiction"],
                          decisionReason="unresolved", chosenGain=None, actionTie=None, nearThreshold=None)
    return result


def stream_uniform(key):
    require(type(key) is dict and set(key) == KEY_FIELDS, "Exact potential-stream key")
    require(key["namespace"] == NAMESPACE and key["conditionId"] in ("C1", "C2", "C3", "C4", "C5")
            and key["logicalRecipient"] in ("A", "B") and key["streamKind"] in ("latent", "report"), "Stream identity")
    integer(key["scheduleVariant"], 0, 8); integer(key["replicateId"], 0, 63)
    integer(key["mainIndex"], 1, 10); integer(key["reportIndex"], 0, 2**53-1)
    require(key["streamKind"] != "latent" or key["reportIndex"] == 0, "Latent index zero")
    hashed = hashlib.sha256(compact(key).encode("utf-8")).hexdigest()
    seed = hashed[:32]
    uniform = float(np.random.Generator(np.random.PCG64(int(seed, 16))).random())
    return {"keyPayload": dict(key), "digest": hashed, "seedHex": seed,
            "uniform": uniform, "uniformBinary64": bits(uniform)}


def categorical(logs, probabilities, uniform):
    require(len(logs) == len(probabilities) == 100 and all(math_finite(x) for x in logs), "Full finite log support")
    require(all(math_finite(x) and x >= 0 for x in probabilities) and 0 <= uniform < 1, "Valid CDF inputs")
    raw = float(np.sum(probabilities))
    require(abs(raw-1.) <= 1e-12, "No mass repair permitted")
    cumulative, previous, last_before = np.float64(0.), np.float64(0.), np.float64(0.)
    category, selected, fallback = None, None, False
    for i, probability in enumerate(probabilities):
        before = cumulative; last_before = before
        cumulative = np.add(cumulative, np.float64(probability))
        if category is None and cumulative > uniform:
            category, previous, selected = i+1, before, cumulative
    if category is None:
        category, previous, selected, fallback = 100, last_before, cumulative, True
    return {"rawMass": raw, "massResidual": raw-1., "category": category,
            "cdfPrevious": float(previous), "cdfSelected": float(selected), "fallback": fallback}


def math_finite(value):
    return bool(np.isfinite(float(value)))


def draw(key, logs):
    stream = stream_uniform(key)
    probabilities = np.exp(logs)
    return {**stream, **categorical(logs, probabilities, stream["uniform"]),
            "samplerMapId": SAMPLER, "logProbabilityBinary64": [bits(float(x)) for x in logs],
            "probabilitiesBinary64": [bits(float(x)) for x in probabilities],
            "zeroExponentiatedCount": int(np.count_nonzero(probabilities == 0))}


def realized(config, budget, index, needs, rho, seconds):
    D = config["allocation_denominator"]
    numerators = [0, 0] if budget == 0 else [budget*index, budget*(D-index)]
    surplus = [(n-need*1000*D)/(1000*D) for n, need in zip(numerators, needs)]
    us = [x**config["alpha"] if x >= 0 else -config["loss_multiplier"]*(-x)**config["alpha"] for x in surplus]
    total = us[0]+us[1]; gpa = 7.+.03*total
    return {"amountNumerators": numerators, "surplus": surplus, "utilities": us,
            "individualGPA": [3.5+.03*x for x in us], "combinedUtility": total, "combinedGPA": gpa,
            "realizedObjective": gpa-rho*seconds,
            "arithmeticMapId": "integer_numerator_then_native_combined_7_plus_0.03_sum_v1"}


def run_model_trial(root, selection, settings, reference, emit):
    """Stream decisions, latent-once draws, paid events and one terminal record."""
    validate_settings(settings)
    require(root_key(root) == root_key(selection) and root["populationId"] == "working", "Selected working root identity")
    selected_scenario = scenario(settings, selection, reference)
    task = BudgetTask(TaskConfig(**root["config"]))
    logs = [np.array([unbits(x) for x in row], dtype=np.float64) for row in root["initialLogBinary64"]]
    cache = SummaryCache(task.config)
    budget, counts, query = task.config.total_millihours, [0, 0], 0
    base_key = {"namespace": NAMESPACE, "conditionId": selection["conditionId"],
                "scheduleVariant": selection["scheduleVariant"], "replicateId": selection["replicateId"],
                "mainIndex": selection["mainIndex"]}
    hidden = None
    while True:
        decision = decision_table(root, task, logs, budget, selected_scenario, settings, cache=cache)
        emit({"kind": "decision", "queryIndex": query, "value": decision})
        if hidden is None:
            latent = []
            for logical in ("A", "B"):
                slot = root["logicalToDisplayed"].index(logical)
                latent.append(draw({**base_key, "logicalRecipient": logical, "streamKind": "latent", "reportIndex": 0},
                                   normalize_log_weights(logs[slot])))
            logical_needs = [x["category"] for x in latent]
            hidden = [logical_needs[("A", "B").index(x)] for x in root["logicalToDisplayed"]]
            emit({"kind": "latent", "draws": latent, "logicalNeeds": logical_needs, "displayedNeeds": hidden})
        if decision["action"] is None:
            result = {"status": "numerical_failure", "reason": "floating_witness_upper_contradiction",
                      "paidCount": query, "finalBudget": budget, "finalLogBinary64": log_bits(logs),
                      **dict.fromkeys(("stopReason", "finalSummary", "shareIndex", "share", "terminalSeconds",
                                      "expectedUtility", "expectedCombinedGPA", "expectedObjective", "realized"))}
            emit({"kind": "history", "value": result})
            return result
        if decision["action"] == "stop":
            break
        who = 0 if decision["action"] == "ask_A" else 1
        key = {**base_key, "logicalRecipient": root["logicalToDisplayed"][who], "streamKind": "report", "reportIndex": counts[who]}
        report = draw(key, task.log_likelihood[hidden[who]-1])
        before, before_counts = log_bits(logs), list(counts)
        logs[who] = task.posterior(logs[who], report["category"])
        counts[who] += 1
        next_budget = budget-task.config.cost_millihours
        emit({"kind": "paid", "queryIndex": query, "displayedSlot": who,
              "logicalRecipient": root["logicalToDisplayed"][who], "recipientReportIndex": before_counts[who],
              "report": report["category"], "draw": report, "budgetBefore": budget, "budgetAfter": next_budget,
              "countsBefore": before_counts, "countsAfter": list(counts), "logsBefore": before,
              "logsAfter": log_bits(logs), "actualUpdateMapId": OBSERVED})
        budget = next_budget; query += 1
    selection_value, summary = decision["currentSelection"], decision["currentSummary"]
    row = summary["macros"][selection_value["selected_macro"]]
    index = selection_value["selected_index"]
    seconds = selection_value["duration_by_macro_seconds"][selection_value["selected_macro"]]
    actual = realized(root["config"], budget, index, hidden, selected_scenario["rho"], seconds)
    require(abs(actual["combinedUtility"]-native_realized(task.config, None if index is None else index/1000,
                                                         budget, hidden)) <= 1e-10, "Native realized parity")
    result = {"status": "completed", "reason": None, "paidCount": query, "finalBudget": budget,
              "finalLogBinary64": log_bits(logs), "stopReason": decision["decisionReason"],
              "finalSummary": summary, "shareIndex": index, "share": None if index is None else index/1000,
              "terminalSeconds": seconds, "expectedUtility": row["expected_utility"],
              "expectedCombinedGPA": row["expected_combined_GPA"],
              "expectedObjective": selection_value["selected_objective"], "realized": actual}
    emit({"kind": "history", "value": result})
    return result
