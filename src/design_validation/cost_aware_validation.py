"""Independent scientific replay of portable selected-trial NDJSON.

Source schedules are independently reconstructed, native initial log bits are
regenerated, and every decision/branch/draw/paid update is checked. This reader
does not import the policy, its terminal helpers, or its bound implementation.
"""
import hashlib
import math
from pathlib import Path
import tempfile
import numpy as np

from ..tutoring_analysis import native_reconstruction as schedule_reference
from . import cost_aware_reference as R
from .cost_aware_inputs import (NAMESPACE, bits, digest, generate_source, log_bits, parse,
                                identical, read, require, root_key, selections, trial_name,
                                unbits, validate_context, validate_source_scope)

BRANCH = "predictive_joint_branch_v1"
OBSERVED = "observed_report_posterior_v1"
SAMPLER = "ordered_float64_cdf_strict_greater_final100_fallback_v1"


class Comparisons:
    def __init__(self):
        self.checks = 0
        self.maximum_error = 0.

    def exact(self, actual, expected, label):
        self.checks += 1
        require(identical(actual, expected), label)

    def near(self, actual, expected, label, tolerance=1e-10):
        if type(expected) is float:
            self.checks += 1
            require(type(actual) is float and math.isfinite(actual) and math.isfinite(expected), label)
            error = abs(actual-expected)
            self.maximum_error = max(self.maximum_error, error)
            require(error <= tolerance, label)
        elif type(expected) is dict:
            self.exact(sorted(actual) if type(actual) is dict else None, sorted(expected), label + " fields")
            for k in expected:
                self.near(actual[k], expected[k], label + "." + k, tolerance)
        elif type(expected) in (list, tuple):
            require(type(actual) is list and len(actual) == len(expected), label + " length")
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.near(a, b, label + "[" + str(i) + "]", tolerance)
        else:
            self.exact(actual, expected, label)


def numeric_payload(value):
    """JavaScript numbers have one binary64 type; do not round source values."""
    if type(value) in (int, float):
        return {"numberBinary64": bits(value)}
    if type(value) is list:
        return [numeric_payload(v) for v in value]
    if type(value) is dict:
        return {k: numeric_payload(v) for k, v in value.items()}
    return value


def validate_roots(inputs, comparisons, node):
    wanted = {root_key(row) for row in inputs["selection"]}
    with tempfile.TemporaryDirectory(prefix="finite-source-") as directory:
        source_path = Path(directory) / "source_states.json"
        generate_source(inputs["context"], source_path, node, selected=inputs["selection"])
        source = read(source_path)
    validate_source_scope(source, inputs["selection"])
    raw_roots = [s for s in source["states"] if root_key(s) in wanted
                 and s["prefixLength"] == 0 and s["prefixKind"] == "fixed"]
    by_key = {root_key(s): s for s in raw_roots}
    require(len(by_key) == len(raw_roots) and set(by_key) == wanted, "Regenerated selected roots complete")
    sessions = {(s["conditionId"], s["scheduleVariant"]): s for s in source["sessions"]}
    roots = {root_key(r): r for r in inputs["roots"]}
    require(len(roots) == len(inputs["roots"]) and set(roots) == wanted, "Exact unique input roots")
    for key, root in roots.items():
        raw = by_key[key]; d = raw["definition"]; common = raw["common"]
        exported = sessions[key[:2]]["exported"]
        options = {k: exported[k] for k in ("profileId", "conditionId", "scheduleVariant",
                   "participantId", "sessionId", "seed", "startedAt")}
        arm, variant, main = key
        comparisons.exact(options, {"profileId": schedule_reference.PROFILE, "conditionId": arm,
            "scheduleVariant": variant, "participantId": f"synthetic-lookup-{arm}-v{variant}",
            "sessionId": f"synthetic-source-{arm}-v{variant}",
            "seed": f"r9-calibration-lookup-v1:{arm}:variant:{variant}",
            "startedAt": "2026-10-11T00:00:00.000Z"}, "Frozen native source options")
        independent_definition = schedule_reference.schedule(options)[main+3]
        comparisons.exact(numeric_payload({k: d[k] for k in independent_definition}),
                          numeric_payload(independent_definition), "Independent native schedule")
        order = [1, 0] if d["sideReversed"] else [0, 1]
        config = {"condition_id": arm, "template_id": d["templateId"], "scale": float(d["scale"]),
                  "gap_offset": float(d["gap"]), "prior_centres": [d["priorCentres"][i] for i in order],
                  "prior_spreads": [d["priorSpreads"][i] for i in order], "report_width": float(d["reportWidth"]),
                  "total_millihours": d["budgetMillihours"], "cost_millihours": d["costMillihours"],
                  "repeat_count_per_30": 0, "allocation_denominator": 1000,
                  "max_assessments": d["budgetMillihours"] // d["costMillihours"],
                  "alpha": common["utility"]["gain_exponent"], "loss_multiplier": common["utility"]["loss_multiplier"],
                  "gpa_baseline": common["gpa"]["baseline"], "gpa_scale": common["gpa"]["utility_scale"]}
        expected = {"populationId": "working", "conditionId": arm, "scheduleVariant": variant, "mainIndex": main,
                    "config": config, "sourceDefinition": d, "sourceOptions": options,
                    "logicalToDisplayed": [["A", "B"][i] for i in order],
                    "initialLogBinary64": log_bits(raw["initialLogs"]),
                    "sourceRecipeFingerprint": raw["meta"]["resolvedRecipeFingerprint"]}
        comparisons.exact(root, expected, "Exact regenerated consumed source fields")
        comparisons.exact(raw["paid"], [], "Unpaid native root")
        comparisons.exact(raw["remainingMillihours"], config["total_millihours"], "Initial native budget")
        comparisons.exact(raw["logicalToDisplayed"], expected["logicalToDisplayed"], "Single side reversal")
        for slot in range(2):
            comparisons.near([unbits(x) for x in root["initialLogBinary64"][slot]],
                             R.prior(config["prior_centres"][slot], config["prior_spreads"][slot]).tolist(),
                             "Independent finite initial prior")
    return roots


class Replay(Comparisons):
    def __init__(self, root, selection, settings, reference):
        super().__init__()
        self.root, self.config, self.selection, self.settings = root, root["config"], selection, settings
        self.profile, multiplier = selection["scenarioId"].split("|rho_")
        self.rho = float(float(multiplier)*reference["rho0_GPA_per_personal_second"])
        self.likelihood = R.channel(self.config["report_width"])
        self.base_key = {"namespace": NAMESPACE, **{k: selection[k] for k in
                         ("conditionId", "scheduleVariant", "replicateId", "mainIndex")}}
        self.decisions = 0
        self.branch_rows = 0

    def summary(self, logs, budget):
        return R.terminal(self.config, logs, budget)

    def decision(self, record, logs, budget):
        initial = budget == self.config["total_millihours"] and log_bits(logs) == self.root["initialLogBinary64"]
        summary = self.summary(logs, budget)
        self.near(record["currentSummary"], summary, "Independent current terminal menu")
        times = R.durations(self.settings, self.profile, budget)
        selected = R.select(record["currentSummary"], times, self.rho)
        independent_selected = R.select(summary, times, self.rho)
        self.exact(record["currentSelection"], selected, "Exact selector on independently checked operands")
        self.near(selected, independent_selected, "Independent current selection")
        expected = {"scenarioId": self.selection["scenarioId"], "budget": budget, "rootState": initial,
                    "mapRole": "initial" if initial else OBSERVED, "orderedLogBinary64": log_bits(logs),
                    "currentSummary": record["currentSummary"], "currentSelection": selected,
                    "reportBlocks": [], "scenarioChildSelections": [], "probabilityResiduals": [],
                    "askSelected": None, "askRaw": None, "bounds": None,
                    "boundsReason": None if initial else "root_annotation_only",
                    "numericalStatus": "valid", "numericalReasons": []}
        cost = self.config["cost_millihours"]
        asks, raw = [], []
        independent_asks, independent_raw = [], []
        if budget >= cost:
            require(len(record["reportBlocks"]) == 20 and len(record["scenarioChildSelections"]) == 2,
                    "Both full report panels required")
            for slot in range(2):
                probabilities, posteriors = R.predictive(logs[slot], self.likelihood)
                require(np.isfinite(probabilities).all() and np.all(probabilities >= 0), "Finite report law")
                residual = float(probabilities.sum())-1.
                require(abs(residual) <= 1e-12, "Unrepaired predictive mass")
                expected["probabilityResiduals"].append(residual)
                children, independent_children = [], []
                for start in range(0, 100, 10):
                    block = record["reportBlocks"][slot*10+start//10]
                    require(len(block["rows"]) == 10, "Complete report block")
                    rows = []
                    for offset, row in enumerate(block["rows"]):
                        index = start+offset; post = posteriors[index]
                        actual = R.actual_posterior(logs[slot], self.likelihood, index+1)
                        parity = float(np.max(abs(post-actual)))
                        require(np.isfinite(post).all() and parity <= 1e-10, "Predictive/observed parity")
                        child_logs = [post, logs[1]] if slot == 0 else [logs[0], post]
                        child = self.summary(child_logs, budget-cost)
                        self.near(row["summary"], child, "Independent full child menu")
                        rows.append({"report": index+1, "probability": float(probabilities[index]),
                            "probabilityBinary64": bits(float(probabilities[index])),
                            "reportProbabilityUnderflow": bool(probabilities[index] == 0),
                            "branchLogBinary64": [bits(float(v)) for v in post],
                            "actualUpdateLogBinary64": [bits(float(v)) for v in actual],
                            "posteriorParityMaximum": parity, "summary": row["summary"],
                            "latentUnderflowCount": int(np.count_nonzero(np.exp(R.normalized(post)) == 0))})
                        child_times = R.durations(self.settings, self.profile, budget-cost)
                        choice = R.select(row["summary"], child_times, self.rho)
                        independent_choice = R.select(child, child_times, self.rho)
                        self.near(choice, independent_choice, "Independent selected child")
                        children.append(choice)
                        independent_children.append(independent_choice)
                        self.branch_rows += 1
                    expected["reportBlocks"].append({"mapId": BRANCH, "displayedSlot": slot,
                        "childBudget": budget-cost, "firstReport": start+1, "lastReport": start+10,
                        "reportProbabilityUnderflowCount": sum(r["reportProbabilityUnderflow"] for r in rows), "rows": rows})
                expected["scenarioChildSelections"].append(children)
                asks.append(float(probabilities @ np.array([x["selected_objective"] for x in children], dtype=np.float64)))
                raw.append(float(probabilities @ np.array([x["raw_max_objective"] for x in children], dtype=np.float64)))
                independent_asks.append(float(probabilities @ np.asarray([x["selected_objective"] for x in independent_children], dtype=np.float64)))
                independent_raw.append(float(probabilities @ np.asarray([x["raw_max_objective"] for x in independent_children], dtype=np.float64)))
            expected.update(askSelected=asks, askRaw=raw)
            self.near(record["askSelected"], independent_asks, "Independent full100 selected backup")
            self.near(record["askRaw"], independent_raw, "Independent full100 raw backup")
        expected.update(R.policy(selected["selected_objective"], asks, budget, cost))
        independent_policy = R.policy(independent_selected["selected_objective"], independent_asks, budget, cost)
        expected["decisionReason"] = ("automatic_zero" if budget == 0 else "structural_unaffordable" if budget < cost
                                      else "voluntary_myopic" if expected["action"] == "stop" else "positive_selected_one_query_gain")
        if initial:
            upper = None if budget < cost else R.continuation_upper(self.config, logs, budget-cost, self.settings, self.profile, self.rho)
            burn = None if budget % cost else self.summary(logs, 0)["raw_grid_max_combined_GPA"]
            grid = record["currentSummary"]["raw_grid_max_combined_GPA"]
            contradiction = upper is not None and any(v > upper+1e-8 for v in asks+raw+([] if burn is None else [burn]))
            bound = {"selectedStopObjective": independent_selected["selected_objective"], "rawMenuStopMaximum": independent_selected["raw_max_objective"],
                     "rawGridCombinedGPAMaximum": grid, "continuationUpper": upper,
                     "continuationReason": "structural_unaffordable" if upper is None else None,
                     "macroClassRootUpper": independent_selected["raw_max_objective"] if upper is None else max(independent_selected["raw_max_objective"], upper),
                     "unrestrictedGridRootUpper": grid if upper is None else max(grid, upper),
                     "oneQueryFloatingWitnessBySlot": expected["askSelected"], "burnZeroWitness": burn,
                     "burnReason": "positive_residual_prevents_exact_zero" if burn is None else None,
                     "macroClassScope": "ten_macro_stopping_only", "unrestrictedScope": "native_grid_nonnegative_stop_penalty_dropped",
                     "classificationMargin": 1e-8, "classifications": {
                         "selectedStopAboveContinuation": None if upper is None else independent_selected["selected_objective"] > upper+1e-8,
                         "oneQueryAboveStopBySlot": None if budget < cost else [v > independent_selected["selected_objective"]+1e-8 for v in asks],
                         "burnAboveStop": None if burn is None else burn > independent_selected["selected_objective"]+1e-8,
                         "witnessUpperContradiction": contradiction}}
            self.near(record["bounds"], bound, "Independent max-before-expectation bounds")
            expected["bounds"] = record["bounds"]
            if contradiction:
                expected.update(action=None, numericalStatus="unresolved", numericalReasons=["floating_witness_upper_contradiction"],
                                decisionReason="unresolved", chosenGain=None, actionTie=None, nearThreshold=None)
        self.exact(record, expected, "Exact complete decision, maps, full100 dots and action")
        if record["numericalStatus"] == "valid":
            self.near({k: record[k] for k in independent_policy}, independent_policy, "Independent all-five myopic rule")
        else:
            self.near(record["bestGain"], independent_policy["bestGain"], "Unresolved independent diagnostic gain")
        self.decisions += 1

    def draw(self, record, logical, kind, index, logs):
        key = {**self.base_key, "logicalRecipient": logical, "streamKind": kind, "reportIndex": index}
        expected = R.stream_uniform(key)
        probabilities = np.exp(logs)
        expected.update(R.categorical(probabilities, expected["uniform"]))
        expected.update(samplerMapId=SAMPLER, logProbabilityBinary64=[bits(float(v)) for v in logs],
                        probabilitiesBinary64=[bits(float(v)) for v in probabilities])
        self.exact(record, expected, "Exact keyed draw, full probability bits and ordered CDF")
        return expected["category"]

    def trial(self, rows, inputs_digest):
        self.exact(next(rows), {"kind": "trial", "selection": self.selection, "inputsDigest": inputs_digest}, "Trial header")
        logs = [np.array([unbits(v) for v in row], dtype=np.float64) for row in self.root["initialLogBinary64"]]
        initial = [x.copy() for x in logs]
        budget, counts, query, observed = self.config["total_millihours"], [0, 0], 0, [[], []]
        hidden = None
        while True:
            event = next(rows)
            self.exact(sorted(event), ["kind", "queryIndex", "value"], "Decision event fields")
            self.exact([event["kind"], event["queryIndex"]], ["decision", query], "Decision ordinal")
            decision = event["value"]
            self.decision(decision, logs, budget)
            if hidden is None:
                latent = next(rows)
                require(latent["kind"] == "latent" and len(latent["draws"]) == 2, "Latent-once pair required")
                needs = [self.draw(latent["draws"][i], logical, "latent", 0,
                                   R.normalized(logs[self.root["logicalToDisplayed"].index(logical)]))
                         for i, logical in enumerate(("A", "B"))]
                hidden = [needs[("A", "B").index(x)] for x in self.root["logicalToDisplayed"]]
                self.exact(latent, {"kind": "latent", "draws": latent["draws"], "logicalNeeds": needs,
                                    "displayedNeeds": hidden}, "Single latent pair and display orientation")
            if decision["action"] in ("stop", None):
                break
            who = 0 if decision["action"] == "ask_A" else 1
            event = next(rows); logical = self.root["logicalToDisplayed"][who]
            report = self.draw(event["draw"], logical, "report", counts[who], self.likelihood[hidden[who]-1])
            before, before_counts = log_bits(logs), list(counts)
            logs[who] = R.actual_posterior(logs[who], self.likelihood, report)
            observed[who].append(report); counts[who] += 1
            next_budget = budget-self.config["cost_millihours"]
            self.exact(event, {"kind": "paid", "queryIndex": query, "displayedSlot": who, "logicalRecipient": logical,
                "recipientReportIndex": before_counts[who], "report": report, "draw": event["draw"],
                "budgetBefore": budget, "budgetAfter": next_budget, "countsBefore": before_counts,
                "countsAfter": list(counts), "logsBefore": before, "logsAfter": log_bits(logs),
                "actualUpdateMapId": OBSERVED}, "Exact observed update and resource conservation")
            budget = next_budget; query += 1
        for slot in range(2):
            full = initial[slot].copy()
            for report in observed[slot]:
                full += self.likelihood[:, report-1]
            self.near(R.normalized(logs[slot]).tolist(), R.normalized(full).tolist(), "Full-prefix Bayes reconstruction")
        history = {"status": "numerical_failure" if decision["action"] is None else "completed",
                   "reason": "floating_witness_upper_contradiction" if decision["action"] is None else None,
                   "paidCount": query, "finalBudget": budget, "finalLogBinary64": log_bits(logs),
                   **dict.fromkeys(("stopReason", "finalSummary", "shareIndex", "share", "terminalSeconds",
                                   "expectedUtility", "expectedCombinedGPA", "expectedObjective", "realized"))}
        if decision["action"] == "stop":
            chosen = decision["currentSelection"]; summary = decision["currentSummary"]
            row = summary["macros"][chosen["selected_macro"]]
            index = chosen["selected_index"]; seconds = chosen["duration_by_macro_seconds"][chosen["selected_macro"]]
            history.update(stopReason=decision["decisionReason"], finalSummary=summary, shareIndex=index,
                           share=None if index is None else index/1000, terminalSeconds=seconds,
                           expectedUtility=row["expected_utility"], expectedCombinedGPA=row["expected_combined_GPA"],
                           expectedObjective=chosen["selected_objective"],
                           realized=R.realized(self.config, budget, index, hidden, self.rho, seconds))
        self.exact(next(rows), {"kind": "history", "value": history}, "Complete final history and scalar realized payoff")
        require(next(rows, None) is None, "Extra records after final history")
        return history


def record_rows(path, expected_hashes):
    require(type(expected_hashes) is list and bool(expected_hashes), "Nonempty raw-row index")
    count = 0
    with Path(path).open("rb") as stream:
        for line in stream:
            require(line.endswith(b"\n") and line != b"\n", "Complete nonempty NDJSON row")
            raw = line[:-1]
            require(count < len(expected_hashes) and hashlib.sha256(raw).hexdigest() == expected_hashes[count],
                    "Raw record-row SHA256 excluding newline")
            count += 1
            yield parse(raw.decode("utf-8"))
    require(count == len(expected_hashes), "Missing NDJSON rows")


def verified_reference(design, reported, comparisons):
    independent = R.reference_record(design)
    comparisons.near(reported, independent, "Independent old symmetric reference")
    return independent


def validate_computation(inputs, records_directory, *, expected_selection, node="node"):
    validate_context(inputs["context"])
    comparisons = Comparisons()
    comparisons.exact(inputs["selection"], selections(expected_selection), "External selected coordinate roster")
    roots = validate_roots(inputs, comparisons, node)
    directory = Path(records_directory)
    manifest = read(directory / "reference.json")
    require(set(manifest) == {"schema", "scope", "inputsDigest", "selection", "roots", "reference", "trials",
                              "historicalAcceptanceClaim"}, "Public reference fields")
    comparisons.exact([manifest["schema"], manifest["scope"], manifest["historicalAcceptanceClaim"]],
                      ["finite-cost-aware-records-v1", "selected_working_trials", False], "No historical/global acceptance claim")
    identity = digest(inputs)
    comparisons.exact(manifest["inputsDigest"], identity, "Public inputs binding")
    comparisons.exact(manifest["selection"], inputs["selection"], "Reference selection")
    comparisons.exact(manifest["roots"], inputs["roots"], "Reference roots")
    independent_reference = verified_reference(inputs["context"]["design"], manifest["reference"], comparisons)
    trials = manifest["trials"]
    comparisons.exact([t["selection"] for t in trials], inputs["selection"], "No duplicate, missing or extra trial files")
    filenames = [trial_name(s)+".ndjson" for s in inputs["selection"]]
    comparisons.exact(sorted(p.name for p in directory.iterdir()), sorted(filenames+["reference.json"]), "Exact output file roster")
    report = {"schema": "finite-cost-aware-verification-v1", "scope": "selected_working_trials",
              "selection": inputs["selection"], "inputsDigest": identity, "trials": [],
              "historicalAcceptanceClaim": False, "sharedBackend": "NumPy_float64_and_PCG64"}
    for trial, filename in zip(trials, filenames):
        comparisons.exact(sorted(trial), ["file", "history", "rowSha256", "selection"], "Trial reference fields")
        comparisons.exact(trial["file"], filename, "Deterministic relative trial filename")
        replay = Replay(roots[root_key(trial["selection"])], trial["selection"], inputs["context"]["settings"], independent_reference)
        rows = record_rows(directory / filename, trial["rowSha256"])
        try:
            history = replay.trial(rows, identity)
        except StopIteration as error:
            raise ValueError("Truncated scientific history") from error
        finally:
            rows.close()
        comparisons.exact(history, trial["history"], "Reference/history terminal agreement")
        report["trials"].append({"selection": trial["selection"], "status": history["status"],
                                 "paidCount": history["paidCount"], "decisionsChecked": replay.decisions,
                                 "branchRowsChecked": replay.branch_rows, "checks": replay.checks,
                                 "maximumAbsoluteError": replay.maximum_error})
    report.update(verification="checked", inputChecks=comparisons.checks,
                  maximumInputAbsoluteError=comparisons.maximum_error,
                  allSelectedNumericallyCompleted=all(t["status"] == "completed" for t in report["trials"]))
    return report
