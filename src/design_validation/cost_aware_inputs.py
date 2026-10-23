"""Portable scientific inputs and transport for selected working-population trials.

Selected roots use genuine existing native session transitions and restore.
Public identifiers are neither historical manifests nor random-number seeds.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

from .bounded_task import TaskConfig

NAMESPACE = "r9-repeated-macro-bridge-v1"
POLICY = "RR_myopic_selected_macro_full_budget_v1"
SELECTION_FIELDS = ("conditionId", "scheduleVariant", "mainIndex", "replicateId", "scenarioId")
KEY_FIELDS = {"namespace", "conditionId", "scheduleVariant", "replicateId", "mainIndex",
              "logicalRecipient", "streamKind", "reportIndex"}
FROZEN_RECIPE = {
    "schema": "finite-cost-aware-recipe-v1", "population": "working", "policy": POLICY,
    "stream_namespace": NAMESPACE, "question_limit": "budget_only",
    "personal_question_time_priced": False, "automatic_zero_budget_duration_seconds": 0.,
    "design": "native_design.json", "source_analysis": "expected_gpa_analysis.json",
    "source_construction": "native_calibration.json", "terminal_time": "tutoring_terminal_time_v01.json",
    "profiles": ["low", "base", "high"], "rho_multipliers": [0., .01, .1, 1., 10., 100.],
    "replicate_ids": [0, 63], "decision_tolerance": 1e-10, "terminal_relative_tolerance": 1e-12,
    "probability_tolerance": 1e-12, "posterior_parity_tolerance": 1e-10, "classification_margin": 1e-8,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, low, high):
    require(type(value) is int and low <= value <= high, "Integer outside scientific support")
    return value


def bits(value):
    require(type(value) in (int, float) and math.isfinite(value), "Finite binary64 number required")
    return struct.pack(">d", float(value)).hex()


def unbits(value):
    require(type(value) is str and len(value) == 16 and all(c in "0123456789abcdef" for c in value),
            "Canonical binary64 required")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    require(math.isfinite(result), "Nonfinite binary64")
    return result


def log_bits(logs):
    require(len(logs) == 2 and all(len(row) == 100 for row in logs), "Two full finite supports required")
    return [[bits(float(x)) for x in row] for row in logs]


def payload(value):
    if type(value) is float:
        return {"binary64": bits(value)}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) in (list, tuple):
        return [payload(x) for x in value]
    require(type(value) is dict and all(type(k) is str for k in value), "JSON value required")
    return {k: payload(v) for k, v in value.items()}


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(compact(payload(value)).encode("utf-8")).hexdigest()


def identical(actual, expected):
    return compact(payload(actual)) == compact(payload(expected))


def parse(text):
    def unique(items):
        result = {}
        for k, v in items:
            require(k not in result, "Duplicate JSON key")
            result[k] = v
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON: " + value)
    value = json.loads(text, object_pairs_hook=unique, parse_constant=constant)
    payload(value)
    return value


def read(path):
    return parse(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(compact(value) + "\n")


def load_context(config):
    config = Path(config)
    recipe = read(config)
    require(identical(recipe, FROZEN_RECIPE), "Unsupported scientific recipe")
    context = {"recipe": recipe, "design": read(config.parent / recipe["design"]),
               "analysis": read(config.parent / recipe["source_analysis"]),
               "construction": read(config.parent / recipe["source_construction"]),
               "settings": read(config.parent / recipe["terminal_time"])}
    validate_context(context)
    return context


def validate_context(context):
    require(set(context) == {"recipe", "design", "analysis", "construction", "settings"}, "Scientific context fields")
    require(identical(context["recipe"], FROZEN_RECIPE), "Unsupported scientific recipe")
    settings = context["settings"]
    frozen = {"common_execution_seconds": 2.65, "personal_question_time_priced": False,
              "automatic_zero_budget_duration_seconds": 0., "working_C1_total_gap_minutes": 1.,
              "C1_gap_scaled": False, "rho_multipliers": [0., .01, .1, 1., 10., 100.],
              "decision_tolerance": 1e-10, "probability_tolerance": 1e-12,
              "posterior_parity_tolerance": 1e-10, "classification_margin": 1e-8}
    require(identical({k: settings[k] for k in frozen}, frozen), "Frozen scientific settings changed")
    names = ("read_scalar", "retrieve_rule", "add_subtract", "multiply_divide", "compare", "clip",
             "EO_inference_seconds", "OPT_inference_seconds")
    profiles = {name: dict(zip(names, row)) for name, row in zip(("low", "base", "high"),
                ((.25, .1, .25, .5, .1, .1, 5., 20.), (.5, .25, .5, 1., .25, .25, 20., 60.),
                 (1., .5, 1., 2., .5, .5, 60., 180.)))}
    require(identical(settings["assumed_profiles"], profiles), "Frozen assumed durations changed")
    common = context["design"]["common"]
    require(common["utility"]["gain_exponent"] == .5 and common["utility"]["loss_multiplier"] == 2
            and common["gpa"]["baseline"] == 3.5 and common["gpa"]["utility_scale"] == .03,
            "Frozen payoff settings changed")
    return context


def selections(rows):
    require(type(rows) is list and bool(rows), "Explicit nonempty trial selection required")
    scenarios = {p + "|rho_" + format(r, "g") for p in FROZEN_RECIPE["profiles"]
                 for r in FROZEN_RECIPE["rho_multipliers"]}
    seen = set()
    for row in rows:
        require(type(row) is dict and set(row) == set(SELECTION_FIELDS), "Exact trial coordinates required")
        require(row["conditionId"] in ("C1", "C2", "C3", "C4", "C5"), "Unknown condition")
        integer(row["scheduleVariant"], 0, 8)
        integer(row["mainIndex"], 1, 10)
        integer(row["replicateId"], 0, 63)
        require(row["scenarioId"] in scenarios, "Unsupported core timing scenario")
        key = tuple(row[k] for k in SELECTION_FIELDS)
        require(key not in seen, "Duplicate selected trial")
        seen.add(key)
    return sorted(rows, key=lambda row: tuple(row[k] for k in SELECTION_FIELDS))


def trial_name(row):
    selections([row])
    return "{conditionId}_v{scheduleVariant}_m{mainIndex}_r{replicateId}_{scenarioId}".format(**row).replace("|", "_")


def root_key(row):
    return row["conditionId"], row["scheduleVariant"], row["mainIndex"]


def source_selection(selected):
    return [dict(zip(("conditionId", "scheduleVariant", "mainIndex"), key))
            for key in sorted({root_key(row) for row in selections(selected)})]


def validate_source_scope(source, selected):
    roots = source_selection(selected)
    require(source["schema"] == "finite-cost-aware-source-roots-v1"
            and source["scope"] == "selected_working_roots" and source["fullPanel"] is False
            and source["syntheticOnly"] is True and source["humanObservations"] is False
            and source["status"] == "constructed_selected_roots", "Honest partial source scope required")
    require(identical(source["rootSelection"], roots), "Exact selected source scope")
    wanted = {root_key(r) for r in roots}
    require(len(source["states"]) == len(wanted)
            and {root_key(r) for r in source["states"]} == wanted, "Exact unique source root roster")
    require(all(r["prefixLength"] == 0 and r["prefixKind"] == "fixed" and r["paid"] == []
                for r in source["states"]), "Only unpaid selected source roots")
    sessions = {(s["conditionId"], s["scheduleVariant"]) for s in source["sessions"]}
    require(len(sessions) == len(source["sessions"]) and sessions == {k[:2] for k in wanted}, "Exact selected session roster")
    for session in source["sessions"]:
        key = session["conditionId"], session["scheduleVariant"]
        last = max(k[2] for k in wanted if k[:2] == key)
        require(session["actualPhase"] == session["exported"]["phase"] == "trial"
                and session["exported"]["status"] == "incomplete"
                and session["completedTrialCount"] == last+3
                and session["exported"]["currentTrialIndex"] == last+3
                and session["completionClaim"] == "selected_root_prefix_only;no_complete_session_claim",
                "Selected prefix is not a completed session")


def source_root(source, session):
    d, common = source["definition"], source["common"]
    require(source["prefixLength"] == 0 and source["prefixKind"] == "fixed" and source["paid"] == [],
            "Unpaid source root required")
    require(d["stage"] == "main" and d["stageIndex"] == source["mainIndex"], "Source main identity")
    order = [1, 0] if d["sideReversed"] else [0, 1]
    mapping = [["A", "B"][i] for i in order]
    require(source["logicalToDisplayed"] == mapping, "Source orientation differs")
    c = TaskConfig(source["conditionId"], d["templateId"], float(d["scale"]), float(d["gap"]),
                   tuple(d["priorCentres"][i] for i in order), tuple(d["priorSpreads"][i] for i in order),
                   float(d["reportWidth"]), d["budgetMillihours"], d["costMillihours"], 0,
                   max_assessments=d["budgetMillihours"] // d["costMillihours"],
                   alpha=common["utility"]["gain_exponent"], loss_multiplier=common["utility"]["loss_multiplier"],
                   gpa_baseline=common["gpa"]["baseline"], gpa_scale=common["gpa"]["utility_scale"])
    config = asdict(c)
    config["prior_centres"], config["prior_spreads"] = list(c.prior_centres), list(c.prior_spreads)
    options = {k: session["exported"][k] for k in ("profileId", "conditionId", "scheduleVariant",
               "participantId", "sessionId", "seed", "startedAt")}
    return {"populationId": "working", "conditionId": source["conditionId"],
            "scheduleVariant": source["scheduleVariant"], "mainIndex": source["mainIndex"],
            "config": config, "sourceDefinition": d, "sourceOptions": options,
            "logicalToDisplayed": mapping, "initialLogBinary64": log_bits(source["initialLogs"]),
            "sourceRecipeFingerprint": source["meta"]["resolvedRecipeFingerprint"]}


def selected_roots(source, selected):
    validate_source_scope(source, selected)
    wanted = {root_key(row) for row in selections(selected)}
    sessions = {(s["conditionId"], s["scheduleVariant"]): s for s in source["sessions"]}
    require(len(sessions) == len(source["sessions"]), "Duplicate source session")
    result = {}
    for row in source["states"]:
        key = root_key(row)
        if row["prefixLength"] == 0 and row["prefixKind"] == "fixed" and key in wanted:
            require(key not in result, "Duplicate source root")
            result[key] = source_root(row, sessions[key[:2]])
    require(set(result) == wanted, "Missing source roots")
    return [result[key] for key in sorted(result)]


def generate_source(context, output, node="node", *, selected):
    helper = Path(__file__).resolve().with_name("cost_aware_source.cjs")
    request = {**context, "roots": source_selection(selected)}
    script = "const fs=require('fs'),m=require(process.argv[1]);const x=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(m.prepare(x.design,x.analysis,x.construction,x.roots)));"
    with Path(output).open("x") as stream:
        subprocess.run([node, "-e", script, str(helper)], input=compact(request),
                       text=True, stdout=stream, check=True)


def prepare_inputs(context, selected, output, node="node"):
    validate_context(context)
    selected = selections(selected)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    generate_source(context, output / "source_states.json", node, selected=selected)
    roots = selected_roots(read(output / "source_states.json"), selected)
    result = {"schema": "finite-cost-aware-inputs-v1", "scope": "selected_working_trials",
              "context": context, "selection": selected, "roots": roots,
              "publicIdentity": "not_a_historical_manifest"}
    write(output / "inputs.json", result)
    return result
