"""Portable full working-population production and exact-roster reduction.

Shards change file ownership only. Every scientific trial retains its original
condition, variant, main, replicate and scenario keys. Existing selected-trial
production and independent replay remain the numerical implementations.
"""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
import scipy

from . import cost_aware_aggregation as A
from .cost_aware_inputs import (FROZEN_RECIPE, compact, digest, identical, load_context,
    log_bits, read, require, root_key, selected_roots, selections, source_selection,
    trial_name, write, prepare_inputs, parse)
from .cost_aware_reproduction import produce, check
from .projection_reference import frozen_plans

ARMS = ("C1", "C2", "C3", "C4", "C5")
SCENARIOS = tuple(p + "|rho_" + format(r, "g") for p in FROZEN_RECIPE["profiles"]
                  for r in FROZEN_RECIPE["rho_multipliers"])
SOURCE_FILES = (
    "design_validation/cost_aware_campaign.py", "design_validation/cost_aware_aggregation.py",
    "design_validation/projection_reference.py",
    "design_validation/cost_aware_inputs.py", "design_validation/cost_aware_reproduction.py",
    "design_validation/cost_aware_validation.py", "design_validation/cost_aware_policy.py",
    "design_validation/cost_aware_reference.py", "design_validation/cost_aware_bounds.py",
    "design_validation/cost_aware_source.cjs", "design_validation/bounded_task.py",
    "design_validation/budget_horizon.py", "design_validation/terminal_time.py",
    "tutoring_analysis/expected_outcomes.py", "tutoring_analysis/nominal_inference.py",
    "tutoring_analysis/native_reconstruction.py", "tutoring_analysis/native_engine.cjs")


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@lru_cache(maxsize=1)
def numerical_runtime():
    """Record loaded numerical binaries once per process; no machine hostname."""
    from numpy.core import _multiarray_umath
    from numpy.linalg import _umath_linalg
    from scipy.linalg import _flapack
    from threadpoolctl import threadpool_info
    libraries = []
    for info in threadpool_info():
        libraries.append({"api": info["internal_api"], "version": info["version"],
                          "threads": info["num_threads"], "prefix": info["prefix"],
                          "binary": file_digest(info["filepath"])})
    require(any(x["api"] in ("mkl", "openblas", "blis") for x in libraries),
            "Loaded numerical library identity unavailable")
    mkl_effective = None
    if any(x["api"] == "mkl" for x in libraries):
        import mkl
        mkl_effective = mkl.cbwr_get()
    return {"python": platform.python_version(), "machine": platform.machine(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "numpyBinary": file_digest(_multiarray_umath.__file__),
            "numpyLinalgBinary": file_digest(_umath_linalg.__file__),
            "lapackBinary": file_digest(_flapack.__file__),
            "loadedLibraries": sorted(libraries, key=lambda x: (x["prefix"], x["binary"])),
            "mklEffective": mkl_effective,
            "threadSettings": {k: os.environ.get(k) for k in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "MKL_CBWR")},
            "cpuFeatures": dict(_multiarray_umath.__cpu_features__)}


def scientific_identity(node="node"):
    root = Path(__file__).resolve().parents[1]
    return {"files": {name: file_digest(root / name) for name in SOURCE_FILES},
            "runtime": numerical_runtime(),
            "node": subprocess.check_output([node, "--version"], text=True).strip()}


def plan(mode="full", people_per_shard=8):
    require(mode in ("smoke", "full"), "Unknown scale")
    require(type(people_per_shard) is int and 1 <= people_per_shard <= 64,
            "Shard size must be 1..64 persons")
    tasks = []
    for scenario in SCENARIOS if mode == "full" else ("base|rho_100",):
        for arm in ARMS if mode == "full" else ("C2",):
            for variant in range(9) if mode == "full" else (0,):
                for start in range(0, 64 if mode == "full" else 1, people_per_shard):
                    tasks.append({"ordinal": len(tasks), "conditionId": arm,
                        "scenarioId": scenario, "scheduleVariant": variant,
                        "replicateIds": list(range(start, min(start + people_per_shard,
                                                              64 if mode == "full" else 1)))})
    return {"schema": "cost-aware-campaign-plan-v1", "mode": mode,
            "peoplePerShard": people_per_shard, "tasks": tasks,
            "persons": 51840 if mode == "full" else 1,
            "histories": 518400 if mode == "full" else 10,
            "humanObservations": False, "historicalReceipt": False}


def task_selection(task):
    return selections([{"conditionId": task["conditionId"],
        "scenarioId": task["scenarioId"], "scheduleVariant": task["scheduleVariant"],
        "replicateId": rep, "mainIndex": main}
        for rep in task["replicateIds"] for main in range(1, 11)])


def prepare(config, output, mode="full", people_per_shard=8, node="node"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    roster = plan(mode, people_per_shard)
    context = load_context(config)
    roots = {}
    for task in roster["tasks"]:
        for row in task_selection(task):
            roots.setdefault(root_key(row), {**row, "replicateId": 0,
                                            "scenarioId": "base|rho_100"})
    selected = selections(list(roots.values()))
    prepared = prepare_inputs(context, selected, output / "roots", node)
    write(output / "plan.json", roster)
    write(output / "prepared.json", {"schema": "cost-aware-campaign-prepared-v1",
          "planDigest": digest(roster), "inputsDigest": digest(prepared),
          "sourceDigest": file_digest(output / "roots/source_states.json"),
          "scientificIdentity": scientific_identity(node)})
    return roster


def load_prepared(config, directory, node="node"):
    directory = Path(directory)
    roster, marker = read(directory / "plan.json"), read(directory / "prepared.json")
    require(identical(roster, plan(roster["mode"], roster["peoplePerShard"])), "Frozen roster changed")
    inputs = read(directory / "roots/inputs.json")
    require(identical(inputs["context"], load_context(config)), "Scientific context changed")
    require(marker == {"schema": "cost-aware-campaign-prepared-v1",
        "planDigest": digest(roster), "inputsDigest": digest(inputs),
        "sourceDigest": file_digest(directory / "roots/source_states.json"),
        "scientificIdentity": scientific_identity(node)}, "Prepared inputs/source/runtime changed")
    return roster, inputs


def shard_path(output, ordinal):
    return Path(output) / "shards" / f"{ordinal:05d}"


def verify_shard(path, task, context, expected_inputs=None, preparation=None):
    """Check the immutable completion commitments, not rerun a trajectory."""
    path = Path(path)
    receipt = read(path / "complete.json")
    require(receipt["schema"] == "cost-aware-campaign-shard-v1"
            and receipt["task"] == task and receipt["contextDigest"] == digest(context),
            "Wrong shard identity")
    wanted = {"binding.json", "selection.json", "inputs/inputs.json", "inputs/source_states.json",
              "records/reference.json", "checked/verification.json", "checked/records_binding.json"}
    wanted.update("records/" + trial_name(s) + ".ndjson" for s in task_selection(task))
    require(set(receipt["files"]) == wanted, "Exact shard artifact roster required")
    for name in sorted(wanted):
        require(file_digest(path / name) == receipt["files"][name], "Changed completed file: " + name)
    reference = read(path / "records/reference.json")
    verified = read(path / "checked/verification.json")
    check_pins = read(path / "checked/records_binding.json")
    expected_pins = {name: receipt["files"][name] for name in wanted
                     if name.startswith("records/") or name.startswith("inputs/")}
    require(check_pins == expected_pins, "Checker is not bound to current records")
    selected = task_selection(task)
    inputs = read(path / "inputs/inputs.json")
    require(identical(inputs["context"], context), "Changed shard scientific context")
    if expected_inputs is not None:
        expected_roots = [r for r in expected_inputs["roots"]
                          if root_key(r) in {root_key(s) for s in selected}]
        require(identical(inputs["roots"], expected_roots), "Shard uses a different prepared root catalog")
        require(read(path / "binding.json") == {"task": task, "preparation": preparation},
                "Shard source/runtime/preparation identity changed")
    require(identical(reference["selection"], selected) and identical(inputs["selection"], selected)
            and identical(verified["selection"], selected), "Changed selected coordinates")
    require(reference["inputsDigest"] == verified["inputsDigest"] == digest(inputs), "Unbound verification")
    require(verified["verification"] == "checked" and verified["allSelectedNumericallyCompleted"],
            "Unresolved independent replay")
    require([r["selection"] for r in reference["trials"]] == selected
            and [r["selection"] for r in verified["trials"]] == selected
            and all(r["history"]["status"] == "completed" for r in reference["trials"]),
            "Missing or unresolved trial")
    return reference


def _run_task(args):
    config, prepared, output, task, node = args
    _, inputs = load_prepared(config, prepared, node)
    preparation = read(Path(prepared) / "prepared.json")
    path = shard_path(output, task["ordinal"])
    if (path / "complete.json").is_file():
        verify_shard(path, task, inputs["context"], inputs, preparation)
        return {"ordinal": task["ordinal"], "status": "reused_verified"}
    binding = {"task": task, "preparation": preparation}
    if path.exists():
        require(read(path / "binding.json") == binding, "Unbound partial shard cannot resume")
    else:
        path.mkdir(parents=True, exist_ok=False)
        write(path / "binding.json", binding)
    selected = task_selection(task)
    if (path / "selection.json").exists():
        require(identical(read(path / "selection.json"), selected), "Resumed selection changed")
    else:
        write(path / "selection.json", selected)
    source = read(Path(prepared) / "roots/source_states.json")
    wanted = {root_key(s) for s in selected}
    source["states"] = [r for r in source["states"] if root_key(r) in wanted]
    source["sessions"] = [s for s in source["sessions"]
                          if (s["conditionId"], s["scheduleVariant"]) in {k[:2] for k in wanted}]
    source["rootSelection"] = source_selection(selected)
    roots = selected_roots(source, selected)
    (path / "inputs").mkdir(exist_ok=True)
    for name, value in (("source_states.json", source),
                        ("inputs.json", {**inputs, "selection": selected, "roots": roots})):
        target = path / "inputs" / name
        if target.exists():
            require(identical(read(target), value), "Resumed inputs changed")
        else:
            write(target, value)
    if not (path / "records/reference.json").exists():
        produce(config, path / "selection.json", path / "inputs", path / "records", resume=True)
    (path / "checked").mkdir(exist_ok=True)
    pins = {p.relative_to(path).as_posix(): file_digest(p)
            for folder in (path / "records", path / "inputs") for p in folder.iterdir() if p.is_file()}
    binding_file = path / "checked/records_binding.json"
    if binding_file.exists():
        require(read(binding_file) == pins, "Records changed after independent checking started")
    else:
        require(not (path / "checked/verification.json").exists(), "Unbound older checker result")
        write(binding_file, pins)
    if (path / "checked/verification.json").exists():
        report = read(path / "checked/verification.json")
    else:
        from .cost_aware_validation import validate_computation
        report = validate_computation(read(path / "inputs/inputs.json"), path / "records",
                                      expected_selection=selected, node=node)
        write(path / "checked/verification.json", report)
    require(report["allSelectedNumericallyCompleted"], "Numerical failure remains visible")
    files = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
    write(path / "complete.json", {"schema": "cost-aware-campaign-shard-v1",
          "task": task, "contextDigest": digest(inputs["context"]),
          "files": {name: file_digest(path / name) for name in files}})
    verify_shard(path, task, inputs["context"], inputs, preparation)
    return {"ordinal": task["ordinal"], "status": "computed_and_checked"}


def run(config, prepared, output, ordinals, workers=1, node="node"):
    roster, _ = load_prepared(config, prepared, node)
    require(type(workers) is int and workers > 0, "Positive worker count required")
    require(bool(ordinals) and len(set(ordinals)) == len(ordinals)
            and all(type(i) is int and 0 <= i < len(roster["tasks"]) for i in ordinals),
            "Unique in-roster ordinals required")
    args = [(config, prepared, output, roster["tasks"][i], node) for i in ordinals]
    if workers == 1:
        for arg in args:
            yield _run_task(arg)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(args))) as pool:
            yield from pool.map(_run_task, args)


def enrich_roots(inputs, manifest):
    """Reconstruct the original response-independent native projection plans."""
    roots = {}
    for source in inputs["roots"]:
        means = [A.finite_mean(v) for v in A.decode_logs(source["initialLogBinary64"])]
        row = {**source, "variant": source["scheduleVariant"], "initialMeans": means,
               "x0": (means[0] - means[1]) / (source["config"]["total_millihours"] / 1000),
               "gapOffset": float(source["sourceDefinition"]["gap"])}
        roots[root_key(row)] = row
    groups = sorted({k[:2] for k in roots})
    predictors = tuple(tuple(roots[c, v, main]["x0"] for main in range(1, 11)) for c, v in groups)
    reference = frozen_plans(predictors)
    require(len(reference["plans"]) == len(groups), "Projection roster changed")
    for (arm, variant), frozen in zip(groups, reference["plans"]):
        batch = [roots[arm, variant, main] for main in range(1, 11)]
        for row in batch:
            row["initialProjectionPlan"] = frozen
            row["projectionRuntime"] = reference["runtime"]
            key = root_key(row)
            roots[key] = A.make_record("root", manifest, [manifest, *key], row)
    return roots


class Store:
    def __init__(self):
        self.rows = {}

    def add(self, row):
        A.validate_record(row)
        ref = A.reference(row)
        require(ref["recordId"] not in self.rows, "Duplicate scientific identity")
        self.rows[ref["recordId"]] = row
        return ref

    def read(self, ref):
        row = self.rows[ref["recordId"]]
        require(A.reference(row) == ref, "Record reference mismatch")
        return row


def reduce_scenario(scenario, tasks, inputs, output, preparation):
    """Reduce one complete scenario, keeping memory bounded across scenarios."""
    identity = A.semantic(preparation)
    roots = enrich_roots(inputs, identity)
    store, people, histories, decisions = Store(), {}, {}, {}
    for task in tasks:
        result = verify_shard(shard_path(output, task["ordinal"]), task, inputs["context"], inputs, preparation)
        for trial in result["trials"]:
            s, h = trial["selection"], trial["history"]
            c, v, rep, main = s["conditionId"], s["scheduleVariant"], s["replicateId"], s["mainIndex"]
            person_key = (c, v, rep)
            prefix = [identity, "working", scenario, c, v, rep]
            root = roots[c, v, main]
            hist = A.make_record("history", identity, prefix + [main], {
                **{k: value for k, value in h.items() if k != "finalSummary"},
                "trialId": A.semantic(prefix + [main]), "personId": A.semantic(prefix),
                "scenarioId": scenario, "root": A.reference(root)})
            histories[c, v, rep, main] = store.add(hist)
            metric = A.model_trial_metrics(hist, root, h["finalSummary"])
            people.setdefault(person_key, []).append(metric)
            if rep == 0:
                with (shard_path(output, task["ordinal"]) / "records" / trial["file"]).open() as stream:
                    next(stream)
                    initial = parse(next(stream))
                require(initial["kind"] == "decision" and initial["queryIndex"] == 0
                        and initial["value"]["rootState"]
                        and initial["value"]["numericalStatus"] == "valid"
                        and initial["value"]["scenarioId"] == scenario
                        and initial["value"]["budget"] == root["config"]["total_millihours"]
                        and initial["value"]["orderedLogBinary64"] == root["initialLogBinary64"],
                        "Original initial decision required")
                decision = A.make_record("decision", identity, [identity, scenario, c, v, main, "initial"],
                    {"rootState": True, "scenarioId": scenario, "numericalStatus": "valid",
                     "action": initial["value"]["action"]})
                decisions[c, v, main] = store.add(decision)
    person_refs = {}
    for key, rows in sorted(people.items()):
        rows.sort(key=lambda r: r["mainIndex"])
        person_refs[key] = store.add(A.model_person_endpoints(rows, inputs["context"]["analysis"],
                                                            rows[0]["initialProjectionPlan"]))
    def estimands():
        for namespace, spec in A.endpoint_specs(inputs["context"]["analysis"]):
            routes = ("deterministic_root_Q",) if spec["endpoint"] == "Q" else (
                "assigned_eligible_person", "standardized_variant_diagnostic")
            for route in routes:
                yield {"spec": spec, "route": route, "scenarioId": scenario,
                    "populationScope": "working", "namespace": namespace,
                    "rootDecisionRefs": lambda arm, v: [decisions[arm, v, i] for i in range(1, 11)],
                    "personRefs": lambda arm, v: [person_refs[arm, v, r] for r in range(64)]}
    def performance():
        for arm in ARMS:
            yield {"conditionId": arm, "scenarioId": scenario, "populationId": "working",
                "historyRefs": lambda v, rep, arm=arm: [histories[arm, v, rep, i] for i in range(1, 11)]}
    manifest = {"manifestHash": identity, "estimandGroups": estimands, "performanceGroups": performance}
    return list(A.summarize_bridge(manifest, store, tick=lambda: None))


def aggregate(config, prepared, output, destination, node="node"):
    roster, inputs = load_prepared(config, prepared, node)
    preparation = read(Path(prepared) / "prepared.json")
    present, missing = [], []
    for task in roster["tasks"]:
        (present if (shard_path(output, task["ordinal"]) / "complete.json").is_file()
         else missing).append(task["ordinal"])
    result = {"schema": "cost-aware-campaign-summary-v1", "mode": roster["mode"],
        "expectedShards": len(roster["tasks"]), "presentShards": len(present),
        "missingOrdinals": missing, "fullStudyComplete": False, "records": [],
        "scope": "original_terminal_cost_only_model;candidate_H6_H7;no_human_inference",
        "preparation": preparation}
    if missing or roster["mode"] != "full":
        for i in present:
            verify_shard(shard_path(output, i), roster["tasks"][i], inputs["context"], inputs, preparation)
        result["reason"] = "missing_shards" if missing else "smoke_is_not_full_study"
    else:
        for scenario in SCENARIOS:
            if "projectionReference" not in result:
                reference_roots = enrich_roots(inputs, A.semantic(preparation))
                result["projectionReference"] = {
                    "runtime": next(iter(reference_roots.values()))["projectionRuntime"],
                    "plans": {f"{c}|{v}": reference_roots[c, v, 1]["initialProjectionPlan"]
                              for c, v in sorted({k[:2] for k in reference_roots})}}
            tasks = [t for t in roster["tasks"] if t["scenarioId"] == scenario]
            result["records"].extend(reduce_scenario(scenario, tasks, inputs, output, preparation))
        kinds = [r["schema"] for r in result["records"]]
        require(len({r["recordId"] for r in result["records"]}) == len(kinds), "Duplicate aggregate identity")
        expected_estimands = {(scenario, spec["id"], route)
            for scenario in SCENARIOS for _, spec in A.endpoint_specs(inputs["context"]["analysis"])
            for route in (("deterministic_root_Q",) if spec["endpoint"] == "Q" else
                          ("assigned_eligible_person", "standardized_variant_diagnostic"))}
        actual_estimands = {(r["scenarioId"], r["testId"], r["route"])
            for r in result["records"] if r["schema"].endswith("estimand-v1")}
        actual_performance = {(r["scenarioId"], r["conditionId"])
            for r in result["records"] if r["schema"].endswith("performance-v1")}
        require(actual_estimands == expected_estimands
                and actual_performance == {(s, c) for s in SCENARIOS for c in ARMS},
                "Exact original aggregate coordinates required")
        require(kinds.count("finite-cost-aware-reduction-performance-v1") == 90
                and kinds.count("finite-cost-aware-reduction-estimand-v1") == 2088,
                "Original 90 performance and 2088 Q/nonQ rows required")
        require(all(r["status"] not in ("UNRESOLVED_NUMERICAL", "UNRESOLVED_COMPUTATION")
                    for r in result["records"]), "Computationally unresolved aggregate")
        result.update(fullStudyComplete=True, persons=51840, histories=518400,
                      originalInitialCoordinates=8100, originalRoots=450, reason=None)
    write(destination, result)
    return result
