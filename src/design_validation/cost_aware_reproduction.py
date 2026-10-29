"""Portable selected-trial file stages, separate from numerical implementations."""
import hashlib
from pathlib import Path

from .cost_aware_inputs import (compact, digest, load_context, identical, read, require,
                                root_key, selected_roots, selections, trial_name, write)


def load_inputs(directory, context, expected_selection):
    directory = Path(directory)
    inputs = read(directory / "inputs.json")
    require(set(inputs) == {"schema", "scope", "context", "selection", "roots", "publicIdentity"}, "Input fields")
    require(inputs["schema"] == "finite-cost-aware-inputs-v1"
            and inputs["scope"] == "selected_working_trials"
            and inputs["publicIdentity"] == "not_a_historical_manifest", "Selected public input schema")
    require(identical(inputs["context"], context), "External scientific settings differ")
    require(identical(inputs["selection"], selections(expected_selection)), "Missing, extra or changed selected coordinates")
    roots = {root_key(r): r for r in inputs["roots"]}
    require(len(roots) == len(inputs["roots"]) and set(roots) == {root_key(r) for r in inputs["selection"]}, "Exact selected root roster")
    return inputs


def produce(config, selection_file, inputs_directory, output, *, resume=False):
    """Compute each selected history once and retain every raw scientific row."""
    from .budget_horizon import budget_templates
    from .terminal_time import reference_rate
    from .cost_aware_policy import run_model_trial

    context = load_context(config)
    inputs = load_inputs(inputs_directory, context, read(selection_file))
    expected_roots = selected_roots(read(Path(inputs_directory) / "source_states.json"), inputs["selection"])
    require(identical(inputs["roots"], expected_roots), "Root adapter/source mismatch")
    output = Path(output); output.mkdir(parents=True, exist_ok=resume)
    require(not (output / "reference.json").exists(), "Completed producer must be reused, not overwritten")
    reference = reference_rate(budget_templates(context["design"]))
    identity = digest(inputs)
    manifest = {"schema": "finite-cost-aware-records-v1", "scope": "selected_working_trials",
                "inputsDigest": identity, "selection": inputs["selection"], "roots": inputs["roots"],
                "reference": reference, "trials": [], "historicalAcceptanceClaim": False}
    roots = {root_key(r): r for r in inputs["roots"]}
    for selection in inputs["selection"]:
        filename = trial_name(selection) + ".ndjson"
        hashes = []
        if resume and (output / filename).exists():
            from .cost_aware_inputs import parse
            history = None
            with (output / filename).open("rb") as stream:
                for line in stream:
                    require(line.endswith(b"\n"), "Partial trial: preserve file separately before retrying this trial")
                    raw = line[:-1]
                    record = parse(raw.decode("utf-8"))
                    require(history is None, "Extra records after completed history")
                    if not hashes:
                        require(record == {"kind": "trial", "selection": selection, "inputsDigest": identity},
                                "Changed resumed trial inputs")
                    if record["kind"] == "history":
                        history = record["value"]
                    hashes.append(hashlib.sha256(raw).hexdigest())
            require(history is not None and history["status"] == "completed",
                    "Incomplete trial: preserve file separately before retrying this trial")
            manifest["trials"].append({"selection": selection, "file": filename,
                                       "rowSha256": hashes, "history": history})
            continue
        with (output / filename).open("xb") as stream:
            def emit(record):
                raw = compact(record).encode("utf-8")
                stream.write(raw + b"\n")
                hashes.append(hashlib.sha256(raw).hexdigest())
            emit({"kind": "trial", "selection": selection, "inputsDigest": identity})
            history = run_model_trial(roots[root_key(selection)], selection, context["settings"], reference, emit)
        manifest["trials"].append({"selection": selection, "file": filename,
                                   "rowSha256": hashes, "history": history})
    # The reference is also the completion marker; partial directories cannot pass.
    write(output / "reference.json", manifest)
    return manifest


def check(config, selection_file, inputs_directory, records_directory, output, node="node"):
    from .cost_aware_validation import validate_computation

    context = load_context(config)
    inputs = load_inputs(inputs_directory, context, read(selection_file))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report = validate_computation(inputs, records_directory, expected_selection=read(selection_file), node=node)
    write(output / "verification.json", report)
    return report
