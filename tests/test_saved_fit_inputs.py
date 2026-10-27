"""Portable input/relocation fixtures, not fitted models or calibration evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.tutoring_statistics import saved_fit_inputs as inputs
from src.tutoring_statistics import independent_saved_fit as reader
from src.tutoring_statistics import power_reference


def fixture(root):
    """Build a labeled synthetic example of the genuine public artifact layout."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    item = save("fixture_runtime.json", {"fixture": True})
    required = ("reproduction.py", "kernel_entry.R", "numerical_environment.json", "native_inputs.py",
        "native_identity.json", "power_reference.py", "power_origins.py", "joint_generation.py", "joint_fit.R",
        "population_fit.R", "cluster_fit.R", "independent_power.py", "independent_transport.py",
        "independent_cluster_rds.R", "source_transport.R", "mixed_fit.R", "joint_design.py")
    source_files = {name: save("source/" + name, {"fixture": True, "role": name}) for name in required}
    sources = save("source_snapshot.json", dict(files=list(source_files.values())))
    manifest = save("power/manifest.json", dict(schema=inputs.PUBLIC_SCHEMA, family="power",
        expectedTotal=80000, identity="new_public_reproduction", sources=sources,
        driver=source_files["reproduction.py"], rDriver=source_files["kernel_entry.R"],
        environment=source_files["numerical_environment.json"], nativeEntry=source_files["native_inputs.py"],
        nativeIdentity=source_files["native_identity.json"],
        native=[save("native/" + name, {"fixture": True}) for name in
                ("resolved_design.json", "initial_panel.json", "h10_canonical.json")],
        numericalRuntime=dict(executable=item, Rscript=item)))
    case = power_reference.case_for(32, 2000)
    people = [(arm + ":fixture", arm) for arm in ("C1", "C2", "C3")]
    rows = [dict(participant=person, condition=arm, need_A=1., need_B=2., y=3.)
            for person, arm in people for _ in range(10)]
    request = save("power/units/00032/actual_request.json", dict(model="H10", reference="C2", rows=rows))
    counts = dict(C1=1, C2=1, C3=1)
    generation = save("power/units/00032/generation.json", dict(case=case, populationCounts=counts,
        recruitedCounts={arm: 193 for arm in counts},
        retentionMask={arm: [True] + [False]*192 for arm in counts},
        retainedParticipantIDs=[p for p, _ in people]))
    spec = save("power/units/00032/spec.json", dict(**inputs.TARGET, fitKey=inputs.FIT_KEY,
        case=case, manifest=manifest, origin=None, populationCounts=counts, request=request, generation=generation))
    # These bytes are intentionally not an RDS; tests never invoke an R reader.
    fit_path = root / "fixture_model.rds"
    fit_path.write_bytes(b"not a fitted model: input admission fixture only")
    fit = dict(path=str(fit_path), sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest())
    raw = dict(status="fitted", fixture=True)
    raw_pin = save("raw.json", raw)
    base = save("baseline.json", dict(manifestSha256=manifest["sha256"], request=request,
        generation=generation, raw=raw, rawArtifact=raw_pin, fitArtifact=fit,
        fitKey=inputs.FIT_KEY, caseID=case["caseID"], repID=1))
    candidate = save("candidate.json", dict(schema="h10-inference-record-v1", manifest=manifest,
        baseline=base, request=request, generation=generation, fitArtifact=fit,
        fitKey=inputs.FIT_KEY, caseID=case["caseID"], repID=1,
        technicalCompleted=True, technicalError=None, candidateMetadata=dict(requestDescriptor=request,
            frameVerified=True, frameRowCount=len(rows), target="fitted_scaled_marginal_Theta",
            inverseVariance=True, cluster="participant", residual="y-Xbeta", priorWeights=False)))
    extracted = save("extracted.json", dict(schema="h10-inference-independent-rds-v1",
        fits={inputs.FIT_KEY: dict(verified=True, requestDescriptor=request, design={})}))
    transport = save("transport.json", dict(schema="h10-power-source-transport-v1",
        newScientificInputs=0, newFitInvocations=0, newCandidateCalculations=0,
        fits={inputs.FIT_KEY: dict(request=request, fitArtifact=fit, exactMatrixInR=True,
            exactResponseInR=True, oneULPMatrixMutationRejected=True, design={})}))
    event = dict(**inputs.TARGET, fitKey=inputs.FIT_KEY, origin="new_population_input", technicalCompleted=True)
    commitment = save("power/events/00032.json", dict(schema=inputs.PUBLIC_SCHEMA, manifest=manifest,
        event=event, eventSha256=hashlib.sha256(json.dumps(event, sort_keys=True, allow_nan=False,
            separators=(",", ":")).encode()).hexdigest(),
        artifacts=[spec, request, generation, base, candidate, raw_pin, fit, extracted, transport]))
    binding = dict(schema=inputs.INPUT_SCHEMA, kind="public_power", target=inputs.TARGET,
                   manifest=manifest, spec=spec, baseline=base, candidate=candidate, commitment=commitment)
    return binding, save


def test_public_actual_layout_and_saved_only_builder(tmp_path):
    binding, save = fixture(tmp_path)
    path = tmp_path / "binding.json"
    inputs.prepare_input(path, reproduction_root=tmp_path)
    assert inputs.load(path)["spec"] == binding["spec"]
    a = inputs.open_input(path)
    b = reader.admit_public_input(binding, reader.Observed())
    assert a["history"]["spec"]["populationCounts"] == b["actualK"] == dict(C1=1, C2=1, C3=1)
    runner = Path(__file__).resolve().parents[1] / "scripts/reproduce_saved_fit_magnitude.py"
    process = subprocess.run([sys.executable, str(runner), "prepare", "--reproduction-root", str(tmp_path),
        "--output", str(tmp_path / "cli_binding.json")], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert inputs.load(tmp_path / "cli_binding.json") == inputs.load(path)


@pytest.mark.parametrize("damage", ["case", "candidate_binding", "metadata_request", "missing_transport", "wrong_exact_flag"])
def test_public_semantic_tamper_with_updated_outer_digests(tmp_path, damage):
    binding, save = fixture(tmp_path)
    commitment = inputs.load(binding["commitment"]["path"])
    if damage == "case":
        old = binding["spec"]
        value = inputs.load(old["path"])
        value["case"]["retentionSeed"] += 1
        new = save("bad_spec.json", value)
        binding["spec"] = new
    elif damage in ("candidate_binding", "metadata_request"):
        old = binding["candidate"]
        value = inputs.load(old["path"])
        if damage == "candidate_binding":
            value["request"] = binding["spec"]
        else:
            value["candidateMetadata"]["requestDescriptor"] = binding["spec"]
        new = save("bad_candidate.json", value)
        binding["candidate"] = new
    else:
        old = next(pin for pin in commitment["artifacts"] if Path(pin["path"]).name == "transport.json")
        if damage == "missing_transport":
            new = None
        else:
            value = inputs.load(old["path"])
            value["fits"][inputs.FIT_KEY]["exactMatrixInR"] = False
            new = save("bad_transport.json", value)
    commitment["artifacts"] = [new if pin == old else pin for pin in commitment["artifacts"]]
    commitment["artifacts"] = [pin for pin in commitment["artifacts"] if pin is not None]
    binding["commitment"] = save("bad_commitment.json", commitment)
    for admit, observe in ((inputs.admit_public, inputs.Observations), (reader.admit_public_input, reader.Observed)):
        with pytest.raises((ValueError, KeyError)):
            admit(binding, observe())


def test_relocation_preserves_logical_descriptor_and_detects_changed_bytes(tmp_path):
    old = tmp_path / "old.json"
    old.write_text('{"value":1}')
    pin = dict(path=str(old), sha256=hashlib.sha256(old.read_bytes()).hexdigest())
    moved = tmp_path / "moved.json"
    old.rename(moved)
    observed = inputs.Observations({pin["path"]: str(moved)})
    assert observed.read(pin) == {"value": 1}
    assert pin["path"] == str(old) and not old.exists()
    moved.write_text('{"value":2}')
    with pytest.raises(ValueError, match="Input changed at closure"):
        observed.fresh()


def test_missing_saved_inputs_never_launch_generation_or_fit(tmp_path):
    with pytest.raises(ValueError):
        inputs.prepare_input(tmp_path / "output.json", reproduction_root=tmp_path)
    assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize("damage", ["missing_role", "replaced_role", "native_roster"])
def test_public_manifest_requires_real_source_roles(tmp_path, damage):
    binding, save = fixture(tmp_path)
    manifest = inputs.load(binding["manifest"]["path"])
    if damage == "missing_role":
        inventory = inputs.load(manifest["sources"]["path"])
        inventory["files"] = [pin for pin in inventory["files"] if Path(pin["path"]).name != "power_reference.py"]
        manifest["sources"] = save("altered_inventory.json", inventory)
    elif damage == "replaced_role":
        manifest["driver"] = manifest["nativeEntry"]
    else:
        manifest["native"] = []
    binding["manifest"] = save("altered_manifest.json", manifest)
    for admit, observe in ((inputs.admit_public, inputs.Observations), (reader.admit_public_input, reader.Observed)):
        with pytest.raises(ValueError, match="source|inventory|native"):
            admit(binding, observe())


@pytest.mark.parametrize("role", ["saved_fit_magnitude.py", "power_reference.py", "publicEntry"])
def test_real_compute_rejects_executing_source_relocation(tmp_path, role):
    from src.tutoring_statistics.saved_fit_runtime import ROOT
    binding, save = fixture(tmp_path)
    entry = Path(__file__).resolve().parents[1] / "scripts/reproduce_saved_fit_magnitude.py"
    path = entry if role == "publicEntry" else ROOT / role
    binding["locations"] = {str(path): binding["spec"]["path"]}
    input_pin = save("input.json", binding)
    process = subprocess.run([sys.executable, str(entry), "compute", "--input", input_pin["path"],
        "--output", str(tmp_path / "no_result.json")], capture_output=True, text=True)
    assert process.returncode != 0 and "Cannot relocate actual execution artifact" in process.stderr
    assert not (tmp_path / "no_result.json").exists()
