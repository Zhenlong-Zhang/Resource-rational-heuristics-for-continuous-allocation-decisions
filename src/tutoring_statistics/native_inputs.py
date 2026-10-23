"""Regenerate both accepted scripted covariate panels from scientific inputs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED_CHANGE = {"condition_id": "C3", "field": "prior_kernel_centres",
                   "old_value": [60, 60], "new_value": [50, 70]}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_revision(base_path: str | Path, patch_path: str | Path) -> tuple[dict, dict, dict]:
    """Resolve exactly one declared edit, returning independent base/resolved dictionaries."""
    patch = json.loads(Path(patch_path).read_text())
    if (patch.get("schema") != "tutoring_design_single_revision_v0.2"
            or patch.get("change") != EXPECTED_CHANGE):
        raise ValueError("Only the frozen C3 [60,60] to [50,70] candidate is supported")
    if file_sha256(base_path) != patch["base_sha256"]:
        raise ValueError("Baseline fixture hash mismatch")
    base = json.loads(Path(base_path).read_text())
    resolved = copy.deepcopy(base)
    row = next(x for x in resolved["conditions"] if x["id"] == "C3")
    if row["prior_kernel_centres"] != patch["change"]["old_value"]:
        raise ValueError("Unexpected baseline C3 centres")
    row["prior_kernel_centres"] = list(patch["change"]["new_value"])
    # Metadata changes are explicit; the sole scientific-field edit is above.
    row["initial_information"] = "differentiated_broad_direct_priors"
    resolved["schema"] = "local_experiment_design_candidates_v0.2"
    resolved["date"] = patch["date"]
    resolved["status"] = "frozen_prospective_v0.2_candidate_computation_pending_not_human_validated"
    for condition in resolved["conditions"]:
        condition["status"] = ("prospective_revised_candidate" if condition["id"] == "C3"
                               else "unchanged_v0.1_finite_task_computational_evidence_retained")
    resolved["revision_provenance"] = {
        "base_sha256": patch["base_sha256"], "patch_sha256": file_sha256(patch_path),
        "engine_sha256": patch["engine_sha256"], "scientific_change": patch["change"],
        "metadata_changes": ["schema", "date", "status", "condition_statuses",
                             "C3_initial_information", "revision_provenance"],
        "status_note": "This frozen pre-run input is never relabeled after outcomes. See stage selection_gate.json and the current plan for the tested disposition.",
        "calibration_executed_note": "Inherited executed=false describes this resolved candidate before the prospective runs, not absence of v0.1 evidence.",
    }
    return base, resolved, patch


def generate(output, *, node="node"):
    from .reproduction import numerical_runtime
    numerical_runtime()
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if not version.startswith("v22."):
        raise ValueError("Use the original Node 22 environment")
    _, design, _ = load_revision(ROOT / "data/base_design.json", ROOT / "data/design_revision.json")
    data = (json.dumps(design, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    identity = json.loads((ROOT / "data/native_identity.json").read_text())
    if hashlib.sha256(data).hexdigest() != identity["resolved_design_sha256"]:
        raise ValueError("Resolved design differs from the accepted exact input")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    design_path = output / "resolved_design.json"
    design_path.write_bytes(data)
    subprocess.run([node, str(ROOT / "initial_panel.cjs"), "--design", str(design_path),
                    "--output", str(output / "initial_panel.json"), "--seed", identity["h3_h5_seed"]], check=True)
    subprocess.run([node, str(ROOT / "joint_panel.cjs"), "build", "--design", str(design_path),
                    "--output", str(output / "native_joint_panel.json")], check=True)
    subprocess.run([node, str(ROOT / "joint_panel.cjs"), "normalize", "--design", str(design_path),
                    "--input", str(output / "native_joint_panel.json"),
                    "--output", str(output / "native_joint_normalized.json")], check=True)
