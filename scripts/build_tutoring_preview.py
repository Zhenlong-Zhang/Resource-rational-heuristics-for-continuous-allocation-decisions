#!/usr/bin/env python3
"""Build the standalone tutoring preview from public inputs; no policy calculation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("engine.js", "ui.js", "ui.css", "preview.js", "local_strategy.js")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def resolve_design(root: Path = ROOT) -> tuple[dict, dict, dict]:
    """Add the display contract to the published, fixed scientific recipe.

    The task engine resolves the budget-only 14-task profile from this base.
    Historical proposal metadata is deliberately not reconstructed: the new
    package has its own fingerprint, while numerical/task fields are preserved.
    """
    config_path = root / "configs/tutoring_preview.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if (config["schema"] != "tutoring_standalone_preview_v1"
            or config["design_file"] != "tutoring_conditions.json"
            or config["profile_id"] != "pilot-r9-design-v1"
            or config["participant_flow_version"] != "r10-prepractice-v1"
            or (config["practice_tasks"], config["main_tasks"]) != (4, 10)):
        raise ValueError("Unsupported standalone preview configuration")
    design_path = root / "configs" / config["design_file"]
    design_bytes = design_path.read_bytes()
    if sha256(design_bytes) != config["design_sha256"]:
        raise ValueError("Scientific input hash differs from the preview contract")
    design = json.loads(design_bytes)
    for key in ("display", "feedback"):
        if key in design["common"]:
            raise ValueError("Preview field would overwrite the scientific input: " + key)
        design["common"][key] = config[key]
    inputs = {str(path.relative_to(root)): sha256(data) for path, data in (
        (config_path, config_bytes), (design_path, design_bytes))}
    return design, config, inputs


def build(output: Path, root: Path = ROOT) -> dict:
    """Write a fresh standalone HTML, resolved input and content-bound manifest."""
    design, config, inputs = resolve_design(root)
    source = root / "src/tutoring_task"
    raw = {name: (source / name).read_bytes() for name in ASSETS}
    assets = {name: data.decode("utf-8") for name, data in raw.items()}
    # Escape JSON before inlining, then reject script/style terminators in source.
    settings = ("window.TUTORING_DESIGN=" + compact(design).replace("<", "\\u003c")
                + ";window.TUTORING_PROFILE=" + compact(config["profile_id"])
                + ";window.TUTORING_LOCAL_STRATEGY=true;")
    scripts = "\n".join((assets["engine.js"], assets["ui.js"], settings,
                          assets["local_strategy.js"], assets["preview.js"]))
    css = assets["ui.css"]
    if "</script" in scripts.lower() or "</style" in css.lower():
        raise ValueError("Unsafe inline source terminator")
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tutoring allocation: standalone preview</title><style>{css}
body{{margin:0;padding:18px 22px 36px;background:#e9eeeb;font-family:ui-sans-serif,system-ui,sans-serif;color:#173b3d}}
.review-banner,#review-controls{{max-width:1140px;margin:0 auto 12px;font-size:12px;box-sizing:border-box}}
.review-banner{{letter-spacing:1.3px;font-weight:650;color:#536861;padding:5px 3px}}
#review-controls{{background:#fff;border:1px solid #d3dfd8;border-radius:10px;padding:12px 16px}}
#review-controls summary{{cursor:pointer;font-weight:600}}#review-controls button,#review-controls select{{font:inherit;padding:8px;margin:5px;border:1px solid #b9cec3;border-radius:7px;background:#f5f9f6;color:#173b3d}}
@media(max-width:460px){{body{{padding:12px 7px 24px}}.review-banner{{font-size:9px;letter-spacing:.7px}}}}
</style></head><body>
<aside class="review-banner">STANDALONE PREVIEW: fictional students, no server submission</aside>
<details id="review-controls"><summary>Researcher preview controls</summary>
<p>New sessions preserve previous browser records. This local selector is not cohort randomization.</p>
<label>Preview condition <select id="preview-condition"><option>C1</option><option>C2</option><option selected>C3</option><option>C4</option><option>C5</option></select></label>
<button id="preview-new" type="button">Create new review session</button>
<button id="preview-export" type="button">Download local JSON</button>
<p id="preview-status" role="status"></p></details>
<main id="tutoring-root"></main><script>{scripts}</script></body></html>'''
    products = {"index.html": html.encode("utf-8"),
                "resolved_design.json": (compact(design) + "\n").encode("utf-8")}
    manifest = {
        "schema": "tutoring_standalone_build_v1", "humanData": False,
        "formalPolicyDraws": 0, "qualtricsKit": False,
        "previewProfileId": config["profile_id"],
        "participantFlowVersion": config["participant_flow_version"],
        "practiceTasks": 4, "mainTasks": 10,
        "questionLimitRule": "budget_only",
        "inputHashes": inputs,
        "sourceHashes": {"src/tutoring_task/" + name: sha256(data)
                         for name, data in raw.items()},
        "builderSHA256": sha256((root / "scripts/build_tutoring_preview.py").read_bytes()),
        "outputHashes": {name: sha256(data) for name, data in products.items()},
        "scope": "Standalone software preview; no formal study deployment or scientific policy run.",
        "identityScope": "Public input metadata gives a new design fingerprint. No automatic migration of historical task sessions.",
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in products.items():
        (output / name).write_bytes(data)
    (output / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
