"""Audit bounded-task/Gaussian-reference differences without importing legacy project code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from design_validation.model_bridge import audit_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = audit_file(args.design)
    outputs = [args.output_dir / x for x in ("model_bridge_summary.json", "model_bridge_templates.csv",
                                            "model_bridge_manifest.json")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Audit outputs already exist; choose a new output directory")
    if any(name == "src" or name.startswith("src.") or name == "mdp" or name.startswith("mdp.")
           or name == "experiments" or name.startswith("experiments.") for name in sys.modules):
        raise RuntimeError("Legacy import detected")
    root = Path(__file__).resolve().parents[1]
    sources = ["src/design_validation/model_bridge.py", "scripts/audit_tutoring_model_bridge.py"]
    manifest = {"design_sha256": summary["design_sha256"], "stochastic_simulation": False,
                "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in sources}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with outputs[1].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs[2].write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
