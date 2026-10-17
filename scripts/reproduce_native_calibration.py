#!/usr/bin/env python3
"""Reproduce the synthetic, budget-only native-state marginal calibration."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tutoring_analysis import native_calibration as calibration
from src.tutoring_analysis import native_readback as readback
from src.tutoring_analysis.expected_outcomes import digest

SMOKE = ("A02:null", "B06:native", "C02:R")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, immutable=True):
    """Publish whole JSON objects; existing scientific records must agree exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError("Existing output has different content: " + str(path))
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        # Contrast weights have a declared arm order; sorted JSON would change it.
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if immutable:
            os.link(temp, path)
        else:
            os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def inputs(folder):
    folder = Path(folder)
    return (read(folder / "lookup.json"), read(folder / "truths.json"),
            read(folder / "analysis.json"), read(folder / "calibration.json"))


def cell_job(arguments):
    folder, output, key, ids = arguments
    lookup, truths, analysis, spec = inputs(folder)
    cell = next(c for c in calibration.requests(analysis, spec)
                if c["id"] + ":" + c["subcase"] == key)
    destination = Path(output) / "batches" / key.replace(":", "__")
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / ".writer.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        old = [read(p) for p in sorted(destination.glob("batch_*.json"))]
        with calibration.PreparedCalibration(lookup, truths, analysis, spec) as prepared:
            result = prepared.run({**cell, "onlyBatchIds": []}, old)
            done = {b["batchId"] for b in old}
            for batch_id in ids:
                if batch_id in done:
                    continue
                result = prepared.run({**cell, "onlyBatchIds": [batch_id]}, result["completedBatches"])
                prepared.verify()
                batch = next(b for b in result["completedBatches"] if b["batchId"] == batch_id)
                write(destination / ("batch_%02d.json" % batch_id), batch)
                print(json.dumps({"cell": key, "batch": batch_id, "attempts": result["completedAttempts"]}), flush=True)
        write(Path(output) / "cells" / (key.replace(":", "__") + ".json"),
              result, immutable=False)
        return {"cell": key, "attempts": result["completedAttempts"],
                "complete": result["complete"]}
    finally:
        lock.unlink()


def summarize(folder, output, selected):
    lookup, truths, analysis, spec = inputs(folder)
    binding = calibration.execution_binding(lookup, truths, analysis, spec)
    calibration._validate_inputs(lookup, binding, analysis, spec)
    if read(Path(output) / "binding.json") != binding:
        raise ValueError("Output was generated with different inputs or numerical code/runtime")
    cells = readback.cells_for(analysis, spec)
    summaries = []
    for key in selected:
        cell = cells[key]
        truth = truths[key]
        calibration._validate_truth_bundle(cell, lookup, truth)
        records = [read(p) for p in sorted((Path(output) / "batches" / key.replace(":", "__")).glob("batch_*.json"))]
        summaries.append(readback.summary(records, cell, readback.specs_for(cell, analysis),
                          truth["candidate"], truth["accepted"], digest(binding),
                          binding["nativeTruthReceiptHash"], analysis))
    counts = {}
    rows = []
    for summary in summaries:
        for t in summary["targets"]:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
            rows.append({"cell": summary["cellId"], "subcase": summary["subcase"],
                         "target": t["testId"], **{name: t[name] for name in
                         ("attempts", "available", "rejections", "coveredAvailableIntervals",
                          "allAttemptRejectionRate", "availableCoverageRate", "status",
                          "typeICalibrationStatus", "coverageCalibrationStatus")},
                         "truth_classification": t["truth"]["classification"],
                         "failures": ";".join(t["failures"])})
    result = {"schema": "native-calibration-readback-v1", "cells": summaries,
              "targetStatusCounts": counts, "humanObservations": False,
              "completedAttempts": sum(x["completedAttempts"] for x in summaries),
              "complete": set(selected) == set(cells) and all(x["complete"] for x in summaries),
              "scope": "pointwise_synthetic_calibration_not_FWER_or_cap20_validation"}
    write(Path(output) / "summary.json", result, immutable=False)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (Path(output) / "targets.csv").write_text(buffer.getvalue())
    return {k: v for k, v in result.items() if k != "cells"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--prepared", type=Path, required=True, help="Generated input directory")
    parser.add_argument("--output", type=Path, help="Generated run directory (run/summarize)")
    parser.add_argument("--source-states", type=Path, help="Optional previously generated source JSON; fully revalidated")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cell", action="append", help="Explicit cell:subcase; repeat for a subset")
    parser.add_argument("--batch-id", type=int, action="append", help="Stable ID 0..39; full batch size remains250")
    parser.add_argument("--workers", type=int, default=1, help="Independent cell processes")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage == "prepare":
        if args.output or args.cell or args.batch_id:
            parser.error("prepare takes --prepared and optional --source-states, not run selectors")
        if args.prepared.exists():
            parser.error("Preparation directory must be new")
        from src.tutoring_analysis.native_lookup import build_lookup
        from src.tutoring_analysis.native_truth import build_truths
        analysis_path = ROOT / "configs/expected_gpa_analysis.json"
        spec_path = ROOT / "configs/native_calibration.json"
        args.prepared.parent.mkdir(parents=True, exist_ok=True)
        if args.source_states is None:
            subprocess.run(["node", str(ROOT / "src/tutoring_analysis/native_source.cjs"),
                            "--design", str(ROOT / "configs/native_design.json"),
                            "--analysis-spec", str(analysis_path), "--calibration-spec", str(spec_path),
                            "--output", str(args.prepared)], check=True)
        else:
            write(args.prepared / "source_states.json", read(args.source_states))
        print("Source states available; reconstructing lookup and independent truths.", flush=True)
        analysis, spec = read(analysis_path), read(spec_path)
        lookup = build_lookup(read(args.prepared / "source_states.json"), analysis, spec)
        print("Finite lookup reconstructed; deriving exact truths.", flush=True)
        truths = build_truths(lookup, analysis, spec)
        for name, value in (("lookup", lookup), ("truths", truths), ("analysis", analysis), ("calibration", spec)):
            write(args.prepared / (name + ".json"), value)
        write(args.prepared / "prepared.json", calibration.execution_binding(lookup, truths, analysis, spec))
        print(json.dumps({"prepared": str(args.prepared), "cells": len(truths)}))
        return
    if args.output is None:
        parser.error("run/summarize require --output")
    if args.source_states is not None:
        parser.error("--source-states applies only to prepare")
    lookup, truths, analysis, spec = inputs(args.prepared)
    binding = calibration.execution_binding(lookup, truths, analysis, spec)
    calibration._validate_inputs(lookup, binding, analysis, spec)
    if read(args.prepared / "prepared.json") != binding:
        raise ValueError("Preparation content/code/runtime differs")
    cells = {c["id"] + ":" + c["subcase"]: c for c in calibration.requests(analysis, spec)}
    selected = args.cell if args.cell is not None else list(cells) if args.mode == "full" else list(SMOKE)
    if len(set(selected)) != len(selected) or not set(selected) <= cells.keys():
        parser.error("Unknown or duplicate cell")
    ids = args.batch_id if args.batch_id is not None else list(range(40)) if args.mode == "full" else [0]
    if len(set(ids)) != len(ids) or any(i not in range(40) for i in ids):
        parser.error("Batch IDs must be unique integers0..39")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / ".writer.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        if args.stage == "run":
            write(args.output / "binding.json", binding)
            jobs = [(str(args.prepared), str(args.output), key, ids) for key in selected]
            if args.workers == 1:
                for job in jobs:
                    print(json.dumps(cell_job(job)), flush=True)
            else:
                with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
                    for result in pool.map(cell_job, jobs):
                        print(json.dumps(result), flush=True)
        print(json.dumps(summarize(args.prepared, args.output, selected)), flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
