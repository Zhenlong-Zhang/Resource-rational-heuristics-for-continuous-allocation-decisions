#!/usr/bin/env python3
"""Deterministic CP planning only; no RNG, fit, calibration or power reduction.

The independent reader uses R qbeta/dbinom and does not import this
implementation. The fixed numerical runtime is required for reproduction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import beta, binom


RULE = dict(confidence=0.95, trueProbability=0.05, maximumHalfWidth=0.01,
            assurance=0.90, selection="first_integer_in_exhaustive_ascending_scan")
WIDTH_GUARD = 1e-12
MASS_GUARD = 1e-10


def evaluate(n):
    k = np.arange(n + 1)
    lo = np.zeros(n + 1)
    hi = np.ones(n + 1)
    lo[1:] = beta.ppf(0.025, k[1:], n - k[1:] + 1)
    hi[:-1] = beta.ppf(0.975, k[:-1] + 1, n - k[:-1])
    mass = binom.pmf(k, n, RULE["trueProbability"])
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()
            and np.isfinite(mass).all() and (mass >= 0).all()
            and (lo >= 0).all() and (hi <= 1).all() and (lo <= hi).all()):
        raise ValueError("NUMERIC_UNRESOLVED: invalid finite interval/mass")
    total = math.fsum(map(float, mass))
    if abs(total - 1) > MASS_GUARD:
        raise ValueError("NUMERIC_UNRESOLVED: binomial mass does not sum to 1")
    half = (hi - lo) / 2
    margin = float(np.min(np.abs(half - RULE["maximumHalfWidth"])))
    if margin <= WIDTH_GUARD:
        raise ValueError("NUMERIC_UNRESOLVED: width too close to threshold")
    passing = half <= RULE["maximumHalfWidth"]
    assurance = math.fsum(map(float, mass[passing]))
    if (assurance < 0 or assurance > 1 + MASS_GUARD
            or abs(assurance - RULE["assurance"]) <= MASS_GUARD):
        raise ValueError("NUMERIC_UNRESOLVED: assurance at/invalid near threshold")
    # Never clamp a numerical result or move the planning threshold.
    return dict(B=n, assurance=assurance, minimumWidthMargin=margin,
                totalMass=total, passingCounts=int(passing.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (platform.python_version(), np.__version__, scipy.__version__) != (
            "3.10.9", "1.23.5", "1.10.0"):
        raise RuntimeError("Changed precision runtime")
    args.output.mkdir(exist_ok=False)
    # Conservative finite stop: Hoeffding tails bound each CP endpoint's
    # distance from k/n by sqrt(log(40)/(2B)). This is not a B search menu.
    bound = math.ceil(math.log(40) / (2 * RULE["maximumHalfWidth"] ** 2))
    source = Path(__file__).resolve()
    reader = source.with_name("read_monte_carlo_precision.R")
    metadata = dict(schema="h10-magnitude-precision-plan-v1", rule=RULE,
                    numericalGuards=dict(width=WIDTH_GUARD, mass=MASS_GUARD),
                    searchBound=bound, source=dict(path=str(source), sha256=
                    hashlib.sha256(source.read_bytes()).hexdigest()),
                    readerSource=dict(path=str(reader), sha256=
                    hashlib.sha256(reader.read_bytes()).hexdigest()),
                    runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__), jobID=os.environ.get("JOB_ID", ""),
                    newDraws=0, newFits=0, powerReductions=0, blockRounding=False)
    (args.output / "binding.json").write_text(json.dumps(metadata, indent=2) + "\n")
    selected = None
    with (args.output / "trace.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["B", "assurance", "minimumWidthMargin",
                                                   "totalMass", "passingCounts"])
        writer.writeheader()
        for n in range(1, bound + 1):
            row = evaluate(n)
            writer.writerow(row)
            if n % 25 == 0:
                stream.flush()
            if row["assurance"] >= RULE["assurance"]:
                selected = row
                break
    if selected is None:
        raise RuntimeError("NUMERIC_UNRESOLVED: finite search ended without selection")
    for path, descriptor in ((source, metadata["source"]), (reader, metadata["readerSource"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
            raise RuntimeError("Precision source changed during enumeration")
    result = dict(metadata, selected=selected, scannedIntegers=selected["B"],
                  traceSha256=hashlib.sha256((args.output / "trace.csv").read_bytes()).hexdigest(),
                  status="NUMERICAL_PLAN_COMPLETE_PENDING_INDEPENDENT_READBACK",
                  scientificAdoption=False, calibrationEstablished=False)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(selected=selected, scannedIntegers=selected["B"],
                          status=result["status"])))


if __name__ == "__main__":
    main()
