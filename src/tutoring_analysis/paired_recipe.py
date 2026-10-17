"""Fixed input recipe and plain-file transport for paired Gamma diagnostics."""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform

BATCHES = (0, 39)
SIZE = 250
ORIGINAL_RUNTIME = {"python": "3.12.14", "numpy": "2.3.5", "scipy": "1.16.3"}
REFERENCE_RUNTIME = {"python": "3.10.9", "numpy": "1.23.5", "scipy": "1.10.0"}
SPEC = dict(id="H05_initial_C4", endpoint="initial_slope", weights={"C4": 1},
            tail="less", primary=True, null=0, alpha=.05, confidence=.95)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def decimal(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite JSON decimal")
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=decimal)


def selected_batches(batches):
    require(type(batches) in (tuple, list), "Explicit batch IDs required")
    require(bool(batches) and all(type(b) is int and b in BATCHES for b in batches),
            "Only original batch IDs 0 and 39 are supported")
    require(list(batches) == sorted(set(batches)), "Duplicate or unordered batch IDs")
    return tuple(batches)


def scope(batches):
    batches = selected_batches(batches)
    return dict(cell="gamma_skew_null", seed=2026100627, batches=list(batches),
                batchSize=SIZE, cases=SIZE * len(batches), people=20,
                condition="C4", window="primary", fullGeneratorDrawOrder=True,
                origin="historical_abstract_summary_audit")


def fixed_cell(config):
    from .calibration import resolved_cells

    config = read_json(config) if isinstance(config, (str, Path)) else config
    require(type(config) is dict and config.get("batchSize") == SIZE, "Fixed batch size changed")
    cells = [c for c in resolved_cells(config) if c["id"] == "gamma_skew_null"]
    require(len(cells) == 1, "Fixed Gamma cell absent or duplicated")
    expected = {
        "id": "gamma_skew_null", "seed": 2026100627, "replications": 10000,
        "generator": "gamma", "effects": "null", "dose": 10,
        "eligibility": "complete_ten",
        "people": {c: 20 for c in ("C1", "C2", "C3", "C4", "C5")},
        "covarianceKind": "exchangeable", "correlation": .3,
        "residualVariance": .01, "randomSlopeVariance": .01,
        "meanTrialSD": {"q": .2, "R": 3, "L": .1},
        "predictorKind": "default", "learning": 0, "orderTrend": 0,
        "participantRetention": 1, "trialRetention": 1, "endpointMissingProbability": 0,
        "commonEndpointComponent": 0,
        "expectedSupportRule": "observed_minimum_two_people_and_source_endpoint_slope_rank_rules; random_missing_support_is_anticipated",
        "nominalStatus": "stress",
    }
    require(json.dumps(cells[0], sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), "Frozen Gamma law changed")
    return cells[0]


def runtime():
    import numpy as np
    import scipy

    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def read_bytes(path, limit=16 * 1024 * 1024):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing/nonregular packet file: {path.name}")
    require(0 < path.stat().st_size <= limit, f"Invalid packet size: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit, f"Invalid packet size: {path.name}")
    return raw


def read_json(path):
    return strict_json(read_bytes(path))


def fresh_directory(path):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "Output must be a new directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def write_bytes(path, value):
    with Path(path).open("xb") as stream:
        stream.write(value)
