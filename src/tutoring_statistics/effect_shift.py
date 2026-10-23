"""effect shift: accepted numerical definitions, portable source projection."""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from copy import deepcopy
from statistics import NormalDist
import copy
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tempfile

import numpy as np
import scipy
from scipy import stats, special, optimize
from scipy.stats import beta as beta_distribution, binom, nct, t as t_distribution
from scipy.optimize import brentq
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, stdtrit

from . import fresh_reference as f
from . import joint_generation as m
require = f.require

PARITY_NAMESPACE = "h10-cr2-shift-parity-v1-20261019"

DERIVED_NAMESPACE = "h10-cr2-derived-power-v1-20261019"

METHOD = f.METHOD
PARITY_TOTAL, DERIVED_TOTAL = 32, 64000

SOURCE_POSITIONS = (0, 3, 7, 10)

MULTIPLIERS = (.5, 1., 2., 4.)

PARITY_MULTIPLIERS = (.5, 4.)

def ids(value, mode):
    maximum = PARITY_TOTAL if mode == "parity" else DERIVED_TOTAL
    require(isinstance(value, list) and value and all(type(x) is int and 0 <= x < maximum for x in value)
            and value == sorted(set(value)), "Invalid duplicate or unordered power IDs")
    return value

def identity(ordinal, mode):
    ids([ordinal], mode)
    multipliers = PARITY_MULTIPLIERS if mode == "parity" else MULTIPLIERS
    per_rep = 8 * len(multipliers)
    rep = ordinal // per_rep + 1
    view_index = (ordinal % per_rep) // (2*len(multipliers))
    direction = 5 + (ordinal // len(multipliers)) % 2
    multiplier = multipliers[ordinal % len(multipliers)]
    source_ordinal = (rep-1)*13 + SOURCE_POSITIONS[view_index]
    source = f.ordinal_case(source_ordinal)
    case = copy.deepcopy(m.measurement_plan()["cases"][6*view_index+direction-1])
    delta = {key: multiplier*value for key, value in case["delta"].items()}
    case_id = f"h10p_{source['view']}_s{direction}_l{int(multiplier*10):02d}_n154"
    case.update(caseID=case_id, repID=rep, delta=delta, trueNullIDs={t: m.null_ids(delta, t) for t in m.TAILS},
        designSeed=source["designSeed"], responseSeed=source["responseSeed"],
        dgpClass="conditional_linear_mean_shift_of_fresh_S1", futureProposedB=2000)
    for arm, offset in (("C2", 0), ("C3", 2)):
        for recipient, key in zip(("A", "B"), m.H10_IDS[offset:offset+2]):
            case["parameters"]["slopes"][arm][recipient] = source["parameters"]["slopes"]["C1"][recipient]-delta[key]
    return dict(ordinal=ordinal, sourceOrdinal=source_ordinal, sourceRepID=rep, sourceCaseID=source["caseID"],
        view=source["view"], direction=direction, multiplier=multiplier, case=case,
        caseID=case_id, repID=rep, fitKey=f"{case_id}__r{rep:04d}")

def shifted_request(actual, delta):
    require(actual.get("model") == "H10" and actual.get("reference") == "C2" and len(actual["rows"]) == 4620,
            "Changed source request")
    shifted = copy.deepcopy(actual)
    for old, row in zip(actual["rows"], shifted["rows"]):
        arm = row["condition"]
        require(arm in m.ARMS and set(row) == {"participant", "condition", "need_A", "need_B", "y"}, "Changed source row")
        da = 0. if arm == "C1" else -delta[f"C1-{arm}:A"]
        db = 0. if arm == "C1" else -delta[f"C1-{arm}:B"]
        row["y"] = old["y"] + da*old["need_A"] + db*old["need_B"]
        require(math.isfinite(row["y"]), "Nonfinite shifted response")
    return shifted
