"""power origins: accepted numerical definitions, portable source projection."""
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



NAMESPACE = "h10-power-design-v1-20261020"

ORIGINAL_NAMESPACE = "h10-gaussian-reference-v1-20261019"

SHIFT_NAMESPACE = "h10-cr2-shift-parity-v1-20261019"

METHOD = "participant_CR2_Satterthwaite_v1"

SCHEMA = "h10-power-legacy-slot-v1"

CONTRASTS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")

class LegacyAdmissionError(ValueError):
    """An incompatible or incomplete alias never grants permission to refit."""

def _require(value, message):
    if not value:
        raise LegacyAdmissionError(message)

def planned_origin(cell_id, rep_id):
    """Mechanical old-slot mapping, independent of B and every numerical result.

    The caller separately checks rep_id <= its source-bound selected B. ``None``
    means a genuinely unmapped slot, not permission to replace a failed alias.
    """
    _require(type(rep_id) is int and 1 <= rep_id <= 2000, "Invalid legacy target rep")
    match = re.fullmatch(
        r"h10pd1_(initial_amount|terminal_amount)_s([1256])_l(05|10|20|40)_"
        r"(fixed154|recruit193_mcar08|fixed193)", cell_id)
    _require(match is not None, "Invalid finite power cell ID")
    view, scenario_text, multiplier_text, population = match.groups()
    scenario = int(scenario_text)
    if population != "fixed154" or scenario not in (5, 6):
        return None
    case_index = (6 if view == "initial_amount" else 18) + scenario
    if multiplier_text == "10" and rep_id <= 1000:
        case_id = f"h10m1_{view}_s{scenario}_n154"
        return dict(kind="original_production", namespace=ORIGINAL_NAMESPACE,
                    caseID=case_id, caseIndex=case_index, repID=rep_id,
                    fitKey=case_id if rep_id == 1 else f"{case_id}__r{rep_id:04d}",
                    recordOrigin="accepted_measurement" if rep_id == 1 else "production",
                    designSeed=202610190 + 10000 * scenario + rep_id,
                    responseSeed=302610190 + 10000 * case_index + rep_id)
    if multiplier_text in ("05", "40") and rep_id <= 2:
        view_index = 1 if view == "initial_amount" else 3
        ordinal = (rep_id - 1) * 16 + view_index * 4 + (scenario - 5) * 2
        ordinal += int(multiplier_text == "40")
        source_case_index = 7 if view == "initial_amount" else 19
        case_id = f"h10p_{view}_s{scenario}_l{multiplier_text}_n154"
        return dict(kind="actual_shift", namespace=SHIFT_NAMESPACE, ordinal=ordinal,
                    caseID=case_id, caseIndex=case_index, repID=rep_id,
                    fitKey=f"{case_id}__r{rep_id:04d}",
                    sourceOrdinal=(rep_id - 1) * 13 + (3 if view_index == 1 else 10),
                    designSeed=402620190 + rep_id,
                    responseSeed=502610190 + 10000 * source_case_index + rep_id)
    return None
