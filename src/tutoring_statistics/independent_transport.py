"""independent transport: accepted numerical definitions, portable source projection."""
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
from .independent_power import population_counts, IDS, METHOD

import numpy as np
import scipy
from scipy import stats, special, optimize
from scipy.stats import beta as beta_distribution, binom, nct, t as t_distribution
from scipy.optimize import brentq
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, stdtrit



SOURCE_TRANSPORT = {}

def require(value, message):
    if not value:
        raise ValueError(message)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def transported_source(request_pin, facts, near):
    key = request_pin["path"], request_pin["sha256"]
    require(key in SOURCE_TRANSPORT, "Missing pinned independently transported source")
    design = SOURCE_TRANSPORT[key]
    require(set(design) == {"X", "y", "participant", "condition"}
            and len(design["X"]) == len(design["y"]) == facts["rows"]
            and all(len(row) == 9 and all(finite(value) for value in row) for row in design["X"])
            and all(finite(value) for value in design["y"]), "Wrong transported source shape/values")
    for field in ("participant", "condition"):
        near(design[field], facts[field], "exact source labels/order", atol=0, rtol=0)
    return design

def independent_fit(raw, evidence, request_pin, rows, modules):
    """Authenticate actual variable-count RDS and the original model references."""
    math_kernel, accepted = modules["math"], modules["accepted"]
    facts = population_counts(rows)
    require(evidence.get("verified") is True and evidence["REML"] is True
            and evidence["formula"] == "y~condition*need_A+condition*need_B+(1|participant)"
            and evidence["requestDescriptor"] == request_pin
            and evidence["frameRowCount"] == evidence["requestRowCount"] == len(rows)
            and evidence["verifiedFields"] == ["participant", "condition", "need_A", "need_B", "y"],
            "Wrong actual RDS admission")
    near = accepted.near
    near(evidence["design"], transported_source(request_pin, facts, near),
         "actual retained model frame", atol=0, rtol=0)
    require(evidence["optimizer"] == raw["optimizer"] == "nloptwrap"
            and raw["REML"] is True and raw["method"] == "lmerTest_Satterthwaite_candidate"
            and "".join(raw["formula"].split()) == evidence["formula"], "Changed original model")
    near(raw["coefficientNames"], evidence["coefficientNames"], "original coefficient names")
    near([raw["coefficients"][key] for key in evidence["coefficientNames"]], evidence["coefficients"], "original beta")
    for raw_name, proof_name in (("vcov", "fixedVcov"), ("randomCovariance", "randomCovariance"),
                                 ("residualVariance", "residualVariance")):
        near(raw[raw_name], evidence[proof_name], "actual original " + raw_name)
    for key in ("converged", "singular", "gradient", "hessianEigenvalues", "usableCandidateInference"):
        near(raw["diagnostics"][key], evidence["diagnostics"][key], "original guard " + key)
    near(raw["diagnostics"]["convergenceMessages"], evidence["diagnostics"]["optimizerMessages"], "optimizer messages")
    L = math_kernel.contrasts(evidence["coefficientNames"])
    saved = evidence["originalSatterthwaite"]
    usable = raw["diagnostics"]["usableCandidateInference"]
    dfs = math_kernel.baseline_degrees_of_freedom(evidence["fixedVcov"], saved["covarianceJacobians"],
                saved["varianceParameterCovariance"], L) if usable else None
    for j, key in enumerate(IDS):
        ref, source = raw["references"][key], evidence["references"][key]
        near(source["L"], L[j].tolist(), "four original signed contrasts", atol=0, rtol=0)
        for field in ("L", "coefficientNames", "estimate", "SE", "df", "statistic"):
            near(ref[field], source[field], "saved original reference " + field)
        if dfs is not None:
            near(ref["df"], float(dfs[j]), "independent original Satterthwaite df")
        for name, source_name in (("CI", "packageConfidenceInterval"), ("twoSidedP", "packageTwoSidedP")):
            near(ref[name], source[source_name], "actual original " + name)
    return facts, L

def independent_candidate(raw, evidence, facts, L, modules):
    kernel, accepted = modules["math"], modules["accepted"]
    result = kernel.cr2_reconstruct(evidence["design"]["X"], evidence["design"]["y"],
        evidence["design"]["participant"], evidence["coefficients"], evidence["theta"],
        evidence["fixedVcov"], evidence["residualVariance"], L,
        expected_cluster_count=facts["participants"], expected_rows_per_cluster=10)
    refs = kernel.references(result, evidence["coefficientNames"], L)
    accepted.near(raw["vcov"], result["covariance"], "independent CR2 covariance")
    require(raw["method"] == METHOD, "Wrong candidate method")
    for key in IDS:
        for field in ("estimate", "SE", "df", "statistic", "CI", "L", "coefficientNames"):
            accepted.near(raw["references"][key][field], refs[key][field], "independent CR2 " + field)
        accepted.near(raw["references"][key]["packageTwoSidedP"], refs[key]["twoSidedP"], "independent CR2 p")
    expected_refs = {key: {**{field: value for field, value in ref.items() if field != "twoSidedP"},
                          "packageTwoSidedP": ref["twoSidedP"],
                          "reconstructedEstimate": ref["estimate"], "reconstructedSE": ref["SE"]}
                     for key, ref in refs.items()}
    return dict(raw, vcov=result["covariance"], references=expected_refs)
