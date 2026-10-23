"""Optional comparison-table projections; no fitted results are supplied."""

def rate_fields(prefix, item):
    if item is None:
        return {f"{prefix}_{key}": None for key in ("numerator", "denominator", "estimate", "lower95", "upper95")}
    interval = item.get("interval95", [None, None])
    return {f"{prefix}_numerator": item["numerator"],
            f"{prefix}_denominator": item["denominator"],
            f"{prefix}_estimate": item["estimate"],
            f"{prefix}_lower95": interval[0], f"{prefix}_upper95": interval[1]}

def null_rows(data):
    rows = []
    for item in data["cases"]:
        case = item["case"]
        for method in ("baseline", "candidate"):
            result = item[method]
            row = {key: case[key] for key in ("caseID", "caseIndex", "H", "condition", "nParticipants", "responseTransform", "intercept", "beta", "fixedTrialTrend", "sdIntercept", "sdTrial", "rho", "sigma", "gridStep")}
            row.update(method=method, planned=case["nReplicates"], available=result["available"], unavailable=result["unavailable"], rejected=result["rejected"], covered=result["covered"])
            for name, field in (("rejection_all", "rejectionAll"), ("rejection_available", "rejectionAmongAvailable"), ("coverage_available", "coverageAmongAvailable")):
                row.update(rate_fields(name, result["rates"][field]))
            rows.append(row)
    return rows

def coverage_rows(data, context):
    rows = []
    for method in ("baseline", "candidate"):
        for contrast, value in data[method]["coverage"].items():
            row = dict(context, method=method, contrast=contrast)
            row.update(rate_fields("coverage_all", value["coverageAll"]))
            row.update(rate_fields("coverage_available", value["coverageAvailable"]))
            rows.append(row)
    return rows

def family_rows(data, context):
    rows = []
    for method in ("baseline", "candidate"):
        for branch in data[method]["branches"]:
            row = dict(context, method=method, tail=branch["tailMode"], true_nulls=";".join(branch["trueNullIDs"]), fwer_applicability=branch["fwerApplicability"])
            row.update({f"count_{k}": branch["counts"][k] for k in ("M", "T", "A4", "R", "F", "D", "technicalMissing")})
            for name, field in (("any_rejection", "anyRejectionAll"), ("false_rejection", "falseRejectionAll"), ("correct_discovery", "alternativeRejectionAll")):
                row.update(rate_fields(name, branch["rates"][field]))
            rows.append(row)
    return rows

def power_context(case):
    cell = case["cell"]
    return {key: cell[key] for key in ("caseID", "cellIndex", "view", "scenarioIndex", "role", "multiplier", "populationRegime", "recruitedPerArm", "eligibilityProbability", "fixedEligiblePerArm")}
