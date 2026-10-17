"""Independent count-based readback of native marginal calibration.

Clopper-Pearson bounds use a continued-fraction implementation independent of
the producer. Normal-reference power integrates over chi-square sampling error.
"""
from __future__ import annotations

import hashlib
import math
from .expected_outcomes import digest, finite, integer

FAMILY = {"Q": "Q", "R": "R_EGPA_pp", "L": "L_lower_mean_share", "initial": "initial_slope"}
OFFSETS = (0, 1, 17, 249)
RECORD_KEYS = {"schema", "bindingHash", "cellHash", "batchId", "seed", "attempts", "status", "targets", "scalarAudits", "contentHash"}
COUNT_KEYS = {"attempts", "available", "rejections", "coveredAvailableIntervals", "requiredArmCountsMinimum",
              "requiredArmCountsMaximum", "unexpectedNumericalFailures"}
AUDIT_KEYS = {"offset", "testId", "available", "status", "route"}

def require(value, label):
    if not value:
        raise ValueError(label)


def fields(value, names, label):
    require(type(value) is dict and set(value) == set(names), label)


def beta_cdf(x, a, b):
    """Regularized beta via a Lentz continued fraction, not SciPy tails."""
    require(finite(x) and finite(a) and finite(b) and a > 0 and b > 0, "Beta arguments")
    if x <= 0:
        return 0.
    if x >= 1:
        return 1.
    def fraction(aa, bb, xx):
        tiny = 1e-300
        c, d = 1., 1. - (aa + bb) * xx / (aa + 1.)
        d = 1. / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, 2001):
            for z in (m * (bb - m) * xx / ((aa + 2*m - 1) * (aa + 2*m)),
                      -(aa + m) * (aa + bb + m) * xx / ((aa + 2*m) * (aa + 2*m + 1))):
                d = 1. + z*d
                c = 1. + z/c
                d = d if abs(d) > tiny else tiny
                c = c if abs(c) > tiny else tiny
                d = 1. / d
                change = d*c
                h *= change
            if abs(change - 1.) < 4e-15:
                return h
        raise ValueError("Independent beta continued fraction did not converge")
    front = math.exp(math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b) + a*math.log(x) + b*math.log1p(-x))
    answer = front * fraction(a,b,x) / a if x < (a+1)/(a+b+2) else 1. - front * fraction(b,a,1-x) / b
    require(-2e-13 <= answer <= 1 + 2e-13, "Independent beta numeric range")
    return max(0., min(1., answer))


def beta_quantile(p, a, b):
    require(0 < p < 1, "Interior beta quantile")
    low, high = 0., 1.
    for _ in range(80):
        middle = (low + high) / 2
        if beta_cdf(middle, a, b) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def cp(successes, total):
    integer(total, "CP attempts")
    integer(successes, "CP successes", 0, total)
    if total == 0:
        return None
    # Preserve source alpha=(1-.95), including its binary64 association.
    alpha = 1. - .95
    return [0. if successes == 0 else beta_quantile(alpha/2, successes, total-successes+1),
            1. if successes == total else beta_quantile(1-alpha/2, successes+1, total-successes)]


def student_cdf(t, df):
    require(finite(t) and finite(df) and df > 0, "Student arguments")
    q = .5 * beta_cdf(df / (df + t*t), df/2, .5)
    return 1. - q if t >= 0 else q


def student_radius(df):
    low, high = 0., 1.
    while student_cdf(high, df) < .975:
        high *= 2
        require(high < 1e16, "Finite Student quantile bracket")
    for _ in range(75):
        middle = (low + high) / 2
        if student_cdf(middle, df) < .975:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def contrast(rows, spec, pooled=False):
    """Independent finite-person sufficient statistics; NaNs mean unavailable."""
    arms = list(spec["weights"])
    samples = [[x for x in rows[a] if math.isfinite(x)] for a in arms]
    counts = [len(a) for a in samples]
    means = [math.fsum(a)/len(a) if a else 0. for a in samples]
    variances = [math.fsum((x-m)**2 for x in a)/(len(a)-1) if len(a)>1 else 0. for a,m in zip(samples,means)]
    effect = math.fsum(spec["weights"][a]*m for a,m in zip(arms,means))
    if pooled:
        require(len(arms) == 2 and list(spec["weights"].values()) == [1,-1] and counts[0] == counts[1], "Separate balanced pooled reference")
        df = counts[0]+counts[1]-2
        se2 = (math.fsum((n-1)*v for n,v in zip(counts,variances))/df * math.fsum(1/n for n in counts)) if df > 0 and all(counts) else math.nan
    else:
        parts = [spec["weights"][a]**2*v/n if n else 0. for a,v,n in zip(arms,variances,counts)]
        se2 = math.fsum(parts)
        den = math.fsum(p*p/(n-1) for p,n in zip(parts,counts) if n>1)
        df = se2*se2/den if den>0 else math.nan
    se = math.sqrt(se2) if math.isfinite(se2) and se2>=0 else math.nan
    available = all(n>=2 for n in counts) and finite(se2) and se2>0 and finite(df) and df>0
    statistic = effect/se if available else math.nan
    p = math.nan
    if available:
        cdf = student_cdf(statistic,df)
        p = 1-cdf if spec["tail"] == "greater" else cdf if spec["tail"] == "less" else 2*min(cdf,1-cdf)
    radius = student_radius(df)*se if finite(df) and df>0 and finite(se) else math.nan
    available = available and finite(p) and finite(radius)
    unexpected = all(n>=2 for n in counts) and (not finite(effect) or not finite(se2) or
                 se2>0 and (not finite(df) or not finite(p) or not finite(radius)))
    return {"effect":effect,"SE":se,"df":df,"statistic":statistic,"p":p,"low":effect-radius,
            "high":effect+radius,"available":available,"reject":available and p<.05,
            "counts":counts,"unexpectedNumericalFailure":unexpected}


def seed(cell, batch):
    integer(batch,"Batch ID",0,39)
    subcase = "forward" if cell["id"] == "B12" else cell["subcase"]
    text = str(cell["seed"])+"|"+cell["id"]+"|"+subcase+"|batch:"+str(batch)
    return str(int.from_bytes(hashlib.sha256(text.encode()).digest()[:16],"big"))


def cells_for(analysis, calibration):
    require(type(calibration["attemptsPerSubcase"]) is int and calibration["attemptsPerSubcase"] == 10000
            and type(calibration["batchSize"]) is int and calibration["batchSize"] == 250
            and calibration["choicePrefixCycle"] == [0,1,3]
            and all(type(x) is int for x in calibration["choicePrefixCycle"]),"Frozen calibration shape")
    cells = []
    for base in calibration["cells"]:
        expected = 20261011100+int(base["id"][1:]) if base["layer"] == "A" else 20261011110+int(base["id"][1:]) if base["layer"] == "B" else 20261011130+int(base["id"][1:])
        require(type(base["seed"]) is int and base["seed"] == expected and base["layer"] in ("A","B","C"),"Frozen cell seed/layer")
        for subcase in base["subcases"]:
            cell = {**base,"subcase":subcase,"analysisSpec":analysis,"calibrationSpec":calibration}
            if base["id"] == "B07":
                cell["population"] = subcase if subcase.startswith("gap_") else "working"
            cells.append(cell)
    result={c["id"]+":"+c["subcase"]:c for c in cells}
    require(len(result)==len(cells),"No duplicate cell/subcase identity")
    return result


def specs_for(cell, analysis):
    if cell["layer"] == "A":
        return analysis["claims"][:2] if cell["id"] in ("A03","A04") or cell["subcase"] == "separate_pooled_reference" else analysis["claims"]
    if cell["id"] == "B07":
        return analysis["diagnostics"]
    family = cell.get("family",cell["subcase"])
    return [s for s in analysis["claims"] if s["endpoint"] == FAMILY[family]]


def arm_size(cell, arm):
    if cell["layer"] == "A":
        other = arm in ("C2","C4") and cell["id"] in ("A03","A04")
        if cell["subcase"] == "variance_count_mirror":
            other = not other
        n = cell.get("otherN",cell["n"]) if other else cell["n"]
    else:
        n = cell.get("armOverrides",{}).get(arm,{}).get("n",cell["n"])
    return integer(n,"Declared generated arm size",1)


def audit_rows(cell, specs, availability):
    route = "separate_pooled_reference" if cell["subcase"] == "separate_pooled_reference" else "selected_Student_or_Welch"
    return [{"offset":o,"testId":s["id"],"available":availability(o,s),"status":"PASS","route":route}
            for o in OFFSETS for s in specs]


def validate_record(record, cell, specs, truths, validation):
    fields(record,RECORD_KEYS,"Completed validation view")
    require(record["schema"] == "native-calibration-completed-batch-v1" and record["bindingHash"] == validation
            and record["cellHash"] == digest(cell) and record["status"] == "complete","Validation view identity")
    batch = integer(record["batchId"],"Strict completed batch ID",0,39)
    require(record["seed"] == seed(cell,batch) and record["attempts"] == 250 and type(record["attempts"]) is int,"Seed/250 attempts")
    require(record["contentHash"] == digest({k:v for k,v in record.items() if k!="contentHash"}),"Validation content hash")
    require(type(record["targets"]) is dict and set(record["targets"]) == {s["id"] for s in specs},"Exact target roster")
    for spec in specs:
        value, truth = record["targets"][spec["id"]], truths[spec["id"]]
        fields(value,COUNT_KEYS,"Per-target counts")
        require(type(value["attempts"]) is int and value["attempts"] == 250,"Per-target attempts")
        available = integer(value["available"],"Available",0,250)
        integer(value["rejections"],"Rejections",0,available)
        require(type(value["unexpectedNumericalFailures"]) is int and value["unexpectedNumericalFailures"] == 0,"No unexpected numerical failures")
        if truth["coverageEligible"]:
            integer(value["coveredAvailableIntervals"],"Covered available",0,available)
        else:
            require(value["coveredAvailableIntervals"] is None,"No unproved coverage count")
        sizes = [arm_size(cell,a) for a in spec["weights"]]
        lows, highs = value["requiredArmCountsMinimum"],value["requiredArmCountsMaximum"]
        require(type(lows) is list and type(highs) is list and len(lows)==len(highs)==len(sizes),"K-arm extrema length")
        for low,high,n in zip(lows,highs,sizes):
            integer(low,"Arm minimum",0,n); integer(high,"Arm maximum",low,n)
            require(not available or high>=2,"Positive availability arm coherence")
    audits = record["scalarAudits"]
    require(type(audits) is list,"Audit list")
    wanted = [(o,s["id"]) for o in OFFSETS for s in specs] if batch in (0,39) else []
    require(len(audits)==len(wanted),"Exact scalar audit roster length")
    route = "separate_pooled_reference" if cell["subcase"] == "separate_pooled_reference" else "selected_Student_or_Welch"
    for row,(offset,test_id) in zip(audits,wanted):
        fields(row,AUDIT_KEYS,"Scalar audit")
        require(row["offset"] == offset and type(row["offset"]) is int and row["testId"] == test_id
                and type(row["available"]) is bool and row["status"] == "PASS" and row["route"] == route,"Scalar audit identity")
        available = record["targets"][test_id]["available"]
        require(not (available==0 and row["available"]) and not (available==250 and not row["available"]),"Audit/count coherence")
    return record


def normal_reference_power(cell, spec):
    """Normal/chi-square conditioning, not the producer's noncentral-t CDF."""
    from scipy.integrate import quad
    n = cell["n"]
    one = len(spec["weights"]) == 1
    df = n - 1 if one else 2 * n - 2
    lo, hi = 0., 16.
    for _ in range(75):
        mid = (lo + hi) / 2
        cdf = 1 - .5 * beta_cdf(df / (df + mid * mid), df / 2, .5)
        if cdf < .95:
            lo = mid
        else:
            hi = mid
    critical, delta = (lo + hi) / 2, .3 * math.sqrt(n if one else n / 2)
    def integrand(log_ratio):
        v = df * math.exp(log_ratio)
        density_dlog = math.exp(df / 2 * math.log(v) - v / 2 - df / 2 * math.log(2) - math.lgamma(df / 2))
        return .5 * math.erfc((critical * math.sqrt(v / df) - delta) / math.sqrt(2)) * density_dlog
    value, error = quad(integrand, -40., 40., epsabs=2e-12, epsrel=2e-12, limit=300)
    if not math.isfinite(value) or error > 1e-9 or not 0 <= value <= 1:
        raise ValueError("Independent Normal/chi-square reference integration failed")
    return value


def summary(records,cell,specs,candidate,accepted,validation,truth_receipt_hash,analysis):
    truths = {t["testId"]:t for t in accepted["targets"]}
    candidates = {t["testId"]:t for t in candidate["targets"]}
    retained = {}
    for record in records:
        validate_record(record,cell,specs,truths,validation)
        require(record["batchId"] not in retained,"No duplicate completed fixture IDs")
        retained[record["batchId"]] = record
    total = len(retained)*250
    targets = []
    for spec in specs:
        truth,cand = truths[spec["id"]],candidates[spec["id"]]
        values = [r["targets"][spec["id"]] for r in retained.values()]
        n = sum(x["available"] for x in values); reject = sum(x["rejections"] for x in values)
        covered = sum(x["coveredAvailableIntervals"] for x in values) if truth["coverageEligible"] else None
        reject_ci, cover_ci = cp(reject,total),None if covered is None else cp(covered,n)
        failures = []
        if truth["typeIEligible"] and reject_ci is not None and reject_ci[1]>.06:
            failures.append("pointwise_null_rejection_CP_upper_above_.06")
        if truth["coverageEligible"] and cover_ci is not None and cover_ci[0]<.93:
            failures.append("pointwise_available_coverage_CP_lower_below_.93")
        reference = None
        if cell["id"] == "A05" and (len(spec["weights"]) == 1 or cell["subcase"] == "separate_pooled_reference"):
            reference = normal_reference_power(cell, spec)
            if total and abs(reject / total - reference) > .02:
                failures.append("identical_normal_reference_power_difference_above_.02")
        unresolved = truth["truthStatus"] == "numerically_unresolved"
        status = "DIAGNOSTIC_ONLY" if truth["truthStatus"] == "diagnostic_not_derived" else "UNAVAILABLE" if n==0 or unresolved or truth["classification"] == "exact_native_nonzero_not_boundary_null" else "MIXED" if failures else "PASS"
        ti = "UNAVAILABLE" if unresolved else "not_a_declared_null" if not truth["typeIEligible"] else "UNAVAILABLE" if n==0 else "MIXED" if reject_ci[1]>.06 else "PASS"
        co = "UNAVAILABLE" if unresolved else "not_derived" if covered is None else "UNAVAILABLE" if cover_ci is None else "MIXED" if cover_ci[0]<.93 else "PASS"
        discrepancy = None if truth["numericalValue"] is None or cand["selectedNativeFloatingTruth"] is None else -(truth["numericalValue"]-cand["selectedNativeFloatingTruth"])
        targets.append({"testId":spec["id"],"truth":truth,"producerCandidate":cand,
            "candidateMinusIndependentDisplayDifference":discrepancy,"producerCandidateStatus":"floating_integration_only_not_accepted_truth",
            "coverageComparison":"exact_rational_vs_Fraction_from_binary64_closed_CI","attempts":total,"available":n,"rejections":reject,
            "allAttemptRejectionRate":None if total==0 else reject/total,"availableConditionalRejectionRate":None if n==0 else reject/n,
            "rejectionCP95":reject_ci,"coveredAvailableIntervals":covered,"availableCoverageRate":None if covered is None or n==0 else covered/n,
            "availableCoverageCP95":cover_ci,"allAttemptIntervalSuccessRate":None if covered is None or total==0 else covered/total,
            "exactNormalReferencePower":reference,"status":status if total==10000 else "PROVISIONAL","failures":failures,
            "provisionalCriterionStatus":None if total==10000 else status,"criteriaFinal":total==10000,
            "typeICalibrationStatus":ti if total==10000 else "PROVISIONAL","coverageCalibrationStatus":co if total==10000 else "PROVISIONAL",
            "provisionalTypeICriterionStatus":None if total==10000 else ti,"provisionalCoverageCriterionStatus":None if total==10000 else co,
            "primary":spec["primary"]})
    return {"schema":"native-cell-calibration-v1","cellId":cell["id"],"subcase":cell["subcase"],"bindingHash":validation,
        "cellHash":digest(cell),"completedBatches":[retained[i] for i in sorted(retained)],"candidateCellHash":digest(candidate),
        "acceptedTruthCellHash":digest(accepted),"independentTruthReceiptHash":truth_receipt_hash,"plannedAttempts":10000,
        "completedAttempts":total,"complete":len(retained)==40,"targets":targets,"pointwiseOnly":True,
        "globalValidityClaim":None,"humanObservations":False,"historicalEvidence":"Earlier unmatched calibration/selection failures are not overturned."}
