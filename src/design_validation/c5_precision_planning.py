"""Conditional fixed-look planning from saved exact C5 design geometry.

No draws, policy evaluations, finite-law enumeration or matrix inversion.
The fixed centered estimator is a prospective proposal, not achieved precision.
"""
from fractions import Fraction as F

MODELS = ('gap', 'partial_low')
EPSILON = F(1, 10)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def fraction(x):
    need(type(x) is list and len(x) == 2 and all(type(v) is str for v in x), 'Exact rational pair')
    result = F(int(x[0]), int(x[1]))
    need(str(result.numerator) == x[0] and str(result.denominator) == x[1], 'Canonical rational')
    return result


def rational(x):
    return [str(x.numerator), str(x.denominator)]


def log_bounds(integer, terms):
    """Rational enclosure from positive atanh series, no binary64 logarithm."""
    k = integer.bit_length()-1
    def atanh(z):
        total, power = F(0), z
        for j in range(terms):
            total += power/F(2*j+1)
            power *= z*z
        return 2*total, 2*total+2*power/((2*terms+1)*(1-z*z))
    lo2, hi2 = atanh(F(1, 3))
    t = F(integer, 2**k)
    lo, hi = atanh((t-1)/(t+1))
    return k*lo2+lo, k*hi2+hi


def ceiling(value):
    return -(-value.numerator//value.denominator)


def bounded_prefix_certificate(radii, variance, maximum):
    """Certify the fixed-look bound for one arbitrary bounded historical prefix."""
    need(len(radii) == 90 and all(x >= 0 for x in radii), 'Exactly 90 nonnegative radii')
    need(variance > 0 and maximum > 0 and max(radii) == maximum, 'Positive geometry')
    lower, upper = log_bounds(80, 96)
    total = sum(radii, F(0))

    def passes(n, log):
        if n < 1:
            return False
        error = EPSILON - total/(90*n)
        return error > 0 and 8100*n*error**2 >= 2*(variance+30*maximum*error)*log

    left, right = 0, 1
    while not passes(right, upper):
        left, right = right, 2*right
    while right-left > 1:
        middle = (left+right)//2
        if passes(middle, upper):
            right = middle
        else:
            left = middle
    need(not passes(right-1, lower), 'Adjacent integer unresolved')
    return dict(totalReplicates=right, futureCanonicalReplicates=right-1,
                currentPassesUpper=True, previousFailsLower=True,
                logBounds=[rational(lower), rational(upper)])


def integer_boundary(coefficient, log_argument, offset=0):
    need(coefficient > 0 and offset in (0, 1), 'Positive planning boundary')
    for terms in (48, 96, 192, 384, 768, 1536):
        low, high = log_bounds(log_argument, terms)
        a, b = offset+ceiling(coefficient*low), offset+ceiling(coefficient*high)
        if a == b:
            n = a
            need(F(n-offset) >= coefficient*high and F(n-1-offset) < coefficient*low,
                 'Adjacent integer certificate')
            # An independent integer-search route checks the adjacent boundary.
            left, right = offset+1, n+1
            while left < right:
                mid = (left+right)//2
                if F(mid-offset) >= coefficient*high:
                    right = mid
                else:
                    left = mid+1
            need(left == n, 'Integer search/formula disagreement')
            return dict(replicates=n, previousReplicates=n-1, coefficient=rational(coefficient),
                        logArgument=log_argument, logBounds=[rational(low), rational(high)],
                        seriesTerms=terms, integerOffset=offset, previousFailsLower=True,
                        currentPassesUpper=True)
    raise ArithmeticError('Integer boundary unresolved; no sample count released')


def calculations(designs, summary):
    """Use saved inverse diagonals and extrema, never sample or reconstruct G."""
    need(len(designs) == 9, 'Nine saved schedules')
    rows = []
    for model in MODELS:
        all_ranges, diagonal_sum = [], F(0)
        for v, design in enumerate(designs):
            d = design[model]
            need(d['status'] == 'available' and len(d['rootRanges']) == 10, 'All nine accepted full-rank designs')
            need([x['rootKey'] for x in d['rootRanges']] == [['C5', v, m] for m in range(1, 11)], 'Exact90 roots')
            diagonal_sum += fraction(d['inverse'][1][1])
            all_ranges.extend(d['rootRanges'])
        minima, maxima, radii = [], [], []
        for row in all_ranges:
            lo, hi = map(fraction, row['singlePrice'])
            need(lo <= 0 <= hi, 'Signed range contains zero')
            if row['weightMinimum'] is None:
                need(row['weightMaximum'] is None and lo == hi == 0, 'All-tie zero contribution')
                radius = F(0)
            else:
                low, high = fraction(row['weightMinimum']), fraction(row['weightMaximum'])
                need(low <= high and lo == min(F(0), low) and hi == max(F(0), high), 'Accepted weight range')
                radius = max(abs(low), abs(high))
            minima.append(lo)
            maxima.append(hi)
            radii.append(radius)
        original = next(x for x in summary['ranges'] if x['model'] == model)
        lo, hi = sum(minima)/90, sum(maxima)/90
        need(original['singlePriceRange'] == [rational(lo), rational(hi)], 'Saved whole-replicate range')
        need([x['replicates'] for x in original['zeroVarianceFloors']] == [16, 32, 64, 128, 256]
             and all(x['status'] == 'unattainable' for x in original['zeroVarianceFloors']), 'Old plan remains unattainable')
        width, sum_sq, b = hi-lo, sum(x*x for x in radii), max(radii)
        variance_sum = F(10, 4)*diagonal_sum
        need(width > 0 and sum_sq > 0 and variance_sum > 0, 'Nondegenerate planned target')
        eb = integer_boundary(7*width/(3*EPSILON), 800, offset=1)
        h = integer_boundary(sum_sq/(2*90**2*EPSILON**2), 80)
        bern = integer_boundary(2*(variance_sum+30*b*EPSILON)/(90**2*EPSILON**2), 80)
        rows.append(dict(model=model, oldWidth=rational(width), centeredRootRadii=[rational(x) for x in radii],
                         centeredSumSquaredRanges=rational(sum_sq), centeredMaxRange=rational(b),
                         centeredVarianceSumUpper=rational(variance_sum), originalEBNecessaryOnly=eb,
                         fixedLookHoeffdingSufficient=h, fixedLookBernsteinSufficient=bern))
    candidates = {name: max(row[name]['replicates'] for row in rows)
                  for name in ('fixedLookHoeffdingSufficient', 'fixedLookBernsteinSufficient')}
    chosen = min(candidates, key=lambda key: (candidates[key], key != 'fixedLookBernsteinSufficient'))
    n = candidates[chosen]
    return dict(rows=rows,
                originalEBNecessaryFullFamilyReplicates=max(row['originalEBNecessaryOnly']['replicates'] for row in rows),
                candidateFullFamilyReplicates=candidates, recommendedMethod=chosen,
                fixedReplicates=n, fixedLowPriceEndpoints=90*n, precision=['1', '10'],
                alpha=['1', '20'], activeTargets=list(MODELS), activeJ=2, inferenceLooks=1,
                estimator='new_fixed_control_variate_same_population_target_not_old_T_samplewise',
                requiredIndependence='root_times_rep_under_declared_PRNG_model',
                selectionBasis='accepted_design_only_min_sufficient_R_tie_Bernstein_before_any_outcome',
                selectionCreatesNoExtraOutcomeMultiplicity=True,
                withinPrior256ReplicateLimit=n <= 256,
                beyond256RequiresNewExplicitDomainRegistration=n > 256,
                policyReleaseAuthorized=False)
