"""Exact implemented finite-law C5 moments and original precision floor.

Ordered integer monomials and Gauss-Jordan inversion are the producer route.
The 53-bit strict-CDF grid is not the ideal exact-real categorical law.
No random draw, policy fit, or replacement precision method is implemented here.
"""
from fractions import Fraction as F
import math

GRID = 1 << 53
MODELS = ('gap', 'partial_low')
LOOKS = (16, 32, 64, 128, 256)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def rational(value):
    value = F(value)
    return [str(value.numerator), str(value.denominator)]


def fraction(value):
    need(type(value) is list and len(value) == 2 and all(type(x) is str for x in value), 'Exact rational encoding')
    answer = F(int(value[0]), int(value[1]))
    need(rational(answer) == value, 'Canonical positive-denominator rational')
    return answer


def matrix_json(matrix):
    return [[rational(x) for x in row] for row in matrix]


def finite_law(raw_logs, np, normalize):
    logs = normalize(np.asarray(raw_logs, dtype=float))  # Exact frozen expression; shared primitive disclosed.
    probabilities = np.exp(logs)
    need(probabilities.shape == (100,) and np.isfinite(probabilities).all() and
         np.all(probabilities >= 0) and abs(float(probabilities.sum()) - 1.) <= 1e-12, 'Frozen finite mass guard')
    cumulative, cuts, total = [], [], 0.
    for p in probabilities:
        total = float(np.add(np.float64(total), np.float64(p)))
        cumulative.append(total.hex())
        f = F.from_float(total) * GRID
        cuts.append(min(GRID, max(0, -(-f.numerator // f.denominator))))
    counts, prior = [], 0
    for cutoff in cuts[:-1]:
        need(cutoff >= prior, 'Nonmonotone CDF')
        counts.append(cutoff - prior)
        prior = cutoff
    counts.append(GRID - prior)  # Includes strict-CDF category100 and final fallback.
    need(sum(counts) == GRID and min(counts) >= 0, 'Complete 53-bit category partition')
    return {'normalizedLogHex': [float(x).hex() for x in logs],
            'probabilityHex': [float(x).hex() for x in probabilities],
            'cumulativeHex': cumulative, 'rawMassHex': float(probabilities.sum()).hex(),
            'counts': counts, 'denominator': GRID, 'lastCategoryIncludesFallback': True}


def root_moments(a, b, budget):
    """Integer monomial accumulation, then exact normalization only once."""
    need(type(budget) is int and budget > 0, 'Positive integer original budget')
    validate_counts(a)
    validate_counts(b)
    terms = [0] * 8  # 1,L,H,L^2,LH,H^2,gap,gap^2 in integer categories.
    for i, ai in enumerate(a, 1):
        if not ai:
            continue
        for j, bj in enumerate(b, 1):
            if not bj or i == j:
                continue
            mass, low, high = ai * bj, min(i, j), max(i, j)
            vals = (1, low, high, low*low, low*high, high*high, high-low, (high-low)**2)
            for k, value in enumerate(vals):
                terms[k] += mass * value
    d, scale = GRID * GRID, F(1000, budget)
    m = [F(x, d) * scale**degree for x, degree in zip(terms, (0, 1, 1, 2, 2, 2, 1, 2))]
    return {'gap': [[m[0], m[6]], [m[6], m[7]]],
            'partial_low': [[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]]}


def inverse(matrix):
    n = len(matrix)
    work = [[F(x) for x in row] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    determinant, sign = F(1), 1
    for col in range(n):
        found = next((i for i in range(col, n) if work[i][col]), None)
        if found is None:
            return None, F(0)
        if found != col:
            work[col], work[found] = work[found], work[col]
            sign *= -1
        pivot = work[col][col]
        determinant *= pivot
        work[col] = [x / pivot for x in work[col]]
        for i in range(n):
            if i != col:
                multiple = work[i][col]
                work[i] = [x - multiple*y for x, y in zip(work[i], work[col])]
    result = [row[n:] for row in work]
    need(all(sum(matrix[i][k]*result[k][j] for k in range(n)) == int(i == j)
             for i in range(n) for j in range(n)), 'Exact inverse identity')
    return result, determinant * sign


def root_range(a, b, budget, inv, model):
    weights = [inv[1][0], inv[1][1]*F(1000, budget)]
    if model == 'partial_low':
        weights.append(inv[1][2]*F(1000, budget))
    denominator = math.lcm(*(x.denominator for x in weights))
    coefficients = [int(x*denominator) for x in weights]
    minimum, maximum = None, None
    for i, ai in enumerate(a, 1):
        if not ai:
            continue
        for j, bj in enumerate(b, 1):
            if not bj or i == j:
                continue
            low, high = min(i, j), max(i, j)
            numerator = coefficients[0] + (coefficients[1]*(high-low) if model == 'gap' else
                         coefficients[1]*low + coefficients[2]*high)
            minimum = numerator if minimum is None else min(minimum, numerator)
            maximum = numerator if maximum is None else max(maximum, numerator)
    if minimum is None:
        return {'weightMinimum': None, 'weightMaximum': None,
                'singlePrice': [rational(0), rational(0)],
                'pairedDifference': [rational(0), rational(0)]}
    lo, hi = F(minimum, denominator), F(maximum, denominator)
    radius = max(abs(lo), abs(hi))
    return {'weightMinimum': rational(lo), 'weightMaximum': rational(hi),
            'singlePrice': [rational(min(F(0), lo)), rational(max(F(0), hi))],
            'pairedDifference': [rational(-radius), rational(radius)]}


def design_payload(root_records):
    need(len(root_records) == 10 and len({tuple(r['rootKey']) for r in root_records}) == 10,
         'Exactly ten distinct roots per design')
    result = {}
    moments = [root_moments(r['laws'][0]['counts'], r['laws'][1]['counts'], r['budget'])
               for r in root_records]
    for model, n in (('gap', 2), ('partial_low', 3)):
        gram = [[sum(moment[model][i][j] for moment in moments) / 10
                 for j in range(n)] for i in range(n)]
        inv, determinant = inverse(gram)
        result[model] = {'G': matrix_json(gram), 'determinant': rational(determinant),
                         'status': 'available' if inv is not None and determinant > 0 else 'singular',
                         'inverse': None if inv is None else matrix_json(inv), 'rootRanges': []}
        if result[model]['status'] == 'available':
            result[model]['rootRanges'] = [dict(rootKey=r['rootKey'], **root_range(
                r['laws'][0]['counts'], r['laws'][1]['counts'], r['budget'], inv, model)) for r in root_records]
    return result


def log_interval(integer):
    """Certified rational log: ln(n)=k ln2+2 atanh((n/2^k-1)/(n/2^k+1))."""
    need(type(integer) is int and integer > 1, 'Positive logarithm argument')
    k = integer.bit_length() - 1
    def series(z):
        terms, power = F(0), z
        for j in range(48):
            terms += power / (2*j+1)
            power *= z*z
        tail = 2*power / (97*(1-z*z))
        return 2*terms, 2*terms+tail
    a, b = series(F(1, 3))
    t = F(integer, 2**k)
    c, d = series((t-1)/(t+1))
    return k*a+c, k*b+d


def summarize_design(designs, symmetry):
    fullrank = len(designs) == 9 and all(d['payload'][m]['status'] == 'available' for d in designs for m in ('gap', 'partial_low'))
    active = 2 if symmetry and fullrank else None
    rows = []
    for model in ('gap', 'partial_low'):
        if not fullrank:
            continue
        all_ranges = [r for d in designs for r in d['payload'][model]['rootRanges']]
        lo = sum(fraction(x['singlePrice'][0]) for x in all_ranges) / 90
        hi = sum(fraction(x['singlePrice'][1]) for x in all_ranges) / 90
        pairlo = sum(fraction(x['pairedDifference'][0]) for x in all_ranges) / 90
        pairhi = sum(fraction(x['pairedDifference'][1]) for x in all_ranges) / 90
        floor = []
        if active:
            loglo, loghi = log_interval(4*active*5*20)
            for count in (16, 32, 64, 128, 256):
                low = 7*(hi-lo)*loglo/(3*(count-1))
                high = 7*(hi-lo)*loghi/(3*(count-1))
                status = 'unattainable' if low > F(1, 10) else ('not_ruled_out' if high <= F(1, 10) else 'numerically_unresolved')
                floor.append({'replicates': count, 'lower': rational(low), 'upper': rational(high), 'status': status})
        rows.append({'model': model, 'singlePriceRange': [rational(lo), rational(hi)],
                     'nominalPairedDifferenceRange': [rational(pairlo), rational(pairhi)], 'zeroVarianceFloors': floor})
    last = [r['zeroVarianceFloors'][-1]['status'] for r in rows] if active else []
    feasible = (False if 'unattainable' in last else
                True if last and all(x == 'not_ruled_out' for x in last) else None)
    return {'activeJ': active, 'precisionBranch': 'analytic_high_zero_low_two' if active else 'unresolved',
            'reportQuantityCount': 6, 'highPriceSlopes': [rational(0), rational(0)] if active else None,
            'pairedDifference': 'negative_low_price' if active else None, 'ranges': rows,
            'policyDrawsReleased': False, 'precisionEstablished': False,
            'floorFeasible': feasible}


def validate_counts(counts):
    need(type(counts) is list and len(counts) == 100
         and all(type(x) is int and x >= 0 for x in counts)
         and sum(counts) == GRID, 'Complete nonnegative integer 53-bit law required')
