"""Independent C5 law/design/range/floor arithmetic.

Normalization, NumPy exponentiation and binary64 addition define the same
implemented law on both sides. Cutoffs (binary search), unordered pairs,
cofactor inversion and prime-series logarithms are independently implemented.
"""
from fractions import Fraction as Q
import math

from . import c5_moments as P


def finite_law(logs, np, normalize):
    normalized = normalize(np.asarray(logs, dtype=float))
    probabilities = np.exp(normalized)
    P.need(probabilities.shape == (100,) and np.isfinite(probabilities).all() and
           np.all(probabilities >= 0) and abs(float(probabilities.sum()) - 1) <= 1e-12, 'Reference finite mass')
    total, previous, counts, cumulatives = 0., 0, [], []
    for j in range(100):
        total = float(np.add(np.float64(total), np.float64(probabilities[j])))
        cumulatives.append(total.hex())
        if j == 99:
            counts.append(P.GRID-previous)
            break
        n, d = total.as_integer_ratio()
        low, high = 0, P.GRID
        while low < high:
            middle = (low+high)//2
            if middle*d < n*P.GRID:
                low = middle+1
            else:
                high = middle
        P.need(low >= previous, 'Reference monotone CDF')
        # The two nearest actual grid points certify the strict boundary.
        P.need(low == 0 or (low-1)*d < n*P.GRID, 'Lower grid boundary')
        P.need(low == P.GRID or low*d >= n*P.GRID, 'Upper grid boundary')
        counts.append(low-previous)
        previous = low
    P.need(sum(counts) == P.GRID and min(counts) >= 0, 'Reference grid partition')
    return {'normalizedLogHex': [float(x).hex() for x in normalized],
            'probabilityHex': [float(x).hex() for x in probabilities],
            'cumulativeHex': cumulatives, 'rawMassHex': float(probabilities.sum()).hex(),
            'counts': counts, 'denominator': P.GRID, 'lastCategoryIncludesFallback': True}


def determinant(a):
    if len(a) == 1:
        return a[0][0]
    return sum((-1)**j*a[0][j]*determinant([row[:j]+row[j+1:] for row in a[1:]]) for j in range(len(a)))


def cofactor_inverse(a):
    d = determinant(a)
    if not d:
        return None, d
    size = len(a)
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            minor = [[a[x][y] for y in range(size) if y != i] for x in range(size) if x != j]
            row.append((-1)**(i+j)*determinant(minor)/d)
        result.append(row)
    return result, d


def unordered_root(root):
    a, b = [law['counts'] for law in root['laws']]
    budget = root['budget']
    total = {'gap': [[0]*2 for _ in range(2)], 'partial_low': [[0]*3 for _ in range(3)]}
    for lower in range(1, 101):
        for higher in range(lower+1, 101):
            weight = a[lower-1]*b[higher-1] + a[higher-1]*b[lower-1]
            if not weight:
                continue
            vectors = {'gap': [budget, 1000*(higher-lower)], 'partial_low': [budget, 1000*lower, 1000*higher]}
            for model, z in vectors.items():
                for i, x in enumerate(z):
                    for j, y in enumerate(z):
                        total[model][i][j] += weight*x*y
    return {model: [[Q(x, P.GRID**2*budget**2) for x in row] for row in a] for model, a in total.items()}


def range_reference(root, inv, model):
    a, b = [law['counts'] for law in root['laws']]
    budget = root['budget']
    # Evaluate each unordered supported point once, with an integer affine form.
    row = [inv[1][0]] + [x*Q(1000, budget) for x in inv[1][1:]]
    den = math.lcm(*(x.denominator for x in row))
    integer = [int(x*den) for x in row]
    values = []
    for lower in range(1, 101):
        for higher in range(lower+1, 101):
            if a[lower-1]*b[higher-1] + a[higher-1]*b[lower-1] == 0:
                continue
            z = [1, higher-lower] if model == 'gap' else [1, lower, higher]
            values.append(sum(x*y for x, y in zip(integer, z)))
    if not values:
        return {'weightMinimum': None, 'weightMaximum': None,
                'singlePrice': [P.rational(0), P.rational(0)], 'pairedDifference': [P.rational(0), P.rational(0)]}
    low, high = Q(min(values), den), Q(max(values), den)
    extreme = max(abs(low), abs(high))
    return dict(weightMinimum=P.rational(low), weightMaximum=P.rational(high),
                singlePrice=[P.rational(min(0, low)), P.rational(max(0, high))],
                pairedDifference=[P.rational(-extreme), P.rational(extreme)])


def design_reference(roots):
    moments = [unordered_root(root) for root in roots]
    output = {}
    for model, size in (('gap', 2), ('partial_low', 3)):
        matrix = [[sum(m[model][i][j] for m in moments)/10 for j in range(size)] for i in range(size)]
        inv, det = cofactor_inverse(matrix)
        value = {'G': P.matrix_json(matrix), 'determinant': P.rational(det),
                 'status': 'available' if inv is not None and det > 0 else 'singular',
                 'inverse': None if inv is None else P.matrix_json(inv), 'rootRanges': []}
        if value['status'] == 'available':
            value['rootRanges'] = [dict(rootKey=root['rootKey'], **range_reference(root, inv, model)) for root in roots]
            P.need(all(sum(matrix[i][k]*inv[k][j] for k in range(size)) == int(i == j)
                       for i in range(size) for j in range(size)), 'Reference inverse identity')
        output[model] = value
    return output


def log_reference(number):
    """Prime-factor logarithms, independent of producer binary range reduction."""
    def bounded_prime(p):
        z = Q(p-1, p+1)
        power, lower = z, Q(0)
        for i in range(256):
            lower += 2*power/(2*i+1)
            power *= z*z
        return lower, lower + 2*power/(513*(1-z*z))
    low, high, remaining = Q(0), Q(0), number
    for p in (2, 3, 5):
        a, b = bounded_prime(p)
        while remaining % p == 0:
            low, high, remaining = low+a, high+b, remaining//p
    P.need(remaining == 1, 'Fixed bound logarithm prime support')
    return low, high


def verify_summary(summary, designs, symmetry):
    rank = len(designs) == 9 and all(value[m]['status'] == 'available' for value in designs for m in ('gap', 'partial_low'))
    active = 2 if rank and symmetry else None
    P.need(summary['activeJ'] == active and summary['precisionBranch'] == ('analytic_high_zero_low_two' if active else 'unresolved'),
           'One prospective active branch')
    P.need(summary['reportQuantityCount'] == 6 and summary['policyDrawsReleased'] is False and
           summary['precisionEstablished'] is False, 'No inference/dispatch from preparation')
    if not rank:
        P.need(summary['ranges'] == [] and summary['highPriceSlopes'] is None
               and summary['pairedDifference'] is None and summary['floorFeasible'] is None, 'Unknown rank')
        return
    independent_log = log_reference(4*2*5*20) if active else None
    P.need(len(summary['ranges']) == 2, 'Exact two model range records')
    final_status = []
    for model, row in zip(('gap', 'partial_low'), summary['ranges']):
        P.need(row['model'] == model, 'Reference target order')
        ranges = [r for value in designs for r in value[model]['rootRanges']]
        low = sum(P.fraction(x['singlePrice'][0]) for x in ranges)/90
        high = sum(P.fraction(x['singlePrice'][1]) for x in ranges)/90
        difference = [sum(P.fraction(x['pairedDifference'][i]) for x in ranges)/90 for i in (0, 1)]
        P.need(row['singlePriceRange'] == [P.rational(low), P.rational(high)] and
               row['nominalPairedDifferenceRange'] == [P.rational(x) for x in difference], 'Independent signed full range')
        if active:
            P.need([r['replicates'] for r in row['zeroVarianceFloors']] == [16, 32, 64, 128, 256], 'Fixed looks')
            for floor in row['zeroVarianceFloors']:
                n = floor['replicates']
                lower = 7*(high-low)*independent_log[0]/(3*(n-1))
                upper = 7*(high-low)*independent_log[1]/(3*(n-1))
                published_low, published_high = P.fraction(floor['lower']), P.fraction(floor['upper'])
                P.need(published_low <= lower and published_high >= upper,
                       'Published enclosure must contain independent certified enclosure')
                expected = 'unattainable' if published_low > Q(1, 10) else (
                    'not_ruled_out' if published_high <= Q(1, 10) else 'numerically_unresolved')
                P.need(floor['status'] == expected, 'Strict zero-variance floor decision')
                if expected == 'unattainable':
                    P.need(lower > Q(1, 10), 'Independent floor not certified above target')
                if expected == 'not_ruled_out':
                    P.need(upper <= Q(1, 10), 'Independent floor not certified within target')
            final_status.append(row['zeroVarianceFloors'][-1]['status'])
        else:
            P.need(row['zeroVarianceFloors'] == [], 'Unknown symmetry is not J6')
    if not active:
        P.need(summary['highPriceSlopes'] is None and summary['pairedDifference'] is None
               and summary['floorFeasible'] is None, 'No analytic claim without symmetry')
    if active:
        feasible = (False if 'unattainable' in final_status else
                    True if all(s == 'not_ruled_out' for s in final_status) else None)
        P.need(summary['highPriceSlopes'] == [P.rational(0), P.rational(0)] and
               summary['pairedDifference'] == 'negative_low_price' and summary['floorFeasible'] is feasible,
               'Analytic zero/negation and floor scope')
