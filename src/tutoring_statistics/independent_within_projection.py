"""Independent Helmert/QR calculation for within-participant H10 slopes.

This numerical reference is deliberately distinct from the centered/SVD
producer. It estimates the same six slopes only under the common within/between
slope model and makes the stated independent Gaussian homoskedastic assumptions.
It never reads old fitted objects, regenerates a row or invokes the producer.
"""
from collections import OrderedDict
import math

ARMS = ('C1', 'C2', 'C3')
SLOPES = ('C1:A', 'C1:B', 'C2:A', 'C2:B', 'C3:A', 'C3:B')
COMPARISONS = ('C1-C2:A', 'C1-C2:B', 'C1-C3:A', 'C1-C3:B')


def require(value, message):
    if not value:
        raise ValueError(message)


def calculate(rows):
    import numpy as np
    from scipy.linalg import qr, solve_triangular
    from scipy.stats import t

    require(type(rows) is list and bool(rows), 'No observed rows')
    subjects = OrderedDict()
    for row in rows:
        require(type(row) is dict and row.get('condition') in ARMS
                and isinstance(row.get('participant'), str) and row['participant'], 'Invalid participant/condition')
        values = (row.get('need_A'), row.get('need_B'), row.get('y'))
        require(all(type(value) in (int, float) and math.isfinite(value) for value in values), 'Invalid numeric row')
        record = subjects.setdefault(row['participant'], {'arm': row['condition'], 'rows': []})
        require(record['arm'] == row['condition'], 'A participant crosses conditions')
        record['rows'].append(values)
    groups, counts = [], {arm: 0 for arm in ARMS}
    for subject in subjects.values():
        values = np.asarray(subject['rows'], dtype=float)
        n = len(values)
        require(n == 10, 'Frozen scope requires exactly ten complete rows per participant')
        counts[subject['arm']] += 1
        block = np.zeros((n, 6), dtype=float)
        j = 2 * ARMS.index(subject['arm'])
        block[:, j:j+2] = values[:, :2]
        q = np.zeros((n-1, n), dtype=float)
        for k in range(1, n):
            denominator = math.sqrt(k * (k + 1))
            q[k-1, :k] = 1.0 / denominator
            q[k-1, k] = -k / denominator
        groups.append((q @ block, q @ values[:, 2]))
    if not all(counts[a] > 0 for a in ARMS):
        return dict(status='statistical_unavailable', reason='within_design_missing_arm',
                    participants=len(subjects), counts=counts)
    matrix = np.vstack([item[0] for item in groups])
    outcome = np.concatenate([item[1] for item in groups])
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale == 0):
        return dict(status='statistical_unavailable', reason='within_design_constant_slope_column',
                    participants=len(subjects), transformedRows=len(outcome), counts=counts)
    scaled = matrix / scale
    singular = np.linalg.svd(scaled, compute_uv=False)
    cutoff = max(scaled.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    if any(.5 * cutoff < value < 2. * cutoff for value in singular):
        return dict(status='technical_unresolved', reason='within_rank_boundary_unresolved', rank=rank,
                    participants=len(subjects), transformedRows=len(outcome), counts=counts)
    if rank != 6:
        return dict(status='statistical_unavailable', reason='within_design_rank_deficient', rank=rank,
                    participants=len(subjects), transformedRows=len(outcome), counts=counts)
    df = len(outcome) - rank
    require(df > 0, 'No residual degrees of freedom')
    q, r, pivot = qr(scaled, mode='economic', pivoting=True)
    permuted_coef = solve_triangular(r, q.T @ outcome)
    scaled_coef = np.empty(6, dtype=float)
    scaled_coef[pivot] = permuted_coef
    beta = scaled_coef / scale
    residual = outcome - matrix @ beta
    rss = float(residual @ residual)
    require(math.isfinite(rss) and rss > 0, 'Residual variance is unavailable')
    inverse_r = solve_triangular(r, np.eye(6))
    cov_permuted = inverse_r @ inverse_r.T
    cov_scaled = np.empty((6, 6), dtype=float)
    cov_scaled[np.ix_(pivot, pivot)] = cov_permuted
    covariance = (rss / df) * cov_scaled / np.outer(scale, scale)
    require(np.all(np.isfinite(beta)) and np.all(np.isfinite(covariance)), 'Nonfinite estimate')
    refs = {}
    for comparison in COMPARISONS:
        difference, need = comparison.split(':')
        left, right = difference.split('-')
        i, j = SLOPES.index(left + ':' + need), SLOPES.index(right + ':' + need)
        for mode, sign in (('D', -1), ('S', 1)):
            vector = np.zeros(6, dtype=float)
            vector[i], vector[j] = 1.0, float(sign)
            estimate = float(vector @ beta)
            variance = float(vector @ covariance @ vector)
            require(math.isfinite(variance) and variance > 0, 'Contrast variance unavailable')
            se = math.sqrt(variance)
            statistic = estimate / se
            critical = float(t.ppf(.975, df))
            less, greater = float(t.cdf(statistic, df)), float(t.sf(statistic, df))
            interval = [estimate-critical*se, estimate+critical*se]
            require(all(math.isfinite(value) for value in (estimate, se, statistic, critical, less, greater, *interval))
                    and 0 <= less <= 1 and 0 <= greater <= 1 and interval[0] < interval[1],
                    'Invalid independent t reference')
            refs[comparison + ':' + mode] = dict(L=vector.tolist(), estimate=estimate, SE=se, df=df,
                statistic=statistic, lessP=less, greaterP=greater, CI=interval)
    return dict(status='available', reason=None, slopeNames=list(SLOPES), beta=beta.tolist(),
                covariance=covariance.tolist(), rss=rss, df=df, rank=rank,
                participants=len(subjects), transformedRows=len(outcome), counts=counts, references=refs,
                assumption='independent Gaussian equal-variance idiosyncratic errors; no outcome-dependent selection')
