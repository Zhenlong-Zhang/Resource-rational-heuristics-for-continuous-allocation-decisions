"""Independent participant whitening and full-design QR oracle check."""
import math
from .saved_fit_magnitude import NAMES

TARGETS = {'s1_rep1': ('C1-C2:B:D',),
           's5_l2_rep1': ('C1-C2:A:S', 'C1-C3:A:S')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def slope(arm, need):
    require(arm in ('C1', 'C2', 'C3') and need in ('A', 'B'), 'Wrong named slope')
    value = [0.0]*9
    value[3 if need == 'A' else 4] = 1.0
    if arm == 'C1':
        value[5 if need == 'A' else 7] = 1.0
    if arm == 'C3':
        value[6 if need == 'A' else 8] = 1.0
    return value


def reconstruct(label, actual_request, case):
    """Whiten each complete participant block and solve the original mean model."""
    import numpy as np
    from scipy.special import ndtri

    require(label in TARGETS and type(actual_request) is dict, 'Wrong fixed target')
    require(actual_request['model'] == 'H10' and actual_request['reference'] == 'C2', 'Wrong model/reference')
    parameters = case['parameters']
    require(finite(parameters['sdU']) and parameters['sdU'] >= 0, 'Wrong known intercept SD')
    require(set(parameters['sigmaByArm']) == {'C1', 'C2', 'C3'} and
            all(finite(value) and value > 0 for value in parameters['sigmaByArm'].values()), 'Wrong known residual SDs')
    require(parameters['sdU'] == 8.0 and all(value == 12.0 for value in parameters['sigmaByArm'].values()),
            'Changed the frozen known simulated covariance')
    rows = actual_request['rows']
    require(type(rows) is list and rows, 'Missing original input rows')
    groups = {}
    for row in rows:
        require(type(row) is dict and set(row) == {'participant', 'condition', 'need_A', 'need_B', 'y'},
                'Changed original input schema')
        person, arm = row['participant'], row['condition']
        require(type(person) is str and person and arm in ('C1', 'C2', 'C3') and
                all(finite(row[name]) for name in ('need_A', 'need_B', 'y')), 'Invalid actual input')
        groups.setdefault(person, []).append(row)
    counts = {arm: 0 for arm in ('C1', 'C2', 'C3')}
    wx, wy = [], []
    for person, own in groups.items():
        require(len(own) == 10 and len({row['condition'] for row in own}) == 1,
                'Each actual participant must have ten rows in one arm')
        arm = own[0]['condition']; counts[arm] += 1
        design = []
        for row in own:
            a, b = float(row['need_A']), float(row['need_B'])
            c1, c3 = float(arm == 'C1'), float(arm == 'C3')
            design.append([1., c1, c3, a, b, c1*a, c3*a, c1*b, c3*b])
        omega = parameters['sigmaByArm'][arm]**2*np.eye(10) + parameters['sdU']**2*np.ones((10, 10))
        chol = np.linalg.cholesky(omega)
        wx.append(np.linalg.solve(chol, np.asarray(design, dtype=float)))
        wy.append(np.linalg.solve(chol, np.asarray([row['y'] for row in own], dtype=float)))
    X, y = np.vstack(wx), np.concatenate(wy)
    require(np.isfinite(X).all() and np.isfinite(y).all(), 'Nonfinite whitened design')
    Q, R = np.linalg.qr(X, mode='reduced')
    require(R.shape == (9, 9) and np.linalg.matrix_rank(R) == 9, 'Singular known-Omega design')
    beta = np.linalg.solve(R, Q.T @ y)
    inverse = np.linalg.solve(R, np.eye(9))
    covariance = inverse @ inverse.T
    require(np.isfinite(beta).all() and np.isfinite(covariance).all(), 'Invalid independent GLS solution')
    critical = float(ndtri(.975)); references = {}
    for key in TARGETS[label]:
        base, kind = key.rsplit(':', 1); arm, need = base[3:5], base[-1]
        left, right = slope('C1', need), slope(arm, need)
        sign = 1. if kind == 'S' else -1.
        L = np.asarray([a+sign*b for a, b in zip(left, right)])
        truth = float(parameters['slopes']['C1'][need]+sign*parameters['slopes'][arm][need])
        estimate = float(L @ beta); variance = float(L @ covariance @ L)
        require(math.isfinite(variance) and variance > 0, 'Invalid conditional GLS variance')
        se = math.sqrt(variance); z = (estimate-truth)/se
        lower, upper = estimate-critical*se, estimate+critical*se
        # Numerical receiving tolerance is not a different confidence level.
        # At an endpoint, preserve uncertainty rather than force a coverage bit.
        endpoint_tolerance = max(2e-9, 2e-8*max(abs(truth), abs(lower), abs(upper)))
        close_to_endpoint = any(math.isclose(truth, endpoint, abs_tol=2e-9, rel_tol=2e-8)
                                for endpoint in (lower, upper))
        covered = (None if close_to_endpoint
                   else lower < truth < upper)
        references[key] = dict(L=L.tolist(), truth=truth, estimate=estimate, SE=se,
                               z=z, CI=[lower, upper], covered=covered,
                               coverageNumericallyUnresolved=close_to_endpoint,
                               endpointNumericalTolerance=endpoint_tolerance)
    return dict(coefficientNames=list(NAMES), beta=beta.tolist(), covariance=covariance.tolist(),
                references=references, actualK=counts, rowCount=len(rows), clusterCount=len(groups),
                algorithm='known_Omega_participant_Cholesky_full_design_QR',
                knownParameters=dict(randomInterceptSD=parameters['sdU'],
                    residualSDByArm=parameters['sigmaByArm'], rowsPerParticipant=10, normalCritical95=critical),
                source='saved_original_input', humanValidityEstablished=False)


def near(actual, expected, label):
    """Accept the previously frozen continuous scale, never an altered identity."""
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), label + ': fields')
        for key in expected:
            near(actual[key], expected[key], label + '.' + key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), label + ': length')
        for index, value in enumerate(expected):
            near(actual[index], value, label + '[' + str(index) + ']')
    elif type(expected) is float:
        require(finite(actual) and math.isclose(actual, expected, abs_tol=2e-9, rel_tol=2e-8), label)
    else:
        require(type(actual) is type(expected) and actual == expected, label)
