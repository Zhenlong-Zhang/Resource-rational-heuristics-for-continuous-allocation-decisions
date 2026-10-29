"""Known simulated covariance GLS diagnostic, not a human-data estimator."""
import math
from statistics import NormalDist
from .saved_fit_magnitude import NAMES, ARMS, require, finite

REFERENCES = {'s1_rep1': {'C1-C2:B:D': ([0,0,0,0,0,0,0,1,0], 0.)},
              's5_l2_rep1': {'C1-C2:A:S': ([0,0,0,2,0,1,0,0,0], .8),
                             'C1-C3:A:S': ([0,0,0,2,0,1,1,0,0], .8)}}


def known_gls(label, actual, case):
    import numpy as np
    require(label in REFERENCES and type(actual) is dict and actual.get('model') == 'H10'
            and actual.get('reference') == 'C2', 'Wrong fixed target/model')
    require(case['parameters']['sdU'] == 8 and case['parameters']['sigmaByArm'] == {a:12. for a in ARMS},
            'Changed known DGP covariance')
    rows = actual['rows']; people = {}
    require(type(rows) is list and rows, 'Missing original input rows')
    for row in rows:
        require(type(row) is dict and set(row) == {'participant','condition','need_A','need_B','y'}
                and type(row['participant']) is str and row['participant'] and row['condition'] in ARMS
                and all(finite(row[k]) for k in ('need_A','need_B','y')), 'Invalid actual source row')
        a,b = row['need_A'],row['need_B']; c1=float(row['condition']=='C1'); c3=float(row['condition']=='C3')
        people.setdefault(row['participant'], []).append(([1.,c1,c3,a,b,c1*a,c3*a,c1*b,c3*b],row['y'],row['condition']))
    info = np.zeros((9,9)); score = np.zeros(9)
    sigma2 = case['parameters']['sigmaByArm']['C1']**2; tau2 = case['parameters']['sdU']**2
    for group in people.values():
        require(len(group) == 10 and len({r[2] for r in group}) == 1, 'Invalid participant block')
        x = np.asarray([r[0] for r in group], dtype=float); y = np.asarray([r[1] for r in group], dtype=float)
        shrink = tau2/(sigma2+len(group)*tau2)
        wx = (x-shrink*x.sum(axis=0))/sigma2; wy = (y-shrink*y.sum())/sigma2
        info += x.T@wx; score += x.T@wy
    require(np.isfinite(info).all() and np.isfinite(score).all(), 'Nonfinite information')
    # Failure is unresolved; no pseudo-inverse or dropped-column fallback.
    require(np.linalg.matrix_rank(info) == 9, 'Singular known-Omega design')
    try:
        np.linalg.cholesky(info)
        beta = np.linalg.solve(info,score); covariance = np.linalg.solve(info,np.eye(9))
    except np.linalg.LinAlgError as exc:
        raise ValueError('Unresolved known-Omega information factorization') from exc
    require(np.isfinite(beta).all() and np.isfinite(covariance).all(), 'Nonfinite known-Omega GLS')
    critical = NormalDist().inv_cdf(.975); refs = {}
    for key,(vector,truth) in REFERENCES[label].items():
        L = np.asarray(vector,dtype=float); estimate = float(L@beta); variance = float(L@covariance@L)
        require(variance > 0 and finite(variance), 'Unresolved known-Omega reference variance')
        se = math.sqrt(variance); ci = [estimate-critical*se,estimate+critical*se]
        ambiguous = any(math.isclose(truth,end,abs_tol=2e-9,rel_tol=2e-8) for end in ci)
        refs[key] = dict(L=vector,truth=truth,estimate=estimate,SE=se,z=(estimate-truth)/se,CI=ci,
                         covered=None if ambiguous else ci[0] <= truth <= ci[1], coverageNumericallyUnresolved=ambiguous)
    return dict(coefficientNames=list(NAMES),beta=beta.tolist(),covariance=covariance.tolist(),references=refs,
                actualK={a:sum(v[0][2]==a for v in people.values()) for a in ARMS},rowCount=len(rows),clusterCount=len(people),
                knownParameters=dict(randomInterceptSD=8.,residualSDByArm={a:12. for a in ARMS},
                                     rowsPerParticipant=10,normalCritical95=critical))


def comparisons(oracle, old_methods):
    result = {}
    for method, old in old_methods.items():
        result[method] = {}
        for key, new in oracle['references'].items():
            previous = old['references'][key]
            result[method][key] = dict(estimateDifference=new['estimate']-previous['estimate'],
                SEratio=new['SE']/previous['SE'], oldStandardizedError=(previous['estimate']-new['truth'])/previous['SE'],
                knownStandardizedError=new['z'])
    return result
