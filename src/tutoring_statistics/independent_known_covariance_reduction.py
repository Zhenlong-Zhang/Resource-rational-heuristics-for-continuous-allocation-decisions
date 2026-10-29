"""Descriptive saved-input oracle coverage, paired transitions and hybrids."""
import math
import statistics
from .saved_fit_magnitude import NAMES as _NAMES, require, finite, exact, near

NAMES = list(_NAMES)
M = 2000
KEY = TARGET = "C1-C2:B:D"
L = [0, 0, 0, 0, 0, 0, 0, 1, 0]


def rep_ids(values):
    require(type(values) is list and values and all(type(v) is int and 1 <= v <= M for v in values)
            and values == sorted(set(values)), "Invalid saved repIDs")
    return values


def interval_event(estimate, endpoints):
    require(finite(estimate) and type(endpoints) is list and len(endpoints) == 2
            and all(finite(v) for v in endpoints) and endpoints[0] < endpoints[1], 'Invalid diagnostic interval')
    ambiguous = any(math.isclose(0., x, abs_tol=2e-9, rel_tol=2e-8) for x in endpoints)
    return dict(CI=endpoints, covered=None if ambiguous else endpoints[0] < 0 < endpoints[1],
        belowTruth=None if ambiguous else endpoints[1] < 0,
        aboveTruth=None if ambiguous else endpoints[0] > 0,
        coverageNumericallyUnresolved=ambiguous)


def compact_event(rep, oracle, scalar):
    require(scalar['repID'] == rep, 'Wrong admitted original repID')
    exact(oracle['coefficientNames'], NAMES, 'nine named coefficients')
    require(set(oracle['references']) == {TARGET}, 'Changed target contrast')
    reference = oracle['references'][TARGET]
    require(reference['L'] == L and all(type(v) in (int, float) for v in reference['L'])
            and type(reference['truth']) is float and reference['truth'] == 0., 'Changed exact L/truth')
    exact(oracle['actualK'], scalar['populationCounts'], 'original actual K')
    require(oracle['rowCount'] == 10*sum(oracle['actualK'].values())
            and oracle['clusterCount'] == sum(oracle['actualK'].values()), 'Changed row/participant count')
    require(all(finite(reference[k]) for k in ('estimate', 'SE', 'z')) and reference['SE'] > 0,
            'Nonfinite known-Omega reference')
    event = interval_event(reference['estimate'], reference['CI'])
    exact(event['covered'], reference['covered'], 'oracle coverage')
    exact(event['coverageNumericallyUnresolved'], reference['coverageNumericallyUnresolved'], 'oracle ambiguity')
    near(reference['z'], reference['estimate']/reference['SE'], 'oracle standardized error')
    comparisons = {}
    hybrids = {}
    for method in ('baseline', 'candidate'):
        old = scalar[method]
        comparisons[method] = dict(estimateDifference=reference['estimate']-old['estimate'],
            SEratio=reference['SE']/old['SE'], oldStandardizedError=old['z'],
            knownStandardizedError=reference['z'])
    widths = {method: (scalar[method]['estimate'], [scalar[method]['lower'], scalar[method]['upper']])
              for method in ('baseline', 'candidate')}
    widths['oracle'] = (reference['estimate'], reference['CI'])
    for center_name, center in (('old', scalar['baseline']['estimate']), ('oracle', reference['estimate'])):
        for width_name, (original_center, bounds) in widths.items():
            ci = [center+(bound-original_center) for bound in bounds]
            hybrids[center_name+'__'+width_name] = interval_event(center, ci)
    return dict(repID=rep, ordinal=40*(rep-1)+32, actualK=dict(oracle['actualK']),
        oracle=dict(estimate=reference['estimate'], SE=reference['SE'], z=reference['z'], **event),
        oldMethods={method: dict(scalar[method]) for method in ('baseline', 'candidate')},
        comparisons=comparisons, hybrids=hybrids)


def moments(values):
    require(values and all(finite(v) for v in values), 'Invalid descriptive scalar set')
    n = len(values); mean = math.fsum(values)/n
    return dict(n=n, mean=mean, sampleSD=(math.sqrt(math.fsum((x-mean)**2 for x in values)/(n-1))
                if n > 1 else None), min=min(values), max=max(values))


def proportion(k, n):
    from scipy.stats import beta
    require(type(k) is int and type(n) is int and 0 <= k <= n, 'Invalid count denominator')
    if n == 0:
        return dict(numerator=k, denominator=n, value=None, interval95=None)
    return dict(numerator=k, denominator=n, value=k/n,
        interval95=[0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                    1. if k == n else float(beta.ppf(.975, k+1, n-k))])


def coverage_report(items, complete):
    known = [v for v in items if type(v['covered']) is bool]
    covered = sum(v['covered'] for v in known)
    return dict(present=len(items), coverageKnown=len(known), covered=covered,
        numericallyUnresolved=len(items)-len(known), belowTruth=sum(v['belowTruth'] for v in known),
        aboveTruth=sum(v['aboveTruth'] for v in known),
        conditional=proportion(covered, len(known)),
        fullM=proportion(covered, M) if complete and len(known) == M else None)


def reduce_events(events):
    """Independent descriptive reducer; a technical gap never becomes a miss."""
    require(type(events) is list, 'Event list required')
    ids = [row['repID'] for row in events]
    if ids:
        rep_ids(ids)
    complete = ids == list(range(1, M+1))
    report = dict(M=M, T=len(events), present=len(events), complete=complete,
        missingRepIDs=sorted(set(range(1, M+1))-set(ids)), technicalMissing=M-len(events),
        oracle=coverage_report([row['oracle'] for row in events], complete),
        oracleRMSSE=None, oracleRMSSEOverEmpiricalSD=None,
        paired={}, hybrids={}, moments={}, comparisons={},
        oldCoverageCountsCalculated={name: sum(e['oldMethods'][name]['covered'] for e in events) if complete else None for name in ('baseline', 'candidate')},
        interpretation='descriptive_same_input_known_DGP_diagnostic_not_causal_or_human_method',
        intervalScope='pointwise_central_95_percent_Clopper_Pearson_not_a_new_MC_test')
    for method in ('baseline', 'candidate'):
        counts = {key: 0 for key in ('00', '01', '10', '11', 'unknown')}
        for row in events:
            old, new = row['oldMethods'][method]['covered'], row['oracle']['covered']
            require(type(old) is bool and (type(new) is bool or new is None), 'Invalid paired coverage states')
            key = str(int(old))+str(int(new)) if new is not None else 'unknown'
            counts[key] += 1
        report['paired'][method] = dict(order='old_first_oracle_second', counts=counts,
            jointlyKnown=len(events)-counts['unknown'])
        report['comparisons'][method] = {key: moments([row['comparisons'][method][key] for row in events])
            for key in ('estimateDifference', 'SEratio', 'oldStandardizedError', 'knownStandardizedError')} if events else {}
    report['hybrids'] = {key: coverage_report([row['hybrids'][key] for row in events], complete)
        for key in ('old__baseline', 'old__candidate', 'old__oracle',
                    'oracle__baseline', 'oracle__candidate', 'oracle__oracle')}
    if events:
        report['moments'] = {key: moments([row['oracle'][key] for row in events]) for key in ('estimate', 'SE', 'z')}
        report['oracleRMSSE'] = math.sqrt(math.fsum(row['oracle']['SE']**2 for row in events)/len(events))
        sd = report['moments']['estimate']['sampleSD']
        report['oracleRMSSEOverEmpiricalSD'] = report['oracleRMSSE']/sd if sd is not None and sd > 0 else None
    return report
