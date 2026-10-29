"""Independent saved CR2-only joint report with binomial-tail inversion."""
from .independent_magnitude_reduction import IDS, truth, require, case_for


def cp(count, size):
    if size == 0: return None
    from scipy.optimize import brentq
    from scipy.special import bdtr
    lower = 0. if count == 0 else float(brentq(lambda x: float(bdtr(count-1, size, x))-.975, 0., 1., xtol=1e-13, rtol=1e-12))
    upper = 1. if count == size else float(brentq(lambda x: float(bdtr(count, size, x))-.025, 0., 1., xtol=1e-13, rtol=1e-12))
    return [lower, upper]

def reduce_events(manifest, events):
    """Independent integer reduction; do not call the producer reducer."""
    require(len({e['ordinal'] for e in events}) == len(events), "Duplicate saved joint ID")
    for event in events:
        expected = case_for(dict(manifest=manifest, expectedSet=set(manifest['expectedOrdinals'])), event['ordinal'])
        require(event['caseID'] == expected['caseID'] and event['repID'] == expected['repID']
                and event['truth'] == truth(expected), "Changed independent saved identity/truth")
    reports = []
    for case in manifest['cases']:
        rows = [e for e in events if e['caseID'] == case['caseID']]
        complete = [e for e in rows if e['technicalCompleted']]
        available = [e for e in complete if e['availableCount'] == 4]
        fixed_truth = truth(case)
        output = dict(caseID=case['caseID'], case=case, truth=fixed_truth, M=2000, T=len(complete), present=len(rows), A4=len(available),
            Aj={k: sum(e['tests'][k]['status'] == 'available' for e in complete) for k in IDS},
            newCounts={k: sum(e['events'][k] is True for e in complete) for k in ('R', 'F', 'D')},
            originalOnCompletedCounts={k: sum(e['original'][k] is True for e in complete) for k in ('R', 'F', 'D')},
            availableCounts={k: sum(e['events'][k] is True for e in available) for k in ('R', 'F', 'D')}, family={})
        output['transitions'] = {k: {bits: sum(str(int(e['original'][k]))+str(int(e['events'][k])) == bits
            for e in complete if e['events'][k] is not None) for bits in ('00', '01', '10', '11')} for k in ('R', 'F', 'D')}
        output['comparisonTransitions'] = {k: {bits: sum(str(int(e['original']['tests'][k]['rejected']))+
            str(int(e['tests'][k]['rejected'])) == bits for e in complete) for bits in ('00', '01', '10', '11')} for k in IDS}
        for key in ('R', 'F', 'D'):
            applicable = key == 'R' or any(v['trueNull'] if key == 'F' else not v['trueNull'] for v in fixed_truth.values())
            k, a, t = output['newCounts'][key], output['availableCounts'][key], len(complete)
            output['family'][key] = None if not applicable else dict(count=k, denominator=2000, technicalCompleted=t,
                rate=k/2000 if t == 2000 else None, CI=cp(k, 2000) if t == 2000 else None,
                availableCount=a, availableDenominator=len(available), availableRate=a/len(available) if available else None,
                availableCI=cp(a, len(available)))
        reports.append(output)
    return reports
