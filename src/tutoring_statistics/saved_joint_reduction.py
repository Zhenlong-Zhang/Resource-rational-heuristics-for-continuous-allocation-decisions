"""Original CR2-only retrospective joint comparison on eight saved cells."""
from .magnitude_design import IDS, truth, require, case_for
B = 2000


def interval(count, n):
    if not n: return None
    from scipy.stats import beta
    return [0. if count == 0 else float(beta.ppf(.025, count, n-count+1)),
            1. if count == n else float(beta.ppf(.975, count+1, n-count))]

def empty_case(case):
    return dict(caseID=case['caseID'], case=case, truth=truth(case), M=B, T=0, present=0, A4=0,
        Aj={key: 0 for key in IDS}, newCounts={key: 0 for key in ('R', 'F', 'D')},
        originalOnCompletedCounts={key: 0 for key in ('R', 'F', 'D')},
        transitions={name: {key: 0 for key in ('00', '01', '10', '11')} for name in ('R', 'F', 'D')},
        comparisonTransitions={name: {key: 0 for key in ('00', '01', '10', '11')} for name in IDS},
        availableCounts={key: 0 for key in ('R', 'F', 'D')})

def accumulate(target, event):
    target['present'] += 1
    if not event['technicalCompleted']: return
    target['T'] += 1; a4 = event['availableCount'] == 4; target['A4'] += int(a4)
    for name in IDS:
        target['Aj'][name] += int(event['tests'][name]['status'] == 'available')
        old, new = event['original']['tests'][name]['rejected'], event['tests'][name]['rejected']
        require(type(old) is type(new) is bool, 'Completed comparison cannot be technical unknown')
        target['comparisonTransitions'][name][str(int(old))+str(int(new))] += 1
    for name in ('R', 'F', 'D'):
        old, new = event['original'][name], event['events'][name]
        if new is None:
            require(old is None, 'Changed applicable scientific event'); continue
        require(type(old) is type(new) is bool, 'Unclassified family event')
        target['newCounts'][name] += int(new); target['originalOnCompletedCounts'][name] += int(old)
        target['availableCounts'][name] += int(a4 and new)
        target['transitions'][name][str(int(old))+str(int(new))] += 1

def finish_case(case):
    nulls = any(v['trueNull'] for v in case['truth'].values()); alternatives = any(not v['trueNull'] for v in case['truth'].values())
    report = dict(case); report['family'] = {}
    for name, applicable in (('R', True), ('F', nulls), ('D', alternatives)):
        report['family'][name] = None if not applicable else dict(
            count=case['newCounts'][name], denominator=B, technicalCompleted=case['T'],
            rate=case['newCounts'][name]/B if case['T'] == B else None,
            CI=interval(case['newCounts'][name], B) if case['T'] == B else None,
            availableCount=case['availableCounts'][name], availableDenominator=case['A4'],
            availableRate=case['availableCounts'][name]/case['A4'] if case['A4'] else None,
            availableCI=interval(case['availableCounts'][name], case['A4']))
    return report


def reduce_events(manifest, events):
    require(len({e['ordinal'] for e in events}) == len(events), "Duplicate saved joint ID")
    reports = {case['caseID']: empty_case(case) for case in manifest['cases']}
    for event in events:
        expected = case_for(manifest, event['ordinal'])
        require(event['caseID'] == expected['caseID'] and event['repID'] == expected['repID']
                and event['truth'] == truth(expected), "Changed saved joint identity/truth")
        accumulate(reports[event['caseID']], event)
    return [finish_case(value) for value in reports.values()]
