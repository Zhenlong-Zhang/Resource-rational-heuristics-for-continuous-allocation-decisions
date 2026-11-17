"""Fixed-weight descriptive summaries; missing roots never become extra weight."""
from collections import Counter
from fractions import Fraction as F
import math

from .inputs import ORIGINAL, need

FIXED = ('equal_minutes', 'equal_projected_gpa', 'all_to_higher_guess',
         'all_to_lower_guess', 'ask_each_equal_projected_gpa')
ADAPTIVE = 'adaptive_one_more'


def coordinate(value):
    need(type(value) is list and len(value) == 3 and type(value[0]) is str
         and type(value[1]) is int and type(value[2]) is int and tuple(value) in ORIGINAL,
         'Original condition, variant and main required')
    return tuple(value)


def adaptive_row(root, result, check):
    """Intersect source-linked bounds for the same actual selected terminal action."""
    need(check['status'] == 'PASS' and result['status'] == check['disposition'],
         'Independently checked adaptive disposition')
    task = root['task']
    initial, cost = task['total_millihours'], task['cost_millihours']
    paid = check['paidReports']
    cap = initial // cost
    need(type(paid) is int and 0 <= paid <= cap, 'Actual affordable paid reports')
    from .fixed import payoff_interval
    support = list(map(F, payoff_interval(task, initial)))
    if result['status'] == 'terminal':
        end = check['terminal']
        need(sum(end['cursors']) == paid and end['budget'] == initial - paid*cost
             and end['target'] == result['target'], 'Paid terminal budget and target')
        p, q = list(map(F, result['externalConditionalExpectedGPAInterval'])), list(map(F, end['independentlyEnclosedGPA']))
        bounds = [max(p[0], q[0], support[0]), min(p[1], q[1], support[1])]
        need(bounds[0] <= bounds[1], 'Nonempty same-estimand interval intersection')
        high = paid
    else:
        need(result['status'].startswith('unresolved_'), 'Known unresolved frontier required')
        bounds, high = support, cap
    return {'card': ADAPTIVE, 'status': result['status'], 'GPAInterval': list(map(str, bounds)),
            'questions': [paid, high], 'tutoringMinutes': [str(F(n*cost, 1000)) for n in (paid, high)],
            'target': result.get('target')}


def summarize(records, *, full=False):
    """Summarize supplied checked scientific rows, not their private provenance."""
    indexed = {}
    for record in records:
        key = coordinate(record['id'])
        need(key not in indexed and record['replicate'] == 0 and type(record['replicate']) is int,
             'Unique original roots and exactly one adaptive rep0')
        rows = record['fixed']
        need(tuple(r['card'] for r in rows) == FIXED, 'Exact ordered five-card roster')
        need(record['adaptive']['card'] == ADAPTIVE, 'Adaptive card required')
        indexed[key] = record
    if full:
        need(set(indexed) == ORIGINAL, 'Full summary requires every original root')
    cells = []
    for condition in sorted({key[0] for key in indexed}):
        members = [indexed[key] for key in sorted(indexed) if key[0] == condition]
        missing = 90 - len(members)
        for index, card in enumerate(FIXED):
            rows = [r['fixed'][index] for r in members]
            need(all(r['status'] in ('scored', 'unresolved', 'unavailable') for r in rows),
                 'Explicit fixed-card evidence status')
            values = [r['expectedGPA'] for r in rows if r['status'] == 'scored']
            need(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'Finite GPA points')
            need(all(r.get('expectedGPA') is None for r in rows if r['status'] != 'scored'),
                 'Unavailable/unresolved is not a point estimate')
            contribution = sum((F.from_float(float(v)) for v in values), F()) / 90
            available = all(r.get('expectedGPAInterval') is not None for r in rows)
            support = [str(sum((F(r['expectedGPAInterval'][side]) for r in rows), F())/90)
                       for side in (0, 1)] if available and not missing else None
            need(support is None or F(support[0]) <= F(support[1]), 'Ordered support bounds')
            costs = {}
            for field in ('expectedQuestions', 'expectedTutoringMinutes'):
                known = not missing and all(r.get(field) is not None for r in rows)
                costs[field] = str(sum((F(str(r[field])) for r in rows), F())/90) if known else None
            cells.append({'condition': condition, 'card': card, 'denominator': 90,
                          'observedRoots': len(rows), 'missingRoots': missing,
                          'statusCounts': dict(Counter(r['status'] for r in rows)),
                          'expectedCombinedGPA': float(contribution) if len(values) == 90 else None,
                          'pointContribution': float(contribution), 'meanPayoffSupport': support,
                          'meanQuestions': costs['expectedQuestions'],
                          'meanTutoringMinutes': costs['expectedTutoringMinutes'],
                          'intervalType': 'payoff_support_not_numerical_error_or_sampling_CI'})
        rows = [r['adaptive'] for r in members]
        means = {}
        for field in ('GPAInterval', 'questions', 'tutoringMinutes'):
            need(all(len(r[field]) == 2 and F(r[field][0]) <= F(r[field][1]) for r in rows),
                 'Ordered adaptive bounds')
            means[field] = [str(sum((F(r[field][side]) for r in rows), F())/90)
                            for side in (0, 1)] if not missing else None
        cells.append({'condition': condition, 'card': ADAPTIVE, 'denominator': 90,
                      'observedRoots': len(rows), 'missingRoots': missing,
                      'statusCounts': dict(Counter(r['status'] for r in rows)), 'means': means,
                      'withinRootMCSE': None,
                      'intervalType': 'numerical_enclosure_and_unresolved_weight_not_sampling_CI'})
    return {'schema': 'strategy-card-summary-v1', 'observedRoots': len(indexed),
            'missingRoots': len(ORIGINAL - indexed.keys()), 'fullFiniteTarget': set(indexed) == ORIGINAL,
            'conditionRootWeight': '1/90', 'overallRootWeight': '1/450', 'adaptiveReplicatesPerRoot': 1,
            'cells': cells, 'withinRootMCSE': None, 'rankingCertified': False,
            'humanTargetsEstablished': False, 'rowSourceAuthenticated': False,
            'scope': 'descriptive_fixed_design_not_report_stream_precision_or_global_optimality'}
