"""Saved endpoint design diagnostics; no coefficient fitting or new outcomes.

The original ten-row OLS values and nine-schedule SDs are preserved verbatim.
Exact centered geometry quantifies potential leverage, not Monte Carlo error.
"""
from fractions import Fraction as Q
import math

from . import true_need_diagnostics as T
from .c5_moments import need

CONDITIONS = T.CONDITIONS
PRICES = T.LABELS
FIELDS = T.FIELDS
METRICS = ('gap_slope', 'lower_need_partial')
ZERO = dict(newDraws=0, newPolicyRuns=0, newFitInvocations=0)


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def endpoint_key(row):
    return row['condition'], row['price'], row['variant'], row['main']


def schedule_key(row):
    return row['condition'], row['price'], row['schedule']


def schedule_roster():
    return [(c, p, s) for c in CONDITIONS for p in PRICES for s in range(9)]


def endpoint_roster():
    return [(*k, m) for k in schedule_roster() for m in range(1, 11)]


def sid(key):
    return '%s:%s:s%02d' % key


def eid(key):
    return sid(key[:3]) + ':m%02d' % key[3]


def indexed(rows, key, expected, label):
    need(type(rows) is list, label + ': list required')
    out = {}
    for row in rows:
        need(type(row) is dict, label + ': object required')
        k = key(row)
        need(k in expected and k not in out, label + ': duplicate/out-of-roster ID ' + repr(k))
        out[k] = row
    need(set(out) == set(expected), label + ': incomplete fixed roster')
    return out


def validate_row(r):
    need(type(r) is dict and set(r) == set(FIELDS), 'Endpoint field set changed')
    need(type(r['condition']) is str and r['condition'] in CONDITIONS and
         type(r['price']) is str and r['price'] in PRICES and
         type(r['variant']) is int and r['variant'] in range(9) and
         type(r['main']) is int and r['main'] in range(1, 11), 'Endpoint identity/type')
    need(type(r['index']) is int and 0 <= r['index'] <= 1000 and finite(r['share']) and
         abs(r['share'] - r['index'] / 1000) <= 1e-12, 'Saved allocation index/share')
    need(type(r['initial_budget_milli']) is int and r['initial_budget_milli'] > 0,
         'Initial budget, not remaining budget, required')
    need(type(r['needs_displayed']) is list and len(r['needs_displayed']) == 2 and
         all(type(n) is int and 1 <= n <= 100 for n in r['needs_displayed']), 'True need categories')
    need(type(r['logical_order']) is list and sorted(r['logical_order']) == ['A', 'B'],
         'Displayed/logical mapping')
    x, y = r['rank_x'], r['rank_y']
    need((x is None and y is None) or
         (finite(x) and x >= 0 and finite(y) and 0 <= y <= 1 and
          min(abs(y - r['share']), abs(y - (1 - r['share']))) <= 1e-12),
         'Legacy initial-mean fields are paired/oriented, not true-need fields')
    need(type(r['initial_ask']) is int and r['initial_ask'] in (0, 1) and
         type(r['mean_paid']) is int and r['mean_paid'] >= 0 and
         r['initial_ask'] == int(r['mean_paid'] > 0), 'Saved ask/count mismatch')
    for field in ('terminal_action', 'selected_rule'):
        need(r[field] is None or type(r[field]) is str and bool(r[field].strip()), 'Missing label must be null')



def rational(q):
    q = Q(q)
    return dict(numerator=str(q.numerator), denominator=str(q.denominator), decimal=float(q))


def project(row):
    a, b = row['needs_displayed']
    if a == b:
        return None
    lo, hi = Q(1000 * min(a, b), row['initial_budget_milli']), Q(1000 * max(a, b), row['initial_budget_milli'])
    return lo, hi, Q(row['index'] if a > b else 1000-row['index'], 1000)


def design_geometry(points):
    """Input ONLY (lo,hi); neither outcome nor saved coefficient reaches this function."""
    n = len(points)
    if n == 0:
        return dict(n=0, gapRank=0, partialRank=0, status='empty', members=[])
    means = [sum((p[j] for p in points), Q(0))/n for j in (0, 1)]
    z = [(a-means[0], b-means[1]) for a, b in points]
    aa = sum((a*a for a, b in z), Q(0))
    bb = sum((b*b for a, b in z), Q(0))
    ab = sum((a*b for a, b in z), Q(0))
    det = aa*bb-ab*ab
    gapss = aa+bb-2*ab
    need(aa >= 0 and bb >= 0 and det >= 0 and gapss >= 0, 'Exact design arithmetic invalid')
    rank = 1 + (2 if det > 0 else int(aa > 0 or bb > 0))
    members = []
    for a, b in z:
        item = dict(gapLeverage=None, gapSlopeWeight=None, partialLeverage=None, partialLoWeight=None)
        if gapss > 0:
            item.update(gapLeverage=rational(Q(1,n)+(b-a)**2/gapss), gapSlopeWeight=rational((b-a)/gapss))
        if det > 0:
            item.update(partialLeverage=rational(Q(1,n)+(bb*a*a-2*ab*a*b+aa*b*b)/det),
                        partialLoWeight=rational((bb*a-ab*b)/det))
        members.append(item)
    return dict(n=n, gapRank=1+int(gapss > 0), partialRank=rank,
                status='full_rank' if gapss > 0 and det > 0 else 'structurally_rank_deficient',
                gapSS=rational(gapss), lowerSS=rational(aa), higherSS=rational(bb), crossSS=rational(ab),
                gramDeterminant=rational(det),
                normalizedDeterminant=rational(det/(aa*bb)) if aa*bb > 0 else None,
                correlationSquared=rational(ab*ab/(aa*bb)) if aa*bb > 0 else None,
                lowerRange=rational(max(p[0] for p in points)-min(p[0] for p in points)),
                higherRange=rational(max(p[1] for p in points)-min(p[1] for p in points)),
                gapRange=rational(max(b-a for a,b in points)-min(b-a for a,b in points)), members=members)


def reduce_saved(data):
    schedule_rows, point_rows, cells = [], [], []
    for key in schedule_roster():
        group = [data['endpoints'][(*key, m)] for m in range(1, 11)]
        projected = [project(r) for r in group]
        eligible = [p for p in projected if p is not None]
        geometry = design_geometry([(p[0],p[1]) for p in eligible])
        saved = data['schedules'][key]
        diagnostics = iter(geometry.pop('members'))
        saved_copy = {f: saved[f] for f in ('gap_rank', 'partial_rank', *METRICS,
                                            'initial_mean_slope', 'initial_mean_eligible')}
        schedule_rows.append(dict(scheduleID=sid(key), coordinate=list(key),
                                  endpointIDs=[eid((*key,m)) for m in range(1,11)],
                                  saved=saved_copy, geometry=geometry,
                                  exactRankAgreesWithSaved=(geometry['gapRank']==saved['gap_rank'] and
                                                           geometry['partialRank']==saved['partial_rank']),
                                  higherNeedShareRange=rational(max(p[2] for p in eligible)-min(p[2] for p in eligible))))
        for row, p in zip(group, projected):
            detail = dict(endpointID=eid(endpoint_key(row)), saved=dict(row), exactTrueNeedTie=p is None,
                          derived=None, design=None)
            if p is not None:
                detail['derived'] = dict(lowerNeedOverInitialBudget=rational(p[0]),
                                         higherNeedOverInitialBudget=rational(p[1]),
                                         gap=rational(p[1]-p[0]), higherNeedShare=rational(p[2]))
                detail['design'] = next(diagnostics)
            point_rows.append(detail)
    for c in CONDITIONS:
        for price in PRICES:
            metrics = {}
            for name in METRICS:
                values = [data['schedules'][c,price,s][name] for s in range(9)]
                metrics[name] = dict(accepted=dict(data['cells'][c,price][name]),
                                     savedValues=values, minimum=min(values), maximum=max(values),
                                     literalFloatSigns=dict(negative=sum(v<0 for v in values),
                                                            zero=sum(v==0 for v in values), positive=sum(v>0 for v in values)))
            cells.append(dict(condition=c, price=price, metrics=metrics))
    return dict(schedules=schedule_rows, points=point_rows, cells=cells)



def admit(rows, summary):
    """Validate the complete scientific population, not private source receipts."""
    need(type(rows) is list, 'Endpoint list required')
    for row in rows:
        validate_row(row)
    endpoints = indexed(rows, endpoint_key, endpoint_roster(), 'Endpoints')
    need(summary['schema'] == T.SCHEMA and summary['fullFiniteTarget'] is True
         and summary['uncertainty'] == 'sample_sd_across_nine_fixed_schedules_ddof1'
         and summary['verifiedTrials'] == 2700, 'Saved summary scope')
    schedules = indexed(summary['schedules'], schedule_key, schedule_roster(), 'Saved fits')
    cells = indexed(summary['conditions'], lambda r: (r['condition'], r['price']),
                    [(c, p) for c in CONDITIONS for p in PRICES], 'Saved cells')
    ties_total = 0
    for key, saved in schedules.items():
        group = [endpoints[(*key, m)] for m in range(1, 11)]
        ties = sum(r['needs_displayed'][0] == r['needs_displayed'][1] for r in group)
        ties_total += ties
        need(saved['complete'] is True and saved['present_mains'] == list(range(1, 11))
             and saved['missing_mains'] == [] and saved['gap_rank'] == 2
             and saved['partial_rank'] == 3, 'Saved complete/full-rank fit')
        for name, value in dict(available_rows=10, missing=0, ties=ties, eligible=10-ties).items():
            need(type(saved[name]) is int and saved[name] == value, 'Saved count discrepancy')
        need(all(finite(saved[name]) for name in METRICS)
             and (saved['initial_mean_slope'] is None or finite(saved['initial_mean_slope'])),
             'Finite saved coefficients required')
    need(ties_total == 66, 'Changed accepted tie union')
    for (condition, price), cell in cells.items():
        need(cell['complete'] is True, 'Complete saved cell required')
        for metric in METRICS:
            values = [schedules[condition, price, s][metric] for s in range(9)]
            mean = math.fsum(values)/9
            sd = math.sqrt(math.fsum((x-mean)**2 for x in values)/8)
            saved = cell[metric]
            need(saved['defined_schedules'] == saved['expected_schedules'] == 9
                 and finite(saved['mean']) and finite(saved['sample_sd'])
                 and abs(saved['mean']-mean) <= 1e-10
                 and abs(saved['sample_sd']-sd) <= 1e-10,
                 'Saved SD/mean mismatch (descriptive only, not a refit)')
    return dict(endpoints=endpoints, schedules=schedules, cells=cells)


def diagnose(rows, summary):
    data = admit(rows, summary)
    return dict(schema='coefficient-design-diagnostics-v1',
                counters={**ZERO, 'newDesignSchedules': 270},
                scope=dict(endpoints=2700, schedules=270, cells=30, trueNeedTies=66, eligible=2634),
                uncertainty='saved_nine_schedule_sample_sd_not_mcse_or_ci',
                **reduce_saved(data))
