"""Independent external-law CDF and conditional-payoff enclosures."""
from decimal import Context, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from math import isqrt

import numpy as np

N = 100
D = 1000
P = 1 << 31
U = 1 << 20


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def exp_enclosure(x):
    bounds = []
    for mode, direction in ((ROUND_FLOOR, 'next_minus'), (ROUND_CEILING, 'next_plus')):
        context = Context(prec=52, rounding=mode)
        rounded = context.divide(x.numerator, x.denominator)
        bounds.append(max(F(0), F(getattr(context, direction)(context.exp(rounded)))))
    return tuple(bounds)


def normalize(rows):
    lower = sum(a for a, _ in rows)
    upper = sum(b for _, b in rows)
    need(lower > 0, 'Independently certified normalizer')
    return tuple((a/upper, min(F(1), b/lower)) for a, b in rows)


@lru_cache(maxsize=16)
def root_law(encoded):
    vals = [F(float.fromhex(x)) for x in encoded]
    need(len(vals) == N, 'Original finite prior')
    return normalize([exp_enclosure(x-max(vals)) for x in vals])


@lru_cache(maxsize=8)
def channel(width):
    need(width > 0, 'Positive width')
    weights = [exp_enclosure(-F(k, 2)*k / (width*width)) for k in range(N)]
    return tuple(normalize([weights[abs(n-y)] for y in range(N)]) for n in range(N))


@lru_cache(maxsize=24)
def terms(budget, loss):
    bounds = [[], []]
    for j in range(D+1):
        low, high = [], []
        for n in range(1, N+1):
            surplus = budget*j - n*1_000_000
            radicand = abs(surplus)*U**2
            a = isqrt(radicand//1_000_000)
            b = a if a*a*1_000_000 == radicand else a+1
            lo, hi = (-loss*b, -loss*a) if surplus < 0 else (a, b)
            low.append(lo)
            high.append(hi)
        bounds[0].append(low)
        bounds[1].append(high)
    return tuple(np.asarray(a, dtype=np.int64) for a in bounds)


def curve(laws, values):
    lo = np.asarray([[(a*P).__floor__() for a, _ in row] for row in laws], dtype=np.int64)
    hi = np.asarray([[(b*P).__ceil__() for _, b in row] for row in laws], dtype=np.int64)
    vmin, vmax = values
    cap = max(max(abs(int(x.min())), abs(int(x.max()))) for x in values)
    need(0 <= lo.min() <= hi.max() <= P and N*P*cap < 2**63, 'Safe exact interval matrix')
    return (lo @ np.maximum(vmin, 0).T + hi @ np.minimum(vmin, 0).T,
            hi @ np.maximum(vmax, 0).T + lo @ np.minimum(vmax, 0).T)


def update(law, channel_rows, y):
    need(type(y) is int and 1 <= y <= N, 'Original report support')
    joint = tuple((a*x,b*z) for (a,b),(x,z) in zip(law,(r[y-1] for r in channel_rows)))
    mass = sum(a for a,_ in joint), min(F(1), sum(b for _,b in joint))
    return normalize(joint), mass


def contained(outer, inner, reason):
    lo, hi = map(F, outer)
    need(lo <= inner[0] <= inner[1] <= hi, reason)


def key_value(key):
    data = json.dumps(key, ensure_ascii=True, separators=(',', ':')).encode()
    word = int.from_bytes(hashlib.sha256(data).digest()[:8], 'big')
    return F(2*word+1, 2**65)


def read_bin(step, prefix, key, law):
    u = key_value(key)
    need(step[prefix+'Key'] == key and F(step[prefix+'Uniform']) == u, 'Exact path-owned RNG key')
    proof = step[prefix+'CDF']
    need(F(proof['uniform']) == u, 'CDF key binding')
    cumulative = [(F(0), F(0))]
    for i,(lo,hi) in enumerate(law,1):
        a,b = cumulative[-1]
        cumulative.append((a+lo, min(F(1), b+hi)) if i<N else (F(1),F(1)))
    selected = proof['selected']
    if selected is None:
        need(len(proof['cdf']) == N, 'Complete unresolved CDF support')
        for i, bounds in enumerate(proof['cdf'], 1):
            contained(bounds, cumulative[i], 'Unresolved CDF enclosure')
        need(all(not (F(a[1]) <= u < F(b[0]))
                 for a,b in zip([(0,0)]+proof['cdf'][:-1],proof['cdf'])), 'Unresolved CDF edge')
        return None
    need(type(selected) is int and 1<=selected<=N, 'CDF bin support')
    contained(proof['previous'], cumulative[selected-1], 'Previous CDF enclosure')
    contained(proof['upperEdge'], cumulative[selected], 'Selected CDF enclosure')
    need(F(proof['previous'][1])<=u<F(proof['upperEdge'][0]), 'Proved strict CDF bin')
    return selected


class Reference:
    def __init__(self, root):
        task = root['task']
        need(task['allocation_denominator'] == D and task['alpha'] == .5
             and task['loss_multiplier']==2 and F(task['gpa_scale'])>0,
             'Source-bound payoff')
        self.root = root
        self.laws = [root_law(tuple(logs)) for logs in root['logsHex']]
        need(len(self.laws)==2, 'Exactly two ordered laws')
        self.channel = channel(F(task['report_width']))
        self.base = 2*F(task['gpa_baseline'])
        self.scale = F(task['gpa_scale'])
        self.cost = task['cost_millihours']
        need(type(self.cost) is int and self.cost>0, 'Positive original integer cost')

    def gpa(self, pair):
        return tuple(self.base+self.scale*x for x in pair)
