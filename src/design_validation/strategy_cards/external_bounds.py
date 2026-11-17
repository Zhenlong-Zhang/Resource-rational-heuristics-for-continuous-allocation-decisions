"""Outward bounds for the adaptive card's external observation and payoff law."""
from decimal import Context, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt

import numpy as np

P = 1 << 30
U = 1 << 20
D = 1000
SUPPORT = 100
LAW_SCALE = 1 << 160


def exp_bounds(q):
    down = Context(prec=44, rounding=ROUND_FLOOR)
    up = Context(prec=44, rounding=ROUND_CEILING)
    # Decimal.exp is correctly rounded in ROUND_HALF_EVEN; spare one ULP.
    lo = down.exp(down.divide(q.numerator, q.denominator))
    hi = up.exp(up.divide(q.numerator, q.denominator))
    return max(F(0), F(down.next_minus(lo))), F(up.next_plus(hi))


def total(intervals):
    return tuple(sum(x[i] for x in intervals) for i in (0, 1))


def normalize(row):
    lo, hi = total(row)
    if lo <= 0:
        raise ValueError("Positive certified normalizer required")
    return tuple((F((a*LAW_SCALE/hi).__floor__(), LAW_SCALE),
                  min(F(1), F((b*LAW_SCALE/lo).__ceil__(), LAW_SCALE)))
                 for a, b in row)


def multiply(x, y):
    return x[0]*y[0], x[1]*y[1]


@lru_cache(maxsize=8)
def channel(width):
    if width <= 0:
        raise ValueError("Positive exact width")
    atoms = [exp_bounds(-F(k*k, 2) / width**2) for k in range(SUPPORT)]
    return tuple(normalize(tuple(atoms[abs(y-n)] for y in range(SUPPORT)))
                 for n in range(SUPPORT))


def cdf(row):
    out = []
    lo = hi = F(0)
    for a, b in row[:-1]:
        lo, hi = lo+a, hi+b
        out.append((lo, min(F(1), hi)))
    out.append((F(1), F(1)))
    return out


@lru_cache(maxsize=64)
def prior(logs):
    x = [F(float.fromhex(i)) for i in logs]
    if len(x) != SUPPORT:
        raise ValueError("Complete finite prior")
    m = max(x)
    return normalize(tuple(exp_bounds(i-m) for i in x))


@lru_cache(maxsize=24)
def utilities(budget, loss):
    lo, hi = [], []
    for j in range(D+1):
        a, b = [], []
        for n in range(1, SUPPORT+1):
            x = budget*j - 1_000_000*n
            scaled = abs(x)*U*U
            k = isqrt(scaled//1_000_000)
            top = k + (k*k*1_000_000 != scaled)
            low, high = ((-loss*top, -loss*k) if x < 0 else (k, top))
            a.append(low)
            b.append(high)
        lo.append(a)
        hi.append(b)
    return np.array(lo, dtype=np.int64), np.array(hi, dtype=np.int64)


def round_law(rows):
    def lower(x): return (x*P).__floor__()
    def upper(x): return (x*P).__ceil__()
    return tuple(np.asarray([[fn(v[side]) for v in row] for row in rows], dtype=np.int64)
                 for side, fn in enumerate((lower, upper)))


def dots(laws, values):
    """Signed outward dot over integer-scaled intervals; verify no int64 overflow."""
    if min(int(a.min()) for a in laws) < 0 or max(int(a.max()) for a in laws) > P:
        raise ValueError("Finite nonnegative dyadic law")
    vmax = max(max(abs(int(a.min())), abs(int(a.max()))) for a in values)
    if laws[0].shape[-1] * P * vmax >= 2**63:
        raise ValueError("Signed dot overflow")
    pl, ph = laws
    vl, vh = values
    return (pl @ np.maximum(vl, 0).T + ph @ np.minimum(vl, 0).T,
            ph @ np.maximum(vh, 0).T + pl @ np.minimum(vh, 0).T)


class CertifiedOracle:
    """Recipient-local bins and certified one-step full-grid comparisons."""
    def __init__(self, root):
        self.root = root
        task = root["task"]
        if task["alpha"] != .5 or task["allocation_denominator"] != D or task["loss_multiplier"] != 2:
            raise ValueError("Admitted signed square-root payoff")
        self.laws = tuple(prior(tuple(logs)) for logs in root["logsHex"])
        self.channel = channel(F(task["report_width"]))
        self.base, self.scale = 2*F(task["gpa_baseline"]), F(task["gpa_scale"])
        if self.scale <= 0 or len(self.laws) != 2:
            raise ValueError("Admitted ordered payoffs")

    def latent_cdf(self, root, slot):
        return cdf(self.laws[slot])

    def report_cdf(self, root, slot, latent):
        return cdf(self.channel[latent-1])

    def belief(self, slot, history):
        law = self.laws[slot]
        for y in history:
            law = tuple(multiply(p, self.channel[n][y-1]) for n, p in enumerate(law))
        return normalize(law) if history else law

    def update_is_certified(self, root, slot, history, report):
        try:
            self.belief(slot, (*history, report))
        except ValueError:
            return False
        return True

    def gpa(self, interval):
        return [str(self.base+self.scale*x) for x in interval]
