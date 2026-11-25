"""Prospective no-draw binary64 report tables, not historical runtime proof.

The same source need not yield identical bits across CPUs, libm IFUNCs, NumPy
builds, or floating-point environments. Widths are explicit binary64 inputs.
"""
import hashlib
import math

import numpy as np

from .bounded_task import NEEDS, logsumexp

GRID = 1 << 53
SCHEMA = 'portable-canonical-report-table-v1'
RULE = 'ordered_binary64_strict_greater_final_category_fallback'


def width_from_hex(encoded):
    if type(encoded) is not str:
        raise ValueError('Canonical binary64 width hex required')
    try:
        width = float.fromhex(encoded)
    except (ValueError, OverflowError) as exc:
        raise ValueError('Canonical binary64 width hex required') from exc
    if not math.isfinite(width) or width <= 0 or width.hex() != encoded:
        raise ValueError('Positive finite canonical binary64 width hex required')
    return width


def _row(logs):
    probabilities = np.exp(logs)
    if not np.isfinite(logs).all() or not np.isfinite(probabilities).all():
        raise ValueError('Finite report row required')
    total, cumulative = 0., []
    for p in probabilities:
        total = float(np.add(np.float64(total), np.float64(p)))
        cumulative.append(total.hex())
    counts, previous = [], 0
    for h in cumulative[:-1]:
        numerator, denominator = float.fromhex(h).as_integer_ratio()
        cutoff = max(previous, min(GRID, (numerator*GRID+denominator-1)//denominator))
        counts.append(cutoff-previous)
        previous = cutoff
    counts.append(GRID-previous)
    return dict(normalizedLogHex=[float(x).hex() for x in logs],
                probabilityHex=[float(x).hex() for x in probabilities],
                cumulativeHex=cumulative, rawMassHex=float(probabilities.sum()).hex(),
                counts=counts, denominator=GRID, lastCategoryIncludesFallback=True)


def generate_report_table(width_hex):
    """Construct all 100 rows without drawing a report or running a policy."""
    width = width_from_hex(width_hex)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        raw = -0.5*((NEEDS[:, None]-NEEDS[None, :])/width)**2
        logs = raw-logsumexp(raw, axis=1)[:, None]
    if not np.isfinite(logs).all():
        raise ValueError('Width produces nonfinite report log values')
    return dict(schema=SCHEMA, reportWidthHex=width_hex,
                key=hashlib.sha256(width_hex.encode('ascii')).hexdigest(),
                needs=list(range(1, 101)), reports=list(range(1, 101)),
                samplingRule=RULE, rows=[_row(logs[i]) for i in range(100)],
                newScientificDraws=0)
