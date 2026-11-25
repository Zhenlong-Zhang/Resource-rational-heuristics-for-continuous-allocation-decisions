"""Independent prospective report-table reconstruction and strict-grid check.

Exact agreement is local to the actual numerical environment; it is not a
cross-CPU, IFUNC, or historical execution equivalence guarantee.
"""
import hashlib
import math

import numpy as np

GRID = 1 << 53


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(_exact(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def grid_partition(cumulative):
    _require(type(cumulative) is list and len(cumulative) == 100, 'Complete CDF required')
    counts, cuts, previous = [], [], 0
    for index, encoded in enumerate(cumulative):
        _require(type(encoded) is str, 'Canonical CDF hex required')
        try:
            value = float.fromhex(encoded)
        except (ValueError, OverflowError) as exc:
            raise ValueError('Canonical CDF hex required') from exc
        _require(math.isfinite(value) and value >= 0 and value.hex() == encoded,
                 'Finite nonnegative canonical CDF required')
        numerator, denominator = value.as_integer_ratio()
        low, high = 0, GRID
        if index != 99:
            while low < high:
                middle = (low+high)//2
                if middle*denominator < numerator*GRID:
                    low = middle+1
                else:
                    high = middle
            boundary = low
            _require(boundary == 0 or (boundary-1)*denominator < numerator*GRID,
                     'Last included grid point')
            _require(boundary == GRID or boundary*denominator >= numerator*GRID,
                     'First excluded grid point')
        else:
            boundary = GRID
        _require(previous <= boundary <= GRID, 'Nonmonotone CDF grid')
        counts.append(boundary-previous)
        cuts.append(boundary)
        previous = boundary
    return counts, cuts


def _reference_rows(width):
    support = np.arange(1, 101, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        distance = support[:, None]-support[None, :]
        scaled = distance/width
        squared = scaled**2
        raw = -0.5*squared
        maximum = np.max(raw, axis=1, keepdims=True)
        exponentials = np.exp(raw-maximum)
        normalizer = maximum+np.log(np.sum(exponentials, axis=1, keepdims=True))
        logs = raw-np.squeeze(normalizer, axis=1)[:, None]
    _require(logs.shape == (100, 100) and np.isfinite(logs).all(), 'Finite report matrix')
    rows = []
    for values in logs:
        mass = np.exp(values)
        _require(np.isfinite(mass).all() and (mass >= 0).all(), 'Finite probabilities')
        cumulative, total = [], 0.0
        for item in mass:
            total = float(np.float64(total)+np.float64(item))
            cumulative.append(total.hex())
        counts, cuts = grid_partition(cumulative)
        _require(sum(counts) == GRID and cuts[-1] == GRID, 'Complete domain')
        rows.append(dict(normalizedLogHex=[float(x).hex() for x in values],
                         probabilityHex=[float(x).hex() for x in mass],
                         cumulativeHex=cumulative, rawMassHex=float(mass.sum()).hex(),
                         counts=counts, denominator=GRID, lastCategoryIncludesFallback=True))
    return rows


def verify_report_table(table, width_hex):
    """Rebuild and compare the complete table for an independently supplied width."""
    _require(type(width_hex) is str, 'Width hex required')
    try:
        width = float.fromhex(width_hex)
    except (ValueError, OverflowError) as exc:
        raise ValueError('Width hex required') from exc
    _require(math.isfinite(width) and width > 0 and width.hex() == width_hex,
             'Positive finite canonical width required')
    expected = dict(schema='portable-canonical-report-table-v1', reportWidthHex=width_hex,
                    key=hashlib.sha256(width_hex.encode('ascii')).hexdigest(),
                    needs=list(range(1, 101)), reports=list(range(1, 101)),
                    samplingRule='ordered_binary64_strict_greater_final_category_fallback',
                    rows=_reference_rows(width), newScientificDraws=0)
    _require(_exact(table, expected), 'Report table mismatch')
    return True
