"""Independent H-only target and event reconstruction; no actor imports."""
import math
from collections import Counter


FAMILIES = ('ES', 'EO_raw_surplus', 'PN', 'AL', 'GN')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def actor_reference(H, family):
    """Reconstruct the original literal formula, native tie and trace counts."""
    need(family in FAMILIES, 'Exactly five precommitted procedures')
    need(type(H['budgetInteger']) is int and H['budgetInteger'] > 0,
         'Positive original resource')
    operations = []

    def event(operation, count=1):
        operations.append((operation, count))

    event('retrieve_rule')
    a = b = budget = None
    if family != 'ES':
        a, b = map(float, H['meanStrings'])
        event('read_scalar', 2)
    if family == 'EO_raw_surplus':
        budget = float(H['resourceString'])
        event('read_scalar')
    share = None
    known = None
    tree = None
    if family == 'ES':
        known = 500
    elif family == 'EO_raw_surplus':
        first = budget + a
        numerator = first - b
        denominator = budget + budget
        share = numerator / denominator
        event('add_subtract', 3)
        event('multiply_divide')
        tree = dict(numerator1=first, numerator=numerator, denominator=denominator,
                    rawShare=share, expression='((B+a)-b)/(B+B)')
    elif family == 'PN':
        share = a / (a+b)
        event('add_subtract')
        event('multiply_divide')
    else:
        difference = abs(a-b)
        epsilon = 1e-9 * max(1, a, b)
        event('add_subtract')
        event('abs')
        event('compare', 2)
        event('multiply_divide')
        event('compare')
        if difference <= epsilon:
            known = 500
        else:
            event('compare')
            lower = a < b
            known = (1000 if lower else 0) if family == 'AL' else (0 if lower else 1000)
    if known is None:
        event('clip')
        clipped = min(1., max(0., share))
        event('multiply_divide')
        location = clipped * 1000
        event('floor')
        event('ceil')
        left, right = math.floor(location), math.ceil(location)
        event('compare')
        candidates = sorted({left, right})
        distances = {}
        for j in candidates:
            event('add_subtract')
            event('abs')
            distances[j] = abs(j-location)
        if len(candidates) > 1:
            event('compare', len(candidates)-1)
        threshold = min(distances.values()) + 1e-12
        event('add_subtract')
        selected = []
        for j in candidates:
            event('compare')
            if distances[j] <= threshold:
                selected.append(j)
        for j in selected:
            event('add_subtract')
            event('abs')
        if len(selected) > 1:
            event('compare', len(selected)-1)
        index = min(selected, key=lambda j: (abs(j-500), j))
    else:
        index = known
    event('local_access')
    event('native_select')
    event('common_execution')
    counts = Counter()
    for operation, count in operations:
        counts[operation] += count
    return dict(target=dict(kind='KnownIndex' if known is not None else 'Fraction',
                            index=index, rawShare=share),
                arithmeticTree=tree, counts=dict(counts), operationSequence=operations)
