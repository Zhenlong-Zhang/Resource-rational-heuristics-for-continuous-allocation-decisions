"""Independent interval-partition check; no sampling or model evaluation."""
from . import canonical_latent as L
from fractions import Fraction


def verify_law(counts):
    if type(counts) is not list or len(counts) != 100:
        raise ValueError('Expected 100 masses')
    previous, checked = 0, 0
    for index, mass in enumerate(counts):
        if type(mass) is not int or mass < 0:
            raise ValueError('Invalid integer mass')
        end = previous + mass
        if end > L.GRID:
            raise ValueError('Partition exceeds grid')
        if mass:
            for integer in (previous, end - 1):
                if L.category_from_integer(counts, integer) != index + 1:
                    raise ValueError('Integer interval boundary mismatch')
                if L.category_from_uniform(counts, integer / L.GRID) != index + 1:
                    raise ValueError('Uniform interval boundary mismatch')
                checked += 1
        previous = end
    if previous != L.GRID:
        raise ValueError('Incomplete integer grid')
    return {'categories': 100, 'denominator': L.GRID,
            'boundaryChecks': checked, 'completeIntegerGrid': True}


def verify_historical_cdf(counts, cumulative_hex):
    """Check the partition against the original sequential CDF bit values."""
    if type(cumulative_hex) is not list or len(cumulative_hex) != 100:
        raise ValueError('Expected 100 historical cumulative values')
    verify_law(counts)
    prior, last_value = 0, 0.0
    for index, encoded in enumerate(cumulative_hex):
        if type(encoded) is not str:
            raise ValueError('Expected encoded binary64 CDF')
        value = float.fromhex(encoded)
        if value.hex() != encoded or not last_value <= value < float('inf'):
            raise ValueError('Noncanonical CDF')
        last_value = value
        if index == 99:
            break
        exact = Fraction.from_float(value) * L.GRID
        cutoff = min(L.GRID, max(0, -(-exact.numerator // exact.denominator)))
        if cutoff < prior or counts[index] != cutoff - prior:
            raise ValueError('CDF cutoff differs from fixed mass')
        prior = cutoff
    if counts[-1] != L.GRID - prior:
        raise ValueError('Final-category fallback differs')
    return {'historicalPartitionMatches': True, 'denominator': L.GRID}
