"""Exact 53-bit integer partitions for the fixed latent categorical law.

These routines consume scientific inputs; they do not regenerate historical
probability bits or claim that a different floating-point environment does so.
"""
from bisect import bisect_right

GRID = 1 << 53


def cutoffs(counts):
    if (type(counts) is not list or len(counts) != 100
            or any(type(x) is not int or x < 0 for x in counts)
            or sum(counts) != GRID):
        raise ValueError('Expected 100 nonnegative integer masses summing to 2**53')
    cumulative, total = [], 0
    for count in counts:
        total += count
        cumulative.append(total)
    return cumulative


def category_from_integer(counts, integer):
    boundaries = cutoffs(counts)
    if type(integer) is not int or not 0 <= integer < GRID:
        raise ValueError('Uniform integer must be in [0, 2**53)')
    return bisect_right(boundaries, integer) + 1


def category_from_uniform(counts, uniform):
    if type(uniform) is not float or not 0 <= uniform < 1:
        raise ValueError('Expected a finite binary64 uniform in [0, 1)')
    numerator, denominator = uniform.as_integer_ratio()
    integer, remainder = divmod(numerator * GRID, denominator)
    if remainder:
        raise ValueError('Uniform is not on the 53-bit grid')
    return category_from_integer(counts, integer)
