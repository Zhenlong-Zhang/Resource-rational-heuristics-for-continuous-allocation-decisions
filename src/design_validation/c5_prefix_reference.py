"""Independent direct sum-tail check for conditional bounded-prefix planning."""
from fractions import Fraction as Q


def family_margins(radii, variance, maximum, count):
    if len(radii) != 90 or any(x < 0 for x in radii) or variance <= 0 or maximum <= 0 or max(radii) != maximum:
        raise ValueError('Invalid bounded-prefix geometry')
    if type(count) is not int or count < 2:
        raise ValueError('Future repetitions required')
    lower = upper = Q(0)
    for multiplier, z in ((6, Q(1, 3)), (1, Q(1, 9))):
        value, power = Q(0), z
        for index in range(128):
            value += 2*power/(2*index+1)
            power *= z*z
        lower += multiplier*value
        upper += multiplier*(value+2*power/(257*(1-z*z)))

    def margin(n, log):
        remaining = Q(1, 10)-sum(radii, Q(0))/(90*n)
        if remaining <= 0:
            return None
        total = 90*n*remaining
        return total*total-2*(n*variance+maximum*total/3)*log

    return margin(count, upper), margin(count-1, lower), margin(count-1, upper)


def verify(radii, variance, maximum, count):
    at, previous, _ = family_margins(radii, variance, maximum, count)
    if at is None or at < 0 or (previous is not None and previous >= 0):
        raise ValueError('Independent adjacent-integer certificate failed')
    return {'currentPassesUpper': True, 'previousFailsLower': True,
            'method': 'independent_original_sum_tail_inequality', 'seriesTerms': 128}
