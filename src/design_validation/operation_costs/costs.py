"""Declared scalar-array operation costs, not measured machine or human time."""
from fractions import Fraction as F

COEFFICIENTS = {
    'read_scalar': F(1, 2), 'retrieve_rule': F(1, 4),
    'add_subtract': F(1, 2), 'multiply_divide': F(1), 'compare': F(1, 4),
    'exp': F(1), 'log': F(1), 'sqrt': F(1), 'store': F(1),
}
COMMON = F(53, 20)
REQUEST = COMMON + 4 * COEFFICIENTS['read_scalar']
RESPONSE = (COMMON + COEFFICIENTS['retrieve_rule']
            + COEFFICIENTS['read_scalar'] + COEFFICIENTS['compare'])


def blocks(budget, fee, reports):
    n, k, d = 100, 1001, 100
    # Both signed branches are materialized; this is an abstract scalar DAG.
    rows = [
        ('channel', dict(add_subtract=4*n*n, multiply_divide=3*n*n,
                        compare=n*(n-1), exp=n*n, log=n)),
        ('reached_updates', dict(add_subtract=4*n*reports, compare=(n-1)*reports,
                                exp=n*reports, log=reports)),
        ('stop_probabilities', dict(add_subtract=6*n, compare=2*(n-1), exp=4*n, log=2)),
        ('current_table', dict(multiply_divide=2*k+k*n, add_subtract=2*k*n,
                              sqrt=2*k*n, compare=3*k*n)),
        ('current_scores', dict(multiply_divide=2*k*n,
                               add_subtract=2*k*(n-1)+k, compare=k-1)),
        ('terminal_target_scan', dict(compare=k)),
        ('criterion_price', dict(multiply_divide=6 if budget >= fee else 2,
                                 add_subtract=10 if budget >= fee else 2)),
        ('action_comparison', dict(compare=3 if budget >= fee else 0)),
    ]
    if budget >= fee:
        rows += [
            ('child_table', dict(multiply_divide=2*k+k*n, add_subtract=2*k*n,
                                sqrt=2*k*n, compare=3*k*n)),
            ('predictive_input_normalization', dict(add_subtract=6*n, compare=2*(n-1),
                                                   exp=2*n, log=2)),
            ('predictive_joint_and_posterior', dict(add_subtract=2*n*d+8*n*d,
                                                  compare=2*d*(n-1),
                                                  exp=4*n*d+2*d, log=2*d)),
            ('child_scores', dict(multiply_divide=2*d*k*n+2*k*n,
                                 add_subtract=2*d*k*(n-1)+2*k*(n-1)+2*d*k,
                                 compare=2*d*(k-1))),
            ('report_expectations', dict(multiply_divide=2*d, add_subtract=2*(d-1))),
        ]
    return [{'block': name, 'counts': value} for name, value in rows]


def computation(budget, fee, reports):
    if (type(budget) is not int or budget < 0 or type(fee) is not int or fee <= 0
            or type(reports) is not int or reports < 0):
        raise ValueError('Exact resource, positive question cost and report count')
    rows = blocks(budget, fee, reports)
    counts = {op: sum(row['counts'].get(op, 0) for row in rows) for op in COEFFICIENTS}
    counts['read_scalar'] = (
        2*(counts['add_subtract']+counts['multiply_divide']+counts['compare'])
        + counts['exp']+counts['log']+counts['sqrt']+200+3+(2 if budget >= fee else 1))
    counts['store'] = sum(counts[op] for op in
                          ('add_subtract', 'multiply_divide', 'compare', 'exp', 'log', 'sqrt'))
    counts['retrieve_rule'] = 1+(2 if budget >= fee else 1)
    seconds = sum((COEFFICIENTS[op]*value for op, value in counts.items()), F())
    return dict(blocks=rows, counts=counts, seconds=str(seconds))


def price(root, budget, history, stop, asks):
    rho = F.from_float(float.fromhex(root['costModel']['rhoHex']))
    task = root['task']
    count = sum(map(len, history))
    own = computation(budget, task['cost_millihours'], count)
    future = (computation(budget-task['cost_millihours'], task['cost_millihours'], count+1)
              if asks else None)
    base, scale = 2*F(task['gpa_baseline']), F(task['gpa_scale'])
    return (base+scale*stop-rho*RESPONSE,
            {slot: base+scale*value-rho*(REQUEST+F(future['seconds'])+RESPONSE)
             for slot, value in asks.items()},
            dict(current=own, future=future, requestSeconds=str(REQUEST),
                 responseSeconds=str(RESPONSE), rho=str(rho)))
