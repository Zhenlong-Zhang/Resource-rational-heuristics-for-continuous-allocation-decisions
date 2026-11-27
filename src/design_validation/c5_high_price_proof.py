"""Conditional high-price population projection proof from checked C5 stages.

This is a source-bound, read-only preparation receiver, not an authentication
service for arbitrary files or a proof of PRNG independence. It requires the
original producer and independent checkpoints, not an asserted pass flag.
"""
from fractions import Fraction

from . import c5_moments as M
from . import c5_reference as V
from . import c5_reproduction as C
from . import c5_public_authority as authority
from . import c5_public_inputs as public_inputs
from . import coefficient_io as IO
from . import personal_time_inputs as I
from . import personal_time_sensitivity as S
from .personal_time_reproduction import compare


def _validate_member(root, cfg, high, independent_high, law):
    budget = root['definition']['budgetMillihours']
    I.need(type(budget) is int and budget > 0, 'Positive integer budget required')
    mapping = root['logicalToDisplayed']
    I.need(type(mapping) is list and sorted(mapping) == ['A', 'B'], 'Invalid display mapping')
    I.need(cfg['valuation']['rhoGPAperSecond'] == 0.01, 'High-price configuration required')
    for value in (high, independent_high):
        I.need(type(value) is dict and set(value) == {'decision', 'terminal'}, 'Actual endpoint required')
        C._first_display(value, budget)
        terminal = value['terminal']
        I.need(terminal['scope'] == 'exact_initial_stop_expected_endpoint_not_realized_history'
               and terminal['searchSeconds'] == 0 and terminal['targetHistoryFabricated'] is False
               and terminal['realizedNetObjective'] is None, 'Expected-only initial endpoint required')
    compare(high, independent_high)
    I.need(law['rootKey'] == list(I.coordinates(root))
           and law['rootSHA256'] == I.scientific_digest(root)
           and law['budget'] == budget and law['logicalToDisplayed'] == mapping,
           'Root/law mapping or budget mismatch')
    laws = law['laws']
    I.need(type(laws) is list and len(laws) == 2, 'Two implemented laws required')
    for item in laws:
        M.validate_counts(item['counts'])
        I.need(item['denominator'] == M.GRID and item['lastCategoryIncludesFallback'] is True,
               'Implemented strict-CDF grid law required')
    I.need(laws[0]['counts'] == laws[1]['counts'], 'Recipient integer laws differ')


def _target_member(root, context, row):
    low_cfg = S.rate_config(context, root, 'rho_0.001')
    definition = root['definition']
    I.need(row['rootKey'] == list(I.coordinates(root)) and row['rootSHA256'] == I.digest(root)
           and row['configSHA256'] == I.digest(low_cfg) and row['config'] == low_cfg
           and row['logicalToDisplayed'] == root['logicalToDisplayed']
           and row['reportWidth'] == definition['reportWidth']
           and low_cfg['task']['total_millihours'] == definition['budgetMillihours']
           and low_cfg['task']['cost_millihours'] == definition['costMillihours'],
           'Root or configuration differs from adopted fixed lookup')


def _source_replay(root, cfg, high, independent):
    # A checkpoint's self-hash is not provenance: replay both actual evaluators.
    actual = C._high_source(root, cfg, False)
    reference = C._high_source(root, cfg, True)
    I.need(I.encoded(high) == I.encoded(actual), 'Producer high-price source replay mismatch')
    I.need(I.encoded(independent) == I.encoded(reference), 'Independent high-price source replay mismatch')
    compare(actual, reference)


def _response_moment(root, model):
    """Ordered masses for E[I*z*Y], with Y=1(first display is higher)."""
    a, b = [item['counts'] for item in root['laws']]
    budget = root['budget']
    size = 2 if model == 'gap' else 3
    total = [Fraction(0) for _ in range(size)]
    for first in range(1, 101):
        for second in range(1, first):
            mass = Fraction(a[first-1] * b[second-1], M.GRID**2)
            low, high = Fraction(1000*second, budget), Fraction(1000*first, budget)
            vector = (Fraction(1), high-low) if model == 'gap' else (Fraction(1), low, high)
            for j in range(size):
                total[j] += mass*vector[j]
    return total


def _full_rank(records):
    I.need(len(records) == 90 and {tuple(x['rootKey']) for x in records} == C.ROSTER,
           'Exactly the 90 distinct C5 roots required')
    result = []
    for variant in range(9):
        group = [r for r in records if r['rootKey'][1] == variant]
        moments = [V.unordered_root(r) for r in group]
        models = {}
        for model, size in (('gap', 2), ('partial_low', 3)):
            matrix = [[sum((m[model][i][j] for m in moments), Fraction(0, 1))/10
                       for j in range(size)] for i in range(size)]
            inverse, det = V.cofactor_inverse(matrix)
            I.need(det > 0, 'Singular or invalid %s design in schedule %d' % (model, variant))
            responses = [_response_moment(r, model) for r in group]
            cross = [sum((r[j] for r in responses), Fraction(0))/10 for j in range(size)]
            beta = [sum((inverse[i][j]*cross[j] for j in range(size)), Fraction(0))
                    for i in range(size)]
            I.need(all(cross[i] == matrix[i][0]/2 for i in range(size))
                   and beta == [Fraction(1, 2)] + [Fraction(0)]*(size-1),
                   'High-price response moment or projection mismatch')
            models[model] = {'determinant': M.rational(det),
                             'responseCrossMoment': [M.rational(x) for x in cross],
                             'coefficients': [M.rational(x) for x in beta]}
        result.append({'schedule': variant, 'models': models})
    return result


def prove(output, config, *, adopted_lookup, policy=None):
    """Read all bound P/R stages and derive the conditional zero-slope result.

    Run the existing full C5 produce and check first. The checked policy input,
    when used during preparation, must be supplied again. The identical-law
    stochastic conclusion additionally assumes independent recipient draws;
    distinct PRNG keys alone cannot establish this assumption.
    """
    lookup = public_inputs.load_fixed_inputs(adopted_lookup, authority.ADOPTED_INITIAL_SHA256)
    rows = {tuple(row['rootKey']): row for row in lookup['roots']}
    I.need(set(rows) == C.ROSTER and len(lookup['roots']) == 90, 'Adopted lookup roster required')
    chosen = C.selection(full=True)
    output, context, roots, schedules, run, policy, _ = C._setup(
        output, config, chosen, policy, checking=True)
    I.need(schedules == list(range(9)) and len(roots) == 90, 'Complete C5 source roots required')
    records = []
    for root in roots:
        _target_member(root, context, rows[I.coordinates(root)])
        high, _ = C._high(output, root, context, run, policy, readonly=True)
        independent, _ = C._high(output, root, context, run, policy, independent=True, readonly=True)
        cfg = S.rate_config(context, root, 'rho_0.01')
        _source_replay(root, cfg, high, independent)
        binding = {**C._root_binding(run, root), 'highSHA256': I.digest(high)}
        claimed, _ = IO.stage(output / 'laws' / (C.stem(root) + '.json'), binding, None, readonly=True)
        checked_binding = {**binding, 'producerSHA256': I.digest(claimed),
                           'independentHighSHA256': I.digest(independent)}
        checked, _ = IO.stage(output / 'independent_laws' / (C.stem(root) + '.json'),
                              checked_binding, None, readonly=True)
        calculated = C.law_record(root, independent, independent=True)
        I.need(claimed == checked == calculated, 'Independent integer-law/checkpoint mismatch')
        _validate_member(root, cfg, high, independent, calculated)
        records.append(calculated)
    ranks = _full_rank(records)
    return {'schema': 'c5-high-price-conditional-zero-proof-v1',
            'runSHA256': I.digest(run), 'rootsSHA256': I.digest(records),
            'checkedLawRecords': records,
            'adoptedLookupSHA256': authority.ADOPTED_INITIAL_SHA256,
            'verifiedRoots': len(records), 'scheduleProjections': ranks,
            'highPricePopulationSlopes': {model: M.rational(sum(
                (M.fraction(row['models'][model]['coefficients'][1]) for row in ranks), Fraction(0))/9)
                for model in ('gap', 'partial_low')},
            'assumption': 'Within each root the separately keyed recipient draws are independent and follow the checked integer laws; key separation alone does not prove independence.',
            'scope': 'Conditional population share projections, not realized fits or historical bitwise identity.',
            'newPolicyDraws': 0}
