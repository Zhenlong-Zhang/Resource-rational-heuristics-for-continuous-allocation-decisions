"""Finite allocation diagnostics and target-key fixed outcomes of the proxy model.

The policy is not rerun here. Inputs are the checked six-price reproduction.
An outcome reconstructed for an initial stop is a fixed model evaluation, not a
historical observation or an additional policy trajectory.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import personal_time_inputs as I
from . import personal_time_sensitivity as S
from . import personal_time_sensitivity_streams as P
from . import personal_time_sensitivity_reference as R
from .bounded_task import normalize_log_weights
from .personal_time_reproduction import compare, csv_bytes

FIELDS = ('expected_u', 'realized_u', 'initial_ask', 'mean_paid',
          'ES_pp', 'EO_pp', 'OPT_pp', 'AL_pp')
RECIPE = {
    'schema': 'finite-allocation-welfare-v1',
    'policyRecipe': 'personal_time_sensitivity.json',
    'relativeNeedTieTolerance': 1e-9,
    'slopeCenteredSSFloor': 1e-12,
    'allocationDenominator': 1000,
    'aggregation': 'ten_trials_then_nine_equal_weight_schedules',
    'payoff': 'recipient_utility_without_personal_time_subtraction',
    'latentLaw': 'original_initial_normalized_logs',
    'scope': 'fixed_finite_roster_not_population_or_human_inference',
}


def average(values):
    I.need(bool(values) and all(math.isfinite(x) for x in values), 'Finite nonempty mean')
    return math.fsum(values) / len(values)


def slope(points, floor=1e-12):
    if len(points) < 2:
        return None
    mx, my = (average(list(v)) for v in zip(*points))
    ss = math.fsum((x - mx)**2 for x, _ in points)
    return math.fsum((x-mx)*(y-my) for x, y in points) / ss if ss > floor else None


def fixed_utility(budget, index, needs, config, *, independent=False):
    """Evaluate the saved integer allocation at the same target's true needs."""
    I.need(type(budget) is int and budget > 0 and type(index) is int
           and 0 <= index <= 1000, 'Positive integer budget and native index')
    I.need(len(needs) == 2 and all(type(n) is int and 1 <= n <= 100 for n in needs),
           'Two target-specific finite needs')
    task = config['task']
    if independent:
        values = [(budget*j - n*1000000) / 1000000
                  for j, n in zip((index, 1000-index), needs)]
        total = math.fsum(math.pow(x, task['alpha']) if x >= 0 else
                         -task['loss_multiplier']*math.pow(-x, task['alpha']) for x in values)
    else:
        x = np.asarray([budget*index-needs[0]*1000000,
                        budget*(1000-index)-needs[1]*1000000], dtype=np.int64) / 1000000
        u = np.where(x >= 0, np.maximum(x, 0)**task['alpha'],
                     -task['loss_multiplier']*np.maximum(-x, 0)**task['alpha'])
        total = float(u[0] + u[1])
    gpa = 2*task['gpa_baseline'] + task['gpa_scale']*total
    # Preserve the reported GPA-to-utility arithmetic, not just its algebra.
    return gpa, (gpa - 2*task['gpa_baseline']) / task['gpa_scale']


def target_latents(root, context, *, independent=False):
    kernels = R if independent else P
    order = root['logicalToDisplayed']
    I.need(sorted(order) == ['A', 'B'], 'Logical/displayed permutation')
    draws = {}
    for logical in ('A', 'B'):
        law = normalize_log_weights(np.asarray(root['initialLogs'][order.index(logical)]))
        draws[logical] = kernels.stream_draw(kernels.make_key(context, logical, 'latent', 0), law)
    return draws


def row(root, label, config, payload, *, independent=False):
    """Reduce one checked endpoint; optional reference arithmetic checks the port."""
    c, v, m = I.coordinates(root)
    I.need(label in dict(S.RATES), 'Frozen price')
    ctx, state, terminal, metrics = (payload[k] for k in ('context', 'finalState', 'terminal', 'metrics'))
    ids = ctx['identity']
    I.need((ids['conditionId'], ids['variant'], ids['mainIndex']) == (c, v, m)
           and ids['replicateId'] == 0 and ids['seed'] == S.RECIPE['seed'], 'Target identity')
    I.need(ids == P.stable_identity(config, S.RECIPE['seed'], S.RECIPE['populationId'], c, v, 0, m),
           'Source-derived person and trial identity')
    I.need(ctx['bindings'] == {k: S.RECIPE[k] for k in ('populationId', 'randomizationGroup')},
           'Original stream namespaces')
    I.need(metrics['choiceAvailable'] and terminal['budget'] == state['budget']
           == metrics['remainingIntegerBudget'], 'Same positive terminal budget')
    j, budget = terminal['intendedIndex'], state['budget']
    I.need(j == metrics['shareIndex'] and metrics['share'] == j/1000, 'Delivered index and share')
    initial_budget = root['definition']['budgetMillihours']
    I.need(initial_budget == metrics['initialIntegerBudget'] == config['task']['total_millihours'],
           'Original initial-budget denominator')
    I.need(state['queryIndex'] == metrics['paidCount'] and metrics['Q'] == int(state['queryIndex'] > 0),
           'Paid-question metric linkage')
    last = [e['value']['value'] for e in payload['events'] if e['name'].startswith('decision_')][-1]
    stop = last['stop']
    I.need(stop['action'] == terminal['action'] and stop['index'] == j,
           'Fixed deterministic terminal decision')
    targets = {name: int(index) for index, names in stop['legacySelection']['allocation_classes'].items()
               for name in names}
    I.need(targets['EO_raw_surplus'] == metrics['jSurplus'] and targets['AL'] == metrics['jAL'],
           'Matched terminal benchmark indices')
    draws = target_latents(root, ctx, independent=independent)
    needs = [draws[logical]['category'] for logical in root['logicalToDisplayed']]
    gpa, u = fixed_utility(budget, j, needs, config, independent=independent)
    if payload['endpointKind'] == 'realized_trajectory':
        saved = {e['name'][7:]: e['value'] for e in payload['events'] if e['name'].startswith('latent_')}
        I.need(set(saved) == {'A', 'B'}, 'Both original latent events')
        for logical in saved:
            for field in ('key', 'keySHA256', 'uniformHex', 'category', 'roundingFallback'):
                I.need(saved[logical][field] == draws[logical][field], 'Target latent ' + field)
        compare(gpa, terminal['realizedCombinedGPA'], 'same_target_trajectory_payoff')
    else:
        I.need(payload['endpointKind'] == 'expected_only'
               and terminal['realizedCombinedGPA'] is None and state['queryIndex'] == 0,
               'Expected-only source is not relabeled a historical trajectory')
    initial = metrics['initialMeans']
    compare(initial, [P.G.finite_mean(logs) for logs in root['initialLogs']], 'target_initial_means')
    gap = initial[0] - initial[1]
    I.need(metrics['rawInitialGap'] == gap, 'Initial mean gap linkage')
    compare(metrics['x0'], gap/(initial_budget/1000), 'initial_normalized_gap')
    tied = abs(gap) <= RECIPE['relativeNeedTieTolerance'] * max(1., *(abs(x) for x in initial))
    means = dict(zip(root['logicalToDisplayed'], initial))
    logical_a_index = j if root['logicalToDisplayed'][0] == 'A' else 1000-j
    return {
        'condition': c, 'variant': v, 'main': m, 'price': label,
        'trial_id': ids['trialId'],
        'rhoGPAperSecond': dict(S.RATES)[label], 'budget_milli': budget,
        'initial_budget_milli': initial_budget, 'index': j, 'share': j/1000,
        'logical_order': list(root['logicalToDisplayed']),
        'needs_displayed': needs,
        'latent_keys': {k: draws[k]['key'] for k in draws},
        'latent_uniforms': {k: draws[k]['uniformHex'] for k in draws},
        'latent_categories': {k: draws[k]['category'] for k in draws},
        'need_a': means['A'], 'need_b': means['B'], 'amount_a': budget*logical_a_index/1000000,
        'rank_x': None if tied else abs(metrics['x0']),
        'rank_y': None if tied else (j/1000 if gap > 0 else 1-j/1000),
        'expected_u': (terminal['actionExpectedCombinedGPA']-2*config['task']['gpa_baseline'])/config['task']['gpa_scale'],
        'realized_gpa': gpa, 'realized_u': u,
        'initial_ask': int(payload['initialDecision']['action'] in ('ask_A', 'ask_B')),
        'mean_paid': state['queryIndex'],
        'ES_pp': abs(j-500)/10,
        'EO_pp': abs(j-targets['EO_raw_surplus'])/10,
        'OPT_pp': abs(j-targets['OPT_expected_GPA'])/10,
        'AL_pp': abs(j-targets['AL'])/10,
        'sourceEndpointKind': payload['endpointKind'],
        'payoffStatus': 'reconstructed_same_target_fixed_outcome',
        'historicalEventClaimed': False,
    }


def summarize(rows, *, full=False, independent=False):
    groups, seen = defaultdict(list), set()
    for value in rows:
        coord = (value['condition'], value['variant'], value['main'])
        key = (value['price'], *coord)
        I.need(coord in I.ORIGINAL and value['price'] in dict(S.RATES) and key not in seen,
               'Unique original endpoint roster')
        seen.add(key)
        I.need(all(type(value[k]) in (int, float) and math.isfinite(value[k]) for k in FIELDS),
               'No unavailable or nonfinite scalar may be silently omitted')
        I.need((value['rank_x'] is None) == (value['rank_y'] is None), 'Paired rank availability')
        groups[value['condition'], value['price'], value['variant']].append(value)
    expected = {(label, *coord) for label, _ in S.RATES for coord in I.ORIGINAL}
    if full:
        I.need(seen == expected, 'Full fixed 2700-endpoint roster required')
    people, partial = [], []
    for (c, label, v), group in sorted(groups.items()):
        group.sort(key=lambda item: item['main'])
        if [r['main'] for r in group] != list(range(1, 11)):
            partial.append({'condition': c, 'price': label, 'variant': v, 'present': len(group)})
            continue
        points = [(r['rank_x'], r['rank_y']) for r in group if r['rank_x'] is not None]
        beta = slope(points)
        if independent and beta is not None:
            x, y = np.asarray(points, dtype=float).T
            xc = x - np.mean(x)
            beta = float(np.linalg.lstsq(xc[:, None], y - np.mean(y), rcond=None)[0][0])
        reducer = (lambda x: float(np.mean(np.asarray(x)))) if independent else average
        people.append({'condition': c, 'price': label, 'variant': v, 'beta_high_need': beta,
                       'rank_eligible': len(points), **{k: reducer([r[k] for r in group]) for k in FIELDS}})
    conditions = []
    for c in I.CONDITIONS:
        for label, _ in S.RATES:
            group = [p for p in people if (p['condition'], p['price']) == (c, label)]
            if len(group) != 9:
                continue
            I.need({p['variant'] for p in group} == set(range(9)), 'Nine equally weighted schedules')
            betas = [p['beta_high_need'] for p in group if p['beta_high_need'] is not None]
            conditions.append({'condition': c, 'price': label, 'endpoints': 90,
                'rank_eligible': sum(p['rank_eligible'] for p in group), 'defined_schedules': len(betas),
                'beta_high_need': average(betas) if len(betas) == 9 else None,
                **{k: average([p[k] for p in group]) for k in FIELDS}})
    return {'schema': RECIPE['schema'], 'fullFiniteTarget': seen == expected,
            'verifiedTrials': len(seen), 'persons': people, 'conditions': conditions,
            'partialPersons': partial, 'historicalEventClaimed': False,
            'policyRerunByAnalysis': False, 'monteCarloPrecisionEstablished': False}


def analyze(input_path, output_path, config_path):
    """Authenticate existing public records before reducing fixed outcomes."""
    input_path, output = Path(input_path).resolve(), S.safe_output(output_path)
    I.need(not input_path.is_relative_to(output.resolve()) and not output.resolve().is_relative_to(input_path),
           'Analysis output must be separate from policy input')
    recipe = I.read(config_path)
    I.need(recipe == RECIPE, 'Frozen allocation/welfare recipe')
    context = S.load_context(Path(config_path).parent / recipe['policyRecipe'])
    selection = I.read(input_path / 'run.json')['selection']
    roots, run, _ = S.load_roots(context, selection, input_path)
    checked = I.read(input_path / 'independent_check.json')
    I.need(checked['binding'] == run and checked['completeIndependentReconstruction'] is True,
           'Completed policy independent reconstruction required')
    expected = {str(S.record_path(input_path, label, root).relative_to(input_path))
                for root in roots for label, _ in S.RATES}
    I.need(set(checked['producerInputs']) == expected and checked['verifiedTrials'] == len(expected),
           'Exactly the requested checked input roster')
    rows, reference_rows, pins = [], [], {}
    for root in roots:
        for label, _ in S.RATES:
            path = S.record_path(input_path, label, root)
            rel = str(path.relative_to(input_path))
            sha = S.file_sha(path)
            I.need(sha == checked['producerInputs'][rel], 'Changed checked policy record')
            cfg = S.rate_config(context, root, label)
            binding = S.record_binding(root, cfg, run, label)
            record = I.read(path)
            S.validate_record(record, binding)
            other_path = S.record_path(input_path, label, root, 'independent')
            other = I.read(other_path)
            I.need(other['producerSHA256'] == sha and other['binding'] == binding
                   and other['absoluteTolerance'] == 1e-10, 'Independent record linkage')
            compare(record['payload'], other['payload'], 'checked_policy_payload')
            actual = row(root, label, cfg, record['payload'])
            reference = row(root, label, cfg, other['payload'], independent=True)
            compare(actual, reference, 'independent_fixed_outcome')
            rows.append(actual)
            reference_rows.append(reference)
            pins[rel] = sha
            pins[str(other_path.relative_to(input_path))] = S.file_sha(other_path)
    full = {I.coordinates(root) for root in roots} == I.ORIGINAL
    summary = summarize(rows, full=full)
    independent_summary = summarize(reference_rows, full=full, independent=True)
    compare(summary, independent_summary, 'independent_allocation_reduction')
    own = {p.name: S.file_sha(p) for p in (Path(__file__), Path(config_path))}
    I.write(output / 'trial_metrics.json', rows)
    I.write(output / 'summary.json', summary)
    I.write(output / 'independent_check.json', {'recipe': recipe, 'sources': own,
        'policyCheckSHA256': S.file_sha(input_path / 'independent_check.json'),
        'policyInputs': pins, 'verifiedTrials': len(rows), 'fullFiniteTarget': full,
        'absoluteTolerance': 1e-10, 'independentArithmetic': 'scalar_power_and_centered_least_squares',
        'sharedKernels': ['native_law_normalization', 'PCG64', 'record_admission', 'group_traversal'],
        'historicalProvenanceReconstructed': False, 'newPolicySimulations': 0})
    if summary['conditions']:
        table = summary['conditions']
        I.write_bytes(output / 'allocation_welfare.csv', csv_bytes(table, tuple(table[0])))
    amount = [{'rate': r['price'], 'condition': r['condition'], 'variant': r['variant'],
               'main': r['main'], 'person': '%s_v%d' % (r['condition'], r['variant']),
               'price': r['rhoGPAperSecond'], 'trial_id': r['trial_id'],
               **{k: r[k] for k in ('need_a', 'need_b', 'amount_a', 'budget_milli')}} for r in rows]
    I.write_bytes(output / 'initial_need_amounts.csv', csv_bytes(amount, tuple(amount[0])))
    return {'verifiedTrials': len(rows), 'fullFiniteTarget': full, 'output': str(output)}
