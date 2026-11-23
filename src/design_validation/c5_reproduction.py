"""Portable C5 finite-law preparation and its independent numerical receiver.

Fresh inputs call the original native root and high-price policy sources.
Alternatively, high-price initial decisions reuse compatible public checked
six-price records. Neither mode draws supplemental outcomes or starts rep1.
"""
from pathlib import Path

import numpy as np

from . import personal_time_inputs as I
from . import personal_time_sensitivity as S
from . import personal_time_sensitivity_streams as P
from . import personal_time_sensitivity_reference as R
from . import budget_horizon as H
from .bounded_task import TaskConfig, normalize_log_weights
from . import c5_moments as M
from . import c5_reference as V
from . import c5_streams as K
from . import coefficient_io as IO
from .personal_time_reproduction import compare, csv_bytes

ROSTER = {('C5', v, m) for v in range(9) for m in range(1, 11)}


def selection(*, full=False, only=None):
    if full:
        I.need(only is None, 'Choose full or only')
        return I.selection([dict(zip(I.FIELDS, key)) for key in sorted(ROSTER)])
    chosen = I.select_roots(only=only)
    I.need(all(I.coordinates(x) in ROSTER for x in chosen), 'C5 roots only')
    return chosen


def stem(root):
    return '%s_v%d_m%d' % I.coordinates(root)


def _first_display(value, budget):
    chosen, terminal = value['decision'], value['terminal']
    I.need(chosen['action'] == chosen['stop']['action'] == terminal['action'] == 'MFN_A'
           and chosen['stop']['index'] == terminal['intendedIndex'] == 1000
           and terminal['budget'] == budget and terminal['newScientificDraws'] == 0
           and terminal['realizedCombinedGPA'] is None, 'High-price first-display/no-question source required')
    return True


def _high_source(root, cfg, independent):
    kernel = R if independent else P
    state = {'budget': root['definition']['budgetMillihours'],
             'logHex': [[float(x).hex() for x in row] for row in root['initialLogs']],
             'counts': [0, 0], 'queryIndex': 0, 'searchSecondsHex': (0.).hex()}
    task = H.BudgetTask(TaskConfig(**cfg['task']))
    decision = kernel.decision(cfg, task, state)
    I.need(decision['action'] not in ('ask_A', 'ask_B', 'random'), 'Expected-only initial decision')
    terminal = kernel.expected_terminal(cfg, state, decision['stop'])
    H.native_utility_table.cache_clear()
    H.native_perfect_information.cache_clear()
    return dict(decision=decision, terminal=terminal)


def _checked_high(policy, root, context, independent):
    path_root, run, roots, checked = (policy[key] for key in ('path', 'run', 'roots', 'checked'))
    I.need(I.coordinates(root) in roots and I.scientific_digest(roots[I.coordinates(root)])
           == I.scientific_digest(root), 'Checked policy/native root mismatch')
    cfg = S.rate_config(context, root, 'rho_0.01')
    # Root transport fields may differ; use the actual authenticated policy root.
    source_root = roots[I.coordinates(root)]
    binding = S.record_binding(source_root, cfg, run, 'rho_0.01')
    role = 'independent' if independent else 'producer'
    path = S.record_path(path_root, 'rho_0.01', root, role)
    record = I.read(path)
    producer_path = S.record_path(path_root, 'rho_0.01', root)
    if independent:
        I.need(type(record) is dict and set(record) == {
            'schema', 'binding', 'producerSHA256', 'payload', 'absoluteTolerance', 'sharedKernels'},
            'Exact independent high-price checkpoint')
        I.need(record['schema'] == 'personal-time-sensitivity-independent-trial-v1'
               and record['binding'] == binding and record['producerSHA256'] == S.file_sha(producer_path)
               and record['absoluteTolerance'] == 1e-10
               and record['sharedKernels'] == ['bounded_task', 'budget_horizon', 'terminal_time',
                                              'authenticated_input_adapter_and_event_traversal'],
               'Independent high-price linkage')
        S.validate_record(S.envelope(binding, record['payload']), binding)
    else:
        S.validate_record(record, binding)
    I.need(S.file_sha(producer_path) == checked['producerInputs'][str(producer_path.relative_to(path_root))],
           'Changed checked producer input')
    payload = record['payload']
    I.need(payload['endpointKind'] == 'expected_only'
           and payload['finalState']['queryIndex'] == 0, 'No fabricated high-price path')
    return dict(decision=payload['initialDecision'], terminal=payload['terminal'])


def law_record(root, high, *, independent=False):
    mapping = root['logicalToDisplayed']
    I.need(type(mapping) is list and sorted(mapping) == ['A', 'B'], 'Exact display/logical mapping')
    fn = V.finite_law if independent else M.finite_law
    laws = [fn(logs, np, normalize_log_weights) for logs in root['initialLogs']]
    same = laws[0]['counts'] == laws[1]['counts']
    first = _first_display(high, root['definition']['budgetMillihours'])
    return dict(rootKey=list(I.coordinates(root)), rootSHA256=I.scientific_digest(root),
                budget=root['definition']['budgetMillihours'], logicalToDisplayed=mapping,
                rawLogHexEqual=([float(x).hex() for x in root['initialLogs'][0]] ==
                                [float(x).hex() for x in root['initialLogs'][1]]),
                laws=laws, recipientCountsExactlyEqual=same,
                highPriceFirstDisplayNoQuestion=first, symmetryCertified=same and first,
                jointLawAssumption='independent recipient finite draws under separately keyed PRNG model',
                futureSamplerRequirement='actual integer cutoffs must exactly equal this law before any policy draw')


def _setup(output, config, selected, policy, *, checking=False, node='node'):
    output = S.safe_output(output)
    context = S.load_context(config)
    I.reference_runtime()
    chosen = I.selection(selected)
    I.need(all(I.coordinates(x) in ROSTER for x in chosen), 'C5 root scope')
    if policy is not None:
        policy, _ = IO.separate(policy, output)
        prior_run = I.read(policy / 'run.json')
        prior_roots, actual_run, _ = S.load_roots(context, prior_run['selection'], policy)
        I.need(actual_run == prior_run, 'Checked policy run binding')
        checked = I.read(policy / 'independent_check.json')
        I.need(checked['schema'] == 'personal-time-sensitivity-independent-check-v1'
               and checked['binding'] == prior_run and checked['completeIndependentReconstruction'] is True
               and checked['verifiedTrials'] == 6*len(prior_roots), 'Complete checked source selection')
        inputs = {}
        for row in chosen:
            for role in ('producer', 'independent'):
                path = S.record_path(policy, 'rho_0.01', row, role)
                relative = str(path.relative_to(policy))
                inputs[relative] = S.file_sha(path)
                if role == 'producer':
                    I.need(inputs[relative] == checked['producerInputs'][relative], 'Changed checked producer')
        policy_identity = {'run': S.file_sha(policy / 'run.json'),
                           'check': S.file_sha(policy / 'independent_check.json'), 'inputs': inputs}
        policy = dict(path=policy, run=prior_run, checked=checked,
                      roots={I.coordinates(x): x for x in prior_roots})
    else:
        policy_identity = None
    binding = dict(schema='c5-finite-preparation-v1', selection=chosen,
                   contextSHA256=I.digest(context), sources=IO.source_identity(),
                   runtime=S.runtime_identity(), checkedPolicy=policy_identity,
                   highSource='checked_public_reuse' if policy else 'fresh_public_initial_decisions',
                   law='strict_CDF_53_bit_uniform_grid_with_category100_fallback',
                   looks=list(M.LOOKS), alpha=['1', '20'], radius=['1', '10'],
                   estimand='equal_nine_conditional_nontie_population_linear_projections')
    if checking:
        I.need(I.read(output / 'run.json') == binding, 'Current C5 source/input/runtime binding')
        roots = I.load_roots(context['reference'], chosen, output / 'native')
        generated = 0
    else:
        I.write(output / 'run.json', binding)
        generated = 0 if (output / 'native/inputs/roots.json').exists() else len(chosen)
        roots = I.prepare(context['reference'], chosen, output / 'native', node)['roots']
    schedules = [v for v in range(9) if {I.coordinates(r) for r in roots if r['scheduleVariant'] == v}
                 == {('C5', v, m) for m in range(1, 11)}]
    expected = {folder + '/' + stem(r) + '.json' for r in roots
                for folder in ('laws', 'independent_laws', 'high', 'independent_high')}
    expected |= {folder + '/C5_v%d.json' % v for v in schedules
                 for folder in ('designs', 'independent_designs')}
    IO.inventory(output, expected)
    return output, context, roots, schedules, binding, policy, generated


def _root_binding(run, root):
    return dict(runSHA256=I.digest(run), rootSHA256=I.scientific_digest(root),
                coordinate=list(I.coordinates(root)))


def _high(output, root, context, run, policy, *, independent=False, readonly=False):
    folder = 'independent_high' if independent else 'high'
    cfg = S.rate_config(context, root, 'rho_0.01')
    bound = {**_root_binding(run, root), 'configSHA256': I.digest(cfg), 'independent': independent}
    def calculate():
        return (_checked_high(policy, root, context, independent) if policy else
                _high_source(root, cfg, independent))
    value, new = IO.stage(output / folder / (stem(root)+'.json'), bound, calculate, readonly=readonly)
    _first_display(value, root['definition']['budgetMillihours'])
    return value, new


def produce(output, config, selected, *, policy=None, node='node'):
    output, context, roots, schedules, run, policy, generated = _setup(
        output, config, selected, policy, node=node)
    counts = dict(newNativeRoots=generated, newInitialPolicyEvaluations=0, newRootLaws=0,
                  newDesigns=0, newScientificDraws=0, newCoefficientFits=0)
    records = []
    for root in roots:
        high, new_high = _high(output, root, context, run, policy)
        counts['newInitialPolicyEvaluations'] += int(new_high and policy is None)
        bound = {**_root_binding(run, root), 'highSHA256': I.digest(high)}
        law, new = IO.stage(output / 'laws' / (stem(root)+'.json'), bound,
                            lambda: law_record(root, high))
        counts['newRootLaws'] += int(new)
        records.append(law)
    designs = []
    for variant in schedules:
        group = [r for r in records if r['rootKey'][1] == variant]
        bound = dict(runSHA256=I.digest(run), lawsSHA256=I.digest(group), schedule=variant)
        design, new = IO.stage(output / 'designs' / ('C5_v%d.json' % variant), bound,
                               lambda: M.design_payload(group))
        counts['newDesigns'] += 2*int(new)
        designs.append(dict(schedule=variant, payload=design))
    full = {tuple(r['rootKey']) for r in records} == ROSTER
    symmetry = full and all(r['symmetryCertified'] for r in records)
    bound = dict(runSHA256=I.digest(run), designsSHA256=I.digest(designs),
                 lawsSHA256=I.digest(records))
    summary, _ = IO.stage(output / 'summary.json', bound, lambda: M.summarize_design(designs, symmetry))
    IO.stage(output / 'domain.json', dict(runSHA256=I.digest(run)), lambda: K.audit(roots, context))
    return dict(output=str(output), fullFiniteTarget=full, counters=counts,
                floorFeasible=summary['floorFeasible'], policyDrawsReleased=False)


def check(output, config, selected, *, policy=None):
    output, context, roots, schedules, run, policy, _ = _setup(
        output, config, selected, policy, checking=True)
    records, designs = [], []
    counts = dict(newInitialPolicyEvaluations=0, newIndependentRootLaws=0,
                  newIndependentDesigns=0, newScientificDraws=0, newCoefficientFits=0)
    for root in roots:
        high, _ = _high(output, root, context, run, policy, readonly=True)
        reference_high, new = _high(output, root, context, run, policy, independent=True)
        counts['newInitialPolicyEvaluations'] += int(new and policy is None)
        compare(high, reference_high)
        bound = {**_root_binding(run, root), 'highSHA256': I.digest(high)}
        claimed, _ = IO.stage(output / 'laws' / (stem(root)+'.json'), bound, None, readonly=True)
        checked_bound = {**bound, 'producerSHA256': I.digest(claimed),
                         'independentHighSHA256': I.digest(reference_high)}
        reference, new = IO.stage(output / 'independent_laws' / (stem(root)+'.json'), checked_bound,
                                  lambda: law_record(root, reference_high, independent=True))
        I.need(claimed == reference, 'Independent full law/symmetry mismatch')
        counts['newIndependentRootLaws'] += int(new)
        records.append(reference)
    for variant in schedules:
        group = [r for r in records if r['rootKey'][1] == variant]
        bound = dict(runSHA256=I.digest(run), lawsSHA256=I.digest(group), schedule=variant)
        claimed, _ = IO.stage(output / 'designs' / ('C5_v%d.json' % variant), bound, None, readonly=True)
        reference, new = IO.stage(output / 'independent_designs' / ('C5_v%d.json' % variant),
                                  {**bound, 'producerSHA256': I.digest(claimed)},
                                  lambda: V.design_reference(group))
        I.need(reference == claimed, 'Independent G/inverse/signed-range mismatch')
        counts['newIndependentDesigns'] += 2*int(new)
        designs.append(dict(schedule=variant, payload=reference))
    full = {tuple(r['rootKey']) for r in records} == ROSTER
    symmetry = full and all(r['symmetryCertified'] for r in records)
    bound = dict(runSHA256=I.digest(run), designsSHA256=I.digest(designs), lawsSHA256=I.digest(records))
    summary, _ = IO.stage(output / 'summary.json', bound, None, readonly=True)
    V.verify_summary(summary, [d['payload'] for d in designs], symmetry)
    domain, _ = IO.stage(output / 'domain.json', dict(runSHA256=I.digest(run)), None, readonly=True)
    I.need(domain == K.audit(roots, context), 'Independent source key-domain admission')
    result = dict(passed=True, verifiedRoots=len(roots), verifiedDesigns=2*len(schedules),
                  fullFiniteTarget=full, symmetryCertified=symmetry, summary=summary,
                  sourceSemantics='public_source_epoch_not_historical_receipts',
                  highSource=run['highSource'], policyDrawsReleased=False,
                  precisionEstablished=False, wholeReplicateRNGReady=False)
    I.write(output / 'independent_check.json', result)
    _tables(output, designs, summary)
    return dict(output=str(output), counters=counts, **result)


def _tables(output, designs, summary):
    rows = []
    for design in designs:
        for model in M.MODELS:
            for root in design['payload'][model]['rootRanges']:
                rows.append(dict(schedule=design['schedule'], model=model,
                                 root=':'.join(map(str, root['rootKey'])),
                                 lower='/'.join(root['singlePrice'][0]),
                                 upper='/'.join(root['singlePrice'][1])))
    if rows:
        I.write_bytes(output / 'signed_root_ranges.csv', csv_bytes(rows, tuple(rows[0])))
    rows = [dict(model=row['model'], replicates=floor['replicates'],
                 lower='/'.join(floor['lower']), upper='/'.join(floor['upper']), status=floor['status'])
            for row in summary['ranges'] for floor in row['zeroVarianceFloors']]
    if rows:
        I.write_bytes(output / 'original_eb_floor.csv', csv_bytes(rows, tuple(rows[0])))
