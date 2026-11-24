"""Actual terminal routines and recipient questions from checked proxy endpoints.

No policy, latent draw, fit or utility evaluation is performed. Pure extraction
checks saved state/event consistency; only analyze admits source-bound records.
Missing endpoints are UNKNOWN, never fabricated zero-question observations.
Scope is the original deterministic terminal-menu roster: a realized random
terminal endpoint (including a zero-question random endpoint) is not supported.
"""
from collections import Counter
from pathlib import Path

from . import personal_time_inputs as I
from . import personal_time_sensitivity as S
from . import true_need_reproduction as T
from .personal_time_reproduction import csv_bytes
from .terminal_time import MACROS

SCHEMA = 'paper-action-recipient-tables-v1'
COUNTS = ('total_questions', 'slot_0_questions', 'slot_1_questions',
          'logical_A_questions', 'logical_B_questions')
ROW_FIELDS = ('condition', 'variant', 'main', 'price', 'trial_id', 'rootSHA256',
              'logical_order', 'endpoint_kind', 'initial_action', 'terminal_action',
              'selected_rule', 'index', 'budget_milli', *COUNTS)
TABLE_FILES = ('trial_actions.json', 'summary.json', 'action_endpoints.csv', 'action_conditions.csv')
CHECK_FIELDS = {'schema', 'sources', 'policyRunSHA256', 'policyCheckSHA256',
                'policyContentSHA256', 'policyBindingSHA256', 'policyInputs', 'rootInputs',
                'outputFiles', 'verifiedTrials', 'fullFiniteTarget', 'newPolicySimulations',
                'historicalProvenanceReconstructed', 'payloadAbsoluteTolerance', 'extractedFieldsComparison'}


def _integer(value, label, lower=0, upper=None):
    I.need(type(value) is int and value >= lower and (upper is None or value <= upper), label)
    return value


def _counts(value):
    I.need(type(value) is list and len(value) == 2, 'Two explicit displayed-slot counts required')
    return [_integer(n, 'Nonnegative integer question count') for n in value]


def _same(actual, expected, label):
    I.need(I.encoded(actual) == I.encoded(expected), label)


def _coordinate(c, v, m):
    I.need(type(c) is str and type(v) is int and type(m) is int
           and (c, v, m) in I.ORIGINAL, 'Strict original root coordinate required')
    return c, v, m


def _order(value):
    I.need(type(value) is list and len(value) == 2 and all(type(x) is str for x in value)
           and sorted(value) == ['A', 'B'],
           'Explicit displayed-slot to logical A/B permutation required')
    return list(value)


def extract(root, label, payload):
    """Read one saved endpoint; this alone does not authenticate its provenance.

    Despite its historical name, root.logicalToDisplayed lists the logical
    recipient occupying displayed slot0 and slot1, in that order.
    """
    try:
        c, v, m = _coordinate(*I.coordinates(root))
        I.need(type(label) is str and label in dict(S.RATES), 'Frozen six-price label required')
        order = _order(root['logicalToDisplayed'])
        context, events = payload['context'], payload['events']
        _same(context['rootSHA256'], I.digest(root), 'Payload/root identity mismatch')
        ids = context['identity']
        _same([ids['conditionId'], ids['variant'], ids['mainIndex']], [c, v, m], 'Endpoint coordinate mismatch')
        _same([ids['seed'], ids['replicateId']], [S.RECIPE['seed'], 0], 'Original seed/replicate required')
        I.need(type(ids['trialId']) is str and bool(ids['trialId']), 'Explicit trial identity required')
        _same(context['bindings'], {k: S.RECIPE[k] for k in ('populationId', 'randomizationGroup')},
              'Original stream namespaces required')
        state, metrics, terminal = (payload[k] for k in ('finalState', 'metrics', 'terminal'))
        counts = _counts(state['counts'])
        q = _integer(state['queryIndex'], 'Explicit nonnegative query index')
        _same(_counts(metrics['counts']), counts, 'Metric/state displayed counts differ')
        _same([metrics['paidCount'], metrics['Q']], [q, int(q > 0)], 'Paid count and Q linkage')
        I.need(sum(counts) == q, 'Displayed counts must exhaust total questions')
        kind = payload['endpointKind']
        I.need(kind in ('expected_only', 'realized_trajectory'), 'Known endpoint kind required')
        if kind == 'expected_only':
            I.need(q == 0 and terminal['realizedCombinedGPA'] is None
                   and terminal['newScientificDraws'] == 0, 'Expected-only endpoint must have no questions/draws')
            names = ['decision_000000', 'expected_terminal', 'metrics']
        else:
            I.need(q > 0 and terminal['realizedCombinedGPA'] is not None, 'Paid trajectory required')
            names = ['decision_000000', 'latent_A', 'latent_B']
            for k in range(q):
                names.extend(['report_%06d' % k, 'paid_%06d' % k, 'decision_%06d' % (k+1)])
            names.extend(['terminal', 'metrics'])
        I.need(type(events) is list and all(type(e) is dict and set(e) == {'name', 'value'} for e in events),
               'Exact saved event schema required')
        I.need([e['name'] for e in events] == names, 'Missing, duplicate, reordered or foreign saved events')
        by_name = {e['name']: e['value'] for e in events}
        _same(events[-1]['value'], metrics, 'Saved metrics event linkage')
        _same(events[-2]['value'], terminal, 'Saved terminal event linkage')
        first = by_name['decision_000000']
        _same(first['value'], payload['initialDecision'], 'Initial decision event linkage')
        before = first['before']
        _same([_counts(before['counts']), before['queryIndex']], [[0, 0], 0], 'Zero-count initial state required')
        initial_budget = _integer(root['definition']['budgetMillihours'], 'Positive initial budget', 1)
        _same([before['budget'], metrics['initialIntegerBudget']], [initial_budget]*2, 'Initial budget linkage')
        for k in range(q):
            decision = by_name['decision_%06d' % k]
            paid = by_name['paid_%06d' % k]
            report = by_name['report_%06d' % k]
            slot = _integer(paid['slot'], 'Displayed paid slot must be 0 or 1', 0, 1)
            _same(decision['before'], before, 'Decision/previous paid-state linkage')
            _same(paid['before'], before, 'Paid before-state linkage')
            I.need(decision['value']['action'] == ('ask_A', 'ask_B')[slot], 'Question action/slot mismatch')
            _same(report['key'], S.P.make_key(context, order[slot], 'report', before['counts'][slot]),
                  'Report key must match logical recipient and per-slot report index')
            _integer(report['category'], 'Finite report category', 1, 100)
            _same(paid['report'], report['category'], 'Paid report linkage')
            after = paid['after']
            updated = list(before['counts'])
            updated[slot] += 1
            _same([_counts(after['counts']), after['queryIndex']], [updated, k+1], 'Exactly one question at the paid slot')
            _integer(after['budget'], 'Nonnegative paid-after budget', 0, before['budget'])
            before = after
        final = by_name['decision_%06d' % q]
        _same(final['before'], before, 'Final decision/paid-state linkage')
        _same(state, before, 'Final state must close the saved event chain')
        fields = T.terminal_fields(payload)
        I.need(fields['terminal_action'] in MACROS, 'Known actual deterministic terminal routine required')
        _same([final['value']['action'], metrics['action']], [fields['terminal_action']]*2,
              'Final action/metric/routine linkage')
        index = _integer(terminal['intendedIndex'], 'Native terminal index', 0, 1000)
        budget = _integer(state['budget'], 'Positive final tutoring budget', 1, initial_budget)
        _same([terminal['budget'], metrics['remainingIntegerBudget']], [budget]*2, 'Final budget linkage')
        _same(metrics['shareIndex'], index, 'Delivered index linkage')
        I.need(payload['initialDecision']['action'] in ('ask_A', 'ask_B') if q else
               payload['initialDecision']['action'] == fields['terminal_action'], 'Initial search/total questions linkage')
        if label == 'rho_0.01':
            I.need(kind == 'expected_only', 'Highest-price original endpoints are expected-only')
        logical = dict(zip(order, counts))
        return dict(condition=c, variant=v, main=m, price=label, trial_id=ids['trialId'],
                    rootSHA256=I.digest(root), logical_order=order, endpoint_kind=kind,
                    initial_action=payload['initialDecision']['action'], **fields,
                    index=index, budget_milli=budget, total_questions=q,
                    slot_0_questions=counts[0], slot_1_questions=counts[1],
                    logical_A_questions=logical['A'], logical_B_questions=logical['B'])
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError('Malformed saved action/count endpoint: ' + str(exc)) from exc


def _validate_row(row):
    I.need(type(row) is dict and set(row) == set(ROW_FIELDS), 'Exact extracted action row required')
    coord = _coordinate(row['condition'], row['variant'], row['main'])
    I.need(type(row['price']) is str and row['price'] in dict(S.RATES), 'Frozen six-price label required')
    order = _order(row['logical_order'])
    for name in COUNTS:
        _integer(row[name], 'Integer extracted count: ' + name)
    q = row['total_questions']
    I.need(q == row['slot_0_questions'] + row['slot_1_questions'], 'Extracted slot count sum')
    logical = dict(zip(order, (row['slot_0_questions'], row['slot_1_questions'])))
    I.need([row['logical_A_questions'], row['logical_B_questions']] == [logical['A'], logical['B']],
           'Extracted logical/displayed count mapping')
    I.need(row['terminal_action'] in MACROS and row['selected_rule'] == row['terminal_action'],
           'Actual selected routine required')
    I.need(row['endpoint_kind'] == ('realized_trajectory' if q else 'expected_only'), 'Endpoint kind/count linkage')
    I.need(row['initial_action'] in ('ask_A', 'ask_B') if q else
           row['initial_action'] == row['terminal_action'], 'Extracted initial action/count linkage')
    I.need(row['price'] != 'rho_0.01' or q == 0, 'Highest-price original endpoint cannot ask')
    _integer(row['index'], 'Extracted native index', 0, 1000)
    _integer(row['budget_milli'], 'Extracted positive budget', 1)
    I.need(type(row['trial_id']) is str and bool(row['trial_id']), 'Extracted trial identity required')
    sha = row['rootSHA256']
    I.need(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef'), 'Root SHA256 required')
    return row['price'], *coord


def summarize(rows, *, full=False):
    """Return all 30 cells; partial means describe observed endpoints only.

    A complete cell's equal-endpoint mean equals ten trials then nine equally
    weighted schedules. Missing cells have None means and UNKNOWN status.
    """
    I.need(type(rows) is list and type(full) is bool, 'Rows list and explicit boolean full flag required')
    seen, roots, trials = {}, {}, {}
    for row in rows:
        key = _validate_row(row)
        I.need(key not in seen, 'Duplicate action endpoint')
        seen[key] = row
        coord = key[1:]
        identity = (row['rootSHA256'], row['logical_order'])
        I.need(coord not in roots or roots[coord] == identity, 'Root/orientation changed across prices')
        roots[coord] = identity
        trial = (row['price'], row['trial_id'])
        I.need(trial not in trials or trials[trial] == coord, 'Trial identity aliases two roots at one price')
        trials[trial] = coord
    expected = {(p, *coord) for p, _ in S.RATES for coord in I.ORIGINAL}
    complete = set(seen) == expected
    I.need(not full or complete, 'Full fixed 2700-endpoint roster required')
    cells = []
    for c in I.CONDITIONS:
        for label, rho in S.RATES:
            present = [seen[label, c, v, m] for v in range(9) for m in range(1, 11)
                       if (label, c, v, m) in seen]
            missing = [[v, m] for v in range(9) for m in range(1, 11) if (label, c, v, m) not in seen]
            n = len(present)
            actions = Counter(r['terminal_action'] for r in present)
            totals = {name: sum(r[name] for r in present) for name in COUNTS}
            cells.append(dict(condition=c, price=label, rhoGPAperSecond=rho, endpoints=n,
                              expectedEndpoints=90, missingEndpoints=90-n, missingCoordinates=missing,
                              completeOriginalCell=n == 90, status='COMPLETE' if n == 90 else 'PARTIAL' if n else 'UNKNOWN',
                              meanDenominator=n, questionTotals=totals,
                              questionMeans={name: totals[name]/n if n else None for name in COUNTS},
                              actionCounts={action: actions[action] for action in MACROS},
                              expectedOnlyEndpoints=sum(r['endpoint_kind'] == 'expected_only' for r in present),
                              realizedEndpoints=sum(r['endpoint_kind'] == 'realized_trajectory' for r in present),
                              initialAskEndpoints=sum(r['total_questions'] > 0 for r in present),
                              zeroQuestionEndpoints=sum(r['total_questions'] == 0 for r in present)))
    return dict(schema=SCHEMA, verifiedTrials=len(seen), expectedEndpoints=2700,
                missingEndpoints=2700-len(seen), fullFiniteTarget=complete, conditions=cells,
                rowSourceAuthenticated=False, policyRerunByAnalysis=False, historicalEventClaimed=False,
                monteCarloPrecisionEstablished=False,
                interpretation='actual_saved_routines_not_nearest_allocation_benchmarks',
                countSource='checked_final_state_metrics_and_paid_events',
                logicalMapping='logical_order[displayed_slot] is logical A or B',
                aggregation='observed_endpoint_mean; full cells have ten trials in each of nine fixed schedules')


def analyze(source, output, config_path, *, full=False):
    """Admit existing checked policy records using the frozen sensitivity config.

    Unlike check_trial, this reader has no reconstruction branch and never writes
    to the source. Numerical payload fields use the established 1e-10 comparison;
    extracted counts, actions, orientations and identities agree exactly.
    """
    source, output = Path(source).resolve(), S.safe_output(output)
    I.need(type(full) is bool, 'Explicit boolean full flag required')
    I.need(not source.is_relative_to(output.resolve()) and not output.resolve().is_relative_to(source),
           'Separate policy input and action output trees required')
    config_path = Path(config_path).resolve()
    context = S.load_context(config_path)
    run_path, check_path = source / 'run.json', source / 'independent_check.json'
    run_sha, check_sha = S.file_sha(run_path), S.file_sha(check_path)
    roots, run, _ = S.load_roots(context, I.read(run_path)['selection'], source)
    selection_full = {I.coordinates(root) for root in roots} == I.ORIGINAL
    I.need(not full or selection_full, 'Full action analysis requires all original roots')
    checked = I.read(check_path)
    _policy_roster(checked, full=full)
    I.need(checked['schema'] == 'personal-time-sensitivity-independent-check-v1'
           and checked['binding'] == run and checked['completeIndependentReconstruction'] is True
           and checked['absoluteTolerance'] == 1e-10 and checked['fullFiniteTarget'] is selection_full,
           'Completed original policy independent reconstruction required')
    expected = {S.record_path(source, label, root) for root in roots for label, _ in S.RATES}
    expected_rel = {str(p.relative_to(source)) for p in expected}
    I.need(set(checked['producerInputs']) == expected_rel and checked['verifiedTrials'] == len(expected),
           'Exact checked root-by-rate inventory required')
    I.need(set((source / 'producer').rglob('*.json')) == expected, 'Missing or foreign producer records')
    readers = {S.record_path(source, label, root, 'independent') for root in roots for label, _ in S.RATES}
    # S.check also writes independent/reduction.json; it is not a trial record.
    actual_readers = set((source / 'independent').rglob('*.json')) - {source / 'independent/reduction.json'}
    I.need(actual_readers == readers, 'Missing or foreign independent trial records; no replay allowed')
    rows, pins = [], {}
    for root in sorted(roots, key=I.coordinates):
        for label, _ in S.RATES:
            cfg = S.rate_config(context, root, label)
            binding = S.record_binding(root, cfg, run, label)
            path, reader = S.record_path(source, label, root), S.record_path(source, label, root, 'independent')
            rel, reader_rel = str(path.relative_to(source)), str(reader.relative_to(source))
            sha, reader_sha = S.file_sha(path), S.file_sha(reader)
            I.need(sha == checked['producerInputs'][rel], 'Changed checked policy record')
            record, other = I.read(path), I.read(reader)
            S.validate_record(record, binding)
            I.need(type(other) is dict and 'payload' in other, 'Malformed independent checkpoint')
            expected_reader = dict(schema='personal-time-sensitivity-independent-trial-v1', binding=binding,
                                   producerSHA256=sha, payload=other['payload'], absoluteTolerance=1e-10,
                                   sharedKernels=['bounded_task', 'budget_horizon', 'terminal_time',
                                                  'authenticated_input_adapter_and_event_traversal'])
            _same(other, expected_reader, 'Exact existing independent checkpoint linkage')
            S.validate_record(S.envelope(binding, other['payload']), binding)
            S.compare(record['payload'], other['payload'], 'checked_action_policy_payload')
            _same(record['payload']['context']['identity'], S.P.stable_identity(
                cfg, S.RECIPE['seed'], S.RECIPE['populationId'], *I.coordinates(root)[:2], 0, root['mainIndex']),
                'Source-derived person and trial identity')
            actual, independent = extract(root, label, record['payload']), extract(root, label, other['payload'])
            _same(actual, independent, 'Independent exact action/count/orientation agreement')
            rows.append(actual)
            pins[rel], pins[reader_rel] = sha, reader_sha
    result = summarize(rows, full=full)
    _same([sum(r['endpoint_kind'] == 'expected_only' for r in rows),
           sum(r['endpoint_kind'] == 'realized_trajectory' for r in rows)],
          [checked['expectedOnlyEndpoints'], checked['genuineRealizedEndpoints']], 'Policy-check endpoint-kind totals')
    I.need(S.file_sha(run_path) == run_sha and S.file_sha(check_path) == check_sha, 'Policy binding changed during admission')
    I.need(all(S.file_sha(source / name) == sha for name, sha in pins.items()), 'Policy input changed during admission')
    result['rowSourceAuthenticated'] = True
    _write_tables(output, rows, result)
    I.write(output / 'source_check.json', dict(
        schema=SCHEMA, sources=_source_identity(config_path),
        policyRunSHA256=run_sha, policyCheckSHA256=check_sha, policyInputs=pins,
        policyContentSHA256=I.digest(checked), policyBindingSHA256=I.digest(run),
        rootInputs={_root_key(root): I.digest(root) for root in roots},
        outputFiles={name: S.file_sha(output / name) for name in TABLE_FILES},
        verifiedTrials=len(rows), fullFiniteTarget=result['fullFiniteTarget'],
        newPolicySimulations=0, historicalProvenanceReconstructed=False,
        payloadAbsoluteTolerance=1e-10, extractedFieldsComparison='exact_typed_json'))
    verify_saved(output, config_path, policy=checked, full=full, policy_sha=check_sha)
    return dict(output=str(output), verifiedTrials=len(rows), fullFiniteTarget=result['fullFiniteTarget'],
                newPolicySimulations=0, tables=[str(output / name) for name in (*TABLE_FILES, 'source_check.json')])


def _root_key(root):
    return '%s_v%d_m%d' % I.coordinates(root)


def _source_identity(config_path):
    config_path = Path(config_path).resolve()
    _same(I.read(config_path), S.RECIPE, 'Frozen personal-time sensitivity config required')
    return {p.name: S.file_sha(p) for p in (Path(__file__), Path(T.__file__), config_path)}


def _sha(value):
    I.need(type(value) is str and len(value) == 64 and set(value) <= set('0123456789abcdef'),
           'Explicit SHA256 pin required')


def _policy_roster(policy, *, full):
    I.need(type(full) is bool and type(policy) is dict and set(policy) == {
        'schema', 'binding', 'producerInputs', 'verifiedTrials', 'fullFiniteTarget',
        'expectedOnlyEndpoints', 'genuineRealizedEndpoints', 'absoluteTolerance',
        'completeIndependentReconstruction', 'historicalParityEstablished', 'monteCarloPrecisionEstablished'},
        'Exact completed policy-check schema required')
    I.need(type(policy['binding']) is dict and {'selection', 'sources'} <= set(policy['binding'])
           and type(policy['producerInputs']) is dict, 'Policy selection/source bindings and pins required')
    I.need(policy['schema'] == 'personal-time-sensitivity-independent-check-v1'
           and policy['completeIndependentReconstruction'] is True and policy['absoluteTolerance'] == 1e-10,
           'Completed policy reconstruction required')
    I.need(policy['binding']['sources'] == S.implementation_identity(), 'Current numerical source binding required')
    selected = I.selection(policy['binding']['selection'])
    is_full = {I.coordinates(r) for r in selected} == I.ORIGINAL
    I.need(policy['fullFiniteTarget'] is is_full and (not full or is_full), 'Requested policy roster scope mismatch')
    expected = {str(S.record_path(Path('.'), label, root)) for root in selected for label, _ in S.RATES}
    I.need(set(policy['producerInputs']) == expected, 'Exact original root-by-rate producer pins required')
    _same(policy['verifiedTrials'], len(expected), 'Policy checked endpoint count')
    for sha in policy['producerInputs'].values():
        _sha(sha)
    kinds = [_integer(policy[k], 'Nonnegative endpoint-kind count')
             for k in ('expectedOnlyEndpoints', 'genuineRealizedEndpoints')]
    I.need(sum(kinds) == len(expected), 'Policy endpoint-kind denominator mismatch')
    if is_full:
        I.need(kinds == [1860, 840], 'Original full endpoint-kind scope')
    return selected, expected


def _write_tables(output, rows, result):
    I.write(output / 'trial_actions.json', rows)
    I.write(output / 'summary.json', result)
    for name, table in (('action_endpoints.csv', rows), ('action_conditions.csv', result['conditions'])):
        flat = T.table_rows(table)
        if flat:
            I.write_bytes(output / name, csv_bytes(flat, tuple(sorted(flat[0]))))


def verify_saved(output, config_path, *, policy, full=True, policy_sha=None):
    """Check a sealed action reduction without accessing the original policy tree.

    Caller must authenticate policy and the saved source_check seal (for example
    through the figure bundle). Supplying policy_sha additionally binds the exact
    original policy-check bytes; parsed policy content is always bound exactly.
    This verifies saved reduction consistency, not a new policy reconstruction.
    """
    output = S.safe_output(output)
    selected, expected = _policy_roster(policy, full=full)
    check = I.read(output / 'source_check.json')
    I.need(type(check) is dict and set(check) == CHECK_FIELDS and check['schema'] == SCHEMA,
           'Exact action source-check schema required')
    _same(check['sources'], _source_identity(config_path), 'Action reduction source/config changed')
    _same(check['policyContentSHA256'], I.digest(policy), 'Saved parsed policy-check identity changed')
    _same(check['policyBindingSHA256'], I.digest(policy['binding']), 'Saved policy binding changed')
    for field in ('policyRunSHA256', 'policyCheckSHA256'):
        _sha(check[field])
    if policy_sha is not None:
        _same(check['policyCheckSHA256'], policy_sha, 'Saved raw policy-check identity changed')
    _same([check['newPolicySimulations'], check['historicalProvenanceReconstructed'],
           check['payloadAbsoluteTolerance'], check['extractedFieldsComparison']],
          [0, False, 1e-10, 'exact_typed_json'], 'Saved action verification scope changed')
    readers = {p.replace('producer/', 'independent/', 1) for p in expected}
    I.need(all(type(check[name]) is dict for name in ('policyInputs', 'rootInputs', 'outputFiles')),
           'Explicit saved action pin mappings required')
    I.need(set(check['policyInputs']) == expected | readers, 'Complete producer/reader action pins required')
    for name, sha in check['policyInputs'].items():
        _sha(sha)
        if name in expected:
            _same(sha, policy['producerInputs'][name], 'Action producer pin differs from policy check')
    I.need(set(check['rootInputs']) == {_root_key(root) for root in selected}, 'Exact saved root-pin roster required')
    for sha in check['rootInputs'].values():
        _sha(sha)
    I.need(set(check['outputFiles']) == set(TABLE_FILES), 'Complete action output roster required')
    for name in TABLE_FILES:
        _same(S.file_sha(output / name), check['outputFiles'][name], 'Changed saved action output: ' + name)
    rows = I.read(output / 'trial_actions.json')
    result = summarize(rows, full=full)
    actual = {(r['price'], r['condition'], r['variant'], r['main']) for r in rows}
    I.need(actual == {(label, *I.coordinates(root)) for root in selected for label, _ in S.RATES},
           'Saved action rows differ from policy root-by-rate selection')
    for row in rows:
        root = dict(zip(I.FIELDS, (row['condition'], row['variant'], row['main'])))
        _same(row['rootSHA256'], check['rootInputs'][_root_key(root)], 'Saved row/root pin mismatch')
    _same([sum(r['endpoint_kind'] == 'expected_only' for r in rows),
           sum(r['endpoint_kind'] == 'realized_trajectory' for r in rows)],
          [policy['expectedOnlyEndpoints'], policy['genuineRealizedEndpoints']], 'Saved endpoint-kind totals mismatch')
    result['rowSourceAuthenticated'] = True
    _same([check['verifiedTrials'], check['fullFiniteTarget']],
          [result['verifiedTrials'], result['fullFiniteTarget']], 'Saved action scope mismatch')
    _same(I.read(output / 'summary.json'), result, 'Saved action summary differs from endpoint reduction')
    for name, table in (('action_endpoints.csv', rows), ('action_conditions.csv', result['conditions'])):
        flat = T.table_rows(table)
        I.need((output / name).read_bytes() == csv_bytes(flat, tuple(sorted(flat[0]))),
               'Saved action CSV differs from endpoint reduction: ' + name)
    return check
