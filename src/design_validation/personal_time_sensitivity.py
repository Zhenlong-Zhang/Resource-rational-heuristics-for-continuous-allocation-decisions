"""Portable six-price original-proxy production and independent reconstruction.

Only explicit original-root selections run. Public transport records are new
source-bound epochs, not historical operational receipts. Check always rebuilds
the scientific chain from authenticated roots and never fills producer gaps.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import platform
import sys

import numpy as np

from . import personal_time_inputs as I
from . import personal_time_sensitivity_streams as P
from . import personal_time_sensitivity_reference as R
from . import personal_time_sensitivity_reduction as D
from . import budget_horizon as H
from .bounded_task import TaskConfig, normalize_log_weights
from .personal_time_reproduction import compare, csv_bytes

RATES = (('rho_0', 0.), ('rho_0.00001', .00001), ('rho_0.0001', .0001), ('rho_0.001', .001),
         ('rho_geometric_0.001_0.01', .0031622776601683794), ('rho_0.01', .01))
DESCRIPTION = 'R11 absolute GPA per personal second; frozen R10 proxy durations; not human calibration'
RECIPE = {'schema': 'personal-time-proxy-sensitivity-v1', 'reference': 'personal_time_reference.json',
          'rates': [list(row) for row in RATES], 'seed': 20261019,
          'populationId': 'r10-working-five-v0-rep0-candidate-not-population-estimate',
          'randomizationGroup': 'r10-original-root-private-stream-candidate-v1', 'replicateId': 0,
          'reference_environment': {'python': '3.10', 'numpy': '1.23.5', 'node': '22.14.0'},
          'aggregation': 'ten_main_trials_per_person_then_nine_equal_weight_schedules',
          'scope': 'frozen_original_proxy_finite_diagnostic_not_literal_or_human_model',
          'historicalReference': 'rho_0.01_expected_only_no_realized_histories'}
REFERENCE_SHA256 = '0f28da0863dc3ad19a866cdcf674c2e0d0287150fdf4d79992aa21e3800d7d12'
SOURCE_FILES = (
    '__init__.py', 'personal_time_sensitivity.py', 'personal_time_sensitivity_analysis.py',
    'personal_time_sensitivity_streams.py', 'personal_time_sensitivity_reference.py',
    'personal_time_sensitivity_reduction.py', 'personal_time_inputs.py', 'personal_time_source.cjs',
    'personal_time_policy.py', 'personal_time_reference.py', 'personal_time_reproduction.py',
    'personal_time_examples.py', 'cost_aware_inputs.py', 'cost_aware_source.cjs', 'bounded_task.py',
    'budget_horizon.py', 'terminal_time.py', '../tutoring_analysis/__init__.py',
    '../tutoring_analysis/expected_outcomes.py', '../tutoring_analysis/nominal_inference.py',
    '../tutoring_analysis/native_engine.cjs')
select_roots = I.select_roots


def file_sha(path):
    path = Path(path)
    I.need(path.is_file() and not path.is_symlink(), 'Ordinary immutable file required: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_context(config_path):
    path = Path(config_path)
    recipe = I.read(path)
    I.need(I.encoded(recipe) == I.encoded(RECIPE), 'Only the frozen six-price scientific recipe is supported')
    reference_path = path.parent / recipe['reference']
    I.need(file_sha(reference_path) == REFERENCE_SHA256, 'Original reference recipe changed')
    reference = I.load_context(reference_path)
    files = {'sensitivity': file_sha(path), 'reference': file_sha(reference_path)}
    files.update({key: file_sha(path.parent / reference['recipe'][key]) for key in I.INPUT_SHA256})
    return {'recipe': recipe, 'reference': reference, 'inputFileSHA256': files}


def implementation_identity():
    here = Path(__file__).resolve().parent
    sources = {name: file_sha(here / name) for name in SOURCE_FILES}
    configs = here.parents[1] / 'configs'
    for name in ('personal_time_sensitivity.json', 'personal_time_reference.json', 'personal_time_base.json',
                 'personal_time_instance.json', 'personal_time_design.json', 'personal_time_root_identity.json'):
        sources['configs/' + name] = file_sha(configs / name)
    return sources


def runtime_identity():
    I.reference_runtime()
    return {'python': platform.python_version(), 'implementation': sys.implementation.name,
            'numpy': np.__version__, 'node': '22.14.0', 'machine': platform.machine(),
            'system': platform.system(), 'byteorder': sys.byteorder,
            'numpyCoreSHA256': file_sha(np.core._multiarray_umath.__file__),
            'numpyLinalgSHA256': file_sha(np.linalg._umath_linalg.__file__),
            'threadSettings': {name: os.environ.get(name) for name in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'MKL_CBWR', 'MKL_DEBUG_CPU_TYPE', 'NPY_DISABLE_CPU_FEATURES')}}


def binding(context, selected):
    I.need(I.encoded(context['recipe']) == I.encoded(RECIPE), 'Frozen recipe required')
    return {'schema': 'personal-time-sensitivity-run-v1', 'selection': I.selection(selected),
            'contextSHA256': I.digest(context), 'sources': implementation_identity(),
            'runtime': runtime_identity(), 'historicalReceipt': False}


def safe_output(output):
    path = Path(output).absolute()
    root = Path(__file__).resolve().parents[2]
    I.need(not path.is_symlink(), 'Output symlink refused')
    target = path.resolve()
    I.need(not root.is_relative_to(target) and not any(target.is_relative_to(root / name)
           for name in ('src', 'scripts', 'configs', 'tests', 'notebooks')), 'Use a generated-output directory')
    if path.exists():
        I.need(path.is_dir() and not any(p.is_symlink() for p in path.rglob('*')), 'Output tree symlink refused')
    return path


def prepare(context, selected, output, node='node'):
    selected, output = I.selection(selected), safe_output(output)
    run = binding(context, selected)
    I.write(output / 'run.json', run)
    data = I.prepare(context['reference'], selected, output / 'inputs', node)
    I.need(data['nodeVersion'] == run['runtime']['node'], 'Native generator actual runtime mismatch')
    return data


def load_roots(context, selected, output):
    selected, output = I.selection(selected), safe_output(output)
    run = binding(context, selected)
    I.need(I.read(output / 'run.json') == run, 'Run/input/source/runtime binding mismatch')
    return I.load_roots(context['reference'], selected, output / 'inputs'), run, output


def rate_config(context, root, label):
    I.need(label in dict(RATES), 'Frozen price required')
    cfg = I.make_config(root, context['reference'])
    if label != 'rho_0.01':
        cfg['valuation'] = {'rhoGPAperSecond': dict(RATES)[label], 'description': DESCRIPTION}
    return cfg


def record_path(output, label, root, role='producer'):
    I.need(label in dict(RATES) and role in ('producer', 'independent'), 'Known record identity')
    return Path(output) / role / label / ('%s_v%d_m%d.json' % I.coordinates(root))


def record_binding(root, cfg, run, label):
    return {'runSHA256': I.digest(run), 'rootSHA256': I.digest(root), 'configSHA256': I.digest(cfg),
            'rateLabel': label, 'coordinates': list(I.coordinates(root))}


def envelope(expected_binding, value):
    result = {'schema': 'personal-time-sensitivity-trial-v1', 'binding': expected_binding, 'payload': value}
    return {**result, 'contentSHA256': I.digest(result)}


def validate_record(record, expected_binding):
    I.need(type(record) is dict and set(record) == {'schema', 'binding', 'payload', 'contentSHA256'},
           'Exact trial envelope')
    I.need(record == envelope(expected_binding, record['payload']), 'Changed trial binding/content')
    value = record['payload']
    I.need(type(value) is dict and set(value) == {'context', 'events', 'initialDecision', 'finalState',
                                               'terminal', 'metrics', 'endpointKind'}, 'Exact trial payload')
    events = value['events']
    I.need(type(events) is list and all(type(e) is dict and set(e) == {'name', 'value'} for e in events),
           'Exact ordered event schema')
    names = [e['name'] for e in events]
    q = value['finalState']['queryIndex']
    I.need(type(q) is int and q >= 0, 'Nonnegative paid count')
    if value['endpointKind'] == 'expected_only':
        expected_names = ['decision_000000', 'expected_terminal', 'metrics']
        I.need(q == 0 and value['terminal']['realizedCombinedGPA'] is None
               and value['terminal']['newScientificDraws'] == 0, 'Expected endpoints have no draws')
    else:
        I.need(value['endpointKind'] == 'realized_trajectory', 'Known endpoint kind')
        expected_names = ['decision_000000', 'latent_A', 'latent_B']
        for k in range(q):
            expected_names.extend(['report_%06d' % k, 'paid_%06d' % k, 'decision_%06d' % (k+1)])
        expected_names += ['terminal', 'metrics']
        I.need(value['terminal']['realizedCombinedGPA'] is not None, 'Realized trajectory endpoint')
    I.need(names == expected_names, 'Missing, reordered or foreign events')
    I.need(value['initialDecision'] == events[0]['value']['value'] and value['terminal'] == events[-2]['value']
           and value['metrics'] == events[-1]['value'], 'Closed payload/event linkage')
    I.need(value['context']['rootSHA256'] == expected_binding['rootSHA256']
           and value['context']['runSHA256'] == expected_binding['runSHA256'], 'Trial context linkage')


def trial(context, root, cfg, run, label, kernels):
    """Shared traversal only; each side derives its own actions, laws and state."""
    recipe = context['recipe']
    c, v, m = I.coordinates(root)
    ctx = {'identity': kernels.stable_identity(cfg, recipe['seed'], recipe['populationId'], c, v, 0, m),
           'model': kernels.model_binding(cfg), 'rootSHA256': I.digest(root), 'runSHA256': I.digest(run),
           'bindings': {k: recipe[k] for k in ('populationId', 'randomizationGroup')}}
    state = {'budget': root['definition']['budgetMillihours'],
             'logHex': [[float(x).hex() for x in row] for row in root['initialLogs']],
             'counts': [0, 0], 'queryIndex': 0, 'searchSecondsHex': (0.).hex()}
    task, events = H.BudgetTask(TaskConfig(**cfg['task'])), []
    def save(name, value):
        events.append({'name': name, 'value': copy.deepcopy(value)})
        return value
    current = kernels.decision(cfg, task, state)
    initial = copy.deepcopy(current)
    save('decision_000000', {'before': state, 'value': current})
    expected_only = current['action'] not in ('ask_A', 'ask_B', 'random')
    if label == 'rho_0.01':
        I.need(expected_only, 'Historical .01 reference must remain expected-only')
    if expected_only:
        terminal = save('expected_terminal', kernels.expected_terminal(cfg, state, current['stop']))
    else:
        latent = {}
        for logical in ('A', 'B'):
            slot = root['logicalToDisplayed'].index(logical)
            law = normalize_log_weights(np.asarray(root['initialLogs'][slot], dtype=float))
            draw = save('latent_' + logical, kernels.stream_draw(kernels.make_key(ctx, logical, 'latent', 0), law))
            latent[logical] = draw['category']
        needs = [latent[logical] for logical in root['logicalToDisplayed']]
        while current['action'] in ('ask_A', 'ask_B'):
            q = state['queryIndex']
            I.need(q < task.config.max_assessments, 'Finite native horizon')
            slot = 0 if current['action'] == 'ask_A' else 1
            key = kernels.make_key(ctx, root['logicalToDisplayed'][slot], 'report', state['counts'][slot])
            draw = save('report_%06d' % q, kernels.stream_draw(key, task.log_likelihood[needs[slot]-1]))
            paid = save('paid_%06d' % q, kernels.paid_value(cfg, task, state, current, draw))
            state = paid['after']
            current = kernels.decision(cfg, task, state)
            save('decision_%06d' % state['queryIndex'], {'before': state, 'value': current})
            H.native_utility_table.cache_clear()
            H.native_perfect_information.cache_clear()
        terminal = save('terminal', kernels.terminal_value(cfg, ctx, state, current['stop'], needs))
    selection = current['stop'].get('legacySelection')
    targets = None if selection is None else {
        macro: int(index) for index, names in selection['allocation_classes'].items() for macro in names}
    metrics = kernels.metric_row(cfg, root, state, terminal, targets)
    if label == 'rho_0.01':
        # This source-derived field is not a relabeled historical GPA summary.
        metrics['expectedFinalUtility'] = current['stop']['expectedUtility']
    save('metrics', metrics)
    return {'context': ctx, 'events': events, 'initialDecision': initial, 'finalState': state,
            'terminal': terminal, 'metrics': metrics,
            'endpointKind': 'expected_only' if expected_only else 'realized_trajectory'}


def produce(context, selected, output):
    roots, run, output = load_roots(context, selected, output)
    for root in roots:
        for label, _ in RATES:
            cfg = rate_config(context, root, label)
            expected = record_binding(root, cfg, run, label)
            target = record_path(output, label, root)
            if target.exists():
                validate_record(I.read(target), expected)
                continue
            value = trial(context, root, cfg, run, label, P)
            record = envelope(expected, value)
            validate_record(record, expected)
            I.write(target, record)
            H.native_utility_table.cache_clear()
            H.native_perfect_information.cache_clear()
    I.write(output / 'producer_complete.json', {'schema': 'personal-time-sensitivity-producer-v1',
            'binding': run, 'selectedTrials': len(roots)*len(RATES), 'independentlyChecked': False})


def check(context, selected, output):
    roots, run, output = load_roots(context, selected, output)
    expected_paths = {record_path(output, label, root) for root in roots for label, _ in RATES}
    actual_paths = set((output / 'producer').rglob('*.json'))
    I.need(actual_paths == expected_paths, 'Missing or foreign producer trial inventory')
    rows, independent_rows, pins = [], [], {}
    for root in roots:
        for label, _ in RATES:
            cfg = rate_config(context, root, label)
            expected = record_binding(root, cfg, run, label)
            path = record_path(output, label, root)
            record = I.read(path)
            validate_record(record, expected)
            independent = trial(context, root, cfg, run, label, R)
            compare(record['payload'], independent, 'independent_complete_trial')
            reader_record = {'schema': 'personal-time-sensitivity-independent-trial-v1', 'binding': expected,
                             'producerSHA256': file_sha(path), 'payload': independent,
                             'absoluteTolerance': 1e-10, 'sharedKernels': ['bounded_task', 'budget_horizon',
                             'terminal_time', 'authenticated_input_adapter_and_event_traversal']}
            I.write(record_path(output, label, root, 'independent'), reader_record)
            pins[str(path.relative_to(output))] = file_sha(path)
            for dest, value in ((rows, record['payload']), (independent_rows, independent)):
                dest.append({'coordinates': list(I.coordinates(root)), 'rateLabel': label,
                             'metrics': value['metrics'], 'initialDecision': value['initialDecision']})
            H.native_utility_table.cache_clear()
            H.native_perfect_information.cache_clear()
    summary = D.aggregate(rows)
    independent_people = []
    for person in summary['persons']:
        metric_rows = sorted((row['metrics'] for row in independent_rows
            if row['rateLabel'] == person['rateLabel'] and row['coordinates'][:2] == [person['condition'], person['variant']]),
            key=lambda row: row['mainIndex'])
        value = R.independent_person(metric_rows)
        for key in ('means', 'choiceMeans', 'choiceEligibleWholeTen', 'realizedAvailableMains', 'availableRealizedMean'):
            compare(person[key], value[key], 'independent_person_' + key)
        compare(person['initialSlope'].get('value'), value['initialSlopeValue'], 'independent_person_initial_slope')
        independent_people.append({k: person[k] for k in ('condition', 'variant', 'rateLabel')} | value)
    conditions = R.independent_conditions(independent_people, D.LABELS)
    compare(summary['conditions'], conditions, 'independent_condition_reduction')
    counts = D.counts(rows)
    compare(counts, D.counts(independent_rows), 'counts_from_independently_reconstructed_records')
    summary.update(counts)
    expected_only = sum(row['expectedOnlyCount'] for row in counts['finalCounts'])
    realized = sum(row['genuineRealizedCount'] for row in counts['finalCounts'])
    summary.update(expectedOnlyEndpoints=expected_only, genuineRealizedEndpoints=realized,
                   utilityScope='posterior_expected_final_utility_not_ex_ante_population_policy_value',
                   historicalUtilityMapping='rho_0.01_kernel_stop_expectedUtility_not_relabelled_handoff_GPA')
    if summary['fullFiniteTarget']:
        I.need((expected_only, realized) == (1860, 840), 'Original full endpoint-kind counts')
        for label, _ in RATES:
            group = [row for row in counts['finalCounts'] if row['rateLabel'] == label]
            actual = (sum(row['expectedOnlyCount'] for row in group), sum(row['genuineRealizedCount'] for row in group))
            expected = (450, 0) if label == 'rho_0.01' else (
                (330, 120) if label == 'rho_geometric_0.001_0.01' else (270, 180))
            I.need(actual == expected, 'Original per-price endpoint-kind counts: ' + label)
    I.write(output / 'trial_rows.json', rows)
    I.write(output / 'person_rows.json', summary['persons'])
    I.write(output / 'aggregate.json', summary)
    I.write(output / 'independent' / 'reduction.json', {'producerInputs': pins, 'persons': independent_people,
                                                      'conditions': conditions, 'absoluteTolerance': 1e-10})
    if summary['fullFiniteTarget']:
        table = []
        for value in summary['conditions']:
            table.append({'condition': value['condition'], 'rateLabel': value['rateLabel'],
                'rhoGPAperSecond': dict(RATES)[value['rateLabel']],
                **{key: value['means'][key]['fullNineMean'] for key in D.SCALARS},
                **{key: value['choiceMeans'][key]['fullNineMean'] for key in D.CHOICE},
                'initialSlope': value['initialSlope']['fullNineMean'], 'MCSE': None,
                'precisionStatus': 'unreplicated_schedule_cell'})
        I.write_bytes(output / 'finite_sensitivity.csv', csv_bytes(table, tuple(table[0])))
    else:
        I.need(not (output / 'finite_sensitivity.csv').exists(), 'Partial output must not carry a full table')
    result = {'schema': 'personal-time-sensitivity-independent-check-v1', 'binding': run,
              'producerInputs': pins, 'verifiedTrials': len(rows), 'fullFiniteTarget': summary['fullFiniteTarget'],
              'expectedOnlyEndpoints': expected_only, 'genuineRealizedEndpoints': realized,
              'absoluteTolerance': 1e-10, 'completeIndependentReconstruction': True,
              'historicalParityEstablished': False, 'monteCarloPrecisionEstablished': False}
    I.write(output / 'independent_check.json', result)
    return result
