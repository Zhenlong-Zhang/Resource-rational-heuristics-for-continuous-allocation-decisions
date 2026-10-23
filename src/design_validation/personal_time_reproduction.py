"""Produce and independently reduce the finite expected-stop reference.

These are expected endpoints of 450 original roots, not 450 realized histories.
The 50 variant-zero roots contribute both request slots (100 margin rows).
"""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

import numpy as np

from . import personal_time_policy as A
from . import personal_time_reference as R
from .bounded_task import TaskConfig
from .budget_horizon import BudgetTask
from .personal_time_inputs import (CONDITIONS, ORIGINAL, binding, coordinates, digest,
    load_context, load_roots, make_config, need, prepare, read, reference_runtime,
    select_roots, write, write_bytes)

SCALARS = ('actionExpectedCombinedGPA', 'expectedNetObjective', 'QMean', 'totalPersonalSeconds')


def compare(got, expected, path='value'):
    if type(expected) is float:
        need(type(got) in (int, float) and math.isfinite(got) and abs(got - expected) <= 1e-10,
             'Mismatch: ' + path)
    elif type(expected) is dict:
        need(type(got) is dict and set(got) == set(expected), 'Fields: ' + path)
        for key in expected:
            compare(got[key], expected[key], path + '.' + key)
    elif type(expected) is list:
        need(type(got) is list and len(got) == len(expected), 'Length: ' + path)
        for i, (left, right) in enumerate(zip(got, expected)):
            compare(left, right, path + '.' + str(i))
    else:
        need(type(got) is type(expected) and got == expected, 'Mismatch: ' + path)


def decision_path(output, root):
    return Path(output) / 'decisions' / ('%s_v%d_m%d.json' % coordinates(root))


def decision_binding(root, cfg, run_binding):
    return {'coordinates': list(coordinates(root)), 'configSHA256': digest(cfg),
            'rootSHA256': digest(root), 'runSHA256': digest(run_binding)}


def is_initial_stop(decision):
    return (decision['action'] == decision['stop']['action']
            and decision['action'] not in ('ask_A', 'ask_B', 'random'))


def validate_record(record, expected_binding):
    need(type(record) is dict and set(record) == {
        'schema', 'binding', 'status', 'decision', 'decisionSHA256', 'realizedOutcomesGenerated', 'newRandomDraws'},
        'Exact decision checkpoint fields')
    need(record['schema'] == 'personal-time-expected-stop-v1' and record['binding'] == expected_binding
         and record['status'] == 'computed', 'Exact matching finished checkpoint required')
    need(record['decisionSHA256'] == digest(record['decision']), 'Decision checkpoint content changed')
    need(record['realizedOutcomesGenerated'] is False and type(record['newRandomDraws']) is int
         and record['newRandomDraws'] == 0,
         'Expected endpoints only')
    need(is_initial_stop(record['decision']), 'Finite reference requires deterministic initial stop')


def produce(context, selected, output):
    reference_runtime()
    roots = load_roots(context, selected, output)
    run_binding = binding(context, selected)
    for root in roots:
        cfg = make_config(root, context)
        expected_binding = decision_binding(root, cfg, run_binding)
        target = decision_path(output, root)
        if target.exists():
            validate_record(read(target), expected_binding)
            continue
        task = BudgetTask(TaskConfig(**cfg['task']))
        # Do not renormalize the original JS-origin log weights at this boundary.
        logs = [np.asarray(row, dtype=float) for row in cfg['state']['logs']]
        decision = A.evaluate(cfg, task, logs, cfg['state']['budget'])
        valid = is_initial_stop(decision)
        record = {'schema': 'personal-time-expected-stop-v1', 'binding': expected_binding,
                  'status': 'computed' if valid else 'unexpected_selected_action', 'decision': decision,
                  'decisionSHA256': digest(decision), 'realizedOutcomesGenerated': False, 'newRandomDraws': 0}
        write(target, record)
        need(valid, 'Unexpected action persisted; finite initial-stop reference not reproduced')
    write(Path(output) / 'producer_complete.json',
          {'schema': 'personal-time-producer-v1', 'binding': run_binding,
           'selectedCoordinates': [list(coordinates(r)) for r in roots],
           'fullFiniteTarget': {coordinates(r) for r in roots} == ORIGINAL,
           'independentlyChecked': False})


def aggregate(rows):
    actual = [tuple(r['coordinates']) for r in rows]
    need(len(actual) == len(set(actual)) and set(actual) <= ORIGINAL, 'Exactly once original scalar ownership')
    if set(actual) != ORIGINAL:
        return {'schema': 'personal-time-expected-scalars-v1', 'fullFiniteTarget': False,
                'verifiedTrials': len(rows), 'conditions': {}, 'missingTrials': len(ORIGINAL - set(actual)),
                'realizedOutcomesGenerated': False, 'hypothesisTestsPerformed': False,
                'monteCarloPrecisionEstablished': False}
    persons = []
    for condition in CONDITIONS:
        for variant in range(9):
            group = sorted((r for r in rows if r['coordinates'][:2] == [condition, variant]),
                           key=lambda r: r['coordinates'][2])
            need([r['coordinates'][2] for r in group] == list(range(1, 11)), 'Whole ten-main participant')
            persons.append({'condition': condition, 'variant': variant,
                            **{key: math.fsum(r[key] for r in group) / 10 for key in (*SCALARS, 'share')}})
    conditions = {}
    for condition in CONDITIONS:
        people = [p for p in persons if p['condition'] == condition]
        conditions[condition] = {
            'assignedPeople': 9, 'choiceCompleteCaseMean': math.fsum(p['share'] for p in people) / 9,
            'scalarEstimates': {key: {'estimate': {'mean': math.fsum(p[key] for p in people) / 9,
                                                 'MCSE': None, 'precisionStatus': 'unreplicated_schedule_cell'}}
                                for key in SCALARS}}
    return {'schema': 'personal-time-expected-scalars-v1', 'fullFiniteTarget': True, 'assignedPeople': 45,
            'verifiedTrials': 450, 'conditions': conditions, 'persons': persons,
            'realizedOutcomesGenerated': False, 'hypothesisTestsPerformed': False,
            'monteCarloPrecisionEstablished': False}


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def check(context, selected, output):
    reference_runtime()
    roots = load_roots(context, selected, output)
    run_binding = binding(context, selected)
    rows, margins = [], []
    for root in roots:
        cfg = make_config(root, context)
        record = read(decision_path(output, root))
        validate_record(record, decision_binding(root, cfg, run_binding))
        task = BudgetTask(TaskConfig(**cfg['task']))
        logs = [np.asarray(row, dtype=float) for row in cfg['state']['logs']]
        decision = R.decision_value(cfg, task, logs, cfg['state']['budget'], lambda: None)
        compare(record['decision'], decision, 'complete_independent_decision')
        need(is_initial_stop(decision), 'Independently selected deterministic initial stop')
        stop = decision['stop']
        coords = coordinates(root)
        rows.append({'coordinates': list(coords), 'actionExpectedCombinedGPA': stop['expectedCombinedGPA'],
                     'expectedNetObjective': stop['objective'], 'QMean': 0,
                     'totalPersonalSeconds': stop['seconds'],
                     'share': stop['index'] / cfg['task']['allocation_denominator']})
        compare(stop['objective'], stop['expectedCombinedGPA'] - cfg['valuation']['rhoGPAperSecond'] * stop['seconds'],
                'priced_expected_endpoint')
        if coords[1] == 0:
            need([ask['slot'] for ask in decision['asks']] == [0, 1], 'Both original request margins')
            for ask in decision['asks']:
                delta = ask['objective'] - stop['objective']
                price = cfg['valuation']['rhoGPAperSecond'] * ask['expectedIncrementalPersonalSeconds']
                margins.append({'condition': coords[0], 'schedule_variant': 0, 'main_trial': coords[2],
                                'stored_slot': ask['slot'], 'net_ask_minus_stop': delta,
                                'margin_without_incremental_request_read_price': delta + price,
                                'scope': 'local_comparison_original_variant0_roots'})
    summary = aggregate(rows)
    expected_margins = {(c, m, slot) for c, v, m in map(coordinates, roots) if v == 0 for slot in (0, 1)}
    need(len(margins) == len(expected_margins)
         and {(r['condition'], r['main_trial'], r['stored_slot']) for r in margins} == expected_margins,
         'Complete selected margin roster')
    if summary['fullFiniteTarget']:
        need(len(margins) == 100, 'All 100 margins from the original 50 variant-zero roots')
    output = Path(output)
    write(output / 'trial_rows.json', rows)
    write(output / 'aggregate.json', summary)
    write(output / 'margin_rows.json', margins)
    write_bytes(output / 'acquisition_margins.csv', csv_bytes(margins, (
        'condition', 'schedule_variant', 'main_trial', 'stored_slot', 'net_ask_minus_stop',
        'margin_without_incremental_request_read_price', 'scope')))
    finite_rows = [{'condition': c, 'expected_combined_GPA': value['scalarEstimates']['actionExpectedCombinedGPA']['estimate']['mean'],
                    'expected_net_objective': value['scalarEstimates']['expectedNetObjective']['estimate']['mean'],
                    'mean_any_paid_request': value['scalarEstimates']['QMean']['estimate']['mean'],
                    'mean_delivered_A_share': value['choiceCompleteCaseMean'],
                    'mean_personal_seconds': value['scalarEstimates']['totalPersonalSeconds']['estimate']['mean']}
                   for c, value in summary['conditions'].items()]
    if summary['fullFiniteTarget']:
        write_bytes(output / 'finite_reference.csv', csv_bytes(finite_rows, (
            'condition', 'expected_combined_GPA', 'expected_net_objective', 'mean_any_paid_request',
            'mean_delivered_A_share', 'mean_personal_seconds')))
    result = {'schema': 'personal-time-independent-check-v1', 'binding': run_binding,
              'verifiedTrials': len(rows), 'fullFiniteTarget': summary['fullFiniteTarget'],
              'completeDecisionsCompared': True, 'absoluteTolerance': 1e-10,
              'originalVariant0Roots': len(margins) // 2, 'marginRows': len(margins),
              'newScientificDraws': 0, 'realizedOutcomesGenerated': False,
              'monteCarloPrecisionEstablished': False,
              'sharedKernels': ['bounded_task', 'budget_horizon', 'terminal_time']}
    write(output / 'independent_check.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'produce', 'check', 'all', 'examples'))
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--config', type=Path, default=root / 'configs' / 'personal_time_reference.json')
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', help='All original 450 expected endpoints and 100 margins')
    group.add_argument('--only', help='Explicit original coordinates, e.g. C3:0:1,C5:0:1')
    group.add_argument('--selection', type=Path, help='JSON list of explicit original coordinates')
    parser.add_argument('--node', default='node')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    need(not root.is_relative_to(output) and not any(output.is_relative_to(root / name)
         for name in ('src', 'scripts', 'configs', 'tests', 'notebooks')), 'Use a generated-output directory')
    if args.mode == 'examples':
        need(not args.full and args.only is None and args.selection is None, 'Examples have four fixed targets')
        from .personal_time_examples import reproduce
        recipe = read(args.config)
        return reproduce(args.config.parent / recipe['equal_target_design'], output)
    selected = select_roots(full=args.full, only=args.only,
                            selected=read(args.selection) if args.selection else None)
    context = load_context(args.config)
    if args.mode in ('prepare', 'all'):
        prepare(context, selected, output, args.node)
    if args.mode in ('produce', 'all'):
        produce(context, selected, output)
    if args.mode in ('check', 'all'):
        return check(context, selected, output)
    return {'mode': args.mode, 'selectedRoots': len(selected), 'output': str(output)}
