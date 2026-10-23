"""Finite myopic action costs, with tutoring and personal time kept separate.

Only the mathematical candidate evaluator is retained. Its historical candidate,
policy and random-law identifiers are scientific identities, not adoption claims.
The terminal settings describe the shared terminal kernel; request/read pricing
is added here, rather than changing that kernel's settings or charging sunk time.
"""
from __future__ import annotations

from dataclasses import fields
import math

import numpy as np

from .bounded_task import TaskConfig, normalize_log_weights
from .budget_horizon import BudgetTask
from .terminal_time import (OPERATIONS, expected_curves, macro_durations,
                            select_macro, terminal_summary, validate_settings)

TOL = 1e-10
KEYS = {'schema', 'candidateId', 'policyId', 'valuation', 'task', 'settings',
        'actionCosts', 'random', 'insufficientBudgetMode', 'state', 'pathReports'}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(value, keys, name):
    need(isinstance(value, dict) and set(value) == set(keys), name+' exact fields')


def scalar(value, name, positive=False):
    need(type(value) in (int, float) and math.isfinite(value) and
         (value > 0 if positive else value >= 0), name+' nonnegative finite number')
    return float(value)


def integer(value, name):
    need(type(value) is int and value >= 0, name+' nonnegative integer')
    return value


def counts(value):
    need(isinstance(value, list) and len(value) == len(OPERATIONS), 'six operation counts')
    for x in value:
        integer(x, 'operation count')
    return value


def seconds(profile, inventory, overhead):
    return scalar(overhead, 'overhead')+math.fsum(n*profile[k] for k, n in zip(OPERATIONS, counts(inventory)))


def report_value(value, report):
    need(isinstance(value, dict) and set(value) in ({'uniform'}, {'byReport'}), 'explicit uniform/byReport encoding')
    if 'uniform' in value:
        return value['uniform']
    need(isinstance(value['byReport'], list) and len(value['byReport']) == 100, 'all100 report values')
    return value['byReport'][report-1]


def validate(config):
    exact(config, KEYS, 'candidate')
    need(config['schema'] == 'r10-action-cost-candidate-v1', 'candidate schema')
    need(config['policyId'] == 'repeated-myopic-action-cost-v1', 'explicit myopic policy')
    need(isinstance(config['candidateId'], str) and bool(config['candidateId']), 'candidate identity')
    exact(config['valuation'], {'rhoGPAperSecond', 'description'}, 'valuation')
    scalar(config['valuation']['rhoGPAperSecond'], 'rho')
    need(isinstance(config['valuation']['description'], str) and bool(config['valuation']['description']), 'valuation convention')
    validate_settings(config['settings'])
    exact(config['task'], {f.name for f in fields(TaskConfig)}, 'native task')
    for name in ('total_millihours','cost_millihours','repeat_count_per_30','allocation_denominator','max_assessments'):
        integer(config['task'][name], name)
    for name in ('scale','report_width','alpha','loss_multiplier','gpa_scale'):
        scalar(config['task'][name], name, positive=True)
    for name in ('gap_offset','gpa_baseline'):
        need(type(config['task'][name]) in (int,float) and math.isfinite(config['task'][name]), 'finite '+name)
    for name in ('condition_id','template_id'):
        need(isinstance(config['task'][name], str) and bool(config['task'][name]), 'explicit '+name)
    for name in ('prior_centres','prior_spreads'):
        need(isinstance(config['task'][name], list) and len(config['task'][name]) == 2, 'two native prior values')
        for value in config['task'][name]:
            need(type(value) in (int,float) and math.isfinite(value), 'finite prior value')
            if name == 'prior_spreads': scalar(value, name, positive=True)
    c = TaskConfig(**config['task'])
    need(type(c.allocation_denominator) is int and c.allocation_denominator > 0, 'native denominator')
    need(type(c.cost_millihours) is int and c.cost_millihours > 0 and type(c.total_millihours) is int, 'integer tutoring resource')
    need(c.max_assessments == c.total_millihours//c.cost_millihours, 'budget-only assessment capacity')
    a = config['actionCosts']
    exact(a, {'terminalProfile', 'commonTerminalSeconds', 'requestCounts', 'requestOverheadSeconds',
              'readCounts', 'readOverheadSeconds'}, 'action costs')
    p = a['terminalProfile']
    exact(p, (*OPERATIONS, 'EO_inference_seconds', 'OPT_inference_seconds'), 'timing profile')
    for value in p.values(): scalar(value, 'primitive seconds')
    macro_durations(p, scalar(a['commonTerminalSeconds'], 'common terminal seconds'), 1)
    for key in ('requestCounts', 'requestOverheadSeconds', 'readCounts', 'readOverheadSeconds'):
        need(isinstance(a[key], list) and len(a[key]) == 2, 'two displayed recipients')
    for who in range(2):
        seconds(p, a['requestCounts'][who], a['requestOverheadSeconds'][who])
        for report in range(1,101):
            seconds(p, report_value(a['readCounts'][who], report), report_value(a['readOverheadSeconds'][who], report))
    r = config['random']
    exact(r, {'probabilities', 'counts', 'overheadSeconds', 'voluntaryEnabled', 'streamNamespace'}, 'random')
    need(type(r['voluntaryEnabled']) is bool, 'random availability boolean')
    need(isinstance(r['streamNamespace'], str) and r['streamNamespace'].startswith('r10-random-terminal:'), 'distinct random stream namespace')
    need(isinstance(r['probabilities'], list) and len(r['probabilities']) == c.allocation_denominator+1, 'native grid probability law')
    for value in r['probabilities']: scalar(value, 'probability')
    need(abs(math.fsum(r['probabilities'])-1.) <= 1e-12, 'unit probability mass; no silent normalization')
    need(seconds(p, r['counts'], r['overheadSeconds']) > 0, 'forced random choice must cost positive personal time')
    need(config['insufficientBudgetMode'] in ('terminal_menu', 'forced_random'), 'explicit insufficient budget mode')
    exact(config['state'], {'budget', 'logs'}, 'state')
    budget = integer(config['state']['budget'], 'budget')
    need(budget <= c.total_millihours, 'state within original resource')
    logs = config['state']['logs']
    need(logs == 'task_initial' or isinstance(logs, list) and len(logs) == 2, 'two log beliefs or explicit task_initial')
    for row in [] if logs == 'task_initial' else logs:
        need(isinstance(row, list) and len(row) == 100 and all(type(x) in (int,float) and not math.isnan(x) and x != math.inf for x in row) and any(math.isfinite(x) for x in row), 'finite-support log weights')
    need(isinstance(config['pathReports'], list), 'finite supplied report path')
    for row in config['pathReports']:
        exact(row, {'slot', 'report'}, 'supplied report')
        need(type(row['slot']) is int and row['slot'] in (0,1) and type(row['report']) is int and 1 <= row['report'] <= 100, 'supplied native report')
    return c


def request_seconds(config, who, report):
    a = config['actionCosts']; p = a['terminalProfile']
    return seconds(p, a['requestCounts'][who], a['requestOverheadSeconds'][who])+seconds(
        p, report_value(a['readCounts'][who], report), report_value(a['readOverheadSeconds'][who], report))


def random_terminal(config, task, logs, budget):
    ua, ub, ga, gb = expected_curves(task.config, *logs, budget)
    mass = np.asarray(config['random']['probabilities'], dtype=float)
    duration = seconds(config['actionCosts']['terminalProfile'], config['random']['counts'], config['random']['overheadSeconds'])
    gpa = float(mass @ (ga+gb)); utility = float(mass @ (ua+ub))
    return {'action': 'random', 'expectedCombinedGPA': gpa, 'expectedUtility': utility,
            'seconds': duration, 'objective': gpa-config['valuation']['rhoGPAperSecond']*duration,
            'law': list(config['random']['probabilities']), 'streamNamespace': config['random']['streamNamespace'],
            'realizedAllocationDrawn': False}


def terminal(config, task, logs, budget):
    random = random_terminal(config, task, logs, budget)
    if budget == 0 or (budget < task.config.cost_millihours and config['insufficientBudgetMode'] == 'forced_random'):
        return {**random, 'forced': True, 'reason': 'zero_budget' if budget == 0 else 'insufficient_budget'}
    summary = terminal_summary(task.config, *logs, budget)
    a = config['actionCosts']
    chosen = select_macro(summary, macro_durations(a['terminalProfile'], a['commonTerminalSeconds'], budget), config['valuation']['rhoGPAperSecond'])
    macro = chosen['selected_macro']; row = summary['macros'][macro]
    result = {'action': macro, 'objective': chosen['selected_objective'], 'expectedCombinedGPA': row['expected_combined_GPA'],
              'expectedUtility': row['expected_utility'], 'seconds': chosen['duration_by_macro_seconds'][macro],
              'index': chosen['selected_index'], 'forced': False, 'reason': 'terminal_menu', 'legacySelection': chosen}
    if config['random']['voluntaryEnabled'] and random['objective'] > result['objective']+TOL:
        return {**random, 'forced': False, 'reason': 'voluntary_random'}
    return result


def evaluate(config, task, logs, budget):
    stop = terminal(config, task, logs, budget)
    result = {'budget': budget, 'stop': stop, 'asks': [], 'action': stop['action'], 'selectedObjective': stop['objective']}
    if budget < task.config.cost_millihours:
        return result
    rho = config['valuation']['rhoGPAperSecond']
    for who in range(2):
        probabilities, posteriors = task.predictive(logs[who])
        need(len(probabilities) == 100 and bool(np.isfinite(probabilities).all()) and bool((probabilities >= 0).all()) and abs(float(probabilities.sum())-1.) <= 1e-12, 'all100 normalized reports')
        values = []; duration = []
        for report in range(1,101):
            child = list(logs); child[who] = posteriors[report-1]
            selected = terminal(config, task, child, budget-task.config.cost_millihours)
            personal = request_seconds(config, who, report)
            duration.append(personal); values.append(selected['objective']-rho*personal)
        result['asks'].append({'slot': who, 'objective': float(probabilities @ np.asarray(values)),
                              'expectedIncrementalPersonalSeconds': float(probabilities @ np.asarray(duration)),
                              'tutoringIntegerCost': task.config.cost_millihours})
    best = max(x['objective'] for x in result['asks'])
    who = 0 if result['asks'][0]['objective'] >= best-TOL else 1
    if best > stop['objective']+TOL:
        result.update(action='ask_A' if who == 0 else 'ask_B', selectedObjective=result['asks'][who]['objective'])
    return result


def run(config):
    c = validate(config); task = BudgetTask(c)
    initial_logs = task.initial_logs if config['state']['logs'] == 'task_initial' else config['state']['logs']
    logs = [normalize_log_weights(np.asarray(row, dtype=float)) for row in initial_logs]
    budget = config['state']['budget']; spent = 0.; steps = []; used = 0
    first = evaluate(config, task, logs, budget); current = first
    for supplied in config['pathReports']:
        need(current['action'] in ('ask_A','ask_B'), 'unused supplied reports after terminal choice')
        who = 0 if current['action'] == 'ask_A' else 1
        need(supplied['slot'] == who, 'supplied report belongs to selected action')
        need(budget >= c.cost_millihours, 'no tutoring overdraft')
        increment = request_seconds(config, who, supplied['report'])
        before = budget; budget -= c.cost_millihours; spent += increment
        logs[who] = task.posterior(logs[who], supplied['report']); used += 1
        steps.append({'slot': who, 'report': supplied['report'], 'budgetBefore': before, 'budgetAfter': budget,
                      'incrementalPersonalSeconds': increment, 'cumulativeSearchSeconds': spent})
        current = evaluate(config, task, logs, budget)
    complete = current['action'] not in ('ask_A','ask_B')
    chosen = current['stop'] if complete else None
    return {'schema': 'r10-action-cost-result-v1', 'candidateId': config['candidateId'], 'policyId': config['policyId'],
            'scope': 'deterministic_supplied_report_fixture_not_population_prediction', 'initial': first,
            'steps': steps, 'reportsConsumed': used, 'finalBudget': budget, 'finalLogHex': [[float(v).hex() for v in x] for x in logs],
            'status': 'terminal' if complete else 'awaiting_supplied_report', 'current': current,
            'searchSeconds': spent, 'terminalSeconds': None if chosen is None else chosen['seconds'],
            'totalPersonalSeconds': None if chosen is None else spent+chosen['seconds'],
            'finalExpectedCombinedGPA': None if chosen is None else chosen['expectedCombinedGPA'],
            'finalNetObjective': None if chosen is None else chosen['objective']-config['valuation']['rhoGPAperSecond']*spent,
            'empiricalCalibration': False, 'paperModelAdopted': False}
