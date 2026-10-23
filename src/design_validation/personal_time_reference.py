"""Independent request/stop decision enumeration using shared finite kernels.

This reader intentionally does not import the producer policy or its terminal
oracle. Independence applies to action accounting/enumeration, not the shared
bounded-task, budget and terminal-macro mathematical kernels.
"""
import math

import numpy as np

from .terminal_time import (OPERATIONS, expected_curves, terminal_summary,
                            macro_durations, select_macro)


def need(ok, label):
    if not ok:
        raise ValueError(label)


def seconds(cfg,slot,report):
    a=cfg['actionCosts'];p=a['terminalProfile']
    def scalar(counts,over):return over+math.fsum(n*p[k] for n,k in zip(counts,OPERATIONS))
    def selected(v):return v['uniform'] if 'uniform' in v else v['byReport'][report-1]
    return scalar(a['requestCounts'][slot],a['requestOverheadSeconds'][slot])+scalar(selected(a['readCounts'][slot]),selected(a['readOverheadSeconds'][slot]))


def stop_value(cfg,task,logs,budget):
    c=task.config;a=cfg['actionCosts'];rho=cfg['valuation']['rhoGPAperSecond']
    ua,ub,ga,gb=expected_curves(c,*logs,budget);law=cfg['random']
    duration=law['overheadSeconds']+math.fsum(n*a['terminalProfile'][k] for n,k in zip(law['counts'],OPERATIONS))
    gpa=float(np.asarray(law['probabilities'])@(ga+gb));utility=float(np.asarray(law['probabilities'])@(ua+ub))
    random={'action':'random','expectedCombinedGPA':gpa,'expectedUtility':utility,'seconds':duration,
            'objective':gpa-rho*duration,'law':list(law['probabilities']),
            'streamNamespace':law['streamNamespace'],'realizedAllocationDrawn':False}
    if budget==0 or (budget<c.cost_millihours and cfg['insufficientBudgetMode']=='forced_random'):
        return {**random,'forced':True,'reason':'zero_budget' if budget==0 else 'insufficient_budget'}
    summary=terminal_summary(c,*logs,budget)
    selected=select_macro(summary,macro_durations(a['terminalProfile'],a['commonTerminalSeconds'],budget),rho)
    name=selected['selected_macro'];row=summary['macros'][name]
    stop={'action':name,'objective':selected['selected_objective'],'expectedCombinedGPA':row['expected_combined_GPA'],
          'expectedUtility':row['expected_utility'],'seconds':selected['duration_by_macro_seconds'][name],
          'index':selected['selected_index'],'forced':False,'reason':'terminal_menu','legacySelection':selected}
    if law['voluntaryEnabled'] and random['objective']>stop['objective']+1e-10:
        return {**random,'forced':False,'reason':'voluntary_random'}
    return stop


def decision_value(cfg,task,logs,budget,tick):
    stop=stop_value(cfg,task,logs,budget)
    result={'budget':budget,'stop':stop,'asks':[],'action':stop['action'],'selectedObjective':stop['objective']}
    if budget<task.config.cost_millihours:return result
    for who in (0,1):
        p,posts=task.predictive(logs[who]);values=[];times=[]
        need(len(p)==100 and np.isfinite(p).all() and np.all(p>=0) and abs(float(p.sum())-1)<=1e-12,'all100 report obligations')
        for r,post in enumerate(posts,1):
            tick();child=list(logs);child[who]=post
            value=stop_value(cfg,task,child,budget-task.config.cost_millihours);cost=seconds(cfg,who,r)
            values.append(value['objective']-cfg['valuation']['rhoGPAperSecond']*cost);times.append(cost)
        result['asks'].append({'slot':who,'objective':float(p@np.asarray(values)),
            'expectedIncrementalPersonalSeconds':float(p@np.asarray(times)),'tutoringIntegerCost':task.config.cost_millihours})
    best=max(v['objective'] for v in result['asks']);who=0 if result['asks'][0]['objective']>=best-1e-10 else 1
    if best>stop['objective']+1e-10:result.update(action='ask_A' if who==0 else 'ask_B',selectedObjective=result['asks'][who]['objective'])
    return result
