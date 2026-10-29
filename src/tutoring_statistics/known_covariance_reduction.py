"""Descriptive saved-input oracle coverage, paired transitions and hybrids."""
import math
import statistics
from .saved_fit_magnitude import NAMES as _NAMES, require, finite, exact, near

NAMES = list(_NAMES)
M = 2000
KEY = TARGET = "C1-C2:B:D"
L = [0, 0, 0, 0, 0, 0, 0, 1, 0]


def rep_ids(values):
    require(type(values) is list and values and all(type(v) is int and 1 <= v <= M for v in values)
            and values == sorted(set(values)), "Invalid saved repIDs")
    return values


def interval_event(center,bounds):
    require(finite(center) and type(bounds) is list and len(bounds)==2 and all(finite(v) for v in bounds)
        and bounds[0]<bounds[1],'Invalid diagnostic CI')
    uncertain=any(math.isclose(v,0.,abs_tol=2e-9,rel_tol=2e-8) for v in bounds)
    return dict(CI=bounds,covered=None if uncertain else bounds[0]<0.<bounds[1],
        belowTruth=None if uncertain else bounds[1]<0.,aboveTruth=None if uncertain else bounds[0]>0.,
        coverageNumericallyUnresolved=uncertain)


def compact_event(record):
    scalar,oracle=record['scalar'],record['oracle'];rep=record['repID'];r=oracle['references'][KEY]
    require(oracle['coefficientNames']==NAMES and r['L']==L and type(r['truth']) is float and r['truth']==0.
        and oracle['actualK']==scalar['populationCounts'] and oracle['rowCount']==10*oracle['clusterCount']
        and oracle['clusterCount']==sum(oracle['actualK'].values()),'Changed oracle structure')
    state=interval_event(r['estimate'],r['CI'])
    require(state['covered'] is r['covered'] and state['coverageNumericallyUnresolved'] is r['coverageNumericallyUnresolved'],
        'Wrong numerical endpoint classification')
    hybrids={}
    for center_name,center in (('old',scalar['baseline']['estimate']),('oracle',r['estimate'])):
        for width in ('baseline','candidate','oracle'):
            src=r if width=='oracle' else scalar[width]
            bounds=src['CI'] if width=='oracle' else [src['lower'],src['upper']]
            hybrids[center_name+'__'+width]=interval_event(center,[center+(end-src['estimate']) for end in bounds])
    return dict(repID=rep,ordinal=40*(rep-1)+32,actualK=oracle['actualK'],
        oracle=dict(estimate=r['estimate'],SE=r['SE'],z=r['z'],**state),oldMethods={k:scalar[k] for k in ('baseline','candidate')},
        comparisons={k:record['comparisons'][k][KEY] for k in ('baseline','candidate')},hybrids=hybrids)


def moments(values):
    require(values and all(finite(v) for v in values),'Invalid diagnostic moments')
    return dict(n=len(values),mean=statistics.mean(values),sampleSD=statistics.stdev(values) if len(values)>1 else None,
        min=min(values),max=max(values))


def proportion(k,n):
    """Invert binomial tails, independent of receiver's beta-quantile API."""
    from scipy.optimize import brentq
    from scipy.stats import binom
    require(type(k) is int and type(n) is int and 0<=k<=n,'Invalid coverage count')
    if n==0:return dict(numerator=k,denominator=n,value=None,interval95=None)
    lo=0. if k==0 else float(brentq(lambda x:binom.sf(k-1,n,x)-.025,0.,1.,xtol=1e-14))
    hi=1. if k==n else float(brentq(lambda x:binom.cdf(k,n,x)-.025,0.,1.,xtol=1e-14))
    return dict(numerator=k,denominator=n,value=k/n,interval95=[lo,hi])


def coverage_report(rows,complete):
    states=[r for r in rows if type(r['covered']) is bool];c=sum(r['covered'] for r in states)
    return dict(present=len(rows),coverageKnown=len(states),covered=c,numericallyUnresolved=len(rows)-len(states),
        belowTruth=sum(r['belowTruth'] for r in states),aboveTruth=sum(r['aboveTruth'] for r in states),
        conditional=proportion(c,len(states)),fullM=proportion(c,M) if complete and len(states)==M else None)


def reduce_events(events):
    ids=[r['repID'] for r in events]
    if ids:rep_ids(ids)
    complete=ids==list(range(1,M+1))
    report=dict(M=M,T=len(events),present=len(events),complete=complete,missingRepIDs=sorted(set(range(1,M+1))-set(ids)),
        technicalMissing=M-len(events),oracle=coverage_report([r['oracle'] for r in events],complete),paired={},hybrids={},
        moments={},comparisons={},oldCoverageCountsCalculated={name: sum(e['oldMethods'][name]['covered'] for e in events) if complete else None for name in ('baseline', 'candidate')},
        interpretation='descriptive_same_input_known_DGP_diagnostic_not_causal_or_human_method',
        intervalScope='pointwise_central_95_percent_Clopper_Pearson_not_a_new_MC_test')
    for method in ('baseline','candidate'):
        counts={s:0 for s in ('00','01','10','11','unknown')}
        for event in events:
            old,new=event['oldMethods'][method]['covered'],event['oracle']['covered']
            require(type(old) is bool and (new is None or type(new) is bool),'Wrong paired coverage state')
            counts['unknown' if new is None else str(int(old))+str(int(new))]+=1
        report['paired'][method]=dict(order='old_first_oracle_second',counts=counts,jointlyKnown=len(events)-counts['unknown'])
        report['comparisons'][method]={k:moments([r['comparisons'][method][k] for r in events])
            for k in ('estimateDifference','SEratio','oldStandardizedError','knownStandardizedError')} if events else {}
    report['hybrids']={k:coverage_report([r['hybrids'][k] for r in events],complete)
        for k in ('old__baseline','old__candidate','old__oracle','oracle__baseline','oracle__candidate','oracle__oracle')}
    if events:report['moments']={k:moments([r['oracle'][k] for r in events]) for k in ('estimate','SE','z')}
    report['oracleRMSSE']=math.sqrt(statistics.mean([r['oracle']['SE']**2 for r in events])) if events else None
    empirical=report['moments'].get('estimate',{}).get('sampleSD')
    report['oracleRMSSEOverEmpiricalSD']=report['oracleRMSSE']/empirical if empirical is not None and empirical>0 else None
    return report
