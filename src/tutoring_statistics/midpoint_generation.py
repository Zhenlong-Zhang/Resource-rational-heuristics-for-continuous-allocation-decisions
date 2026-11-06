"""Frozen centered-Gaussian midpoint PCG64 generation and scalar projection."""
import math
from .within_design import MIDPOINT_NAMESPACE as NAMESPACE, MIDPOINT_CASE as CASE_ID
from .within_design import ARMS, COMPARISONS, MIDPOINT_SLOPES as SLOPES

def require(value, message):
    if not value:
        raise ValueError(message)

def generate(case,library):
    import numpy as np
    require(np.__version__=='1.23.5','Changed PCG64 environment')
    n=193;seeds=case['seeds']
    uniforms=np.random.Generator(np.random.PCG64(seeds['retention'])).random((3,n));mask=uniforms<.8
    indices=np.random.Generator(np.random.PCG64(seeds['design'])).integers(0,9,size=(3,n),dtype=np.int64)
    rng=np.random.Generator(np.random.PCG64(seeds['response']));intercepts=rng.standard_normal(3*n);errors=rng.standard_normal((3*n,10))
    rows=[];retained=[]
    for ai,arm in enumerate(ARMS):
        for pi in range(n):
            person=f'{NAMESPACE}:r{case["repID"]}:{arm}:p{pi+1:03d}'
            person_rows=[];source=library[arm][int(indices[ai,pi])];j=ai*n+pi
            for ti,original in enumerate(source):
                a,b=original['need_A'],original['need_B'];beta=case['law']['slopes'][arm]
                mean=60.+beta['A']*(a-50.)+beta['B']*(b-50.)
                y=mean+8.*float(intercepts[j])+12.*float(errors[j,ti])
                person_rows.append(dict(participant=person,condition=arm,need_A=a,need_B=b,y=y))
            if bool(mask[ai,pi]):retained.append(person);rows.extend(person_rows)
    generation=dict(scheduleIndices=indices.tolist(),retentionUniforms=uniforms.tolist(),
        retentionMask={a:mask[i].tolist() for i,a in enumerate(ARMS)},interceptZ=intercepts.tolist(),
        residualZ=errors.tolist(),actualK={a:int(mask[i].sum()) for i,a in enumerate(ARMS)},
        retainedParticipantIDs=retained,responseTransform='none',rowOrder='C1,C2,C3;person1..193;numeric_main1..10')
    return dict(model='H10',reference='C2',rows=rows),generation


def evaluation(within):
    targets={}
    for key,ref in within['references'].items():
        comparison,mode=key.rsplit(':',1);pair,need=comparison.split(':');left,right=pair.split('-')
        truth=SLOPES[left][need]+(-SLOPES[right][need] if mode=='D' else SLOPES[right][need])
        ambiguous=ref['status']=='available' and any(math.isclose(truth,v,rel_tol=2e-8,abs_tol=2e-9) for v in ref['CI'])
        targets[key]=dict(truth=truth,covered=(ref['CI'][0]<truth<ref['CI'][1])
            if ref['status']=='available' and not ambiguous else None,endpointUnresolved=ambiguous,referenceStatus=ref['status'])
    nulls={key:abs(SLOPES['C1'][key[-1]])>=abs(SLOPES[key[3:5]][key[-1]]) for key in COMPARISONS}
    require([nulls[k] for k in COMPARISONS]==[False,True,False,True],'Changed finite mixed-null identity')
    return dict(schema='h10-midpoint-truth-evaluation-v1',usedByEstimator=False,references=targets,magnitudeTrueNull=nulls)


def compact_event(ordinal,within,truth,counts):
    family=within['composition'];known=family['status']=='complete'
    rejected={k:family['tests'][k]['holmReject'] for k in COMPARISONS}
    event=dict(ordinal=ordinal,repID=ordinal+1,caseID=CASE_ID,actualK=counts,methodStatus=within['status'],
        methodReason=within['reason'],references=within['references'],evaluation=truth,composition=family,
        technicalCompleted=known,availableComparisons=family['counts']['available'],A4=family['completeFamilyAvailable'],
        R=any(rejected.values()) if known else None,
        F=any(rejected[k] for k in COMPARISONS if truth['magnitudeTrueNull'][k]) if known else None,
        D=any(rejected[k] for k in COMPARISONS if not truth['magnitudeTrueNull'][k]) if known else None,
        originalGuardUsedForSelection=False)
    return event
