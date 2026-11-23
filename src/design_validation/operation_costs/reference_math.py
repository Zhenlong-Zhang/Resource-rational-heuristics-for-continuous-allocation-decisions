"""Independent proper-law, signed reduction, actor and cost reader; no P imports."""
import hashlib,json
from collections import Counter
from decimal import Context,ROUND_FLOOR,ROUND_CEILING
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt
from pathlib import Path

PRICES=('0x0.0p+0','0x1.5798ee2308c3ap-27','0x1.0c6f7a0b5ed8dp-20','0x1.4f8b588e368f1p-17','0x1.a36e2eb1c432dp-14','0x1.0624dd2f1a9fcp-10','0x1.9e7c6e43390b7p-9','0x1.47ae147ae147bp-7')
KNOWN=dict(read_scalar='1/2',retrieve_rule='1/4',add_subtract='1/2',multiply_divide='1',compare='1/4',clip='1/4',common_execution='53/20',request_execution='53/20')
VERSION='ask_once_post4_retained_raw_inputs_v1'

def need(x,message):
    if not x:raise ValueError(message)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rawsha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def I(z):
    def n(v):return F(int(v[0]),int(v[1])) if isinstance(v,list) else F(v)
    lo,hi=map(n,z);need(lo<=hi,'Ordered finite interval');return lo,hi
def enc(z):return [str(x) for x in z]
def add(a,b):return a[0]+b[0],a[1]+b[1]
def mul(a,b):
    z=[x*y for x in a for y in b];return min(z),max(z)
def summed(rows):
    lo=hi=F(0)
    for a,b in rows:lo+=a;hi+=b
    return lo,hi
def dot(p,u):
    need(len(p)==len(u)==100,'Complete100 need support')
    return summed(mul(a,b) for a,b in zip(p,u))
def generator_regression_reference():
    cases={}
    for key,values,expected in (
        ('positive_single_use_generator',[(F(2),F(3))],(F(2),F(3))),
        ('negative_single_use_generator',[(F(-4),F(-2))],(F(-4),F(-2))),
        ('mixed_single_use_generator',[(F(-3),F(-1)),(F(2),F(5))],(F(-1),F(4)))):
        actual=summed(z for z in values)
        need(actual==expected,'Independent one-pass generator regression '+key);cases[key]=enc(actual)
    weights=[(F(1,2),F(1,2))]*2+[(F(0),F(0))]*98
    values=[(F(-4),F(-2)),(F(2),F(6))]+[(F(0),F(0))]*98
    primary=dot(weights,values);need(primary==(F(-1),F(2)),'Independent signed100 dot regression')
    cases['signed_dot']=enc(primary)
    complete=add((F(2),F(2)),add(primary,mul((F(1),F(1)),dot(weights,values))))
    need(complete==(F(0),F(6)),'Independent opposite and baseline regression')
    cases['signed_joint_with_opposite_and_baseline']=enc(complete)
    return dict(status='PASS',cases=cases,scope='actual_single_pass_functions_exact_synthetic_algebra_no_new_law_actor_U')
def contains(saved,reference,why):
    s=I(saved);need(s[0]<=reference[0]<=reference[1]<=s[1],why)
def proper(row):
    need(len(row)==100 and all(0<=a<=b<=1 for a,b in row),'Proper full support probabilities')
    lo,hi=summed(row);need(lo<=1<=hi,'Normalization encloses one')
def rounded(z):
    d=1<<192
    return F((z[0]*d).__floor__(),d),F((z[1]*d).__ceil__(),d)
def expbox(x):
    result=[]
    for mode,direction in ((ROUND_FLOOR,'next_minus'),(ROUND_CEILING,'next_plus')):
        ctx=Context(prec=52,rounding=mode)
        a=ctx.divide(x.numerator,x.denominator)
        result.append(max(F(0),F(getattr(ctx,direction)(ctx.exp(a)))))
    return tuple(result)
def normalized(weights):
    a,b=summed(weights);need(a>0,'Certified positive normalization')
    row=[rounded((lo/b,min(F(1),hi/a))) for lo,hi in weights]
    proper(row);return row
@lru_cache(maxsize=4)
def law(logs):
    z=[F(float.fromhex(v)) for v in logs];need(len(z)==100,'Original100 log operands')
    shift=max(z);return normalized([expbox(v-shift) for v in z])
@lru_cache(maxsize=2)
def channel(width):
    need(width>0,'Positive original channel width')
    w=[expbox(-F(k*k,2)/(width*width)) for k in range(100)]
    return [normalized([w[abs(n-y)] for y in range(100)]) for n in range(100)]
@lru_cache(maxsize=800)
def utility(budget,j):
    need(type(budget)is int and budget>0 and type(j)is int and 0<=j<=1000,'Physical scalar domain')
    d=1<<96;out=[]
    for n in range(1,101):
        q=budget*j-n*1000000;v=abs(q)*d*d;a=isqrt(v//1000000);b=a if a*a*1000000==v else a+1
        out.append((-F(2*b,d),-F(2*a,d)) if q<0 else (F(a,d),F(b,d)))
    return out

def ledger(root,record,actor,reference):
    H=record['H'];slot=record['orderedSlot'];y=record['report']
    ref=reference.actor_reference(H,'EO_raw_surplus')
    need(actor['family']=='EO_raw_surplus' and actor['target']==ref['target'] and actor['arithmeticTree']==ref['arithmeticTree']
         and actor['algorithmId']=='H-literal-bplusb-v1' and actor['actorInputPermission']=='H_only'
         and actor['actor_target_certified'] is True,'Independent raw/native algorithm and permission')
    trace=actor['trace'];counts=Counter()
    need(len({z['id'] for z in trace})==len(trace),'Unique original actor events')
    need([e['operation'] for e in trace for _ in range(e['count'])]
         ==[op for op,n in ref['operationSequence'] for _ in range(n)],'Ordered actual actor events')
    for pos,e in enumerate(trace):
        prefix=H['stateId']+':EO_raw_surplus'
        need(type(e['count'])is int and e['count']>0 and e['stateVersion']==H['stateId']
             and e['id']==prefix+':'+str(pos) and e['outputRef']==prefix+':value:'+str(pos)
             and e['coefficientId']==e['operation'] and e['lifetime']=='invocation'
             and e['inputRef'] and e['inputOrigin']==('declared_algorithm' if e['operation']=='retrieve_rule'
                 else 'task_supplied_H' if e['block']=='input' else 'algorithm_local')
             and e['block'] in ('input','target','native','response')
             and (e['operation']!='common_execution' or e['block']=='response'),'Actual state/event ownership')
        counts[e['operation']]+=e['count']
    need(dict(counts)==ref['counts'] and actor['counts']==ref['counts'],'Original actor operation counts')
    fields={'report_integer':y,'updated_queried_mean':H['meanStrings'][slot],'unchanged_other_mean':H['meanStrings'][1-slot],'remaining_budget':H['resourceString']}
    blocks=[dict(block='plan_and_affordability',counts=dict(retrieve_rule=1,read_scalar=2,compare=1),fields=['initial_remaining_budget','question_cost']),dict(block='precommitted_label_and_request',counts=dict(read_scalar=1,request_execution=1),fields=['selected_physical_label']),dict(block='post_request_four_reads',counts=dict(read_scalar=4),fields=list(fields)),dict(block='asked_bit_store_and_sequence_stop',counts=dict(store=1,read_scalar=1,compare=1),fields=['asked_bit'])]
    binding={f'meanStrings[{slot}]':'updated_queried_mean',f'meanStrings[{1-slot}]':'unchanged_other_mean','resourceString':'remaining_budget'}
    owned=[];events=[];count=Counter()
    for e in trace:
        z=dict(e,feeOwner='terminal_raw',charged=True)
        if e['operation']=='read_scalar':
            need(e['count']==1 and e['block']=='input' and e['inputRef'] in binding,'One retained read per named public field')
            owned.append(e['inputRef']);z.update(feeOwner='post_request_four_reads',charged=False,alreadyChargedField=binding[e['inputRef']])
        else:count[e['operation']]+=e['count']
        events.append(z)
    need(sorted(owned)==sorted(binding),'Exactly three input read owners')
    for block in blocks:count.update(block['counts'])
    need(count['request_execution']==count['common_execution']==1,'One request and one response')
    known=sum((F(KNOWN[k])*n for k,n in count.items() if k in KNOWN),F())
    symbols={k:n for k,n in count.items() if k not in KNOWN}
    return dict(version=VERSION,blocks=blocks,postReadFieldValues=fields,rawReadOwnership=binding,originalTraceWithOwnership=events,counts=dict(count),knownSeconds=str(known),symbols=symbols,oneSecondMemberSeconds=str(known+sum(symbols.values())),unknownMember='each_used_unknown_one_second_not_measured',knownCoefficients=KNOWN,workingMemoryGrant='same_invocation_retention_paid_local_access_no_persistent_posterior_or_cross_trial_cache',updateGrant='task_supplied_display_no_actor_Bayesian_recipe',controlReadScope='four_post_reads_plus_explicit_initial_label_sequence_reads_not_four_total')
