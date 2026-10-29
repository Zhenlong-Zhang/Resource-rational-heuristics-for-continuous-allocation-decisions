"""Independent interval diagnostics, exact binomial tails and paired Holm3."""
import math
from .independent_magnitude_reduction import require

METHODS = ('baseline','candidate','KR')
TARGETS = {1: {'C1-C2:B:D': 0.}, 2: {'C1-C2:A:S': .8, 'C1-C3:A:S': .8}}
LROWS = {'C1-C2:B:D': [0,0,0,0,0,0,0,1,0],
         'C1-C2:A:S': [0,0,0,2,0,1,0,0,0], 'C1-C3:A:S': [0,0,0,2,0,1,1,0,0]}


def audit_event(row):
    ordinal=row['ordinal'];case=1+(ordinal%2);rep=1+(ordinal//2)
    require(type(ordinal) is int and 0<=ordinal<4258 and row['caseIndex']==case and row['repID']==rep,
        'Wrong independent scientific ID')
    expected='h10covv1_initial_amount_'+('s1' if case==1 else 's5_l20')+'_recruit193_mcar08'
    require(row['caseID']==expected and set(row['actualK'])=={'C1','C2','C3'}
        and all(type(v) is int and 0<=v<=193 for v in row['actualK'].values()),'Changed independent law/K')
    require(set(row['methods'])==set(METHODS),'Changed method family')
    for name in METHODS+('oracle',):
        item=row['oracle'] if name=='oracle' else row['methods'][name]
        require(item['status'] in ('complete','technical_unresolved','statistical_unavailable'),'Unknown method state')
        expected_refs=set(TARGETS[case]) if name!='oracle' or item['status']=='complete' else set()
        require(set(item['references'])==expected_refs,'Changed reference roster')
        for key,reference in item['references'].items():
            require(reference['truth']==TARGETS[case][key],'Changed independent target truth')
            if item['status']=='complete':
                require(reference['L']==LROWS[key] and reference.get('status','available')=='available'
                    and type(reference['SE']) in (float,int) and math.isfinite(reference['SE']) and reference['SE']>0,
                    'Changed available target')
                require(len(reference['CI'])==2 and all(math.isfinite(v) for v in reference['CI'])
                    and reference['CI'][0]<reference['CI'][1],'Invalid accepted interval')
                require((reference['coverageNumericallyUnresolved'] is True and reference['covered'] is None)
                    or (reference['coverageNumericallyUnresolved'] is False and type(reference['covered']) is bool),
                    'Changed endpoint classification')
            else:require(reference['status']==item['status'] and reference['covered'] is None,
                'Unknown converted to coverage')


def exact_cp(k,n):
    from scipy.stats import binomtest
    if n==0:return None
    interval=binomtest(k,n).proportion_ci(confidence_level=.95,method='exact')
    return [float(interval.low),float(interval.high)]


def summarize_numbers(values):
    count=len(values)
    if not count:return dict(n=0,mean=None,sd=None,minimum=None,maximum=None)
    mean=sum(values)/count
    return dict(n=count,mean=mean,sd=math.sqrt(sum((value-mean)**2 for value in values)/(count-1)) if count>1 else None,
        minimum=min(values),maximum=max(values))


def method_at(row,name):
    return row['oracle'] if name=='oracle' else row['methods'][name]


def table_for(rows,key,left,right):
    from scipy.stats import binom
    cells=[0,0,0,0];excluded=[]
    for row in rows:
        lm,rm=method_at(row,left),method_at(row,right)
        a,b=lm['references'].get(key,{}),rm['references'].get(key,{})
        if type(a.get('covered')) is bool and type(b.get('covered')) is bool:
            cells[2*int(a['covered'])+int(b['covered'])]+=1
        else:excluded.append(dict(ordinal=row['ordinal'],leftStatus=lm['status'],rightStatus=rm['status'],
            leftReason=lm.get('reason') or a.get('reason'),rightReason=rm.get('reason') or b.get('reason'),
            leftEndpointUnknown=a.get('coverageNumericallyUnresolved',False),rightEndpointUnknown=b.get('coverageNumericallyUnresolved',False)))
    Q=sum(cells);gain,loss=cells[1:3];d=gain+loss
    value=(gain-loss)/Q if Q else None
    return dict(left=left,right=right,Q=Q,table=dict(zip(('00','01','10','11'),cells)),excluded=excluded,
        delta=value,MCSE=math.sqrt((d*Q-(gain-loss)**2)/(Q*Q*(Q-1))) if Q>1 else None,
        exactTwoSidedMcNemarP=(float(min(1.,2*binom.cdf(min(gain,loss),d,.5))) if d else 1.) if Q else None,
        fullLawPrimaryEligible=len(rows)==Q==2129,holm3P=None)


def reduce_independently(events):
    ids=[row['ordinal'] for row in events];require(len(set(ids))==len(ids),'Duplicate independent ID')
    output={};family=[]
    for case in (1,2):
        rows=[row for row in events if row['caseIndex']==case];reports={}
        for name in METHODS+('oracle',):
            references={}
            for key,truth in TARGETS[case].items():
                refs=[];unknown=[];technical=statistical=0
                for row in rows:
                    method=method_at(row,name);ref=method['references'].get(key)
                    technical+=int(method['status']=='technical_unresolved');statistical+=int(method['status']=='statistical_unavailable')
                    if method['status']!='complete':unknown.append(dict(ordinal=row['ordinal'],status=method['status'],
                        reason=method.get('reason') or (ref or {}).get('reason')))
                    if ref is not None:refs.append(ref)
                classified=[ref for ref in refs if type(ref.get('covered')) is bool]
                continuous=[ref for ref in refs if ref.get('status','available')=='available' and 'estimate' in ref]
                covered=sum(ref['covered'] for ref in classified);available=len(classified)
                proportion=covered/available if available else None
                errors=summarize_numbers([ref['estimate']-ref['truth'] for ref in continuous])
                rms=math.sqrt(sum(ref['SE']*ref['SE'] for ref in continuous)/len(continuous)) if continuous else None
                full=len(rows)==available==2129;interval=exact_cp(covered,2129) if full else None
                references[key]=dict(M=2129,present=len(rows),missingRows=2129-len(rows),T=len(rows)-technical,
                    A=available,C=covered,technicalUnknown=technical,statisticalUnavailable=statistical,
                    endpointUnknown=sum(ref.get('coverageNumericallyUnresolved') is True for ref in refs),
                    unavailableReasons=unknown,conditionalCoverage=proportion,conditionalCP95=exact_cp(covered,available),
                    fullCoverage=covered/2129 if full else None,fullCP95=interval,
                    MCSE=math.sqrt(proportion*(1-proportion)/available) if available else None,
                    plannedClassificationBounds=[covered/2129,(covered+2129-available)/2129],signedError=errors,
                    meanErrorMCSE=errors['sd']/math.sqrt(errors['n']) if errors['sd'] is not None else None,
                    RMSSE=rms,RMSSEoverEmpiricalSD=rms/errors['sd'] if rms is not None and errors['sd'] else None,
                    widths=summarize_numbers([ref['CI'][1]-ref['CI'][0] for ref in continuous]),
                    dfs=summarize_numbers([ref['df'] for ref in continuous if 'df' in ref]),
                    scales=summarize_numbers([ref.get('scale',1.) for ref in continuous]),
                    lowerMiss=sum(ref['covered'] is False and truth<ref['CI'][0] for ref in classified),
                    upperMiss=sum(ref['covered'] is False and truth>ref['CI'][1] for ref in classified),
                    absoluteNominalAdverse=interval[1]<.95 if full else None)
            reports[name]=references
        primary={};secondary={}
        for key in TARGETS[case]:
            primary[key]=table_for(rows,key,'candidate','KR');family.append(primary[key])
            secondary[key]={label:table_for(rows,key,a,b) for label,a,b in
                (('baseline_vs_KR','baseline','KR'),('oracle_vs_KR','oracle','KR'),('oracle_vs_CR2','oracle','candidate'))}
        output[str(case)]=dict(M=2129,present=len(rows),methods=reports,pairedKRminusCR2=primary,diagnosticPairs=secondary,
            retainedK={arm:summarize_numbers([row['actualK'][arm] for row in rows]) for arm in ('C1','C2','C3')})
    if len(family)==3 and all(item['fullLawPrimaryEligible'] for item in family):
        order=sorted(range(3),key=lambda i:(family[i]['exactTwoSidedMcNemarP'],i));scaled=[]
        for position,index in enumerate(order):
            scaled.append(min(1.,(3-position)*family[index]['exactTwoSidedMcNemarP']))
            family[index]['holm3P']=max(scaled)
    for item in family:item['holm3Reject']=(item['holm3P']<=.05) if item['holm3P'] is not None else None
    return output
