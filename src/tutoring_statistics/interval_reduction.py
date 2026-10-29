"""Closed two-law interval coverage and prespecified paired Holm3 report."""
import math
from .saved_fit_magnitude import require

METHODS = ('baseline','candidate','KR')
TARGETS = {1: {'C1-C2:B:D': ([0,0,0,0,0,0,0,1,0], 0.)},
           2: {'C1-C2:A:S': ([0,0,0,2,0,1,0,0,0], .8),
               'C1-C3:A:S': ([0,0,0,2,0,1,1,0,0], .8)}}


def cp(count,total):
    from scipy.stats import beta
    if not total:return None
    return [0. if count==0 else float(beta.ppf(.025,count,total-count+1)),
        1. if count==total else float(beta.ppf(.975,count+1,total-count))]


def moments(values):
    n=len(values);mean=math.fsum(values)/n if n else None
    sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/(n-1)) if n>1 else None
    return dict(n=n,mean=mean,sd=sd,minimum=min(values) if n else None,maximum=max(values) if n else None)


def compare_pair(rows,key,left,right):
    from scipy.stats import binomtest
    table={k:0 for k in ('00','01','10','11')};excluded=[]
    for event in rows:
        refs=[]
        for name in (left,right):
            method=event['oracle'] if name=='oracle' else event['methods'][name]
            ref=method['references'].get(key,{})
            refs.append((method,ref))
        if all(type(ref.get('covered')) is bool for method,ref in refs):
            table[''.join(str(int(ref['covered'])) for method,ref in refs)]+=1
        else:
            excluded.append(dict(ordinal=event['ordinal'],leftStatus=refs[0][0]['status'],rightStatus=refs[1][0]['status'],
                leftReason=refs[0][0].get('reason') or refs[0][1].get('reason'),
                rightReason=refs[1][0].get('reason') or refs[1][1].get('reason'),
                leftEndpointUnknown=refs[0][1].get('coverageNumericallyUnresolved',False),
                rightEndpointUnknown=refs[1][1].get('coverageNumericallyUnresolved',False)))
    Q=sum(table.values());gain=table['01'];loss=table['10'];delta=(gain-loss)/Q if Q else None
    discord=(gain+loss)/Q if Q else None
    return dict(left=left,right=right,Q=Q,table=table,excluded=excluded,delta=delta,
        MCSE=math.sqrt(max(0.,discord-delta**2)/(Q-1)) if Q>1 else None,
        exactTwoSidedMcNemarP=(float(binomtest(gain,gain+loss,.5).pvalue) if gain+loss else 1.) if Q else None,
        fullLawPrimaryEligible=Q==2129 and len(rows)==2129,holm3P=None)


def reduce_rows(events):
    require(len({r['ordinal'] for r in events})==len(events),'Duplicate final event')
    result={};primary=[]
    for case in (1,2):
        rows=[r for r in events if r['caseIndex']==case];methods={}
        for name in METHODS+('oracle',):
            report={}
            for key in TARGETS[case]:
                refs=[];unknown=[];technical=statistical=0
                for row in rows:
                    method=row['oracle'] if name=='oracle' else row['methods'][name]
                    if method['status']=='technical_unresolved':technical+=1
                    if method['status']=='statistical_unavailable':statistical+=1
                    ref=method['references'].get(key)
                    if method['status']!='complete':unknown.append(dict(ordinal=row['ordinal'],status=method['status'],
                        reason=method.get('reason') or (ref or {}).get('reason')))
                    if ref is not None:refs.append(ref)
                continuous=[r for r in refs if r.get('status','available')=='available' and 'estimate' in r]
                resolved=[r for r in refs if type(r.get('covered')) is bool]
                C=sum(int(r['covered']) for r in resolved);A=len(resolved);T=len(rows)-technical
                errors=[r['estimate']-r['truth'] for r in continuous];error=moments(errors)
                rms=math.sqrt(math.fsum(r['SE']**2 for r in continuous)/len(continuous)) if continuous else None
                cov=C/A if A else None;full=len(rows)==A==2129
                report[key]=dict(M=2129,present=len(rows),missingRows=2129-len(rows),T=T,A=A,C=C,
                    technicalUnknown=technical,statisticalUnavailable=statistical,
                    endpointUnknown=sum(r.get('coverageNumericallyUnresolved') is True for r in refs),
                    unavailableReasons=unknown,conditionalCoverage=cov,conditionalCP95=cp(C,A),
                    fullCoverage=C/2129 if full else None,fullCP95=cp(C,2129) if full else None,
                    MCSE=math.sqrt(cov*(1-cov)/A) if A else None,
                    plannedClassificationBounds=[C/2129,(C+2129-A)/2129],
                    signedError=error,meanErrorMCSE=error['sd']/math.sqrt(error['n']) if error['sd'] is not None else None,
                    RMSSE=rms,RMSSEoverEmpiricalSD=rms/error['sd'] if rms is not None and error['sd'] else None,
                    widths=moments([r['CI'][1]-r['CI'][0] for r in continuous]),
                    dfs=moments([r['df'] for r in continuous if 'df' in r]),
                    scales=moments([r.get('scale',1.) for r in continuous]),
                    lowerMiss=sum(r['covered'] is False and r['truth']<r['CI'][0] for r in resolved),
                    upperMiss=sum(r['covered'] is False and r['truth']>r['CI'][1] for r in resolved))
                report[key]['absoluteNominalAdverse']=(report[key]['fullCP95'][1]<.95) if full else None
            methods[name]=report
        paired={};secondary={}
        for key in TARGETS[case]:
            paired[key]=compare_pair(rows,key,'candidate','KR');primary.append(paired[key])
            secondary[key]={name:compare_pair(rows,key,left,right) for name,left,right in
                (('baseline_vs_KR','baseline','KR'),('oracle_vs_KR','oracle','KR'),('oracle_vs_CR2','oracle','candidate'))}
        result[str(case)]=dict(M=2129,present=len(rows),methods=methods,pairedKRminusCR2=paired,diagnosticPairs=secondary,
            retainedK={arm:moments([r['actualK'][arm] for r in rows]) for arm in ('C1','C2','C3')})
    if all(p['fullLawPrimaryEligible'] for p in primary):
        previous=0.
        for rank,index in enumerate(sorted(range(3),key=lambda j:(primary[j]['exactTwoSidedMcNemarP'],j))):
            previous=max(previous,min(1.,(3-rank)*primary[index]['exactTwoSidedMcNemarP']));primary[index]['holm3P']=previous
    for item in primary:item['holm3Reject']=(item['holm3P']<=.05) if item['holm3P'] is not None else None
    return result
