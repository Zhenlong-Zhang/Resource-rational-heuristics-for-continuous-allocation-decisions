"""Fixed-denominator joint reports, paired IUT transitions and diagnostics."""
import math
from .magnitude_design import IDS, require


def reduce_events(ctx, events):
    """Producer counts; the independent receiver uses its separate frozen reducer."""
    require(len({e['ordinal'] for e in events}) == len(events),'Duplicate final event')
    for event in events:
        case = ctx['modules']['audit'].case_for(ctx['manifest'], event['ordinal'])
        require(event['caseID'] == case['caseID'] and event['repID'] == case['repID']
                and event['truth'] == ctx['modules']['audit'].truth(case), 'Changed joint event identity/truth')
    from scipy.stats import beta
    def cp(k,n):
        if not n: return None
        return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
    def moments(values):
        if not values:return dict(count=0,mean=None,min=None,max=None)
        return dict(count=len(values),mean=math.fsum(values)/len(values),min=min(values),max=max(values))
    reports=[]
    for case in ctx['manifest']['cases']:
        rows=[e for e in events if e['caseID']==case['caseID']]; truth=ctx['modules']['audit'].truth(case); methods={}
        for name in ('baseline','candidate'):
            done=[e['methods'][name] for e in rows if e['methods'][name]['technicalCompleted']]
            available=[m for m in done if m['availableCount']==4]; family={}
            for kind in ('R','F','D'):
                applicable=kind=='R' or any(v['trueNull'] if kind=='F' else not v['trueNull'] for v in truth.values())
                k=sum(v['events'][kind] is True for v in done); a=sum(v['events'][kind] is True for v in available)
                family[kind]=None if not applicable else dict(count=k,plannedDenominator=ctx['manifest']['B'],technicalCompleted=len(done),
                    rate=k/ctx['manifest']['B'] if len(done)==ctx['manifest']['B'] else None,CI=cp(k,ctx['manifest']['B']) if len(done)==ctx['manifest']['B'] else None,
                    availableCount=a,availableDenominator=len(available),availableRate=a/len(available) if available else None,
                    availableCI=cp(a,len(available)))
            methods[name]=dict(M=ctx['manifest']['B'],T=len(done),A4=len(available),Aj={k:sum(v['tests'][k]['status']=='available' for v in done) for k in IDS},
                technicalUnresolvedPresent=sum(not e['methods'][name]['technicalCompleted'] for e in rows),
                statisticalUnavailableCompleted=sum(v['availableCount']<4 for v in done),family=family)
        paired={}
        for name in ('baseline','candidate'):
            comparable=[e['methods'][name] for e in rows if e['methods'][name]['technicalCompleted']
                        and e['methods'][name]['original']['technicalCompleted']]
            paired[name]=dict(jointCompleted=len(comparable),familyTransitions={},comparisonTransitions={})
            for kind in ('R','F','D'):
                values=[m for m in comparable if m['events'][kind] is not None and m['original'][kind] is not None]
                paired[name]['familyTransitions'][kind]={bits:sum(str(int(m['original'][kind]))+str(int(m['events'][kind]))==bits
                    for m in values) for bits in ('00','01','10','11')}
            paired[name]['comparisonTransitions']={k:{bits:sum(str(int(m['original']['tests'][k]['rejected']))+
                str(int(m['tests'][k]['rejected']))==bits for m in comparable) for bits in ('00','01','10','11')} for k in IDS}
            diagnostics={}
            for key in IDS:
                data=[e['diagnostics'][name][key] for e in rows]
                states={state:sum(d['jointTest']['status']==state for d in data) for state in
                    ('available','statistical_unavailable','technical_unresolved')}
                valid=[d for d in data if d['jointTest']['status']=='available'];t=states['available']+states['statistical_unavailable']
                thresholds={}
                if truth[key]['trueNull']:
                    for level in (.0125,.05/3,.025,.05):
                        k=sum(d['jointTest']['pJoint']<level for d in valid);n=len(valid)
                        thresholds[str(level)]=dict(count=k,plannedDenominator=ctx['manifest']['B'],technicalCompleted=t,
                            rate=k/ctx['manifest']['B'] if t==ctx['manifest']['B'] else None,CI=cp(k,ctx['manifest']['B']) if t==ctx['manifest']['B'] else None,
                            availableDenominator=n,availableRate=k/n if n else None,availableCI=cp(k,n))
                rho=[d['jointTest']['reference']['rho'] for d in valid]
                bracket_counts={};bracket_values={field:[] for field in ('lower','upper','gap')}
                for d in valid:
                    b=d['jointTest']['reference']['referenceBracket'];status='same_sign_or_zero' if b is None else b['numericalStatus']
                    bracket_counts[status]=bracket_counts.get(status,0)+1
                    if b is not None:
                        for field in bracket_values:bracket_values[field].append(b[field])
                old={};nuisance={}
                for linear,target in (('D',truth[key]['D']),('S',truth[key]['S'])):
                    refs=[d[linear] for d in data];av=[r for r in refs if r['status']=='available']
                    old[linear]=dict(statusCounts={status:sum(r['status']==status for r in refs) for status in
                        ('available','statistical_unavailable','technical_unresolved')},
                        scalars={field:moments([r[field] for r in av]) for field in
                            ('estimate','SE','df','statistic','lessP','greaterP','twoSidedP')},
                        coveredCount=sum(r['CI'][0]<=target<=r['CI'][1] for r in av),availableDenominator=len(av))
                    nuisance[linear]=dict(applicable=target!=0.,zeroTruthCoordinate=target==0.,
                        values=moments([abs(target)/r['SE'] for r in av]) if target!=0. else moments([]))
                diagnostics[key]=dict(trueNull=truth[key]['trueNull'],trueD=truth[key]['D'],trueS=truth[key]['S'],
                    M=ctx['manifest']['B'],present=len(data),technicalCompleted=t,statusCounts=states,thresholds=thresholds if truth[key]['trueNull'] else None,
                    available=len(valid),rho=moments(rho),rhoSignCounts=dict(negative=sum(v<0 for v in rho),nonnegative=sum(v>=0 for v in rho)),
                    probabilityBracket=dict(statusCounts=bracket_counts,
                    scalars={field:moments(values) for field,values in bracket_values.items()}),
                    nonzeroStandardizedNuisance=nuisance,nuisanceScale='absolute_true_coordinate / original_estimated_SE; descriptive_not_oracle',
                    oldReferences=old,diagnosticSource='admitted_public_joint_and_magnitude_records',
                    pointwiseMC=True,simultaneousMCGuarantee=False)
            methods[name]['comparisonDiagnostics']=diagnostics
            f=methods[name]['family']['F']
            methods[name]['nominalFamilyFAt05'] = None if f is None else (
                'not_established' if f['CI'] is None else 'supported' if f['CI'][1]<=.05 else
                'adverse' if f['CI'][0]>.05 else 'unresolved')
        reports.append(dict(caseID=case['caseID'],case=case,truth=truth,present=len(rows),methods=methods,paired=paired))
    return reports
