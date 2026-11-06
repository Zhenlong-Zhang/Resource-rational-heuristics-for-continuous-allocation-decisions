"""Frozen three-method coverage reduction; no generation, fitting or archive I/O."""
import math
from array import array
from collections import Counter

METHODS=('baseline','candidate','within')
COMPARISONS=('C1-C2:A','C1-C2:B','C1-C3:A','C1-C3:B')
LINEAR_IDS=tuple(c+':'+k for c in COMPARISONS for k in ('D','S'))
M={1:10000,2:2129}

def require(value, message):
    if not value:
        raise ValueError(message)

def validate_event(event):
    ordinal=event['ordinal'];require(type(ordinal) is int and 0<=ordinal<12129,'Invalid ordinal')
    index=1 if ordinal<10000 else 2;rep=ordinal+1 if index==1 else ordinal-9999
    require(event['caseIndex']==index and event['repID']==rep and set(event['methods'])==set(METHODS)
        and event['magnitudeTrueNull']=={k:index==1 for k in COMPARISONS}
        and event['methods']['within']['conditionedOnOriginalGuard'] is False,'Changed event identity/guard boundary')
    require(set(event['actualK'])=={'C1','C2','C3'} and all(type(v) is int and 0<=v<=193 for v in event['actualK'].values()),'Invalid actual K')
    for method in event['methods'].values():
        require(set(method['references'])==set(LINEAR_IDS),'Changed eight-reference roster')
        family=method['composition'];require(family['familySize']==4 and family['alpha']==.05
            and family['comparisonIDs']==list(COMPARISONS) and set(family['tests'])==set(COMPARISONS),'Changed fixed Holm family')
        for ref in method['references'].values():
            require(ref['status'] in ('available','statistical_unavailable','technical_unresolved')
                and (type(ref['covered']) is bool or ref['covered'] is None),'Invalid availability')
            if ref['status']=='available':
                require(all(type(ref[k]) in (int,float) and math.isfinite(ref[k]) for k in ('estimate','SE','df','truth'))
                    and ref['SE']>0 and ref['df']>0 and len(ref['CI'])==2 and all(math.isfinite(x) for x in ref['CI'])
                    and ref['CI'][0]<ref['CI'][1],'Invalid available interval')
                require((ref['covered'] is None)==ref['coverageNumericallyUnresolved'],'Changed unresolved endpoint')
            else:require(ref['covered'] is None and ref['CI'] is None,'Unknown interval supplied a classification')

def cp(count,n):
    if not n:return None
    from scipy.stats import beta
    return [0. if count==0 else float(beta.ppf(.025,count,n-count+1)),
            1. if count==n else float(beta.ppf(.975,count+1,n-count))]


def proportion(count,n,eligible=True):
    return dict(denominator=n,count=count,rate=count/n if n and eligible else None,
        CP95=cp(count,n) if eligible else None,MCSE=math.sqrt((count/n)*(1-count/n)/n) if n and eligible else None)


def describe(values):
    n=len(values)
    if not n:return dict(n=0,mean=None,sd=None,min=None,max=None,quantiles={k:None for k in ('p05','p25','p50','p75','p95')})
    mean=math.fsum(values)/n;sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(n-1)) if n>1 else None
    ordered=sorted(values);quantiles={}
    for key,q in (('p05',.05),('p25',.25),('p50',.5),('p75',.75),('p95',.95)):
        at=(n-1)*q;lo=int(at);hi=min(lo+1,n-1);quantiles[key]=ordered[lo]+(at-lo)*(ordered[hi]-ordered[lo])
    return dict(n=n,mean=mean,sd=sd,min=ordered[0],max=ordered[-1],quantiles=quantiles)


def family_events(event,name):
    family=event['methods'][name]['composition'];truth=event['magnitudeTrueNull']
    if family['status']=='technical_unresolved':return dict(R=None,F=None,D=None)
    require(family['status']=='complete','Unknown family status')
    tests=family['tests'];rejected={k:tests[k]['holmReject'] for k in COMPARISONS}
    require(all(type(v) is bool for v in rejected.values()) and family['anyMagnitudeRejects']==any(rejected.values()),'Changed family events')
    return dict(R=any(rejected.values()),F=any(rejected[k] for k in COMPARISONS if truth[k]) if any(truth.values()) else None,
        D=any(rejected[k] for k in COMPARISONS if not truth[k]) if not all(truth.values()) else None)


def family_accumulator():
    return dict(count=Counter(),Aj=Counter(),comparisons={k:Counter() for k in COMPARISONS},
        events={k:Counter() for k in ('R','F','D')})


def add_family(acc,event,name):
    family=event['methods'][name]['composition'];available=family['counts']['available']
    acc['count']['present']+=1;acc['count']['T']+=family['status']=='complete'
    acc['count']['A4']+=available==4;acc['count']['partialAvailable']+=0<available<4
    acc['count']['zeroAvailable']+=available==0
    for key in COMPARISONS:
        test=family['tests'][key];acc['Aj'][key]+=test['status']=='available';row=acc['comparisons'][key]
        row['T']+=test['status']!='technical_unresolved';row['available']+=test['status']=='available'
        row['rejected']+=test['holmReject'] is True
    for key,value in family_events(event,name).items():
        acc['events'][key]['count']+=value is True
        acc['events'][key]['A4count']+=value is True and available==4


def family_report(acc,index,total):
    c=acc['count'];present=c['present'];full=total>0 and present==c['T']==total;events={}
    for key,values in acc['events'].items():
        applies=key=='R' or (key=='F' and index==1) or (key=='D' and index==2)
        events[key]=dict(applicable=applies,all=proportion(values['count'],total) if applies and full else None,
            technical=proportion(values['count'],c['T']) if applies else None,
            available=proportion(values['A4count'],c['A4']) if applies else None)
    return dict(M=total,present=present,T=c['T'],A4=c['A4'],partialAvailable=c['partialAvailable'],zeroAvailable=c['zeroAvailable'],
        technicalUnknown=present-c['T'],Aj={k:acc['Aj'][k] for k in COMPARISONS},
        comparisons={k:dict(T=acc['comparisons'][k]['T'],available=acc['comparisons'][k]['available'],
            rejected=acc['comparisons'][k]['rejected'],all=proportion(acc['comparisons'][k]['rejected'],total) if full else None)
            for k in COMPARISONS},events=events)


def add_reference_count(c,ref):
    c['present']+=1;c['T']+=ref['status']!='technical_unresolved';c['A']+=ref['status']=='available'
    c['coverageKnown']+=type(ref['covered']) is bool;c['C']+=ref['covered'] is True
    c['statisticalUnavailable']+=ref['status']=='statistical_unavailable'
    c['endpointUnknown']+=ref['coverageNumericallyUnresolved'] is True
    if ref['covered'] is False:
        c['lowerMiss']+=ref['truth']<ref['CI'][0];c['upperMiss']+=ref['truth']>ref['CI'][1]


def reference_counts(c,total):
    return dict(M=total,present=c['present'],T=c['T'],A=c['A'],coverageKnown=c['coverageKnown'],C=c['C'],
        missing=total-c['present'],technicalUnknown=c['present']-c['T'],statisticalUnavailable=c['statisticalUnavailable'],
        endpointUnknown=c['endpointUnknown'],conditional=proportion(c['C'],c['coverageKnown']))


class Accumulator:
    """Bounded scalar arrays only; no full event/metadata trees retained."""
    def __init__(self):
        self.seen=set();self.cases={}
        for index in M:
            self.cases[index]=dict(present=0,K={k:array('d') for k in ('C1','C2','C3','total')},guard=Counter(),
                refs={(n,k):dict(count=Counter(),reasons=Counter(),values={x:array('d') for x in ('estimate','error','SE','df','width')}) for n in METHODS for k in LINEAR_IDS},
                families={n:dict(family_accumulator(),strata={}) for n in METHODS},
                pairs={k:dict(table=Counter(),excluded=[],values={x:array('d') for x in ('estimateDifference','widthDifference','widthRatio','squaredErrorDifference')}) for k in LINEAR_IDS},
                transitions={k:Counter() for k in ('R','F','D')})

    def add(self,event):
        validate_event(event);ordinal=event['ordinal'];require(ordinal not in self.seen,'Duplicate final ordinal');self.seen.add(ordinal)
        index=event['caseIndex'];case=self.cases[index];case['present']+=1
        guard=event['originalGuard'];stratum='no_fitted_object' if guard is None else ('passed' if all(guard.values()) else 'failed')
        case['guard'][stratum]+=1
        if guard is not None:
            for key,value in guard.items():
                require(type(value) is bool,'Malformed original guard');case['guard']['false:'+key]+=int(not value)
        for key,values in case['K'].items():values.append(sum(event['actualK'].values()) if key=='total' else event['actualK'][key])
        for name in METHODS:
            method=event['methods'][name];acc=case['families'][name];add_family(acc,event,name)
            strata=acc['strata'].setdefault(stratum,dict(family=family_accumulator(),references={k:Counter() for k in LINEAR_IDS}))
            add_family(strata['family'],event,name)
            for key,ref in method['references'].items():
                row=case['refs'][(name,key)];c=row['count'];add_reference_count(c,ref);add_reference_count(strata['references'][key],ref)
                if ref['status']=='available':
                    for field,value in dict(estimate=ref['estimate'],error=ref['estimate']-ref['truth'],SE=ref['SE'],df=ref['df'],width=ref['CI'][1]-ref['CI'][0]).items():row['values'][field].append(value)
                if ref['reason'] is not None:row['reasons'][str(ref['reason'])]+=1
        left=event['methods']['candidate'];right=event['methods']['within']
        for key in LINEAR_IDS:
            a=left['references'][key];b=right['references'][key];pair=case['pairs'][key]
            if type(a['covered']) is bool and type(b['covered']) is bool:pair['table'][str(int(a['covered']))+str(int(b['covered']))]+=1
            else:pair['excluded'].append(dict(ordinal=ordinal,leftStatus=a['status'],rightStatus=b['status'],
                leftEndpointUnknown=a['coverageNumericallyUnresolved'],rightEndpointUnknown=b['coverageNumericallyUnresolved']))
            if a['status']==b['status']=='available':
                aw=a['CI'][1]-a['CI'][0];bw=b['CI'][1]-b['CI'][0]
                for field,value in dict(estimateDifference=b['estimate']-a['estimate'],widthDifference=bw-aw,widthRatio=bw/aw,
                    squaredErrorDifference=(b['estimate']-b['truth'])**2-(a['estimate']-a['truth'])**2).items():pair['values'][field].append(value)
        le=family_events(event,'candidate');re=family_events(event,'within')
        for key in le:
            if type(le[key]) is bool and type(re[key]) is bool:case['transitions'][key][str(int(le[key]))+str(int(re[key]))]+=1
            elif (index==1 and key!='D') or (index==2 and key!='F'):case['transitions'][key]['unknown']+=1

    def report(self):
        cases={}
        for index,case in self.cases.items():
            total=M[index];present=case['present'];methods={};pairs={}
            for name in METHODS:
                refs={}
                for key in LINEAR_IDS:
                    row=case['refs'][(name,key)];c=row['count'];v=row['values'];n=len(v['estimate'])
                    estimates=describe(v['estimate']);errors=describe(v['error']);ses=describe(v['SE']);rms=math.sqrt(math.fsum(x*x for x in v['SE'])/n) if n else None
                    full=c['coverageKnown']==total and present==total;interval=cp(c['C'],total) if full else None
                    refs[key]=dict(M=total,present=present,T=c['T'],A=c['A'],coverageKnown=c['coverageKnown'],C=c['C'],missing=total-present,
                        technicalUnknown=present-c['T'],statisticalUnavailable=c['statisticalUnavailable'],endpointUnknown=c['endpointUnknown'],
                        lowerMiss=c['lowerMiss'],upperMiss=c['upperMiss'],reasons=dict(sorted(row['reasons'].items())),
                        conditional=proportion(c['C'],c['coverageKnown']),full=proportion(c['C'],total) if full else None,
                        plannedClassificationBounds=[c['C']/total,(c['C']+total-c['coverageKnown'])/total],
                        pointwiseNominalAssessment='unresolved' if interval is None else 'undercoverage' if interval[1]<.95 else 'overcoverage' if interval[0]>.95 else 'inconclusive',
                        moments=dict(n=n,meanEstimate=estimates['mean'],bias=errors['mean'],biasMCSE=errors['sd']/math.sqrt(n) if n>1 else None,
                            empiricalSD=estimates['sd'],meanSE=ses['mean'],sdSE=ses['sd'],RMSSE=rms,
                            RMSSEOverEmpiricalSD=rms/estimates['sd'] if n>1 and estimates['sd']>0 else None,
                            signedError=errors,SE=ses,width=describe(v['width']),df=describe(v['df'])))
                acc=case['families'][name];strata={}
                for key,value in sorted(acc['strata'].items()):
                    n=value['family']['count']['present']
                    strata[key]=dict(present=n,diagnosticConditioningOnly=True,
                        references={k:reference_counts(c,n) for k,c in value['references'].items()},
                        family=family_report(value['family'],index,n))
                methods[name]=dict(references=refs,family=family_report(acc,index,total),guardStrata=strata)
            for key,pair in case['pairs'].items():
                table={k:pair['table'][k] for k in ('00','01','10','11')};q=sum(table.values());g,l=table['01'],table['10']
                delta=(g-l)/q if q else None;discordance=(g+l)/q if q else None;primary=index==1 and key=='C1-C2:B:D';full=present==total and q==total
                exact=None
                if primary and full:
                    from scipy.stats import binomtest
                    exact=float(binomtest(g,g+l,.5,alternative='two-sided').pvalue) if g+l else 1.
                decision=('relative_benefit' if delta>0 else 'relative_harm') if exact is not None and exact<.05 else 'unresolved' if primary else None
                pairs[key]=dict(M=total,present=present,Q=q,table=table,excluded=pair['excluded'],delta=delta,discordance=discordance,
                    discordanceCP95=cp(g+l,q),MCSE=math.sqrt(max(0.,discordance-delta*delta)/(q-1)) if q>1 else None,
                    primary=primary,fullLawEligible=full,primaryUnresolved=primary and not full,exactMcNemarP=exact,primaryDecision=decision,
                    continuousN=len(pair['values']['widthRatio']),**{k:describe(v) for k,v in pair['values'].items()})
            transitions={}
            for key,counts in case['transitions'].items():
                applies=key=='R' or (key=='F' and index==1) or (key=='D' and index==2)
                transitions[key]=dict(applicable=applies,table={k:counts[k] for k in ('00','01','10','11')} if applies else None,
                    Q=sum(counts[k] for k in ('00','01','10','11')) if applies else None,unknown=counts['unknown'] if applies else None,
                    left='candidate',right='within')
            cases[str(index)]=dict(M=total,present=present,methods=methods,pairedWithinMinusCR2=pairs,familyTransitions=transitions,
                retainedK={k:describe(v) for k,v in case['K'].items()},originalGuard=dict(sorted(case['guard'].items())))
        return dict(schema='h10-within-final-statistics-v1',cases=cases,solePrimary=dict(caseIndex=1,linearID='C1-C2:B:D',
            alternative='two-sided',alpha=.05,multiplicity='one_prespecified_primary_no_Holm3'),
            primarySelectionUnchanged=True,withinConditionedOnOriginalGuard=False,pointwiseIntervalsOnly=True,scientificAdoption=False)
