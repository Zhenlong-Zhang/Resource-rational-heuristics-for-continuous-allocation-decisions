"""Independent coverage counting, summaries and intervals from accepted scalar events."""
import math

METHODS=('baseline','candidate','within')
COMPARISONS=('C1-C2:A','C1-C2:B','C1-C3:A','C1-C3:B')
REFS=tuple(key+':'+kind for key in COMPARISONS for kind in ('D','S'))
M={1:10000,2:2129}
STATES=('available','statistical_unavailable','technical_unresolved')
NAMES=['(Intercept)','conditionC1','conditionC3','need_A','need_B','conditionC1:need_A',
       'conditionC3:need_A','conditionC1:need_B','conditionC3:need_B']
SLOPES=['C1:A','C1:B','C2:A','C2:B','C3:A','C3:B']


def require(value,message):
    if not value:raise ValueError(message)


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def compare(actual,expected,label='independent'):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual)==set(expected),label+' keys')
        for key in expected:compare(actual[key],expected[key],label+'.'+str(key))
    elif type(expected) is list:
        require(type(actual) is list and len(actual)==len(expected),label+' length')
        for i,value in enumerate(expected):compare(actual[i],value,label+'['+str(i)+']')
    elif type(expected) is float:
        require(finite(actual) and math.isclose(actual,expected,abs_tol=2e-9,rel_tol=2e-8),label+' finite value')
    else:require(type(actual) is type(expected) and actual==expected,label+' exact value')


def exact_cp(successes,total):
    from scipy.stats import binomtest
    require(type(successes) is int and type(total) is int and 0<=successes<=total,'Invalid integer binomial counts')
    if not total:return None
    interval=binomtest(successes,total).proportion_ci(confidence_level=.95,method='exact')
    return [float(interval.low),float(interval.high)]


def summarize(values):
    n=len(values)
    if not n:return dict(n=0,mean=None,sd=None,min=None,max=None,
        quantiles={key:None for key in ('p05','p25','p50','p75','p95')})
    require(all(finite(value) for value in values),'Nonfinite accepted continuous summary')
    ordered=sorted(values);mean=math.fsum(values)/n
    def quantile(p):
        position=(n-1)*p;lo=math.floor(position);hi=math.ceil(position)
        return float(ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo]))
    return dict(n=n,mean=mean,sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(n-1)) if n>1 else None,
        min=float(ordered[0]),max=float(ordered[-1]),
        quantiles={key:quantile(p) for key,p in (('p05',.05),('p25',.25),('p50',.5),('p75',.75),('p95',.95))})


def proportion(successes,total):
    require(type(successes) is int and type(total) is int and 0<=successes<=total,'Invalid proportion counts')
    rate=successes/total if total else None
    return dict(denominator=total,count=successes,rate=rate,CP95=exact_cp(successes,total),
        MCSE=math.sqrt(rate*(1-rate)/total) if total else None)


def target(case,key,within=False):
    pair,need,kind=key.split(':');a,b=pair.split('-');sign=-1. if kind=='D' else 1.
    names=SLOPES if within else NAMES
    def vector(arm):
        values=[0.]*len(names)
        if within:values[names.index(arm+':'+need)]=1.
        else:
            values[names.index('need_'+need)]=1.
            if arm!='C2':values[names.index('condition'+arm+':need_'+need)]=1.
        return values
    slopes={arm:dict(A=.25,B=-.25) for arm in ('C1','C2','C3')}
    if case==2:
        slopes['C2']=dict(A=.55,B=-.55);slopes['C3']=dict(A=.55,B=-.55)
    return dict(L=[x+sign*y for x,y in zip(vector(a),vector(b))],truth=slopes[a][need]+sign*slopes[b][need])


def audit_event(event):
    ordinal=event['ordinal'];require(type(ordinal) is int and 0<=ordinal<12129,'Wrong independent stable ordinal')
    case=1 if ordinal<10000 else 2;rep=ordinal+1 if case==1 else ordinal-9999
    expected_id='h10wcv1_initial_amount_'+('s1' if case==1 else 's5_l20')+'_recruit193_mcar08'
    require(type(event['caseIndex']) is int and event['caseIndex']==case and type(event['repID']) is int
        and event['repID']==rep and event['caseID']==expected_id,'Wrong fixed law/rep identity')
    require(set(event['actualK'])=={'C1','C2','C3'} and all(type(v) is int and 0<=v<=193 for v in event['actualK'].values()),
        'Changed actual retained population')
    require(set(event['magnitudeTrueNull'])==set(COMPARISONS)
        and all(type(value) is bool and value==(case==1) for value in event['magnitudeTrueNull'].values()),
        'Changed fixed magnitude scientific truth')
    guard=event['originalGuard']
    require(guard is None or (set(guard)=={'converged','nonsingular','hessianOK','gradientOK','finiteBeta','finiteV'}
        and all(type(v) is bool for v in guard.values())),'Changed original guard description')
    require(set(event['methods'])==set(METHODS),'Changed method roster')
    for name,method in event['methods'].items():
        require(method['status'] in ('available','complete','statistical_unavailable','technical_unresolved')
            and set(method['references'])==set(REFS),'Unknown method/reference state')
        if name=='within':require(method['conditionedOnOriginalGuard'] is False,'Within inference was guard-selected')
        for key,ref in method['references'].items():
            wanted=target(case,key,name=='within')
            require(type(ref['L']) is list and all(finite(v) for v in ref['L']) and finite(ref['truth'])
                and ref['L']==wanted['L'] and ref['truth']==wanted['truth'],'Changed independent named L/canonical truth')
            require(ref['status'] in STATES,'Unknown linear-reference state')
            if ref['status']=='available':
                require(all(finite(ref[field]) for field in ('estimate','SE','df','statistic','lessP','greaterP','twoSidedP'))
                    and ref['SE']>0 and ref['df']>0 and all(0<=ref[field]<=1 for field in ('lessP','greaterP','twoSidedP'))
                    and type(ref['CI']) is list and len(ref['CI'])==2 and all(finite(v) for v in ref['CI'])
                    and ref['CI'][0]<ref['CI'][1],'Invalid accepted interval')
                require((ref['coverageNumericallyUnresolved'] is True and ref['covered'] is None)
                    or (ref['coverageNumericallyUnresolved'] is False and type(ref['covered']) is bool),
                    'Changed endpoint unknown classification')
            else:
                require(ref['covered'] is None and ref['CI'] is None and ref['coverageNumericallyUnresolved'] is False
                    and all(ref[field] is None for field in ('estimate','SE','df','statistic','lessP','greaterP','twoSidedP')),
                    'Unknown converted to a scientific number or miss')
        family_events(event,name)  # Pure categorical/strict-threshold audit.


def family_events(event,name):
    method=event['methods'][name];family=method['composition'];tests=family['tests']
    require(family['schema']=='h10-magnitude-family-candidate-v1'
        and family['methodID']=='h10_magnitude_DS_IUT_Bonferroni2_Holm4_candidate_v1'
        and family['familySize']==4 and family['alpha']==.05
        and family['comparisonIDs']==list(COMPARISONS) and set(tests)==set(COMPARISONS),
        'Changed fixed scientific Holm family')
    available=unavailable=technical=0
    for key in COMPARISONS:
        test=tests[key];d=method['references'][key+':D'];s=method['references'][key+':S']
        state=('technical_unresolved' if 'technical_unresolved' in (d['status'],s['status'])
            else 'statistical_unavailable' if 'statistical_unavailable' in (d['status'],s['status']) else 'available')
        require(test['comparisonID']==key and test['status']==state,'Changed comparison availability')
        available+=state=='available';unavailable+=state=='statistical_unavailable';technical+=state=='technical_unresolved'
        if state=='available':
            compare(test['inputP'],dict(D_less=d['lessP'],D_greater=d['greaterP'],S_less=s['lessP'],S_greater=s['greaterP']),
                'Accepted signed tails')
            require(finite(test['pMagnitude']) and 0<=test['pMagnitude']<=1,'Invalid accepted magnitude p')
        else:require(test['pMagnitude'] is None and test['inputP'] is None,'Unavailable supplied a p value')
    compare(family['counts'],dict(planned=4,technicalCompleted=4-technical,available=available,
        statisticalUnavailable=unavailable,technicalUnresolved=technical),'Exact family availability')
    require(family['completeFamilyAvailable']==(available==4),'Changed A4')
    if technical:
        require(family['status']=='technical_unresolved' and family['anyMagnitudeRejects'] is None and family['holmComputed'] is False
            and all(test['holmReject'] is None and test['holmP'] is None for test in tests.values()),
            'Technical unknown became nonrejection')
        result=dict(R=None,F=None,D=None)
    else:
        require(family['status']=='complete' and family['holmComputed'] is True,'Unclosed scientific family')
        decisions={}
        for key,test in tests.items():
            if test['status']=='available':
                require(finite(test['holmP']) and 0<=test['holmP']<=1 and type(test['holmReject']) is bool
                    and test['holmReject']==(test['holmP']<.05),'Changed strict Holm rejection')
            else:require(test['holmP'] is None and test['holmPaddingP']==1. and test['holmReject'] is False,
                'Changed statistical-unavailable convention')
            decisions[key]=test['holmReject']
        result=dict(R=any(decisions.values()),F=any(decisions[k] for k in COMPARISONS if event['magnitudeTrueNull'][k]),
            D=any(decisions[k] for k in COMPARISONS if not event['magnitudeTrueNull'][k]))
        require(type(family['anyMagnitudeRejects']) is bool and family['anyMagnitudeRejects']==result['R'],
            'Changed family-any decision')
    if not any(event['magnitudeTrueNull'].values()):result['F']=None
    if all(event['magnitudeTrueNull'].values()):result['D']=None
    return result


def guard_label(event):
    guard=event['originalGuard']
    return 'no_fitted_object' if guard is None else 'passed' if all(guard.values()) else 'failed'


class Reference:
    def __init__(self):
        self.present=self.T=self.A=self.known=self.C=self.statistical=self.technical=self.endpoint=0
        self.lower=self.upper=0;self.reasons={};self.errors=[];self.ses=[];self.widths=[];self.dfs=[]

    def add(self,event,ref,moments=True):
        self.present+=1;state=ref['status'];self.T+=state!='technical_unresolved'
        self.A+=state=='available';self.statistical+=state=='statistical_unavailable';self.technical+=state=='technical_unresolved'
        self.endpoint+=ref['coverageNumericallyUnresolved'] is True
        if type(ref['covered']) is bool:
            self.known+=1;self.C+=ref['covered']
            self.lower+=ref['covered'] is False and ref['truth']<ref['CI'][0]
            self.upper+=ref['covered'] is False and ref['truth']>ref['CI'][1]
        if ref.get('reason') is not None:
            reason=str(ref['reason']);self.reasons[reason]=self.reasons.get(reason,0)+1
        if moments and state=='available':
            self.errors.append(ref['estimate']-ref['truth']);self.ses.append(ref['SE'])
            self.widths.append(ref['CI'][1]-ref['CI'][0]);self.dfs.append(ref['df'])

    def counts(self,M):
        return dict(M=M,present=self.present,T=self.T,A=self.A,coverageKnown=self.known,C=self.C,missing=M-self.present,
            statisticalUnavailable=self.statistical,technicalUnknown=self.technical,endpointUnknown=self.endpoint,
            conditional=proportion(self.C,self.known))

    def report(self,M,truth):
        full=M>0 and self.present==self.known==M;interval=exact_cp(self.C,M) if full else None;errors=summarize(self.errors)
        rms=math.sqrt(math.fsum(se*se for se in self.ses)/self.A) if self.A else None
        ses=summarize(self.ses)
        return dict(self.counts(M),reasons=dict(sorted(self.reasons.items())),full=proportion(self.C,M) if full else None,
            plannedClassificationBounds=[self.C/M,(self.C+M-self.known)/M],
            pointwiseNominalAssessment=('undercoverage' if interval[1]<.95 else 'overcoverage' if interval[0]>.95 else 'inconclusive') if full else 'unresolved',
            moments=dict(n=self.A,meanEstimate=errors['mean']+truth if self.A else None,bias=errors['mean'],
                biasMCSE=errors['sd']/math.sqrt(errors['n']) if errors['sd'] is not None else None,
                empiricalSD=errors['sd'],meanSE=ses['mean'],sdSE=ses['sd'],RMSSE=rms,
                RMSSEOverEmpiricalSD=rms/errors['sd'] if rms is not None and errors['sd'] else None,
                width=summarize(self.widths),df=summarize(self.dfs),SE=ses,signedError=errors),
            lowerMiss=self.lower,upperMiss=self.upper)


class Paired:
    def __init__(self):
        self.table={key:0 for key in ('00','01','10','11')};self.excluded=[]
        self.width_difference=[];self.width_ratio=[];self.signed_difference=[];self.squared_difference=[]

    def add(self,event,key):
        left=event['methods']['candidate']['references'][key];right=event['methods']['within']['references'][key]
        if type(left['covered']) is bool and type(right['covered']) is bool:
            self.table[str(int(left['covered']))+str(int(right['covered']))]+=1
        else:self.excluded.append(dict(ordinal=event['ordinal'],leftStatus=left['status'],rightStatus=right['status'],
            leftEndpointUnknown=left['coverageNumericallyUnresolved'],rightEndpointUnknown=right['coverageNumericallyUnresolved']))
        if left['status']==right['status']=='available':
            lw=left['CI'][1]-left['CI'][0];rw=right['CI'][1]-right['CI'][0]
            le=left['estimate']-left['truth'];re=right['estimate']-right['truth']
            self.width_difference.append(rw-lw);self.width_ratio.append(rw/lw);self.signed_difference.append(re-le)
            self.squared_difference.append(re*re-le*le)

    def report(self,M,present,primary):
        from scipy.stats import binomtest
        Q=sum(self.table.values());gain=self.table['01'];loss=self.table['10'];discordant=gain+loss
        delta=(gain-loss)/Q if Q else None;q=discordant/Q if Q else None
        full=present==Q==M
        p=(float(binomtest(gain,discordant,.5,alternative='two-sided').pvalue) if discordant else 1.) if primary and full else None
        decision=(p<.05) if primary and full else None
        disposition=('relative_benefit' if decision and delta>0 else 'relative_harm' if decision and delta<0 else 'unresolved') if primary else None
        return dict(M=M,present=present,Q=Q,table=dict(self.table),excluded=self.excluded,
            delta=delta,discordance=q,discordanceCP95=exact_cp(discordant,Q),MCSE=math.sqrt(max(0.,q-delta*delta)/(Q-1)) if Q>1 else None,
            exactMcNemarP=p,fullLawEligible=full,primary=primary,primaryUnresolved=primary and not full,
            primaryDecision=disposition,continuousN=len(self.width_difference),
            estimateDifference=summarize(self.signed_difference),widthDifference=summarize(self.width_difference),
            widthRatio=summarize(self.width_ratio),squaredErrorDifference=summarize(self.squared_difference))


class Family:
    def __init__(self,case):
        self.case=case;self.present=self.T=0;self.hist={str(i):0 for i in range(5)}
        self.events={name:dict(known=0,successes=0) for name in ('R','F','D')};self.reasons=[]
        self.available_successes={key:0 for key in ('R','F','D')}
        self.comparisons={key:dict(T=0,available=0,rejected=0) for key in COMPARISONS}

    def add(self,event,name):
        family=event['methods'][name]['composition'];values=family_events(event,name)
        self.present+=1;self.T+=family['status']=='complete';self.hist[str(family['counts']['available'])]+=1
        if family['status']!='complete':self.reasons.append(dict(ordinal=event['ordinal'],reason=event['methods'][name]['reason']))
        for key,test in family['tests'].items():
            count=self.comparisons[key];count['T']+=test['status']!='technical_unresolved';count['available']+=test['status']=='available'
            count['rejected']+=test['holmReject'] is True
        for key,value in values.items():
            if type(value) is bool:self.events[key]['known']+=1;self.events[key]['successes']+=value
            if type(value) is bool and family['counts']['available']==4:self.available_successes[key]+=value

    def report(self,M):
        reports={}
        for name,value in self.events.items():
            applicable=not ((self.case==1 and name=='D') or (self.case==2 and name=='F'));known=value['known'];successes=value['successes']
            full=applicable and M>0 and self.present==known==M
            reports[name]=dict(applicable=applicable,all=proportion(successes,M) if full else None,
                technical=proportion(successes,known) if applicable else None,
                available=proportion(self.available_successes[name],self.hist['4']) if applicable else None)
        return dict(M=M,present=self.present,T=self.T,technicalUnknown=self.present-self.T,
            A4=self.hist['4'],partialAvailable=sum(self.hist[str(i)] for i in (1,2,3)),zeroAvailable=self.hist['0'],
            Aj={key:value['available'] for key,value in self.comparisons.items()},
            comparisons={key:dict(value,all=proportion(value['rejected'],M) if M>0 and self.present==self.T==M else None)
                for key,value in self.comparisons.items()},events=reports)


class Transition:
    def __init__(self):self.cells={key:0 for key in ('00','01','10','11')}

    def add(self,event,key):
        a=family_events(event,'candidate')[key];b=family_events(event,'within')[key]
        if type(a) is bool and type(b) is bool:self.cells[str(int(a))+str(int(b))]+=1

    def report(self,M,present,applicable):
        Q=sum(self.cells.values())
        return dict(left='candidate',right='within',applicable=applicable,Q=Q if applicable else None,
            unknown=present-Q if applicable else None,
            table=dict(self.cells) if applicable else None)


class Accumulator:
    """Streaming counts and float sequences, never a full nested-event archive."""
    def __init__(self):
        self.seen=set();self.rows={case:0 for case in M}
        self.references={(case,name,key):Reference() for case in M for name in METHODS for key in REFS}
        self.families={(case,name):Family(case) for case in M for name in METHODS}
        self.paired={(case,key):Paired() for case in M for key in REFS}
        self.transitions={(case,key):Transition() for case in M for key in ('R','F','D')}
        self.K={(case,arm):[] for case in M for arm in ('C1','C2','C3','total')}
        self.guards={case:{} for case in M}
        self.guard_refs={(case,name,stratum,key):Reference() for case in M for name in METHODS
            for stratum in ('passed','failed','no_fitted_object') for key in REFS}
        self.guard_families={(case,name,stratum):Family(case) for case in M for name in METHODS for stratum in ('passed','failed','no_fitted_object')}

    def add(self,event):
        audit_event(event);ordinal=event['ordinal'];require(ordinal not in self.seen,'Duplicate received scientific ID')
        self.seen.add(ordinal);case=event['caseIndex'];self.rows[case]+=1;stratum=guard_label(event)
        self.guards[case][stratum]=self.guards[case].get(stratum,0)+1
        if event['originalGuard'] is not None:
            for key,value in event['originalGuard'].items():
                label='false:'+key;self.guards[case][label]=self.guards[case].get(label,0)+int(not value)
        for arm,value in event['actualK'].items():self.K[case,arm].append(value)
        self.K[case,'total'].append(sum(event['actualK'].values()))
        for name in METHODS:
            for key,ref in event['methods'][name]['references'].items():
                self.references[case,name,key].add(event,ref);self.guard_refs[case,name,stratum,key].add(event,ref,moments=False)
            self.families[case,name].add(event,name);self.guard_families[case,name,stratum].add(event,name)
        for key in REFS:self.paired[case,key].add(event,key)
        for key in ('R','F','D'):self.transitions[case,key].add(event,key)

    def report(self):
        result={}
        for case,total in M.items():
            methods={}
            for name in METHODS:
                methods[name]=dict(references={key:self.references[case,name,key].report(total,target(case,key)['truth']) for key in REFS},
                    family=self.families[case,name].report(total),guardStrata={})
                for stratum in ('passed','failed','no_fitted_object'):
                    n=self.guards[case].get(stratum,0)
                    if not n:continue
                    methods[name]['guardStrata'][stratum]=dict(present=n,diagnosticConditioningOnly=True,
                        references={key:self.guard_refs[case,name,stratum,key].counts(n) for key in REFS},
                        family=self.guard_families[case,name,stratum].report(n))
            result[str(case)]=dict(M=total,present=self.rows[case],methods=methods,
                pairedWithinMinusCR2={key:self.paired[case,key].report(total,self.rows[case],case==1 and key=='C1-C2:B:D') for key in REFS},
                familyTransitions={key:self.transitions[case,key].report(total,self.rows[case],
                    not ((case==1 and key=='D') or (case==2 and key=='F'))) for key in ('R','F','D')},
                retainedK={arm:summarize(self.K[case,arm]) for arm in ('C1','C2','C3','total')},originalGuard=self.guards[case])
        return dict(schema='h10-within-final-statistics-v1',cases=result,
            solePrimary=dict(caseIndex=1,linearID='C1-C2:B:D',alternative='two-sided',alpha=.05,
                multiplicity='one_prespecified_primary_no_Holm3'),primarySelectionUnchanged=True,
            withinConditionedOnOriginalGuard=False,pointwiseIntervalsOnly=True,scientificAdoption=False)
