"""Frozen full-flow midpoint reduction; descriptive CP is not the primary band."""
import math

M=4612
COMPARISONS=('C1-C2:A','C1-C2:B','C1-C3:A','C1-C3:B')
LINEAR_IDS=tuple(key+':'+kind for key in COMPARISONS for kind in ('D','S'))
NULLS=dict(zip(COMPARISONS,(False,True,False,True)))
STATES=('available','statistical_unavailable','technical_unresolved')
def require(value,message):
    if not value:raise ValueError(message)

def proportion(k,n):
    require(type(k) is int and type(n) is int and 0<=k<=n,'Invalid descriptive count')
    if n==0:return dict(count=k,denominator=n,rate=None,MCSE=None,CP95=None)
    from scipy.stats import beta
    rate=k/n
    ci=[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
    require(all(math.isfinite(v) and 0<=v<=1 for v in ci),'Nonfinite descriptive interval')
    return dict(count=k,denominator=n,rate=rate,MCSE=math.sqrt(rate*(1-rate)/n),CP95=ci)


def primary(true,false,M=M):
    require(type(M) is int and M>0 and type(true) is int and type(false) is int and 0<=true<=true+false<=M,'Bad primary counts')
    unknown=M-true-false;eligible=unknown==0;rate=true/M if eligible else None
    decision='unresolved'
    if eligible:
        if 50*true>=41*M:decision='supports_at_least_80_percent'
        elif 50*true<39*M:decision='below_80_percent'
    return dict(estimand='P(at_least_one_true_alternative_rejected)',known=true+false,true=true,false=false,
        unknown=unknown,pHat=rate,hoeffdingBand=[max(0.,rate-.02),min(1.,rate+.02)] if eligible else None,
        epsilon=.02,confidence=.95,threshold=.8,fullFlowEligible=eligible,decision=decision,
        possibleFraction=[true/M,(true+unknown)/M])


def event_fields(event,ordinal):
    require(event['ordinal']==ordinal and event['repID']==ordinal+1 and event['originalGuardUsedForSelection'] is False
        and event['methodStatus'] in STATES and set(event['references'])==set(event['evaluation']['references'])==set(LINEAR_IDS)
        and event['evaluation']['magnitudeTrueNull']==NULLS,'Changed accepted event identity')
    tests=event['composition']['tests'];family=event['composition'];status=event['methodStatus']
    require(family['familySize']==4 and family['alpha']==.05 and family['comparisonIDs']==list(COMPARISONS)
        and set(tests)==set(COMPARISONS),'Changed fixed Holm family')
    available=0;rejected=[]
    for key in COMPARISONS:
        t=tests[key];require(t['status']==status and t['comparisonID']==key,'Changed comparison status')
        if t['status']=='available':
            available+=1
            require(type(t['holmP']) in (int,float) and math.isfinite(t['holmP']) and 0<=t['holmP']<=1
                and t['holmReject'] is (t['holmP']<.05),'Changed accepted strict Holm flag')
            if t['holmReject']:rejected.append(key)
        elif t['status']=='statistical_unavailable':
            require(t['pMagnitude'] is None and t['holmP'] is None and t['holmPaddingP']==1.
                and t['holmReject'] is False,'Statistical non-discovery changed')
        else:require(t['holmReject'] is None and t['holmPaddingP'] is None,'Technical unknown padded')
    known=status!='technical_unresolved'
    flags=dict(R=bool(rejected),F=any(NULLS[k] for k in rejected),D=any(not NULLS[k] for k in rejected)) if known else dict(R=None,F=None,D=None)
    require(event['technicalCompleted'] is known and event['availableComparisons']==available and event['A4'] is (available==4)
        and all(event[k] is v for k,v in flags.items()) and family['anyMagnitudeRejects'] is flags['R'],
        'Changed R/F/D distinction or full-flow status')
    return tests,flags


class Accumulator:
    def __init__(self,planned=M):
        self.M=planned;self.seen=set();self.T=0;self.A4=0;self.stat=0;self.tech=0;self.reasons={}
        self.aj=dict.fromkeys(COMPARISONS,0);self.events={k:[0,0] for k in ('R','F','D')}
        self.refs={k:dict(available=0,coverageKnown=0,covered=0,endpointUnknown=0,statisticalUnavailable=0,
            technicalUnknown=0,reasons={}) for k in LINEAR_IDS}
        self.K={k:dict(n=0,sum=0,minimum=None,maximum=None) for k in ('C1','C2','C3')}
    def add(self,event,ordinal):
        require(type(ordinal) is int and 0<=ordinal<self.M and ordinal not in self.seen,'Duplicate/out-of-range scientific member')
        tests,flags=event_fields(event,ordinal);self.seen.add(ordinal);status=event['methodStatus']
        self.T+=int(event['technicalCompleted']);self.A4+=int(event['A4'])
        self.stat+=int(status=='statistical_unavailable');self.tech+=int(status=='technical_unresolved')
        if event['methodReason'] is not None:self.reasons[event['methodReason']]=self.reasons.get(event['methodReason'],0)+1
        for key,t in tests.items():self.aj[key]+=int(t['status']=='available')
        for key,flag in flags.items():
            if flag is not None:self.events[key][0 if flag else 1]+=1
        for key,v in self.refs.items():
            ref=event['references'][key];ev=event['evaluation']['references'][key];state=ref['status'];covered=ev['covered']
            require(state in STATES and ev['referenceStatus']==state and (covered is None or type(covered) is bool)
                and type(ev['endpointUnresolved']) is bool,'Changed linear availability/coverage type')
            v['available']+=int(state=='available');v['coverageKnown']+=int(type(covered) is bool);v['covered']+=int(covered is True)
            v['endpointUnknown']+=int(ev['endpointUnresolved']);v['statisticalUnavailable']+=int(state=='statistical_unavailable')
            v['technicalUnknown']+=int(state=='technical_unresolved')
            require((state=='available' and ((type(covered) is bool and ev['endpointUnresolved'] is False)
                    or (covered is None and ev['endpointUnresolved'] is True)))
                or (state!='available' and covered is None and ev['endpointUnresolved'] is False),'Unavailable interval supplied coverage')
            if ref['reason'] is not None:v['reasons'][ref['reason']]=v['reasons'].get(ref['reason'],0)+1
        require(set(event['actualK'])==set(self.K),'Changed three-arm K')
        for key,n in event['actualK'].items():
            require(type(n) is int and 0<=n<=193,'Invalid retained K');v=self.K[key];v['n']+=1;v['sum']+=n
            v['minimum']=n if v['minimum'] is None else min(v['minimum'],n);v['maximum']=n if v['maximum'] is None else max(v['maximum'],n)
    def report(self):
        present=len(self.seen);refs={};events={}
        for key,(yes,no) in self.events.items():
            unknown=self.M-yes-no;events[key]=dict(true=yes,false=no,unknown=unknown,conditional=proportion(yes,yes+no),
                full=proportion(yes,self.M) if unknown==0 else None,possibleFraction=[yes/self.M,(yes+unknown)/self.M])
        for key,v in self.refs.items():
            refs[key]=dict(v,missing=self.M-present,conditional=proportion(v['covered'],v['coverageKnown']),
                full=proportion(v['covered'],self.M) if v['coverageKnown']==self.M else None)
        return dict(counts=dict(M=self.M,present=present,received=present,missing=self.M-present,T=self.T,A4=self.A4,Aj=self.aj,
            statisticalUnavailable=self.stat,technicalUnknown=self.tech),reasons=self.reasons,events=events,references=refs,
            retainedK=self.K,primary=primary(*self.events['D'],M=self.M))
