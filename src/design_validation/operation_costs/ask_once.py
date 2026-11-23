"""Full ex-ante integration of a precommitted question followed by the raw rule."""
from fractions import Fraction as F

from ..strategy_cards import external_bounds as external
from ..personal_time_inputs import digest
from . import scalars as S
from .actors import actor
from .fees import compose, VERSION
from .inputs import record, price_values


def produce(root, physical, baseline, workspace):
    task=root['task']; sid=root['id']; slot=root['orderedPhysical'].index(physical)
    B=task['total_millihours']-task['cost_millihours']
    S.need(sid in [['C3',0,1],['C5',0,1]] and physical in ('A','B')
           and task['alpha']==.5 and task['loss_multiplier']==2
           and task['allocation_denominator']==1000 and type(B)is int and B>0,
           'Original affordable one-question policy')
    root_hash=digest(root)
    context=dict(rootSHA256=root_hash, logsHex=root['logsHex'],
                 widthHex=float(task['report_width']).hex(), precisionBits=160)

    def make_law():
        return dict(priors=[[S.encode(x) for x in external.prior(tuple(logs))]
                            for logs in root['logsHex']],
                    channel=[[S.encode(x) for x in row]
                             for row in external.channel(F(task['report_width']))],
                    latentSpecificRowNormalization=True,support=[1,100])

    law=workspace.obtain('law_'+sid[0],context,make_law)
    priors=[[S.interval(x) for x in row] for row in law['priors']]
    channel=[[S.interval(x) for x in row] for row in law['channel']]
    for row in priors+channel:S.proper(row)
    vectors={}

    def vector(j):
        key=f'{B}:{j}'
        if key not in vectors:
            ctx=dict(budget=B,amountIndex=j,alpha='1/2',lossMultiplier=2,
                     denominator=1000,support=[1,100],precisionBits=80)
            vectors[key]=workspace.obtain(f'u_{B}_{j}',ctx,
                                         lambda:[S.encode(x) for x in S.utility(B,j)])
        return [S.interval(x) for x in vectors[key]]

    reports=[]
    for y in range(1,101):
        rec=record(root,physical,y)
        ctx=dict(rootSHA256=root_hash,HSHA256=rec['HSHA256'],lawSHA256=digest(law),
                 physicalRecipient=physical,orderedSlot=slot,report=y,feeVersion=VERSION)

        def make_row():
            act=actor(rec['H'],'EO_raw_surplus'); j=act['target']['index']; amounts=[j,1000-j]
            joint,mass=S.report_joint(priors[slot],channel,y)
            ledger=compose(root,rec,act)
            G=S.report_value(task,joint,mass,priors[1-slot],vector(amounts[slot]),vector(amounts[1-slot]))
            K,T=F(ledger['knownSeconds']),F(ledger['oneSecondMemberSeconds'])
            return dict(**rec,originalActor=act,nativeIndex=j,amountIndicesBySlot=amounts,
                        ledger=ledger,jointIntervals=[S.encode(x) for x in joint],massInterval=S.encode(mass),
                        queriedVectorRef=f'{B}:{amounts[slot]}',otherVectorRef=f'{B}:{amounts[1-slot]}',
                        weightedGPAInterval=S.encode(G),
                        weightedKnownSecondsInterval=S.encode(S.times(mass,(K,K))),
                        weightedMemberSecondsInterval=S.encode(S.times(mass,(T,T))),
                        weightedSymbolCountsIntervals={k:S.encode(S.times(mass,(F(n),F(n))))
                                                      for k,n in ledger['symbols'].items()})

        row=workspace.obtain(f'report_{sid[0]}_{physical}_{y:03}',ctx,make_row)
        for key in (row['queriedVectorRef'],row['otherVectorRef']):
            budget,j=map(int,key.split(':'));S.need(budget==B,'Exact vector budget');vector(j)
        reports.append(row)
    result=reduce(root,physical,reports,baseline)
    result.update(law=law,utilityVectors=vectors,reports=reports,feeVersion=VERSION,
                  scalarGeneration='fresh_or_identical_public_checkpoint_80bit_signed_sqrt',
                  actualReportsDrawn=0,newPolicyTrajectories=0)
    return result


def reduce(root,physical,rows,baseline):
    """Reduce complete signed rows without dividing by a tiny event probability."""
    S.need([r['report'] for r in rows]==list(range(1,101)), 'Complete report roster')
    def summed(field):return S.total(S.interval(r[field]) for r in rows)
    M,G,T,K=(summed(f) for f in ('massInterval','weightedGPAInterval',
                                'weightedMemberSecondsInterval','weightedKnownSecondsInterval'))
    S.need(M[0]<=1<=M[1], 'Whole report mass enclosure')
    symbols={}
    for row in rows:
        for key,v in row['weightedSymbolCountsIntervals'].items():
            symbols.setdefault(key,[]).append(S.interval(v))
    prices=[dict(rho=r,criterionInterval=[str(G[0]-F(r)*T[1]),str(G[1]-F(r)*T[0])])
            for r in price_values()]
    comparisons=[]
    for family in ('ES','EO_raw_surplus'):
        old=next(a for a in baseline['actors'] if a['family']==family)
        for a,b in zip(prices,old['prices']):
            S.need(a['rho']==b['rho'],'Matched initial price')
            x,z=S.interval(a['criterionInterval']),S.interval(b['criterionInterval'])
            d=x[0]-z[1],x[1]-z[0]
            comparisons.append(dict(baselineFamily=family,rho=a['rho'],differenceInterval=S.encode(d),
                status='ask_strictly_better' if d[0]>0 else 'baseline_strictly_better' if d[1]<0 else 'indeterminate'))
    return dict(id=root['id'],physicalRecipient=physical,orderedSlot=root['orderedPhysical'].index(physical),
                nextBudget=root['task']['total_millihours']-root['task']['cost_millihours'],
                reportMassInterval=S.encode(M),expectedGPAInterval=S.encode(G),
                expectedMemberSecondsInterval=S.encode(T),expectedKnownSecondsInterval=S.encode(K),
                expectedSymbolCountsIntervals={k:S.encode(S.total(v)) for k,v in symbols.items()},
                prices=prices,matchedComparisons=comparisons,expectedQuestionCount='1',
                scope='full_exante_precommitted_direction_not_reached_posterior',
                unknownMember='each_used_unknown_one_second_not_measured')
