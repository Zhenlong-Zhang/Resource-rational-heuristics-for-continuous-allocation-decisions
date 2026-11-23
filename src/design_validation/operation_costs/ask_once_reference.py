"""Independent signed-law and cost audit; no producer arithmetic imports."""
from fractions import Fraction as F

from . import reference_math as R, actor_reference as A
from .inputs import record, price_values


def verify(root,physical,p,baseline):
    task=root['task']; slot=root['orderedPhysical'].index(physical)
    B=task['total_millihours']-task['cost_millihours']
    R.need(p['id']==root['id'] and p['physicalRecipient']==physical
           and p['orderedSlot']==slot and p['nextBudget']==B
           and p['feeVersion']==R.VERSION and p['actualReportsDrawn']==p['newPolicyTrajectories']==0
           and p['scope']=='full_exante_precommitted_direction_not_reached_posterior'
           and p['unknownMember']=='each_used_unknown_one_second_not_measured', 'Fixed direction and scope')
    priors=[R.law(tuple(x)) for x in root['logsHex']]; ch=R.channel(F(task['report_width']))
    lr=p['law']
    R.need(len(lr['priors'])==2 and len(lr['channel'])==100
           and lr['latentSpecificRowNormalization'] is True and lr['support']==[1,100], 'Full canonical law')
    for saved,ref in zip(lr['priors']+lr['channel'],priors+ch):
        R.proper([R.I(x) for x in saved])
        for a,b in zip(saved,ref):R.contains(a,b,'Independent normalized finite law')
    used=set(); G=[]; T=[]; K=[]; M=[]; symbols={}
    R.need([r['report'] for r in p['reports']]==list(range(1,101)), 'Every report, including tiny mass')
    def vector(key):
        budget,j=map(int,key.split(':'));R.need(budget==B,'Exact after-question budget')
        saved=p['utilityVectors'][key];ref=R.utility(B,j)
        R.need(len(saved)==100,'Complete utility support');used.add(key)
        for a,b in zip(saved,ref):R.contains(a,b,'Independent96-bit utility')
        return [R.I(z) for z in saved]
    for row in p['reports']:
        y=row['report'];rec=record(root,physical,y)
        R.need(all(row[k]==v for k,v in rec.items()),'Exact independently attributed public H')
        ledger=R.ledger(root,rec,row['originalActor'],A)
        R.need(ledger==row['ledger'],'Independent retained-read and once-only fee ownership')
        j=row['originalActor']['target']['index'];indices=[j,1000-j]
        R.need(row['nativeIndex']==j and row['amountIndicesBySlot']==indices
               and row['queriedVectorRef']==f'{B}:{indices[slot]}'
               and row['otherVectorRef']==f'{B}:{indices[1-slot]}','Physical allocation complements')
        u,v=vector(row['queriedVectorRef']),vector(row['otherVectorRef'])
        joint=[R.mul(priors[slot][n],ch[n][y-1]) for n in range(100)]
        mass=R.summed(joint);mass=mass[0],min(F(1),mass[1])
        R.need(mass[1]>0 and len(row['jointIntervals'])==100,'Retain positive-upper tiny branches')
        for a,b in zip(row['jointIntervals'],joint):R.contains(a,b,'Independent report joint')
        R.contains(row['massInterval'],mass,'Independent report mass')
        base,scale=2*F(task['gpa_baseline']),F(task['gpa_scale'])
        expected=R.add(R.mul((base,base),mass),R.mul((scale,scale),
                       R.add(R.dot(joint,u),R.mul(mass,R.dot(priors[1-slot],v)))))
        R.contains(row['weightedGPAInterval'],expected,'Independent signed joint GPA')
        for field,amount in (('weightedKnownSecondsInterval',F(ledger['knownSeconds'])),
                             ('weightedMemberSecondsInterval',F(ledger['oneSecondMemberSeconds']))):
            R.contains(row[field],R.mul(mass,(amount,amount)),'Weighted actual fee')
        R.need(set(row['weightedSymbolCountsIntervals'])==set(ledger['symbols']),'All unknown coefficients')
        for key,n in ledger['symbols'].items():
            box=row['weightedSymbolCountsIntervals'][key]
            R.contains(box,R.mul(mass,(F(n),F(n))),'Weighted symbolic operation')
            symbols.setdefault(key,[]).append(R.I(box))
        G.append(R.I(row['weightedGPAInterval']));T.append(R.I(row['weightedMemberSecondsInterval']))
        K.append(R.I(row['weightedKnownSecondsInterval']));M.append(R.I(row['massInterval']))
    R.need(used==set(p['utilityVectors']), 'Exact used utility roster')
    g,t,k,m=map(R.summed,(G,T,K,M));R.need(m[0]<=1<=m[1], 'Complete predictive mass')
    R.need(p['expectedGPAInterval']==R.enc(g) and p['expectedMemberSecondsInterval']==R.enc(t)
           and p['expectedKnownSecondsInterval']==R.enc(k) and p['reportMassInterval']==R.enc(m)
           and p['expectedSymbolCountsIntervals']=={key:R.enc(R.summed(v)) for key,v in symbols.items()},
           'Exact full-report reduction')
    prices=[dict(rho=r,criterionInterval=[str(g[0]-F(r)*t[1]),str(g[1]-F(r)*t[0])]) for r in price_values()]
    R.need(p['prices']==prices,'Eight exact affine prices')
    contrasts=[]
    for family in ('ES','EO_raw_surplus'):
        old=next(a for a in baseline['actors'] if a['family']==family)
        for a,b in zip(prices,old['prices']):
            R.need(a['rho']==b['rho'],'Same initial price')
            x,z=R.I(a['criterionInterval']),R.I(b['criterionInterval']);d=(x[0]-z[1],x[1]-z[0])
            contrasts.append(dict(baselineFamily=family,rho=a['rho'],differenceInterval=R.enc(d),
                status='ask_strictly_better' if d[0]>0 else 'baseline_strictly_better' if d[1]<0 else 'indeterminate'))
    R.need(p['matchedComparisons']==contrasts,'Same-root ES/raw contrasts')
    return dict(status='PASS',id=root['id'],physicalRecipient=physical,reportRows=100,
                utilityVectors=len(used),comparisons=len(contrasts),
                generatorRegression=R.generator_regression_reference(),
                scope='independent52digit_law_96bit_scalar_signed_joint_and_fee')
