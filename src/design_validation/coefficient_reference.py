"""Independent pair-difference and residualized geometry for saved coefficients."""
from fractions import Fraction as Q

from . import coefficient_diagnostics as P


def encode(value):
    value = Q(value)
    return dict(numerator=str(value.numerator), denominator=str(value.denominator), decimal=float(value))


def unq(value):
    return Q(int(value['numerator']), int(value['denominator']))


def independent_geometry(points):
    """Pair differences; partial leverage by orthogonal residualization."""
    n = len(points)
    if not n:
        return dict(n=0,gapRank=0,partialRank=0,status='empty',members=[])
    differences = [(points[i][0]-points[j][0], points[i][1]-points[j][1])
                   for i in range(n) for j in range(i)]
    aa = sum((a*a for a,b in differences), Q(0))/n
    bb = sum((b*b for a,b in differences), Q(0))/n
    ab = sum((a*b for a,b in differences), Q(0))/n
    ss = sum(((b-a)**2 for a,b in differences), Q(0))/n
    det = aa*bb-ab**2
    centered = [(sum((a-other[0] for other in points),Q(0))/n,
                 sum((b-other[1] for other in points),Q(0))/n) for a,b in points]
    members = []
    for a,b in centered:
        row = dict(gapLeverage=None,gapSlopeWeight=None,partialLeverage=None,partialLoWeight=None)
        if ss:
            row.update(gapLeverage=encode(Q(1,n)+(b-a)**2/ss),gapSlopeWeight=encode((b-a)/ss))
        if det:
            residual = a-ab*b/bb
            conditional_ss = aa-ab**2/bb
            row.update(partialLeverage=encode(Q(1,n)+b*b/bb+residual**2/conditional_ss),
                       partialLoWeight=encode(residual/conditional_ss))
        members.append(row)
    return dict(n=n,gapRank=1+int(ss>0),partialRank=1+(2 if det>0 else int(aa>0 or bb>0)),
                status='full_rank' if ss>0 and det>0 else 'structurally_rank_deficient',
                gapSS=encode(ss),lowerSS=encode(aa),higherSS=encode(bb),crossSS=encode(ab),
                gramDeterminant=encode(det),normalizedDeterminant=encode(det/(aa*bb)) if aa*bb else None,
                correlationSquared=encode(ab**2/(aa*bb)) if aa*bb else None,
                lowerRange=encode(max(p[0] for p in points)-min(p[0] for p in points)),
                higherRange=encode(max(p[1] for p in points)-min(p[1] for p in points)),
                gapRange=encode(max(b-a for a,b in points)-min(b-a for a,b in points)),members=members)


def verify_geometry_invariants(p, coordinates, result):
    n = len(coordinates)
    if not n:
        return
    rows = result['members']
    for name, rank in (('gapLeverage',2),('partialLeverage',3)):
        if rows[0][name] is not None:
            h = [unq(r[name]) for r in rows]
            p.need(all(0<=v<=1 for v in h) and sum(h,Q(0))==rank,'Exact hat diagonal/trace')
    if rows[0]['gapSlopeWeight'] is not None:
        w = [unq(r['gapSlopeWeight']) for r in rows]
        p.need(sum(w,Q(0))==0 and sum((v*(b-a) for v,(a,b) in zip(w,coordinates)),Q(0))==1,
               'Gap weight moments')
    if rows[0]['partialLoWeight'] is not None:
        w = [unq(r['partialLoWeight']) for r in rows]
        p.need(sum(w,Q(0))==0 and sum((v*a for v,(a,b) in zip(w,coordinates)),Q(0))==1 and
               sum((v*b for v,(a,b) in zip(w,coordinates)),Q(0))==0,'Partial weight moments')


def check(p, data, result):
    p.need(result['schema']=='coefficient-design-diagnostics-v1', 'Diagnostic schema')
    p.need(result['counters']=={**p.ZERO,'newDesignSchedules':270},'Actual work counters')
    p.need(result['scope']==dict(endpoints=2700,schedules=270,cells=30,trueNeedTies=66,eligible=2634),'Scope')
    expected_schedule_ids=[p.sid(k) for k in p.schedule_roster()]
    expected_point_ids=[p.eid(k) for k in p.endpoint_roster()]
    p.need([r['scheduleID'] for r in result['schedules']]==expected_schedule_ids,'Schedule ID order/union')
    p.need([r['endpointID'] for r in result['points']]==expected_point_ids,'Point ID order/union')
    point_result={r['endpointID']:r for r in result['points']}
    for key, claimed in zip(p.schedule_roster(),result['schedules']):
        coordinates, outcomes, eligible_ids=[],[],[]
        for main in range(1,11):
            k=(*key,main)
            old=data['endpoints'][k]
            actual=point_result[p.eid(k)]
            p.need(actual['saved']==old,'Changed original endpoint bytes/fields')
            need_first,need_second=old['needs_displayed']
            tied=need_first==need_second
            p.need(actual['exactTrueNeedTie'] is tied,'Exact category tie')
            if tied:
                p.need(actual['derived'] is None and actual['design'] is None,'Tie was filled as an eligible point')
                continue
            lower=Q(min(need_first,need_second)*1000,old['initial_budget_milli'])
            higher=Q(max(need_first,need_second)*1000,old['initial_budget_milli'])
            y=Q(old['index'],1000) if need_first>need_second else Q(1000-old['index'],1000)
            p.need(actual['derived']==dict(lowerNeedOverInitialBudget=encode(lower),
                                           higherNeedOverInitialBudget=encode(higher),
                                           gap=encode(higher-lower),higherNeedShare=encode(y)),
                   'True-need scale/outcome/display mapping')
            coordinates.append((lower,higher));outcomes.append(y);eligible_ids.append(p.eid(k))
        reference=independent_geometry(coordinates)
        verify_geometry_invariants(p,coordinates,reference)
        member_values=reference.pop('members')
        p.need(claimed['geometry']==reference,'Independent exact design geometry')
        for identity,reference_point in zip(eligible_ids,member_values):
            p.need(point_result[identity]['design']==reference_point,'Independent per-point leverage/weight')
        old=data['schedules'][key]
        preserved={name:old[name] for name in ('gap_rank','partial_rank',*p.METRICS,
                                               'initial_mean_slope','initial_mean_eligible')}
        p.need(claimed['saved']==preserved and claimed['coordinate']==list(key) and
               claimed['endpointIDs']==[p.eid((*key,m)) for m in range(1,11)],'Saved fit/roster changed')
        agrees=reference['gapRank']==old['gap_rank'] and reference['partialRank']==old['partial_rank']
        p.need(claimed['exactRankAgreesWithSaved'] is agrees,'Float/exact rank comparison flag')
        p.need(claimed['higherNeedShareRange']==encode(max(outcomes)-min(outcomes)),'Outcome diagnostic range')
    p.need([(c['condition'],c['price']) for c in result['cells']]==
           [(c,price) for c in p.CONDITIONS for price in p.PRICES],'Cell roster')
    for cell in result['cells']:
        k=(cell['condition'],cell['price'])
        for metric in p.METRICS:
            saved=[data['schedules'][(k[0],k[1],s)][metric] for s in range(9)]
            signs={'negative':0,'zero':0,'positive':0}
            for value in saved:
                signs['negative' if value<0 else 'positive' if value>0 else 'zero']+=1
            p.need(cell['metrics'][metric]==dict(accepted=data['cells'][k][metric],savedValues=saved,
                                                minimum=min(saved),maximum=max(saved),literalFloatSigns=signs),
                   'Saved-only cell description changed')



def verify(rows, summary, result):
    data = P.admit(rows, summary)
    check(P, data, result)
    P.need(result['uncertainty'] == 'saved_nine_schedule_sample_sd_not_mcse_or_ci',
           'Preserve uncertainty interpretation')
    return dict(passed=True, verifiedEndpoints=2700, verifiedSchedules=270,
                newDraws=0, newPolicyRuns=0, newFitInvocations=0)
