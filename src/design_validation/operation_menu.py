"""Exact saved-interval finite-menu algebra; no actor, law or utility calls.

The adapter must supply checked public scientific products. This module does
not infer historical provenance, human durations, or free online selection.
"""
from fractions import Fraction as F
from itertools import combinations
from .personal_time_inputs import digest

HEX=('0x0.0p+0','0x1.5798ee2308c3ap-27','0x1.0c6f7a0b5ed8dp-20','0x1.4f8b588e368f1p-17','0x1.a36e2eb1c432dp-14','0x1.0624dd2f1a9fcp-10','0x1.9e7c6e43390b7p-9','0x1.47ae147ae147bp-7')
FAMILIES=('ES','EO_raw_surplus','PN','AL','GN')
FULL_CONDITIONS=('C3','C5')

def need(x,m):
    if not x:raise ValueError(m)
def box(x):
    a,b=map(F,x);need(a<=b,'Ordered accepted interval');return a,b
def enc(x):return [str(v) for v in x]
def difference(x,y):return x[0]-y[1],x[1]-y[0]
def region(a,b):
    # Exact solution of a-rho*b>0 on rho>=0; sufficient bound, not point truth.
    if b==0:return dict(kind='all_nonnegative' if a>0 else 'empty')
    t=a/b
    if b>0:return dict(kind='below',threshold=str(t),lower='0',lowerClosed=True,upperClosed=False) if t>0 else dict(kind='empty')
    if t<0:return dict(kind='all_nonnegative')
    return dict(kind='above',threshold=str(t),lowerClosed=False)
def identity(root,a,b):
    sameG=False;sameT=False;sameSymbolic=False;why=[]
    if a['kind']==b['kind']=='noask':
        sameG=a['budget']==b['budget'] and a['index']==b['index'] and a['GPAInterval']==b['GPAInterval']
        sameT=box(a['memberSecondsInterval'])==box(b['memberSecondsInterval']) and box(a['memberSecondsInterval'])[0]==box(a['memberSecondsInterval'])[1]
        sameSymbolic=a['knownSecondsInterval']==b['knownSecondsInterval'] and a['symbolCountIntervals']==b['symbolCountIntervals']
        if sameG:why.append('same_initial_root_native_budget_index_and_shared_saved_score')
    elif a['kind']==b['kind']=='ask_once' and {a['orderedSlot'],b['orderedSlot']}=={0,1}:
        rowsA=root['askRows'][a['physicalRecipient']];rowsB=root['askRows'][b['physicalRecipient']]
        sameLaw=root['root']['logsHex'][0]==root['root']['logsHex'][1] and a['budget']==b['budget'] and [z['report'] for z in rowsA]==[z['report'] for z in rowsB]==list(range(1,101))
        if sameLaw:
            sameG=all(x['queriedVectorRef']==y['queriedVectorRef'] and x['otherVectorRef']==y['otherVectorRef'] for x,y in zip(rowsA,rowsB))
            sameT=all(x['ledger']['oneSecondMemberSeconds']==y['ledger']['oneSecondMemberSeconds'] for x,y in zip(rowsA,rowsB))
            sameSymbolic=all(x['ledger']['knownSeconds']==y['ledger']['knownSeconds'] and x['ledger']['symbols']==y['ledger']['symbols'] for x,y in zip(rowsA,rowsB))
            if sameG:why.append('exact_equal_input_priors_shared_channel_payoff_and100role_operand_exchange')
    return dict(sameUnderlyingG=sameG,sameMemberTime=sameT,sameSymbolicTime=sameSymbolic,evidence=why)
def orient_condition(g,k,counts,policy):
    return dict(gpaLowerMargin=str(g),knownSecondsUpperDifference=str(k),unknownCountUpperDifferences={n:str(v) for n,v in sorted(counts.items())},extraPersonalSecondsPolicy=policy,domains='rho>=0; every_unknown_seconds>=0; extra_personal_seconds_to_this_policy>=0',sufficientStrictInequality='gpaLowerMargin > rho*(knownSecondsUpperDifference + sum(unknownCountUpperDifferences[k]*unknown_seconds[k]) + extra_personal_seconds_to_this_policy)',scope='sufficient_only_no_symbolic_global_winner_no_game_budget_or_policy_change')
def pair(root,a,b,rhos):
    proof=identity(root,a,b);G=(F(0),F(0)) if proof['sameUnderlyingG'] else difference(box(a['GPAInterval']),box(b['GPAInterval']))
    T=(F(0),F(0)) if proof['sameMemberTime'] else difference(box(a['memberSecondsInterval']),box(b['memberSecondsInterval']))
    K=(F(0),F(0)) if proof['sameSymbolicTime'] else difference(box(a['knownSecondsInterval']),box(b['knownSecondsInterval']))
    symbols={}
    for n in sorted(set(a['symbolCountIntervals'])|set(b['symbolCountIntervals'])):
        symbols[n]=(F(0),F(0)) if proof['sameSymbolicTime'] else difference(box(a['symbolCountIntervals'].get(n,['0','0'])),box(b['symbolCountIntervals'].get(n,['0','0'])))
    cells=[]
    for raw in rhos:
        rho=F(raw);need(rho>=0,'Nonnegative accepted price');d=(G[0]-rho*T[1],G[1]-rho*T[0]);tie=(proof['sameUnderlyingG'] and (rho==0 or proof['sameMemberTime'])) or d==(F(0),F(0))
        status='provable_tie' if tie else 'first_strictly_better' if d[0]>0 else 'second_strictly_better' if d[1]<0 else 'overlap'
        extra=dict(status='no_positive_extra_time_certificate')
        if tie:extra=dict(status='personal_extra_time_unpriced' if rho==0 else 'tie_only_at_zero_added_time',positiveExtraToFirst=None if rho==0 else 'second_strictly_better',positiveExtraToSecond=None if rho==0 else 'first_strictly_better',scope='personal_overhead_only_unchanged_outcome_and_other_policy')
        if status in ('first_strictly_better','second_strictly_better'):
            winner=a['id'] if status=='first_strictly_better' else b['id'];margin=d[0] if status=='first_strictly_better' else -d[1]
            extra=dict(status='all_nonnegative_personal_extra_time' if rho==0 else 'strictly_below',policy=winner,secondsThreshold=None if rho==0 else str(margin/rho),scope='extra_personal_seconds_only_not_tutoring_budget_debit_or_changed_policy')
        cells.append(dict(rho=raw,differenceInterval=enc(d),status=status,tieBasis='structural_shared_value_and_effective_fee' if tie and proof['sameUnderlyingG'] else 'exact_point_interval' if tie else None,extraTimeCondition=extra))
    tieRegion=dict(kind='none_proven')
    if G==(F(0),F(0)):tieRegion=dict(kind='all_nonnegative' if T==(F(0),F(0)) else 'exact_price',rho=None if T==(F(0),F(0)) else '0')
    elif G[0]==G[1] and T[0]==T[1] and T[0]!=0 and G[0]/T[0]>=0:tieRegion=dict(kind='exact_price',rho=str(G[0]/T[0]))
    return dict(first=a['id'],second=b['id'],identityProof=proof,gpaDifferenceInterval=enc(G),memberTimeDifferenceInterval=enc(T),oneSecondMemberRhoConditions=dict(firstStrict=region(G[0],T[1]),secondStrict=region(-G[1],-T[0]),provableTie=tieRegion,scope='sufficient_open_strict_regions; complement_not_a_tie_certificate'),symbolicConditions=dict(firstStrict=orient_condition(G[0],K[1],{n:v[1] for n,v in symbols.items()},a['id']),secondStrict=orient_condition(-G[1],-K[0],{n:-v[0] for n,v in symbols.items()},b['id']),provableTieScope='all_nonnegative_rho_and_unknown_seconds' if G==(F(0),F(0)) and K==(F(0),F(0)) and all(v==(F(0),F(0)) for v in symbols.values()) else 'rho_zero_only_all_nonnegative_unknown_seconds' if G==(F(0),F(0)) else 'none_proven'),prices=cells)
def menu_sets(candidates,pairs,rhos):
    answers=[]
    for n,rho in enumerate(rhos):
        groups=[{c['id']} for c in candidates]
        for p in pairs:
            if p['prices'][n]['status']=='provable_tie':
                left=next(g for g in groups if p['first'] in g);right=next(g for g in groups if p['second'] in g)
                if left is not right:left.update(right);groups.remove(right)
        order={c['id']:i for i,c in enumerate(candidates)};groups.sort(key=lambda g:min(order[k] for k in g))
        values={c['id']:(box(c['GPAInterval'])[0]-F(rho)*box(c['memberSecondsInterval'])[1],box(c['GPAInterval'])[1]-F(rho)*box(c['memberSecondsInterval'])[0]) for c in candidates}
        bounds=[(max(values[k][0] for k in g),min(values[k][1] for k in g)) for g in groups];need(all(a<=b for a,b in bounds),'Proved equal objective groups have nonempty bound intersection')
        best=[i for i,v in enumerate(bounds) if all(i==j or v[0]>w[1] for j,w in enumerate(bounds))]
        answers.append(dict(rho=rho,status='certified_best_within_evaluated_menu' if len(best)==1 else 'unresolved',bestPolicies=sorted(groups[best[0]]) if len(best)==1 else [],equivalenceGroups=[sorted(g) for g in groups],scope='evaluator_only_not_free_runtime_selection_or_global_optimum'))
    return answers
def produce(b, *, conditions=FULL_CONDITIONS):
    """Evaluate checked saved inputs; an explicit subset is not the full result."""
    need(type(conditions) in (tuple,list) and bool(conditions)
         and list(conditions)==sorted(set(conditions)) and set(conditions)<=set(FULL_CONDITIONS),
         'Explicit canonical C3/C5 selection')
    need([r['id'] for r in b['roots']]==[[c,0,1] for c in conditions], 'Exact selected original roots')
    need(b['rhoValues']==[str(F.from_float(float.fromhex(x))) for x in HEX],
         'Exact eight binary64 price identities')
    need(b['sourceEvidenceSHA256']==digest(b['sourceEvidence']), 'Portable scientific source commitment')
    roots=[]
    for root in b['roots']:
        need(root['rootSHA256']==digest(root['root']) and root['root']['id']==root['id'],
             'Public scientific root content identity')
        c=root['menu'];need(len(c)==7 and len({x['id'] for x in c})==7,'Exact seven-item menu')
        need([x['id'] for x in c]==['noask:'+f for f in FAMILIES]+['ask:A','ask:B'],
             'Original ordered five noask/two ask policy roster')
        need(all(box(z['knownSecondsInterval'])[0]>=0 and box(z['memberSecondsInterval'])[0]>=0 and all(box(v)[0]>=0 for v in z['symbolCountIntervals'].values()) for z in c),'Nonnegative accepted costs/counts')
        pairs=[pair(root,x,y,b['rhoValues']) for x,y in combinations(c,2)];need(len(pairs)==21,'Every unordered pair including directions')
        roots.append(dict(id=root['id'],rootSHA256=root['rootSHA256'],policies=c,pairs=pairs,menuBestSets=menu_sets(c,pairs,b['rhoValues'])))
    n=len(roots)
    return dict(schema='operation-menu-v1',roots=roots,rhoValues=b['rhoValues'],sourceEvidenceSHA256=b['sourceEvidenceSHA256'],unknownMember='each_used_unknown_one_second_not_measured',symbolicScope='nonnegative_global_operation_seconds_sufficient_halfspaces_only',newActors=0,newLaws=0,newUtilities=0,newIntegrations=0,newTrajectories=0,
                scope='two_selected_roots_seven_precommitted_alternatives_not_free_selector' if n==2 else 'one_selected_root_seven_precommitted_alternatives_not_free_selector',
                selectedConditions=list(conditions),fullSelectedScope=tuple(conditions)==FULL_CONDITIONS,
                counts=dict(roots=n,policies=7*n,pairs=21*n,pairPriceCells=168*n,menuPriceCells=8*n))
