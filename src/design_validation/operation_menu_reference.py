"""Independent saved-only finite-menu and sufficient-region checker.

No producer arithmetic imports. Structural equality follows saved root/role
operands and typed fees, never overlap of unrelated interval enclosures.
"""
import copy
from fractions import Fraction as F
from .personal_time_inputs import digest

HEX=('0x0.0p+0','0x1.5798ee2308c3ap-27','0x1.0c6f7a0b5ed8dp-20','0x1.4f8b588e368f1p-17','0x1.a36e2eb1c432dp-14','0x1.0624dd2f1a9fcp-10','0x1.9e7c6e43390b7p-9','0x1.47ae147ae147bp-7')
FAMILIES=('ES','EO_raw_surplus','PN','AL','GN')
KNOWN=dict(read_scalar='1/2',retrieve_rule='1/4',add_subtract='1/2',multiply_divide='1',compare='1/4',clip='1/4',common_execution='53/20')

def need(x,m):
    if not x:raise ValueError(m)
def bounds(v):
    a,b=map(F,v);need(a<=b,'Ordered finite accepted bounds');return a,b
def enc(v):return [str(z) for z in v]
def minus(a,b):return a[0]-b[1],a[1]-b[0]
def point(z):return [str(z),str(z)]
def positive_prices(intercept,penalty):
    # Solve the strict linear inequality directly over the nonnegative axis.
    if not penalty:return {'kind':'all_nonnegative' if intercept>0 else 'empty'}
    crossing=intercept/penalty
    if penalty<0:
        return {'kind':'all_nonnegative'} if intercept>0 else dict(kind='above',threshold=str(crossing),lowerClosed=False)
    if intercept<=0:return {'kind':'empty'}
    return dict(kind='below',threshold=str(crossing),lower='0',lowerClosed=True,upperClosed=False)
def condition(g,k,symbols,policy):
    return dict(gpaLowerMargin=str(g),knownSecondsUpperDifference=str(k),unknownCountUpperDifferences={n:str(symbols[n]) for n in sorted(symbols)},extraPersonalSecondsPolicy=policy,
        domains='rho>=0; every_unknown_seconds>=0; extra_personal_seconds_to_this_policy>=0',
        sufficientStrictInequality='gpaLowerMargin > rho*(knownSecondsUpperDifference + sum(unknownCountUpperDifferences[k]*unknown_seconds[k]) + extra_personal_seconds_to_this_policy)',
        scope='sufficient_only_no_symbolic_global_winner_no_game_budget_or_policy_change')

def grant_semantics(root,side=None,report=None):
    """Rebuild displayed scientific grants without the producer's public_h."""
    task=root['task'];display=root['display']
    need(type(task['total_millihours']) is int and type(task['cost_millihours']) is int
         and task['cost_millihours']>=0 and type(task['report_width']) in (int,float)
         and F(task['report_width'])>0, 'Canonical budget and positive finite channel width')
    need(len(display['root'])==2 and all(type(x) is str for x in display['root'])
         and len(display['singleReport'])==2
         and all(len(row)==100 and all(type(x) is str for x in row) for row in display['singleReport']),
         'Complete original displayed root and hypothetical report means')
    means=list(display['root']);latest=[None,None];history=[];budget=task['total_millihours']
    if side is not None:
        need(side in ('A','B') and type(report) is int and 1<=report<=100, 'Attributed one-report grant')
        slot=root['orderedPhysical'].index(side)
        means[slot]=display['singleReport'][slot][report-1]
        latest[slot]=report;history=[dict(slot=slot,report=report)]
        budget-=task['cost_millihours']
    else:need(report is None, 'Initial grant has no report')
    need(budget>0, 'Original affordable positive remaining resource')
    return dict(budgetInteger=budget,meanStrings=means,resourceString=f'{budget//1000}.{budget%1000:03d}',
                ranges=[dict(lower=1,upper=100),dict(lower=1,upper=100)],
                channel=dict(source='task_width_metadata_not_actor_operand',
                             reportWidthHex=float(task['report_width']).hex()),latest=latest,history=history)

def check_grant(actual,root,side=None,report=None):
    expected=grant_semantics(root,side,report)
    need(set(actual)==set(expected)|{'stateId'} and type(actual['stateId']) is str and bool(actual['stateId'])
         and {k:v for k,v in actual.items() if k!='stateId'}==expected,
         'Exact root/display/budget/channel/history grant; stateId is an opaque provenance label')

def law_context(root):
    """Bind the canonical law inputs/semantics, not a new law evaluation."""
    return dict(schema='operation-menu-law-context-v1',rootSHA256=digest(root),logsHex=root['logsHex'],
                widthHex=float(root['task']['report_width']).hex(),precisionBits=160,
                law='normalized_real_exp_of_admitted_binary64_logs_and_width_dyadic160',
                support=[1,100],latentSpecificRowNormalization=True)

def source_menu(bundle,source):
    condition_id=source['id'][0];raw=bundle['rawCheap'][condition_id];root=source['root']
    need(root['id']==raw['id']==source['id'] and source['rootSHA256']==digest(root)
         and raw['HSourceSHA256']==digest(raw['H'])
         and raw['H']['budgetInteger']==root['task']['total_millihours']
         and raw['H']['meanStrings']==root['display']['root'],'Same original initial-root budget and visible inputs')
    need(raw['H']['history']==[] and raw['H']['latest']==[None,None], 'Initial public H, not a post-query grant')
    need(root['task']['alpha']==.5 and root['task']['loss_multiplier']==2
         and root['task']['allocation_denominator']==1000 and sorted(root['orderedPhysical'])==['A','B'],
         'Same admitted task/payoff and physical mapping')
    check_grant(raw['H'],root)
    need(source['lawContext']==law_context(root), 'Common canonical initial-root law inputs and semantics')
    need(raw['newReports']==0 and raw['scope']=='five_precommitted_noask_rules_not_free_best_of_five'
         and raw['unknownMember']=='each_used_unknown_one_second_not_measured', 'Saved public noask scope')
    need([a['family'] for a in raw['actors']]==list(FAMILIES),'Complete five precommitted noask rules')
    result=[]
    for a in raw['actors']:
        ledger=a['ledger'];need(ledger['knownCoefficients']==KNOWN,'Original noask operation-seconds units')
        need(a['GPAInterval']==raw['scores'][str(a['index'])]['interval'],'Same-index shared saved score source')
        need([z['rho'] for z in a['prices']]==bundle['rhoValues'], 'Same noask price identities')
        result.append(dict(id='noask:'+a['family'],kind='noask',family=a['family'],index=a['index'],budget=root['task']['total_millihours'],GPAInterval=a['GPAInterval'],knownSecondsInterval=point(ledger['knownSeconds']),symbolCountIntervals={n:point(v) for n,v in ledger['symbols'].items()},memberSecondsInterval=point(ledger['oneSecondMemberSeconds']),sourceRaw=bundle['sourceEvidence']['cheap']['producers'][condition_id]['sha256'],knownCoefficients=KNOWN))
    for side in ('A','B'):
        d=next(z for z in bundle['rawAskDirections'] if z['id']==source['id'] and z['physicalRecipient']==side)
        slot=root['orderedPhysical'].index(side)
        need(d['orderedSlot']==slot and d['nextBudget']==root['task']['total_millihours']-root['task']['cost_millihours']
             and d['scope']=='full_exante_precommitted_direction_not_reached_posterior'
             and d['actualReportsDrawn']==0 and d['newPolicyTrajectories']==0
             and d['unknownMember']=='each_used_unknown_one_second_not_measured'
             and [z['rho'] for z in d['prices']]==bundle['rhoValues']
             and [z['report'] for z in source['askRows'][side]]==list(range(1,101)), 'Same initial-root full exante query, not reached conditional value')
        need([z['report'] for z in d['reports']]==list(range(1,101)), 'Full public saved report roster')
        compact=[]
        for z in d['reports']:
            need(z['id']==source['id'] and z['physicalRecipient']==side and z['orderedSlot']==slot,
                 'Saved report/root/physical role join')
            check_grant(z['H'],root,side,z['report'])
            need(z['HSHA256']==digest(z['H']), 'Report grant hash commits the actual unchanged H')
            ledger=z['ledger']
            need(ledger['knownCoefficients']==dict(KNOWN,request_execution='53/20'),
                 'Original saved query/response operation-seconds units')
            compact.append(dict(report=z['report'],queriedVectorRef=z['queriedVectorRef'],
                                otherVectorRef=z['otherVectorRef'],
                                ledger={k:ledger[k] for k in ('knownSeconds','symbols','oneSecondMemberSeconds')},
                                rawSHA256=digest(z)))
        need(source['askRows'][side]==compact, 'Exact compact projection of public saved reports')
        result.append(dict(id='ask:'+side,kind='ask_once',physicalRecipient=side,orderedSlot=slot,budget=d['nextBudget'],GPAInterval=d['expectedGPAInterval'],knownSecondsInterval=d['expectedKnownSecondsInterval'],symbolCountIntervals=d['expectedSymbolCountsIntervals'],memberSecondsInterval=d['expectedMemberSecondsInterval'],sourceRaw=bundle['sourceEvidence']['ask']['producer']['sha256'],knownCoefficients=dict(KNOWN,request_execution='53/20')))
    need(source['menu']==result,'Independent normalization of original accepted scalar evidence')
    for x in result:
        for key in ('GPAInterval','knownSecondsInterval','memberSecondsInterval'):bounds(x[key])
        need(bounds(x['knownSecondsInterval'])[0]>=0 and bounds(x['memberSecondsInterval'])[0]>=0
             and all(bounds(z)[0]>=0 for z in x['symbolCountIntervals'].values()),'Nonnegative personal cost/count bounds')
    return result

def shared(root,first,second):
    eqG=eqT=eqS=False;why=[]
    if first['kind']=='noask' and second['kind']=='noask':
        eqG=(first['index'],first['budget'],first['GPAInterval'])==(second['index'],second['budget'],second['GPAInterval'])
        left,right=bounds(first['memberSecondsInterval']),bounds(second['memberSecondsInterval'])
        eqT=left==right and left[0]==left[1]
        eqS=first['knownSecondsInterval']==second['knownSecondsInterval'] and first['symbolCountIntervals']==second['symbolCountIntervals']
        if eqG:why=['same_initial_root_native_budget_index_and_shared_saved_score']
    elif first['kind']=='ask_once' and second['kind']=='ask_once' and sorted([first['orderedSlot'],second['orderedSlot']])==[0,1]:
        x,y=root['askRows'][first['physicalRecipient']],root['askRows'][second['physicalRecipient']]
        symmetry=(root['root']['logsHex'][0]==root['root']['logsHex'][1] and first['budget']==second['budget'] and [z['report'] for z in x]==[z['report'] for z in y]==list(range(1,101)))
        if symmetry:
            eqG=all((a['queriedVectorRef'],a['otherVectorRef'])==(b['queriedVectorRef'],b['otherVectorRef']) for a,b in zip(x,y))
            eqT=all(a['ledger']['oneSecondMemberSeconds']==b['ledger']['oneSecondMemberSeconds'] for a,b in zip(x,y))
            eqS=all((a['ledger']['knownSeconds'],a['ledger']['symbols'])==(b['ledger']['knownSeconds'],b['ledger']['symbols']) for a,b in zip(x,y))
            if eqG:why=['exact_equal_input_priors_shared_channel_payoff_and100role_operand_exchange']
    return dict(sameUnderlyingG=eqG,sameMemberTime=eqT,sameSymbolicTime=eqS,evidence=why)

def checked_pair(root,a,b,rhos):
    proof=shared(root,a,b)
    G=(F(0),F(0)) if proof['sameUnderlyingG'] else minus(bounds(a['GPAInterval']),bounds(b['GPAInterval']))
    T=(F(0),F(0)) if proof['sameMemberTime'] else minus(bounds(a['memberSecondsInterval']),bounds(b['memberSecondsInterval']))
    K=(F(0),F(0)) if proof['sameSymbolicTime'] else minus(bounds(a['knownSecondsInterval']),bounds(b['knownSecondsInterval']))
    counts={n:(F(0),F(0)) if proof['sameSymbolicTime'] else minus(bounds(a['symbolCountIntervals'].get(n,['0','0'])),bounds(b['symbolCountIntervals'].get(n,['0','0']))) for n in sorted(set(a['symbolCountIntervals'])|set(b['symbolCountIntervals']))}
    cells=[]
    for raw in rhos:
        rho=F(raw);D=(G[0]-rho*T[1],G[1]-rho*T[0])
        exact=proof['sameUnderlyingG'] and (rho==0 or proof['sameMemberTime']) or D==(F(0),F(0))
        if exact:status='provable_tie'
        elif D[0]>0:status='first_strictly_better'
        elif D[1]<0:status='second_strictly_better'
        else:status='overlap'
        extra={'status':'no_positive_extra_time_certificate'}
        if exact:
            extra=dict(status='personal_extra_time_unpriced' if rho==0 else 'tie_only_at_zero_added_time',positiveExtraToFirst=None if rho==0 else 'second_strictly_better',positiveExtraToSecond=None if rho==0 else 'first_strictly_better',scope='personal_overhead_only_unchanged_outcome_and_other_policy')
        elif status in ('first_strictly_better','second_strictly_better'):
            who=a['id'] if status=='first_strictly_better' else b['id'];margin=D[0] if who==a['id'] else -D[1]
            extra=dict(status='all_nonnegative_personal_extra_time' if rho==0 else 'strictly_below',policy=who,secondsThreshold=None if rho==0 else str(margin/rho),scope='extra_personal_seconds_only_not_tutoring_budget_debit_or_changed_policy')
        cells.append(dict(rho=raw,differenceInterval=enc(D),status=status,tieBasis='structural_shared_value_and_effective_fee' if exact and proof['sameUnderlyingG'] else 'exact_point_interval' if exact else None,extraTimeCondition=extra))
    tie={'kind':'none_proven'}
    if G==(F(0),F(0)):
        tie=dict(kind='all_nonnegative' if T==(F(0),F(0)) else 'exact_price',rho=None if T==(F(0),F(0)) else '0')
    elif G[0]==G[1] and T[0]==T[1] and T[0]!=0:
        price=G[0]/T[0]
        if price>=0:tie=dict(kind='exact_price',rho=str(price))
    symbolic_tie='none_proven'
    if G==(F(0),F(0)):
        symbolic_tie='all_nonnegative_rho_and_unknown_seconds' if K==(F(0),F(0)) and all(z==(F(0),F(0)) for z in counts.values()) else 'rho_zero_only_all_nonnegative_unknown_seconds'
    return dict(first=a['id'],second=b['id'],identityProof=proof,gpaDifferenceInterval=enc(G),memberTimeDifferenceInterval=enc(T),
        oneSecondMemberRhoConditions=dict(firstStrict=positive_prices(G[0],T[1]),secondStrict=positive_prices(-G[1],-T[0]),provableTie=tie,scope='sufficient_open_strict_regions; complement_not_a_tie_certificate'),
        symbolicConditions=dict(firstStrict=condition(G[0],K[1],{k:z[1] for k,z in counts.items()},a['id']),secondStrict=condition(-G[1],-K[0],{k:-z[0] for k,z in counts.items()},b['id']),provableTieScope=symbolic_tie),prices=cells)


def best_sets(candidates,pairs,rhos):
    out=[]
    for position,rho in enumerate(rhos):
        # Equivalence components follow only proved equality, never overlap.
        labels={c['id']:c['id'] for c in candidates}
        for pair in pairs:
            if pair['prices'][position]['status']!='provable_tie':continue
            old,new=labels[pair['second']],labels[pair['first']]
            for key in labels:
                if labels[key]==old:labels[key]=new
        components=[]
        for c in candidates:
            group=next((g for g in components if labels[c['id']]==labels[next(iter(g))]),None)
            if group is None:components.append({c['id']})
            else:group.add(c['id'])
        intervals={c['id']:(bounds(c['GPAInterval'])[0]-F(rho)*bounds(c['memberSecondsInterval'])[1],bounds(c['GPAInterval'])[1]-F(rho)*bounds(c['memberSecondsInterval'])[0]) for c in candidates}
        group_bounds=[]
        for group in components:
            box=(max(intervals[n][0] for n in group),min(intervals[n][1] for n in group));need(box[0]<=box[1],'True equality group bounds intersect');group_bounds.append(box)
        winners=[]
        for i,(lo,_) in enumerate(group_bounds):
            if not any(hi>=lo for k,(_,hi) in enumerate(group_bounds) if k!=i):winners.append(i)
        out.append(dict(rho=rho,status='certified_best_within_evaluated_menu' if len(winners)==1 else 'unresolved',bestPolicies=sorted(components[winners[0]]) if len(winners)==1 else [],equivalenceGroups=[sorted(g) for g in components],scope='evaluator_only_not_free_runtime_selection_or_global_optimum'))
    return out


def verify(b,p, *, conditions=('C3','C5')):
    """Check complete selected science, not private execution provenance."""
    rhos=[str(F.from_float(float.fromhex(v))) for v in HEX]
    need(p['schema']=='operation-menu-v1' and p['rhoValues']==b['rhoValues']==rhos,'Exact eight source-bound binary64-price identities')
    need(type(conditions) in (tuple,list) and bool(conditions)
         and list(conditions)==[c for c in ('C3','C5') if c in conditions]
         and len(conditions)==len(set(conditions)), 'Explicit canonical selected scope')
    n=len(conditions)
    need([r['id'] for r in b['roots']]==[[c,0,1] for c in conditions] and len(p['roots'])==n,
         'Exact selected original roots only')
    need(set(b['rawCheap'])==set(conditions)
         and set(b['sourceEvidence']['cheap']['producers'])==set(conditions)
         and [(d['id'],d['physicalRecipient']) for d in b['rawAskDirections']]==
             [([c,0,1],side) for c in conditions for side in ('A','B')], 'Exact raw scientific source roster')
    need(b['sourceEvidenceSHA256']==digest(b['sourceEvidence'])
         and all(b['sourceEvidence']['cheap']['producers'][c]['sha256']==digest(b['rawCheap'][c]) for c in conditions)
         and b['sourceEvidence']['ask']['producer']['sha256']==digest(b['rawAskDirections']),
         'Canonical saved scientific payload commitments, not old receipt hashes')
    need(p['selectedConditions']==list(conditions) and p['fullSelectedScope'] is (n==2)
         and p['counts']==dict(roots=n,policies=7*n,pairs=21*n,pairPriceCells=168*n,menuPriceCells=8*n),
         'Truthful full/subset counts')
    need(p['sourceEvidenceSHA256']==b['sourceEvidenceSHA256'] and p['unknownMember']=='each_used_unknown_one_second_not_measured'
         and p['symbolicScope']=='nonnegative_global_operation_seconds_sufficient_halfspaces_only'
         and p['scope']==('two_selected_roots_seven_precommitted_alternatives_not_free_selector' if n==2 else 'one_selected_root_seven_precommitted_alternatives_not_free_selector')
         and all(p[k]==0 for k in ('newActors','newLaws','newUtilities','newIntegrations','newTrajectories')),'Saved-only scope and illustrative cost member')
    for source,saved in zip(b['roots'],p['roots']):
        candidates=source_menu(b,source)
        need(saved['id']==source['id'] and saved['rootSHA256']==source['rootSHA256'] and saved['policies']==candidates,'Original input and complete frozen seven-item roster')
        pairs=[]
        for i,a in enumerate(candidates):
            for b2 in candidates[i+1:]:pairs.append(checked_pair(source,a,b2,rhos))
        need(saved['pairs']==pairs,'Independent complete pair/strict boundary/tie/extra-time/symbolic inequalities')
        need(saved['menuBestSets']==best_sets(candidates,pairs,rhos),'Independent finite menu equality classes and certified best sets')
    return dict(status='PASS',verifiedRoots=n,verifiedPolicies=7*n,verifiedPairs=21*n,verifiedPairPriceCells=168*n,verifiedMenuPriceCells=8*n,newActors=0,newLaws=0,newUtilities=0,newIntegrations=0,newTrajectories=0,symbolicWinner='not_inferred_sufficient_nonnegative_parameter_conditions_only',fullSelectedScope=n==2)


REQUIRED=('wrong_source','wrong_root','dropped_policy','false_overlap_tie','wrong_pair_sign','wrong_rho_region','wrong_extra_time_policy','wrong_symbolic_inequality','false_menu_winner','missing_pair','wrong_price','conditional_as_exante')
def semantic_checks(b,p, *, conditions=('C3','C5')):
    results={}
    for name in REQUIRED:
        q=copy.deepcopy(p);root=q['roots'][0];pair=root['pairs'][0]
        if name=='wrong_source':q['sourceEvidenceSHA256']='0'*64
        elif name=='wrong_root':root['rootSHA256']='0'*64
        elif name=='dropped_policy':root['policies'].pop()
        elif name=='false_overlap_tie':pair['prices'][0].update(status='provable_tie',tieBasis='overlap_is_equality')
        elif name=='wrong_pair_sign':pair['gpaDifferenceInterval']=['9999','9999']
        elif name=='wrong_rho_region':pair['oneSecondMemberRhoConditions']['firstStrict']={'kind':'forged'}
        elif name=='wrong_extra_time_policy':pair['symbolicConditions']['secondStrict']['extraPersonalSecondsPolicy']=pair['first']
        elif name=='wrong_symbolic_inequality':pair['symbolicConditions']['firstStrict']['knownSecondsUpperDifference']='9999'
        elif name=='false_menu_winner':root['menuBestSets'][0]['bestPolicies']=['uncharged_oracle']
        elif name=='missing_pair':root['pairs'].pop()
        elif name=='wrong_price':q['rhoValues'][0]='-1'
        elif name=='conditional_as_exante':root['policies'][-1]['kind']='conditional_reached_posterior'
        try:verify(b,q,conditions=conditions)
        except (ValueError,KeyError,AssertionError):results[name]='rejected'
        else:raise ValueError('Independent saved-menu mutation accepted: '+name)
    return dict(status='PASS',required=list(REQUIRED),results=results,scope='in_memory_saved_results_only_no_source_writes_or_scientific_kernel')
