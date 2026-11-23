"""Five independently precommitted public-H rules; no personal meta-selector."""
from fractions import Fraction as F
from math import isqrt

FAMILIES=('ES','EO_raw_surplus','PN','AL','GN')
KNOWN={'read_scalar':'1/2','retrieve_rule':'1/4','add_subtract':'1/2','multiply_divide':'1',
       'compare':'1/4','clip':'1/4','common_execution':'53/20'}


def need(ok,why):
    if not ok:raise ValueError(why)


def ledger(actor):
    events=[];counts={}
    for row in actor['trace']:
        op=row['operation'];translated=dict(row)
        # Explicit new known-index response decomposition; no added motor chunk.
        if actor['family']=='ES' and op in ('local_access','native_select'):
            translated['originalOperation']=op
            op={'local_access':'read_scalar','native_select':'compare'}[op]
            translated.update(operation=op,coefficientId=op,
                              decomposition='known_index_access_or_response_selection')
        events.append(translated);counts[op]=counts.get(op,0)+row['count']
    need(counts.get('common_execution')==1,'One motor response')
    known=sum((F(KNOWN[k])*n for k,n in counts.items() if k in KNOWN),F())
    symbols={k:n for k,n in counts.items() if k not in KNOWN}
    return dict(events=events,counts=counts,knownSeconds=str(known),symbols=symbols,
                unknownMember='each_used_unknown_one_second_not_measured',
                oneSecondMemberSeconds=str(known+sum(symbols.values())),knownCoefficients=KNOWN)


def weighted(prob,values):
    lo=hi=F()
    for (a,b),(x,y) in zip(prob,values):
        products=(a*x,a*y,b*x,b*y);lo+=min(products);hi+=max(products)
    return lo,hi


def utility(amount,need):
    numerator=amount-need*1_000_000;scale=1<<80
    k=isqrt(abs(numerator)*scale*scale//1_000_000)
    h=k+(k*k*1_000_000!=abs(numerator)*scale*scale)
    return (-F(2*h,scale),-F(2*k,scale)) if numerator<0 else (F(k,scale),F(h,scale))


def score(root,j,external,cache):
    t=root['task'];need(t['alpha']==.5 and t['loss_multiplier']==2 and t['allocation_denominator']==1000,'Signed square-root task')
    for index in (j,1000-j):
        if index not in cache:
            cache[index]=[utility(t['total_millihours']*index,n) for n in range(1,101)]
    total=[weighted(external.prior(tuple(logs)),cache[index]) for logs,index in zip(root['logsHex'],(j,1000-j))]
    return [str(2*F(t['gpa_baseline'])+F(t['gpa_scale'])*sum(z[k] for z in total)) for k in (0,1)]


def superiority(a,b):
    """Sufficient open strict region of rho>=0 for the explicit1-second member."""
    margin=F(a['GPAInterval'][0])-F(b['GPAInterval'][1])
    slope=F(a['ledger']['oneSecondMemberSeconds'])-F(b['ledger']['oneSecondMemberSeconds'])
    if slope==0:
        return dict(status='all_nonnegative' if margin>0 else 'not_certified',margin=str(margin),timeDifference=str(slope))
    threshold=margin/slope
    if slope>0:
        return dict(status='rho_below' if threshold>0 else 'not_certified',strictThreshold=str(threshold),rhoAtLeast='0')
    return dict(status='rho_above',strictThreshold=str(threshold),rhoAtLeast='0',zeroIncluded=margin>0)


def produce(root, H, external, workspace):
    """Evaluate the five fixed rules once, then reuse each score at eight prices."""
    from .actors import actor
    from .inputs import price_values
    from ..personal_time_inputs import digest
    cache, scores, actors = {}, {}, []
    for family in FAMILIES:
        act = actor(H, family)
        j = act['target']['index']
        context = dict(rootSHA256=digest(root), index=j, utilityPrecisionBits=80,
                       law='canonical_real_exp_dyadic160', algorithm='signed_scalar_pair')
        if str(j) not in scores:
            value = workspace.obtain('score_'+root['id'][0]+'_'+str(j), context,
                                     lambda: score(root, j, external, cache))
            scores[str(j)] = dict(interval=value, origin=dict(
                kind='fresh_or_identical_public_checkpoint_scalar_pair',
                utilityPrecisionBits=80, nativeIndex=j))
        typed = ledger(act)
        G = scores[str(j)]['interval']
        T = F(typed['oneSecondMemberSeconds'])
        prices = [dict(rho=rho, criterionInterval=[str(F(x)-F(rho)*T) for x in G])
                  for rho in price_values()]
        actors.append(dict(family=family, index=j, originalActor=act, ledger=typed,
                           GPAInterval=G, valueSource=scores[str(j)]['origin'], prices=prices))
    return dict(id=root['id'], H=H, HSourceSHA256=digest(H), actors=actors, scores=scores,
                pairwise=[dict(better=a['family'], worse=b['family'], region=superiority(a,b))
                          for a in actors for b in actors if a is not b],
                newReports=0, scope='five_precommitted_noask_rules_not_free_best_of_five',
                unknownMember='each_used_unknown_one_second_not_measured')


def winners(row):
    """Certify winners only against interval upper bounds; do not invent a selector."""
    cells = []
    for p in range(8):
        groups = {}
        for a in row['actors']:
            price = a['prices'][p]
            rho = F(price['rho'])
            key = (a['index'], rho*F(a['ledger']['oneSecondMemberSeconds']))
            groups.setdefault(key, []).append(a)
        boxes = {k: tuple(map(F, members[0]['prices'][p]['criterionInterval']))
                 for k, members in groups.items()}
        certified = [k for k,v in boxes.items()
                     if all(v[0] > other[1] for j,other in boxes.items() if j != k)]
        if len(groups) == 1:
            certified = list(groups)
        winner = certified[0] if len(certified) == 1 else None
        selected = [] if winner is None else [a['family'] for a in groups[winner]]
        cells.append(dict(id=row['id'], rho=row['actors'][0]['prices'][p]['rho'],
                          families=selected,
                          status='indeterminate' if winner is None else
                          'unique' if len(selected)==1 else 'exact_shared_action_effective_fee_tie',
                          scope='one_second_member_finite_precommitted_menu'))
    return cells
