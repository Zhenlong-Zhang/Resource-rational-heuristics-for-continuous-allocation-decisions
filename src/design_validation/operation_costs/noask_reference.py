"""Independent actor, fee and signed scalar checks for no-question procedures."""
from collections import Counter
from fractions import Fraction as F

from .actor_reference import actor_reference, FAMILIES
from .reference_math import law, utility, dot, add, mul, contains, need
from .inputs import price_values
from ..personal_time_inputs import digest

KNOWN = dict(read_scalar='1/2', retrieve_rule='1/4', add_subtract='1/2',
             multiply_divide='1', compare='1/4', clip='1/4', common_execution='53/20')


def verify(root, H, row):
    need(row['id']==root['id'] and row['H']==H and row['HSourceSHA256']==digest(H),
         'Original displayed input binding')
    need([a['family'] for a in row['actors']]==list(FAMILIES) and row['newReports']==0,
         'Exactly five no-question rules')
    need(row['scope']=='five_precommitted_noask_rules_not_free_best_of_five'
         and row['unknownMember']=='each_used_unknown_one_second_not_measured', 'No cost or selector scope transfer')
    laws = [law(tuple(x)) for x in root['logsHex']]
    for a in row['actors']:
        family=a['family']; expected=actor_reference(H, family); old=a['originalActor']
        need(old['target']==expected['target'] and old['arithmeticTree']==expected['arithmeticTree']
             and old['counts']==expected['counts'] and old['family']==family
             and a['index']==expected['target']['index']
             and old['algorithmId']=='H-literal-bplusb-v1'
             and old['actorInputPermission']=='H_only' and old['actor_target_certified'] is True,
             'Independent target, arithmetic and actor permission')
        need(all(type(x['count']) is int and x['count']>0 for x in old['trace']),
             'Nonboolean positive exact event counts')
        need([x['operation'] for x in old['trace'] for _ in range(x['count'])]
             == [op for op,count in expected['operationSequence'] for _ in range(count)],
             'Independent complete event sequence')
        events=[]; counts=Counter()
        for pos,event in enumerate(old['trace']):
            prefix=H['stateId']+':'+family
            need(event['id']==prefix+':'+str(pos) and event['stateVersion']==H['stateId']
                 and event['outputRef']==prefix+':value:'+str(pos)
                 and event['coefficientId']==event['operation'] and event['lifetime']=='invocation'
                 and event['inputRef'] and event['inputOrigin']==(
                     'declared_algorithm' if event['operation']=='retrieve_rule' else
                     'task_supplied_H' if event['block']=='input' else 'algorithm_local')
                 and event['block'] in ('input','target','native','response')
                 and (event['operation']!='common_execution' or event['block']=='response'), 'Trace ownership')
            item=dict(event); op=event['operation']
            if family=='ES' and op in ('local_access','native_select'):
                item['originalOperation']=op
                op={'local_access':'read_scalar','native_select':'compare'}[op]
                item.update(operation=op,coefficientId=op,
                            decomposition='known_index_access_or_response_selection')
            counts[op]+=event['count'];events.append(item)
        fixed=sum((F(KNOWN[k])*v for k,v in counts.items() if k in KNOWN),F())
        symbols={k:v for k,v in counts.items() if k not in KNOWN}; T=fixed+sum(symbols.values())
        need(a['ledger']==dict(events=events,counts=dict(counts),knownSeconds=str(fixed),
             symbols=symbols,unknownMember='each_used_unknown_one_second_not_measured',
             oneSecondMemberSeconds=str(T),knownCoefficients=KNOWN), 'Independent noask fee')
        t=root['task']; j=a['index']; B=t['total_millihours']
        g=add(dot(laws[0],utility(B,j)),dot(laws[1],utility(B,1000-j)))
        ref=add((2*F(t['gpa_baseline']),)*2,mul((F(t['gpa_scale']),)*2,g))
        contains(a['GPAInterval'],ref,'Independent real-law signed scalar GPA')
        need(row['scores'][str(j)]['interval']==a['GPAInterval'], 'Shared index score')
        need(a['valueSource']==row['scores'][str(j)]['origin']==dict(
             kind='fresh_or_identical_public_checkpoint_scalar_pair',utilityPrecisionBits=80,nativeIndex=j),
             'Fresh/reused numerical provenance, not a historical acceptance claim')
        need(a['prices']==[dict(rho=r,criterionInterval=[str(F(x)-F(r)*T) for x in a['GPAInterval']])
                           for r in price_values()], 'Every exact affine price')
    expected_pairs=[]
    for a in row['actors']:
        for b in row['actors']:
            if a is b:continue
            margin=F(a['GPAInterval'][0])-F(b['GPAInterval'][1])
            slope=F(a['ledger']['oneSecondMemberSeconds'])-F(b['ledger']['oneSecondMemberSeconds'])
            if slope==0:
                region=dict(status='all_nonnegative' if margin>0 else 'not_certified',
                            margin=str(margin),timeDifference=str(slope))
            elif slope>0:
                region=dict(status='rho_below' if margin/slope>0 else 'not_certified',
                            strictThreshold=str(margin/slope),rhoAtLeast='0')
            else:
                region=dict(status='rho_above',strictThreshold=str(margin/slope),
                            rhoAtLeast='0',zeroIncluded=margin>0)
            expected_pairs.append(dict(better=a['family'],worse=b['family'],region=region))
    need(row['pairwise']==expected_pairs,'Independent twenty sufficient directed regions')
    return dict(status='PASS',id=root['id'],actors=5,priceRows=40,
                scope='independent_targets_fees_96bit_utilities_52digit_law')


def winners(row):
    """Independent strict finite-menu reducer; overlap alone is never an exact tie."""
    result=[]
    for index,rho in enumerate(price_values()):
        groups={}
        for a in row['actors']:
            key=(a['index'],F(rho)*F(a['ledger']['oneSecondMemberSeconds']))
            groups.setdefault(key,[]).append(a)
        candidates=[]
        for key,members in groups.items():
            low=F(members[0]['prices'][index]['criterionInterval'][0])
            if all(low>F(others[0]['prices'][index]['criterionInterval'][1])
                   for other,others in groups.items() if other!=key):candidates.append(key)
        selected=[] if len(candidates)!=1 else [x['family'] for x in groups[candidates[0]]]
        status=('indeterminate' if not selected else 'unique' if len(selected)==1
                else 'exact_shared_action_effective_fee_tie')
        result.append(dict(id=row['id'],rho=rho,families=selected,status=status,
                           scope='one_second_member_finite_precommitted_menu'))
    return result
