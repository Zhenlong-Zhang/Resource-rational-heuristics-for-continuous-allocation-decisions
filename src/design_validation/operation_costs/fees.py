"""Composite perception ownership: four real post-query fields, no precision input."""
from fractions import Fraction as F
KNOWN={'read_scalar':'1/2','retrieve_rule':'1/4','add_subtract':'1/2','multiply_divide':'1','compare':'1/4','clip':'1/4','common_execution':'53/20','request_execution':'53/20'}
VERSION='ask_once_post4_retained_raw_inputs_v1'


def need(ok,why):
    if not ok:raise ValueError(why)

def compose(root,record,actor):
    H=record['H'];slot=record['orderedSlot'];y=record['report']
    need(actor['family']=='EO_raw_surplus' and H['latest'][slot]==y,'Actual admitted report/raw trace')
    fields={'report_integer':y,'updated_queried_mean':H['meanStrings'][slot],
            'unchanged_other_mean':H['meanStrings'][1-slot],'remaining_budget':H['resourceString']}
    blocks=[dict(block='plan_and_affordability',counts=dict(retrieve_rule=1,read_scalar=2,compare=1),
                 fields=['initial_remaining_budget','question_cost']),
            dict(block='precommitted_label_and_request',counts=dict(read_scalar=1,request_execution=1),fields=['selected_physical_label']),
            dict(block='post_request_four_reads',counts=dict(read_scalar=4),fields=list(fields)),
            dict(block='asked_bit_store_and_sequence_stop',counts=dict(store=1,read_scalar=1,compare=1),fields=['asked_bit'])]
    binding={f'meanStrings[{slot}]':'updated_queried_mean',f'meanStrings[{1-slot}]':'unchanged_other_mean','resourceString':'remaining_budget'}
    events=[];counts={};owned=[]
    for event in actor['trace']:
        row=dict(event);row['feeOwner']='terminal_raw';row['charged']=True
        if event['operation']=='read_scalar':
            need(event['block']=='input' and event['inputRef'] in binding and event['count']==1,'Only duplicate screen reads are reassigned')
            row.update(feeOwner='post_request_four_reads',charged=False,alreadyChargedField=binding[event['inputRef']]);owned.append(event['inputRef'])
        else:counts[event['operation']]=counts.get(event['operation'],0)+event['count']
        events.append(row)
    need(sorted(owned)==sorted(binding) and counts.get('common_execution')==1,'Exactly three retained inputs; one final motor episode')
    for block in blocks:
        for op,n in block['counts'].items():counts[op]=counts.get(op,0)+n
    need(counts['request_execution']==counts['common_execution']==1,'Request and response paid once each')
    known=sum((F(KNOWN[k])*n for k,n in counts.items() if k in KNOWN),F())
    symbols={k:n for k,n in counts.items() if k not in KNOWN}
    return dict(version=VERSION,blocks=blocks,postReadFieldValues=fields,rawReadOwnership=binding,originalTraceWithOwnership=events,
         counts=counts,knownSeconds=str(known),symbols=symbols,oneSecondMemberSeconds=str(known+sum(symbols.values())),
         unknownMember='each_used_unknown_one_second_not_measured',knownCoefficients=KNOWN,
         workingMemoryGrant='same_invocation_retention_paid_local_access_no_persistent_posterior_or_cross_trial_cache',
         updateGrant='task_supplied_display_no_actor_Bayesian_recipe',controlReadScope='four_post_reads_plus_explicit_initial_label_sequence_reads_not_four_total')
