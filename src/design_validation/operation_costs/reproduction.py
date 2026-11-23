"""Portable phase routing with independent receiving before table publication."""
from concurrent.futures import ProcessPoolExecutor
import csv
from io import StringIO
from pathlib import Path

from ..personal_time_inputs import digest, need, read, write, write_bytes
from ..strategy_cards import external_bounds
from . import inputs as I, noask, noask_reference, ask_once, ask_once_reference
from .checkpoint import Workspace


def units(roots,families,directions):
    rows=[]
    for root in roots:
        c=root['id'][0]
        rows.append(('noask_'+c,'noask',root,None))
        if 'literal' in families:
            rows.extend((f'literal_{c}_{i}','literal',root,rho)
                        for i,rho in enumerate(I.PRICES[:3]))
        if 'ask_once' in families and c in ('C3','C5'):
            rows.extend((f'ask_{c}_{d}','ask_once',root,d) for d in directions)
    return rows


def scientific_context(unit):
    name,family,root,arg=unit
    return dict(unit=name,family=family,rootSHA256=digest(root),argument=arg)


def _process(task):
    output,binding,unit,phase=task
    name,family,root,arg=unit; output=Path(output); w=Workspace(output,binding)
    context=scientific_context(unit);result_path=output/'products'/(name+'.json')
    def baseline():
        return noask.produce(root,I.public_h(root),external_bounds,w)
    if phase=='produce':
        if family=='noask':generate=baseline
        elif family=='ask_once':generate=lambda:ask_once.produce(root,arg,baseline(),w)
        else:
            from . import literal
            generate=lambda:literal.produce(root,arg,workspace=w)
        value=w.obtain('result_'+name,context,generate)
        write(result_path,dict(context=context,result=value))
        return dict(unit=name,phase=phase,newCheckpoints=w.created,reusedCheckpoints=w.reused)
    payload=read(result_path)
    need(payload['context']==context and set(payload)=={'context','result'}, 'Exact product source scope')
    need(w.get('result_'+name,context)==payload['result'], 'Product/checkpoint identity')
    p=payload['result']
    if family=='noask':checked=noask_reference.verify(root,I.public_h(root),p)
    elif family=='ask_once':
        base=read(output/'products'/('noask_'+root['id'][0]+'.json'))['result']
        noask_reference.verify(root,I.public_h(root),base)
        checked=ask_once_reference.verify(root,arg,p,base)
    else:
        from . import literal_reference
        checked=literal_reference.verify(root,arg,p)
    receipt=dict(unit=name,status='PASS',bindingSHA256=digest(binding),
                 productSHA256=digest(payload),independent=checked)
    write(output/'checks'/(name+'.json'),receipt)
    return receipt


def process(output,config,phase,families=('noask','literal','ask_once'),
            directions=('A','B'),workers=1,node=None):
    need(phase in ('produce','check') and type(workers)is int and workers>=1,'Phase and useful workers')
    need(bool(families) and set(families)<= {'noask','literal','ask_once'}
         and len(set(families))==len(families), 'Distinct scientific result families')
    need(bool(directions) and set(directions)<= {'A','B'} and len(set(directions))==len(directions),
         'Distinct fixed physical directions')
    binding,roots=I.load(output,config,node=node)
    jobs=[(str(output),binding,u,phase) for u in units(roots,families,directions)]
    # Initial-root references precede ask-once receiving; each durable output is disjoint.
    batches=([j for j in jobs if j[2][1]=='noask'],[j for j in jobs if j[2][1]!='noask'])
    answers=[]
    for batch in batches:
        if workers==1:answers.extend(_process(j) for j in batch)
        elif batch:
            with ProcessPoolExecutor(max_workers=min(workers,len(batch))) as pool:
                answers.extend(pool.map(_process,batch))
    return dict(phase=phase,units=len(answers),details=answers,
                scope='selected_original_roots_and_precommitted_directions')


def csv_bytes(rows):
    need(bool(rows),'Nonempty table')
    buffer=StringIO(newline=''); writer=csv.DictWriter(buffer,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);return buffer.getvalue().encode()


def tables(output,config,families=('noask','literal','ask_once'),directions=('A','B')):
    output=Path(output);binding,roots=I.load(output,config)
    tables={'noask':[],'menu_winners':[],'pairwise_regions':[],'literal':[],
            'literal_es_comparisons':[],'ask_once':[],'ask_once_contrasts':[]}
    for unit in units(roots,families,directions):
        name,family,root,arg=unit;payload=read(output/'products'/(name+'.json'))
        check=read(output/'checks'/(name+'.json'))
        need(check['status']=='PASS' and check['unit']==name
             and check['bindingSHA256']==digest(binding) and check['productSHA256']==digest(payload)
             and payload['context']==scientific_context(unit), 'Only independently received products')
        p=payload['result'];c=root['id'][0]
        if family=='noask':
            for a in p['actors']:
                for row in a['prices']:
                    tables['noask'].append(dict(condition=c,procedure=a['family'],native_index=a['index'],
                        rho=row['rho'],gpa_lower=a['GPAInterval'][0],gpa_upper=a['GPAInterval'][1],
                        known_seconds=a['ledger']['knownSeconds'],
                        one_second_member_seconds=a['ledger']['oneSecondMemberSeconds'],
                        criterion_lower=row['criterionInterval'][0],criterion_upper=row['criterionInterval'][1]))
            winner_rows=noask.winners(p)
            need(winner_rows==noask_reference.winners(p),'Independent finite-menu readout')
            for row in winner_rows:
                tables['menu_winners'].append(dict(condition=c,rho=row['rho'],status=row['status'],
                                                   procedures='|'.join(row['families']),scope=row['scope']))
            for pair in p['pairwise']:
                region=pair['region']
                tables['pairwise_regions'].append(dict(condition=c,better=pair['better'],worse=pair['worse'],
                    status=region['status'],strict_threshold=region.get('strictThreshold'),
                    zero_included=region.get('zeroIncluded'),margin=region.get('margin'),
                    scope='sufficient_strict_one_second_member_not_exact_switch'))
        elif family=='ask_once':
            for row in p['prices']:
                tables['ask_once'].append(dict(condition=c,direction=arg,rho=row['rho'],
                    gpa_lower=p['expectedGPAInterval'][0],gpa_upper=p['expectedGPAInterval'][1],
                    seconds_lower=p['expectedMemberSecondsInterval'][0],seconds_upper=p['expectedMemberSecondsInterval'][1],
                    criterion_lower=row['criterionInterval'][0],criterion_upper=row['criterionInterval'][1],scope=p['scope']))
            for row in p['matchedComparisons']:
                tables['ask_once_contrasts'].append(dict(condition=c,direction=arg,baseline=row['baselineFamily'],
                    rho=row['rho'],difference_lower=row['differenceInterval'][0],
                    difference_upper=row['differenceInterval'][1],status=row['status']))
        else:
            interval=p.get('externalConditionalExpectedGPAInterval',[None,None])
            criterion=p.get('externalCriterionInterval',[None,None])
            tables['literal'].append(dict(condition=c,rho_hex=arg,status=p['status'],
                questions=sum('report' in s for s in p['steps']),native_index=p.get('target'),
                personal_seconds=p['actualPersonalSeconds'],conditional_gpa_lower=interval[0],
                conditional_gpa_upper=interval[1],criterion_lower=criterion[0],criterion_upper=criterion[1],
                scope='reached_history_conditional_not_exante_after_search'))
            comparison=p['comparison'];delta=comparison.get('costedMinusESCriterionInterval',[None,None])
            tables['literal_es_comparisons'].append(dict(condition=c,rho_hex=arg,
                scope=comparison['comparisonScope'],status=comparison.get('comparison','unresolved'),
                difference_lower=delta[0],difference_upper=delta[1],
                es_seconds=comparison['actualESSeconds']))
    emitted={}
    for name,rows in tables.items():
        if rows:
            write_bytes(output/'tables'/(name+'.csv'),csv_bytes(rows));emitted[name]=len(rows)
    return dict(tables=emitted,status='PASS',scope='checked_selected_units_only')
