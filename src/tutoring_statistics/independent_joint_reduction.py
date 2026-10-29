"""Separate joint family, paired-event and constituent diagnostic reduction."""
import math
from .independent_magnitude_reduction import IDS, require


def base_reduce(ctx, events):
    """Independent fixed-denominator report, including an honest partial prefix."""
    require(len({row['ordinal'] for row in events}) == len(events), 'Duplicate independent event')
    from .independent_magnitude_reduction import case_for, truth
    for row in events:
        case = case_for(dict(manifest=ctx['manifest'], expectedSet=set(ctx['manifest']['expectedOrdinals'])), row['ordinal'])
        require(row['caseID'] == case['caseID'] and row['repID'] == case['repID']
                and row['truth'] == truth(case), 'Changed independent joint identity/truth')
    result = []
    for case in ctx['manifest']['cases']:
        rows = [row for row in events if row['caseID'] == case['caseID']]
        truth = ctx['modules']['audit'].truth(case); reports = {}
        for name in ('baseline','candidate'):
            done = [row['methods'][name] for row in rows if row['methods'][name]['technicalCompleted']]
            available = [v for v in done if v['availableCount'] == 4]
            family = {}
            for key in ('R','F','D'):
                applicable = key == 'R' or any(t['trueNull'] if key == 'F' else not t['trueNull'] for t in truth.values())
                k = sum(v['events'][key] is True for v in done)
                a = sum(v['events'][key] is True for v in available)
                family[key] = None if not applicable else dict(count=k, plannedDenominator=ctx['manifest']['B'],
                    technicalCompleted=len(done), rate=k/ctx['manifest']['B'] if len(done)==ctx['manifest']['B'] else None,
                    CI=ctx['originalReader'].interval(k,ctx['manifest']['B']) if len(done)==ctx['manifest']['B'] else None,
                    availableCount=a, availableDenominator=len(available),
                    availableRate=a/len(available) if available else None,
                    availableCI=ctx['originalReader'].interval(a,len(available)) if available else None)
            reports[name] = dict(M=ctx['manifest']['B'],T=len(done),A4=len(available),
                Aj={key:sum(v['tests'][key]['status']=='available' for v in done) for key in IDS},
                technicalUnresolvedPresent=sum(not row['methods'][name]['technicalCompleted'] for row in rows),
                statisticalUnavailableCompleted=sum(v['availableCount']<4 for v in done),family=family)
        result.append(dict(caseID=case['caseID'],case=case,truth=truth,present=len(rows),methods=reports))
    return result



def reduce_events(ctx,events):
    """Frozen independent M/T/A and CP plus separately written paired counts."""
    reports=base_reduce(ctx,events)
    for report in reports:
        rows=[e for e in events if e['caseID']==report['caseID']];pair={}
        for label in ('baseline','candidate'):
            methods=[e['methods'][label] for e in rows]
            done=[m for m in methods if m['technicalCompleted'] and m['original']['technicalCompleted']]
            trans={}
            for metric in ('R','F','D'):
                counts={key:0 for key in ('00','01','10','11')}
                for m in done:
                    before,after=m['original'][metric],m['events'][metric]
                    if before is not None and after is not None:counts[('%d%d'%(before,after))]+=1
                trans[metric]=counts
            comp={}
            for key in IDS:
                counts={bits:0 for bits in ('00','01','10','11')}
                for m in done:counts['%d%d'%(m['original']['tests'][key]['rejected'],m['tests'][key]['rejected'])]+=1
                comp[key]=counts
            pair[label]=dict(jointCompleted=len(done),familyTransitions=trans,comparisonTransitions=comp)
            diagnostics={}
            def summary(values):
                return dict(count=len(values),mean=math.fsum(values)/len(values) if values else None,
                    min=min(values) if values else None,max=max(values) if values else None)
            def interval(k,n):return ctx['originalReader'].interval(k,n) if n else None
            for key in IDS:
                selected=[e['diagnostics'][label][key] for e in rows]
                statuses={state:0 for state in ('available','statistical_unavailable','technical_unresolved')}
                for d in selected:statuses[d['jointTest']['status']]+=1
                valid=[d for d in selected if d['jointTest']['status']=='available']
                # Use the independently calculated per-fit p for threshold
                # membership. Integer disagreement at a boundary is not hidden
                # by the continuous numerical comparison tolerance.
                independent_p=[m['tests'][key]['pJoint'] for m in methods if m['tests'][key]['status']=='available']
                thresholds={};technical=statuses['available']+statuses['statistical_unavailable']
                if report['truth'][key]['trueNull']:
                    for level in (.0125,.05/3,.025,.05):
                        count=len([p for p in independent_p if p<level]);n=len(independent_p)
                        thresholds[str(level)]=dict(count=count,plannedDenominator=ctx['manifest']['B'],technicalCompleted=technical,
                            rate=count/ctx['manifest']['B'] if technical==ctx['manifest']['B'] else None,CI=interval(count,ctx['manifest']['B']) if technical==ctx['manifest']['B'] else None,
                            availableDenominator=n,availableRate=count/n if n else None,availableCI=interval(count,n))
                brackets={};fields={k:[] for k in ('lower','upper','gap')}
                for d in valid:
                    value=d['jointTest']['reference']['referenceBracket']
                    state='same_sign_or_zero' if value is None else value['numericalStatus']
                    brackets[state]=brackets.get(state,0)+1
                    if value is not None:
                        for field in fields:fields[field].append(value[field])
                originals={};nuisance={}
                for kind in ('D','S'):
                    target=report['truth'][key][kind];refs=[d[kind] for d in selected]
                    active=[r for r in refs if r['status']=='available'];counts={state:0 for state in statuses}
                    for ref in refs:counts[ref['status']]+=1
                    originals[kind]=dict(statusCounts=counts,
                        scalars={field:summary([ref[field] for ref in active]) for field in
                            ('estimate','SE','df','statistic','lessP','greaterP','twoSidedP')},
                        coveredCount=len([r for r in active if r['CI'][0]<=target<=r['CI'][1]]),availableDenominator=len(active))
                    nuisance[kind]=dict(applicable=target!=0.,zeroTruthCoordinate=target==0.,
                        values=summary([abs(target)/ref['SE'] for ref in active]) if target!=0. else summary([]))
                diagnostics[key]=dict(trueNull=report['truth'][key]['trueNull'],trueD=report['truth'][key]['D'],
                    trueS=report['truth'][key]['S'],M=ctx['manifest']['B'],present=len(selected),technicalCompleted=technical,
                    statusCounts=statuses,thresholds=thresholds if report['truth'][key]['trueNull'] else None,available=len(valid),
                    rho=summary([d['jointTest']['reference']['rho'] for d in valid]),
                    rhoSignCounts=dict(negative=len([d for d in valid if d['jointTest']['reference']['rho']<0]),
                        nonnegative=len([d for d in valid if d['jointTest']['reference']['rho']>=0])),
                    probabilityBracket=dict(statusCounts=brackets,scalars={k:summary(values) for k,values in fields.items()}),
                    nonzeroStandardizedNuisance=nuisance,nuisanceScale='absolute_true_coordinate / original_estimated_SE; descriptive_not_oracle',
                    oldReferences=originals,diagnosticSource='admitted_public_joint_and_magnitude_records',
                    pointwiseMC=True,simultaneousMCGuarantee=False)
            report['methods'][label]['comparisonDiagnostics']=diagnostics
            f=report['methods'][label]['family']['F']
            disposition=None
            if f is not None:
                if f['CI'] is None:disposition='not_established'
                elif f['CI'][1]<=.05:disposition='supported'
                elif f['CI'][0]>.05:disposition='adverse'
                else:disposition='unresolved'
            report['methods'][label]['nominalFamilyFAt05']=disposition
        report['paired']=pair
    return reports
