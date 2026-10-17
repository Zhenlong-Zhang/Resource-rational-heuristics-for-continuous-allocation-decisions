"""Same-input Student and projection attribution, not calibration acceptance."""
from __future__ import annotations

import io
import time

import numpy as np

from . import calibration as cal
from . import nominal_inference as nominal
from . import inference as legacy
from .paired_recipe import (BATCHES, SIZE, SPEC, ORIGINAL_RUNTIME, REFERENCE_RUNTIME,
                            fixed_cell, fresh_directory, runtime, scope,
                            selected_batches, strict_json, write_bytes, write_json)

LIMITATIONS = [
 '500 fixed original-seed cases are not added to historical10000; no Type-I repair criterion.',
 'Current runtime differs from historical3.12.14/NumPy2.3.5/SciPy1.16.3; no historical bit replay.',
 'Abstract unbounded Gamma stress inputs are not native-task or human observations.',
 'Frozen full generator is shared input-law primitive; independent reader recomputes projection and Student statistics.',
 'All historical111 failures and accepted formal89/22/7 dispositions remain unchanged.',
 'No estimator correction is implemented; agreement ends supplementary computation with skewness limitation.',
]

def people(values):
    return [dict(participantId='C4:%02d'%j,conditionId='C4',
             endpoints={'initial_slope':dict(status='available',value=float(v))},
             auditOrigin='historical_abstract_summary_audit') for j,v in enumerate(values)]

def legacy_people(values):
    return [dict(participantId='C4:%02d'%j,conditionId='C4',status='available',value=float(v))
            for j,v in enumerate(values)]

def negative_tests():
    names = []

    def rejects(name, call):
        try:
            call()
        except (ValueError, TypeError):
            names.append(name)
            return
        raise AssertionError("Negative test failed: " + name)

    rejects("duplicate_json_key", lambda: strict_json('{"x":1,"x":2}'))
    rejects("nonfinite_json", lambda: strict_json('{"x":NaN}'))
    rejects("overflow_json_decimal", lambda: strict_json('{"x":1e999}'))
    rejects("changed_designated_tail", lambda: nominal.test_contrast(
        people([0., 1., 2.]), {**SPEC, "tail": "greater"}))
    rejects("duplicate_independent_person", lambda: nominal.test_contrast(
        people([0., 1., 2.]) + people([0.]), SPEC))
    if legacy.contrast(legacy_people([0., 0., 0.]), {"C4": 1}, "less")["status"] != "unavailable":
        raise AssertionError("Legacy zero-SE must be unavailable")
    if nominal.test_contrast(people([0., 0., 0.]), SPEC)["status"] != "unavailable":
        raise AssertionError("Current zero-SE must be unavailable")
    names.append("both_zero_SE_unavailable")
    if legacy.participant_slope([{"shareA": .5, "xInitial": 0.}] * 10, "xInitial")["status"] != "unavailable":
        raise AssertionError("Constant predictor must be unavailable")
    names.append("constant_predictor_unavailable")
    boundary = decisions({"pValue": .05, "interval": [-1., 1.]},
                         {"p": .05, "CI": [-1., 1.]}, {"p": .05, "CI": [-1., 1.]})
    if not boundary["legacyLE"] or boundary["currentLT"] or boundary["projectionLT"]:
        raise AssertionError("Exact boundary conventions")
    names.append("exact_alpha_boundary_conventions")
    return dict(passed=len(names), failed=0, skipped=0, names=names)


def decisions(old,current,projection):
    return dict(legacyLE=old['pValue']<=.05,currentLT=current['p']<.05,
      projectionLT=projection['p']<.05,legacyCovered=old['interval'][0]<=0<=old['interval'][1],
      currentCovered=current['CI'][0]<=0<=current['CI'][1],
      projectionCovered=projection['CI'][0]<=0<=projection['CI'][1],
      legacyBoundary=old['pValue']==.05,currentBoundary=current['p']==.05,
      projectionBoundary=projection['p']==.05)

def produce(config, output, batches=BATCHES):
    """Write fixed-shape selected batches; summary.json is the completion record."""
    batches = selected_batches(batches)
    cell = fixed_cell(config)
    output = fresh_directory(output)
    phase = time.monotonic()
    tests = negative_tests()
    write_json(output / "tests.json", tests)
    batch_summaries=[]; differences={k:0. for k in ('legacyCurrentEffect','legacyCurrentSE','legacyCurrentDF',
       'legacyCurrentStatistic','legacyCurrentP','legacyCurrentCILow','legacyCurrentCIHigh',
       'vectorRawSlope','vectorProjectionSlope','rawProjectionSlope')}
    nonzero={k:0 for k in differences}; all_counts={}; discordances=0
    plan=None
    for number in batches:
        start=time.monotonic()
        batch=cal.generate_batch(cell,number,SIZE)
        raw=batch['raw']['C4']; original=batch['populations']['primary']['C4']['initial']
        if not all(np.all(raw[key]) for key in ('records','personKeep')) or not all(np.all(a) for a in raw['endpointMasks'].values()):
            raise AssertionError('Fixed full-eligibility law changed')
        if not np.all(original['available']) or not np.all(original['observations']==10) or not np.all(original['truth']==0):
            raise AssertionError('Fixed original primary/null support changed')
        arrays={k:raw[k] for k in ('y','x','records','personKeep','expectedY','scale','gap')}
        arrays.update({'endpoint_'+k:v for k,v in raw['endpointMasks'].items()})
        if any(a.dtype.kind not in 'bfiu' or not np.all(np.isfinite(a)) for a in arrays.values()):
            raise AssertionError('Invalid raw packet')
        buffer=io.BytesIO(); np.savez(buffer,**arrays); write_bytes(output / ('batch_%d.npz' % number), buffer.getvalue())
        diagnostics=[]; weights=[]
        for x in raw['x']:
            centered=x-np.mean(x); design=np.column_stack([np.ones(10),centered])
            rank=int(np.linalg.matrix_rank(design))
            d=dict(observations=10,predictorLevels=len(set(map(float,x))),Sxx=float(np.sum(centered**2)),rank=rank,residualDf=8)
            if rank!=2 or d['predictorLevels']<2 or d['Sxx']<=1e-12: raise AssertionError('Diagnostic schedule support')
            diagnostics.append(d); weights.append(np.linalg.lstsq(design,np.eye(10),rcond=None)[0][1])
        proposed=dict(kind='diagnostic_unadjusted_same_raw_schedule_not_native_plan',weights=np.asarray(weights).tolist(),diagnostics=diagnostics)
        if plan is None: plan=proposed; write_json(output / 'plan.json', plan)
        elif plan!=proposed: raise AssertionError('Fixed schedule changed between batches')
        projections=nominal.canonical_projection(np.asarray(weights),raw['y'])
        vector=cal.vector_contrast(batch['populations']['primary'],'initial',{'C4':1},'less')
        records=[]; counts={}; batchdiscord=0
        for i in range(SIZE):
            values=original['values'][i]
            qualification=[]
            for j in range(20):
                rows=[dict(shareA=float(raw['y'][i,j,k]),xInitial=float(raw['x'][j,k]),
                       stageIndex=k+1,scale=float(raw['scale'][k]),gapOffset=float(raw['gap'][k])) for k in range(10)]
                q=legacy.participant_slope(rows,'xInitial')
                if q['status']!='available': raise AssertionError('Unexpected raw OLS unavailability')
                qualification.append(q)
            raw_slopes=np.asarray([q['value'] for q in qualification])
            old=legacy.contrast(legacy_people(values),{'C4':1},'less')
            current=nominal.test_contrast(people(values),SPEC)
            projection=nominal.test_contrast(people(projections[i]),SPEC)
            if any(x['status']!='available' for x in (old,current,projection)):
                raise AssertionError('Unexpected paired inference unavailability')
            vv={key:value[i].item() for key,value in vector.items()}
            dec=decisions(old,current,projection)
            for key,value in dec.items(): counts[key]=counts.get(key,0)+int(value)
            if dec['legacyLE']!=dec['currentLT']: batchdiscord+=1
            deltas=dict(legacyCurrentEffect=abs(old['estimate']-current['effect']),
              legacyCurrentSE=abs(old['standardError']-current['SE']),legacyCurrentDF=abs(old['df']-current['df']),
              legacyCurrentStatistic=abs(old['statistic']-current['statistic']),legacyCurrentP=abs(old['pValue']-current['p']),
              legacyCurrentCILow=abs(old['interval'][0]-current['CI'][0]),legacyCurrentCIHigh=abs(old['interval'][1]-current['CI'][1]),
              vectorRawSlope=float(np.max(np.abs(values-raw_slopes))),
              vectorProjectionSlope=float(np.max(np.abs(values-projections[i]))),rawProjectionSlope=float(np.max(np.abs(raw_slopes-projections[i]))))
            for key,value in deltas.items(): differences[key]=max(differences[key],value); nonzero[key]+=int(value!=0.)
            records.append(dict(batch=number,index=i,replicationId=number*SIZE+i,
              peopleIds=['C4:%02d'%j for j in range(20)],vectorSlopes=values.tolist(),legacyRawSlopes=raw_slopes.tolist(),
              projectionSlopes=projections[i].tolist(),qualifications=qualification,
              legacy=old,legacyVector=vv,currentSameInput=current,currentProjection=projection,decisions=dec))
        write_json(output / ('cases_%d.json' % number), records)
        for key,value in counts.items(): all_counts[key]=all_counts.get(key,0)+value
        discordances+=batchdiscord
        batch_summaries.append(dict(batch=number,cases=SIZE,seconds=time.monotonic()-start,decisionCounts=counts,sameInputRejectDiscordances=batchdiscord))
        print('FIXED_BATCH_CLOSED '+str(number),flush=True)
    summary=dict(schema='paired-gamma-diagnostic-v1',scope=scope(batches),cases=SIZE * len(batches),
      selectedComplete=True,fixed500Complete=batches == BATCHES,referenceRuntime=REFERENCE_RUNTIME,
      auditOrigin='historical_abstract_summary_audit',testResults=tests,batchSummaries=batch_summaries,
      originalRuntime=ORIGINAL_RUNTIME, observedRuntime=runtime(),
      aggregateDifferences=dict(maxAbs=differences,nonzeroCaseCounts=nonzero,decisionCounts=all_counts,
          sameInputRejectDiscordances=discordances),
      elapsedEntryScienceSeconds=time.monotonic()-phase,limitations=LIMITATIONS,
      typeIRepairClaim=False,historicalFailuresChanged=0,formalDispositionsChanged=False)
    write_json(output / 'summary.json', summary)
    return summary
