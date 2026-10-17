/* Synthetic standalone replay only. No native transport or human adapter. */
'use strict';
const E=require('./native_engine.cjs');
const assert=(ok,message)=>{if(!ok)throw Error(message);};
function near(actual,expected,label){
  if(typeof expected==='number'){assert(typeof actual==='number'&&Number.isFinite(actual),label+' is not finite');const integerAccounting=/(Millihours|Microhours|trueNeeds|potentialReports|orientation|^report$|shareIndex|eventIndex|scheduleIndex|stageIndex|reportIndex|assessmentCounts|open counts|event counts|open budget|event budget|question prebudget|report index|committed share)/.test(label);assert(integerAccounting?Number.isInteger(actual)&&actual===expected:Math.abs(actual-expected)<=1e-10,label+' differs from replay');}
  else if(Array.isArray(expected)){assert(Array.isArray(actual)&&actual.length===expected.length,label+' shape mismatch');expected.forEach((x,i)=>near(actual[i],x,label+'['+i+']'));}
  else if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',label+' missing');for(const [key,value]of Object.entries(expected))near(actual[key],value,label+'.'+key);}
  else assert(actual===expected,label+' differs from replay');
}
function sessionMeta(data,design){
  assert(data.schema==='tutoring-task-export-1','Unsupported export schema');
  assert(data.profileId===E.PILOT||data.profileId===E.ENGAGEMENT||data.profileId===E.R9,'Only supported pilot profiles may be replayed');
  if(data.profileId===E.ENGAGEMENT)assert(data.questionLimitRule==='budget_only'&&data.maximumOnlineAssessments===null&&data.presentationVersion==='trial-presentation-v1'&&data.feedbackStyle==='outcome-engagement-v1','New profile metadata mismatch');
  assert(data.engineVersion===E.VERSION,'Session engine version mismatch');
  assert(data.designFingerprint===E.sessionFingerprint(design,data),'Design fingerprint mismatch');
  const options={participantId:data.participantId,sessionId:data.sessionId,conditionId:data.conditionId,seed:data.seed,profileId:data.profileId,scheduleVariant:data.scheduleVariant};
  for(const key of ['participantId','sessionId','seed'])assert(typeof options[key]==='string'&&options[key].trim().length>0,'Missing nonblank '+key);
  assert(Number.isInteger(options.scheduleVariant)&&options.scheduleVariant>=0&&options.scheduleVariant<=8,'Missing or invalid pilot schedule variant');
  const s=E.createSession(design,options);assert(E.fingerprint(data.schedule)===E.fingerprint(s.schedule),'Schedule differs from selected design');
  assert(['complete','incomplete','mechanism_checks_not_passed'].includes(data.status),'Invalid session status');
  const phases=['instructions','instructionsReview','comprehension','ineligible','practiceCheck','trial','stopConfirm','allocation','feedback','pendingCommit','break','posttask','complete',...(data.profileId===E.R9?['practiceGuide']:[])];
  assert(phases.includes(data.phase),'Invalid session phase');
  assert(data.status===(data.phase==='complete'?'complete':data.phase==='ineligible'?'mechanism_checks_not_passed':'incomplete'),'Session status/phase mismatch');
  assert(Number.isInteger(data.completedTrialCount)&&data.completedTrialCount>=0&&data.completedTrialCount<=14,'Invalid completed trial count');
  assert(Number.isInteger(data.archivedTrialCount)&&data.archivedTrialCount>=0&&data.archivedTrialCount<=14,'Invalid archived trial count');
  assert(Array.isArray(data.trials),'Missing completed trial records array');
  assert(data.completedTrialCount===data.archivedTrialCount+new Set(data.trials.map(t=>t.trialId)).size,'Completed trial count/records mismatch');
  assert(data.status!=='complete'||data.completedTrialCount===14,'Complete session dose mismatch');
  assert(data.exportIsFullHistory===(data.archivedTrialCount===0),'History availability/archived count mismatch');
  if(data.profileId===E.R9){
    assert(data.questionLimitRule==='budget_only'&&data.maximumOnlineAssessments===null,'R9 horizon metadata mismatch');
    const gate={...s,...data,current:data.currentTrial||null,completedTrials:data.trials,practiceGuideHistory:data.practiceGuideHistory};
    E.validateR9Instrumentation(gate,design);
    assert(Number.isInteger(data.currentTrialIndex)&&data.currentTrialIndex>=0&&data.currentTrialIndex<=14,'R9 exported progress missing');
    assert(data.currentTrialIndex===data.completedTrialCount-(data.phase==='pendingCommit'?1:0),'R9 completed/current progress mismatch');
  }
  return {...options,questionLimitRule:E.profile(data).budgetOnly?'budget_only':'count_cap_20',phase:data.phase,status:data.status,reportedCompletedTrialCount:data.completedTrialCount,archivedTrialCount:data.archivedTrialCount||0,reportedFullHistory:data.exportIsFullHistory===true,
    ...(data.profileId===E.R9?{...E.r9Metadata(design),guideFixtureId:data.guideFixtureId,guideFixtureHash:data.guideFixtureHash,descriptiveOnly:true,equalOutcomeSemantics:'legacy_equal_mean_raw_surplus_supporting_only'}:{})};
}
function validateTrial(raw,meta,design,current=false,replayContext=null){
  assert(raw.designFingerprint===E.sessionFingerprint(design,meta),'Trial design fingerprint mismatch');
  assert(raw.seedNamespaceVersion===E.PILOT,'Trial seed namespace mismatch');
  assert(current||raw.status==='complete','Invalid completed trial status');
  const t=E.expandTrial(raw),base=E.createSession(design,meta),index=base.schedule.findIndex(d=>d.trialId===t.trialId);
  if(raw.compactSchema){for(const k of ['initialLogs','terminalLogs'])if(raw[k])near(raw[k],t[k],'compact provided '+k);
    raw.events.forEach((e,i)=>{for(const k of ['logsBefore','logsAfter','displayAfter'])if(e[k])near(e[k],t.events[i][k],'compact provided '+k);});}
  assert(index>=0,'Unknown trial identity');assert(E.fingerprint(t.definition)===E.fingerprint(base.schedule[index]),'Definition differs from selected design');
  assert(t.conditionId===meta.conditionId&&t.participantId===meta.participantId,'Trial participant/condition mismatch');
  assert(t.profileId===meta.profileId&&t.engineVersion===E.VERSION,'Trial profile/version mismatch');
  if(E.profile(meta).budgetOnly)assert(t.questionLimitRule==='budget_only'&&t.questionCapacity===E.questionCapacity(meta,t.definition),'Trial budget-only metadata mismatch');
  let state;
  if(meta.profileId===E.R9){
    assert(replayContext&&replayContext.state&&replayContext.state.currentTrialIndex===index&&replayContext.state.phase==='trial','R9 record requires its actual prior gate/trial prefix');
    for(const [key,value]of Object.entries(E.r9Metadata(design)))assert(E.fingerprint(t[key])===E.fingerprint(value),'Trial R9 identity '+key);
    state=replayContext.state;
  }else{
    base.currentTrialIndex=index;
    state=E.transition(base,{id:'replay-begin',expectedRevision:base.revision,type:'beginChecks'},design);
    state=E.transition(state,{id:'replay-open',expectedRevision:state.revision,type:'submitChecks',answers:E.checkQuestions(state).map(q=>q.correct)},design);
  }
  near(t.trueNeeds,state.current.trueNeeds,'trueNeeds');near(t.potentialReports,state.current.potentialReports,'potentialReports');
  near(t.logicalToDisplayed,state.current.logicalToDisplayed,'orientation');near(t.initialLogs,state.current.initialLogs,'initialLogs');
  assert(Array.isArray(t.events)&&t.events.length>0&&t.events[0].type==='trial_open','Missing trial_open');
  let committed=false,advanced=false;const originalActions=new Set();
  const paid=[];
  for(let i=0;i<t.events.length;i++){
    const e=t.events[i];assert(e.eventIndex===i,'Event index mismatch');assert(typeof e.id==='string'&&e.id.trim(),'Missing action identity');
    const directCompanion=e.type==='stop_confirm'&&e.directMainStop===true;
    if(directCompanion)assert(t.definition.stage==='main'&&state.phase==='allocation'&&t.events[i-1]?.type==='stop_open'&&t.events[i-1].id===e.id,'Direct stop companion identity mismatch');
    else{assert(!originalActions.has(e.id),'Duplicate original action identity');originalActions.add(e.id);}
    assert(typeof e.at==='string'&&Number.isFinite(Date.parse(e.at)),'Invalid event timestamp');
    if(i===0){near(e.budgetMillihours,state.current.budgetMillihours,'open budget');near(e.assessmentCounts,[0,0],'open counts');continue;}
    let action;
    if(e.type==='assessment'){
      assert(!committed,'Question after allocation commit');const before=state.current;
      if(!raw.compactSchema)assert(Array.isArray(e.logsBefore)&&Array.isArray(e.logsAfter)&&Array.isArray(e.displayAfter),'Expanded question summaries missing');
      assert(e.student===0||e.student===1,'Invalid recipient');near(e.reportIndex,before.assessmentCounts[e.student],'report index');
      near(e.budgetBeforeMillihours,before.budgetMillihours,'question prebudget');
      if(e.logsBefore)near(e.logsBefore,before.logs,'logsBefore');action={type:'assess',student:e.student};paid.push({student:e.student,report:e.report});
    }else if(e.type==='stop_open')action={type:'openStop'};
    else if(e.type==='stop_cancel')action={type:'cancelStop'};
    else if(e.type==='stop_confirm'){
      // In main trials openStop already writes stop_confirm and enters allocation.
      if(state.phase==='allocation'&&e.directMainStop===true)action=null;else action={type:'confirmStop'};
    }else if(e.type==='allocation_edit'){
      assert(state.phase==='allocation','Edit outside allocation phase');
      assert(E.parseAllocation(e.rawValue)===e.shareIndex,'Edited raw value/share mismatch');
      if(e.editCount!==undefined){assert(Number.isSafeInteger(e.editCount)&&e.editCount>=1,'Invalid editCount');assert(E.parseAllocation(e.firstRawValue)===e.firstShareIndex,'First edit value/share mismatch');assert(typeof e.firstEditId==='string'&&e.firstEditId.trim()&&Number.isFinite(Date.parse(e.firstEditAt))&&e.lastEditAt===e.at,'Invalid edit summary');assert(e.editCount===1?e.firstEditId===e.id&&e.firstRawValue===e.rawValue:e.firstEditId!==e.id&&!originalActions.has(e.firstEditId),'Invalid first/last edit identity');originalActions.add(e.firstEditId);}
      action=e.shareIndex===null?{type:'editAllocation',value:e.rawValue}:{type:'editAllocationIndex',shareIndex:e.shareIndex};
    }else if(e.type==='allocation_confirm'){assert(!committed,'Repeated allocation commit');action={type:'confirmAllocation'};committed=true;}
    else if(e.type==='trial_complete'){assert(committed&&!advanced,'Trial completion out of order');action={type:'advance'};advanced=true;}
    else throw Error('Unknown trial event: '+e.type);
    if(action)state=E.transition(state,{id:'replay-'+index+'-'+i,at:e.at,expectedRevision:state.revision,...action},design);
    near(e.budgetMillihours,state.current.budgetMillihours,'event budget');near(e.assessmentCounts,state.current.assessmentCounts,'event counts');
    if(e.type==='assessment'){
      const actual=state.current.events.filter(x=>x.type==='assessment').at(-1);near(e.report,actual.report,'report');
      if(e.logsAfter)near(e.logsAfter,state.current.logs,'logsAfter');if(e.displayAfter)near(e.displayAfter,actual.displayAfter,'displayAfter');
    }
    if(e.type==='allocation_confirm')near(e.shareIndex,state.current.result.shareIndex,'committed share');
  }
  assert(committed,'No committed allocation');assert(current||advanced,'Completed record missing advance');
  if(current)assert(meta.phase===state.phase,'Current trial/session phase mismatch');
  near(t.terminalLogs,state.current.logs,'terminalLogs');near(t.result,state.current.result,'result');
  if(replayContext)replayContext.state=state;
  const semantic={definition:t.definition,trueNeeds:t.trueNeeds,initialLogs:state.current.initialLogs,terminalLogs:state.current.logs,
    potentialReports:t.potentialReports,logicalToDisplayed:t.logicalToDisplayed,paid,result:state.current.result};
  // Only trial_complete may be appended by monotonic advancement. Retained scientific
  // acquisition/stopping/edit/commit history (including original IDs) must still agree.
  const committedHistory=t.events.filter(e=>e.type!=='trial_complete').map(e=>{
    const copy={...e};for(const k of ['logsBefore','logsAfter','displayAfter'])delete copy[k];return copy;
  });
  return {trialId:t.trialId,conditionId:meta.conditionId,participantId:meta.participantId,sessionId:meta.sessionId,profileId:meta.profileId,questionLimitRule:meta.questionLimitRule,
    ...semantic,events:t.events,advanced,committedNotAdvanced:!advanced,semanticFingerprint:E.fingerprint({...semantic,committedHistory})};
}
function r9ReplayContext(exported,meta,design){
  const context={state:E.createSession(design,{...meta,startedAt:exported.startedAt}),counter:0};
  context.act=(type,fields={})=>{context.state=E.transition(context.state,{id:'r9-prefix-'+context.counter++,expectedRevision:context.state.revision,type,...fields},design);};
  context.align=revision=>{assert(Number.isInteger(revision)&&revision>=context.state.revision,'R9 prefix revision reversal');context.state.revision=revision;};
  context.checkSet=kind=>{
    const attempts=exported.checks.filter(x=>x.kind===kind);
    if(kind==='comprehension'&&attempts.length)context.act('beginChecks',{at:attempts[0].at});
    for(const c of attempts){
      if(context.state.phase==='instructionsReview')context.act('beginChecks',{at:c.at});
      if(kind==='practiceCheck'){
        const g=exported.practiceGuideHistory[c.guideHistoryIndex];assert(g,'Missing actual R9 guide entry');
        assert(context.state.phase==='practiceGuide','Post-practice attempt without guide progression');
        near(context.state.practiceGuideHistory.at(-1).fixtureHash,g.fixtureHash,'guide fixture');
        context.align(g.acknowledgedRevision-1);
        context.act('beginPracticeChecks',{at:g.acknowledgedAt,fixtureId:g.fixtureId,fixtureHash:g.fixtureHash});
      }
      context.align(c.attemptRevision-1);context.act('submitChecks',{answers:c.answers,at:c.at});
      const generated=context.state.checks.at(-1);
      for(const key of ['kind','attempt','answers','missed','presentedOptionIds','questionKeys','missedQuestionKeys','checkVersion','guideHistoryIndex','attemptRevision'])if(c[key]!==undefined)near(generated[key],c[key],'R9 actual check '+key);
    }
  };
  context.checkSet('comprehension');return context;
}
function normalize(data,design){
  const errors=[],flags=[];let exported,kind,rawTrials=[];
  if(data.schema==='tutoring-task-export-1'){kind='standalone_export';exported=data;rawTrials=(data.trials||[]).map(t=>({t,current:false}));}
  else throw Error('Only synthetic standalone exports are supported');
  const meta=sessionMeta(exported,design),current=exported.currentTrial;
  if(exported.pendingCommit?.record)rawTrials.push({t:exported.pendingCommit.record,current:false});
  if(current&&current.result){rawTrials.push({current:true,t:{trialId:current.definition.trialId,conditionId:meta.conditionId,participantId:meta.participantId,profileId:meta.profileId,engineVersion:E.VERSION,
    designFingerprint:exported.designFingerprint,seedNamespaceVersion:E.PILOT,
    ...(E.profile(meta).budgetOnly?{questionLimitRule:'budget_only',questionCapacity:E.questionCapacity(meta,current.definition)}:{}),
    ...(meta.profileId===E.R9?{...E.r9Metadata(design),guideFixtureId:exported.guideFixtureId,guideFixtureHash:exported.guideFixtureHash}:{}),definition:current.definition,initialLogs:current.initialLogs,terminalLogs:current.logs,trueNeeds:current.trueNeeds,potentialReports:current.potentialReports,logicalToDisplayed:current.logicalToDisplayed,events:current.events,result:current.result}});}
  else if(current)flags.push('current_uncommitted');
  const validated=new Map();
  if(meta.profileId===E.R9){
    const copyIdentity=raw=>{const t=E.expandTrial(raw),events=t.events.filter(e=>e.type!=='trial_complete').map(e=>{const x={...e};for(const k of ['logsBefore','logsAfter','displayAfter'])delete x[k];return x;});return E.fingerprint({definition:t.definition,trueNeeds:t.trueNeeds,initialLogs:t.initialLogs,terminalLogs:t.terminalLogs,potentialReports:t.potentialReports,logicalToDisplayed:t.logicalToDisplayed,result:t.result,events});};
    const copies=new Map();for(const item of rawTrials){const id=item.t.trialId,old=copies.get(id);if(old){assert(copyIdentity(old.t)===copyIdentity(item.t),'Conflicting R9 trial copies');if(old.current&&!item.current)copies.set(id,item);}else copies.set(id,item);}
    const items=[...copies.values()].sort((a,b)=>a.t.definition.scheduleIndex-b.t.definition.scheduleIndex),context=r9ReplayContext(exported,meta,design);
    for(const {t,current}of items){try{
      const index=t.definition.scheduleIndex;assert(index===context.state.currentTrialIndex,'R9 prior trial prefix unavailable');
      if(context.state.phase==='practiceGuide')context.checkSet('practiceCheck');
      if(context.state.phase==='break')context.act('continueBreak',{at:E.expandTrial(t).events[0].at});
      const record=validateTrial(t,meta,design,current,context);validated.set(record.trialId,record);
      if(record.advanced){
        const fields={at:record.events.at(-1).at};
        if(index===3){const g=exported.practiceGuideHistory[0];assert(g,'R9 first guide witness missing');context.align(g.shownRevision-1);fields.at=g.shownAt;}
        context.act('acknowledgeAdvance',fields);
      }
    }catch(e){errors.push({trialId:t.trialId||null,message:e.message});break;}}
    if(items.length&&items[0].t.definition.scheduleIndex>0)flags.push('r9_prior_prefix_unavailable');
  }else for(const {t,current}of rawTrials){try{const record=validateTrial(t,meta,design,current),old=validated.get(record.trialId);if(old&&old.semanticFingerprint!==record.semanticFingerprint){flags.push('conflicting_trial_copies');throw Error('Conflicting within-input trial copies');}if(!old||record.advanced)validated.set(record.trialId,record);}catch(e){errors.push({trialId:t.trialId||t.definition?.trialId||null,message:e.message});}}
  if(errors.length)flags.push('invalid_trials');
  if(validated.size<Math.max(meta.reportedCompletedTrialCount,meta.archivedTrialCount))flags.push('reported_history_unavailable');
  return {schema:'tutoring_normalized_v1',kind,meta,flags,errors,trials:[...validated.values()],posttask:exported.posttask||{},checks:exported.checks||[],...(meta.profileId===E.R9?{practiceGuideHistory:exported.practiceGuideHistory}: {})};
}
module.exports={normalize,validateTrial,near};
if(require.main===module){let text='';process.stdin.setEncoding('utf8');process.stdin.on('data',x=>text+=x);process.stdin.on('end',()=>{try{const request=JSON.parse(text);process.stdout.write(JSON.stringify({ok:true,normalized:normalize(request.data,request.design)}));}catch(e){process.stdout.write(JSON.stringify({ok:false,error:e.message}));process.exitCode=1;}});}
