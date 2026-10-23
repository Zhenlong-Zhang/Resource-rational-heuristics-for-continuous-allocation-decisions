/* One declared conditional software panel. Genuine frozen engine transitions;
 * no human/Theory observations, seed search, extra scientific replications or fit. */
'use strict';

const fs=require('node:fs'), path=require('node:path'), assert=require('node:assert/strict');
const E=require('./panel_engine.cjs');
const N=require('./panel_normalizer.cjs');
const SEED='h10-alternatives-paid-panel-v1-20261019';
const META={origin:'conditional_scripted_engine_panel',profileId:E.R9,flowVersion:E.R10_FLOW,
  resourceUnit:'minutes',seed:SEED,conditions:['C1','C2','C3'],variants:[0,1,2,3,4,5,6,7,8],
  questionRule:'1+((stageIndex+variant)%3)',studentRule:'(stageIndex+variant+k)%2',
  choiceRule:'250+((stageIndex*37+variant*13)%501)',humanObservations:false,theoryPolicy:false};
function atomic(value,destination) {
  fs.mkdirSync(path.dirname(destination),{recursive:true});
  const temporary=destination+'.uncommitted-'+process.pid;
  fs.writeFileSync(temporary,JSON.stringify(value,null,2)+'\n',{flag:'wx'});
  try {fs.linkSync(temporary,destination);} finally {fs.unlinkSync(temporary);}
}
function one(design,condition,variant) {
  const participant=`h10-alternatives-${condition}-v${variant}`;
  const options={profileId:E.R9,participantFlowVersion:E.R10_FLOW,conditionId:condition,
    participantId:participant,sessionId:participant+'-session',scheduleVariant:variant,
    seed:SEED,startedAt:'2026-10-19T00:00:00.000Z'};
  let state=E.createSession(design,options),counter=0;
  const act=(type,fields={})=>{state=E.transition(state,{id:`h10-${counter}`,
    expectedRevision:state.revision,at:new Date(Date.parse(options.startedAt)+100*counter++).toISOString(),
    type,...fields},design);};
  act('beginChecks');
  for(let step=0;step<4;step++) act('advanceDisplayGuide',{
    fixtureId:state.guideFixtureId,fixtureHash:state.guideFixtureHash,step});
  act('submitChecks',{answers:E.checkQuestions(state).map(q=>q.correct)});
  while(state.phase!=='posttask') {
    if(state.phase==='practiceGuide') {act('beginPracticeChecks',{
      fixtureId:state.guideFixtureId,fixtureHash:state.guideFixtureHash});continue;}
    if(state.phase==='practiceCheck') {act('submitChecks',{answers:E.checkQuestions(state).map(q=>q.correct)});continue;}
    if(state.phase==='break') {act('continueBreak');continue;}
    assert.equal(state.phase,'trial');
    const d=state.current.definition;
    const logical=d.priorCentres.map((m,i)=>E.prior(m,d.priorSpreads[i]));
    assert.deepEqual(state.current.initialLogs,(d.sideReversed?[1,0]:[0,1]).map(i=>logical[i]));
    const questions=1+((d.stageIndex+variant)%3);
    assert.ok(state.current.budgetMillihours-questions*d.costMillihours>0,
      'Declared question script must leave positive budget; do not shorten or redraw');
    for(let k=0;k<questions;k++) act('assess',{student:(d.stageIndex+variant+k)%2});
    const shareIndex=250+((d.stageIndex*37+variant*13)%501);
    act('openStop');if(state.phase==='stopConfirm')act('confirmStop');
    act('editAllocationIndex',{shareIndex});act('confirmAllocation');act('advance');act('acknowledgeAdvance');
  }
  const answers=Object.fromEntries(E.PILOT_TEXT_FIELDS.map(key=>[key,'fixed conditional software fixture']));
  for(const key of ['costClarity','uncertaintyClarity','gpaClarity','realism','duration','repetition'])answers[key]='4';
  act('complete',{answers});
  const exported=E.exportData(state),normalized=N.normalize(exported,design);
  assert.deepEqual(normalized.errors,[]);assert.equal(normalized.meta.status,'complete');
  assert.equal(normalized.trials.filter(t=>t.definition.stage==='main').length,10);
  return exported;
}
function build(design) {
  E.validateDesign(design);
  const records=[];
  for(const condition of META.conditions)for(const variant of META.variants)records.push(one(design,condition,variant));
  return {schema:'h10-alternatives-source-panel-v1',metadata:META,records};
}
function normalize(panel,design) {
  assert.equal(panel.schema,'h10-alternatives-source-panel-v1');assert.deepEqual(panel.metadata,META);
  assert.equal(panel.records.length,27);
  return panel.records.map((raw,i)=>{
    const normalized=N.normalize(raw,design);
    assert.deepEqual(normalized.errors,[]);assert.equal(normalized.meta.status,'complete');
    const condition=META.conditions[Math.floor(i/9)],variant=i%9;
    assert.equal(normalized.meta.participantId,`h10-alternatives-${condition}-v${variant}`);
    assert.equal(normalized.meta.sessionId,normalized.meta.participantId+'-session');
    assert.equal(normalized.meta.conditionId,condition);assert.equal(normalized.meta.profileId,E.R9);
    assert.equal(normalized.meta.participantFlowVersion,E.R10_FLOW);
    assert.equal(normalized.meta.seed,SEED);assert.equal(normalized.meta.scheduleVariant,variant);
    assert.equal(raw.displayTimeMapping,'1canonical model hour =1scenario minute');
    for(const t of normalized.trials) {
      const d=t.definition,r=t.result,questions=1+((d.stageIndex+variant)%3);
      assert.equal(r.assessmentCounts.reduce((a,b)=>a+b,0),questions);
      assert.equal(r.remainingMillihours,d.budgetMillihours-questions*d.costMillihours);
      assert.ok(r.remainingMillihours>0);
      assert.equal(r.shareIndex,250+((d.stageIndex*37+variant*13)%501));
      const events=t.events.filter(e=>e.type==='assessment');
      assert.equal(events.length,questions);
      events.forEach((e,k)=>assert.equal(e.student,(d.stageIndex+variant+k)%2));
    }
    return normalized;
  });
}
module.exports={META,build,normalize};
if(require.main===module) {
  const [mode,...args]=process.argv.slice(2),opts={};
  assert.ok(['build','normalize'].includes(mode),'Expected build or normalize');
  for(let i=0;i<args.length;i+=2) {
    assert.ok(['--design','--input','--output'].includes(args[i])&&args[i+1]&&!opts[args[i]],'Invalid arguments');
    opts[args[i]]=args[i+1];
  }
  assert.ok(opts['--design']&&opts['--output']);
  const design=JSON.parse(fs.readFileSync(opts['--design'],'utf8'));
  const result=mode==='build'?build(design):normalize(JSON.parse(fs.readFileSync(opts['--input'],'utf8')),design);
  atomic(result,opts['--output']);
  process.stdout.write(JSON.stringify({mode,output:opts['--output'],sessions:27,mainRows:270})+'\n');
}
