/* Purpose: synthetic software checks for durable strategy responses and export compatibility. */
'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const R=require('../src/tutoring_task/local_strategy.js');
class Memory {constructor(){this.m=new Map();this.fail=false;}getItem(k){return this.m.has(k)?this.m.get(k):null;}setItem(k,v){if(this.fail)throw Error('simulated write failure');this.m.set(k,v);}}
const identity=conditionId=>({sessionId:'session',participantId:'person',conditionId,profileId:'pilot-r9-design-v1',participantFlowVersion:'r10-prepractice-v1',designFingerprint:'fingerprint'});
const clock=()=>new Date('2026-11-17T12:00:00Z').toISOString();
function create(storage,condition='C1',random=()=>0){return R.open({storage,key:'sidecar',identity:identity(condition),random,now:clock});}
test('five conditions, six judgement permutations and stable ordering',()=>{
 const perms=new Set();for(const condition of ['C1','C2','C3','C4','C5'])for(let a=0;a<3;a++)for(let b=0;b<2;b++){
  const values=[0,0,0,0,0,a,b],storage=new Memory(),m=create(storage,condition,()=>values.shift());let st=m.get();perms.add(st.questionOrder.slice(0,3).join(','));assert.equal(st.questionOrder[3],'self_report');assert.equal(new Set(st.cardOrder).size,6);
  for(let i=0;i<4;i++){const before=m.get(),id=before.questionOrder[i];m.answer(id,before.cardOrder[i],null);assert.deepEqual(m.get().cardOrder,st.cardOrder);assert.deepEqual(m.get().questionOrder,st.questionOrder);}
  st=m.get();assert.equal(st.answers.length,4);assert.equal(st.itemStartedAt.length,4);assert(st.completedAt);assert.deepEqual(create(storage,condition).get(),st);
 }assert.equal(perms.size,6);
});
test('other, skip, pending, failed save and stale tabs remain distinct',()=>{
 const storage=new Memory(),m=create(storage);let st=m.get();assert.equal(st.answers.length,0);assert.equal(st.completedAt,null);
 const first=st.questionOrder[0];assert.throws(()=>m.answer(first,'other',''),/Describe your approach/);m.answer(first,'other','first approach');const second=m.get().questionOrder[1];m.answer(second,'other','words');
 st=m.get();assert.equal(st.answers[0].otherText,'first approach');assert.equal(st.answers[1].otherText,'words');
 const third=st.questionOrder[2],prior=storage.getItem('sidecar');storage.fail=true;assert.throws(()=>m.skip(third),/simulated/);assert.equal(storage.getItem('sidecar'),prior);assert.equal(m.get().answers.length,2);storage.fail=false;m.skip(third);assert.equal(m.get().answers[2].status,'skipped');
 assert.throws(()=>m.skip(third),/no longer current/);const stale=create(storage);m.skip(m.get().questionOrder[3]);assert.throws(()=>stale.skip(stale.get().questionOrder[3]),/another tab/);
});
test('recovery and export reader reject corrupt or mismatched evidence',()=>{
 const storage=new Memory(),m=create(storage);const task=identity('C1');let envelope={schema:'r13-local-export-v1',task,strategy:m.get()};assert.deepEqual(R.readExport(envelope).strategy,m.get());
 assert.throws(()=>R.readExport({...envelope,strategyRecovery:{raw:'broken'}}),/recovery/);assert.throws(()=>R.readExport({...envelope,task:identity('C2')}),/corrupt/);
 storage.setItem('sidecar','{invalid');assert.throws(()=>create(storage),/cannot be opened/);assert.equal(storage.getItem('sidecar'),'{invalid');
 assert.throws(()=>R.open({storage:new Memory(),key:'missing',identity:identity('C1'),random:()=>0,requiredExisting:true}),/no saved strategy/);
});
