/* Purpose: synthetic software checks for durable strategy responses and export compatibility. */
'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const R=require('../src/tutoring_task/local_strategy.js');
const identity={sessionId:'other-session',participantId:'software-fixture',conditionId:'C4',profileId:'pilot-r9-design-v1',participantFlowVersion:'r10-prepractice-v1',designFingerprint:'software-fixture-only'};
class Memory {constructor(){this.values=new Map();this.fail=false;}getItem(k){return this.values.get(k)??null;}setItem(k,v){if(this.fail)throw Error('Synthetic storage quota failure');this.values.set(k,v);}}
function open(storage){return R.open({storage,key:'other-test',identity,random:()=>0,now:()=>new Date('2026-11-24T12:00:00Z').toISOString()});}
test('Other requires non-whitespace text without mutating saved data; all items remain skippable',()=>{
 const storage=new Memory(),m=open(storage),before=storage.getItem('other-test'),item=m.get().questionOrder[0];
 for(const text of [null,undefined,'',' \n\t\u00a0 '])assert.throws(()=>m.answer(item,'other',text),/Describe your approach/);
 assert.equal(storage.getItem('other-test'),before);
 for(const id of m.get().questionOrder)m.skip(id);
 assert.equal(m.get().answers.length,4);assert(m.get().answers.every(a=>a.status==='skipped'&&a.otherText===null));
 assert(m.get().completedAt);
});
test('long Unicode and markup text survives storage failure retry reload and export exactly',()=>{
 const text='  '+('策略 🧠 café <tag> & \"quoted\"\n').repeat(2000)+'  ',storage=new Memory(),m=open(storage),item=m.get().questionOrder[0];
 assert(text.length>50000);const prior=storage.getItem('other-test');storage.fail=true;
 assert.throws(()=>m.answer(item,'other',text),/quota failure/);assert.equal(storage.getItem('other-test'),prior);assert.equal(m.get().answers.length,0);
 storage.fail=false;m.answer(item,'other',text);assert.equal(m.get().answers[0].otherText,text);
 let resumed=open(storage);assert.equal(resumed.get().answers[0].otherText,text);
 while(resumed.get().answers.length<4)resumed.skip(resumed.get().questionOrder[resumed.get().answers.length]);
 const envelope=JSON.parse(JSON.stringify({schema:'r13-local-export-v1',task:identity,strategy:resumed.get()}));
 assert.equal(R.readExport(envelope).strategy.answers[0].otherText,text);
 assert.equal(open(storage).get().answers[0].otherText,text);
 envelope.strategy.answers[0].otherText=' \n ';assert.throws(()=>R.readExport(envelope),/corrupt/);
});
test('legacy saved answers are retained without retroactively calling blank Other a skip',()=>{
 const storage=new Memory(),m=open(storage);m.answer(m.get().questionOrder[0],'other','temporary');
 const legacy=m.get();legacy.schema='r13-local-strategy-v1';legacy.answers[0].otherText='';delete legacy.answers[0].responsePolicyVersion;
 storage.setItem('other-test',JSON.stringify(legacy));const resumed=open(storage);assert.deepEqual(resumed.get(),legacy);
 const next=resumed.get().questionOrder[1];assert.throws(()=>resumed.answer(next,'other',''),/Describe your approach/);
 const long='beyond500 '.repeat(100);resumed.answer(next,'other',long);
 const exported=R.readExport({schema:'r13-local-export-v1',task:identity,strategy:resumed.get()});
 assert.equal(exported.strategy.answers[0].status,'answered');assert.equal(exported.strategy.answers[0].otherText,'');
 assert.equal(exported.strategy.answers[1].otherText,long);assert.equal(exported.strategy.answers[1].responsePolicyVersion,R.VERSION);
 exported.strategy.answers[1].otherText='';assert.throws(()=>R.readExport({schema:'r13-local-export-v1',task:identity,strategy:exported.strategy}),/corrupt/);
});
