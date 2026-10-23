'use strict';
const E = require('./panel_engine.cjs');
  'use strict';
  const LIMIT=14000, CHUNKS=4;
  const clone=x=>JSON.parse(JSON.stringify(x));
  function digest(text){
    // Integrity/error-detection only, not authentication or anti-cheating.
    let a=2166136261,b=0x9e3779b9;
    for(const byte of new TextEncoder().encode(text)){a=Math.imul(a^byte,16777619);b=Math.imul(b^byte,2246822519);}
    return (a>>>0).toString(16).padStart(8,'0')+(b>>>0).toString(16).padStart(8,'0');
  }
  function pack(value,extra={}){
    const text=JSON.stringify(value),encoder=new TextEncoder(),parts=[];
    let current='',bytes=0;
    for(const char of text){const n=encoder.encode(char).length;
      if(bytes+n>LIMIT){parts.push(current);current='';bytes=0;} current+=char;bytes+=n;}
    parts.push(current);
    if(parts.length>CHUNKS) throw new Error('Record exceeds fixed embedded-data capacity; export locally and stop. Nothing was truncated.');
    return {parts,meta:{schema:'rr_transport_v1',...extra,count:parts.length,bytes:encoder.encode(text).length,digest:digest(text)}};
  }
  function unpack(parts,meta){
    if(!meta||meta.schema!=='rr_transport_v1'||!Number.isInteger(meta.count)||meta.count<1||meta.count>CHUNKS) throw new Error('Invalid saved transport metadata');
    if(parts.slice(0,meta.count).some(p=>typeof p!=='string'||new TextEncoder().encode(p).length>LIMIT)) throw new Error('Missing or oversized saved chunk');
    const text=parts.slice(0,meta.count).join('');
    if(new TextEncoder().encode(text).length!==meta.bytes||digest(text)!==meta.digest) throw new Error('Incomplete or corrupt platform checkpoint');
    return JSON.parse(text);
  }
  function fields(stem,payload){
    return [...Array.from({length:CHUNKS},(_,i)=>[stem+'_'+i,payload.parts[i]||'']),[stem+'_meta',JSON.stringify(payload.meta)]];
  }
  function read(host,stem){
    const raw=host.get(stem+'_meta');
    if(!raw) return null;
    return {meta:JSON.parse(raw),parts:Array.from({length:CHUNKS},(_,i)=>host.get(stem+'_'+i))};
  }
  // Versioned participant-flow receipt, not a completed-study or analysis label.
  // The legacy final-record route remains unchanged; no statistical inference here.
  function readTerminalCapture(host,design) {
    const final=read(host,'rr_final');if(!final)return null;
    const record=unpack(final.parts,final.meta);
    if(record.schema==='rr_final_record_v1')return null;
    if(record.schema!=='rr_terminal_record_v1')throw new Error('Unsupported terminal record schema');
    const cp=read(host,'rr_checkpoint');if(!cp)throw new Error('Terminal checkpoint missing');
    const checkpoint=unpack(cp.parts,cp.meta),state=E.restore(checkpoint,design),last=state.checks.at(-1);
    if(!E.hasR10Flow(state)||state.phase!=='ineligible'||state.current!==null||state.pendingCommit!==null||![0,4].includes(state.currentTrialIndex))throw new Error('Terminal state is not a supported failed understanding stage');
    if(record.terminalReason!=='three_failed_understanding_attempts'||!last||!['comprehension','practiceCheck'].includes(last.kind)||record.failedCheckKind!==last.kind||record.endedAt!==last.at||record.originalSlot!==state.currentTrialIndex)throw new Error('Terminal reason or stage mismatch');
    const attempts=state.checks.filter(x=>x.kind===last.kind);
    if(attempts.length!==3||attempts.some(x=>x.missed.length===0)||state.completedAt!==null)throw new Error('Terminal failure evidence mismatch');
    if(record.sessionId!==state.sessionId||record.conditionId!==state.conditionId||JSON.stringify(record.exported)!==JSON.stringify(E.exportData(state)))throw new Error('Terminal export/checkpoint mismatch');
    const slot=state.currentTrialIndex;
    if(state.archivedTrialCount!==slot||state.completedTrials.length!==0||state.historyStoredExternally!==(slot>0))throw new Error('Terminal archived history count mismatch');
    for(const meta of [cp.meta,final.meta])if(meta.sessionId!==state.sessionId||meta.fromSlot!==slot||meta.toSlot!==slot+1||typeof meta.txn!=='string')throw new Error('Terminal native identity/slot mismatch');
    if(cp.meta.txn!==final.meta.txn||cp.meta.txn!==digest(JSON.stringify({slot,sessionId:state.sessionId,record,checkpoint,final:true})))throw new Error('Terminal native transaction mismatch');
    for(let i=1;i<=14;i++){
      const stem='rr_t'+String(i).padStart(2,'0'),packet=read(host,stem);
      if(!!packet!==(i<=slot))throw new Error('Terminal round record count mismatch');
      if(packet){
        const value=unpack(packet.parts,packet.meta),trial=value.trial,definition=state.schedule[i-1];
        if(value.schema!=='rr_native_trial_v1'||value.sessionId!==state.sessionId||value.conditionId!==state.conditionId||packet.meta.sessionId!==state.sessionId||packet.meta.fromSlot!==i-1||packet.meta.toSlot!==i||typeof packet.meta.txn!=='string'||!/^[0-9a-f]{16}$/.test(packet.meta.txn))throw new Error('Terminal prior-round transport identity mismatch');
        if(!trial||trial.status!=='complete'||trial.participantId!==state.participantId||trial.conditionId!==state.conditionId||trial.profileId!==state.profileId||trial.designFingerprint!==state.designFingerprint||trial.participantFlowVersion!==state.participantFlowVersion||trial.instructionVersion!==state.instructionVersion||trial.checkVersion!==state.checkVersion||trial.trialId!==definition.trialId||JSON.stringify(trial.definition)!==JSON.stringify(definition)||trial.definition.scheduleIndex!==i-1)throw new Error('Terminal prior-round definition/version mismatch');
      }
    }
    return {state,record,checkpoint,meta:final.meta};
  }
module.exports={LIMIT,CHUNKS,digest,pack,unpack,fields,read,readTerminalCapture};
