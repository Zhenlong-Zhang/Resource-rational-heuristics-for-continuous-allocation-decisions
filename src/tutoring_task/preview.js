/* Local research-review shell. Never included in the participant Qualtrics kit. */
(function () {
  'use strict';
  const profileId=typeof TUTORING_PROFILE!=='undefined'?TUTORING_PROFILE:'legacy-4p30m-v1';
  const localStrategy=typeof TUTORING_LOCAL_STRATEGY!=='undefined'&&TUTORING_LOCAL_STRATEGY===true;
  const pilot=TutoringTaskEngine.profile({profileId}).pilot;
  const POINTER = localStrategy?'rr-tutoring-review-r13-local-active':pilot?'rr-tutoring-review-'+profileId+'-active':'rr-tutoring-review-v1-active';
  const root = document.getElementById('tutoring-root');
  const status = document.getElementById('preview-status');
  let app, active, strategy;
  const strategyRoot=localStrategy?document.createElement('div'):null;
  if(strategyRoot){strategyRoot.id='r13-root';strategyRoot.hidden=true;root.after(strategyRoot);}
  function randomIndex(bound){const limit=Math.floor(0x100000000/bound)*bound,draw=new Uint32Array(1);do{crypto.getRandomValues(draw);}while(draw[0]>=limit);return draw[0]%bound;}
  function identity(data,state){return {sessionId:data.options.sessionId,participantId:data.options.participantId,conditionId:data.options.conditionId,profileId:state.profileId,participantFlowVersion:state.participantFlowVersion,designFingerprint:state.designFingerprint};}
  function ensureStrategy(state){
    if(!localStrategy||strategy||!['posttask','complete'].includes(state.phase))return;
    if(state.currentTrialIndex!==14||state.pendingCommit!==null||(state.archivedTrialCount||0)+state.completedTrials.length!==14||state.schedule.filter(x=>x.stage==='practice').length!==4||state.schedule.filter(x=>x.stage==='main').length!==10)throw new Error('Final task record has not been acknowledged.');
    strategy=TutoringLocalStrategy.mount({element:strategyRoot,taskRoot:root,storage:localStorage,key:active.key+'-strategy',identity:identity(active,state),random:randomIndex,requiredExisting:state.phase==='complete'});
  }
  function token() {
    if (!globalThis.crypto || !crypto.getRandomValues) throw new Error('Secure session identity unavailable. Use a localhost preview.');
    return Array.from(crypto.getRandomValues(new Uint32Array(4)), x => x.toString(16).padStart(8, '0')).join('');
  }
  function launch(data) {
    if (app) app.destroy();
    if (strategy) strategy.destroy();
    strategy=null;
    root.hidden=false;
    active = data;
    document.getElementById('preview-condition').value = data.options.conditionId;
    app = TutoringTaskUI.mount(root, {design:TUTORING_DESIGN,sessionOptions:data.options,
      persistenceKey:data.key, storage:localStorage, reviewMode:true,
      ...(localStrategy?{onChange:ensureStrategy}:{}),
      ...(profileId===TutoringTaskEngine.R9?{guideAsset:typeof TUTORING_R9_GUIDE_ASSET!=='undefined'?TUTORING_R9_GUIDE_ASSET:null}: {})});
    if(localStrategy){const state=app.getState();if(state)ensureStrategy(state);}
    status.textContent = 'Local review session · '+data.options.conditionId+' · '+data.options.sessionId;
  }
  function create() {
    const id = token();
    const data = {key:'rr-review-session-'+(pilot?profileId+'-':'')+id, options:{conditionId:document.getElementById('preview-condition').value,
      sessionId:id, participantId:'review-'+id, seed:token(),...(pilot?{profileId}: {}),...(localStrategy?{participantFlowVersion:TutoringTaskEngine.R10_FLOW}: {})}};
    // Verify durable identity storage before rendering a new task; do not erase old sessions.
    localStorage.setItem(POINTER,JSON.stringify(data));
    launch(data);
  }
  function fail(error) { status.textContent = 'Cannot open safely: '+error.message; }
  document.getElementById('preview-new').addEventListener('click', () => {
    if (active && !confirm('Create a new local review session? The current session will remain in this browser storage. Download it first if you need a portable copy.')) return;
    try { create(); } catch (error) { fail(error); }
  });
  function downloadReview() {
    try {
      const task = app ? app.exportData() : {recoveryRaw:active && localStorage.getItem(active.key),error:'No mounted task'};
      const record = localStrategy?{schema:'r13-local-export-v1',task,strategy:strategy?strategy.get():null,...(strategy&&strategy.recovery()?{strategyRecovery:strategy.recovery()}:{})}:task;
      const blob = new Blob([typeof record === 'string' ? record : JSON.stringify(record,null,2)],{type:'application/json'});
      const link = document.createElement('a'); link.href=URL.createObjectURL(blob);
      link.download='tutoring-local-review-'+(active ? active.options.sessionId : 'recovery')+'.json';
      link.click(); setTimeout(()=>URL.revokeObjectURL(link.href),1000);
    } catch (error) { fail(error); }
  }
  document.getElementById('preview-export').addEventListener('click', downloadReview);
  if(localStrategy)root.addEventListener('click',event=>{const target=event.target.closest('[data-action="export"]');if(target){event.preventDefault();event.stopImmediatePropagation();downloadReview();}},true);
  try {
    const raw = localStorage.getItem(POINTER);
    if (raw) {
      const data=JSON.parse(raw);
      if (!data.key || !data.options || !data.options.sessionId || !/^C[1-5]$/.test(data.options.conditionId)) throw new Error('Invalid local session pointer; existing data were not overwritten.');
      launch(data);
    } else create();
  } catch (error) { fail(error); }
}());
