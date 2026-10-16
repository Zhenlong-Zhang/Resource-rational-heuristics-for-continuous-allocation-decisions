/* Pure deterministic schedule extraction. No UI, participant data or normalizer. */
'use strict';
const VERSION = 'tutoring-task-0.1.0';
const PILOT = 'pilot-4p10m-v1';
function profile(options) {
  const id=options && options.profileId || 'legacy-4p30m-v1';
  assert(id===PILOT||id==='legacy-4p30m-v1','Unsupported task profile');
  return {id,pilot:id===PILOT,practiceTrials:4,mainTrials:id===PILOT?10:30,totalTrials:id===PILOT?14:34,maxCheckAttempts:id===PILOT?3:2};
}
function assert(ok, message) { if (!ok) throw new Error(message); }
function stable(x) {
  if (Array.isArray(x)) return '[' + x.map(stable).join(',') + ']';
  if (x && typeof x === 'object') return '{' + Object.keys(x).sort().map(k => JSON.stringify(k) + ':' + stable(x[k])).join(',') + '}';
  return JSON.stringify(x);
}
function hashWords(text) {
  let a = 1779033703, b = 3144134277, c = 1013904242, d = 2773480762;
  for (let i = 0; i < text.length; i++) {
    const k = text.charCodeAt(i);
    a = b ^ Math.imul(a ^ k, 597399067); b = c ^ Math.imul(b ^ k, 2869860233);
    c = d ^ Math.imul(c ^ k, 951274213); d = a ^ Math.imul(d ^ k, 2716044179);
  }
  a = Math.imul(c ^ (a >>> 18), 597399067); b = Math.imul(d ^ (b >>> 22), 2869860233);
  c = Math.imul(a ^ (c >>> 17), 951274213); d = Math.imul(b ^ (d >>> 19), 2716044179);
  return [(a ^ b ^ c ^ d) >>> 0, (b ^ a) >>> 0, (c ^ a) >>> 0, (d ^ a) >>> 0];
}
function rng(parts) {
  let [a, b, c, d] = hashWords(stable(parts));
  const next = function () {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0; a = b ^ (b >>> 9); b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11); d = (d + 1) | 0; t = (t + d) | 0; c = (c + t) | 0;
    return ((t >>> 0) + 0.5) / 4294967296;
  };
  for (let i = 0; i < 12; i++) next();
  return next;
}
function validateDesign(design) {
  // Validate only inputs consumed by this extraction; no display/approval fields.
  assert(stable(design.common.variation.scale_factors)==='[0.95,1,1.05]', 'Unsupported scales');
  assert(stable(design.common.variation.gap_scale_cell_counts_for_three_gap_conditions)==='[[4,3,3],[3,4,3],[3,3,4]]', 'Unsupported balance');
  const expected = [ [[50,50],[3,3],30,120,1.2], [[45,75],[2,2],6,150,1.2], [[50,70],[18,18],6,170,.5], [[45,65],[2,2],6,55,.6], [[55,55],[8,8],3,55,.3] ];
  assert(design.conditions.length===5, 'Five conditions required');
  design.conditions.forEach((row,i)=>{
    assert(row.id==='C'+(i+1), 'Unexpected condition order');
    assert(stable([row.prior_kernel_centres,row.prior_kernel_spreads,row.report_kernel_width,row.total_time,row.sample_time_cost])===stable(expected[i]), 'Unsupported scientific recipe');
    assert(stable(row.mean_gap_offsets)===(i===1||i===3?'[-5,0,5]':'[0]'), 'Unsupported gaps');
  });
}
function shuffled(values, random) {
  const a = values.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function seedKey(options, namespace) { const key=[VERSION, String(options.seed), options.participantId, options.sessionId, options.conditionId, namespace];return profile(options).pilot?key.concat(PILOT):key; }
function makeSchedule(design, options) {
  validateDesign(design);
  const row = design.conditions.find(x => x.id === options.conditionId);
  assert(row, 'A C1-C5 assignment is required');
  const cfg=profile(options),variant=cfg.pilot?(options.scheduleVariant===undefined?hashWords(stable(seedKey(options,'pilot-variant')))[0]%9:options.scheduleVariant):null;
  if(cfg.pilot)assert(Number.isInteger(variant)&&variant>=0&&variant<=8,'Schedule variant must be 0-8');
  const scales = design.common.variation.scale_factors, cells = design.common.variation.gap_scale_cell_counts_for_three_gap_conditions;
  const items = [];
  scales.forEach((scale, si) => row.mean_gap_offsets.forEach((gap, gi) => {
    const count = cfg.pilot?(row.mean_gap_offsets.length===1?3+(si===variant%3?1:0):1+(si===Math.floor(variant/3)&&gi===variant%3?1:0)):(row.mean_gap_offsets.length === 1 ? 10 : cells[si][gi]);
    for (let k = 0; k < count; k++) items.push({scale, gap, templateId: row.id + '_s' + si + '_g' + gi});
  }));
  const random = rng(seedKey(options, 'main-order'));
  let ordered;
  for (let attempt = 0; attempt < 10000; attempt++) {
    const test = shuffled(items, random);
    if (!test.some((x, i) => i >= 3 && x.scale === test[i-1].scale && x.scale === test[i-2].scale && x.scale === test[i-3].scale)) { ordered = test; break; }
  }
  assert(ordered, 'Could not construct balanced trial order');
  const sides = shuffled(Array(cfg.mainTrials/2).fill(false).concat(Array(cfg.mainTrials/2).fill(true)), rng(seedKey(options, 'main-sides')));
  const practiceSides = shuffled([false,false,true,true], rng(seedKey(options, 'practice-sides')));
  const practice = Array.from({length:4}, (_, i) => ({scale:1,gap:0,templateId:row.id + '_practice_central',sideReversed:practiceSides[i],stage:'practice',stageIndex:i+1}));
  const main = ordered.map((x, i) => Object.assign({}, x, {sideReversed:sides[i],stage:'main',stageIndex:i+1}));
  return practice.concat(main).map((x, i) => {
    const m = row.prior_kernel_centres;
    return Object.assign(x, {scheduleIndex:i,trialId:x.stage + '-' + String(x.stageIndex).padStart(2,'0'),
      priorCentres:[(m[0]+x.gap)*x.scale,(m[1]-x.gap)*x.scale], priorSpreads:row.prior_kernel_spreads.map(v=>v*x.scale),
      reportWidth:row.report_kernel_width*x.scale,budgetMillihours:Math.round(row.total_time*x.scale*1000),costMillihours:Math.round(row.sample_time_cost*x.scale*1000)});
  });
}
function options(cid,variant,seed,prefix){
  const id=prefix+'-'+cid+'-v'+variant;
  return {conditionId:cid,scheduleVariant:variant,seed:String(seed),participantId:id,
    sessionId:id+'-session',profileId:PILOT,startedAt:'2026-10-07T00:00:00.000Z'};
}
function schedules(design,spec){
  return spec.conditions.map(cid=>{
    const o=options(cid,spec.scheduleVariant,spec.seed,spec.identityPrefix);
    return {conditionId:cid,options:o,definitions:makeSchedule(design,o).filter(x=>x.stage==='main')};
  });
}
module.exports={schedules,makeSchedule,stable,hashWords,rng};
if(require.main===module){let raw='';process.stdin.setEncoding('utf8');process.stdin.on('data',x=>raw+=x);process.stdin.on('end',()=>{
  try{const r=JSON.parse(raw);if(r.mode!=='schedules')throw Error('Only deterministic schedules are supported');
    process.stdout.write(JSON.stringify({ok:true,result:schedules(r.design,r.spec)}));}
  catch(e){process.stdout.write(JSON.stringify({ok:false,error:e.message}));process.exitCode=1;}
});}
