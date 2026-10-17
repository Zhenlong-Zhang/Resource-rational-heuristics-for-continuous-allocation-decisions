/* Shared finite-task engine. No network, DOM, legacy model, or policy recommendations. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.TutoringTaskEngine = factory();
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const VERSION = "tutoring-task-0.1.0";
  const SCHEMA = "tutoring-session-1";
  const PILOT = "pilot-4p10m-v1";
  const ENGAGEMENT = "pilot-engagement-budget-v1";
  const R9 = "pilot-r9-design-v1";
  const R9_VERSIONS = Object.freeze({resolvedRecipeVersion:"r9-c1-gap1-unscaled-v1",
    instructionVersion:"r9-instructions-v1",displayVersion:"r9-finite95-channel80-decimal-v1",
    checkVersion:"r9-understanding-sets-v1",feedbackStyle:"r9-outcome-two-decimal-v1"});
  const R9_GUIDE = "r9-c3-report50-guide-v1";
  function profile(options) {
    const id=options && options.profileId || "legacy-4p30m-v1";
    assert(id===PILOT||id===ENGAGEMENT||id===R9||id==="legacy-4p30m-v1","Unsupported task profile");
    const pilot=id===PILOT||id===ENGAGEMENT||id===R9;
    return {id,pilot,practiceTrials:4,mainTrials:pilot?10:30,totalTrials:pilot?14:34,maxCheckAttempts:pilot?3:2,...(id===ENGAGEMENT||id===R9?{budgetOnly:true}: {}),...(id===R9?{r9:true}: {})};
  }
  function questionCapacity(state,definition=state.current.definition) {
    assert(Number.isSafeInteger(definition.budgetMillihours)&&definition.budgetMillihours>=0&&Number.isSafeInteger(definition.costMillihours)&&definition.costMillihours>0,"Question cost must be a positive integer");
    return profile(state).budgetOnly?Math.floor(definition.budgetMillihours/definition.costMillihours):20;
  }
  const clone = x => JSON.parse(JSON.stringify(x));
  function assert(ok, message) { if (!ok) throw new Error(message); }
  function stable(x) {
    if (Array.isArray(x)) return "[" + x.map(stable).join(",") + "]";
    if (x && typeof x === "object") return "{" + Object.keys(x).sort().map(k => JSON.stringify(k) + ":" + stable(x[k])).join(",") + "}";
    return JSON.stringify(x);
  }
  // cyrb128 is a reproducible non-cryptographic seed/checksum, not authentication.
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
  function fingerprint(value) { return hashWords(stable(value)).map(v => v.toString(16).padStart(8, "0")).join(""); }
  function resolveProfile(design,options) {
    validateDesign(design);
    const id=profile(options).id;
    return {profileId:id,baseDesignFingerprint:fingerprint(design),...(id===R9?{
      resolvedRecipeVersion:R9_VERSIONS.resolvedRecipeVersion,c1TotalCentreGapMinutes:1,
      c1CentreRule:"50*scale-minus/plus-totalGap/2;gap-is-unscaled",
      gapConvention:"C1-total-centre-gap-unscaled;other-conditions-scaled-row-offset",
      gapFieldMeaning:"schedule-row-offset;not-C1-total-centre-gap",scientificSeedNamespace:PILOT,
      guideRecipe:clone(design.conditions.find(row=>row.id==="C3"))}: {})};
  }
  function sessionFingerprint(design,options) {
    return profile(options).r9?fingerprint({design,profileId:R9,resolvedRecipe:resolveProfile(design,options)}):fingerprint(design);
  }
  function r9Metadata(design) {
    const resolvedRecipe=resolveProfile(design,{profileId:R9});
    return {...R9_VERSIONS,resolvedRecipe,resolvedRecipeFingerprint:fingerprint(resolvedRecipe)};
  }
  function formatR9Number(value,places) {
    assert(Number.isFinite(value)&&Math.abs(value)<1e21&&Number.isInteger(places)&&places>=0&&places<=3,"Invalid R9 display number");
    const text=value.toFixed(places);
    return Number(text)===0?text.replace(/^-/,''):text;
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
  function normalize(logs) {
    assert(Array.isArray(logs) && logs.length === 100 && logs.every(Number.isFinite), "Expected 100 finite log weights");
    const m = Math.max.apply(null, logs), z = m + Math.log(logs.reduce((s, x) => s + Math.exp(x - m), 0));
    return logs.map(x => x - z);
  }
  function prior(centre, spread) {
    assert(Number.isFinite(centre) && spread > 0 && Number.isFinite(spread), "Invalid prior");
    return normalize(Array.from({length: 100}, (_, i) => -0.5 * ((i + 1 - centre) / spread) ** 2));
  }
  const likelihoodCache = new Map();
  function likelihood(width) {
    assert(width > 0 && Number.isFinite(width), "Invalid report width");
    if (!likelihoodCache.has(width)) {
      const rows = Array.from({length: 100}, (_, n) => Object.freeze(normalize(Array.from({length: 100}, (_, y) => -0.5 * ((y - n) / width) ** 2))));
      likelihoodCache.set(width, Object.freeze(rows));
    }
    return likelihoodCache.get(width);
  }
  function posterior(logPrior, report, logLikelihood) {
    assert(Number.isInteger(report) && report >= 1 && report <= 100, "Report outside 1–100");
    return normalize(logPrior.map((x, n) => x + logLikelihood[n][report - 1]));
  }
  function moments(logs) {
    const probabilities = normalize(logs).map(Math.exp);
    const mean = probabilities.reduce((s, p, i) => s + p * (i + 1), 0);
    const sd = Math.sqrt(probabilities.reduce((s, p, i) => s + p * (i + 1 - mean) ** 2, 0));
    return {mean, sd, probabilities};
  }
  function centralRange(probabilities,coverage=.9) {
    assert(probabilities.length===100&&coverage>0&&coverage<1,"Invalid central range");
    const tail=(1-coverage)/2;let sum=0,lower=null,upper=100;
    for(let i=0;i<100;i++){sum+=probabilities[i];if(lower===null&&sum>=tail)lower=i+1;if(sum>=1-tail){upper=i+1;break;}}
    return {lower,upper,coverage:probabilities.slice(lower-1,upper).reduce((a,b)=>a+b,0),targetCoverage:coverage};
  }
  function r9CredibleRange(probabilities) {
    assert(Array.isArray(probabilities)&&probabilities.length===100&&probabilities.every(p=>Number.isFinite(p)&&p>=0),"Invalid R9 posterior probabilities");
    const total=probabilities.reduce((a,b)=>a+b,0);
    assert(Math.abs(total-1)<=1e-10,"R9 posterior must be normalized");
    let sum=0,lower=null,upper=null;
    for(let i=0;i<100;i++){sum+=probabilities[i];if(lower===null&&sum>=.025)lower=i+1;if(upper===null&&sum>=.975)upper=i+1;}
    assert(lower!==null&&upper!==null,"R9 interval quantiles unavailable");
    return {lower,upper,coverage:probabilities.slice(lower-1,upper).reduce((a,b)=>a+b,0),targetCoverage:.95,semanticsVersion:R9_VERSIONS.displayVersion};
  }
  const precisionCache=new Map();
  function sharedChannelPrecision(width,target=.8) {
    assert(Number.isFinite(target)&&target>0&&target<1,"Invalid channel coverage target");
    const key=String(width)+"|"+String(target);
    if(!precisionCache.has(key)){
      const rows=likelihood(width),cumulative=rows.map((row,n)=>{
        const mass=Array(100).fill(0);row.forEach((x,y)=>{mass[Math.abs(y-n)]+=Math.exp(x);});
        for(let e=1;e<100;e++)mass[e]+=mass[e-1];return mass;
      });
      let error=0;while(error<99&&Math.min(...cumulative.map(row=>row[error]))<target)error++;
      const shown=Math.min(100,5*Math.ceil(error/5));
      const coverage=cumulative.map(row=>row[Math.min(99,shown)]),minimum=Math.min(...coverage);
      assert(minimum>=target,"Channel guarantee cannot meet target");
      precisionCache.set(key,Object.freeze({targetCoverage:target,minimumIntegerError:error,displayedError:shown,
        minimumCoverage:minimum,worstNeed:coverage.indexOf(minimum)+1,semanticsVersion:R9_VERSIONS.displayVersion}));
    }
    return clone(precisionCache.get(key));
  }
  function predictiveRMSE(logs, logLikelihood) {
    const p = normalize(logs).map(Math.exp);
    let square = 0;
    for (let n = 0; n < 100; n++) for (let y = 0; y < 100; y++) square += p[n] * Math.exp(logLikelihood[n][y]) * (y - n) ** 2;
    return Math.sqrt(square);
  }
  function reportAccuracy(probabilities,width,coverage=.8) {
    const rows=likelihood(width),errors=Array(100).fill(0);
    for(let n=0;n<100;n++)for(let y=0;y<100;y++)errors[Math.abs(y-n)]+=probabilities[n]*Math.exp(rows[n][y]);
    let mass=0;for(let error=0;error<100;error++){mass+=errors[error];if(mass>=coverage)return {error,coverage:mass,targetCoverage:coverage};}
    return {error:99,coverage:mass,targetCoverage:coverage};
  }
  function outcome(budgetMillihours, shareIndex, trueNeeds) {
    assert(Number.isSafeInteger(budgetMillihours) && budgetMillihours >= 0, "Invalid budget");
    assert(Array.isArray(trueNeeds) && trueNeeds.length === 2 && trueNeeds.every(x => Number.isInteger(x) && x >= 1 && x <= 100), "Invalid true need");
    assert(budgetMillihours === 0 ? shareIndex === null : Number.isInteger(shareIndex) && shareIndex >= 0 && shareIndex <= 1000, "Allocation must be actively entered");
    const amountsMicrohours = budgetMillihours ? [budgetMillihours * shareIndex, budgetMillihours * (1000 - shareIndex)] : [0, 0];
    assert(amountsMicrohours.every(Number.isSafeInteger), "Allocation exceeds exact integer range");
    const surplus = amountsMicrohours.map((x, i) => (x - trueNeeds[i] * 1000000) / 1000000);
    const utilities = surplus.map(x => x >= 0 ? Math.sqrt(x) : -2 * Math.sqrt(-x));
    const gpas = utilities.map(x => 3.5 + 0.03 * x);
    const totalUtility = utilities[0] + utilities[1];
    return {amountsMicrohours, surplus, utilities, gpas, totalUtility, totalGPA: 7 + 0.03 * totalUtility};
  }
  function validateDesign(design) {
    const c = design.common;
    assert(c && design.assignment.design === "between_subjects" && design.assignment.conditions_per_participant === 1, "Unsupported assignment");
    assert(design.assignment.main_trials === 30 && design.assignment.practice_trials === 4, "Unsupported session dose");
    assert(design.assignment.blocks === 3 && design.assignment.trials_per_block === 10 && c.goal === "maximize_combined_final_GPA_under_supplied_task_rule", "Unsupported block structure/objective");
    assert(c.posterior === "exact_discrete_bayes_over_100_points" && c.initial_information_mode === "direct_specified_prior_not_simulated_historical_prior_samples", "Unsupported belief semantics");
    assert(c.latent_need.family === "finite_normalized_gaussian_kernel" && c.observation.family === "finite_normalized_gaussian_kernel_conditional_on_true_need", "Unsupported probability family");
    for (const x of [c.latent_need, c.observation]) assert(x.support_min === 1 && x.support_max === 100 && x.support_step === 1, "Unsupported support");
    assert(c.latent_need.independent_recipients && c.latent_need.fixed_within_trial && c.latent_need.independent_new_students_across_trials && c.observation.independent_given_latent_need && c.observation.assessment_does_not_teach && c.observation.normalization_depends_on_latent_need, "Unsupported independence/learning");
    assert(c.stop_time_cost === 0 && c.wall_clock_time_cost === 0 && c.maximum_online_assessments === 20 && c.pretrial_sample_count_per_student === 0 && stable(c.learning_rates) === "[1,1]", "Unsupported resource rules");
    assert(c.allocation.share_grid_denominator === 1000 && c.allocation.initial_response === null && c.allocation.input_step_percentage_points === 0.1 && c.allocation.stop_irreversible, "Unsupported allocation interface");
    assert(c.fixed_point_time_unit_hours === .001 && c.allocation.canonical_amount_unit_hours === .000001 && stable(c.allocation.canonical_share_index_range) === "[0,1000]" && c.allocation.include_exact_zero_and_all_remaining === true, "Unsupported canonical amount units");
    assert(c.utility.gain_exponent === .5 && c.utility.loss_exponent === .5 && c.utility.loss_multiplier === 2 && c.gpa.baseline === 3.5 && c.gpa.utility_scale === .03 && c.gpa.clipping === false, "Unsupported GPA payoff");
    assert(stable(c.variation.scale_factors) === "[0.95,1,1.05]" && stable(c.variation.side_assignment_counts) === "[15,15]", "Unsupported recipe scales/sides");
    assert(stable(c.variation.gap_scale_cell_counts_for_three_gap_conditions) === "[[4,3,3],[3,4,3],[3,3,4]]", "Unsupported cell balance");
    assert(c.variation.trials_per_scale === 10 && c.variation.maximum_consecutive_same_scale === 3 && c.variation.human_trials_selected_by_target_rule_performance === false && c.display.probability_plot === true, "Unsupported order/selection/distribution display");
    assert(c.feedback.every_trial === true && c.feedback.show_true_need === true && c.feedback.show_student_final_GPA === true && c.feedback.show_total_GPA === true && c.feedback.show_resource_accounting === true && c.feedback.show_optimal_allocation === false && c.feedback.show_heuristic_labels === false, "Unsupported feedback");
    const expected = [ [[50,50],[3,3],30,120,1.2], [[45,75],[2,2],6,150,1.2], [[50,70],[18,18],6,170,.5], [[45,65],[2,2],6,55,.6], [[55,55],[8,8],3,55,.3] ];
    assert(design.conditions.length === 5, "Five conditions required");
    design.conditions.forEach((row, i) => {
      assert(row.id === "C" + (i + 1), "Unexpected condition order");
      assert(stable([row.prior_kernel_centres,row.prior_kernel_spreads,row.report_kernel_width,row.total_time,row.sample_time_cost]) === stable(expected[i]), "Only selected D111 scientific recipes are supported");
      assert(stable(row.mean_gap_offsets) === (i === 1 || i === 3 ? "[-5,0,5]" : "[0]"), "Unsupported gap recipe");
    });
    return design;
  }
  function shuffled(values, random) {
    const a = values.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }
  function seedKey(options, namespace) { const key=[VERSION, String(options.seed), options.participantId, options.sessionId, options.conditionId, namespace];return profile(options).pilot?key.concat(PILOT):key; }
  function presentations(options) {
    // Presentation seeds exclude condition, recipe, need, outcome and scientific RNG.
    const key=["trial-presentation-v1",String(options.seed),options.participantId,options.sessionId];
    const names=shuffled(["Alex","Jordan","Taylor","Morgan","Casey","Riley","Jamie","Avery","Quinn","Sam","Charlie","Drew","Blake","Robin","Cameron","Reese","Rowan","Parker","Finley","Skyler","Dakota","Emerson","Hayden","Sage","Remy","Jules","Kendall","Lane"],rng(key.concat("names")));
    const subjects=["mathematics","biology","chemistry","physics","history","literature","geography"];
    const random=rng(key.concat("subjects"));let order;
    for(let i=0;i<10000;i++){const trial=shuffled(subjects.concat(subjects),random);if(trial.every((x,j)=>j===0||x!==trial[j-1])){order=trial;break;}}
    assert(order,"Could not construct varied subject order");
    return order.map((subject,i)=>({version:"trial-presentation-v1",names:names.slice(2*i,2*i+2),subject}));
  }
  function makeSchedule(design, options) {
    validateDesign(design);
    const row = design.conditions.find(x => x.id === options.conditionId);
    assert(row, "An external C1–C5 assignment is required");
    const cfg=profile(options),variant=cfg.pilot?(options.scheduleVariant===undefined?hashWords(stable(seedKey(options,"pilot-variant")))[0]%9:options.scheduleVariant):null;
    if(cfg.pilot)assert(Number.isInteger(variant)&&variant>=0&&variant<=8,"Pilot schedule variant must be0–8");
    const scales = design.common.variation.scale_factors, cells = design.common.variation.gap_scale_cell_counts_for_three_gap_conditions;
    const items = [];
    scales.forEach((scale, si) => row.mean_gap_offsets.forEach((gap, gi) => {
      const count = cfg.pilot?(row.mean_gap_offsets.length===1?3+(si===variant%3?1:0):1+(si===Math.floor(variant/3)&&gi===variant%3?1:0)):(row.mean_gap_offsets.length === 1 ? 10 : cells[si][gi]);
      for (let k = 0; k < count; k++) items.push({scale, gap, templateId: cfg.r9&&row.id==="C1"?"C1_s"+si+"_d1000":row.id + "_s" + si + "_g" + gi});
    }));
    const random = rng(seedKey(options, "main-order"));
    let ordered;
    for (let attempt = 0; attempt < 10000; attempt++) {
      const test = shuffled(items, random);
      if (!test.some((x, i) => i >= 3 && x.scale === test[i-1].scale && x.scale === test[i-2].scale && x.scale === test[i-3].scale)) { ordered = test; break; }
    }
    assert(ordered, "Could not construct balanced trial order");
    const sides = shuffled(Array(cfg.mainTrials/2).fill(false).concat(Array(cfg.mainTrials/2).fill(true)), rng(seedKey(options, "main-sides")));
    const practiceSides = shuffled([false,false,true,true], rng(seedKey(options, "practice-sides")));
    const practice = Array.from({length:4}, (_, i) => ({scale:1,gap:0,templateId:cfg.r9&&row.id==="C1"?"C1_practice_r9_d1000":row.id + "_practice_central",sideReversed:practiceSides[i],stage:"practice",stageIndex:i+1}));
    const main = ordered.map((x, i) => Object.assign({}, x, {sideReversed:sides[i],stage:"main",stageIndex:i+1}));
    const labels=cfg.budgetOnly?presentations(options):null;
    return practice.concat(main).map((x, i) => {
      const m = row.prior_kernel_centres;
      return Object.assign(x, {scheduleIndex:i,trialId:x.stage + "-" + String(x.stageIndex).padStart(2,"0"),
        priorCentres:cfg.r9&&row.id==="C1"?[50*x.scale-.5,50*x.scale+.5]:[(m[0]+x.gap)*x.scale,(m[1]-x.gap)*x.scale], priorSpreads:row.prior_kernel_spreads.map(v=>v*x.scale),
        reportWidth:row.report_kernel_width*x.scale,budgetMillihours:Math.round(row.total_time*x.scale*1000),costMillihours:Math.round(row.sample_time_cost*x.scale*1000),...(labels?{presentation:labels[i]}: {}),
        ...(cfg.r9?{profileId:R9,resolvedRecipeVersion:R9_VERSIONS.resolvedRecipeVersion,displayVersion:R9_VERSIONS.displayVersion,gapConvention:"C1-total-centre-gap-unscaled;other-conditions-scaled-row-offset",...(row.id==="C1"?{c1TotalCentreGapMinutes:1}: {})}: {})});
    });
  }
  function draw(logs, random) {
    const ps = normalize(logs).map(Math.exp), u = random(); let sum = 0;
    for (let i = 0; i < ps.length; i++) { sum += ps[i]; if (u < sum) return i + 1; }
    return 100;
  }
  function scenario(state, definition) {
    const prefix = seedKey(state, "scenario").concat(definition.trialId);
    const logicalNeeds = definition.priorCentres.map((m, i) => draw(prior(m,definition.priorSpreads[i]), rng(prefix.concat("latent",i))));
    const ll = likelihood(definition.reportWidth);
    const logicalReports = logicalNeeds.map((n, i) => Array.from({length:questionCapacity(state,definition)}, (_, k) => draw(ll[n-1],rng(prefix.concat("report",i,k)))));
    const mapping = definition.sideReversed ? [1,0] : [0,1];
    return {trueNeeds:mapping.map(i=>logicalNeeds[i]),potentialReports:mapping.map(i=>logicalReports[i]),logicalToDisplayed:definition.sideReversed ? ["B","A"] : ["A","B"]};
  }
  function now(action) { return action && action.at ? action.at : new Date().toISOString(); }
  function audit(state, type, data, action) { state.audit.push({type,at:now(action),revision:state.revision+1,...data}); }
  function event(state, type, action, details) {
    const t = state.current;
    const entry={eventIndex:t.events.length,id:action.id,type,at:now(action),budgetMillihours:t.budgetMillihours,assessmentCounts:t.assessmentCounts.slice(),...(details||{})};
    if(profile(state).pilot&&type==="allocation_edit"){
      const previous=t.events[t.events.length-1];
      if(previous&&previous.type==="allocation_edit"){
        t.events[t.events.length-1]={...entry,eventIndex:previous.eventIndex,
          firstEditId:previous.firstEditId||previous.id,firstEditAt:previous.firstEditAt||previous.at,
          firstShareIndex:previous.firstShareIndex===undefined?previous.shareIndex:previous.firstShareIndex,firstRawValue:previous.firstRawValue===undefined?previous.rawValue:previous.firstRawValue,
          editCount:(previous.editCount||1)+1,lastEditAt:entry.at,allocationEditLogging:"first_last_and_count"};return;
      }
      Object.assign(entry,{firstEditId:entry.id,firstEditAt:entry.at,firstShareIndex:entry.shareIndex,
        firstRawValue:entry.rawValue,editCount:1,lastEditAt:entry.at,allocationEditLogging:"first_last_and_count"});
    }
    t.events.push(entry);
  }
  function openTrial(state, action) {
    const definition = state.schedule[state.currentTrialIndex], hidden = scenario(state,definition);
    const mapping = definition.sideReversed ? [1,0] : [0,1];
    const logs = mapping.map(i=>prior(definition.priorCentres[i],definition.priorSpreads[i]));
    state.current = {definition:clone(definition),...hidden,initialLogs:clone(logs),logs,budgetMillihours:definition.budgetMillihours,
      assessmentCounts:[0,0],events:[],draftAllocation:"",shareIndex:null,stopReason:null,result:null};
    state.phase="trial"; event(state,"trial_open",action,{});
  }
  function createSession(design, options) {
    validateDesign(design);
    for (const key of ["seed","participantId","sessionId","conditionId"]) assert(options && typeof options[key] === "string" && options[key].length > 0, "Missing string session option: " + key);
    const cfg=profile(options),extra=cfg.pilot?{profileId:cfg.id,scheduleVariant:options.scheduleVariant===undefined?hashWords(stable(seedKey(options,"pilot-variant")))[0]%9:options.scheduleVariant,...(cfg.budgetOnly?{questionLimitRule:"budget_only",maximumOnlineAssessments:null,presentationVersion:"trial-presentation-v1",feedbackStyle:"outcome-engagement-v1"}: {})}:{};
    const r9=cfg.r9?{...r9Metadata(design),guideFixtureId:R9_GUIDE,guideFixtureHash:guideExample(design).fixtureHash,
      guideShownAt:null,guideAcknowledgedAt:null,practiceGuideHistory:[]}:{};
    return {schema:SCHEMA,engineVersion:VERSION,...extra,...r9,designFingerprint:sessionFingerprint(design,options),fingerprintAlgorithm:"cyrb128-stable-json-not-security",
      conditionId:options.conditionId,participantId:options.participantId,sessionId:options.sessionId,seed:options.seed,
      startedAt:options.startedAt||new Date().toISOString(),completedAt:null,phase:"instructions",revision:0,
      currentTrialIndex:0,schedule:makeSchedule(design,options),current:null,completedTrials:[],archivedTrialCount:0,
      historyStoredExternally:false,acceptedActionIds:[],audit:[],checks:[],posttaskDraft:{},pendingCommit:null};
  }
  const CHECKS = [
    {key:"cost",question:"60 tutoring hours; each assessment costs 3 hours. After two assessments, how many hours remain?",options:[["54","54 hours"],["60","60 hours"],["57","57 hours"]],correct:"54",explanation:"Each assessment costs 3 hours: 60 − 2 × 3 = 54 hours remain."},
    {key:"diagnosis",question:"Does an assessment teach the student or change their actual need?",options:[["no","No. It provides imperfect information."],["yes","Yes. It teaches the student."]],correct:"no",explanation:"Assessment is diagnostic, not tutoring. Actual need stays fixed."},
    {key:"uncertainty",question:"Estimated need 50, SD 3: must the actual need be between 47 and 53 hours?",options:[["no","No. SD describes spread, not a guaranteed range."],["yes","Yes. It must be in that range."]],correct:"no",explanation:"SD summarizes uncertainty; it is not a guaranteed bound. Possible needs are integers 1–100."},
    {key:"gpa",question:"Which description matches the GPA rule?",options:[["rule","Exactly enough: 3.50. Four hours short: 3.38. Four extra: 3.56."],["symmetric","Exactly enough: 3.50. Four hours short: 3.44. Four extra: 3.56."]],correct:"rule",explanation:"Exactly enough maintains 3.5. Four missing hours lose 0.12; four extra hours gain 0.06 under the supplied rule."},
    {key:"goal",question:"What is your task goal?",options:[["total","Achieve the highest combined final GPA."],["same","Always make the students' GPAs identical."],["questions","Ask as many questions as possible."]],correct:"total",explanation:"Your goal is the highest combined final GPA, not a prescribed division or number of questions."},
    {key:"stop",question:"After confirming stop, can you ask again? Does reading time use tutoring hours?",options:[["no_no","No to both."],["yes_no","You can ask again; reading is free."],["no_yes","You cannot ask again; reading costs tutoring hours."]],correct:"no_no",explanation:"Confirmed stopping is final. Only assessments consume tutoring hours; reading is free."}
  ];
  const PRACTICE_CHECKS = [
    {key:"cost",question:"120 tutoring hours; one assessment costs 4 hours. After three assessments, how many remain?",options:[["108","108 hours"],["116","116 hours"],["120","120 hours"]],correct:"108",explanation:"120 − 3 × 4 = 108 hours remain."},
    {key:"stop",question:"Can you request another assessment after confirming stop?",options:[["no","No."],["yes","Yes."]],correct:"no",explanation:"Confirmed stopping is irreversible in that trial."}
  ];
  const PILOT_CHECKS = [
    {key:"cost",question:"Imagine you have 60 minutes for tutoring. Asking a student to explain a concept takes 3 minutes. After asking twice, how many minutes are left for tutoring?",options:[["54","54 minutes"],["60","60 minutes"],["57","57 minutes"]],correct:"54",explanation:"The two questions use 6 minutes, so54 minutes remain for tutoring."},
    {key:"diagnosis",question:"When you ask a student to explain a concept, does this teach them or reduce the tutoring they need?",options:[["no","No. It helps you learn about their need, but does not teach them."],["yes","Yes. The question counts as teaching."]],correct:"no",explanation:"You listen to the student's answer without teaching. Their actual need stays the same."},
    {key:"uncertainty",question:"The screen says there is about a 90% chance that a student's need is between 47 and 53 minutes. Can their actual need be outside that range?",options:[["no","Yes. The range includes most possibilities, but is not a guarantee."],["yes","No. The need must be inside the range."]],correct:"no",explanation:"Some possibilities lie outside the displayed range. The screen states how much of the current estimate is inside it."},
    {key:"gpa",question:"Compare being short of the needed tutoring by a certain amount with giving the same amount of extra tutoring. Which change is larger?",options:[["rule","The GPA loss from too little tutoring is larger than the gain from the same amount of extra tutoring."],["symmetric","The loss and gain are the same size."]],correct:"rule",explanation:"A shortfall lowers GPA twice as much as the same amount of extra tutoring raises it."},
    {key:"goal",question:"What are you trying to achieve?",options:[["total","The highest combined final GPA for the two students."],["same","Always give both students the same final GPA."],["questions","Ask as many questions as possible."]],correct:"total",explanation:"Try to achieve the highest combined final GPA. You choose whether to ask and how to divide the time."},
    {key:"stop",question:"After you stop asking and move to the allocation screen, can you ask again? Does reading or deciding use the tutoring budget?",options:[["no_no","No to both."],["yes_no","You can ask again; reading does not use the budget."],["no_yes","You cannot ask again; reading uses the budget."]],correct:"no_no",explanation:"Stopping is final for that trial. Only asking a student deducts time from the tutoring budget."}
  ];
  const PILOT_PRACTICE_CHECKS = [
    {key:"cost",question:"Imagine a new trial with 120 minutes for tutoring. Each question takes 4 minutes. If you ask three questions, how many minutes remain?",options:[["108","108 minutes"],["116","116 minutes"],["120","120 minutes"]],correct:"108",explanation:"Three questions use 12 minutes, leaving 108 minutes for tutoring."},
    {key:"stop",question:"In that hypothetical trial, could you ask another question after moving to the allocation screen?",options:[["no","No."],["yes","Yes."]],correct:"no",explanation:"Once you stop asking, you can only divide the time that remains."}
  ];
  function r9InitialQuestions() {
    return PILOT_CHECKS.filter(q=>q.key!=="uncertainty").map(q=>{
      const copy=clone(q);copy.question=copy.question.replace(/\btrial\b/g,"round");
      copy.explanation=copy.explanation.replace(/\btrial\b/g,"round");
      if(copy.key==="cost")copy.explanation=copy.explanation.replace("so54","so 54");
      if(copy.key==="gpa")copy.explanation="For the same number of minutes, too little tutoring lowers GPA more than extra tutoring raises it. Four minutes short gives 3.38; four extra gives 3.56.";
      return copy;
    });
  }
  function r9PracticeQuestions(state) {
    const f=guideFromRecipe(state.resolvedRecipe.guideRecipe),v=f.view,a=v.students[0],b=v.students[1];
    const reading=x=>formatR9Number(x.mean,1)+" minutes; range "+x.interval.lower+"–"+x.interval.upper+" minutes";
    const remaining=formatR9Number(v.remainingMillihours/1000,3),spent=formatR9Number((v.totalMillihours-v.remainingMillihours)/1000,3);
    return [
      {key:"guide_reading",question:"In the example, which current estimate and range are shown for "+a.name+"?",options:[["shown",reading(a)],["other",reading(b)],["exact","The exact tutoring need is already known."]],correct:"shown",explanation:a.name+"'s displayed current estimate and range are "+reading(a)+". They are not an exact revealed need."},
      {key:"guide_range",question:"Could "+a.name+"'s actual tutoring need be outside the displayed example range?",options:[["possible","Yes. The range includes at least 95% of current possibilities, not every possibility."],["guaranteed","No. The actual amount must be inside it."]],correct:"possible",explanation:"This model-based range describes current uncertainty. The example range does not cover every possible amount."},
      {key:"guide_measurement",question:"What does the shared 80% answer-precision statement describe?",options:[["answer","How far a new answer can be from the student's actual need."],["belief","How certain the current estimate is."],["learning","How much the question teaches the student."]],correct:"answer",explanation:"It describes a new imperfect answer for either student, not the current estimate's range. Asking does not teach."},
      {key:"guide_resource",question:"After the example's one question, how is its tutoring budget accounted for?",options:[["accounted",spent+" minutes asking + "+remaining+" minutes available for tutoring."],["unchanged",formatR9Number(v.totalMillihours/1000,3)+" minutes are still available for tutoring."],["teaching","The question time also counts as tutoring."]],correct:"accounted",explanation:"The displayed total equals time asking plus time remaining for tutoring. Reading and deciding use no tutoring time."},
      {key:"guide_stop",question:"After stopping and reaching the slider, could you ask another question in that round?",options:[["no","No."],["yes","Yes."]],correct:"no",explanation:"Stopping is irreversible. Use the slider to divide the remaining tutoring time."}
    ];
  }
  function checkQuestions(state,kind) {
    const stage=kind||state.remediationKind||state.phase,practice=stage==="practiceCheck";
    const questions=profile(state).r9?(practice?r9PracticeQuestions(state):r9InitialQuestions()):profile(state).pilot?(practice?PILOT_PRACTICE_CHECKS:PILOT_CHECKS):(practice?PRACTICE_CHECKS:CHECKS);
    if(!profile(state).pilot)return questions;
    const rows=questions.map(q=>({...q,options:shuffled(q.options,rng(seedKey(state,"check-order:"+stage+":"+q.key+(profile(state).r9?":"+R9_VERSIONS.checkVersion:""))))}));
    // Remove the specific all-first shortcut while retaining stable option identities.
    if(rows.every(q=>q.options[0][0]===q.correct))rows[rows.length-1].options.reverse();
    return rows;
  }
  const PILOT_TEXT_FIELDS=["assessmentStrategy","allocationStrategy","goal","confusing","improvements","bugs","rushing","hypothesis","comments"];
  function validatePosttask(answers) {
    assert(answers&&typeof answers==="object"&&!Array.isArray(answers),"Invalid reflection answers");
    const keys=PILOT_TEXT_FIELDS.concat(["costClarity","uncertaintyClarity","gpaClarity","realism","duration","repetition"]);
    for(const key of Object.keys(answers))assert(keys.includes(key),"Unknown reflection field");
    for(const key of PILOT_TEXT_FIELDS)if(answers[key]!==undefined)assert(typeof answers[key]==="string"&&answers[key].length<=500,"Reflection text must be 500 characters or less");
    return answers;
  }
  function parseAllocation(text) {
    const clean = String(text).trim();
    if (!/^(?:\d{1,3})(?:\.\d)?$/.test(clean)) return null;
    const index = Math.round(Number(clean)*10);
    return index >= 0 && index <= 1000 ? index : null;
  }
  function visibleStudent(t,i) {
    const m=moments(t.logs[i]);
    if(t.definition.profileId===R9)return {name:t.definition.presentation.names[i],subject:t.definition.presentation.subject,
      mean:m.mean,probabilities:m.probabilities,interval:r9CredibleRange(m.probabilities),
      latestReport:t.events.filter(e=>e.type==="assessment"&&e.student===i).map(e=>e.report).slice(-1)[0]||null};
    return {label:i===0?"A":"B",...(t.definition.presentation?{name:t.definition.presentation.names[i],subject:t.definition.presentation.subject}: {}),mean:m.mean,sd:m.sd,probabilities:m.probabilities,reportRMSE:predictiveRMSE(t.logs[i],likelihood(t.definition.reportWidth)),latestReport:t.events.filter(e=>e.type==="assessment"&&e.student===i).map(e=>e.report).slice(-1)[0]||null};
  }
  function trialView(state) {
    if (!state.current) return null;
    const t=state.current,count=t.assessmentCounts[0]+t.assessmentCounts[1],d=t.definition;
    const budgetOnly=profile(state).budgetOnly;
    const view={trialId:d.trialId,stage:d.stage,stageIndex:d.stageIndex,phase:state.phase,totalMillihours:d.budgetMillihours,
      remainingMillihours:t.budgetMillihours,costMillihours:d.costMillihours,assessmentCount:count,maxAssessments:budgetOnly?null:20,
      ...(budgetOnly?{questionLimitRule:"budget_only",questionCapacity:questionCapacity(state,d),presentation:clone(d.presentation)}: {}),
      canAssess:state.phase==="trial"&&(budgetOnly||count<20)&&t.budgetMillihours>=d.costMillihours,
      disabledReason:!budgetOnly&&count>=20?"assessment_cap":t.budgetMillihours<d.costMillihours?"insufficient_budget":null,
      students:[visibleStudent(t,0),visibleStudent(t,1)],draftAllocation:t.draftAllocation,shareIndex:t.shareIndex,
      assessmentReports:t.events.filter(e=>e.type==="assessment").map(e=>({student:e.student,report:e.report,budgetMillihours:e.budgetMillihours}))};
    if (t.result) view.feedback=clone(t.result);
    if(profile(state).r9)Object.assign(view,{profileId:R9,displayVersion:R9_VERSIONS.displayVersion,sharedPrecision:sharedChannelPrecision(d.reportWidth)});
    return view;
  }
  const guideCache=new Map();
  function guideFromRecipe(row) {
    const key=fingerprint(row);
    if(!guideCache.has(key)){
      const definition={profileId:R9,trialId:"example",stage:"example",stageIndex:0,
        templateId:"C3_s1_g0",scale:1,gap:0,sideReversed:false,priorCentres:row.prior_kernel_centres.slice(),
        priorSpreads:row.prior_kernel_spreads.slice(),reportWidth:row.report_kernel_width,
        budgetMillihours:Math.round(row.total_time*1000),costMillihours:Math.round(row.sample_time_cost*1000),
        presentation:{version:"trial-presentation-v1",names:["Alex","Jordan"],subject:"mathematics"}};
      const logs=definition.priorCentres.map((m,i)=>prior(m,definition.priorSpreads[i]));
      logs[0]=posterior(logs[0],50,likelihood(definition.reportWidth));
      const current={definition,logs,budgetMillihours:definition.budgetMillihours-definition.costMillihours,
        assessmentCounts:[1,0],events:[{type:"assessment",student:0,report:50,budgetMillihours:definition.budgetMillihours-definition.costMillihours}],draftAllocation:"",shareIndex:null};
      const semantic={schema:"tutoring-r9-guide-v1",fixtureId:R9_GUIDE,...R9_VERSIONS,view:trialView({profileId:R9,phase:"trial",current})};
      guideCache.set(key,{...semantic,fixtureHash:fingerprint(semantic)});
    }
    return clone(guideCache.get(key));
  }
  function guideExample(design) {validateDesign(design);return guideFromRecipe(design.conditions.find(x=>x.id==="C3"));}
  function enterPracticeGuide(state,action) {
    const attempt=state.checks.filter(x=>x.kind==="practiceCheck").length+1,at=now(action);
    assert(attempt<=3&&state.practiceGuideHistory.length===attempt-1,"Invalid guide attempt progression");
    state.practiceGuideHistory.push({forAttempt:attempt,fixtureId:state.guideFixtureId,fixtureHash:state.guideFixtureHash,
      ...R9_VERSIONS,shownAt:at,acknowledgedAt:null,shownRevision:state.revision+1,acknowledgedRevision:null});
    state.guideShownAt=at;state.guideAcknowledgedAt=null;state.phase="practiceGuide";state.remediationKind="practiceCheck";
  }
  function validateR9Instrumentation(state,design) {
    if(!profile(state).r9)return;
    const meta=r9Metadata(design),fixture=guideExample(design);
    for(const key of Object.keys(meta))assert(stable(state[key])===stable(meta[key]),"Saved R9 version/recipe mismatch: "+key);
    assert(state.guideFixtureId===fixture.fixtureId&&state.guideFixtureHash===fixture.fixtureHash,"Saved guide fixture mismatch");
    assert(Array.isArray(state.checks)&&Array.isArray(state.practiceGuideHistory)&&state.practiceGuideHistory.length<=3,"Invalid R9 gate history");
    const initial=state.checks.filter(x=>x.kind==="comprehension"),practice=state.checks.filter(x=>x.kind==="practiceCheck");
    assert(initial.length+practice.length===state.checks.length&&initial.length<=3&&practice.length<=3,"Invalid understanding sets");
    assert(stable(state.checks)===stable(initial.concat(practice))&&state.practiceGuideHistory.length>=practice.length&&state.practiceGuideHistory.length<=practice.length+1,"Invalid understanding/guide sequence");
    let previousRevision=-1,previousAt=-Infinity;
    for(const kind of ["comprehension","practiceCheck"]){
      const qs=checkQuestions(state,kind),attempts=kind==="comprehension"?initial:practice;
      attempts.forEach((c,index)=>{
        assert(c.attempt===index+1&&Array.isArray(c.answers)&&c.answers.length===qs.length&&c.answers.every((x,i)=>qs[i].options.some(o=>o[0]===x)),"Invalid R9 check responses");
        const missed=qs.map((q,i)=>c.answers[i]!==q.correct?i:null).filter(i=>i!==null);
        assert(stable(c.missed)===stable(missed)&&stable(c.questionKeys)===stable(qs.map(q=>q.key))&&stable(c.missedQuestionKeys)===stable(missed.map(i=>qs[i].key)),"R9 check semantic mismatch");
        assert(stable(c.presentedOptionIds)===stable(qs.map(q=>q.options.map(o=>o[0])))&&c.checkVersion===R9_VERSIONS.checkVersion,"R9 check option/version mismatch");
        assert(Number.isInteger(c.attemptRevision)&&c.attemptRevision>previousRevision&&c.attemptRevision<=state.revision&&Number.isFinite(Date.parse(c.at))&&Date.parse(c.at)>=previousAt,"Invalid R9 attempt ordering");
        assert(index===attempts.length-1||missed.length>0,"Attempt after a passed understanding set");
        if(kind==="practiceCheck"){
          const g=state.practiceGuideHistory[index];assert(c.guideHistoryIndex===index&&g&&g.acknowledgedAt!==null&&g.acknowledgedRevision<c.attemptRevision&&Date.parse(g.acknowledgedAt)<=Date.parse(c.at),"Check bypassed guide acknowledgment");
        }
        previousRevision=c.attemptRevision;previousAt=Date.parse(c.at);
      });
    }
    state.practiceGuideHistory.forEach((g,index)=>{
      assert(g.forAttempt===index+1&&g.fixtureId===fixture.fixtureId&&g.fixtureHash===fixture.fixtureHash,"Guide attempt/fixture mismatch");
      for(const key of Object.keys(R9_VERSIONS))assert(g[key]===R9_VERSIONS[key],"Guide version mismatch");
      assert(Number.isInteger(g.shownRevision)&&g.shownRevision>0&&g.shownRevision<=state.revision&&Number.isFinite(Date.parse(g.shownAt)),"Invalid guide shown witness");
      if(index===0)assert(initial.length&&g.shownRevision>initial[initial.length-1].attemptRevision&&Date.parse(g.shownAt)>=Date.parse(initial[initial.length-1].at),"Guide preceded initial understanding pass");
      if(index>0)assert(practice[index-1]&&practice[index-1].missed.length>0&&g.shownRevision===practice[index-1].attemptRevision&&Date.parse(g.shownAt)>=Date.parse(practice[index-1].at),"Guide remediation order mismatch");
      if(g.acknowledgedAt===null)assert(g.acknowledgedRevision===null&&index===state.practiceGuideHistory.length-1&&state.phase==="practiceGuide","Invalid pending guide witness");
      else assert(Number.isFinite(Date.parse(g.acknowledgedAt))&&Date.parse(g.shownAt)<=Date.parse(g.acknowledgedAt)&&Number.isInteger(g.acknowledgedRevision)&&g.acknowledgedRevision>g.shownRevision&&g.acknowledgedRevision<=state.revision,"Invalid guide acknowledgment witness");
    });
    const passed=xs=>xs.length>0&&xs[xs.length-1].missed.length===0;
    assert(["instructions","instructionsReview","comprehension","ineligible","practiceGuide","practiceCheck","trial","stopConfirm","allocation","feedback","pendingCommit","break","posttask","complete"].includes(state.phase),"Invalid R9 saved phase");
    if(["instructions","instructionsReview","comprehension"].includes(state.phase)){
      assert(state.currentTrialIndex===0&&state.current===null&&state.completedTrials.length===0&&state.archivedTrialCount===0&&practice.length===0&&!passed(initial),"Invalid initial-set progression");
      if(state.phase==="instructions")assert(initial.length===0,"Instructions after initial attempt");
      else assert(initial.length<3,"Initial attempt ceiling exceeded");
      if(state.phase==="instructionsReview")assert(initial.length>0&&state.remediationKind==="comprehension","Missing initial remediation");
    }
    if(state.phase==="ineligible")assert(state.current===null&&((state.currentTrialIndex===0&&initial.length===3&&!passed(initial)&&practice.length===0)||(state.currentTrialIndex===4&&passed(initial)&&practice.length===3&&!passed(practice))),"Ineligibility requires three failed attempts at a set");
    if(practice.length)assert(state.currentTrialIndex>=4,"Post-practice attempts before practice");
    if(state.current!==null||state.currentTrialIndex>0||state.practiceGuideHistory.length||state.archivedTrialCount>0||state.completedTrials.length>0||["trial","stopConfirm","allocation","feedback","pendingCommit","break","posttask","complete"].includes(state.phase))assert(passed(initial),"Initial understanding set not passed");
    if(state.currentTrialIndex>=4)assert(state.practiceGuideHistory.length>0,"Post-practice guide missing");
    if(state.phase==="practiceGuide"||state.phase==="practiceCheck"){
      assert(state.currentTrialIndex===4&&state.current===null&&!passed(practice)&&state.practiceGuideHistory.length===practice.length+1,"Invalid post-practice progression");
      assert((state.practiceGuideHistory[state.practiceGuideHistory.length-1].acknowledgedAt===null)===(state.phase==="practiceGuide"),"Guide/check phase mismatch");
    }else{
      assert(state.practiceGuideHistory.length===practice.length,"Unexpected extra guide witness");
      if(state.currentTrialIndex>=4&&state.phase!=="ineligible")assert(passed(practice),"Main round before post-practice pass");
    }
    if(state.practiceGuideHistory.length){const last=state.practiceGuideHistory[state.practiceGuideHistory.length-1];assert(state.guideShownAt===last.shownAt&&state.guideAcknowledgedAt===last.acknowledgedAt,"Latest guide witness mismatch");}
    else assert(state.guideShownAt===null&&state.guideAcknowledgedAt===null,"Unexpected guide witness");
  }
  function validateState(state,design,options) {
    assert(state&&state.schema===SCHEMA&&state.engineVersion===VERSION,"Unsupported saved session version");
    assert(state.designFingerprint===sessionFingerprint(design,state),"Saved session belongs to a different design");
    if (options) {assert(profile(state).id===profile(options).id,"Saved session profile mismatch");for (const k of ["conditionId","participantId","sessionId","seed","scheduleVariant"]) if (options[k]!==undefined) assert(state[k]===options[k],"Saved session identity/condition mismatch");}
    assert(Number.isInteger(state.revision)&&state.revision>=0&&Number.isInteger(state.currentTrialIndex)&&state.currentTrialIndex>=0&&state.currentTrialIndex<=profile(state).totalTrials,"Invalid saved progression");
    assert(stable(state.schedule)===stable(makeSchedule(design,state)),"Saved schedule mismatch");
    if(profile(state).budgetOnly)assert(state.questionLimitRule==="budget_only"&&state.maximumOnlineAssessments===null&&state.presentationVersion==="trial-presentation-v1"&&state.feedbackStyle===(profile(state).r9?R9_VERSIONS.feedbackStyle:"outcome-engagement-v1"),"Saved profile rules mismatch");
    validateR9Instrumentation(state,design);
    if(state.current){
      const t=state.current; assert(profile(state).budgetOnly?stable(t.definition)===stable(state.schedule[state.currentTrialIndex]):t.definition.trialId===state.schedule[state.currentTrialIndex].trialId,"Trial identity mismatch");
      assert(stable({...scenario(state,t.definition)})===stable({trueNeeds:t.trueNeeds,potentialReports:t.potentialReports,logicalToDisplayed:t.logicalToDisplayed}),"Saved scenario mismatch");
      const count=t.assessmentCounts.reduce((s,n)=>s+n,0);
      assert(t.assessmentCounts.every(x=>Number.isInteger(x)&&x>=0)&&count<=questionCapacity(state,t.definition),"Invalid assessment counts");
      assert(t.budgetMillihours===t.definition.budgetMillihours-count*t.definition.costMillihours&&t.budgetMillihours>=0,"Saved budget mismatch");
      if(profile(state).budgetOnly){
        const counts=[0,0];let budget=t.definition.budgetMillihours;
        for(const e of t.events.filter(x=>x.type==="assessment")){
          assert((e.student===0||e.student===1)&&e.reportIndex===counts[e.student]&&e.report===t.potentialReports[e.student][counts[e.student]]&&e.budgetBeforeMillihours===budget,"Saved question sequence mismatch");
          counts[e.student]++;budget-=t.definition.costMillihours;
          assert(e.budgetMillihours===budget&&stable(e.assessmentCounts)===stable(counts),"Saved question accounting mismatch");
        }
        assert(stable(counts)===stable(t.assessmentCounts),"Saved question counts mismatch");
      }
      t.logs.forEach(normalize);
    }
    return state;
  }
  function transition(previous,action,design) {
    assert(action&&typeof action.id==="string"&&action.id.length>0,"Action identity required");
    if(previous.acceptedActionIds.includes(action.id))return previous;
    assert(action.expectedRevision===previous.revision,"Stale action; reload the saved state");
    assert(previous.designFingerprint===sessionFingerprint(design,previous),"Design changed during session");
    validateR9Instrumentation(previous,design);
    const s=clone(previous),t=s.current;
    const requirePhase=(...phases)=>assert(phases.includes(s.phase),"Action unavailable in "+s.phase);
    switch(action.type){
      case "beginChecks": requirePhase("instructions","instructionsReview");s.phase=s.remediationKind||"comprehension";delete s.remediationKind;break;
      case "beginPracticeChecks": {
        assert(profile(s).r9,"R9 guide action is unavailable");requirePhase("practiceGuide");
        const g=s.practiceGuideHistory[s.practiceGuideHistory.length-1];
        assert(action.fixtureId===s.guideFixtureId&&action.fixtureHash===s.guideFixtureHash&&g&&g.acknowledgedAt===null,"Guide identity/acknowledgment mismatch");
        g.acknowledgedAt=now(action);g.acknowledgedRevision=s.revision+1;s.guideAcknowledgedAt=g.acknowledgedAt;
        s.phase="practiceCheck";delete s.remediationKind;break;
      }
      case "submitChecks": {
        requirePhase("comprehension","practiceCheck"); const kind=s.phase,questions=checkQuestions(s,kind);
        assert(Array.isArray(action.answers)&&action.answers.length===questions.length,"Answer every check");
        if(profile(s).r9)assert(action.answers.every((x,i)=>questions[i].options.some(o=>o[0]===x)),"Unknown understanding option");
        const missed=questions.map((q,i)=>action.answers[i]!==q.correct?i:null).filter(x=>x!==null);
        const attempt=s.checks.filter(x=>x.kind===kind).length+1;
        s.checks.push({kind,attempt,answers:action.answers.slice(),missed,at:now(action),...(profile(s).pilot?{presentedOptionIds:questions.map(q=>q.options.map(o=>o[0]))}:{}),
          ...(profile(s).r9?{checkVersion:R9_VERSIONS.checkVersion,questionKeys:questions.map(q=>q.key),missedQuestionKeys:missed.map(i=>questions[i].key),attemptRevision:s.revision+1,...(kind==="practiceCheck"?{guideHistoryIndex:attempt-1}: {})}: {})});
        if(!missed.length)openTrial(s,action); else if(attempt>=profile(s).maxCheckAttempts)s.phase="ineligible";
        else if(profile(s).r9&&kind==="practiceCheck")enterPracticeGuide(s,action);
        else if(profile(s).pilot){s.remediationKind=kind;s.phase="instructionsReview";}
        break;
      }
      case "assess": {
        requirePhase("trial");assert(action.student===0||action.student===1,"Student must be A or B");
        const count=t.assessmentCounts[0]+t.assessmentCounts[1];
        assert((profile(s).budgetOnly||count<20)&&t.budgetMillihours>=t.definition.costMillihours,"Another assessment is not available");
        const before=clone(t.logs),budgetBefore=t.budgetMillihours,i=action.student,index=t.assessmentCounts[i];
        const report=t.potentialReports[i][index];t.logs[i]=posterior(t.logs[i],report,likelihood(t.definition.reportWidth));
        t.assessmentCounts[i]++;t.budgetMillihours-=t.definition.costMillihours;
        event(s,"assessment",action,{student:i,report,reportIndex:index,budgetBeforeMillihours:budgetBefore,logsBefore:before,logsAfter:clone(t.logs)});
        t.events[t.events.length-1].displayAfter=[visibleStudent(t,0),visibleStudent(t,1)];
        if(t.budgetMillihours===0){t.stopReason="zero_budget";t.shareIndex=null;s.phase="allocation";}
        break;
      }
      case "openStop": requirePhase("trial");s.phase="stopConfirm";event(s,"stop_open",action);if(profile(s).pilot&&t.definition.stage==="main"){s.phase="allocation";t.stopReason=!profile(s).budgetOnly&&t.assessmentCounts.reduce((a,b)=>a+b,0)>=20?"assessment_cap":t.budgetMillihours<t.definition.costMillihours?"insufficient_budget":"voluntary";event(s,"stop_confirm",action,{directMainStop:true});}break;
      case "cancelStop": requirePhase("stopConfirm");s.phase="trial";event(s,"stop_cancel",action);break;
      case "confirmStop": requirePhase("stopConfirm");s.phase="allocation";t.stopReason=!profile(s).budgetOnly&&t.assessmentCounts.reduce((a,b)=>a+b,0)>=20?"assessment_cap":t.budgetMillihours<t.definition.costMillihours?"insufficient_budget":"voluntary";event(s,"stop_confirm",action);break;
      case "editAllocation": requirePhase("allocation");assert(String(action.value).length<=12,"Allocation entry is too long");t.draftAllocation=String(action.value);t.shareIndex=parseAllocation(t.draftAllocation);event(s,"allocation_edit",action,{rawValue:t.draftAllocation,shareIndex:t.shareIndex});break;
      case "editAllocationIndex":requirePhase("allocation");assert(profile(s).pilot&&Number.isInteger(action.shareIndex)&&action.shareIndex>=0&&action.shareIndex<=1000,"Select a valid allocation");t.shareIndex=action.shareIndex;t.draftAllocation=String(action.shareIndex/10);event(s,"allocation_edit",action,{rawValue:t.draftAllocation,shareIndex:t.shareIndex,control:"resource_slider"});break;
      case "confirmAllocation": {
        requirePhase("allocation"); const j=t.budgetMillihours===0?null:t.shareIndex;
        const value=outcome(t.budgetMillihours,j,t.trueNeeds);event(s,"allocation_confirm",action,{shareIndex:j});
        t.result={...value,trueNeeds:t.trueNeeds.slice(),shareIndex:j,remainingMillihours:t.budgetMillihours,
          assessmentCounts:t.assessmentCounts.slice(),assessmentTimeMillihours:t.definition.budgetMillihours-t.budgetMillihours,stopReason:t.stopReason,confirmedAt:now(action)};
        s.phase="feedback";break;
      }
      case "advance": {
        requirePhase("feedback");event(s,"trial_complete",action);
        let record={trialId:t.definition.trialId,conditionId:s.conditionId,participantId:s.participantId,definition:clone(t.definition),
          initialLogs:clone(t.initialLogs),terminalLogs:clone(t.logs),trueNeeds:t.trueNeeds.slice(),potentialReports:clone(t.potentialReports),logicalToDisplayed:t.logicalToDisplayed,
          events:clone(t.events),result:clone(t.result),status:"complete",engineVersion:VERSION,designFingerprint:s.designFingerprint,seedNamespaceVersion:profile(s).pilot?PILOT:VERSION,...(profile(s).pilot?{profileId:profile(s).id}: {}),...(profile(s).budgetOnly?{questionLimitRule:"budget_only",questionCapacity:questionCapacity(s,t.definition)}: {})};
        if(profile(s).r9)Object.assign(record,r9Metadata(design),{guideFixtureId:s.guideFixtureId,guideFixtureHash:s.guideFixtureHash});
        if(profile(s).budgetOnly)record=compactTrial(record);
        s.completedTrials.push(record);s.pendingCommit={trialId:record.trialId,trialIndex:s.currentTrialIndex,record};s.phase="pendingCommit";break;
      }
      case "acknowledgeAdvance":
        requirePhase("pendingCommit");assert(s.pendingCommit&&s.pendingCommit.trialIndex===s.currentTrialIndex,"No matching pending trial");
        s.currentTrialIndex++;s.current=null;s.pendingCommit=null;
        if(s.currentTrialIndex===4){if(profile(s).r9)enterPracticeGuide(s,action);else s.phase="practiceCheck";}
        else if(s.currentTrialIndex===profile(s).totalTrials)s.phase="posttask";
        else if(profile(s).pilot?s.currentTrialIndex===9:s.currentTrialIndex===14||s.currentTrialIndex===24)s.phase="break";
        else openTrial(s,action);break;
      case "continueBreak":requirePhase("break");openTrial(s,action);break;
      case "editPosttask":requirePhase("posttask");s.posttaskDraft=clone(profile(s).pilot?validatePosttask(action.answers||{}):action.answers||{});break;
      case "complete":
        requirePhase("posttask");s.posttaskDraft=clone(profile(s).pilot?validatePosttask(action.answers||s.posttaskDraft):action.answers||s.posttaskDraft);
        for(const key of ["costClarity","uncertaintyClarity","gpaClarity","realism"].concat(profile(s).pilot?["duration","repetition"]:[]))assert(Number.isInteger(Number(s.posttaskDraft[key]))&&Number(s.posttaskDraft[key])>=1&&Number(s.posttaskDraft[key])<=7,"Select every1–7rating");
        s.phase="complete";s.completedAt=now(action);break;
      case "resume":audit(s,"resume",{},action);break;
      case "focus":audit(s,"focus",{focused:!!action.focused},action);break;
      default:throw new Error("Unknown action: "+action.type);
    }
    s.revision++;s.acceptedActionIds.push(action.id);validateR9Instrumentation(s,design);return s;
  }
  function compactEvents(events,tuples=false){
    return events.map(e=>{const x=clone(e);if(x.type==="assessment"){delete x.logsBefore;delete x.logsAfter;delete x.displayAfter;if(tuples)return [x.eventIndex,x.id,x.at,x.budgetMillihours,...x.assessmentCounts,x.student,x.report,x.reportIndex,x.budgetBeforeMillihours];}return x;});
  }
  function expandedEvent(raw){
    if(!Array.isArray(raw))return clone(raw);
    assert(raw.length===10,"Invalid compact assessment tuple");
    const [eventIndex,id,at,budgetMillihours,a,b,student,report,reportIndex,budgetBeforeMillihours]=raw;
    assert([eventIndex,budgetMillihours,a,b,student,report,reportIndex,budgetBeforeMillihours].every(Number.isSafeInteger)&&typeof id==="string"&&id.length>0&&typeof at==="string"&&Number.isFinite(Date.parse(at))&&(student===0||student===1)&&report>=1&&report<=100,"Invalid compact assessment fields");
    return {eventIndex,id,type:"assessment",at,budgetMillihours,assessmentCounts:[a,b],student,report,reportIndex,budgetBeforeMillihours};
  }
  function reconstruct(definition,events){
    const mapping=definition.sideReversed?[1,0]:[0,1];
    const initialLogs=mapping.map(i=>prior(definition.priorCentres[i],definition.priorSpreads[i]));
    const t={definition,logs:clone(initialLogs),events:[]};
    for(const raw of events){
      const e=expandedEvent(raw);
      if(e.type==="assessment"){
        e.logsBefore=clone(t.logs);t.logs[e.student]=posterior(t.logs[e.student],e.report,likelihood(definition.reportWidth));e.logsAfter=clone(t.logs);
        t.events.push(e);e.displayAfter=[visibleStudent(t,0),visibleStudent(t,1)];
      }else t.events.push(e);
    }
    return {initialLogs,terminalLogs:t.logs,events:t.events};
  }
  function compactTrial(record){
    if(record.compactSchema==="tutoring-trial-replay-1"||record.compactSchema==="tutoring-trial-replay-2")return clone(record);
    const r=clone(record),tuples=record.profileId===ENGAGEMENT||record.profileId===R9;delete r.initialLogs;delete r.terminalLogs;r.events=compactEvents(r.events,tuples);r.compactSchema=tuples?"tutoring-trial-replay-2":"tutoring-trial-replay-1";return r;
  }
  function expandTrial(record){
    const r=clone(record);if(!r.compactSchema)return r;
    assert(r.compactSchema==="tutoring-trial-replay-1"||r.compactSchema==="tutoring-trial-replay-2","Unsupported compact trial version");
    const replay=reconstruct(r.definition,r.events);Object.assign(r,replay);delete r.compactSchema;return r;
  }
  function compactCurrent(current,tuples=false){
    if(!current)return null;const t=clone(current);delete t.initialLogs;delete t.logs;t.events=compactEvents(t.events,tuples);return t;
  }
  function serialize(state){
    const payload=clone(state);payload.compactSchema="tutoring-session-replay-1";
    payload.completedTrials=payload.completedTrials.map(compactTrial);payload.current=compactCurrent(payload.current,profile(state).budgetOnly);
    if(payload.pendingCommit)payload.pendingCommit.record=compactTrial(payload.pendingCommit.record);
    return JSON.stringify({schema:SCHEMA,payload,checksum:fingerprint(payload)});
  }
  function restore(text,design,options){
    const envelope=typeof text==="string"?JSON.parse(text):text;
    assert(envelope.schema===SCHEMA&&envelope.checksum===fingerprint(envelope.payload),"Saved data is corrupt or incomplete; do not reset it");
    const s=clone(envelope.payload);
    if(s.compactSchema){
      assert(s.compactSchema==="tutoring-session-replay-1","Unsupported compact session version");
      if(!profile(s).budgetOnly)s.completedTrials=s.completedTrials.map(expandTrial);
      if(s.pendingCommit&&!profile(s).budgetOnly)s.pendingCommit.record=expandTrial(s.pendingCommit.record);
      if(s.current){const replay=reconstruct(s.current.definition,s.current.events);s.current.initialLogs=replay.initialLogs;s.current.logs=replay.terminalLogs;s.current.events=replay.events;}
      delete s.compactSchema;
    }
    return validateState(s,design,options);
  }
  function checkpoint(state){
    const c=clone(state);c.archivedTrialCount+=c.completedTrials.length;c.completedTrials=[];c.historyStoredExternally=c.archivedTrialCount>0;
    c.archivedAuditCount=(c.archivedAuditCount||0)+c.audit.length;c.audit=[];
    const currentIds=c.current?c.current.events.map(x=>x.id):[];
    c.acceptedActionIds=c.acceptedActionIds.filter(id=>currentIds.includes(id)||id===c.acceptedActionIds[c.acceptedActionIds.length-1]);
    return serialize(c);
  }
  function exportData(state){
    return {schema:"tutoring-task-export-1",engineVersion:VERSION,participantId:state.participantId,sessionId:state.sessionId,conditionId:state.conditionId,
      ...(profile(state).pilot?{profileId:profile(state).id,scheduleVariant:state.scheduleVariant,displayTimeMapping:"1canonical model hour =1scenario minute",
        allocationEditLogging:"First and last allocation edits and total editCount retained; intermediate slider trajectory omitted. Each accepted draft is saved durably."}:{}),
      ...(profile(state).budgetOnly?{questionLimitRule:"budget_only",maximumOnlineAssessments:null,presentationVersion:state.presentationVersion,feedbackStyle:state.feedbackStyle,historyEncoding:"tutoring-trial-replay-2; lossless paid-assessment tuples",scientificScope:"Object-level rules and first20 scientific inputs preserved; historical capped RR predictions/calibration are not automatically extended to this profile."}: {}),
      ...(profile(state).r9?{...R9_VERSIONS,revision:state.revision,currentTrialIndex:state.currentTrialIndex,resolvedRecipe:clone(state.resolvedRecipe),resolvedRecipeFingerprint:state.resolvedRecipeFingerprint,
        guideFixtureId:state.guideFixtureId,guideFixtureHash:state.guideFixtureHash,guideShownAt:state.guideShownAt,
        guideAcknowledgedAt:state.guideAcknowledgedAt,practiceGuideHistory:clone(state.practiceGuideHistory),
        scientificScope:"R9 unscaled total-centre-gap1 C1 is new; C2–C5 scientific RNG namespace/recipes preserved. Budget-only R9 inference is not released."}: {}),
      seed:state.seed,designFingerprint:state.designFingerprint,fingerprintAlgorithm:state.fingerprintAlgorithm,
      phase:state.phase,status:state.phase==="complete"?"complete":state.phase==="ineligible"?"mechanism_checks_not_passed":"incomplete",
      startedAt:state.startedAt,completedAt:state.completedAt,completedTrialCount:state.archivedTrialCount+state.completedTrials.length,
      historyStoredExternally:state.historyStoredExternally,archivedTrialCount:state.archivedTrialCount,
      exportIsFullHistory:!state.historyStoredExternally,schedule:clone(state.schedule),checks:clone(state.checks),trials:clone(state.completedTrials),
      currentTrial:clone(state.current),pendingCommit:clone(state.pendingCommit),audit:clone(state.audit),posttask:clone(state.posttaskDraft),
      researchUse:"Local prototype; not consent, participant approval or an anti-cheating boundary"};
  }
  return {VERSION,SCHEMA,prior,likelihood,posterior,moments,predictiveRMSE,outcome,normalize,fingerprint,
    validateDesign,makeSchedule,createSession,transition,trialView,serialize,restore,checkpoint,exportData,compactTrial,expandTrial,parseAllocation,CHECKS,PRACTICE_CHECKS,PILOT,ENGAGEMENT,R9,R9_VERSIONS,R9_GUIDE,profile,questionCapacity,centralRange,reportAccuracy,checkQuestions,PILOT_TEXT_FIELDS,
    resolveProfile,sessionFingerprint,r9Metadata,formatR9Number,r9CredibleRange,sharedChannelPrecision,guideExample,validateR9Instrumentation};
}));
