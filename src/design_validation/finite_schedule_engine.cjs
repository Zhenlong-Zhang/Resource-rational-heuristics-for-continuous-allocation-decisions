/* Pure ENGAGEMENT schedule/Bayes/payoff closure; no session or browser runtime. */
'use strict';
const VERSION = "tutoring-task-0.1.0";
const PILOT = "pilot-4p10m-v1";
const ENGAGEMENT = "pilot-engagement-budget-v1";
function profile(options) {
  const id=options && options.profileId || "legacy-4p30m-v1";
  assert(id===PILOT||id===ENGAGEMENT||id==="legacy-4p30m-v1","Unsupported task profile");
  const pilot=id===PILOT||id===ENGAGEMENT;
  return {id,pilot,practiceTrials:4,mainTrials:pilot?10:30,totalTrials:pilot?14:34,maxCheckAttempts:pilot?3:2,...(id===ENGAGEMENT?{budgetOnly:true}: {})};
}
function questionCapacity(state,definition=state.current.definition) {
  assert(Number.isSafeInteger(definition.budgetMillihours)&&definition.budgetMillihours>=0&&Number.isSafeInteger(definition.costMillihours)&&definition.costMillihours>0,"Question cost must be a positive integer");
  return profile(state).budgetOnly?Math.floor(definition.budgetMillihours/definition.costMillihours):20;
}
function assert(ok, message) { if (!ok) throw new Error(message); }
function stable(x) {
  if (Array.isArray(x)) return "[" + x.map(stable).join(",") + "]";
  if (x && typeof x === "object") return "{" + Object.keys(x).sort().map(k => JSON.stringify(k) + ":" + stable(x[k])).join(",") + "}";
  return JSON.stringify(x);
}
// cyrb128 is a reproducible non-cryptographic seed, not authentication.
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
  assert(Number.isInteger(report) && report >= 1 && report <= 100, "Report outside 1-100");
  return normalize(logPrior.map((x, n) => x + logLikelihood[n][report - 1]));
}
function moments(logs) {
  const probabilities = normalize(logs).map(Math.exp);
  const mean = probabilities.reduce((s, p, i) => s + p * (i + 1), 0);
  const sd = Math.sqrt(probabilities.reduce((s, p, i) => s + p * (i + 1 - mean) ** 2, 0));
  return {mean, sd, probabilities};
}
function predictiveRMSE(logs, logLikelihood) {
  const p = normalize(logs).map(Math.exp);
  let square = 0;
  for (let n = 0; n < 100; n++) for (let y = 0; y < 100; y++) square += p[n] * Math.exp(logLikelihood[n][y]) * (y - n) ** 2;
  return Math.sqrt(square);
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
  // Display and feedback flags are not inputs to this scientific closure.
  assert(c.variation.trials_per_scale === 10 && c.variation.maximum_consecutive_same_scale === 3 && c.variation.human_trials_selected_by_target_rule_performance === false, "Unsupported order/selection");
  const expected = [ [[50,50],[3,3],30,120,1.2], [[45,75],[2,2],6,150,1.2], [[50,70],[18,18],6,170,.5], [[45,65],[2,2],6,55,.6], [[55,55],[8,8],3,55,.3] ];
  assert(design.conditions.length === 5, "Five conditions required");
  design.conditions.forEach((row, i) => {
    assert(row.id === "C" + (i + 1), "Unexpected condition order");
    assert(stable([row.prior_kernel_centres,row.prior_kernel_spreads,row.report_kernel_width,row.total_time,row.sample_time_cost]) === stable(expected[i]), "Only the selected scientific recipes are supported");
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
  assert(row, "An external C1-C5 assignment is required");
  const cfg=profile(options),variant=cfg.pilot?(options.scheduleVariant===undefined?hashWords(stable(seedKey(options,"pilot-variant")))[0]%9:options.scheduleVariant):null;
  if(cfg.pilot)assert(Number.isInteger(variant)&&variant>=0&&variant<=8,"Pilot schedule variant must be0-8");
  const scales = design.common.variation.scale_factors, cells = design.common.variation.gap_scale_cell_counts_for_three_gap_conditions;
  const items = [];
  scales.forEach((scale, si) => row.mean_gap_offsets.forEach((gap, gi) => {
    const count = cfg.pilot?(row.mean_gap_offsets.length===1?3+(si===variant%3?1:0):1+(si===Math.floor(variant/3)&&gi===variant%3?1:0)):(row.mean_gap_offsets.length === 1 ? 10 : cells[si][gi]);
    for (let k = 0; k < count; k++) items.push({scale, gap, templateId: row.id + "_s" + si + "_g" + gi});
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
  const practice = Array.from({length:4}, (_, i) => ({scale:1,gap:0,templateId:row.id + "_practice_central",sideReversed:practiceSides[i],stage:"practice",stageIndex:i+1}));
  const main = ordered.map((x, i) => Object.assign({}, x, {sideReversed:sides[i],stage:"main",stageIndex:i+1}));
  const labels=cfg.budgetOnly?presentations(options):null;
  return practice.concat(main).map((x, i) => {
    const m = row.prior_kernel_centres;
    return Object.assign(x, {scheduleIndex:i,trialId:x.stage + "-" + String(x.stageIndex).padStart(2,"0"),
      priorCentres:[(m[0]+x.gap)*x.scale,(m[1]-x.gap)*x.scale], priorSpreads:row.prior_kernel_spreads.map(v=>v*x.scale),
      reportWidth:row.report_kernel_width*x.scale,budgetMillihours:Math.round(row.total_time*x.scale*1000),costMillihours:Math.round(row.sample_time_cost*x.scale*1000),...(labels?{presentation:labels[i]}: {})});
  });
}
module.exports = {ENGAGEMENT, validateDesign, makeSchedule, questionCapacity,
                  prior, posterior, likelihood, moments, predictiveRMSE, outcome};
