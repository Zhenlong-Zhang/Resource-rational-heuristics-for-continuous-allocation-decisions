/* Actual R9 scientific recipes and R10 teaching transitions; finite software panel.
 * No theory policy, calibration draws, human observations or UI changes.
 * Output rows use displayed A/B ordering.
 */
'use strict';

const fs = require('node:fs');
const assert = require('node:assert/strict');
const E = require('./panel_engine.cjs');
const N = require('./panel_normalizer.cjs');

function orderedPriors(definition) {
  const logical = definition.priorCentres.map((m, i) => E.prior(m, definition.priorSpreads[i]));
  return (definition.sideReversed ? [1, 0] : [0, 1]).map(i => logical[i]);
}

function verifyMapping(definition, logs) {
  assert.deepEqual(logs, orderedPriors(definition), 'Initial finite priors do not match displayed A/B mapping');
}

function oneSession(design, condition, variant, seed) {
  const participant = `candidate-panel-${condition}-v${variant}`;
  const options = {profileId:E.R9, participantFlowVersion:E.R10_FLOW,
    participantId:participant, sessionId:participant+'-session', conditionId:condition,
    scheduleVariant:variant, seed:String(seed), startedAt:'2026-10-19T00:00:00.000Z'};
  let state = E.createSession(design, options);
  let counter = 0;
  const mapping = {actualMainTrials:0, normal:0, reversed:0, asymmetricWrongMappingRejected:0,
    symmetricSwapInvariant:0};
  const act = (type, fields={}) => {
    state = E.transition(state, {id:`candidate-${counter}`, expectedRevision:state.revision,
      at:new Date(Date.parse(options.startedAt)+100*counter++).toISOString(), type, ...fields}, design);
  };
  act('beginChecks');
  for (let step=0; step<4; step++) act('advanceDisplayGuide', {
    fixtureId:state.guideFixtureId, fixtureHash:state.guideFixtureHash, step});
  act('submitChecks', {answers:E.checkQuestions(state).map(q=>q.correct)});
  while (state.phase !== 'posttask') {
    if (state.phase === 'practiceGuide') {
      act('beginPracticeChecks', {fixtureId:state.guideFixtureId, fixtureHash:state.guideFixtureHash});
      continue;
    }
    if (state.phase === 'practiceCheck') {
      act('submitChecks', {answers:E.checkQuestions(state).map(q=>q.correct)});
      continue;
    }
    if (state.phase === 'break') {act('continueBreak'); continue;}
    assert.equal(state.phase, 'trial', 'Unexpected genuine-transition phase');
    const definition = state.current.definition;
    verifyMapping(definition, state.current.initialLogs);
    if (definition.stage === 'main') {
      mapping.actualMainTrials++;
      mapping[definition.sideReversed ? 'reversed' : 'normal']++;
      const reversed = state.current.initialLogs.slice().reverse();
      if (JSON.stringify(reversed) === JSON.stringify(state.current.initialLogs)) {
        verifyMapping(definition, reversed);
        mapping.symmetricSwapInvariant++;
      } else {
        assert.throws(()=>verifyMapping(definition, reversed), /displayed A\/B/);
        mapping.asymmetricWrongMappingRejected++;
      }
    }
    // These are declared scripted choices, not theoretical/human predictions.
    // No search is needed to obtain the actual initial design.
    const j = 250 + ((definition.stageIndex*37 + variant*13) % 501);
    act('openStop');
    if (state.phase === 'stopConfirm') act('confirmStop');
    act('editAllocationIndex', {shareIndex:j});
    act('confirmAllocation'); act('advance'); act('acknowledgeAdvance');
  }
  const answers = Object.fromEntries(E.PILOT_TEXT_FIELDS.map(key=>[key, 'scripted software fixture']));
  for (const key of ['costClarity','uncertaintyClarity','gpaClarity','realism','duration','repetition']) answers[key] = '4';
  act('complete', {answers});
  const exported = E.exportData(state);
  const normalized = N.normalize(exported, design);
  assert.deepEqual(normalized.errors, [], 'Source panel must pass the actual reader');
  assert.equal(normalized.meta.status, 'complete');
  const main = normalized.trials.filter(t=>t.definition.stage==='main');
  assert.equal(main.length, 10);
  const rows = main.map(t=>{
    const d=t.definition, r=t.result;
    verifyMapping(d, t.initialLogs);
    const initial=t.initialLogs.map(logs=>E.moments(logs).mean);
    const terminal=t.terminalLogs.map(logs=>E.moments(logs).mean);
    return {participant:t.participantId, session:t.sessionId, condition:t.conditionId,
      profileId:t.profileId, flowVersion:normalized.meta.participantFlowVersion,
      mainIndex:d.stageIndex, template:d.templateId, scale:d.scale, gap:d.gap,
      sideReversed:d.sideReversed, meanAInitial:initial[0], meanBInitial:initial[1],
      meanATerminal:terminal[0], meanBTerminal:terminal[1], shareIndex:r.shareIndex,
      share:r.remainingMillihours===0 ? null : r.shareIndex/1000,
      initialBudgetMs:d.budgetMillihours, remainingBudgetMs:r.remainingMillihours};
  });
  assert.equal(mapping.normal, 5); assert.equal(mapping.reversed, 5);
  return {rows, mapping, identity:{participant, session:options.sessionId, condition, variant,
    profileId:E.R9, flowVersion:E.R10_FLOW, seed:String(seed),
    resolvedRecipeVersion:normalized.meta.resolvedRecipeVersion,
    resolvedRecipeFingerprint:normalized.meta.resolvedRecipeFingerprint}};
}

function buildPanel(design, seed='candidate-initial-design-v1') {
  E.validateDesign(design);
  const rows=[], sessions=[];
  for (const condition of ['C1','C2','C3','C4','C5']) for (let variant=0; variant<9; variant++) {
    const session=oneSession(design, condition, variant, seed);
    rows.push(...session.rows);
    sessions.push({identity:session.identity, mapping:session.mapping});
  }
  assert.equal(sessions.length, 45); assert.equal(rows.length, 450);
  return {schema:'tutoring-statistics-fixture-rows-v1', origin:'synthetic_software_fixture', rows,
    metadata:{schema:'candidate-initial-design-panel-v1', dataRole:'scripted_source_software_fixture',
    profileId:E.R9, flowVersion:E.R10_FLOW, seed:String(seed), sessionCount:45, mainRows:450,
    source:'current_engine_makeSchedule_prior_moments_and_genuine_transition_normalizer',
    policy:'no_questions_scripted_share_grid_no_theoretical_claim',
    scope:'nine_variants_per_condition_one_declared_seed_identity_set_not_all_side_masks',
    humanObservations:false, scientificMethodAdopted:false, sessions}};
}

module.exports={orderedPriors, verifyMapping, oneSession, buildPanel};
if (require.main===module) {
  const args=process.argv.slice(2);
  if (args.length===1 && args[0]==='--help') {
    process.stdout.write('node initial_panel.cjs --design resolved_design.json --output panel.json [--seed value]\n');
  } else {
    try {
      const options={};
      for (let i=0; i<args.length; i+=2) {
        if (!['--design','--output','--seed'].includes(args[i]) || !args[i+1] || options[args[i]]!==undefined)
          throw Error('Expected unique --design, --output and optional --seed values');
        options[args[i]]=args[i+1];
      }
      if (!options['--design'] || !options['--output']) throw Error('--design and --output are required');
      const design=JSON.parse(fs.readFileSync(options['--design'],'utf8'));
      const panel=buildPanel(design, options['--seed']);
      fs.writeFileSync(options['--output'], JSON.stringify(panel, null, 2)+'\n', {flag:'wx'});
      process.stdout.write(JSON.stringify({ok:true, sessionCount:45, mainRows:450, output:options['--output']})+'\n');
    } catch(error) {
      process.stderr.write(String(error.stack || error)+'\n'); process.exitCode=1;
    }
  }
}
