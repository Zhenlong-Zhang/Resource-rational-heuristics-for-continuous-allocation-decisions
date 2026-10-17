/* Deterministic synthetic transitions; not browser or human-session validation. */
'use strict';
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const E = require('./native_engine.cjs'), N = require('./native_normalizer.cjs');
const assert = (x, message) => { if (!x) throw Error(message); };
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const stable = value => value && typeof value === 'object'
  ? (Array.isArray(value) ? '[' + value.map(stable).join(',') + ']'
    : '{' + Object.keys(value).sort().map(k => JSON.stringify(k) + ':' + stable(value[k])).join(',') + '}')
  : JSON.stringify(value);
function numericBits(value) {
  if(typeof value==='number') { assert(Number.isFinite(value),'Nonfinite receipt number');
    assert(!Number.isInteger(value)||Number.isSafeInteger(value),'Large integer receipt identities must be decimal strings');
    const b=Buffer.alloc(8);b.writeDoubleBE(value);return {number_binary64:b.toString('hex')}; }
  if(Array.isArray(value))return value.map(numericBits);
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,numericBits(v)]));
  return value;
}
const digest = value => sha(stable(numericBits(value)));

// Frozen engine identities and RNG salt are scientific inputs, not release labels.
const SOURCE_FILES = ['native_source.cjs', 'native_engine.cjs', 'native_normalizer.cjs'];
const snapshot = () => Object.fromEntries(SOURCE_FILES.map(f => [f, sha(fs.readFileSync(path.join(__dirname, f)))]));
function prepare(design, analysis, calibration) {
  E.validateDesign(design);
  assert(analysis.profileId === 'finite-budget-only-v1' && analysis.analysisId === 'person-level-nominal-v1'
    && analysis.questionLimitRule === 'budget_only', 'Wrong scientific analysis');
  assert(calibration.schema === 'native-calibration-spec-v1'
    && stable(calibration.lookupVariants) === stable([0,1,2,3,4,5,6,7,8])
    && stable(calibration.lookupConditions) === stable(['C1','C2','C3','C4','C5'])
    && stable(calibration.lookupPrefixes) === stable([[],[0],[0,1,0]])
    && calibration.lookupSeedNamespace === 'r9-calibration-lookup-v1', 'Frozen construction changed');
  const sourceFiles = snapshot(), inputHash = digest({design, analysis, calibration});
  const states = [], sessions = [];
  for (const condition of calibration.lookupConditions) for (const variant of calibration.lookupVariants) {
    const seed = calibration.lookupSeedNamespace + ':' + condition + ':variant:' + variant;
    const options = {profileId:E.R9, conditionId:condition, scheduleVariant:variant,
      participantId:'synthetic-lookup-' + condition + '-v' + variant, sessionId:'synthetic-source-' + condition + '-v' + variant,
      seed, startedAt:'2026-10-11T00:00:00.000Z'};
    let state = E.createSession(design, options), counter = 0;
    const act = (s, type, fields={}) => E.transition(s, {id:'lookup-' + condition + '-' + variant + '-' + counter,
      expectedRevision:s.revision, type, at:new Date(Date.UTC(2026,9,11) + counter++ * 1000).toISOString(), ...fields}, design);
    const meta = {...E.r9Metadata(design), profileId:state.profileId, guideFixtureId:state.guideFixtureId,
      guideFixtureHash:state.guideFixtureHash, presentationVersion:state.presentationVersion,
      feedbackStyle:state.feedbackStyle, designFingerprint:state.designFingerprint};
    state = act(state, 'beginChecks');
    state = act(state, 'submitChecks', {answers:E.checkQuestions(state).map(q => q.correct)});
    while (state.currentTrialIndex < 14) {
      if (state.phase === 'practiceGuide') {
        state = act(state, 'beginPracticeChecks', {fixtureId:state.guideFixtureId, fixtureHash:state.guideFixtureHash});
        state = act(state, 'submitChecks', {answers:E.checkQuestions(state).map(q => q.correct)});
      }
      if (state.phase === 'break') state = act(state, 'continueBreak');
      assert(state.phase === 'trial' && state.current, 'Unexpected genuine phase before trial');
      const d = state.current.definition;
      if (d.stage === 'main') {
        assert(d.gapConvention === meta.resolvedRecipe.gapConvention, 'Definition gap convention missing');
        if (condition === 'C1') assert(d.gap === 0 && d.c1TotalCentreGapMinutes === 1, 'C1 row-offset/total-gap conflation');
        const prefixes = calibration.lookupPrefixes.map(x => x.slice());
        if (d.stageIndex === 1 && d.budgetMillihours % d.costMillihours === 0)
          prefixes.push(Array.from({length:d.budgetMillihours/d.costMillihours}, (_,i) => i % 2));
        for (const prefix of prefixes) {
          let fork = E.restore(E.serialize(state), design, options);
          for (const student of prefix) fork = act(fork, 'assess', {student});
          const t = fork.current, exhaustion = prefix.length > 3;
          assert(t.budgetMillihours === d.budgetMillihours - prefix.length*d.costMillihours, 'Integer prefix budget mismatch');
          assert(exhaustion ? t.budgetMillihours === 0 : t.budgetMillihours > 0, 'Unexpected prefix affordability');
          const paid = t.events.filter(e => e.type === 'assessment').map(e => ({student:e.student,report:e.report}));
          const serialized = E.serialize(fork);
          states.push({conditionId:condition, scheduleVariant:variant, mainIndex:d.stageIndex,
            prefixLength:prefix.length, prefixKind:exhaustion?'exact_exhaustion':'fixed', seed,
            definition:t.definition, logicalToDisplayed:t.logicalToDisplayed, initialLogs:t.initialLogs,
            terminalLogs:t.logs, paid, remainingMillihours:t.budgetMillihours,
            stateSnapshotSHA256:sha(serialized), semanticFingerprint:E.fingerprint({definition:t.definition,paid,logs:t.logs,budget:t.budgetMillihours}),
            meta, common:design.common, enginePhase:fork.phase, syntheticOnly:true});
        }
      }
      // Baseline ES is only a source-schedule advancing policy, not a null/RR claim.
      state = act(state, 'openStop');
      if (state.phase === 'stopConfirm') state = act(state, 'confirmStop');
      state = act(state, 'editAllocationIndex', {shareIndex:500});
      state = act(state, 'confirmAllocation'); state = act(state, 'advance'); state = act(state, 'acknowledgeAdvance');
    }
    assert(state.phase === 'posttask' && state.currentTrialIndex === 14, 'Source dose not completed');
    const normalized = N.normalize(E.exportData(state), design);
    assert(normalized.errors.length === 0 && normalized.trials.length === 14, 'Baseline normalizer replay failed');
    sessions.push({conditionId:condition, scheduleVariant:variant, seed, identity:meta,
      checks:state.checks, practiceGuideHistory:state.practiceGuideHistory,
      completedTrialCount:state.completedTrials.length, snapshotSHA256:sha(E.serialize(state)),
      actualPhase:state.phase, exported:E.exportData(state),
      completionClaim:'genuine_posttask_only;no_final_reflection_complete_or_native_ack_claim'});
  }

  assert(stable(snapshot()) === stable(sourceFiles), 'Source bytes changed during preparation');
  assert(digest({design, analysis, calibration}) === inputHash, 'Scientific inputs changed');
  const sourceReceipt = {schema:'native-source-content-v1', designHash:digest(design),
    common:design.common, identity:sessions[0].identity, files:sourceFiles,
    scope:'synthetic_finite_transitions_only', humanObservations:false};
  return {schema:'native-source-states-v1', syntheticOnly:true, humanObservations:false,
    sourceFiles, sourceReceipt, sourceReceiptHash:digest(sourceReceipt),
    analysisSpecHash:digest(analysis), calibrationSpecHash:digest(calibration),
    designHash:digest(design), design, engineVersion:E.VERSION,
    construction:'genuine_public_transitions;baseline_ES;prefixes_0_1_3;fixed_round1_exact_exhaustion',
    states, sessions, status:'constructed'};
}
function verify(source, analysis, calibration) {
  const reconstructed = prepare(source.design, analysis, calibration);
  assert(digest(reconstructed) === digest(source), 'Source reconstruction mismatch');
  return {schema:'native-source-check-v1', sourceStatesHash:digest(source), sessions:45,
    states:source.states.length, humanObservations:false};
}
function main(argv) {
  if (argv.length === 1 && argv[0] === '--verify-stdin') {
    const request = JSON.parse(fs.readFileSync(0, 'utf8'));
    process.stdout.write(JSON.stringify(verify(request.source, request.analysis, request.calibration)));
    return;
  }
  if (argv.includes('--help')) {
    process.stdout.write('node native_source.cjs --design FILE --analysis-spec FILE --calibration-spec FILE --output DIR\n');
    return;
  }
  const args = {};
  for (let i=0;i<argv.length;i+=2) {
    assert(argv[i].startsWith('--') && argv[i+1], 'Expected --name value');
    const key = argv[i].slice(2); assert(!(key in args), 'Duplicate argument');
    args[key] = argv[i+1];
  }
  assert(Object.keys(args).sort().join(',') === 'analysis-spec,calibration-spec,design,output', 'Exact CLI inputs required');
  assert(!fs.existsSync(args.output), 'Output directory must be fresh');
  const inputs = Object.fromEntries(['design','analysis-spec','calibration-spec'].map(key => {
    const bytes = fs.readFileSync(args[key]);
    return [key, {value:JSON.parse(bytes.toString('utf8')), hash:sha(bytes)}];
  }));
  const result = prepare(inputs.design.value, inputs['analysis-spec'].value, inputs['calibration-spec'].value);
  for (const [key, input] of Object.entries(inputs))
    assert(sha(fs.readFileSync(args[key])) === input.hash, 'Input file changed: '+key);
  assert(stable(snapshot()) === stable(result.sourceFiles), 'Source changed before output');
  fs.mkdirSync(args.output);
  const temporary = path.join(args.output, 'source_states.json.tmp');
  const fd = fs.openSync(temporary, 'wx');
  try { fs.writeFileSync(fd, JSON.stringify(result)); fs.fsyncSync(fd); }
  finally { fs.closeSync(fd); }
  fs.renameSync(temporary, path.join(args.output, 'source_states.json'));
  const directory = fs.openSync(args.output, 'r');
  try { fs.fsyncSync(directory); } finally { fs.closeSync(directory); }
}
module.exports = {prepare, verify, stable, digest, SOURCE_FILES};
if (require.main === module) {
  try { main(process.argv.slice(2)); }
  catch (error) { process.stderr.write(error.message+'\n'); process.exitCode=1; }
}
