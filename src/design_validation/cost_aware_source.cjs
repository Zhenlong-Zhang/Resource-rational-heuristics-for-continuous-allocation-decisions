/* Selected unpaid roots through genuine native session ancestry, not a full panel. */
'use strict';
const crypto = require('crypto');
const E = require('../tutoring_analysis/native_engine.cjs');
const assert = (ok, message) => { if (!ok) throw Error(message); };
const sha = value => crypto.createHash('sha256').update(value).digest('hex');

function rootSelection(rows) {
  assert(Array.isArray(rows) && rows.length > 0, 'Explicit selected roots required');
  const seen = new Set();
  const result = rows.map(row => {
    assert(row && Object.keys(row).sort().join(',') === 'conditionId,mainIndex,scheduleVariant', 'Exact root coordinates required');
    assert(['C1','C2','C3','C4','C5'].includes(row.conditionId)
      && Number.isInteger(row.scheduleVariant) && row.scheduleVariant >= 0 && row.scheduleVariant <= 8
      && Number.isInteger(row.mainIndex) && row.mainIndex >= 1 && row.mainIndex <= 10, 'Root outside scientific support');
    const key = [row.conditionId, row.scheduleVariant, row.mainIndex].join(':');
    assert(!seen.has(key), 'Duplicate selected root'); seen.add(key);
    return {conditionId:row.conditionId, scheduleVariant:row.scheduleVariant, mainIndex:row.mainIndex};
  });
  return result.sort((a,b) => a.conditionId.localeCompare(b.conditionId)
    || a.scheduleVariant-b.scheduleVariant || a.mainIndex-b.mainIndex);
}

function skippedPrefixQueries(definition, construction) {
  assert(Number.isSafeInteger(definition.budgetMillihours) && definition.budgetMillihours > 0
    && Number.isSafeInteger(definition.costMillihours) && definition.costMillihours > 0, 'Integer source budget/cost');
  let count = construction.lookupPrefixes.reduce((sum, prefix) => sum + prefix.length, 0);
  if (definition.stageIndex === 1 && definition.budgetMillihours % definition.costMillihours === 0)
    count += definition.budgetMillihours / definition.costMillihours;
  return count;
}

function prepare(design, analysis, construction, selected) {
  E.validateDesign(design);
  assert(analysis.profileId === 'finite-budget-only-v1' && analysis.analysisId === 'person-level-nominal-v1'
    && analysis.questionLimitRule === 'budget_only', 'Wrong scientific analysis');
  assert(construction.schema === 'native-calibration-spec-v1'
    && JSON.stringify(construction.lookupVariants) === JSON.stringify([0,1,2,3,4,5,6,7,8])
    && JSON.stringify(construction.lookupConditions) === JSON.stringify(['C1','C2','C3','C4','C5'])
    && JSON.stringify(construction.lookupPrefixes) === JSON.stringify([[],[0],[0,1,0]])
    && construction.lookupSeedNamespace === 'r9-calibration-lookup-v1', 'Frozen construction changed');
  const roots = rootSelection(selected), groups = new Map(), states = [], sessions = [];
  for (const root of roots) {
    const key = root.conditionId + ':' + root.scheduleVariant;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(root);
  }
  for (const group of groups.values()) {
    const condition = group[0].conditionId, variant = group[0].scheduleVariant;
    const wanted = new Set(group.map(row => row.mainIndex)), last = Math.max(...wanted);
    const seed = construction.lookupSeedNamespace + ':' + condition + ':variant:' + variant;
    const options = {profileId:E.R9, conditionId:condition, scheduleVariant:variant,
      participantId:'synthetic-lookup-' + condition + '-v' + variant,
      sessionId:'synthetic-source-' + condition + '-v' + variant,
      seed, startedAt:'2026-10-11T00:00:00.000Z'};
    let state = E.createSession(design, options), counter = 0;
    const act = (s, type, fields={}) => E.transition(s, {
      id:'lookup-' + condition + '-' + variant + '-' + counter,
      expectedRevision:s.revision, type,
      at:new Date(Date.UTC(2026,9,11) + counter++ * 1000).toISOString(), ...fields}, design);
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
      assert(state.phase === 'trial' && state.current, 'Unexpected genuine phase before selected root');
      const d = state.current.definition;
      if (d.stage === 'main') {
        assert(d.gapConvention === meta.resolvedRecipe.gapConvention, 'Definition gap convention missing');
        if (condition === 'C1') assert(d.gap === 0 && d.c1TotalCentreGapMinutes === 1, 'C1 gap convention');
        if (wanted.has(d.stageIndex)) {
          const fork = E.restore(E.serialize(state), design, options), t = fork.current;
          const paid = t.events.filter(e => e.type === 'assessment').map(e => ({student:e.student,report:e.report}));
          assert(paid.length === 0 && t.budgetMillihours === d.budgetMillihours, 'Unpaid selected root required');
          states.push({conditionId:condition, scheduleVariant:variant, mainIndex:d.stageIndex,
            prefixLength:0, prefixKind:'fixed', seed, definition:t.definition,
            logicalToDisplayed:t.logicalToDisplayed, initialLogs:t.initialLogs, terminalLogs:t.logs,
            paid, remainingMillihours:t.budgetMillihours, stateSnapshotSHA256:sha(E.serialize(fork)),
            semanticFingerprint:E.fingerprint({definition:t.definition,paid,logs:t.logs,budget:t.budgetMillihours}),
            meta, common:design.common, enginePhase:fork.phase, syntheticOnly:true});
        }
        if (d.stageIndex === last) break;
        // Discarded paid forks change only this shared clock, not the parent.
        // Reserve their calls so later baseline actions retain original IDs/times.
        counter += skippedPrefixQueries(d, construction);
      }
      state = act(state, 'openStop');
      if (state.phase === 'stopConfirm') state = act(state, 'confirmStop');
      state = act(state, 'editAllocationIndex', {shareIndex:500});
      state = act(state, 'confirmAllocation'); state = act(state, 'advance');
      state = act(state, 'acknowledgeAdvance');
    }
    assert(state.phase === 'trial' && state.current.definition.stage === 'main'
      && state.current.definition.stageIndex === last, 'Selected ancestry ended at the wrong root');
    sessions.push({conditionId:condition, scheduleVariant:variant, seed, identity:meta,
      checks:state.checks, practiceGuideHistory:state.practiceGuideHistory,
      completedTrialCount:state.completedTrials.length, snapshotSHA256:sha(E.serialize(state)),
      actualPhase:state.phase, exported:E.exportData(state),
      completionClaim:'selected_root_prefix_only;no_complete_session_claim'});
  }
  assert(states.length === roots.length, 'Selected root roster incomplete');
  return {schema:'finite-cost-aware-source-roots-v1', scope:'selected_working_roots',
    fullPanel:false, syntheticOnly:true, humanObservations:false, rootSelection:roots,
    engineVersion:E.VERSION, construction:'genuine_native_ancestry;baseline_ES;unpaid_selected_roots;reserved_fork_action_counters',
    states, sessions, status:'constructed_selected_roots'};
}

module.exports = {prepare, rootSelection, skippedPrefixQueries};
