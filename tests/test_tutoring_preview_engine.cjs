/* Purpose: check the actual standalone build's 14-task profile and input semantics. */
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const E = require('../src/tutoring_task/engine.js');
assert.ok(process.env.TUTORING_PREVIEW_DIR, 'Build the preview and set TUTORING_PREVIEW_DIR');
const design = JSON.parse(fs.readFileSync(path.join(process.env.TUTORING_PREVIEW_DIR, 'resolved_design.json')));
const options = (conditionId, scheduleVariant = 0) => ({
  conditionId, scheduleVariant, profileId: E.R9, participantFlowVersion: E.R10_FLOW,
  seed: 'standalone-software-fixture', participantId: 'synthetic', sessionId: 'preview-test',
  startedAt: '2000-01-01T00:00:00.000Z',
});

test('five conditions and nine schedules retain four practice and ten main tasks', () => {
  for (let arm = 1; arm <= 5; arm++) for (let variant = 0; variant < 9; variant++) {
    const state = E.createSession(design, options('C' + arm, variant));
    assert.equal(state.schedule.length, 14);
    assert.equal(state.schedule.filter(x => x.stage === 'practice').length, 4);
    assert.equal(state.schedule.filter(x => x.stage === 'main').length, 10);
    assert.equal(state.questionLimitRule, 'budget_only');
    assert.equal(state.maximumOnlineAssessments, null);
    assert.equal(state.participantFlowVersion, E.R10_FLOW);
    for (const trial of state.schedule) {
      assert.equal(E.questionCapacity(state, trial), Math.floor(trial.budgetMillihours / trial.costMillihours));
      if (arm === 1) assert.equal(trial.priorCentres[1] - trial.priorCentres[0], 1);
    }
    assert.deepEqual(E.restore(E.serialize(state), design), state);
  }
});

test('design mismatch is rejected without modifying a committed session', () => {
  const stored = E.serialize(E.createSession(design, options('C3')));
  const changed = structuredClone(design);
  changed.schema = 'different-packaging-identity';
  assert.throws(() => E.restore(stored, changed), /design|fingerprint/i);
  assert.equal(E.serialize(E.restore(stored, design)), stored);
});

test('unsupported scientific recipe is not silently accepted', () => {
  const changed = structuredClone(design);
  changed.conditions[2].sample_time_cost = 0.1;
  assert.throws(() => E.validateDesign(changed), /recipe/i);
});
