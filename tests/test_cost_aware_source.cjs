/* Test purpose: selected-root support and skipped-fork clock arithmetic only. */
'use strict';
const assert = require('node:assert/strict');
const S = require('../src/design_validation/cost_aware_source.cjs');

const construction = {lookupPrefixes:[[],[0],[0,1,0]]};
const definition = {stageIndex:1, budgetMillihours:150000, costMillihours:1200};
assert.equal(S.skippedPrefixQueries(definition, construction), 4 + 125);
assert.equal(S.skippedPrefixQueries({...definition, stageIndex:2}, construction), 4);
assert.equal(S.skippedPrefixQueries({...definition, budgetMillihours:150001}, construction), 4);
assert.equal(S.skippedPrefixQueries({...definition, budgetMillihours:170000, costMillihours:500}, construction), 4 + 340);

const roots = [];
for (const conditionId of ['C1','C2','C3','C4','C5'])
  for (let scheduleVariant=0; scheduleVariant<=8; scheduleVariant++)
    for (let mainIndex=1; mainIndex<=10; mainIndex++)
      roots.push({conditionId, scheduleVariant, mainIndex});
assert.deepEqual(S.rootSelection(roots.slice().reverse()), roots);
const first = roots[0];
for (const invalid of [[], [first, first], [{...first, mainIndex:0}], [{...first, mainIndex:11}],
  [{...first, scheduleVariant:9}], [{...first, scheduleVariant:true}], [{...first, conditionId:'C6'}],
  [{...first, replicateId:1}]]) assert.throws(() => S.rootSelection(invalid));
console.log('Selected-root support and reserved counter checks passed; no sessions or trials generated.');
