/* Purpose: source-adapter binary64 identity and original transition-clock contract.
 * Full original-root agreement is checked separately by the explicit prepare CLI.
 */
'use strict';
const assert = require('assert');
const input = require('../src/design_validation/personal_time_source.cjs');
const source = require('../src/design_validation/cost_aware_source.cjs');
assert.deepStrictEqual(input.numericBits(1), {number_binary64:'3ff0000000000000'});
assert.notDeepStrictEqual(input.numericBits(-0), input.numericBits(0));
assert.throws(() => input.numericBits(Infinity));
assert.throws(() => input.numericBits(Number.MAX_SAFE_INTEGER + 1));
assert.strictEqual(input.scientificDigest({a:1,b:[2]}), input.scientificDigest({b:[2],a:1}));
const construction = {lookupPrefixes:[[],[0],[0,1,0]]};
assert.strictEqual(source.skippedPrefixQueries({stageIndex:1,budgetMillihours:120,costMillihours:3}, construction), 44);
assert.strictEqual(source.skippedPrefixQueries({stageIndex:2,budgetMillihours:120,costMillihours:3}, construction), 4);
assert.strictEqual(source.skippedPrefixQueries({stageIndex:1,budgetMillihours:121,costMillihours:3}, construction), 4);
assert.throws(() => source.rootSelection([{conditionId:'C1',scheduleVariant:0,mainIndex:1},
  {conditionId:'C1',scheduleVariant:0,mainIndex:1}]));
assert.throws(() => input.prepare({selection:[],identity:{}}));
process.stdout.write('Personal-time source interface invariants passed.\n');
