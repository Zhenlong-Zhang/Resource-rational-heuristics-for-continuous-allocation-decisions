/* Original unpaid roots through the existing public native ancestry constructor. */
'use strict';
const fs = require('fs');
const crypto = require('crypto');
const source = require('./cost_aware_source.cjs');
const assert = (ok, message) => { if (!ok) throw Error(message); };
const FIELDS = ['conditionId','scheduleVariant','mainIndex','seed','definition',
  'logicalToDisplayed','initialLogs','common'];
const select = row => Object.fromEntries(FIELDS.map(key => [key, row[key]]));
const stable = value => value && typeof value === 'object'
  ? (Array.isArray(value) ? '[' + value.map(stable).join(',') + ']'
    : '{' + Object.keys(value).sort().map(k => JSON.stringify(k) + ':' + stable(value[k])).join(',') + '}')
  : JSON.stringify(value);

function numericBits(value) {
  if (typeof value === 'number') {
    assert(Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value)), 'Finite binary64 input');
    const b = Buffer.alloc(8); b.writeDoubleBE(value);
    return {number_binary64:b.toString('hex')};
  }
  if (Array.isArray(value)) return value.map(numericBits);
  if (value && typeof value === 'object')
    return Object.fromEntries(Object.entries(value).map(([k,v]) => [k,numericBits(v)]));
  return value;
}

function scientificDigest(value) {
  return crypto.createHash('sha256').update(stable(numericBits(value))).digest('hex');
}

function prepare(request) {
  const selected = source.rootSelection(request.selection);
  const reference = request.identity;
  assert(reference.schema === 'r10-original-root-input-identity-v1', 'Original root identity schema');
  const expected = new Map(reference.coordinates.map(r => [[r.condition,r.variant,r.main].join(':'), r]));
  assert(reference.coordinates.length === 450 && expected.size === 450, '450 unique reference coordinates');
  const result = source.prepare(request.design, request.analysis, request.construction, selected);
  const sessions = new Map(result.sessions.map(s => [[s.conditionId,s.scheduleVariant].join(':'), s]));
  const roots = result.states.map(row => {
    const key = [row.conditionId,row.scheduleVariant,row.mainIndex].join(':');
    const wanted = expected.get(key);
    assert(wanted && scientificDigest(select(row)) === wanted.scientificSHA256, 'Original scientific binary64 mismatch: ' + key);
    assert(row.stateSnapshotSHA256 === wanted.snapshotSHA256, 'Original snapshot mismatch: ' + key);
    const session = sessions.get([row.conditionId,row.scheduleVariant].join(':'));
    // Keep only constructor inputs, never the baseline session's realized outcomes.
    const exported = Object.fromEntries(['profileId','conditionId','scheduleVariant','participantId',
      'sessionId','seed','startedAt'].map(k => [k,session.exported[k]]));
    return {...select(row), stateSnapshotSHA256:row.stateSnapshotSHA256,
      prefixLength:row.prefixLength, prefixKind:row.prefixKind, paid:row.paid,
      meta:{resolvedRecipeFingerprint:row.meta.resolvedRecipeFingerprint}, session:{exported}};
  });
  assert(roots.length === selected.length, 'Exact selected original root count');
  return {schema:'personal-time-original-roots-v1', selection:selected, roots,
    originalScientificBinary64Matched:true, originalSnapshotsMatched:true,
    realizedPolicyHistoriesGenerated:false, nodeVersion:process.versions.node};
}

function main(argv) {
  if (argv.length === 1 && argv[0] === '--help') {
    process.stdout.write('Read a personal-time source request as JSON on stdin; emit checked original roots.\n');
    return;
  }
  assert(argv.length === 0, 'Use stdin JSON, or --help');
  assert(process.versions.node === '22.14.0', 'Reference Node 22.14.0 is required');
  process.stdout.write(JSON.stringify(prepare(JSON.parse(fs.readFileSync(0, 'utf8')))) + '\n');
}

module.exports = {prepare, scientificDigest, numericBits, select};
if (require.main === module) {
  try { main(process.argv.slice(2)); }
  catch (error) { process.stderr.write(error.message + '\n'); process.exitCode = 1; }
}
