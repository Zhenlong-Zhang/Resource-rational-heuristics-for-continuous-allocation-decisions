/* Render only scientific display operands; no private UI build is needed. */
'use strict';
const fs = require('node:fs');
const E = require('../../tutoring_analysis/native_engine.cjs');
const need = (ok, message) => { if (!ok) throw Error(message); };

function bits(value) {
  const buffer = Buffer.alloc(8);
  buffer.writeDoubleBE(value);
  return buffer.toString('hex');
}

function render(root) {
  const prior = root.logBits.map(row => row.map(value => Buffer.from(value, 'hex').readDoubleBE()));
  need(prior.length === 2 && prior.every(row => row.length === 100 && row.every(Number.isFinite)),
    'Two complete finite priors');
  need(JSON.stringify(prior.map(row => row.map(bits))) === JSON.stringify(root.logBits), 'Exact prior bits');
  const channel = E.likelihood(root.task.report_width);
  const display = logs => E.formatR9Number(E.moments(logs).mean, 1);
  return {id: root.id, status: 'authenticated', root: prior.map(display),
    priorLogBits: prior.map(row => row.map(bits)), orderEquivalent: true,
    singleReport: prior.map(logs => Array.from({length: 100}, (_, i) =>
      display(E.posterior(logs, i + 1, channel))))};
}

if (process.argv.slice(2).join(' ') === '--help') {
  process.stdout.write('Render original root and one-report guesses from stdin JSON.\n');
} else {
  need(process.argv.length === 2 && process.versions.node === '22.14.0', 'Reference Node 22.14.0');
  process.stdout.write(JSON.stringify(JSON.parse(fs.readFileSync(0, 'utf8')).map(render)) + '\n');
}
