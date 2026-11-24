/* Verify the public standalone build in one fresh synthetic C3 browser session.
 * Requires Node 22+ (native WebSocket/fetch) and an installed Chromium headless
 * browser with its runtime libraries. No npm packages or runtime downloads.
 * Set TUTORING_CHROME to the browser's absolute executable path.
 * Build first: python3 scripts/build_tutoring_preview.py --output results/preview
 * Run: node tests/test_tutoring_standalone_browser.cjs results/preview/index.html
 * Optional: --output results/preview/browser-check.json (must not already exist).
 * This is synthetic software verification, not a policy campaign or human study.
 * Uses local files and loopback CDP only; no screenshots or remote services.
 * Same-build committed-answer recovery is tested, not cross-build migration.
 */
'use strict';

const assert = require('node:assert/strict');
const { withChrome } = require('./helpers/tutoring_preview_chrome.cjs');
const { startTask, exposeReadback } = require('./helpers/tutoring_preview_flow.cjs');
const { readExport } = require('../src/tutoring_task/local_strategy.js');

const OTHER_TEXT = '  ' + '策略 🧠 café <tag> & quoted\n'.repeat(2000) + '  ';
const moduleVisible = '!!document.querySelector("[data-r13-skip]") && !document.querySelector("[data-r13-skip]").closest("[hidden]")';
const reflectionVisible = '!!document.querySelector(\'form[data-form="posttask"]\') && !document.querySelector(\'form[data-form="posttask"]\').closest("[hidden]")';

async function readDownload({ evaluate }, selector = "#preview-export") {
  return evaluate(`(async () => {
    const realURL = URL.createObjectURL;
    const realClick = HTMLAnchorElement.prototype.click;
    let capture = null;
    URL.createObjectURL = blob => { capture = blob; return realURL(blob); };
    HTMLAnchorElement.prototype.click = function () {};
    try {
      document.querySelector(${JSON.stringify(selector)}).click();
      if (!capture) throw new Error('Preview did not offer a JSON export');
      return JSON.parse(await capture.text());
    } finally {
      URL.createObjectURL = realURL;
      HTMLAnchorElement.prototype.click = realClick;
    }
  })()`);
}

async function run(browser) {
  const { evaluate, waitFor, call } = browser;
  async function reloadAndWait(ready) {
    await evaluate('window.r13PriorDocument = true');
    await call('Page.reload');
    await waitFor('!window.r13PriorDocument && (' + ready + ')');
  }
  const atFinal = await startTask(browser);
  assert.equal(await evaluate(moduleVisible), false);
  assert.equal(await evaluate(reflectionVisible), false);
  await evaluate(`window.setItemOriginal = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key === taskReadback().key && JSON.parse(value).payload.phase === 'posttask') {
        throw new Error('Synthetic final acknowledgment storage failure');
      }
      return setItemOriginal.call(this, key, value);
    };
    document.querySelector('[data-action="advance"]').click();`);
  await waitFor('!!document.querySelector(\'[data-action="retryPending"]\')');
  assert.equal(await evaluate('taskReadback().task.phase'), 'pendingCommit');
  assert.equal(await evaluate(moduleVisible), false);
  assert.equal(await evaluate(reflectionVisible), false);
  await evaluate(`Storage.prototype.setItem = setItemOriginal;
    document.querySelector('[data-action="retryPending"]').click();`);
  await waitFor(moduleVisible);
  assert.equal(await evaluate(reflectionVisible), false);
  const taskAtModule = await evaluate('taskReadback().task.completedTrials');
  assert.deepEqual(taskAtModule.map(record => record.trialId),
    [...atFinal.refs, atFinal.finalAtFeedback]);
  const readOrder = () => evaluate(`[...document.querySelectorAll('[data-r13-choice]')]
    .map(x => x.dataset.r13Choice || x.value)`);
  const cardOrder = await readOrder();
  assert.equal(new Set(cardOrder).size, 7);
  assert.equal(cardOrder.at(-1), 'other');
  assert.deepEqual([...cardOrder.slice(0, -1)].sort(), [
    'adaptive_combined_gpa', 'all_to_higher_guess', 'all_to_lower_guess',
    'ask_each_equal_projected_gpa', 'equal_minutes', 'equal_projected_gpa',
  ]);
  const prior = await readDownload(browser);
  assert.equal(prior.schema, 'r13-local-export-v1');
  assert.equal(prior.task.conditionId, 'C3');
  assert.equal(prior.task.profileId, 'pilot-r9-design-v1');
  assert.equal(prior.task.participantFlowVersion, 'r10-prepractice-v1');
  assert.equal(prior.task.questionLimitRule, 'budget_only');
  assert.equal(prior.task.schedule.filter(task => task.stage === 'practice').length, 4);
  assert.equal(prior.task.schedule.filter(task => task.stage === 'main').length, 10);
  assert.equal(prior.task.exportIsFullHistory, true);
  assert.equal(prior.task.designFingerprint, await evaluate(
    'TutoringTaskEngine.sessionFingerprint(TUTORING_DESIGN, {profileId:TUTORING_PROFILE})'));
  assert.equal(prior.strategy.answers.length, 0);
  assert.equal(prior.strategy.completedAt, null);
  assert.equal(prior.strategy.questionOrder.at(-1), 'self_report');

  await evaluate(`document.querySelector('[data-r13-choice][value="other"]').click()`);
  assert.equal(await evaluate('document.querySelector("#r13-other").required'), true);
  assert.equal(await evaluate('document.querySelector("#r13-other").hasAttribute("maxlength")'), false);
  await evaluate(`document.querySelector('[data-r13-form]').requestSubmit()`);
  assert.deepEqual((await readDownload(browser)).strategy, prior.strategy);
  await evaluate(`document.querySelector('#r13-other').value = ' \\n\\t ';
    document.querySelector('[data-r13-form]').dispatchEvent(new Event('submit', {bubbles:true,cancelable:true}))`);
  await waitFor('document.querySelector("#r13-root [role=alert]")?.textContent.includes("Describe your approach")');
  assert.deepEqual((await readDownload(browser)).strategy, prior.strategy);

  await evaluate(`window.setItemOriginal = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key !== taskReadback().key) throw new Error('Synthetic strategy storage failure');
      return setItemOriginal.call(this, key, value);
    };
    document.querySelector('[data-r13-skip]').click();`);
  await waitFor('document.querySelector("#r13-root [role=alert]")?.textContent.includes("Synthetic strategy storage failure")');
  assert.deepEqual((await readDownload(browser)).strategy, prior.strategy);
  assert.deepEqual(await readOrder(), cardOrder);
  await evaluate(`Storage.prototype.setItem = setItemOriginal;
    document.querySelector('[data-r13-skip]').click();`);
  await waitFor('document.querySelector("#r13-root")?.textContent.includes("2 OF 4")');
  const partial = await readDownload(browser);
  assert.equal(partial.strategy.answers.length, 1);
  assert.equal(partial.strategy.answers[0].status, 'skipped');
  assert.equal(partial.strategy.answers[0].choiceId, null);

  await reloadAndWait(moduleVisible);
  await exposeReadback(browser);
  assert.deepEqual((await readDownload(browser)).strategy, partial.strategy);
  assert.deepEqual(await readOrder(), cardOrder);
  assert.deepEqual(await evaluate('taskReadback().task.completedTrials'), taskAtModule);

  await evaluate(`(async () => {
    const other = document.querySelector('[data-r13-choice][value="other"]');
    if (!other) throw new Error('Other choice unavailable');
    other.click();
    await new Promise(resolve => setTimeout(resolve, 0));
    const textArea = document.querySelector('#r13-root textarea');
    if (!textArea) throw new Error('Other explanation is unavailable');
    textArea.value = ${JSON.stringify(OTHER_TEXT)};
    textArea.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector('[data-r13-submit]').click();
  })()`);
  await waitFor('!!document.querySelector("[data-r13-skip]") && !document.querySelector("[data-r13-skip]").disabled');
  assert.deepEqual(await readOrder(), cardOrder);
  const savedLong = await readDownload(browser);
  assert.equal(savedLong.strategy.answers[1].otherText, OTHER_TEXT);
  await reloadAndWait(moduleVisible);
  await exposeReadback(browser);
  assert.deepEqual((await readDownload(browser)).strategy, savedLong.strategy);
  await evaluate(`(async () => {
    const nextTurn = () => new Promise(resolve => setTimeout(resolve, 60));
    document.querySelector('[data-r13-skip]').click();
    await nextTurn();
    document.querySelector('[data-r13-skip]').click();
    await nextTurn();
  })()`);
  await waitFor(reflectionVisible);
  assert.equal(await evaluate(moduleVisible), false);
  await evaluate(`(() => {
    const form = document.querySelector('form[data-form="posttask"]');
    for (const select of form.querySelectorAll('select')) select.value = '4';
    form.requestSubmit();
  })()`);
  await waitFor('taskReadback().task.phase === "complete"');
  const downloaded = await readDownload(browser);
  const exported = readExport(downloaded);
  assert.equal(exported.task.phase, 'complete');
  assert.ok(exported.strategy.completedAt);
  assert.deepEqual(exported.strategy.questionOrder, partial.strategy.questionOrder);
  assert.deepEqual(exported.strategy.cardOrder, partial.strategy.cardOrder);
  assert.deepEqual(exported.strategy.answers.map(answer => answer.status),
    ['skipped', 'answered', 'skipped', 'skipped']);
  assert.equal(exported.strategy.answers[1].choiceId, 'other');
  assert.equal(exported.strategy.answers[1].otherText, OTHER_TEXT);
  assert.deepEqual(exported.task.trials, taskAtModule);
  assert.deepEqual(await evaluate('taskReadback().task.completedTrials'), taskAtModule);
  assert.deepEqual(await readDownload(browser, '[data-action="export"]'), downloaded);
  await reloadAndWait('!!document.querySelector(\'[data-action="export"]\')');
  const reopened = await readDownload(browser, '[data-action="export"]');
  assert.deepEqual(reopened.strategy, exported.strategy);
  assert.deepEqual(reopened.task.trials, taskAtModule);
  return { partial, exported: downloaded, cardOrder };
}

if (require.main === module) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}

async function main() {
  const { parseArgs } = require('node:util');
  const { values, positionals } = parseArgs({
    options: { help: { type: 'boolean' }, output: { type: 'string' } },
    allowPositionals: true,
  });
  if (values.help) {
    console.log('Usage: node tests/test_tutoring_standalone_browser.cjs INDEX_HTML [--output NEW_JSON]\n' +
      'Build INDEX_HTML using scripts/build_tutoring_preview.py --output DIR.\n' +
      'Requires Node 22+ and TUTORING_CHROME set to an installed headless Chromium executable.\n' +
      'No npm dependencies, downloads, screenshots, formal policy draws or human data.\n' +
      'Fresh synthetic C3 session; local file and loopback CDP only.\n' +
      'JSON goes to stdout or a new file in an existing output directory.');
    return;
  }
  assert.equal(positionals.length, 1, 'Supply one built index.html path; see --help');
  await withChrome(positionals[0], async browser => {
    const readback = await run(browser);
    const record = {
      schema: 'tutoring_standalone_browser_check_v1',
      humanData: false,
      formalPolicyDraws: 0,
      syntheticTaskRecords: 14,
      crossBuildMigrationChecked: false,
      ...readback,
    };
    const json = JSON.stringify(record, null, 2) + '\n';
    if (values.output) {
      await require('node:fs/promises').writeFile(values.output, json, { flag: 'wx' });
    } else {
      process.stdout.write(json);
    }
  });
}

module.exports = { run };
