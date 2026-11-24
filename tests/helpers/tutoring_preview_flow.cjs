/* Synthetic UI actions for the fixed four-practice, ten-main preview profile. */
'use strict';

const assert = require('node:assert/strict');

async function exposeReadback({ evaluate }) {
  await evaluate(`window.taskReadback = function () {
    for (const key of Object.keys(localStorage)) {
      const value = JSON.parse(localStorage.getItem(key));
      if (value?.payload?.completedTrials && value.payload?.schedule) {
        return { key, task: value.payload };
      }
    }
    throw new Error('Saved task state not found');
  };`);
}

async function startTask(browser) {
  const { evaluate, waitFor } = browser;
  await waitFor('!!document.querySelector("#tutoring-root [data-action]")');
  await exposeReadback(browser);
  const initial = await evaluate('taskReadback().task.phase');
  assert.equal(initial, 'instructions');
  const progress = await evaluate(`(async () => {
    const nextTurn = () => new Promise(resolve => setTimeout(resolve, 0));
    const click = async name => {
      const target = document.querySelector('#tutoring-root [data-action="' + name + '"]');
      if (!target || target.disabled) throw new Error('Unavailable task action: ' + name);
      target.click();
      await nextTurn();
    };
    for (let step = 0; step < 180; step++) {
      const { task } = taskReadback();
      const phase = task.phase;
      if (phase === 'feedback' && task.currentTrialIndex === task.schedule.length - 1) {
        return { refs: task.completedTrials.map(r => r.trialId),
          finalAtFeedback: task.current.definition.trialId };
      }
      if (phase === 'instructions') await click('beginChecks');
      else if (phase === 'displayGuide') await click('advanceDisplayGuide');
      else if (phase === 'practiceGuide') await click('beginPracticeChecks');
      else if (phase === 'break') await click('continueBreak');
      else if (phase === 'trial') await click('openStop');
      else if (phase === 'stopConfirm') await click('confirmStop');
      else if (phase === 'allocation') {
        if (task.current.budgetMillihours > 0) {
          const endpoint = document.querySelector('#tutoring-root [data-action="allocationZero"]');
          if (endpoint) await click('allocationZero');
          else {
            const input = document.querySelector('#tt-allocation');
            input.value = '45';
            input.dispatchEvent(new Event('input', { bubbles: true }));
            await nextTurn();
          }
        }
        await click('confirmAllocation');
      } else if (phase === 'feedback') await click('advance');
      else if (phase === 'comprehension' || phase === 'practiceCheck') {
        const form = document.querySelector('form[data-form="checks"]');
        TutoringTaskEngine.checkQuestions(task).forEach((question, index) => {
          const radios = Array.from(form.querySelectorAll('input[name="check' + index + '"]'));
          const input = radios.find(x => x.value === question.correct);
          if (!input) throw new Error('Correct check option missing for index ' + index);
          input.click();
        });
        form.requestSubmit();
        await nextTurn();
      } else throw new Error('Unexpected task phase: ' + phase);
    }
    throw new Error('Did not reach the final round feedback');
  })()`);
  assert.equal(progress.refs.length, 13);
  assert.equal(progress.finalAtFeedback, 'main-10');
  return progress;
}

module.exports = { startTask, exposeReadback };
