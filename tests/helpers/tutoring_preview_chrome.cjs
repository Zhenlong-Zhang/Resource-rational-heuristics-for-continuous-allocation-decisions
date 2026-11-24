/* Local-file browser harness: fresh profile, loopback CDP, no external requests. */
'use strict';

const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs/promises');
const { once } = require('node:events');
const os = require('node:os');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');
const { pathToFileURL } = require('node:url');

async function withChrome(pagePath, inspect) {
  const chrome = process.env.TUTORING_CHROME;
  assert.ok(chrome && path.isAbsolute(chrome),
    'Set TUTORING_CHROME to the absolute path of an installed headless browser');
  assert.equal(typeof WebSocket, 'function', 'Node 22+ with native WebSocket is required');
  assert.equal(typeof fetch, 'function', 'Node with native fetch is required');
  await fs.access(path.resolve(pagePath));
  await fs.access(chrome, require('node:fs').constants.X_OK);
  const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tutoring-preview-chrome-'));
  const child = spawn(chrome, ['--headless=new', '--no-first-run',
    '--disable-background-networking', '--remote-debugging-address=127.0.0.1',
    '--remote-debugging-port=0',
    '--user-data-dir=' + profile, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  const exit = once(child, 'exit');
  let socket;
  let startup = '';
  child.stderr.on('data', chunk => { startup = (startup + chunk).slice(-2500); });
  try {
    let port;
    for (let waited = 0; waited < 25000 && !port; waited += 100) {
      assert.equal(child.exitCode, null, 'Chrome exited before opening DevTools: ' + startup);
      const listener = startup.match(/DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)\//);
      if (listener) {
        port = listener[1];
        break;
      }
      try {
        const content = await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8');
        [port] = content.trim().split('\n');
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
        await delay(100);
      }
    }
    assert.ok(port, 'Chrome did not start a local debugging session: ' + startup);
    const response = await fetch('http://127.0.0.1:' + port + '/json/list');
    assert.ok(response.ok);
    const targets = await response.json();
    socket = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
    await once(socket, 'open');
    const pending = new Map();
    let seq = 0;
    socket.addEventListener('message', ({ data }) => {
      const message = JSON.parse(data);
      pending.get(message.id)?.(message);
    });
    async function call(method, params = {}) {
      const id = ++seq;
      const message = await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          pending.delete(id);
          reject(new Error('Chrome did not respond to ' + method));
        }, 10000);
        pending.set(id, result => {
          pending.delete(id);
          clearTimeout(timeout);
          resolve(result);
        });
        socket.send(JSON.stringify({ id, method, params }));
      });
      if (message.error) throw new Error(method + ': ' + message.error.message);
      return message.result;
    }
    async function evaluate(expression) {
      const result = await call('Runtime.evaluate', {
        expression, awaitPromise: true, returnByValue: true,
      });
      if (result.exceptionDetails) {
        throw new Error(result.exceptionDetails.exception?.description ||
          result.exceptionDetails.text);
      }
      return result.result.value;
    }
    async function waitFor(expression) {
      for (let attempt = 0; attempt < 70; attempt++) {
        try {
          if (await evaluate(expression)) return;
        } catch (error) {
          if (!/Cannot find context|Execution context was destroyed/.test(error.message)) throw error;
        }
        await delay(50);
      }
      throw new Error('Browser condition failed: ' + expression);
    }
    await call('Page.enable');
    await call('Network.enable');
    await call('Network.setBlockedURLs', {
      urls: ['http://*', 'https://*', 'ws://*', 'wss://*', 'ftp://*'],
    });
    const url = pathToFileURL(path.resolve(pagePath)).href;
    await call('Page.navigate', { url });
    await waitFor('location.href === ' + JSON.stringify(url) +
      ' && document.readyState === "complete"');
    await inspect({ evaluate, waitFor, call, url });
  } finally {
    socket?.close();
    child.kill();
    await exit;
    await fs.rm(profile, { recursive: true, force: true, maxRetries: 6, retryDelay: 125 });
  }
}

module.exports = { withChrome };
