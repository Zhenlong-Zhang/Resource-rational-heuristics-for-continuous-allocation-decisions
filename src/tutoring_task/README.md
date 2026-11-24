# Standalone Tutoring Preview

Build the browser preview from a repository checkout with Python 3.10 or later:

```bash
python scripts/build_tutoring_preview.py --output results/tutoring_preview
python -m http.server 8000 --bind 127.0.0.1 --directory results/tutoring_preview
```

Open `http://localhost:8000`. Use a fresh output directory for each build.
The build uses only Python's standard library and makes no network requests or
policy calculations. The [preview notebook](../../notebooks/finite_tutoring/standalone_preview.ipynb)
provides the same source-calling entry.

The page contains four practice tasks, ten main tasks and a post-task block:
wisdom, morality and fairness in shuffled order, followed by self-report.
Six fixed strategy cards retain their session order; Other is always last.
The condition selector is for review, not participant randomization. This is
an accepted software prototype, not the finalized or deployed human study.

## Source and Input

- `engine.js` implements discrete Bayes updates, budgets, schedules and task state.
- `ui.js` and `ui.css` render the task and handle local state transitions.
- `preview.js` supplies review controls and connects the strategy questionnaire.
- `local_strategy.js` stores and validates strategy answers and exports.
- `configs/tutoring_conditions.json` supplies the fixed numerical recipes;
  `configs/tutoring_preview.json` pins that input and adds display/feedback settings.

The retained base input contains historical 30-main-task/cap20 fields. The
fixed browser profile explicitly resolves **10 main tasks and budget-only
question availability**, with the unscaled one-minute C1 centre gap. Those
base fields are not the effective preview dose or question limit. The engine
validates both the base recipe and effective profile.

Similarly, the inherited display metadata records the older mean/SD/RMS
interface. This fixed browser profile actually shows one-decimal guesses,
finite 95% need intervals and the shared 80% report-accuracy indicator.

The build writes `index.html`, `resolved_design.json` and `build_manifest.json`.
The manifest binds source, input and output hashes. Public input metadata
differs from the historical proposal envelope, so the design fingerprint is
new. Numerical fields, seeded schedules and display semantics are preserved;
old task sessions are not automatically migrated into a differently identified
build. Legacy strategy-answer compatibility is separate from task migration.

## Response State

`local_strategy.js` is a standalone, dependency-free response-state and export
reader module. In Node, load it with `require('./src/tutoring_task/local_strategy.js')`;
in a browser it exposes `TutoringLocalStrategy`.

- `open(...)` accepts a storage adapter, explicit task identity and random/clock
  providers. It creates or resumes stable card/question order and exposes
  `get`, `answer` and `skip` operations.
- Other responses require non-whitespace text. Committed text, including Unicode
  and markup, is preserved without a project-imposed text-length cap. Actual
  storage limits still apply: failed writes leave the committed state unchanged.
- `readExport(...)` validates the response envelope against its task identity.
  Existing serialization identifiers and legacy blank Other answers remain
  readable; new answers use the current response policy. Pending, skipped and
  answered states are distinct. Recovery payloads are not validated answers.

## Software Checks

Run the packaging and focused state checks with Python 3.10+ and Node 22.14.0:

```bash
python -m unittest discover -s tests -p test_tutoring_preview_build.py
python scripts/build_tutoring_preview.py --output results/tutoring_preview_test
node --test tests/test_tutoring_local_strategy.cjs tests/test_tutoring_strategy_other.cjs
TUTORING_PREVIEW_DIR=results/tutoring_preview_test node --test tests/test_tutoring_preview_engine.cjs
```

The browser check uses a locally installed Chromium headless executable and
Node's built-in DevTools client; no npm package is needed. Its command is
documented in the test guide. It completes the synthetic C3 flow and checks
required Other text, Skip, failed-save retry, reload and complete JSON export.
It makes no screenshots and produces no human or policy-evaluation data.

Responses stay in browser storage until manually exported. Storage capacity is
finite; failed writes are reported rather than truncated. Recovery covers
committed answers, not unsent textarea drafts. Formal platform persistence,
participant comprehension, study timing and human-data collection are outside
this preview's verification scope.
