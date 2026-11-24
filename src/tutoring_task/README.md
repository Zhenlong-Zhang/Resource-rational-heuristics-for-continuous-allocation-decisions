# Strategy Response State

`local_strategy.js` is a standalone, dependency-free response-state and export
reader module. In Node, load it with `require('./src/tutoring_task/local_strategy.js')`;
in a browser it exposes `TutoringLocalStrategy`. It does not build or deploy an
experiment, generate task inputs, or provide the participant UI.

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

Run the focused software checks with Node 22.14.0:

```bash
node --test tests/test_tutoring_local_strategy.cjs tests/test_tutoring_strategy_other.cjs
```

These tests use in-memory storage. They do not establish participant-study
deployment, browser/tenant storage capacity or recovery of uncommitted text.
The full standalone preview and formal study integration are not part of this
module-only reproduction surface.
