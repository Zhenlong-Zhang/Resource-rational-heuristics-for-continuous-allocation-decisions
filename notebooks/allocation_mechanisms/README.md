# Allocation Objectives and Solver Diagnostics

[objective_and_solver_diagnostics.ipynb](objective_and_solver_diagnostics.ipynb)
connects the full manifest workflow to the validated report reader. The first
code cell installs dependencies and computation defaults to `RUN = False`.

## Full Scientific Recipes

Every full-stage command is declared in the notebook; generated configurations
are produced before their consumers. Counts below are per environment, not total
episodes. Distinct stage seeds and task boundaries are preserved.

| Stage | Family | Episodes/environment | Episodes/task | Seed namespace offset |
| --- | --- | --- | --- | --- |
| Objective oracle | `oracle` | 1200 | 100 | 0 |
| Discovery | `six_sample` | 120 | 5 | 0 |
| Sampling-cost sweep | `custom_rr` | 120 | 5 | 0 |
| One-factor-at-a-time sweep | `custom_rr` | 120 | 5 | 0 |
| Fixed sampling budgets | `fixed_budget` | 1200 | 5 | 300000 |
| Independent confirmation | `custom_rr` | 1200 | 10 | 1000000 |
| Held-out myopic comparator | `custom_rr` | 1200 | 10 | 900000 |
| Held-out DP | `custom_rr` | 1200 | 1200 | 900000 |

All full stages retain 500 observation draws, allocation tolerance .05 and an
oracle grid of 4,001. Fixed budgets are `0,2,4,6,8,10,12`. The solver pair uses
the same three held-out configurations and episode seeds; DP maximum samples
are 10, mean-grid size 50/radius 3 SD, with seven Gauss-Hermite observation
branches. The configurations retain their own integration methods and seeds;
the notebook does not overwrite them with a common replacement model.

## Producer and Reader Coverage

| Result / report input | Public producer-to-reader chain |
| --- | --- |
| Objective and oracle strata | `active_search_evaluation_workflow.py` oracle create/run-task/collect -> `analyze_active_search.py oracle` -> sign summary, anchors, objective JSON and sampling/OAT configuration files |
| Repeated-information value and sweep figures | Generated anchors/sampling/OAT configurations -> workflow fixed_budget/custom_rr create/run-task/collect -> `analyze_active_search.py formal-summaries` -> evidence CSV, formal sweep CSV and SVGs |
| Discovery and independent confirmation | Workflow six_sample -> `analyze_active_search.py select-confirmation` -> selected configuration -> workflow custom_rr with independent seed -> confirmation summary |
| Held-out solver comparison | `configs/solver_heldout_configs.json` -> matched custom_rr myopic/DP runs with common episode seeds -> `analyze_active_search.py solver-comparison` -> paired CSV/JSON and SVG |
| Validated report and confirmation figure | All six newly generated oracle/objective/formal/discovery/confirmation/solver directories -> `generate_active_search_report.py` -> report, supporting tables and figures |

The report validates oracle 129,600 episodes / 1,296 tasks / 108 summaries;
discovery 38,880 / 7,776 / 324; confirmation 14,400 / 1,440 / 12. It requires
different discovery and confirmation seed namespaces and three held-out solver
rows. No cached result folder or private manifest is consumed. The oracle
analyzer generates four anchors and both sweep configuration files; the discovery
analyzer selects up to twelve confirmation configurations with its fixed rule.

## Smoke and Dry Run

With `RUN = False`, commands print without reading missing generated manifests
or configurations. A representative task-zero command is shown for each family;
actual execution reads its new manifest and executes every task before strict
collection. Outputs are under `results/allocation_mechanisms/<mode>/`.

Smoke follows the same producer/reader chain with five episodes/environment,
at most five/task and 24 VOI draws, retaining the seed namespaces, environment
definitions, solver geometry and selection rules. It may fail selection at low
precision; no fallback anchor or confirmation candidate is injected. The final
report is full-only because its required counts are scientific checks, not
optional validation. Smoke is wiring evidence only, not an independent result.

Expected average utility is the performance criterion. Joint point-estimate
success, lower-bound confirmation and solver utility gains are separate claims;
retain failed confirmation, baselines and mixed solver findings.
