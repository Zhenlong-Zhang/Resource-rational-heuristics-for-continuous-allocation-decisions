# Scientific Reproduction Notebooks

Entry points are organized by scientific question and result family. They call
the shared implementation in `src/` and portable runners in `scripts/`; no model
or inference method is duplicated in notebook cells. Both `main` and the
maintained `20-limit` variant use this same thematic layout where compatible.

## Questions and Outputs

| Scientific question / result family | Notebook | Public producer and full settings |
| --- | --- | --- |
| Which heuristics emerge across Gaussian regimes? Final-choice/acquisition tables, behavior profiles, parameter sweeps, target candidates, DP sensitivity and Gauss-Hermite diagnostics | [Gaussian regimes](heuristic_mapping/gaussian_regimes.ipynb) | `run_parallel_experiments.py`, sections `step7,sweeps,dp,gh` plus three targeted grids; 1200 episodes, 500 VOI draws and 500 common observations/person; 32 grid chunks |
| Does active search approach true-state equal outcome? Observation invariant and focused/narrow-followup regime tables | [Active-search regimes](heuristic_mapping/active_search_regimes.ipynb) | `check_observation_streams.py` (120 episodes, 500 observations/person, correlation threshold .95); parallel runner's two active-search grids (1200/500/500, zero prior samples, 48 chunks) |
| Can a constructed manual-search baseline outperform equal split, and does RR match it? Three-policy environment/profile/candidate tables | [Constructed benchmark](heuristic_mapping/constructed_benchmark.ipynb) | Parallel runner, `active_search_diagnostic`, 972-environment `active_search_benchmark`; 1200/500/500, three manual samples/person, 81 chunks |
| How do objectives, repeated information, confirmation and non-myopic solvers differ? Objective strata, fixed-budget/sweep evidence, independent confirmation, paired solver tables and report figures | [Objective and solver diagnostics](allocation_mechanisms/objective_and_solver_diagnostics.ipynb) | `active_search_evaluation_workflow.py` -> `analyze_active_search.py` -> `generate_active_search_report.py`; all eight scientific recipes and generated inputs connected; see [settings and coverage](allocation_mechanisms/README.md) |
| When does scarcity favor lower effort-to-goal? Object gate, development selection, confirmation, lower-need metrics and non-positive Gaussian probabilities | [Scarcity](heuristic_mapping/scarcity.ipynb) | `run_scarcity_public.py --mode serious`, `scarcity.py` constants and `scarcity_frozen_settings.json`; 120 object/development episodes and 1200 confirmation episodes, unchanged stage-specific seeds |
| Is the capped finite task numerically sound and are response rules identifiable? Baseline, C3 revision, C2 recovery and finite-prior bridge | [Finite-task validation](finite_tutoring/design_validation.ipynb) | Four validation/audit runners with baseline/revised conditions and fixed stage recipes; [settings and output map](finite_tutoring/README.md) |
| How do the retained statistical methods calibrate, and what follows under declared planning assumptions? Inference examples, 28 planning scenarios, three calibration routes, failed criteria, C1 non-adoption and cohort bounds | [Calibration and power](statistical_methods/calibration_and_power.ipynb) | Historical Student/Welch 390000 attempts, known-bound 100000, empirical-Bernstein 120000, batch size 250 and unchanged per-cell seeds; [methods and output map](statistical_methods/README.md) |

## Computational Outputs

The Gaussian notebook produces `step7_final_choice_comparison.csv`,
`step7_information_acquisition_comparison.csv`, `step7_behavior_profiles.csv`,
`rr_approximation_methods_comparison.csv`, all four `sweep_*` tables,
`dp_sensitivity_analysis.csv` and `gauss_hermite_diagnostics.csv`. Each targeted
grid produces `targeted_regime_final_choice_comparison.csv`,
`targeted_regime_rr_behavior_profiles.csv`,
`targeted_regime_final_choice_candidates.csv` and
`targeted_regime_behavior_candidates.csv`. These include negative/near-zero
regimes rather than only favorable candidates.

The active-search notebook uses the same four targeted-grid outputs for its
two distinct grids, plus `observation_stream_check.json`. The constructed
benchmark produces `active_search_diagnostic_environment_summary.csv`,
`active_search_diagnostic_policy_profiles.csv` and
`active_search_diagnostic_manual_advantage_candidates.csv`. The shared parallel
runner combines shard tables at the output root and generates the associated
SVG summaries. Notebook previews read combined tables, not arbitrary shards.

The scarcity producer generates oracle rows/summary, object gate/selected anchors,
development and confirmation episode rows/summaries, confirmation selection and
run metadata. The notebook also preserves the deterministic mechanism preview
and untruncated Gaussian non-positive-probability diagnostic. An object gate
that stops later stages remains an explicit negative/ineligible result.

The allocation-mechanism notebook generates the oracle, anchors, sampling/OAT
configurations, fixed-budget comparison, discovery-selected confirmation and
matched myopic/DP results before the report. Its linked settings map records
each stage's episode/task counts and seed namespace. Full mode generates the
validated report, supporting tables and figures; smoke does not bypass its
full-count gates or force candidate selection.

## Running and Interpreting

The first code cell installs dependencies. Open from a repository checkout,
choose the stated smoke/full mode and inspect printed commands before setting
`RUN = True`. Scarcity calls its full mode `serious`. The earlier Gaussian
runners retain the numerical preset named `server` for full settings; this is
a portable preset name, not a requirement for particular hardware. Full seeds,
solver defaults, common random streams and sharding settings are unchanged.

Generated outputs belong under ignored `results/` paths; choose a fresh output
directory. Smoke checks wiring only, does not support scientific claims and may
overlap full-run samples. The statistical smoke uses full-run batch zero and
keeps its RNG shape; its alternate-sized fixture mode is not scientific evidence.
Statistical schedules additionally require Node.js, not browser or npm tooling.

The finite task has a 20-query bound and discrete positive support. Do not apply
that bound retroactively to the older continuous Gaussian comparisons or treat
their result tables as finite-task evidence. Historical statistical policies and
failed calibration remain method-labelled supporting diagnostics, not automatic
replacements for the latest primary method. Expected average utility is the
performance criterion; allocation resemblance and sampling behavior are
diagnostics. Neither synthetic recovery nor conditional power proves human
validity, fixes informative selection, or determines a final recruitment size.
