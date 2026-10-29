# Scientific Reproduction Notebooks

Entry points are organized by scientific question and result family. They call
the shared implementation in `src/` and portable runners in `scripts/`; no model
or inference method is duplicated in notebook cells.

Use `main` for the maintained reproduction workflows. Retained historical
analyses are identified by their scientific scope and fixed source versions,
not by separate maintained branches. See the [scientific scope](../README.md#scientific-scope)
for the archived capped release. Do not substitute a newer model or estimator
when reproducing a historical result.

## Start Here

1. Follow the [installation guide](../README.md#installation), then open a notebook below from the checkout. Native calibration and paired Gamma use the [reference environment](statistical_methods/README.md#native-state-calibration).
2. Run the dependency-install cell. Inspect the notebook's settings and linked [configurations](../configs/README.md), choose smoke/full mode and an output directory, and preview the printed commands with execution disabled.
3. Enable `RUN` only when ready; the paired Gamma section has its own `RUN_PAIRED` switch. Use the notebook's readers and the output maps below to inspect all results, including failures and unavailable endpoints.

Outputs are generated under ignored `results/` paths, not bundled with the
notebooks. Smoke checks wiring only. Use fresh destinations unless a route
explicitly documents compatible resume.

## Questions and Outputs

| Scientific question / result family | Notebook | Public producer and full settings |
| --- | --- | --- |
| Which heuristics emerge across Gaussian regimes? Final-choice/acquisition tables, behavior profiles, parameter sweeps, target candidates, DP sensitivity and Gauss-Hermite diagnostics | [Gaussian regimes](heuristic_mapping/gaussian_regimes.ipynb) | `run_parallel_experiments.py`, sections `step7,sweeps,dp,gh` plus three targeted grids; 1200 episodes, 500 VOI draws and 500 common observations/person; 32 grid chunks |
| What did the historical Gaussian broad and focused grids compute? Three 120-episode grids and three 1200-episode confirmations | [Historical Gaussian reference](heuristic_mapping/historical_gaussian_grids.ipynb) | `reproduce_gaussian_reference.py`; pinned source commit, serious preset, 500 VOI draws, 200 requested common observations; original grid-specific chunk counts |
| Does active search approach true-state equal outcome? Observation invariant and focused/narrow-followup regime tables | [Active-search regimes](heuristic_mapping/active_search_regimes.ipynb) | `check_observation_streams.py` (120 episodes, 500 observations/person, correlation threshold .95); parallel runner's two active-search grids (1200/500/500, zero prior samples, 48 chunks) |
| Can a constructed manual-search baseline outperform equal split, and does RR match it? Three-policy environment/profile/candidate tables | [Constructed benchmark](heuristic_mapping/constructed_benchmark.ipynb) | Parallel runner, `active_search_diagnostic`, 972-environment `active_search_benchmark`; 1200/500/500, three manual samples/person, 81 chunks |
| How do objectives, repeated information, confirmation and non-myopic solvers differ? Objective strata, fixed-budget/sweep evidence, independent confirmation, paired solver tables and report figures | [Objective and solver diagnostics](allocation_mechanisms/objective_and_solver_diagnostics.ipynb) | `active_search_evaluation_workflow.py` -> `analyze_active_search.py` -> `generate_active_search_report.py`; all eight scientific recipes and generated inputs connected; see [settings and coverage](allocation_mechanisms/README.md) |
| When does scarcity favor lower effort-to-goal? Object gate, development selection, confirmation, lower-need metrics and non-positive Gaussian probabilities | [Scarcity](heuristic_mapping/scarcity.ipynb) | `run_scarcity_public.py --mode serious`, `scarcity.py` constants and `scarcity_frozen_settings.json`; 120 object/development episodes and 1200 confirmation episodes, unchanged stage-specific seeds |
| Is the capped finite task numerically sound and are response rules identifiable? Baseline, C3 revision, C2 recovery and finite-prior bridge | [Finite-task validation](finite_tutoring/design_validation.ipynb) | Four validation/audit runners with baseline/revised conditions and fixed stage recipes; [settings and output map](finite_tutoring/README.md) |
| Does a query cap bind, how do terminal time prices affect choice, and what numerical bounds hold? | [Horizons, terminal time and bounds](finite_tutoring/horizon_terminal_bounds.ipynb) | `reproduce_finite_tutoring.py`: 400 paired episodes per each of six templates (2400 pairs); 36 initial timing states; conditional sensitivity/bounds at the generated old terminals; six zero-terminal-cost root refinements |
| How does repeated selected-macro questioning respond to terminal time prices? Full report branches, person-level behavior, utility and hypothesis contrasts | [Cost-aware computation](finite_tutoring/cost_aware_computation.ipynb) | Selected-trial runner plus `reproduce_cost_aware_study.py prepare/run/aggregate`; original 450 roots, 18 scenarios, 51,840 model persons and 518,400 histories; 90 performance and 2,088 Q/non-Q aggregate rows |
| What changes when information requests and termination both have personal-time costs? | [Personal-time reference and sensitivity](finite_tutoring/personal_time_reference.ipynb) | Native input generation; 450 expected endpoints, 100 original-root request margins and four equal-outcome targets; separate six-price comparison with 1,860 expected endpoints and 840 realized histories, independent trajectory checks and equal-schedule summaries |
| How do the retained statistical methods calibrate, and what follows under declared planning assumptions? Inference examples, 28 planning scenarios, three calibration routes, failed criteria, C1 identifiability limits, cohort bounds and same-input Gamma attribution | [Calibration and power](statistical_methods/calibration_and_power.ipynb) | Historical Student/Welch 390000 attempts, known-bound 100000, empirical-Bernstein 120000; supplementary fixed500 paired producer and independent reader; fixed250 batch shape and unchanged seeds; [methods and output map](statistical_methods/README.md) |
| How are expected-GPA resemblance, nominal person-level contrasts and conditional sample sizes defined? | [Expected GPA and nominal inference](statistical_methods/expected_gpa_and_nominal_inference.ipynb) | `reproduce_expected_gpa.py`: 27 initial targets x three rules, optional synthetic raw-history inference and 24 conditional sample-size cases; not full native-state calibration |
| How do the budget-only nominal methods calibrate on native finite states? Source schedules, lookup maps, independent exact truths, calibration batches and count readback | [Native-state calibration](statistical_methods/native_state_calibration.ipynb) | `reproduce_native_calibration.py prepare/run/summarize`; 40 subcases x 40 batches x 250 attempts, pinned numerical environment; [settings](../configs/README.md#statistical-settings) |
| How does the refined mixed-effects procedure behave under declared nulls? | [Mixed-model null validation](statistical_methods/mixed_model_null_validation.ipynb) | Original 12 cases x 2,000 inputs; baseline, strict restart and PORT consistency checks; independent fits/readback and full versus available denominators |
| What do paired coverage and recruitment/effect assumptions imply for the candidate analysis? | [Paired coverage and conditional power](statistical_methods/paired_coverage_and_conditional_power.ipynb) | Fixed 10,000 paired CR2/known-covariance GLS inputs and 40 x 2,000 direct-fit power inputs; original legacy prefixes, actual shifts, whole-person retention, independent reductions and Monte Carlo bounds |
| How do the retained baseline and cluster-robust methods compare? | [Inference comparisons](statistical_methods/inference_comparisons.ipynb) | Separately identified 40,000-input baseline, 480 saved-fit diagnostic and 26,000-input fresh validation; original estimators, rosters, guards and negative findings, not relabelled paired-validation results |
| How are candidate magnitude p-values combined, and how many Monte Carlo replicates meet the fixed precision rule? | [Magnitude composition and precision](statistical_methods/magnitude_composition_and_precision.ipynb) | Supplied-p-value API with four fixed comparisons; exhaustive CP planning and independent R enumeration. Not constituent-test calibration, magnitude power or human sample-size selection |
| What explains the saved target's coverage deficit, and do eight named contrasts agree on one saved fit? | [Saved coverage and contrasts](statistical_methods/saved_coverage_and_contrasts.ipynb) | Saved-only input binding, exact 2,000-member CI diagnosis, single-fit eight-L computation and independent readers; optional existing public input producer, separately disabled. Descriptive diagnosis and one-object agreement, not repaired coverage or population validity |
| How do magnitude tests and interval methods behave under their fixed validation and power designs? | [Magnitude validation and conditional power](statistical_methods/magnitude_validation_and_power.ipynb) | Original magnitude and joint-Gaussian comparisons, saved-fit power, known-covariance diagnostics and paired CR2/Kenward-Roger coverage. Public input/fit production and independent reduction; not human-method adoption or sample-size selection |

## Computational Outputs

[Native-state marginal calibration](statistical_methods/native_state_calibration.ipynb)
provides the complete synthetic budget-only chain: source states, finite lookup,
independent rational truths, fixed 40x40x250 calibration, and independently
recomputed `summary.json`/`targets.csv`. Smoke runs three fixed 250-attempt batches
and remains provisional. Full reference dispositions are 89 PASS, 22 UNAVAILABLE,
7 DIAGNOSTIC_ONLY; this does not repair unmatched historical selection failures
or establish latest cap20 compatibility.

The historical reference is a separate identified implementation, not a switch
back to an obsolete solver in the current package. It regenerates the historical
comparison/profile/candidate tables and figures under `tables/`, with the source
identity and execution status in `reference_recipe.json`. A full Git clone or
an explicit fetch of the documented commit is required. The current Gaussian
notebook is a current-model analysis; matching episode counts alone does not
make it a reproduction of historical tables.

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

The finite horizon notebook regenerates paired histories before the two terminal
consumers. Its initial/root analyses do not need a cached panel. It provides raw
scenario records, paired summaries, macro counts and numerical bound witnesses;
`reproduce_finite_auxiliary.py` adds the 45 ENGAGEMENT schedules, JavaScript
parity, 6804-case geometry, complete initial/terminal descriptive summaries and
compact raw versus symmetry-derived bounds. This is not a new cost-aware
adaptive-policy calculation or a validation of budget-only results under cap20.

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
