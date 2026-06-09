# Scripts

`generate_results.py` is the main command-line runner.

It can generate:

- Step 7 final-choice comparisons
- information-acquisition comparisons
- behavioral-profile diagnostics
- RR approximation-method comparisons
- one-dimensional sweeps across environment parameters
- candidate regime tables for near-50/50 and equal-outcome behavior
- DP sensitivity diagnostics
- Gauss-Hermite diagnostics
- simple SVG heatmaps and a run summary

Example:

```bash
python3 scripts/generate_results.py --preset smoke --sections all --output-dir results/round2_smoke
```

Scale-up example:

```bash
python3 scripts/generate_results.py \
  --preset server \
  --episodes 1200 \
  --voi-samples 500 \
  --common-observations on \
  --sections step7,sweeps,dp,gh \
  --output-dir results/round2_server
```

Generated outputs are written under `results/`, which is ignored by Git.

`run_parallel_r2.py` is the preferred runner for larger Round 2 jobs. It splits the run into section/environment or section/feature shards, runs shards in parallel, writes one stdout/stderr log per shard, and combines successful shard outputs.

Example:

```bash
python3 scripts/run_parallel_r2.py \
  --preset serious \
  --sections all \
  --max-workers 7 \
  --output-dir results/r2_parallel_serious
```

If a task fails, inspect `parallel_run_status.csv`, `parallel_summary.md`, and the corresponding file under `logs/`.
