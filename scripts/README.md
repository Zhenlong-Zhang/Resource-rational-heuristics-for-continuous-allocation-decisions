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
