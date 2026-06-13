#!/usr/bin/env bash
set -euo pipefail

# Submit the Round 2 Hoffman2 jobs used for targeted regime search and
# 1200-episode confirmation. Run this script from the Hoffman2 clone.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
SLOTS="${SLOTS:-8}"
H_RT="${H_RT:-24:00:00}"
H_DATA="${H_DATA:-8G}"

mkdir -p "${PROJECT_ROOT}/jobs" "${PROJECT_ROOT}/logs"

submit_regime_job() {
  local job_key="$1"
  local job_name="$2"
  local log_prefix="$3"
  local preset="$4"
  local grid="$5"
  local chunks="$6"
  local output_dir="$7"
  local job_file="${PROJECT_ROOT}/jobs/${job_key}.job"

  cat > "${job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/${log_prefix}.\$JOB_ID.log
#$ -j y
#$ -pe shared ${SLOTS}
#$ -l h_rt=${H_RT},h_data=${H_DATA}
#$ -N ${job_name}

export LC_ALL=C
export LANG=C
PYTHON=${PYTHON_BIN}
\$PYTHON --version

\$PYTHON scripts/run_parallel_r2.py \\
  --preset ${preset} \\
  --sections regime_grid \\
  --regime-grid ${grid} \\
  --regime-grid-chunks ${chunks} \\
  --common-observations on \\
  --max-workers \${NSLOTS:-${SLOTS}} \\
  --output-dir ${output_dir}
EOF

  chmod +x "${job_file}"
  echo "Submitting ${job_file}"
  (cd "${PROJECT_ROOT}" && qsub "${job_file}")
}

submit_regime_job \
  "run_distinct_serious" \
  "rr_distinct_serious" \
  "distinct_serious" \
  "serious" \
  "equal_outcome_distinct_focused" \
  "32" \
  "results/r2_equal_outcome_distinct_serious"

submit_regime_job \
  "run_full_near_serious" \
  "rr_full_near" \
  "full_near_serious" \
  "serious" \
  "near_50_50" \
  "64" \
  "results/r2_full_near_50_50_serious"

submit_regime_job \
  "run_full_equal_serious" \
  "rr_full_equal" \
  "full_equal_serious" \
  "serious" \
  "equal_outcome" \
  "64" \
  "results/r2_full_equal_outcome_serious"

submit_regime_job \
  "run_confirm_near_1200" \
  "rr_near_1200" \
  "confirm_near_1200" \
  "server" \
  "near_50_50_focused" \
  "36" \
  "results/r2_confirm_near_50_50_focused_1200"

submit_regime_job \
  "run_confirm_equal_1200" \
  "rr_equal_1200" \
  "confirm_equal_1200" \
  "server" \
  "equal_outcome_focused" \
  "48" \
  "results/r2_confirm_equal_outcome_focused_1200"

submit_regime_job \
  "run_confirm_distinct_1200" \
  "rr_dist_1200" \
  "confirm_distinct_1200" \
  "server" \
  "equal_outcome_distinct_focused" \
  "64" \
  "results/r2_confirm_equal_outcome_distinct_1200"

echo
echo "Submitted Round 2 Hoffman2 jobs. Check status with:"
echo "  qstat -u \${USER}"
