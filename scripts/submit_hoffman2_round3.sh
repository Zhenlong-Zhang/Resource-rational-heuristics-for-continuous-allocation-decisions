#!/usr/bin/env bash
set -euo pipefail

# Submit Round 3 Hoffman2 jobs.
#
# Defaults:
# - run the observation-stream sanity diagnostic on a compute node
# - run the active-search true-equal-outcome targeted grid at server scale
# - keep the 10x approximation-method comparison available but off by default
#
# Run this script from the Hoffman2 clone after pulling the latest GitHub commit.

export LC_ALL=C
export LANG=C

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
SLOTS="${SLOTS:-8}"
H_DATA="${H_DATA:-8G}"

RUN_OBSERVATION_CHECK="${RUN_OBSERVATION_CHECK:-1}"
RUN_ACTIVE_SEARCH="${RUN_ACTIVE_SEARCH:-1}"
RUN_METHOD_COMPARISON_10X="${RUN_METHOD_COMPARISON_10X:-0}"

ACTIVE_CHUNKS="${ACTIVE_CHUNKS:-81}"
ACTIVE_H_RT="${ACTIVE_H_RT:-24:00:00}"
ACTIVE_OUTPUT_DIR="${ACTIVE_OUTPUT_DIR:-results/r3_active_search_equal_outcome_server}"

METHOD_H_RT="${METHOD_H_RT:-24:00:00}"
METHOD_OUTPUT_DIR="${METHOD_OUTPUT_DIR:-results/r3_method_comparison_10x_server}"

CHECK_H_RT="${CHECK_H_RT:-02:00:00}"
CHECK_OUTPUT_JSON="${CHECK_OUTPUT_JSON:-results/r3_observation_stream_check_server.json}"

mkdir -p "${PROJECT_ROOT}/jobs" "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"

submit_job_file() {
  local job_file="$1"
  chmod +x "${job_file}"
  echo "Submitting ${job_file}"
  (cd "${PROJECT_ROOT}" && env LC_ALL=C LANG=C qsub "${job_file}")
}

if [[ "${RUN_OBSERVATION_CHECK}" == "1" ]]; then
  job_file="${PROJECT_ROOT}/jobs/r3_observation_stream_check.job"
  cat > "${job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_observation_stream_check.\$JOB_ID.log
#$ -j y
#$ -l h_rt=${CHECK_H_RT},h_data=${H_DATA}
#$ -N rr_r3_obs

export LC_ALL=C
export LANG=C
PYTHON=${PYTHON_BIN}
\$PYTHON --version

\$PYTHON scripts/check_observation_streams.py \\
  --episodes 120 \\
  --observations-per-person 500 \\
  --min-correlation 0.95 \\
  --output-json ${CHECK_OUTPUT_JSON}
EOF
  submit_job_file "${job_file}"
fi

if [[ "${RUN_ACTIVE_SEARCH}" == "1" ]]; then
  job_file="${PROJECT_ROOT}/jobs/r3_active_search_equal_outcome_server.job"
  cat > "${job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_active_search_equal_outcome_server.\$JOB_ID.log
#$ -j y
#$ -pe shared ${SLOTS}
#$ -l h_rt=${ACTIVE_H_RT},h_data=${H_DATA}
#$ -N rr_r3_active

export LC_ALL=C
export LANG=C
PYTHON=${PYTHON_BIN}
\$PYTHON --version

\$PYTHON scripts/run_parallel_r2.py \\
  --preset server \\
  --sections regime_grid \\
  --regime-grid active_search_equal_outcome_focused \\
  --regime-grid-chunks ${ACTIVE_CHUNKS} \\
  --episodes 1200 \\
  --voi-samples 500 \\
  --common-observations on \\
  --max-workers \${NSLOTS:-${SLOTS}} \\
  --output-dir ${ACTIVE_OUTPUT_DIR}
EOF
  submit_job_file "${job_file}"
fi

if [[ "${RUN_METHOD_COMPARISON_10X}" == "1" ]]; then
  job_file="${PROJECT_ROOT}/jobs/r3_method_comparison_10x_server.job"
  cat > "${job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_method_comparison_10x_server.\$JOB_ID.log
#$ -j y
#$ -pe shared ${SLOTS}
#$ -l h_rt=${METHOD_H_RT},h_data=${H_DATA}
#$ -N rr_r3_methods

export LC_ALL=C
export LANG=C
PYTHON=${PYTHON_BIN}
\$PYTHON --version

\$PYTHON scripts/run_parallel_r2.py \\
  --preset server \\
  --sections step7 \\
  --episodes 1200 \\
  --voi-samples 500 \\
  --common-observations on \\
  --max-workers \${NSLOTS:-${SLOTS}} \\
  --output-dir ${METHOD_OUTPUT_DIR}
EOF
  submit_job_file "${job_file}"
fi

echo
echo "Submitted requested Round 3 Hoffman2 jobs. Check status with:"
echo "  qstat -u \${USER}"
echo
echo "Useful result paths:"
echo "  ${CHECK_OUTPUT_JSON}"
echo "  ${ACTIVE_OUTPUT_DIR}"
echo "  ${METHOD_OUTPUT_DIR}  # only if RUN_METHOD_COMPARISON_10X=1"
