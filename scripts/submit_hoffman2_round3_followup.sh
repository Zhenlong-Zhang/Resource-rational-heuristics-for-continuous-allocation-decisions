#!/usr/bin/env bash
set -euo pipefail

# Submit Round 3 follow-up Hoffman2 jobs.
#
# This script launches two independent workflows:
# 1. A high-parallelism array job for the narrow active-search true-outcome grid.
# 2. A higher-episode Step 7 approximation-method comparison.
#
# Run from the Hoffman2 clone after pulling the latest GitHub commit.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"

GRID="${GRID:-active_search_equal_outcome_narrow_followup}"
ACTIVE_OUTPUT_DIR="${ACTIVE_OUTPUT_DIR:-results/r3_active_search_equal_outcome_narrow_followup_1200ep}"
ACTIVE_EPISODES="${ACTIVE_EPISODES:-1200}"
VOI_SAMPLES="${VOI_SAMPLES:-500}"
OBSERVATIONS_PER_PERSON="${OBSERVATIONS_PER_PERSON:-500}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-160}"
TASK_H_RT="${TASK_H_RT:-12:00:00}"
TASK_H_DATA="${TASK_H_DATA:-2G}"

COMBINE_SLOTS="${COMBINE_SLOTS:-12}"
COMBINE_H_RT="${COMBINE_H_RT:-08:00:00}"
COMBINE_H_DATA="${COMBINE_H_DATA:-2G}"

APPROX_OUTPUT_DIR="${APPROX_OUTPUT_DIR:-results/r3_approximation_methods_1200ep}"
APPROX_EPISODES="${APPROX_EPISODES:-1200}"
APPROX_SLOTS="${APPROX_SLOTS:-12}"
APPROX_H_RT="${APPROX_H_RT:-48:00:00}"
APPROX_H_DATA="${APPROX_H_DATA:-3G}"
BLINKERED_SAMPLES="${BLINKERED_SAMPLES:-250}"

PACKAGE_H_RT="${PACKAGE_H_RT:-02:00:00}"
PACKAGE_H_DATA="${PACKAGE_H_DATA:-2G}"

SUBMIT_ACTIVE="${SUBMIT_ACTIVE:-1}"
SUBMIT_APPROX="${SUBMIT_APPROX:-1}"

mkdir -p "${PROJECT_ROOT}/jobs" "${PROJECT_ROOT}/logs"

grid_tasks="$(
  cd "${PROJECT_ROOT}"
  "${PYTHON_BIN}" - <<PY
from src.experiments.sweeps import build_targeted_regime_grid_configs
print(len(build_targeted_regime_grid_configs(["${GRID}"])))
PY
)"

if [[ "${grid_tasks}" -le 0 ]]; then
  echo "No grid tasks found for ${GRID}" >&2
  exit 1
fi

echo "Round 3 follow-up settings"
echo "  grid: ${GRID}"
echo "  grid tasks: ${grid_tasks}"
echo "  active output: ${ACTIVE_OUTPUT_DIR}"
echo "  approximation output: ${APPROX_OUTPUT_DIR}"
echo "  active episodes: ${ACTIVE_EPISODES}"
echo "  approximation episodes: ${APPROX_EPISODES}"
echo

active_package_job_id=""
if [[ "${SUBMIT_ACTIVE}" == "1" ]]; then
  mkdir -p "${PROJECT_ROOT}/${ACTIVE_OUTPUT_DIR}"
  array_job_file="${PROJECT_ROOT}/jobs/r3_active_search_narrow_followup_array.job"
  cat > "${array_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_active_search_narrow_followup.\$JOB_ID.\$TASK_ID.log
#$ -j y
#$ -t 1-${grid_tasks}
#$ -tc ${MAX_CONCURRENT_TASKS}
#$ -l h_rt=${TASK_H_RT},h_data=${TASK_H_DATA}
#$ -N rr_r3_nf_arr

export LC_ALL=C
export LANG=C
set -euo pipefail

PYTHON=${PYTHON_BIN}
TASK_INDEX=\$((SGE_TASK_ID - 1))
SHARD=\$(printf "${GRID}_chunk%04d_of${grid_tasks}" "\${TASK_INDEX}")
TASK_OUTPUT_DIR="${ACTIVE_OUTPUT_DIR}/tasks/regime_grid/\${SHARD}"

mkdir -p "\${TASK_OUTPUT_DIR}"
echo "Task \${SGE_TASK_ID}/${grid_tasks}; chunk index \${TASK_INDEX}; grid ${GRID}"
\$PYTHON --version
\$PYTHON scripts/generate_results.py \\
  --preset server \\
  --sections regime_grid \\
  --output-dir "\${TASK_OUTPUT_DIR}" \\
  --regime-grid "${GRID}" \\
  --regime-grid-chunk-index "\${TASK_INDEX}" \\
  --regime-grid-chunks "${grid_tasks}" \\
  --episodes "${ACTIVE_EPISODES}" \\
  --voi-samples "${VOI_SAMPLES}" \\
  --common-observations on \\
  --observations-per-person "${OBSERVATIONS_PER_PERSON}" \\
  --gauss-hermite-order 15 \\
  --dp-max-samples-values 2,4,6,10 \\
  --dp-mean-grid-sizes 7,11,21,50 \\
  --dp-observation-branches 3,5
EOF
  chmod +x "${array_job_file}"

  echo "Submitting active-search array job ${array_job_file}"
  array_submit_output=$(cd "${PROJECT_ROOT}" && env LC_ALL=C LANG=C qsub "${array_job_file}")
  echo "${array_submit_output}"
  array_job_id=$(echo "${array_submit_output}" | awk '/Your job-array/ {print $3} /Your job/ {print $3}' | tr -d '"' | sed 's/[.:].*$//')
  if [[ -z "${array_job_id}" ]]; then
    echo "Could not parse active-search array job id" >&2
    exit 1
  fi

  combine_job_file="${PROJECT_ROOT}/jobs/r3_active_search_narrow_followup_combine.job"
  cat > "${combine_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_active_search_narrow_followup_combine.\$JOB_ID.log
#$ -j y
#$ -pe shared ${COMBINE_SLOTS}
#$ -l h_rt=${COMBINE_H_RT},h_data=${COMBINE_H_DATA}
#$ -N rr_r3_nf_comb

export LC_ALL=C
export LANG=C
set -euo pipefail

PYTHON=${PYTHON_BIN}
\$PYTHON --version
\$PYTHON scripts/run_parallel_r2.py \\
  --preset server \\
  --sections regime_grid \\
  --regime-grid "${GRID}" \\
  --regime-grid-chunks "${grid_tasks}" \\
  --episodes "${ACTIVE_EPISODES}" \\
  --voi-samples "${VOI_SAMPLES}" \\
  --common-observations on \\
  --observations-per-person "${OBSERVATIONS_PER_PERSON}" \\
  --resume \\
  --max-workers \${NSLOTS:-${COMBINE_SLOTS}} \\
  --output-dir "${ACTIVE_OUTPUT_DIR}"
EOF
  chmod +x "${combine_job_file}"

  echo "Submitting active-search combine job ${combine_job_file}"
  combine_submit_output=$(cd "${PROJECT_ROOT}" && env LC_ALL=C LANG=C qsub -hold_jid "${array_job_id}" "${combine_job_file}")
  echo "${combine_submit_output}"
  combine_job_id=$(echo "${combine_submit_output}" | awk '/Your job/ {print $3}' | tr -d '"')
  if [[ -z "${combine_job_id}" ]]; then
    echo "Could not parse active-search combine job id" >&2
    exit 1
  fi

  package_job_file="${PROJECT_ROOT}/jobs/r3_active_search_narrow_followup_package.job"
  cat > "${package_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_active_search_narrow_followup_package.\$JOB_ID.log
#$ -j y
#$ -l h_rt=${PACKAGE_H_RT},h_data=${PACKAGE_H_DATA}
#$ -N rr_r3_nf_pkg

export LC_ALL=C
export LANG=C
set -euo pipefail

date
tar -czf "${ACTIVE_OUTPUT_DIR}.tar.gz" "${ACTIVE_OUTPUT_DIR}" \\
  logs/r3_active_search_narrow_followup.*.log \\
  logs/r3_active_search_narrow_followup_combine.*.log
ls -lh "${ACTIVE_OUTPUT_DIR}.tar.gz"
date
EOF
  chmod +x "${package_job_file}"

  echo "Submitting active-search package job ${package_job_file}"
  package_submit_output=$(cd "${PROJECT_ROOT}" && env LC_ALL=C LANG=C qsub -hold_jid "${combine_job_id}" "${package_job_file}")
  echo "${package_submit_output}"
  active_package_job_id=$(echo "${package_submit_output}" | awk '/Your job/ {print $3}' | tr -d '"')
fi

if [[ "${SUBMIT_APPROX}" == "1" ]]; then
  mkdir -p "${PROJECT_ROOT}/${APPROX_OUTPUT_DIR}"
  approx_job_file="${PROJECT_ROOT}/jobs/r3_approximation_methods_1200ep.job"
  cat > "${approx_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_approximation_methods_1200ep.\$JOB_ID.log
#$ -j y
#$ -pe shared ${APPROX_SLOTS}
#$ -l h_rt=${APPROX_H_RT},h_data=${APPROX_H_DATA}
#$ -N rr_r3_methods

export LC_ALL=C
export LANG=C
set -euo pipefail

PYTHON=${PYTHON_BIN}
\$PYTHON --version
\$PYTHON scripts/run_parallel_r2.py \\
  --preset server \\
  --sections step7 \\
  --episodes "${APPROX_EPISODES}" \\
  --voi-samples "${VOI_SAMPLES}" \\
  --blinkered-samples "${BLINKERED_SAMPLES}" \\
  --common-observations on \\
  --observations-per-person "${OBSERVATIONS_PER_PERSON}" \\
  --resume \\
  --max-workers \${NSLOTS:-${APPROX_SLOTS}} \\
  --output-dir "${APPROX_OUTPUT_DIR}"

tar -czf "${APPROX_OUTPUT_DIR}.tar.gz" "${APPROX_OUTPUT_DIR}" \\
  logs/r3_approximation_methods_1200ep.*.log
ls -lh "${APPROX_OUTPUT_DIR}.tar.gz"
EOF
  chmod +x "${approx_job_file}"

  echo "Submitting approximation-method comparison job ${approx_job_file}"
  approx_submit_output=$(cd "${PROJECT_ROOT}" && env LC_ALL=C LANG=C qsub "${approx_job_file}")
  echo "${approx_submit_output}"
fi

echo
echo "Submitted requested Round 3 follow-up jobs."
echo "Check status with:"
echo "  qstat -u \${USER}"

