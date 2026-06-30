#!/usr/bin/env bash
set -euo pipefail

# Submit the Round 3 active-search grid as a Hoffman2 array job.
#
# Why this exists:
# - A single shared-memory job with many slots can wait a long time in queue.
# - This array job splits the 486-grid active-search run into 486 one-slot
#   tasks and lets Hoffman2 schedule them independently.
# - The `-tc` throttle controls how many tasks may run at the same time.
#
# Run this script from the Hoffman2 clone after pulling the latest GitHub commit.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"

OUTPUT_DIR="${OUTPUT_DIR:-results/r3_active_search_equal_outcome_server_array486}"
ARRAY_TASKS="${ARRAY_TASKS:-486}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-81}"
TASK_H_RT="${TASK_H_RT:-08:00:00}"
TASK_H_DATA="${TASK_H_DATA:-2G}"

COMBINE_SLOTS="${COMBINE_SLOTS:-8}"
COMBINE_H_RT="${COMBINE_H_RT:-08:00:00}"
COMBINE_H_DATA="${COMBINE_H_DATA:-2G}"

PACKAGE_H_RT="${PACKAGE_H_RT:-01:00:00}"
PACKAGE_H_DATA="${PACKAGE_H_DATA:-2G}"
ARCHIVE="${ARCHIVE:-${OUTPUT_DIR}.tar.gz}"

mkdir -p "${PROJECT_ROOT}/jobs" "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/${OUTPUT_DIR}"

array_job_file="${PROJECT_ROOT}/jobs/r3_active_search_array486.job"
cat > "${array_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_active_search_array486.\$JOB_ID.\$TASK_ID.log
#$ -j y
#$ -t 1-${ARRAY_TASKS}
#$ -tc ${MAX_CONCURRENT_TASKS}
#$ -l h_rt=${TASK_H_RT},h_data=${TASK_H_DATA}
#$ -N rr_r3_arr

export LC_ALL=C
export LANG=C
set -euo pipefail

PYTHON=${PYTHON_BIN}
TASK_INDEX=\$((SGE_TASK_ID - 1))
SHARD=\$(printf "active_search_equal_outcome_focused_chunk%02d_of${ARRAY_TASKS}" "\${TASK_INDEX}")
TASK_OUTPUT_DIR="${OUTPUT_DIR}/tasks/regime_grid/\${SHARD}"

mkdir -p "\${TASK_OUTPUT_DIR}"
echo "Task \${SGE_TASK_ID}/${ARRAY_TASKS}; chunk index \${TASK_INDEX}"
\$PYTHON --version
\$PYTHON scripts/generate_results.py \\
  --preset server \\
  --sections regime_grid \\
  --output-dir "\${TASK_OUTPUT_DIR}" \\
  --regime-grid active_search_equal_outcome_focused \\
  --regime-grid-chunk-index "\${TASK_INDEX}" \\
  --regime-grid-chunks ${ARRAY_TASKS} \\
  --episodes 1200 \\
  --voi-samples 500 \\
  --common-observations on \\
  --gauss-hermite-order 15 \\
  --dp-max-samples-values 2,4,6,10 \\
  --dp-mean-grid-sizes 7,11,21,50 \\
  --dp-observation-branches 3,5
EOF
chmod +x "${array_job_file}"

echo "Submitting array job ${array_job_file}"
array_submit_output=$(cd "${PROJECT_ROOT}" && env LC_ALL=C LANG=C qsub "${array_job_file}")
echo "${array_submit_output}"
array_job_id=$(echo "${array_submit_output}" | awk '/Your job-array/ {print $3} /Your job/ {print $3}' | tr -d '"' | sed 's/[.:].*$//')
if [[ -z "${array_job_id}" ]]; then
  echo "Could not parse array job id from qsub output" >&2
  exit 1
fi

combine_job_file="${PROJECT_ROOT}/jobs/r3_active_search_array486_combine.job"
cat > "${combine_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_active_search_array486_combine.\$JOB_ID.log
#$ -j y
#$ -pe shared ${COMBINE_SLOTS}
#$ -l h_rt=${COMBINE_H_RT},h_data=${COMBINE_H_DATA}
#$ -N rr_r3_comb

export LC_ALL=C
export LANG=C
set -euo pipefail

PYTHON=${PYTHON_BIN}
\$PYTHON --version
\$PYTHON scripts/run_parallel_r2.py \\
  --preset server \\
  --sections regime_grid \\
  --regime-grid active_search_equal_outcome_focused \\
  --regime-grid-chunks ${ARRAY_TASKS} \\
  --episodes 1200 \\
  --voi-samples 500 \\
  --common-observations on \\
  --resume \\
  --max-workers \${NSLOTS:-${COMBINE_SLOTS}} \\
  --output-dir "${OUTPUT_DIR}"
EOF
chmod +x "${combine_job_file}"

echo "Submitting combine job ${combine_job_file}"
combine_submit_output=$(cd "${PROJECT_ROOT}" && env LC_ALL=C LANG=C qsub -hold_jid "${array_job_id}" "${combine_job_file}")
echo "${combine_submit_output}"
combine_job_id=$(echo "${combine_submit_output}" | awk '/Your job/ {print $3}' | tr -d '"')
if [[ -z "${combine_job_id}" ]]; then
  echo "Could not parse combine job id from qsub output" >&2
  exit 1
fi

package_job_file="${PROJECT_ROOT}/jobs/r3_active_search_array486_package.job"
cat > "${package_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_active_search_array486_package.\$JOB_ID.log
#$ -j y
#$ -l h_rt=${PACKAGE_H_RT},h_data=${PACKAGE_H_DATA}
#$ -N rr_r3_pkg

export LC_ALL=C
export LANG=C
set -euo pipefail

date
tar -czf "${ARCHIVE}" "${OUTPUT_DIR}" \\
  results/r3_observation_stream_check_server.json \\
  logs/r3_active_search_array486.*.log \\
  logs/r3_active_search_array486_combine.*.log
ls -lh "${ARCHIVE}"
date
EOF
chmod +x "${package_job_file}"

echo "Submitting package job ${package_job_file}"
(cd "${PROJECT_ROOT}" && env LC_ALL=C LANG=C qsub -hold_jid "${combine_job_id}" "${package_job_file}")

echo
echo "Submitted Round 3 array workflow:"
echo "  array job:   ${array_job_id}"
echo "  combine job: ${combine_job_id}"
echo "  archive:     ${ARCHIVE}"
echo
echo "Check status with:"
echo "  qstat -u \${USER}"
