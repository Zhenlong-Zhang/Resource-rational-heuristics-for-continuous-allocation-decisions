#!/usr/bin/env bash
set -euo pipefail

# Submit the R4 diagnostic grid as independent one-slot Hoffman2 array tasks.

export LC_ALL=C
export LANG=C

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-results/r4_diagnostic_active_search_server}"
GRID="${GRID:-r4_diagnostic_active_search}"
PRESET="${PRESET:-server}"
EPISODES="${EPISODES:-1200}"
VOI_SAMPLES="${VOI_SAMPLES:-500}"
OBSERVATIONS_PER_PERSON="${OBSERVATIONS_PER_PERSON:-500}"
MANUAL_ACTIVE_SAMPLES_PER_PERSON="${MANUAL_ACTIVE_SAMPLES_PER_PERSON:-3}"
MAX_GRID_POINTS="${MAX_GRID_POINTS:-}"
ARRAY_TASKS="${ARRAY_TASKS:-486}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-160}"
TASK_H_RT="${TASK_H_RT:-24:00:00}"
TASK_H_DATA="${TASK_H_DATA:-2G}"
COLLECT_H_RT="${COLLECT_H_RT:-02:00:00}"
COLLECT_H_DATA="${COLLECT_H_DATA:-2G}"
BASELINE_COMMIT="e92d64d"

cd "${PROJECT_ROOT}"
mkdir -p jobs logs "${OUTPUT_DIR}"

if ! git diff --quiet "${BASELINE_COMMIT}" -- \
  scripts/generate_results.py \
  src/mdp \
  src/policies \
  src/experiments \
  src/solvers \
  src/simulator; then
  echo "Scientific source differs from R4 baseline ${BASELINE_COMMIT}; refusing submission." >&2
  exit 1
fi

SOURCE_COMMIT="$(git rev-parse HEAD)"
GRID_SIZE="$(${PYTHON_BIN} - "${GRID}" "${MAX_GRID_POINTS}" <<'PY'
import sys
from src.experiments.sweeps import build_targeted_regime_grid_configs

grid = sys.argv[1]
limit = int(sys.argv[2]) if sys.argv[2] else None
print(len(build_targeted_regime_grid_configs([grid], max_grid_points=limit)))
PY
)"
GRID_TASKS="${ARRAY_TASKS}"
if [[ "${GRID_SIZE}" -lt "${GRID_TASKS}" ]]; then
  GRID_TASKS="${GRID_SIZE}"
fi
if [[ "${GRID_TASKS}" -le 0 ]]; then
  echo "No R4 grid tasks were generated." >&2
  exit 1
fi

manifest_command=(
  "${PYTHON_BIN}" scripts/r4_array_workflow.py create-manifest
  --run-dir "${OUTPUT_DIR}"
  --git-commit "${SOURCE_COMMIT}"
  --baseline-commit "${BASELINE_COMMIT}"
  --throttle "${MAX_CONCURRENT_TASKS}"
  --grid "${GRID}"
  --grid-size "${GRID_SIZE}"
  --chunks "${GRID_TASKS}"
  --preset "${PRESET}"
  --episodes "${EPISODES}"
  --voi-samples "${VOI_SAMPLES}"
  --common-observations on
  --observations-per-person "${OBSERVATIONS_PER_PERSON}"
  --manual-active-samples-per-person "${MANUAL_ACTIVE_SAMPLES_PER_PERSON}"
)
if [[ -n "${MAX_GRID_POINTS}" ]]; then
  manifest_command+=(--max-grid-points "${MAX_GRID_POINTS}")
fi
"${manifest_command[@]}"

MANIFEST="${PROJECT_ROOT}/${OUTPUT_DIR}/r4_array_manifest.json"
ARRAY_JOB_FILE="${PROJECT_ROOT}/jobs/r4_diagnostic_active_search_array.job"
cat > "${ARRAY_JOB_FILE}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r4_diagnostic_active_search_array.\$JOB_ID.\$TASK_ID.log
#$ -j y
#$ -t 1-${GRID_TASKS}
#$ -tc ${MAX_CONCURRENT_TASKS}
#$ -l h_rt=${TASK_H_RT},h_data=${TASK_H_DATA}
#$ -N rr_r4_arr

export LC_ALL=C
export LANG=C
set -euo pipefail

TASK_INDEX=\$((SGE_TASK_ID - 1))
${PYTHON_BIN} scripts/r4_array_workflow.py run-shard \\
  --manifest "${MANIFEST}" \\
  --chunk-index "\${TASK_INDEX}"
EOF
chmod +x "${ARRAY_JOB_FILE}"

array_submit_output="$(env LC_ALL=C LANG=C qsub "${ARRAY_JOB_FILE}")"
echo "${array_submit_output}"
array_job_id="$(echo "${array_submit_output}" | awk '/Your job-array/ {print $3} /Your job/ {print $3}' | tr -d '"' | sed 's/[.:].*$//')"
if [[ -z "${array_job_id}" ]]; then
  echo "Could not parse R4 array job id." >&2
  exit 1
fi

COLLECT_JOB_FILE="${PROJECT_ROOT}/jobs/r4_diagnostic_active_search_collect.job"
cat > "${COLLECT_JOB_FILE}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r4_diagnostic_active_search_collect.\$JOB_ID.log
#$ -j y
#$ -l h_rt=${COLLECT_H_RT},h_data=${COLLECT_H_DATA}
#$ -N rr_r4_col

export LC_ALL=C
export LANG=C
set -euo pipefail

${PYTHON_BIN} scripts/r4_array_workflow.py collect --manifest "${MANIFEST}"
tar -czf "${OUTPUT_DIR}.tar.gz" "${OUTPUT_DIR}" \\
  logs/r4_diagnostic_active_search_array.${array_job_id}.*.log \\
  logs/r4_diagnostic_active_search_collect.\$JOB_ID.log
ls -lh "${OUTPUT_DIR}.tar.gz"
EOF
chmod +x "${COLLECT_JOB_FILE}"

collect_submit_output="$(env LC_ALL=C LANG=C qsub -hold_jid "${array_job_id}" "${COLLECT_JOB_FILE}")"
echo "${collect_submit_output}"
collector_job_id="$(echo "${collect_submit_output}" | awk '/Your job/ {print $3}' | tr -d '"' | sed 's/[.:].*$//')"
if [[ -z "${collector_job_id}" ]]; then
  echo "Could not parse R4 collector job id." >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/r4_array_workflow.py record-jobs \\
  --manifest "${MANIFEST}" \\
  --array-job-id "${array_job_id}" \\
  --collector-job-id "${collector_job_id}"

echo "R4 array job: ${array_job_id}"
echo "R4 collector job: ${collector_job_id}"
echo "R4 tasks: ${GRID_TASKS}; throttle: ${MAX_CONCURRENT_TASKS}"
echo "Progress command:"
echo "  ${PYTHON_BIN} scripts/r4_array_workflow.py progress --manifest ${MANIFEST}"
