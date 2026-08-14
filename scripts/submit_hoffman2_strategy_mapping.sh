#!/usr/bin/env bash
set -euo pipefail

# Submit the three StrategyMapping families from one immutable manifest.

export LC_ALL=C
export LANG=C

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
source "${PROJECT_ROOT}/scripts/hoffman2_scheduler.sh"
RUN_MODE="${RUN_MODE:-smoke}"
QUEUE="${QUEUE:-campus2.q}"
DIAGNOSTIC_ACTIVE_SEARCH_DIR="${DIAGNOSTIC_ACTIVE_SEARCH_DIR:-results/active_search_benchmark_server_20260714_array486}"
TASK_H_RT="${TASK_H_RT:-04:00:00}"
TASK_H_DATA="${TASK_H_DATA:-2G}"
COLLECT_H_RT="${COLLECT_H_RT:-01:00:00}"
COLLECT_H_DATA="${COLLECT_H_DATA:-4G}"

if [[ "${RUN_MODE}" == "serious" ]]; then
  MODE_TAG="s"
  EPISODES=1200
  EPISODES_PER_TASK=10
  SEED_NAMESPACE_OFFSET=60000000
  : "${FOUR_WAY_THROTTLE:?Set FOUR_WAY_THROTTLE from the current Compute Ceiling Report}"
  : "${SIGMA_THROTTLE:?Set SIGMA_THROTTLE from the current Compute Ceiling Report}"
  : "${FIXED_THROTTLE:?Set FIXED_THROTTLE from the current Compute Ceiling Report}"
elif [[ "${RUN_MODE}" == "smoke" ]]; then
  MODE_TAG="m"
  EPISODES=2
  EPISODES_PER_TASK=2
  SEED_NAMESPACE_OFFSET=70000000
  FOUR_WAY_THROTTLE="${FOUR_WAY_THROTTLE:-3}"
  SIGMA_THROTTLE="${SIGMA_THROTTLE:-3}"
  FIXED_THROTTLE="${FIXED_THROTTLE:-5}"
else
  echo "RUN_MODE must be smoke or serious." >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/strategy_mapping_${RUN_MODE}_$(date +%Y%m%d_%H%M%S)}"
if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_DIR_ABS="${OUTPUT_DIR}"
else
  OUTPUT_DIR_ABS="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi
OBSERVATION_DRAWS=500
ORACLE_GRID_SIZE=4001
OBSERVATIONS_PER_PERSON=500

cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "StrategyMapping submission requires a clean committed worktree." >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/strategy_mapping_workflow.py create \
  --analysis-mode "${RUN_MODE}" \
  --output-dir "${OUTPUT_DIR_ABS}" \
  --diagnostic_active_search-dir "${DIAGNOSTIC_ACTIVE_SEARCH_DIR}" \
  --episodes "${EPISODES}" \
  --episodes-per-task "${EPISODES_PER_TASK}" \
  --observation-draws "${OBSERVATION_DRAWS}" \
  --oracle-grid-size "${ORACLE_GRID_SIZE}" \
  --observations-per-person "${OBSERVATIONS_PER_PERSON}" \
  --seed-namespace-offset "${SEED_NAMESPACE_OFFSET}"

MANIFEST="${OUTPUT_DIR_ABS}/strategy_mapping_manifest.json"
SCHEDULER_DIR="${OUTPUT_DIR_ABS}/scheduler"
LOG_DIR="${OUTPUT_DIR_ABS}/logs"
mkdir -p "${SCHEDULER_DIR}" "${LOG_DIR}"
SUBMISSION_EVIDENCE_DIR="${SCHEDULER_DIR}/submission_evidence"
RUN_TOKEN="$(date -u +%m%d%H%M%S)${RANDOM}"
strategy_mapping_init_submission_tracking "${SUBMISSION_EVIDENCE_DIR}"

task_range() {
  "${PYTHON_BIN}" - "${MANIFEST}" "$1" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
indices = [int(task["task_index"]) for task in manifest["tasks"] if task["analysis"] == sys.argv[2]]
if not indices or indices != list(range(min(indices), max(indices) + 1)):
    raise SystemExit("non-contiguous StrategyMapping task family")
print(min(indices), len(indices))
PY
}

read -r FOUR_WAY_OFFSET FOUR_WAY_TASKS < <(task_range four_way)
read -r SIGMA_OFFSET SIGMA_TASKS < <(task_range sigma_need)
read -r FIXED_OFFSET FIXED_TASKS < <(task_range fixed_total_need)

rollback_partial_submission() {
  strategy_mapping_rollback_partial_submission
}
trap rollback_partial_submission ERR

submit_family() {
  local family="$1"
  local offset="$2"
  local task_count="$3"
  local throttle="$4"
  local job_file="${SCHEDULER_DIR}/strategy_mapping_${RUN_MODE}_${family}_array.job"
  local job_name="strategy_mapping${MODE_TAG}${family:0:1}${RUN_TOKEN}"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N strategy_mapping${MODE_TAG}_${family}
#$ -q ${QUEUE}
#$ -j y
#$ -o ${LOG_DIR}/strategy_mapping_${RUN_MODE}_${family}.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=${TASK_H_RT}
#$ -l h_data=${TASK_H_DATA}
#$ -t 1-${task_count}
#$ -tc ${throttle}
set -euo pipefail
export LANG=C
export LC_ALL=C
cd "${PROJECT_ROOT}"
GLOBAL_TASK_INDEX=\$(( ${offset} + SGE_TASK_ID - 1 ))
"${PYTHON_BIN}" scripts/strategy_mapping_workflow.py run-task \
  --manifest "${MANIFEST}" \
  --task-index "\${GLOBAL_TASK_INDEX}"
EOF
  chmod +x "${job_file}"
  strategy_mapping_submit_job "${family}" "${job_name}" "${job_file}"
  LAST_JOB_ID="${STRATEGY_MAPPING_LAST_JOB_ID}"
}

LAST_JOB_ID=""
submit_family four_way "${FOUR_WAY_OFFSET}" "${FOUR_WAY_TASKS}" "${FOUR_WAY_THROTTLE}"
FOUR_WAY_JOB_ID="${LAST_JOB_ID}"
submit_family sigma "${SIGMA_OFFSET}" "${SIGMA_TASKS}" "${SIGMA_THROTTLE}"
SIGMA_JOB_ID="${LAST_JOB_ID}"
submit_family fixed "${FIXED_OFFSET}" "${FIXED_TASKS}" "${FIXED_THROTTLE}"
FIXED_JOB_ID="${LAST_JOB_ID}"

COLLECT_JOB="${SCHEDULER_DIR}/strategy_mapping_${RUN_MODE}_collect.job"
cat > "${COLLECT_JOB}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N strategy_mapping${MODE_TAG}_collect
#$ -q ${QUEUE}
#$ -j y
#$ -o ${LOG_DIR}/strategy_mapping_${RUN_MODE}_collect.\$JOB_ID.log
#$ -l h_rt=${COLLECT_H_RT}
#$ -l h_data=${COLLECT_H_DATA}
#$ -hold_jid ${FOUR_WAY_JOB_ID},${SIGMA_JOB_ID},${FIXED_JOB_ID}
set -euo pipefail
export LANG=C
export LC_ALL=C
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/strategy_mapping_workflow.py collect --manifest "${MANIFEST}"
EOF
chmod +x "${COLLECT_JOB}"
COLLECT_JOB_NAME="strategy_mapping${MODE_TAG}c${RUN_TOKEN}"
strategy_mapping_submit_job "collector" "${COLLECT_JOB_NAME}" "${COLLECT_JOB}"
COLLECT_JOB_ID="${STRATEGY_MAPPING_LAST_JOB_ID}"

"${PYTHON_BIN}" scripts/strategy_mapping_workflow.py record-jobs \
  --manifest "${MANIFEST}" \
  --array-job-id "${FOUR_WAY_JOB_ID},${SIGMA_JOB_ID},${FIXED_JOB_ID}" \
  --collector-job-id "${COLLECT_JOB_ID}"
trap - ERR

printf 'mode=%s\noutput_dir=%s\nmanifest=%s\n' "${RUN_MODE}" "${OUTPUT_DIR}" "${MANIFEST}"
printf 'four_way_job=%s sigma_job=%s fixed_job=%s collector_job=%s\n' \
  "${FOUR_WAY_JOB_ID}" "${SIGMA_JOB_ID}" "${FIXED_JOB_ID}" "${COLLECT_JOB_ID}"
printf 'progress: %s scripts/strategy_mapping_workflow.py progress --manifest %s\n' \
  "${PYTHON_BIN}" "${MANIFEST}"
