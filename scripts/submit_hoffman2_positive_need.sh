#!/usr/bin/env bash
set -euo pipefail

# Submit one StrategyMapping pre-feedback development or confirmation array.

export LC_ALL=C
export LANG=C

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
QUEUE="${QUEUE:-campus2.q}"
PHASE="${PHASE:-development}"
TASK_H_RT="${TASK_H_RT:-08:00:00}"
TASK_H_DATA="${TASK_H_DATA:-2G}"
COLLECT_H_RT="${COLLECT_H_RT:-}"
COLLECT_H_DATA="${COLLECT_H_DATA:-4G}"
source "${PROJECT_ROOT}/scripts/hoffman2_scheduler.sh"

case "${PHASE}" in
  development)
    : "${THROTTLE:?Set THROTTLE from the current Hoffman2 Compute Ceiling Report}"
    COLLECT_H_RT="${COLLECT_H_RT:-08:00:00}"
    OUTPUT_DIR="${OUTPUT_DIR:-results/positive_need_development_$(date +%Y%m%d_%H%M%S)}"
    ;;
  smoke)
    THROTTLE="${THROTTLE:-36}"
    COLLECT_H_RT="${COLLECT_H_RT:-02:00:00}"
    : "${DEVELOPMENT_DIR:?Set DEVELOPMENT_DIR to the validated development output}"
    OUTPUT_DIR="${OUTPUT_DIR:-results/positive_need_smoke_$(date +%Y%m%d_%H%M%S)}"
    ;;
  serious)
    : "${THROTTLE:?Set THROTTLE from the current Hoffman2 Compute Ceiling Report}"
    : "${DEVELOPMENT_DIR:?Set DEVELOPMENT_DIR to the validated development output}"
    COLLECT_H_RT="${COLLECT_H_RT:-04:00:00}"
    OUTPUT_DIR="${OUTPUT_DIR:-results/positive_need_serious_$(date +%Y%m%d_%H%M%S)}"
    ;;
  *)
    echo "PHASE must be development, smoke, or serious." >&2
    exit 2
    ;;
esac

if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_DIR_ABS="${OUTPUT_DIR}"
else
  OUTPUT_DIR_ABS="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Submission requires a clean committed worktree." >&2
  exit 1
fi

if [[ "${PHASE}" == "development" ]]; then
  "${PYTHON_BIN}" scripts/positive_need_workflow.py create-development \
    --output-dir "${OUTPUT_DIR_ABS}"
else
  if [[ "${DEVELOPMENT_DIR}" = /* ]]; then
    DEVELOPMENT_DIR_ABS="${DEVELOPMENT_DIR}"
  else
    DEVELOPMENT_DIR_ABS="${PROJECT_ROOT}/${DEVELOPMENT_DIR}"
  fi
  "${PYTHON_BIN}" scripts/positive_need_workflow.py create-confirmation \
    --output-dir "${OUTPUT_DIR_ABS}" \
    --development-dir "${DEVELOPMENT_DIR_ABS}" \
    --mode "${PHASE}"
fi

MANIFEST="${OUTPUT_DIR_ABS}/positive_need_manifest.json"
SCHEDULER_DIR="${OUTPUT_DIR_ABS}/scheduler"
LOG_DIR="${OUTPUT_DIR_ABS}/logs"
mkdir -p "${SCHEDULER_DIR}" "${LOG_DIR}"
SUBMISSION_EVIDENCE_DIR="${SCHEDULER_DIR}/submission_evidence"
strategy_mapping_init_submission_tracking "${SUBMISSION_EVIDENCE_DIR}"

TASK_COUNT="$("${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["task_count"])
PY
)"
RUN_TOKEN="$(date -u +%m%d%H%M%S)${RANDOM}"
PHASE_TAG="${PHASE:0:1}"
ARRAY_JOB="${SCHEDULER_DIR}/positive_need_${PHASE}_array.job"
cat > "${ARRAY_JOB}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N rrp${PHASE_TAG}
#$ -q ${QUEUE}
#$ -j y
#$ -o ${LOG_DIR}/positive_need_${PHASE}.$JOB_ID.$TASK_ID.log
#$ -l h_rt=${TASK_H_RT}
#$ -l h_data=${TASK_H_DATA}
#$ -t 1-${TASK_COUNT}
#$ -tc ${THROTTLE}
set -euo pipefail
export LC_ALL=C
export LANG=C
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/positive_need_workflow.py run-task \
  --manifest "${MANIFEST}" \
  --task-index "$((SGE_TASK_ID - 1))"
EOF
chmod +x "${ARRAY_JOB}"

rollback_partial_submission() {
  strategy_mapping_rollback_partial_submission
}
trap rollback_partial_submission ERR

ARRAY_JOB_NAME="rrp${PHASE_TAG}a${RUN_TOKEN}"
strategy_mapping_submit_job "array" "${ARRAY_JOB_NAME}" "${ARRAY_JOB}"
ARRAY_JOB_ID="${STRATEGY_MAPPING_LAST_JOB_ID}"

COLLECT_JOB="${SCHEDULER_DIR}/positive_need_${PHASE}_collect.job"
cat > "${COLLECT_JOB}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N rrp${PHASE_TAG}c
#$ -q ${QUEUE}
#$ -j y
#$ -o ${LOG_DIR}/positive_need_${PHASE}_collect.$JOB_ID.log
#$ -l h_rt=${COLLECT_H_RT}
#$ -l h_data=${COLLECT_H_DATA}
#$ -hold_jid ${ARRAY_JOB_ID}
set -euo pipefail
export LC_ALL=C
export LANG=C
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/positive_need_workflow.py collect --manifest "${MANIFEST}"
EOF
chmod +x "${COLLECT_JOB}"
COLLECT_JOB_NAME="rrp${PHASE_TAG}c${RUN_TOKEN}"
strategy_mapping_submit_job "collector" "${COLLECT_JOB_NAME}" "${COLLECT_JOB}"
COLLECT_JOB_ID="${STRATEGY_MAPPING_LAST_JOB_ID}"

"${PYTHON_BIN}" - \
  "${OUTPUT_DIR_ABS}" "${ARRAY_JOB_ID}" "${COLLECT_JOB_ID}" "${THROTTLE}" \
  "${QUEUE}" "${TASK_H_RT}" "${TASK_H_DATA}" "${COLLECT_H_RT}" "${COLLECT_H_DATA}" \
  "${ARRAY_JOB}" "${COLLECT_JOB}" "${TASK_COUNT}" "${PHASE}" \
  "${ARRAY_JOB_NAME}" "${COLLECT_JOB_NAME}" "${PYTHON_BIN}" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
(
    output, array_id, collector_id, throttle, queue, task_h_rt, task_h_data,
    collect_h_rt, collect_h_data, array_job, collector_job, task_count, phase,
    array_name, collector_name, python_bin,
) = sys.argv[1:]

def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

value = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "array_job_id": array_id,
    "collector_job_id": collector_id,
    "array_job_name": array_name,
    "collector_job_name": collector_name,
    "throttle": int(throttle),
    "queue": queue,
    "task_h_rt": task_h_rt,
    "task_h_data": task_h_data,
    "collector_h_rt": collect_h_rt,
    "collector_h_data": collect_h_data,
    "task_slots": 1,
    "collector_slots": 1,
    "task_count": int(task_count),
    "phase": phase,
    "python_bin": python_bin,
    "array_job_path": os.path.relpath(array_job, output),
    "collector_job_path": os.path.relpath(collector_job, output),
    "array_job_sha256": sha256(array_job),
    "collector_job_sha256": sha256(collector_job),
}
submission_dir = os.path.join(output, "scheduler", "submission_evidence")
value["submission_evidence"] = {
    os.path.relpath(os.path.join(submission_dir, name), output): {
        "sha256": sha256(os.path.join(submission_dir, name)),
        "bytes": os.path.getsize(os.path.join(submission_dir, name)),
    }
    for name in sorted(os.listdir(submission_dir))
    if os.path.isfile(os.path.join(submission_dir, name))
}
path = os.path.join(output, "scheduler", "jobs.json")
fd, temporary = tempfile.mkstemp(prefix=".jobs.", suffix=".tmp", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
trap - ERR

printf 'phase=%s\noutput_dir=%s\nmanifest=%s\n' "${PHASE}" "${OUTPUT_DIR_ABS}" "${MANIFEST}"
printf 'array_job=%s collector_job=%s task_count=%s throttle=%s\n' \
  "${ARRAY_JOB_ID}" "${COLLECT_JOB_ID}" "${TASK_COUNT}" "${THROTTLE}"
printf 'progress: %s scripts/positive_need_workflow.py progress --manifest %s\n' \
  "${PYTHON_BIN}" "${MANIFEST}"
