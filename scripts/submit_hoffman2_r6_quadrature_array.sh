#!/usr/bin/env bash
set -euo pipefail

# Submit the frozen 90-case R6 quadrature diagnostic outside the clean checkout.

export LC_ALL=C
export LANG=C

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
QUEUE="${QUEUE:-campus2.q}"
TASK_H_RT="${TASK_H_RT:-02:00:00}"
TASK_H_DATA="${TASK_H_DATA:-2G}"
COLLECT_H_RT="${COLLECT_H_RT:-00:30:00}"
COLLECT_H_DATA="${COLLECT_H_DATA:-1G}"
: "${THROTTLE:?Set THROTTLE from the current Hoffman2 Compute Ceiling Report}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/u/scratch/${USER:0:1}/${USER}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/r6_quadrature_diagnostic_$(date +%Y%m%d_%H%M%S)}"
source "${PROJECT_ROOT}/scripts/r6_scheduler.sh"

OUTPUT_DIR_ABS="$("${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
PROJECT_ROOT_ABS="$(cd "${PROJECT_ROOT}" && pwd -P)"
case "${OUTPUT_DIR_ABS}/" in
  "${PROJECT_ROOT_ABS}/"*)
    echo "Quadrature diagnostic output and logs must be outside the clean checkout." >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT_ABS}"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Submission requires a clean committed worktree." >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/r6_quadrature_array.py create --output-dir "${OUTPUT_DIR_ABS}"
MANIFEST="${OUTPUT_DIR_ABS}/r6_quadrature_diagnostic_manifest.json"
SCHEDULER_DIR="${OUTPUT_DIR_ABS}/scheduler"
LOG_DIR="${OUTPUT_DIR_ABS}/logs"
SUBMISSION_EVIDENCE_DIR="${SCHEDULER_DIR}/submission_evidence"
mkdir -p "${SCHEDULER_DIR}" "${LOG_DIR}"
r6_init_submission_tracking "${SUBMISSION_EVIDENCE_DIR}"

RUN_TOKEN="$(date -u +%m%d%H%M%S)${RANDOM}"
ARRAY_JOB_NAME="r6qda${RUN_TOKEN}"
COLLECT_JOB_NAME="r6qdc${RUN_TOKEN}"
ARRAY_JOB="${SCHEDULER_DIR}/quadrature_array.job"
COLLECT_JOB="${SCHEDULER_DIR}/quadrature_collector.job"

cat >"${ARRAY_JOB}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N r6qda
#$ -q ${QUEUE}
#$ -j y
#$ -o ${LOG_DIR}/quadrature_array.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=${TASK_H_RT}
#$ -l h_data=${TASK_H_DATA}
#$ -t 1-90
#$ -tc ${THROTTLE}
set -euo pipefail
export LC_ALL=C
export LANG=C
cd "${PROJECT_ROOT_ABS}"
"${PYTHON_BIN}" scripts/r6_quadrature_array.py run-task \
  --manifest "${MANIFEST}" \
  --task-index "\$((SGE_TASK_ID - 1))"
EOF
chmod +x "${ARRAY_JOB}"

rollback_partial_submission() {
  r6_rollback_partial_submission
}
trap rollback_partial_submission ERR

r6_submit_job "array" "${ARRAY_JOB_NAME}" "${ARRAY_JOB}"
ARRAY_JOB_ID="${R6_LAST_JOB_ID}"

cat >"${COLLECT_JOB}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N r6qdc
#$ -q ${QUEUE}
#$ -j y
#$ -o ${LOG_DIR}/quadrature_collector.\$JOB_ID.log
#$ -l h_rt=${COLLECT_H_RT}
#$ -l h_data=${COLLECT_H_DATA}
#$ -hold_jid ${ARRAY_JOB_ID}
set -euo pipefail
export LC_ALL=C
export LANG=C
cd "${PROJECT_ROOT_ABS}"
for _ in \$(seq 1 60); do
  [[ -s "${SCHEDULER_DIR}/jobs.json" ]] && break
  sleep 1
done
[[ -s "${SCHEDULER_DIR}/jobs.json" ]]
"${PYTHON_BIN}" scripts/r6_quadrature_array.py collect --manifest "${MANIFEST}"
EOF
chmod +x "${COLLECT_JOB}"

r6_submit_job "collector" "${COLLECT_JOB_NAME}" "${COLLECT_JOB}"
COLLECT_JOB_ID="${R6_LAST_JOB_ID}"

"${PYTHON_BIN}" - \
  "${OUTPUT_DIR_ABS}" "${ARRAY_JOB_ID}" "${COLLECT_JOB_ID}" \
  "${ARRAY_JOB_NAME}" "${COLLECT_JOB_NAME}" "${THROTTLE}" "${QUEUE}" \
  "${ARRAY_JOB}" "${COLLECT_JOB}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

(
    output, array_id, collector_id, array_name, collector_name, throttle,
    queue, array_job, collector_job,
) = sys.argv[1:]

def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

submission_dir = os.path.join(output, "scheduler", "submission_evidence")
value = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "array_job_id": array_id,
    "collector_job_id": collector_id,
    "array_job_name": array_name,
    "collector_job_name": collector_name,
    "throttle": int(throttle),
    "queue": queue,
    "task_count": 90,
    "task_slots": 1,
    "collector_slots": 1,
    "array_job_path": os.path.relpath(array_job, output),
    "collector_job_path": os.path.relpath(collector_job, output),
    "array_job_sha256": sha256(array_job),
    "collector_job_sha256": sha256(collector_job),
    "submission_evidence": {
        os.path.relpath(os.path.join(submission_dir, name), output): {
            "sha256": sha256(os.path.join(submission_dir, name)),
            "bytes": os.path.getsize(os.path.join(submission_dir, name)),
        }
        for name in sorted(os.listdir(submission_dir))
        if os.path.isfile(os.path.join(submission_dir, name))
    },
}
path = os.path.join(output, "scheduler", "jobs.json")
fd, temporary = tempfile.mkstemp(prefix=".jobs.", suffix=".tmp", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
trap - ERR

printf 'output_dir=%s\nmanifest=%s\n' "${OUTPUT_DIR_ABS}" "${MANIFEST}"
printf 'array_job=%s collector_job=%s tasks=90 throttle=%s\n' \
  "${ARRAY_JOB_ID}" "${COLLECT_JOB_ID}" "${THROTTLE}"
printf 'after qacct is available: %s scripts/r6_quadrature_array.py audit-qacct --manifest %s --job-id %s --job-id %s\n' \
  "${PYTHON_BIN}" "${MANIFEST}" "${ARRAY_JOB_ID}" "${COLLECT_JOB_ID}"
printf 'then finalize: %s scripts/r6_quadrature_array.py finalize --manifest %s\n' \
  "${PYTHON_BIN}" "${MANIFEST}"
