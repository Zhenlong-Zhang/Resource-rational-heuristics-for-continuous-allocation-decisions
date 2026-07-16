#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

PROJECT_ROOT="${PROJECT_ROOT:-/u/home/z/zzl/Resource-rational-heuristics-for-continuous-allocation-decisions}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
WORKFLOW_RUN_DIR="${WORKFLOW_RUN_DIR:-results/r3_approximation_methods_episode_array}"
MANIFEST="${MANIFEST:-${WORKFLOW_RUN_DIR}/r3_episode_array_manifest.json}"
TASK_H_RT="${TASK_H_RT:-43200}"
TASK_H_DATA="${TASK_H_DATA:-3G}"
COLLECT_H_RT="${COLLECT_H_RT:-14400}"
COLLECT_H_DATA="${COLLECT_H_DATA:-4G}"
QSUB_BIN="${QSUB_BIN:-qsub}"
QDEL_BIN="${QDEL_BIN:-qdel}"
QSTAT_BIN="${QSTAT_BIN:-qstat}"
ROLLBACK_POLLS="${ROLLBACK_POLLS:-30}"
ROLLBACK_SLEEP_SECONDS="${ROLLBACK_SLEEP_SECONDS:-2}"
JOB_DIR="${JOB_DIR:-${PROJECT_ROOT}/jobs}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
SUBMISSION_RECEIPT_DIR="${SUBMISSION_RECEIPT_DIR:-${WORKFLOW_RUN_DIR}/submission_receipts}"

cd "${PROJECT_ROOT}"
mkdir -p "${JOB_DIR}" "${LOG_DIR}"
mkdir -p "${SUBMISSION_RECEIPT_DIR}"

manifest_summary=$("${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(len(manifest["tasks"]), manifest["array_lanes"], manifest["lane_throttle"])
PY
)
read -r task_count array_lanes lane_throttle <<< "${manifest_summary}"

if [[ "${task_count}" -eq 0 ]]; then
  echo "The frozen checkpoint is already complete; no episode array was submitted."
  "${PYTHON_BIN}" scripts/r3_episode_array_workflow.py status --manifest "${MANIFEST}"
  exit 0
fi
if [[ "${lane_throttle}" -gt 100 ]]; then
  echo "Manifest lane_throttle=${lane_throttle} exceeds max_aj_instances=100" >&2
  exit 2
fi
if [[ $((array_lanes * (lane_throttle + 1) + 1)) -gt 500 ]]; then
  echo "Manifest lane concurrency exceeds max_u_jobs=500 after reserving the collector" >&2
  exit 2
fi

submission_json=$("${PYTHON_BIN}" scripts/r3_episode_array_workflow.py submission \
  --manifest "${MANIFEST}" --action init)
submission_complete=0

rollback_submission() {
  exit_status=$?
  trap - ERR
  if [[ "${submission_complete}" -eq 1 ]]; then
    return "${exit_status}"
  fi
  set +e
  current=$("${PYTHON_BIN}" scripts/r3_episode_array_workflow.py submission \
    --manifest "${MANIFEST}" --action show 2>/dev/null)
  submitted_ids=$(printf '%s' "${current}" | "${PYTHON_BIN}" -c '
import json, sys
value = json.load(sys.stdin)
ids = list(value.get("lane_jobs", {}).values())
if value.get("collector_job_id"):
    ids.append(value["collector_job_id"])
print(" ".join(str(item) for item in ids))
' 2>/dev/null)
  receipt_ids=$(find "${SUBMISSION_RECEIPT_DIR}" -maxdepth 1 -type f -name '*.txt' \
    -exec cat {} \; 2>/dev/null | sed 's/[.:].*$//' | grep -E '^[0-9]+$' || true)
  submitted_ids=$(printf '%s\n%s\n' "${submitted_ids}" "${receipt_ids}" | \
    tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')
  rollback_ok=yes
  for submitted_id in ${submitted_ids}; do
    "${QDEL_BIN}" "${submitted_id}" || true
  done
  rollback_qstat="${SUBMISSION_RECEIPT_DIR}/rollback_qstat.txt"
  for ((poll=1; poll<=ROLLBACK_POLLS; poll++)); do
    rollback_ok=yes
    if ! "${QSTAT_BIN}" -u "${USER}" > "${rollback_qstat}"; then
      rollback_ok=no
    else
      for submitted_id in ${submitted_ids}; do
        if awk -v id="${submitted_id}" '$1 == id {found=1} END {exit !found}' "${rollback_qstat}"; then
          rollback_ok=no
        fi
      done
    fi
    [[ "${rollback_ok}" == yes ]] && break
    sleep "${ROLLBACK_SLEEP_SECONDS}"
  done
  "${PYTHON_BIN}" scripts/r3_episode_array_workflow.py submission \
    --manifest "${MANIFEST}" --action rollback \
    --rollback-complete "${rollback_ok}" \
    --detail "submission failed with exit ${exit_status}; qdel ids: ${submitted_ids}" >/dev/null 2>&1
  exit "${exit_status}"
}
trap rollback_submission ERR

existing_collector=$(printf '%s' "${submission_json}" | "${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(sys.stdin).get("collector_job_id", ""))')
if [[ -n "${existing_collector}" ]]; then
  submission_complete=1
  echo "Submission already recorded with collector ${existing_collector}; no duplicate jobs submitted."
  exit 0
fi

lane_job_ids=()
for ((lane_id=1; lane_id<=array_lanes; lane_id++)); do
  lane_record=$("${PYTHON_BIN}" - "${MANIFEST}" "${lane_id}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
lane_id = int(sys.argv[2])
lane = next(record for record in manifest["lanes"] if int(record["lane_id"]) == lane_id)
print(f'{lane["task_file"]}\t{len(lane["task_ids"])}')
PY
)
  IFS=$'\t' read -r lane_file lane_task_count <<< "${lane_record}"
  if [[ "${lane_task_count}" -eq 0 ]]; then
    continue
  fi
  existing_lane_job=$(printf '%s' "${submission_json}" | "${PYTHON_BIN}" -c \
    'import json,sys; value=json.load(sys.stdin); print(value.get("lane_jobs", {}).get(sys.argv[1], ""))' \
    "${lane_id}")
  if [[ -n "${existing_lane_job}" ]]; then
    lane_job_ids+=("${existing_lane_job}")
    continue
  fi
  lane_job_file="${JOB_DIR}/r3_episode_array_lane_${lane_id}.job"
  cat > "${lane_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o ${LOG_DIR}/r3_episode_array_lane${lane_id}.\$JOB_ID.\$TASK_ID.log
#$ -j y
#$ -l h_rt=${TASK_H_RT},h_data=${TASK_H_DATA}
#$ -N rr_r3_e${lane_id}

export LC_ALL=C
export LANG=C
set -euo pipefail

TASK_ID=\$(sed -n "\${SGE_TASK_ID}p" "${lane_file}")
if [[ -z "\${TASK_ID}" ]]; then
  echo "No manifest task for lane ${lane_id}, SGE_TASK_ID=\${SGE_TASK_ID}" >&2
  exit 1
fi
ATTEMPT_ID="job\${JOB_ID}_lane${lane_id}_task\${SGE_TASK_ID}"
"${PYTHON_BIN}" scripts/r3_episode_array_workflow.py run-shard \
  --manifest "${MANIFEST}" \
  --task-id "\${TASK_ID}" \
  --attempt-id "\${ATTEMPT_ID}"
EOF
  chmod +x "${lane_job_file}"
  receipt="${SUBMISSION_RECEIPT_DIR}/lane_${lane_id}.txt"
  env LC_ALL=C LANG=C "${QSUB_BIN}" -terse \
    -t "1-${lane_task_count}" \
    -tc "${lane_throttle}" \
    "${lane_job_file}" > "${receipt}"
  submit_output=$(cat "${receipt}")
  echo "${submit_output}"
  lane_job_id=$(printf '%s\n' "${submit_output}" | tail -n 1 | tr -d '"' | sed 's/[.:].*$//')
  if [[ -z "${lane_job_id}" ]]; then
    echo "Could not parse lane ${lane_id} job ID" >&2
    exit 3
  fi
  submission_json=$("${PYTHON_BIN}" scripts/r3_episode_array_workflow.py submission \
    --manifest "${MANIFEST}" --action record-lane \
    --lane-id "${lane_id}" --job-id "${lane_job_id}")
  lane_job_ids+=("${lane_job_id}")
done

if [[ "${#lane_job_ids[@]}" -eq 0 ]]; then
  echo "No non-empty lane was submitted" >&2
  exit 4
fi
hold_ids=$(IFS=,; echo "${lane_job_ids[*]}")

collector_job_file="${JOB_DIR}/r3_episode_array_collector.job"
cat > "${collector_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o ${LOG_DIR}/r3_episode_array_collector.\$JOB_ID.log
#$ -j y
#$ -l h_rt=${COLLECT_H_RT},h_data=${COLLECT_H_DATA}
#$ -N rr_r3_ecol

export LC_ALL=C
export LANG=C
set -euo pipefail

"${PYTHON_BIN}" scripts/r3_episode_array_workflow.py collect \
  --manifest "${MANIFEST}" \
  --promote
"${PYTHON_BIN}" scripts/r3_episode_array_workflow.py status --manifest "${MANIFEST}"
EOF
chmod +x "${collector_job_file}"
collector_receipt="${SUBMISSION_RECEIPT_DIR}/collector.txt"
env LC_ALL=C LANG=C "${QSUB_BIN}" -terse -hold_jid "${hold_ids}" \
  "${collector_job_file}" > "${collector_receipt}"
collector_output=$(cat "${collector_receipt}")
echo "${collector_output}"
collector_job_id=$(printf '%s\n' "${collector_output}" | tail -n 1 | tr -d '"' | sed 's/[.:].*$//')
if [[ -z "${collector_job_id}" ]]; then
  echo "Could not parse collector job ID" >&2
  exit 5
fi

"${PYTHON_BIN}" scripts/r3_episode_array_workflow.py submission \
  --manifest "${MANIFEST}" --action record-collector --job-id "${collector_job_id}" >/dev/null
submission_complete=1
trap - ERR

echo "Submitted ${#lane_job_ids[@]} disjoint one-slot array lanes for ${task_count} episodes."
echo "Per-lane throttle: ${lane_throttle}; collector ${collector_job_id} holds on ${hold_ids}."
echo "Status: ${PYTHON_BIN} scripts/r3_episode_array_workflow.py status --manifest ${MANIFEST}"
