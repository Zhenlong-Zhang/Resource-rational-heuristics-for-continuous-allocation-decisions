#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

PROJECT_ROOT="${PROJECT_ROOT:-/u/home/z/zzl/Resource-rational-heuristics-for-continuous-allocation-decisions}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
CANONICAL_OUTPUT_DIR="${CANONICAL_OUTPUT_DIR:-results/r3_approximation_methods_checkpointed_1200ep_20260713}"
WORKFLOW_RUN_DIR="${WORKFLOW_RUN_DIR:-results/r3_approximation_methods_episode_array}"
CUTOVER_EVIDENCE_DIR="${CUTOVER_EVIDENCE_DIR:-${WORKFLOW_RUN_DIR}_cutover_evidence}"
TASK_MANIFEST="${TASK_MANIFEST:-jobs/r3_approx_methods_tasks.tsv}"
SMOKE_GATE="${SMOKE_GATE:-}"
REVIEWED_COMMIT="${REVIEWED_COMMIT:-}"
WRITER_JOB_IDS="${WRITER_JOB_IDS:-14055056,14055057}"
SUCCESSOR_JOB_IDS="${SUCCESSOR_JOB_IDS:-}"
WRITER_JOB_NAME_PATTERN="${WRITER_JOB_NAME_PATTERN:-rr_r3_m}"
CUTOVER_ACTION="${CUTOVER_ACTION:-verify}"
ARRAY_LANES="${ARRAY_LANES:-2}"
LANE_THROTTLE="${LANE_THROTTLE:-80}"
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-700}"
QUIESCENCE_POLLS="${QUIESCENCE_POLLS:-60}"
QUIESCENCE_SLEEP_SECONDS="${QUIESCENCE_SLEEP_SECONDS:-10}"
QSTAT_BIN="${QSTAT_BIN:-qstat}"
QDEL_BIN="${QDEL_BIN:-qdel}"
QACCT_BIN="${QACCT_BIN:-qacct}"

if [[ "${CUTOVER_ACTION}" != "verify" && "${CUTOVER_ACTION}" != "cancel" ]]; then
  echo "CUTOVER_ACTION must be verify or cancel" >&2
  exit 2
fi
if [[ -z "${SMOKE_GATE}" || ! -f "${SMOKE_GATE}" ]]; then
  echo "SMOKE_GATE must point to completed isolated smoke evidence" >&2
  exit 2
fi
if [[ -z "${REVIEWED_COMMIT}" ]]; then
  echo "REVIEWED_COMMIT must name the exact reviewed commit" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${CUTOVER_EVIDENCE_DIR}" logs jobs

if [[ "$(git rev-parse HEAD)" != "${REVIEWED_COMMIT}" ]]; then
  echo "Checkout HEAD does not match REVIEWED_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked checkout is dirty; refusing cutover" >&2
  exit 2
fi
"${PYTHON_BIN}" scripts/r3_episode_array_workflow.py verify-smoke-gate \
  --gate "${SMOKE_GATE}" --reviewed-commit "${REVIEWED_COMMIT}" >/dev/null

all_writer_ids="${WRITER_JOB_IDS}"
if [[ -n "${SUCCESSOR_JOB_IDS}" ]]; then
  all_writer_ids="${all_writer_ids},${SUCCESSOR_JOB_IDS}"
fi
IFS=',' read -r -a writer_ids <<< "${all_writer_ids}"
if [[ "${#writer_ids[@]}" -eq 0 ]]; then
  echo "WRITER_JOB_IDS must explicitly account for prior writers" >&2
  exit 2
fi

before_file="${CUTOVER_EVIDENCE_DIR}/qstat_before.txt"
after_file="${CUTOVER_EVIDENCE_DIR}/qstat_after.txt"
"${QSTAT_BIN}" -u "${USER}" > "${before_file}"

unaccounted_ids=$(awk -v pattern="${WRITER_JOB_NAME_PATTERN}" -v known_csv="${all_writer_ids}" '
  BEGIN {
    count = split(known_csv, known, ",")
    for (item_index = 1; item_index <= count; item_index++) {
      gsub(/[[:space:]]/, "", known[item_index])
      accounted[known[item_index]] = 1
    }
  }
  NR > 2 && $3 ~ pattern && !($1 in accounted) {print $1}
' "${before_file}")
if [[ -n "${unaccounted_ids}" ]]; then
  echo "Unaccounted possible R3 writer jobs: ${unaccounted_ids}" >&2
  exit 3
fi

active_ids=()
for raw_job_id in "${writer_ids[@]}"; do
  job_id="${raw_job_id//[[:space:]]/}"
  if [[ -n "${job_id}" ]] && awk -v id="${job_id}" '$1 == id {found=1} END {exit !found}' "${before_file}"; then
    active_ids+=("${job_id}")
  fi
done

if [[ "${#active_ids[@]}" -gt 0 && "${CUTOVER_ACTION}" == "verify" ]]; then
  echo "Canonical writers are still active: ${active_ids[*]}" >&2
  echo "Run the isolated smoke/retry/negative checks, then explicitly use CUTOVER_ACTION=cancel or wait." >&2
  exit 3
fi
if [[ "${#active_ids[@]}" -gt 0 ]]; then
  "${QDEL_BIN}" "${active_ids[@]}"
fi

for ((poll=1; poll<=QUIESCENCE_POLLS; poll++)); do
  "${QSTAT_BIN}" -u "${USER}" > "${after_file}"
  still_active=0
  for raw_job_id in "${writer_ids[@]}"; do
    job_id="${raw_job_id//[[:space:]]/}"
    if [[ -n "${job_id}" ]] && awk -v id="${job_id}" '$1 == id {found=1} END {exit !found}' "${after_file}"; then
      still_active=1
      break
    fi
  done
  if [[ "${still_active}" -eq 0 ]]; then
    break
  fi
  sleep "${QUIESCENCE_SLEEP_SECONDS}"
done
if [[ "${still_active:-1}" -ne 0 ]]; then
  echo "Timed out waiting for canonical writers to leave qstat" >&2
  exit 4
fi

qacct_dir="${CUTOVER_EVIDENCE_DIR}/qacct"
mkdir -p "${qacct_dir}"
for raw_job_id in "${writer_ids[@]}"; do
  job_id="${raw_job_id//[[:space:]]/}"
  [[ -z "${job_id}" ]] && continue
  accounted=0
  for ((poll=1; poll<=QUIESCENCE_POLLS; poll++)); do
    if "${QACCT_BIN}" -j "${job_id}" > "${qacct_dir}/${job_id}.txt" 2>&1; then
      accounted=1
      break
    fi
    sleep "${QUIESCENCE_SLEEP_SECONDS}"
  done
  if [[ "${accounted}" -ne 1 ]]; then
    echo "No qacct evidence for writer ${job_id}; refusing freeze" >&2
    exit 5
  fi
done

final_file="${CUTOVER_EVIDENCE_DIR}/qstat_final.txt"
"${QSTAT_BIN}" -u "${USER}" > "${final_file}"
final_unaccounted_ids=$(awk -v pattern="${WRITER_JOB_NAME_PATTERN}" -v known_csv="${all_writer_ids}" '
  BEGIN {
    count = split(known_csv, known, ",")
    for (item_index = 1; item_index <= count; item_index++) {
      gsub(/[[:space:]]/, "", known[item_index])
      accounted[known[item_index]] = 1
    }
  }
  NR > 2 && (($1 in accounted) || $3 ~ pattern) {print $1}
' "${final_file}")
if [[ -n "${final_unaccounted_ids}" ]]; then
  echo "Final qstat found a known or successor R3 writer: ${final_unaccounted_ids}" >&2
  exit 6
fi

evidence_file="${CUTOVER_EVIDENCE_DIR}/writer_quiescence.json"
"${PYTHON_BIN}" - "${evidence_file}" "${before_file}" "${after_file}" "${final_file}" "${qacct_dir}" "${CUTOVER_ACTION}" "${all_writer_ids}" "${SUCCESSOR_JOB_IDS}" "${WRITER_JOB_NAME_PATTERN}" <<'PY'
import hashlib
import json
import pathlib
import sys

output, before, after, final, qacct_dir, action, raw_ids, raw_successors, pattern = sys.argv[1:]
job_ids = [value.strip() for value in raw_ids.split(",") if value.strip()]
successor_ids = {value.strip() for value in raw_successors.split(",") if value.strip()}
def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def file_record(path):
    value = pathlib.Path(path).resolve()
    return {"path": str(value), "size": value.stat().st_size, "sha256": sha(value)}
def job_record(job_id):
    return {
        "job_id": job_id,
        "can_write": False,
        "scheduler_disposition": "absent_from_final_qstat_and_present_in_qacct",
        "qacct": file_record(pathlib.Path(qacct_dir) / f"{job_id}.txt"),
    }
payload = {
    "writers_quiescent": True,
    "cutover_action": action,
    "accounted_job_ids": job_ids,
    "writer_job_name_pattern": pattern,
    "qstat_before": file_record(before),
    "qstat_after": file_record(after),
    "qstat_final": file_record(final),
    "jobs": [job_record(job_id) for job_id in job_ids if job_id not in successor_ids],
    "successors": [job_record(job_id) for job_id in job_ids if job_id in successor_ids],
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
payload["evidence_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

freeze_command=(
  "${PYTHON_BIN}" scripts/r3_episode_array_workflow.py freeze
  --canonical-run-dir "${CANONICAL_OUTPUT_DIR}"
  --workflow-run-dir "${WORKFLOW_RUN_DIR}"
  --task-manifest "${TASK_MANIFEST}"
  --writer-evidence "${evidence_file}"
  --git-commit "${REVIEWED_COMMIT}"
  --reviewed-commit "${REVIEWED_COMMIT}"
  --run-mode production
  --array-lanes "${ARRAY_LANES}"
  --lane-throttle "${LANE_THROTTLE}"
  --expected-task-count "${EXPECTED_TASK_COUNT}"
)
for raw_job_id in "${writer_ids[@]}"; do
  job_id="${raw_job_id//[[:space:]]/}"
  [[ -n "${job_id}" ]] && freeze_command+=(--required-writer-job-id "${job_id}")
done
"${freeze_command[@]}"

echo "Checkpoint frozen after verified writer quiescence."
echo "No episode jobs have been submitted yet."
