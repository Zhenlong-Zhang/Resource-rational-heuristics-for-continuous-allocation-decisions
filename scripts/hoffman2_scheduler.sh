#!/usr/bin/env bash

# Fail-closed scheduler helpers shared by the StrategyMapping submitter and its tests.

STRATEGY_MAPPING_QSUB_BIN="${STRATEGY_MAPPING_QSUB_BIN:-qsub}"
STRATEGY_MAPPING_QSTAT_BIN="${STRATEGY_MAPPING_QSTAT_BIN:-qstat}"
STRATEGY_MAPPING_QDEL_BIN="${STRATEGY_MAPPING_QDEL_BIN:-qdel}"
STRATEGY_MAPPING_SCHEDULER_PYTHON="${STRATEGY_MAPPING_SCHEDULER_PYTHON:-${PYTHON_BIN:-python3}}"
STRATEGY_MAPPING_SCHEDULER_USER="${STRATEGY_MAPPING_SCHEDULER_USER:-${USER:-$(id -un)}}"

declare -a STRATEGY_MAPPING_SUBMITTED_JOB_IDS=()

strategy_mapping_init_submission_tracking() {
  STRATEGY_MAPPING_SUBMISSION_EVIDENCE_DIR="$1"
  mkdir -p "${STRATEGY_MAPPING_SUBMISSION_EVIDENCE_DIR}"
  STRATEGY_MAPPING_SUBMITTED_JOB_IDS=()
  STRATEGY_MAPPING_LAST_JOB_ID=""
}

strategy_mapping_append_job_id() {
  local candidate="$1"
  local existing
  for existing in "${STRATEGY_MAPPING_SUBMITTED_JOB_IDS[@]:-}"; do
    if [[ "${existing}" == "${candidate}" ]]; then
      return 0
    fi
  done
  STRATEGY_MAPPING_SUBMITTED_JOB_IDS+=("${candidate}")
}

strategy_mapping_job_ids_named() {
  local job_name="$1"
  local xml_file
  xml_file="$(mktemp "${STRATEGY_MAPPING_SUBMISSION_EVIDENCE_DIR}/.qstat.XXXXXX.xml")"
  if ! "${STRATEGY_MAPPING_QSTAT_BIN}" -xml -u "${STRATEGY_MAPPING_SCHEDULER_USER}" >"${xml_file}" 2>/dev/null; then
    rm -f "${xml_file}"
    return 1
  fi
  "${STRATEGY_MAPPING_SCHEDULER_PYTHON}" - "${xml_file}" "${job_name}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
target = sys.argv[2]
seen = set()
for job in root.iter("job_list"):
    name = job.findtext("JB_name")
    number = job.findtext("JB_job_number")
    if name == target and number and number.isdigit() and number not in seen:
        seen.add(number)
        print(number)
PY
  local status=$?
  rm -f "${xml_file}"
  return "${status}"
}

strategy_mapping_cancel_ids() {
  if [[ "$#" -gt 0 ]]; then
    "${STRATEGY_MAPPING_QDEL_BIN}" "$@" >/dev/null 2>&1 || true
  fi
}

strategy_mapping_rollback_partial_submission() {
  if [[ "${#STRATEGY_MAPPING_SUBMITTED_JOB_IDS[@]}" -gt 0 ]]; then
    strategy_mapping_cancel_ids "${STRATEGY_MAPPING_SUBMITTED_JOB_IDS[@]}"
    printf 'Rolled back partial StrategyMapping submission: %s\n' \
      "${STRATEGY_MAPPING_SUBMITTED_JOB_IDS[*]}" >&2
  fi
}

strategy_mapping_cleanup_unparseable_submission() {
  local job_name="$1"
  local -a discovered=()
  local attempt job_id query_output remaining

  # The unique name is a second recovery handle when terse output is malformed.
  "${STRATEGY_MAPPING_QDEL_BIN}" "${job_name}" >/dev/null 2>&1 || true
  for attempt in 1 2 3 4 5; do
    if query_output="$(strategy_mapping_job_ids_named "${job_name}")"; then
      if [[ -z "${query_output}" ]]; then
        return 0
      fi
      discovered=()
      while IFS= read -r job_id; do
        if [[ -n "${job_id}" ]]; then
          discovered+=("${job_id}")
          strategy_mapping_append_job_id "${job_id}"
        fi
      done <<<"${query_output}"
      strategy_mapping_cancel_ids "${discovered[@]}"
    fi
    sleep 1
  done

  if ! remaining="$(strategy_mapping_job_ids_named "${job_name}")"; then
    printf 'Unable to verify cleanup for scheduler job name %s: qstat failed.\n' \
      "${job_name}" >&2
    return 1
  fi
  if [[ -n "${remaining}" ]]; then
    printf 'Unable to prove cleanup for scheduler job name %s; remaining IDs: %s\n' \
      "${job_name}" "${remaining}" >&2
    return 1
  fi
}

strategy_mapping_submit_job() {
  local label="$1"
  local job_name="$2"
  local job_file="$3"
  local stdout_file="${STRATEGY_MAPPING_SUBMISSION_EVIDENCE_DIR}/${label}.qsub.stdout"
  local stderr_file="${STRATEGY_MAPPING_SUBMISSION_EVIDENCE_DIR}/${label}.qsub.stderr"
  local metadata_file="${STRATEGY_MAPPING_SUBMISSION_EVIDENCE_DIR}/${label}.qsub.meta"
  local output job_id exit_status

  if env LC_ALL=C LANG=C "${STRATEGY_MAPPING_QSUB_BIN}" -terse -N "${job_name}" "${job_file}" \
    >"${stdout_file}" 2>"${stderr_file}"; then
    exit_status=0
  else
    exit_status=$?
  fi
  output="$(tr -d '\r\n' <"${stdout_file}")"
  {
    printf 'label=%s\n' "${label}"
    printf 'job_name=%s\n' "${job_name}"
    printf 'job_file=%s\n' "${job_file}"
    printf 'qsub_exit_status=%s\n' "${exit_status}"
    printf 'terse_output=%s\n' "${output}"
  } >"${metadata_file}"
  cat "${stderr_file}" >&2

  if [[ "${exit_status}" -ne 0 ]]; then
    printf 'qsub failed for %s with exit status %s.\n' \
      "${label}" "${exit_status}" >&2
    return "${exit_status}"
  fi
  if [[ "${output}" =~ ^([0-9]+)(\.[0-9]+-[0-9]+(:[0-9]+)?)?$ ]]; then
    job_id="${BASH_REMATCH[1]}"
    strategy_mapping_append_job_id "${job_id}"
    STRATEGY_MAPPING_LAST_JOB_ID="${job_id}"
    printf '%s\n' "${output}" >&2
    return 0
  fi

  printf 'Malformed terse qsub output for %s: %s\n' "${label}" "${output}" >&2
  local cleanup_status=0
  if ! strategy_mapping_cleanup_unparseable_submission "${job_name}"; then
    cleanup_status=1
  fi
  strategy_mapping_rollback_partial_submission
  if [[ "${cleanup_status}" -ne 0 ]]; then
    printf 'Scheduler cleanup could not be verified for %s; manual intervention is required.\n' \
      "${job_name}" >&2
    return 70
  fi
  return 1
}
