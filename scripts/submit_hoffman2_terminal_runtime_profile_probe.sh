#!/usr/bin/env bash
set -euo pipefail

# Submit or collect one immutable one-slot Hoffman2 runtime-profile probe.

export LANG=C
export LC_ALL=C
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONSTARTUP PYTHONINSPECT PYTHONHOME

ACTION="${1:-}"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/terminal_runtime_profile_probe_runs}"
QSUB_BIN="${QSUB_BIN:-qsub}"
QSTAT_BIN="${QSTAT_BIN:-qstat}"
QACCT_BIN="${QACCT_BIN:-qacct}"

require_approved_hashes() {
  : "${APPROVED_EVIDENCE_MODULE_HASH:?Set the Reviewer-approved evidence-module SHA-256}"
  : "${APPROVED_PROBE_COLLECTOR_HASH:?Set the Reviewer-approved collector SHA-256}"
  : "${APPROVED_PROBE_JOB_SCRIPT_HASH:?Set the Reviewer-approved probe job-script SHA-256}"
  : "${APPROVED_PROBE_SUBMITTER_HASH:?Set the Reviewer-approved submitter SHA-256}"
  for value in "${APPROVED_EVIDENCE_MODULE_HASH}" \
    "${APPROVED_PROBE_COLLECTOR_HASH}" \
    "${APPROVED_PROBE_JOB_SCRIPT_HASH}" \
    "${APPROVED_PROBE_SUBMITTER_HASH}"; do
    [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || {
      echo "Malformed externally approved runtime-probe hash." >&2
      exit 2
    }
  done
}

check_approved_file() {
  local path="$1"
  local expected_hash="$2"
  local label="$3"
  [[ -f "${path}" ]] || { echo "Missing approved ${label}: ${path}" >&2; exit 2; }
  local actual_hash
  actual_hash="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual_hash}" == "${expected_hash}" ]] || {
    echo "${label} differs from the externally approved hash." >&2
    exit 2
  }
}

check_approved_tree() {
  local root="$1"
  local context="$2"
  check_approved_file \
    "${root}/scripts/collect_hoffman2_runtime_profile_probe.py" \
    "${APPROVED_PROBE_COLLECTOR_HASH}" "${context} collector"
  check_approved_file \
    "${root}/scripts/hoffman2_terminal_runtime_profile_probe.job" \
    "${APPROVED_PROBE_JOB_SCRIPT_HASH}" "${context} probe job"
  check_approved_file \
    "${root}/scripts/submit_hoffman2_terminal_runtime_profile_probe.sh" \
    "${APPROVED_PROBE_SUBMITTER_HASH}" "${context} submitter"
  check_approved_file \
    "${root}/src/experiments/terminal_migration_evidence.py" \
    "${APPROVED_EVIDENCE_MODULE_HASH}" "${context} evidence module"
}

APPROVED_HASH_ARGS=(
  --approved-evidence-module-hash "${APPROVED_EVIDENCE_MODULE_HASH:-}"
  --approved-collector-hash "${APPROVED_PROBE_COLLECTOR_HASH:-}"
  --approved-probe-job-script-hash "${APPROVED_PROBE_JOB_SCRIPT_HASH:-}"
  --approved-submitter-hash "${APPROVED_PROBE_SUBMITTER_HASH:-}"
)

if [[ "${ACTION}" == "submit" ]]; then
  : "${RUN_ID:?Set a unique immutable RUN_ID}"
  : "${PYTHON_BIN:?Set the Hoffman2 Python executable to probe}"
  : "${CONDA_SH:?Set conda.sh for explicit activation}"
  : "${CONDA_ENV_PATH:?Set the conda environment path to probe}"
  require_approved_hashes
  APPROVED_HASH_ARGS=(
    --approved-evidence-module-hash "${APPROVED_EVIDENCE_MODULE_HASH}"
    --approved-collector-hash "${APPROVED_PROBE_COLLECTOR_HASH}"
    --approved-probe-job-script-hash "${APPROVED_PROBE_JOB_SCRIPT_HASH}"
    --approved-submitter-hash "${APPROVED_PROBE_SUBMITTER_HASH}"
  )
  check_approved_tree "${SOURCE_ROOT}" "source"
  for value in "${PYTHON_BIN}" "${CONDA_SH}" "${CONDA_ENV_PATH}"; do
    if [[ "${value}" == *','* || "${value}" == *$'\n'* ]]; then
      echo "Unsafe qsub environment value: ${value}" >&2
      exit 2
    fi
  done

  mkdir -p "${RUNS_ROOT}"
  RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
  mkdir "${RUN_DIR}"
  EVIDENCE_DIR="${RUN_DIR}/submission_evidence"
  mkdir -p "${EVIDENCE_DIR}/scripts" "${EVIDENCE_DIR}/src/experiments"
  cp "${SOURCE_ROOT}/scripts/hoffman2_terminal_runtime_profile_probe.job" \
    "${EVIDENCE_DIR}/scripts/hoffman2_terminal_runtime_profile_probe.job"
  cp "${SOURCE_ROOT}/scripts/submit_hoffman2_terminal_runtime_profile_probe.sh" \
    "${EVIDENCE_DIR}/scripts/submit_hoffman2_terminal_runtime_profile_probe.sh"
  cp "${SOURCE_ROOT}/scripts/collect_hoffman2_runtime_profile_probe.py" \
    "${EVIDENCE_DIR}/scripts/collect_hoffman2_runtime_profile_probe.py"
  cp "${SOURCE_ROOT}/src/experiments/terminal_migration_evidence.py" \
    "${EVIDENCE_DIR}/src/experiments/terminal_migration_evidence.py"
  check_approved_tree "${EVIDENCE_DIR}" "evidence copy"
  FROZEN_JOB_SCRIPT="${EVIDENCE_DIR}/scripts/hoffman2_terminal_runtime_profile_probe.job"
  FROZEN_COLLECTOR="${EVIDENCE_DIR}/scripts/collect_hoffman2_runtime_profile_probe.py"
  chmod 0555 \
    "${FROZEN_JOB_SCRIPT}" \
    "${FROZEN_COLLECTOR}" \
    "${EVIDENCE_DIR}/scripts/submit_hoffman2_terminal_runtime_profile_probe.sh"
  chmod 0444 "${EVIDENCE_DIR}/src/experiments/terminal_migration_evidence.py"
  for value in "${RUN_DIR}" "${PYTHON_BIN}" "${CONDA_SH}" \
    "${CONDA_ENV_PATH}" "${FROZEN_JOB_SCRIPT}"; do
    if [[ "${value}" == *','* || "${value}" == *$'\n'* ]]; then
      echo "Unsafe qsub environment value: ${value}" >&2
      exit 2
    fi
  done
  printf '%s\n' "${APPROVED_PROBE_JOB_SCRIPT_HASH}" > \
    "${EVIDENCE_DIR}/approved_probe_job_script_sha256.txt"
  python3 "${FROZEN_COLLECTOR}" preflight \
    --source-root "${SOURCE_ROOT}" \
    --evidence-root "${EVIDENCE_DIR}" \
    --python-executable "$(readlink -f "${PYTHON_BIN}")" \
    --conda-env-path "${CONDA_ENV_PATH}" \
    --preflight "${EVIDENCE_DIR}/preflight.json" \
    "${APPROVED_HASH_ARGS[@]}"

  JOB_STDOUT="${RUN_DIR}/job.stdout"
  JOB_STDERR="${RUN_DIR}/job.stderr"
  QSUB_STDOUT="${EVIDENCE_DIR}/qsub.stdout"
  QSUB_STDERR="${EVIDENCE_DIR}/qsub.stderr"
  QSUB_COMMAND="${EVIDENCE_DIR}/qsub.command"
  for path in "${JOB_STDOUT}" "${JOB_STDERR}" "${QSUB_STDOUT}" \
    "${QSUB_STDERR}" "${QSUB_COMMAND}"; do
    [[ ! -e "${path}" ]] || { echo "Runtime-probe evidence already exists: ${path}" >&2; exit 2; }
  done

  QSUB_ARGS=(
    -terse -N terminal_runtime_profile_probe -cwd -pe shared 1
    -l h_rt=00:05:00,h_data=1G -r n -j n
    -o "${JOB_STDOUT}" -e "${JOB_STDERR}"
    -v "RUN_DIR=${RUN_DIR},PYTHON_BIN=${PYTHON_BIN},CONDA_SH=${CONDA_SH},CONDA_ENV_PATH=${CONDA_ENV_PATH},FROZEN_JOB_SCRIPT=${FROZEN_JOB_SCRIPT}"
    "${FROZEN_JOB_SCRIPT}"
  )
  {
    printf '%q ' "${QSUB_BIN}" "${QSUB_ARGS[@]}"
    printf '\n'
  } > "${QSUB_COMMAND}"
  set +e
  "${QSUB_BIN}" "${QSUB_ARGS[@]}" > "${QSUB_STDOUT}" 2> "${QSUB_STDERR}"
  QSUB_STATUS=$?
  set -e
  printf '%s\n' "${QSUB_STATUS}" > "${EVIDENCE_DIR}/qsub.exit_status"
  [[ "${QSUB_STATUS}" -eq 0 ]] || { echo "Runtime-profile qsub failed." >&2; exit "${QSUB_STATUS}"; }
  JOB_ID="$(tr -d '\r\n' < "${QSUB_STDOUT}")"
  [[ "${JOB_ID}" =~ ^[0-9]+$ && "${JOB_ID}" != "0" ]] || {
    echo "Malformed runtime-profile probe job ID." >&2
    exit 2
  }
  printf '%s\n' "${JOB_ID}" > "${EVIDENCE_DIR}/job_id.txt"
  printf 'run_dir=%s\njob_id=%s\n' "${RUN_DIR}" "${JOB_ID}"

elif [[ "${ACTION}" == "collect" ]]; then
  : "${RUN_DIR:?Set the immutable runtime-profile probe RUN_DIR}"
  require_approved_hashes
  APPROVED_HASH_ARGS=(
    --approved-evidence-module-hash "${APPROVED_EVIDENCE_MODULE_HASH}"
    --approved-collector-hash "${APPROVED_PROBE_COLLECTOR_HASH}"
    --approved-probe-job-script-hash "${APPROVED_PROBE_JOB_SCRIPT_HASH}"
    --approved-submitter-hash "${APPROVED_PROBE_SUBMITTER_HASH}"
  )
  check_approved_tree "${SOURCE_ROOT}" "source"
  EVIDENCE_DIR="${RUN_DIR}/submission_evidence"
  check_approved_tree "${EVIDENCE_DIR}" "evidence copy"
  FROZEN_COLLECTOR="${EVIDENCE_DIR}/scripts/collect_hoffman2_runtime_profile_probe.py"
  JOB_ID_FILE="${RUN_DIR}/submission_evidence/job_id.txt"
  [[ -f "${JOB_ID_FILE}" ]] || { echo "Missing runtime-probe job ID." >&2; exit 2; }
  JOB_ID="$(tr -d '\r\n' < "${JOB_ID_FILE}")"
  [[ "${JOB_ID}" =~ ^[0-9]+$ && "${JOB_ID}" != "0" ]] || {
    echo "Malformed runtime-probe job ID." >&2
    exit 2
  }
  if "${QSTAT_BIN}" -j "${JOB_ID}" >/dev/null 2>&1; then
    echo "Runtime-profile probe ${JOB_ID} is still in qstat." >&2
    exit 75
  fi
  QACCT_RAW="${RUN_DIR}/submission_evidence/qacct.raw"
  GATE_JSON="${RUN_DIR}/runtime_profile_scheduler_gate.json"
  [[ ! -e "${QACCT_RAW}" && ! -e "${GATE_JSON}" ]] || {
    echo "This runtime-profile probe already has collector evidence." >&2
    exit 2
  }
  set +e
  "${QACCT_BIN}" -j "${JOB_ID}" > "${QACCT_RAW}" 2>&1
  QACCT_STATUS=$?
  set -e
  [[ "${QACCT_STATUS}" -eq 0 ]] || { echo "Runtime-profile qacct failed." >&2; exit 2; }
  python3 "${FROZEN_COLLECTOR}" collect \
    --run-dir "${RUN_DIR}" \
    --qacct "${QACCT_RAW}" \
    --submitted-job-id "${JOB_ID}" \
    --gate "${GATE_JSON}" \
    "${APPROVED_HASH_ARGS[@]}"
  printf 'runtime_profile_gate=%s\n' "${GATE_JSON}"
else
  echo "Usage: $0 submit | collect" >&2
  exit 2
fi
