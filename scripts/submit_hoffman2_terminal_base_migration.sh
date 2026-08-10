#!/usr/bin/env bash
set -euo pipefail

# Prepare, submit, and later qacct-gate one immutable one-slot migration run.

export LANG=C
export LC_ALL=C
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONSTARTUP PYTHONINSPECT PYTHONHOME

ACTION="${1:-}"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/terminal_base_migration_runs}"
QSUB_BIN="${QSUB_BIN:-qsub}"
QSTAT_BIN="${QSTAT_BIN:-qstat}"
QACCT_BIN="${QACCT_BIN:-qacct}"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_safe_qsub_value() {
  if [[ "$1" == *','* || "$1" == *$'\n'* ]]; then
    echo "Unsafe qsub environment value: $1" >&2
    exit 2
  fi
}

git_in_directory() {
  local directory="$1"
  shift
  (
    cd "${directory}"
    git "$@"
  )
}

if [[ "${ACTION}" == "submit" ]]; then
  : "${RUN_ID:?Set a unique immutable RUN_ID}"
  : "${APPROVAL_PATH:?Set the Reviewer-approved execution profile path}"
  : "${APPROVAL_HASH:?Set its externally recorded SHA-256}"
  : "${PYTHON_BIN:?Set the approved Hoffman2 Python executable}"
  : "${CONDA_SH:?Set the conda.sh path used for explicit activation}"
  : "${CONDA_ENV_PATH:?Set the approved conda environment path}"
  : "${ORIGINAL_MANIFEST:?Set the preserved original quadrature manifest path}"

  mkdir -p "${RUNS_ROOT}"
  RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
  mkdir "${RUN_DIR}"
  STAGE_ROOT="${RUN_DIR}/authoritative_checkout"
  EVIDENCE_DIR="${RUN_DIR}/submission_evidence"
  mkdir "${EVIDENCE_DIR}"

  if [[ "$(sha256_file "${APPROVAL_PATH}")" != "${APPROVAL_HASH}" ]]; then
    echo "Execution approval does not match the external Reviewer hash." >&2
    exit 2
  fi
  if [[ "$(sha256_file "${ORIGINAL_MANIFEST}")" != "9215d3e3823c1f01b070d6a575f214e2bff0f1617262a9445b66a451d02753d2" ]]; then
    echo "Original manifest byte hash mismatch." >&2
    exit 2
  fi

  git_in_directory "${RUN_DIR}" clone --no-checkout "${SOURCE_ROOT}" "${STAGE_ROOT}"
  git_in_directory "${STAGE_ROOT}" checkout --detach \
    7376c5d70cf2520600894853e2a1275e8d0a89e1
  mkdir -p "${STAGE_ROOT}/src/experiments" "${STAGE_ROOT}/scripts" \
    "${STAGE_ROOT}/results/r6_prefeedback_quadrature_7376c5d_v1"
  cp "${SOURCE_ROOT}/src/experiments/terminal_base_migration.py" \
    "${STAGE_ROOT}/src/experiments/terminal_base_migration.py"
  cp "${SOURCE_ROOT}/scripts/export_terminal_base_migration.py" \
    "${STAGE_ROOT}/scripts/export_terminal_base_migration.py"
  cp "${ORIGINAL_MANIFEST}" \
    "${STAGE_ROOT}/results/r6_prefeedback_quadrature_7376c5d_v1/r6_quadrature_diagnostic_manifest.json"
  cp "${SOURCE_ROOT}/scripts/hoffman2_terminal_base_migration.job" \
    "${RUN_DIR}/hoffman2_terminal_base_migration.job"
  cp "${SOURCE_ROOT}/scripts/submit_hoffman2_terminal_base_migration.sh" \
    "${EVIDENCE_DIR}/submitter.sh"
  cp "${APPROVAL_PATH}" "${EVIDENCE_DIR}/execution_approval.json"
  printf '%s\n' "${APPROVAL_HASH}" > \
    "${EVIDENCE_DIR}/approved_execution_approval_sha256.txt"

  FROZEN_JOB_SCRIPT="${RUN_DIR}/hoffman2_terminal_base_migration.job"
  FROZEN_APPROVAL_PATH="${EVIDENCE_DIR}/execution_approval.json"
  chmod 0555 "${FROZEN_JOB_SCRIPT}"
  chmod 0444 "${FROZEN_APPROVAL_PATH}"
  "${PYTHON_BIN}" - "${APPROVAL_PATH}" "${APPROVAL_HASH}" \
    "${STAGE_ROOT}" "${FROZEN_JOB_SCRIPT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

approval_path, approval_hash, stage, job = sys.argv[1:]
raw = Path(approval_path).read_bytes()
if hashlib.sha256(raw).hexdigest() != approval_hash:
    raise SystemExit("approval hash changed during staging")
approval = json.loads(raw)
expected = dict(approval["migration_tool_hashes"])
actual = {
    "scripts/export_terminal_base_migration.py": hashlib.sha256(
        (Path(stage) / "scripts/export_terminal_base_migration.py").read_bytes()
    ).hexdigest(),
    "src/experiments/terminal_base_migration.py": hashlib.sha256(
        (Path(stage) / "src/experiments/terminal_base_migration.py").read_bytes()
    ).hexdigest(),
    "scripts/hoffman2_terminal_base_migration.job": hashlib.sha256(
        Path(job).read_bytes()
    ).hexdigest(),
}
if actual != expected:
    raise SystemExit("staged migration tool hashes differ from Reviewer approval")
PY

  for value in "${STAGE_ROOT}" "${RUN_DIR}" "${FROZEN_APPROVAL_PATH}" "${APPROVAL_HASH}" \
    "${PYTHON_BIN}" "${CONDA_SH}" "${CONDA_ENV_PATH}" "${FROZEN_JOB_SCRIPT}"; do
    require_safe_qsub_value "${value}"
  done

  {
    printf 'commit=%s\n' "$(git_in_directory "${STAGE_ROOT}" rev-parse HEAD)"
    printf 'tree=%s\n' "$(git_in_directory "${STAGE_ROOT}" rev-parse 'HEAD^{tree}')"
    printf 'approval_sha256=%s\n' "${APPROVAL_HASH}"
    printf 'manifest_sha256=%s\n' "$(sha256_file "${ORIGINAL_MANIFEST}")"
    printf 'module_sha256=%s\n' "$(sha256_file "${STAGE_ROOT}/src/experiments/terminal_base_migration.py")"
    printf 'cli_sha256=%s\n' "$(sha256_file "${STAGE_ROOT}/scripts/export_terminal_base_migration.py")"
    printf 'job_script_sha256=%s\n' "$(sha256_file "${FROZEN_JOB_SCRIPT}")"
    printf 'submitter_sha256=%s\n' "$(sha256_file "${EVIDENCE_DIR}/submitter.sh")"
    printf 'python_bin=%s\n' "$(readlink -f "${PYTHON_BIN}")"
    printf 'staged_untracked=%s\n' "$(git_in_directory "${STAGE_ROOT}" ls-files --others -- | paste -sd, -)"
  } > "${EVIDENCE_DIR}/preflight.txt"
  git_in_directory "${STAGE_ROOT}" diff --quiet HEAD --
  git_in_directory "${STAGE_ROOT}" diff --cached --quiet HEAD --

  JOB_STDOUT="${RUN_DIR}/job.stdout"
  JOB_STDERR="${RUN_DIR}/job.stderr"
  QSUB_STDOUT="${EVIDENCE_DIR}/qsub.stdout"
  QSUB_STDERR="${EVIDENCE_DIR}/qsub.stderr"
  QSUB_COMMAND="${EVIDENCE_DIR}/qsub.command"
  JOB_ID_FILE="${EVIDENCE_DIR}/job_id.txt"
  for path in "${JOB_STDOUT}" "${JOB_STDERR}" "${QSUB_STDOUT}" "${QSUB_STDERR}" \
    "${QSUB_COMMAND}" "${JOB_ID_FILE}"; do
    [[ ! -e "${path}" ]] || { echo "Evidence path already exists: ${path}" >&2; exit 2; }
  done

  cd "${STAGE_ROOT}"
  QSUB_ARGS=(
    -terse -N terminal_base_migration_7376c5d -cwd -pe shared 1
    -l h_rt=00:10:00,h_data=2G -r n -j n
    -o "${JOB_STDOUT}" -e "${JOB_STDERR}"
    -v "STAGE_ROOT=${STAGE_ROOT},RUN_DIR=${RUN_DIR},APPROVAL_PATH=${FROZEN_APPROVAL_PATH},APPROVAL_HASH=${APPROVAL_HASH},PYTHON_BIN=${PYTHON_BIN},CONDA_SH=${CONDA_SH},CONDA_ENV_PATH=${CONDA_ENV_PATH},FROZEN_JOB_SCRIPT=${FROZEN_JOB_SCRIPT}"
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
  [[ "${QSUB_STATUS}" -eq 0 ]] || { echo "qsub failed; run evidence is retained." >&2; exit "${QSUB_STATUS}"; }
  JOB_ID="$(tr -d '\r\n' < "${QSUB_STDOUT}")"
  [[ "${JOB_ID}" =~ ^[0-9]+$ ]] || { echo "Malformed non-array terse job ID." >&2; exit 2; }
  printf '%s\n' "${JOB_ID}" > "${JOB_ID_FILE}"
  printf 'run_dir=%s\njob_id=%s\n' "${RUN_DIR}" "${JOB_ID}"

elif [[ "${ACTION}" == "collect" ]]; then
  : "${RUN_DIR:?Set the immutable completed RUN_DIR}"
  : "${APPROVAL_HASH:?Set the externally Reviewer-approved execution profile SHA-256}"
  JOB_ID_FILE="${RUN_DIR}/submission_evidence/job_id.txt"
  [[ -f "${JOB_ID_FILE}" ]] || { echo "Missing submitted job ID." >&2; exit 2; }
  JOB_ID="$(tr -d '\r\n' < "${JOB_ID_FILE}")"
  [[ "${JOB_ID}" =~ ^[0-9]+$ ]] || { echo "Malformed job ID." >&2; exit 2; }
  if "${QSTAT_BIN}" -j "${JOB_ID}" >/dev/null 2>&1; then
    echo "Job ${JOB_ID} is still present in qstat; no qacct evidence was written." >&2
    exit 75
  fi
  QACCT_RAW="${RUN_DIR}/submission_evidence/qacct.raw"
  GATE_JSON="${RUN_DIR}/scheduler_gate.json"
  [[ ! -e "${QACCT_RAW}" && ! -e "${GATE_JSON}" ]] || {
    echo "This run already has scheduler evidence and cannot be reused." >&2
    exit 2
  }
  set +e
  "${QACCT_BIN}" -j "${JOB_ID}" > "${QACCT_RAW}" 2>&1
  QACCT_STATUS=$?
  set -e
  [[ "${QACCT_STATUS}" -eq 0 ]] || { echo "qacct failed; retain this run as failed evidence." >&2; exit 2; }
  python3 "${SOURCE_ROOT}/scripts/collect_terminal_base_migration_evidence.py" \
    --run-dir "${RUN_DIR}" \
    --qacct "${QACCT_RAW}" \
    --submitted-job-id "${JOB_ID}" \
    --approved-execution-approval-file-hash "${APPROVAL_HASH}" \
    --gate "${GATE_JSON}"
  printf 'scheduler_gate=%s\n' "${GATE_JSON}"
else
  echo "Usage: $0 submit | collect" >&2
  exit 2
fi
