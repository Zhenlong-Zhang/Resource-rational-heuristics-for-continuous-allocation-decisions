#!/usr/bin/env bash
# Capture immutable scheduler evidence and audit a completed P2 smoke setup.

set -euo pipefail

: "${SETUP_ROOT:?Set the completed immutable SETUP_ROOT}"
: "${COMPUTE_CEILING:?Set the Compute Ceiling Report bound by the setup}"
: "${COMPUTE_CEILING_EVIDENCE_ROOT:?Set the raw Compute Ceiling evidence directory}"

PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
QSTAT_BIN="${QSTAT_BIN:-qstat}"
QACCT_BIN="${QACCT_BIN:-qacct}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_ROOT="$(cd "${SETUP_ROOT}" && pwd)"
COMPUTE_CEILING="$(cd "$(dirname "${COMPUTE_CEILING}")" && pwd)/$(basename "${COMPUTE_CEILING}")"
COMPUTE_CEILING_EVIDENCE_ROOT="$(cd "${COMPUTE_CEILING_EVIDENCE_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

if [[ -e "${SETUP_ROOT}/setup_audit.json" || -e "${SETUP_ROOT}/qacct" || \
      -e "${SETUP_ROOT}/qacct_attempts" || -e "${SETUP_ROOT}/final_qstat" ]]; then
  echo "Refusing to overwrite terminal setup audit evidence." >&2
  exit 1
fi
mkdir "${SETUP_ROOT}/qacct" "${SETUP_ROOT}/qacct_attempts" "${SETUP_ROOT}/final_qstat"
qstat_xml="${SETUP_ROOT}/final_qstat/snapshot.xml"
set +e
"${QSTAT_BIN}" -xml -u "$(id -un)" > "${qstat_xml}" 2>&1
qstat_status=$?
set -e
printf '%s\n' "${qstat_status}" > "${SETUP_ROOT}/final_qstat/snapshot.status"
if [[ "${qstat_status}" -ne 0 ]]; then
  echo "Unable to capture an authoritative qstat XML snapshot." >&2
  exit 2
fi

while IFS=$'\t' read -r role job_id _job_file _qsub_file; do
  final="${SETUP_ROOT}/qacct/${role}.raw"
  expected=16
  [[ "${role}" == "plan_merge" ]] && expected=1
  complete=0
  for attempt in $(seq 1 30); do
    attempt_path="${SETUP_ROOT}/qacct_attempts/${role}.$(printf '%02d' "${attempt}").raw"
    if "${QACCT_BIN}" -j "${job_id}" > "${attempt_path}" && \
       "${PYTHON_BIN}" - "${attempt_path}" "${expected}" <<'PY'
from pathlib import Path
import sys
from src.experiments.terminal_execution import parse_qacct_records
records = parse_qacct_records(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if len(records) == int(sys.argv[2]) else 1)
PY
    then
      ln "${attempt_path}" "${final}"
      complete=1
      break
    fi
    sleep 10
  done
  if [[ "${complete}" -ne 1 ]]; then
    echo "Complete qacct evidence did not become available for setup job ${job_id}." >&2
    exit 2
  fi
done < "${SETUP_ROOT}/setup_submissions.tsv"

"${PYTHON_BIN}" scripts/audit_terminal_manifest_setup.py \
  --setup-root "${SETUP_ROOT}" --compute-ceiling "${COMPUTE_CEILING}" \
  --compute-ceiling-evidence-root "${COMPUTE_CEILING_EVIDENCE_ROOT}" \
  --output "${SETUP_ROOT}/setup_audit.json"
printf 'P2 terminal manifest setup audit passed: %s\n' "${SETUP_ROOT}/setup_audit.json"
