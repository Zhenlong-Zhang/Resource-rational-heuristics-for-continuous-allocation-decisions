#!/usr/bin/env bash
# Capture immutable Hoffman2 scheduler/resource evidence and derive its bounded report.

set -euo pipefail

: "${EVIDENCE_ROOT:?Set a new immutable EVIDENCE_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
QUEUE="${QUEUE:-campus2.q}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVIDENCE_ROOT="$(cd "$(dirname "${EVIDENCE_ROOT}")" && pwd)/$(basename "${EVIDENCE_ROOT}")"
if [[ -e "${EVIDENCE_ROOT}" ]]; then
  echo "Refusing to overwrite compute-ceiling evidence: ${EVIDENCE_ROOT}" >&2
  exit 1
fi
mkdir "${EVIDENCE_ROOT}"
export LANG=C LC_ALL=C

myresources > "${EVIDENCE_ROOT}/myresources.raw"
qconf -sconf > "${EVIDENCE_ROOT}/qconf_sconf_global.raw"
qconf -sq "${QUEUE}" > "${EVIDENCE_ROOT}/qconf_sq_campus2.raw"
qhost > "${EVIDENCE_ROOT}/qhost.raw"
qquota -u "$(id -un)" > "${EVIDENCE_ROOT}/qquota.raw"
qstat -g c > "${EVIDENCE_ROOT}/qstat_g_c.raw"
qstat -xml -u "$(id -un)" > "${EVIDENCE_ROOT}/qstat_user.xml"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/create_hoffman2_compute_ceiling_report.py \
  --evidence-root "${EVIDENCE_ROOT}" --queue "${QUEUE}" \
  --output "${EVIDENCE_ROOT}/compute_ceiling_report.json"
printf 'Compute-ceiling evidence captured: %s\n' "${EVIDENCE_ROOT}"
