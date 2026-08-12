#!/usr/bin/env bash
# Submit or audit two immutable plan-only diagnostic replicates on Hoffman2.

set -euo pipefail

: "${ACTION:?Set ACTION to submit or audit}"
: "${RUN_ROOT:?Set a new RUN_ROOT for submit, or an existing RUN_ROOT for audit}"

PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
QSUB_BIN="${QSUB_BIN:-qsub}"
QDEL_BIN="${QDEL_BIN:-qdel}"
QSTAT_BIN="${QSTAT_BIN:-qstat}"
QACCT_BIN="${QACCT_BIN:-qacct}"
QUEUE="${QUEUE:-campus2.q}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$(cd "$(dirname "${RUN_ROOT}")" && pwd)/$(basename "${RUN_ROOT}")"
cd "${PROJECT_ROOT}"

if [[ "${ACTION}" != "submit" && "${ACTION}" != "audit" ]]; then
  echo "ACTION must be submit or audit." >&2
  exit 1
fi

submit_run() {
  if [[ -e "${RUN_ROOT}" ]]; then
    echo "Refusing to overwrite plan diagnostic run: ${RUN_ROOT}" >&2
    exit 1
  fi
  if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all)" ]]; then
    echo "Plan diagnostics require a clean committed worktree." >&2
    exit 1
  fi
  read -r commit tree source_identity_hash provider_hash < <(
    "${PYTHON_BIN}" - "${PROJECT_ROOT}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from src.experiments import terminal_execution as execution
source = execution.capture_clean_source_identity(root, execution.TERMINAL_SOURCE_PATHS)
provider, _ = execution.load_accepted_canonical_base_provider()
print(source["commit"], source["tree"], source["identity_hash"], provider.provider_hash)
PY
  )

  mkdir -p \
    "${RUN_ROOT}/jobs" \
    "${RUN_ROOT}/logs" \
    "${RUN_ROOT}/qsub_raw" \
    "${RUN_ROOT}/replicate_a" \
    "${RUN_ROOT}/replicate_b"
  declare -a submitted_jobs=()
  declare -a replicates=()
  declare -a job_ids=()

  rollback() {
    if [[ "${#submitted_jobs[@]}" -gt 0 ]]; then
      "${QDEL_BIN}" "${submitted_jobs[@]}" >/dev/null 2>&1 || true
    fi
  }
  trap rollback ERR

  for replicate in a b; do
    job_file="${RUN_ROOT}/jobs/plan_${replicate}.job"
    cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N tvp1_${replicate}
#$ -q ${QUEUE}
#$ -j y
#$ -o ${RUN_ROOT}/logs/plan_${replicate}.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=00:10:00
#$ -l h_data=2G
#$ -t 1-16
#$ -tc 16
set -euo pipefail
export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/terminal_validation_array.py diagnose-plan \
  --stage smoke --descriptor-position "\${SGE_TASK_ID}" --mode plan-only \
  --output "${RUN_ROOT}/replicate_${replicate}/diagnostic_\$(printf '%03d' \${SGE_TASK_ID}).json"
EOF
    chmod 500 "${job_file}"
    raw_file="${RUN_ROOT}/qsub_raw/plan_${replicate}.txt"
    "${QSUB_BIN}" -terse "${job_file}" > "${raw_file}"
    raw="$(tr -d '\r\n' < "${raw_file}")"
    job_id="$(printf '%s' "${raw}" | sed -E 's/^([0-9]+).*/\1/')"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
      echo "Unable to parse qsub job ID: ${raw}" >&2
      exit 1
    fi
    submitted_jobs+=("${job_id}")
    replicates+=("${replicate}")
    job_ids+=("${job_id}")
  done

  "${PYTHON_BIN}" - "${RUN_ROOT}/run_metadata.json" "${QUEUE}" "${commit}" "${tree}" \
    "${source_identity_hash}" "${provider_hash}" \
    "${replicates[0]}" "${job_ids[0]}" "${replicates[1]}" "${job_ids[1]}" <<'PY'
import json, os, sys, tempfile
path, queue, commit, tree, source_identity, provider, rep_a, job_a, rep_b, job_b = sys.argv[1:]
payload = {
    "schema": "terminal_validation_plan_diagnostic_run_v1",
    "stage": "smoke",
    "descriptor_count": 16,
    "queue": queue,
    "requested_slots": 1,
    "requested_memory_bytes": 2 * 1024**3,
    "max_wall_seconds": 300,
    "source_commit": commit,
    "source_tree": tree,
    "source_identity_hash": source_identity,
    "provider_hash": provider,
    "jobs": [
        {"replicate": rep_a, "job_id": job_a},
        {"replicate": rep_b, "job_id": job_b},
    ],
}
encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
directory = os.path.dirname(path)
fd, temporary = tempfile.mkstemp(prefix=".run_metadata.", dir=directory, text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
finally:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
PY
  trap - ERR
  printf 'Submitted plan-only P1 replicates as jobs %s and %s.\n' \
    "${job_ids[0]}" "${job_ids[1]}"
}

audit_run() {
  if [[ ! -f "${RUN_ROOT}/run_metadata.json" ]]; then
    echo "Plan diagnostic run metadata is missing." >&2
    exit 1
  fi
  if [[ -e "${RUN_ROOT}/audit.json" || -e "${RUN_ROOT}/qacct" ]]; then
    echo "Refusing to overwrite plan diagnostic audit evidence." >&2
    exit 1
  fi
  mkdir "${RUN_ROOT}/qacct"
  mkdir "${RUN_ROOT}/final_qstat"
  while read -r replicate job_id; do
    qstat_raw="${RUN_ROOT}/final_qstat/plan_${replicate}.raw"
    set +e
    "${QSTAT_BIN}" -j "${job_id}" > "${qstat_raw}" 2>&1
    qstat_status=$?
    set -e
    printf '%s\n' "${qstat_status}" > "${RUN_ROOT}/final_qstat/plan_${replicate}.status"
    if [[ "${qstat_status}" -eq 0 ]]; then
      echo "Job ${job_id} is still visible in qstat; audit is not allowed." >&2
      exit 2
    fi
    if ! "${PYTHON_BIN}" - "${qstat_raw}" "${job_id}" "${qstat_status}" <<'PY'
from pathlib import Path
import sys
from src.experiments.terminal_plan_diagnostics import validate_qstat_absence_text
validate_qstat_absence_text(Path(sys.argv[1]).read_text(encoding="utf-8"), sys.argv[2], int(sys.argv[3]))
PY
    then
      echo "qstat did not authoritatively prove that job ${job_id} is absent." >&2
      exit 2
    fi
    temporary="${RUN_ROOT}/qacct/.plan_${replicate}.raw.tmp.$$"
    if ! "${QACCT_BIN}" -j "${job_id}" > "${temporary}"; then
      rm -f "${temporary}"
      echo "qacct failed for job ${job_id}." >&2
      exit 2
    fi
    ln "${temporary}" "${RUN_ROOT}/qacct/plan_${replicate}.raw"
    rm -f "${temporary}"
  done < <("${PYTHON_BIN}" - "${RUN_ROOT}/run_metadata.json" <<'PY'
import json, sys
metadata = json.load(open(sys.argv[1], encoding="utf-8"))
for job in metadata["jobs"]:
    print(job["replicate"], job["job_id"])
PY
  )
  "${PYTHON_BIN}" scripts/audit_terminal_plan_diagnostics.py \
    --run-root "${RUN_ROOT}" --output "${RUN_ROOT}/audit.json"
  printf 'P1 plan-only diagnostic audit passed: %s\n' "${RUN_ROOT}/audit.json"
}

if [[ "${ACTION}" == "submit" ]]; then
  submit_run
else
  audit_run
fi
