#!/usr/bin/env bash
# Submit deterministic smoke or distributed full terminal-manifest setup.

set -euo pipefail

: "${STAGE:?Set STAGE to smoke or full}"
: "${SETUP_ROOT:?Set a new immutable SETUP_ROOT}"
: "${COMPUTE_CEILING:?Set the fresh Compute Ceiling Report path}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
QSUB_BIN="${QSUB_BIN:-qsub}"
QUEUE="${QUEUE:-campus2.q}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_ROOT="$(cd "$(dirname "${SETUP_ROOT}")" && pwd)/$(basename "${SETUP_ROOT}")"
COMPUTE_CEILING="$(cd "$(dirname "${COMPUTE_CEILING}")" && pwd)/$(basename "${COMPUTE_CEILING}")"

if [[ "${STAGE}" != "smoke" && "${STAGE}" != "full" ]]; then
  echo "STAGE must be smoke or full." >&2
  exit 1
fi
if [[ -e "${SETUP_ROOT}" ]]; then
  echo "Refusing to overwrite terminal manifest setup: ${SETUP_ROOT}" >&2
  exit 1
fi
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Terminal manifest setup requires a clean committed worktree." >&2
  exit 1
fi

mkdir -p "${SETUP_ROOT}/jobs" "${SETUP_ROOT}/logs" "${SETUP_ROOT}/qsub_raw"
manifest="${SETUP_ROOT}/terminal_${STAGE}_manifest.json"
submission_file="${SETUP_ROOT}/setup_submissions.tsv"
: > "${submission_file}"

submit() {
  local role="$1" job_file="$2" raw_file="${SETUP_ROOT}/qsub_raw/${role}.txt"
  "${QSUB_BIN}" -terse "${job_file}" > "${raw_file}"
  local raw job_id
  raw="$(tr -d '\r\n' < "${raw_file}")"
  job_id="$(printf '%s' "${raw}" | sed -E 's/^([0-9]+).*/\1/')"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Unable to parse qsub job ID: ${raw}" >&2
    return 1
  fi
  printf '%s\t%s\t%s\t%s\n' "${role}" "${job_id}" "${job_file}" "${raw_file}" >> "${submission_file}"
  printf '%s' "${job_id}"
}

if [[ "${STAGE}" == "smoke" ]]; then
  job_file="${SETUP_ROOT}/jobs/smoke_setup.job"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N tv_smsetup
#$ -q ${QUEUE}
#$ -pe shared 16
#$ -j y
#$ -o ${SETUP_ROOT}/logs/smoke_setup.\$JOB_ID.log
#$ -l h_rt=24:00:00
#$ -l h_data=2G
set -euo pipefail
export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT}"
TERMINAL_MANIFEST_WORKERS=1 "${PYTHON_BIN}" -m unittest tests.test_terminal_execution
export TERMINAL_MANIFEST_WORKERS="\${NSLOTS:-16}"
"${PYTHON_BIN}" scripts/terminal_validation_array.py freeze-manifest \
  --stage smoke --output "${manifest}" --compute-ceiling "${COMPUTE_CEILING}" \
  --max-descriptors-per-subshard 450 --queue "${QUEUE}" \
  --h-rt-seconds 86400 --memory-bytes 8589934592 --throttle 4
EOF
  chmod 500 "${job_file}"
  smoke_job="$(submit smoke_setup "${job_file}")"
  printf 'Submitted smoke setup job %s.\n' "${smoke_job}"
  exit 0
fi

shard_count="${MANIFEST_PLAN_SHARDS:-90}"
if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ || "${shard_count}" -gt 2000 ]]; then
  echo "MANIFEST_PLAN_SHARDS must be between 1 and 2000." >&2
  exit 1
fi
mkdir -p "${SETUP_ROOT}/plan_a" "${SETUP_ROOT}/plan_b"

write_fragment_job() {
  local replicate="$1" target_dir="$2" job_file="$3"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N tv_plan_${replicate}
#$ -q ${QUEUE}
#$ -j y
#$ -o ${SETUP_ROOT}/logs/plan_${replicate}.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=24:00:00
#$ -l h_data=2G
#$ -t 1-${shard_count}
#$ -tc 100
set -euo pipefail
export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/terminal_validation_array.py freeze-plan-fragment \
  --stage full --shard-index "\${SGE_TASK_ID}" --shard-count "${shard_count}" \
  --output "${target_dir}/fragment_\$(printf '%03d' \${SGE_TASK_ID}).json"
EOF
  chmod 500 "${job_file}"
}

job_a="${SETUP_ROOT}/jobs/full_plan_a.job"
job_b="${SETUP_ROOT}/jobs/full_plan_b.job"
write_fragment_job a "${SETUP_ROOT}/plan_a" "${job_a}"
write_fragment_job b "${SETUP_ROOT}/plan_b" "${job_b}"
plan_a_job="$(submit full_plan_a "${job_a}")"
plan_b_job="$(submit full_plan_b "${job_b}")"

merge_job_file="${SETUP_ROOT}/jobs/full_plan_merge.job"
cat > "${merge_job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N tv_plan_merge
#$ -q ${QUEUE}
#$ -j y
#$ -o ${SETUP_ROOT}/logs/full_plan_merge.\$JOB_ID.log
#$ -l h_rt=02:00:00
#$ -l h_data=4G
#$ -hold_jid ${plan_a_job},${plan_b_job}
set -euo pipefail
export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/terminal_validation_array.py merge-plan-fragments \
  --stage full --replicate-a-dir "${SETUP_ROOT}/plan_a" \
  --replicate-b-dir "${SETUP_ROOT}/plan_b" --shard-count "${shard_count}" \
  --output "${manifest}" --assembly-output "${SETUP_ROOT}/manifest_plan_assembly.json" \
  --compute-ceiling "${COMPUTE_CEILING}" --max-descriptors-per-subshard 450 \
  --queue "${QUEUE}" --h-rt-seconds 86400 --memory-bytes 8589934592 --throttle 90
EOF
chmod 500 "${merge_job_file}"
merge_job="$(submit full_plan_merge "${merge_job_file}")"

printf 'Submitted full setup arrays %s and %s, followed by merge job %s.\n' \
  "${plan_a_job}" "${plan_b_job}" "${merge_job}"
