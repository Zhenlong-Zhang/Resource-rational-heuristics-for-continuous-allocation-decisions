#!/usr/bin/env bash
# Submit deterministic dual-replicate terminal-manifest planning on Hoffman2.

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

if [[ "${STAGE}" == "smoke" ]]; then
  shard_count="${MANIFEST_PLAN_SHARDS:-16}"
  segment_size="${MANIFEST_PLAN_SEGMENT_SIZE:-16}"
  formal_throttle=4
else
  shard_count="${MANIFEST_PLAN_SHARDS:-2000}"
  segment_size="${MANIFEST_PLAN_SEGMENT_SIZE:-100}"
  formal_throttle=90
fi
if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ || "${shard_count}" -gt 2000 ]]; then
  echo "MANIFEST_PLAN_SHARDS must be between 1 and 2000." >&2
  exit 1
fi
if [[ ! "${segment_size}" =~ ^[1-9][0-9]*$ || "${segment_size}" -gt 100 ]]; then
  echo "MANIFEST_PLAN_SEGMENT_SIZE must be between 1 and 100." >&2
  exit 1
fi

mkdir -p \
  "${SETUP_ROOT}/jobs" \
  "${SETUP_ROOT}/logs" \
  "${SETUP_ROOT}/qsub_raw" \
  "${SETUP_ROOT}/plan_a" \
  "${SETUP_ROOT}/plan_b"
manifest="${SETUP_ROOT}/terminal_${STAGE}_manifest.json"
submission_file="${SETUP_ROOT}/setup_submissions.tsv"
: > "${submission_file}"
declare -a submitted_jobs=()

submit() {
  local role="$1"
  local job_file="$2"
  local raw_file="${SETUP_ROOT}/qsub_raw/${role}.txt"
  "${QSUB_BIN}" -terse "${job_file}" > "${raw_file}"
  local raw job_id
  raw="$(tr -d '\r\n' < "${raw_file}")"
  job_id="$(printf '%s' "${raw}" | sed -E 's/^([0-9]+).*/\1/')"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Unable to parse qsub job ID: ${raw}" >&2
    return 1
  fi
  submitted_jobs+=("${job_id}")
  printf '%s\t%s\t%s\t%s\n' \
    "${role}" "${job_id}" "${job_file}" "${raw_file}" >> "${submission_file}"
  printf '%s' "${job_id}"
}

segment_index=0
for replicate in a b; do
  start=1
  while [[ "${start}" -le "${shard_count}" ]]; do
    end=$((start + segment_size - 1))
    if [[ "${end}" -gt "${shard_count}" ]]; then
      end="${shard_count}"
    fi
    segment_index=$((segment_index + 1))
    role="plan_${replicate}_$(printf '%03d' "${segment_index}")"
    job_file="${SETUP_ROOT}/jobs/${role}.job"
    cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N tv_${STAGE:0:1}${replicate}${segment_index}
#$ -q ${QUEUE}
#$ -j y
#$ -o ${SETUP_ROOT}/logs/${role}.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=24:00:00
#$ -l h_data=2G
#$ -t ${start}-${end}
#$ -tc ${segment_size}
set -euo pipefail
export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/terminal_validation_array.py freeze-plan-fragment \
  --stage "${STAGE}" --shard-index "\${SGE_TASK_ID}" --shard-count "${shard_count}" \
  --output "${SETUP_ROOT}/plan_${replicate}/fragment_\$(printf '%03d' \${SGE_TASK_ID}).json"
EOF
    chmod 500 "${job_file}"
    submit "${role}" "${job_file}" >/dev/null
    start=$((end + 1))
  done
done

hold_ids="$(IFS=,; echo "${submitted_jobs[*]}")"
merge_job_file="${SETUP_ROOT}/jobs/plan_merge.job"
cat > "${merge_job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N tv_${STAGE:0:1}merge
#$ -q ${QUEUE}
#$ -j y
#$ -o ${SETUP_ROOT}/logs/plan_merge.\$JOB_ID.log
#$ -l h_rt=24:00:00
#$ -l h_data=4G
#$ -hold_jid ${hold_ids}
set -euo pipefail
export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/terminal_validation_array.py merge-plan-fragments \
  --stage "${STAGE}" --replicate-a-dir "${SETUP_ROOT}/plan_a" \
  --replicate-b-dir "${SETUP_ROOT}/plan_b" --shard-count "${shard_count}" \
  --output "${manifest}" --assembly-output "${SETUP_ROOT}/manifest_plan_assembly.json" \
  --compute-ceiling "${COMPUTE_CEILING}" --max-descriptors-per-subshard 450 \
  --queue "${QUEUE}" --h-rt-seconds 86400 --memory-bytes 8589934592 \
  --throttle "${formal_throttle}"
EOF
chmod 500 "${merge_job_file}"
merge_job="$(submit plan_merge "${merge_job_file}")"

printf 'Submitted %s setup as %s planning arrays followed by merge job %s.\n' \
  "${STAGE}" "${#submitted_jobs[@]}" "${merge_job}"
