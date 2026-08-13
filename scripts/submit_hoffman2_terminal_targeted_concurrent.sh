#!/usr/bin/env bash
# Submit the six immutable shared-two Reference-B repair validations.

set -Eeuo pipefail

: "${OUTPUT_ROOT:?Set a new immutable OUTPUT_ROOT}"
: "${EXPECTED_COMMIT:?Set the reviewed clean source commit}"

PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
QSUB_BIN="${QSUB_BIN:-qsub}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$(cd "$(dirname "${OUTPUT_ROOT}")" && pwd)/$(basename "${OUTPUT_ROOT}")"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite targeted validation run: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ "$(cd "${PROJECT_ROOT}" && git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "Targeted validation source commit mismatch." >&2
  exit 1
fi
if [[ -n "$(cd "${PROJECT_ROOT}" && git status --porcelain --untracked-files=all)" ]]; then
  echo "Targeted validation requires a clean committed worktree." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/scheduler"
job_file="${OUTPUT_ROOT}/scheduler/targeted_concurrent.job"
cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N tvtargeted
#$ -q campus2.q
#$ -j y
#$ -o ${OUTPUT_ROOT}/logs/targeted.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=02:00:00
#$ -l h_data=8589934592
#$ -t 1-6
#$ -tc 6
#$ -pe shared 2
set -euo pipefail
export LANG=C
export LC_ALL=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
cd "${PROJECT_ROOT}"
case "\${SGE_TASK_ID}" in
  1) target=base_72; repeat=1 ;;
  2) target=base_72; repeat=2 ;;
  3) target=one_step_28517; repeat=1 ;;
  4) target=one_step_28517; repeat=2 ;;
  5) target=one_step_28715; repeat=1 ;;
  6) target=one_step_28715; repeat=2 ;;
  *) echo "Unexpected task ID: \${SGE_TASK_ID}" >&2; exit 2 ;;
esac
"${PYTHON_BIN}" scripts/run_terminal_targeted_concurrent.py \
  --target "\${target}" \
  --repeat "\${repeat}" \
  --output-root "${OUTPUT_ROOT}/tasks" \
  --project-root "${PROJECT_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}"
EOF
chmod 500 "${job_file}"

set +e
"${QSUB_BIN}" -terse "${job_file}" > "${OUTPUT_ROOT}/scheduler/qsub.raw" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${OUTPUT_ROOT}/scheduler/qsub.status"
if [[ "${status}" -ne 0 ]]; then
  cat "${OUTPUT_ROOT}/scheduler/qsub.raw" >&2
  exit "${status}"
fi
job_id="$(sed -E 's/^([0-9]+).*/\1/' "${OUTPUT_ROOT}/scheduler/qsub.raw")"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "Unable to parse targeted qsub job ID." >&2
  exit 1
fi
printf '%s\n' "${job_id}" > "${OUTPUT_ROOT}/scheduler/job_id"
printf 'Submitted targeted concurrent validation job %s.\n' "${job_id}"
printf 'After completion, capture qacct -j %s as scheduler/qacct.raw and run the tracked auditor.\n' "${job_id}"
