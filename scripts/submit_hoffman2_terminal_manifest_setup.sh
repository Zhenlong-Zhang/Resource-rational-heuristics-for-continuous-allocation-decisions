#!/usr/bin/env bash
# Submit deterministic dual-replicate terminal-manifest planning on Hoffman2.

set -euo pipefail

: "${STAGE:?Set STAGE to smoke or full}"
: "${SETUP_ROOT:?Set a new immutable SETUP_ROOT}"
: "${COMPUTE_CEILING:?Set the fresh Compute Ceiling Report path}"

PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
QSUB_BIN="${QSUB_BIN:-qsub}"
QDEL_BIN="${QDEL_BIN:-qdel}"
QSTAT_BIN="${QSTAT_BIN:-qstat}"
GIT_BIN="${GIT_BIN:-git}"
QUEUE="${QUEUE:-campus2.q}"
export LANG=C LC_ALL=C
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SETUP_ROOT="$(cd "$(dirname "${SETUP_ROOT}")" && pwd -P)/$(basename "${SETUP_ROOT}")"
COMPUTE_CEILING="$(cd "$(dirname "${COMPUTE_CEILING}")" && pwd -P)/$(basename "${COMPUTE_CEILING}")"

if [[ "${STAGE}" != "smoke" && "${STAGE}" != "full" ]]; then
  echo "STAGE must be smoke or full." >&2
  exit 1
fi
if [[ -e "${SETUP_ROOT}" ]]; then
  echo "Refusing to overwrite terminal manifest setup: ${SETUP_ROOT}" >&2
  exit 1
fi
if [[ -n "$(cd "${PROJECT_ROOT}" && "${GIT_BIN}" status --porcelain --untracked-files=all)" ]]; then
  echo "Terminal manifest setup requires a clean committed worktree." >&2
  exit 1
fi

if [[ "${STAGE}" == "smoke" ]]; then
  shard_count="${MANIFEST_PLAN_SHARDS:-16}"
  segment_size="${MANIFEST_PLAN_SEGMENT_SIZE:-16}"
  formal_throttle=16
  formal_h_rt_seconds=28800
  fragment_h_rt="00:10:00"
  merge_h_rt="00:05:00"
else
  shard_count="${MANIFEST_PLAN_SHARDS:-2000}"
  segment_size="${MANIFEST_PLAN_SEGMENT_SIZE:-100}"
  formal_throttle=90
  formal_h_rt_seconds=86400
  fragment_h_rt="24:00:00"
  merge_h_rt="24:00:00"
fi
if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ || "${shard_count}" -gt 2000 ]]; then
  echo "MANIFEST_PLAN_SHARDS must be between 1 and 2000." >&2
  exit 1
fi
if [[ ! "${segment_size}" =~ ^[1-9][0-9]*$ || "${segment_size}" -gt 100 ]]; then
  echo "MANIFEST_PLAN_SEGMENT_SIZE must be between 1 and 100." >&2
  exit 1
fi
run_tag="$("${PYTHON_BIN}" - "${SETUP_ROOT}" <<'PY'
import hashlib, os, sys
print(hashlib.sha256(os.fsencode(sys.argv[1])).hexdigest()[:8])
PY
)"

mkdir -p \
  "${SETUP_ROOT}/jobs" \
  "${SETUP_ROOT}/logs" \
  "${SETUP_ROOT}/qsub_raw" \
  "${SETUP_ROOT}/plan_a" \
  "${SETUP_ROOT}/plan_b" \
  "${SETUP_ROOT}/profiles_a" \
  "${SETUP_ROOT}/profiles_b" \
  "${SETUP_ROOT}/profiles_merge" \
  "${SETUP_ROOT}/rollback"
manifest="${SETUP_ROOT}/terminal_${STAGE}_manifest.json"
submission_file="${SETUP_ROOT}/setup_submissions.tsv"
: > "${submission_file}"
declare -a submitted_jobs=()
LAST_JOB_ID=""
CLEANUP_UNCERTAIN=0

rollback() {
  local original_status="${1:-1}"
  trap - ERR INT TERM
  set +e
  "${QSTAT_BIN}" -xml -u "$(id -un)" > "${SETUP_ROOT}/rollback/discovery.xml" 2>&1
  discovery_status=$?
  set -e
  printf '%s\n' "${discovery_status}" > "${SETUP_ROOT}/rollback/discovery.status"
  if [[ "${discovery_status}" -eq 0 ]]; then
    discovery_ids="${SETUP_ROOT}/rollback/discovered_job_ids"
    set +e
    "${PYTHON_BIN}" - "${SETUP_ROOT}/rollback/discovery.xml" "${run_tag}" > "${discovery_ids}" <<'PY'
from pathlib import Path
import sys
from xml.etree import ElementTree
root = ElementTree.fromstring(Path(sys.argv[1]).read_text(encoding="utf-8"))
for job in root.iter():
    values = {str(child.tag).rsplit("}", 1)[-1]: (child.text or "").strip() for child in job}
    if values.get("JB_name", "").endswith("_" + sys.argv[2]) and values.get("JB_job_number", "").isdigit():
        print(values["JB_job_number"])
PY
    discovery_parse_status=$?
    set -e
    if [[ "${discovery_parse_status}" -eq 0 ]]; then
      while IFS= read -r discovered_id; do
        if [[ -n "${discovered_id}" && " ${submitted_jobs[*]} " != *" ${discovered_id} "* ]]; then
          submitted_jobs+=("${discovered_id}")
        fi
      done < "${discovery_ids}"
    else
      CLEANUP_UNCERTAIN=1
    fi
  else
    CLEANUP_UNCERTAIN=1
  fi
  set +e
  if [[ "${#submitted_jobs[@]}" -gt 0 ]]; then
    "${QDEL_BIN}" "${submitted_jobs[@]}" > "${SETUP_ROOT}/rollback/qdel.raw" 2>&1
    qdel_status=$?
  else
    printf '%s\n' "no_discovered_jobs" > "${SETUP_ROOT}/rollback/qdel.raw"
    qdel_status=0
  fi
  "${QSTAT_BIN}" -xml -u "$(id -un)" > "${SETUP_ROOT}/rollback/qstat.xml" 2>&1
  qstat_status=$?
  set -e
  printf '%s\n' "${qdel_status}" > "${SETUP_ROOT}/rollback/qdel.status"
  printf '%s\n' "${qstat_status}" > "${SETUP_ROOT}/rollback/qstat.status"
  if [[ "${CLEANUP_UNCERTAIN}" -ne 0 || "${qstat_status}" -ne 0 ]] || ! "${PYTHON_BIN}" - \
      "${SETUP_ROOT}/rollback/qstat.xml" "${qstat_status}" "${run_tag}" \
      ${submitted_jobs[@]+"${submitted_jobs[@]}"} <<'PY'
from pathlib import Path
import sys
from xml.etree import ElementTree
from src.experiments.terminal_plan_diagnostics import validate_qstat_absence_text
text = Path(sys.argv[1]).read_text(encoding="utf-8")
status = int(sys.argv[2])
for job_id in sys.argv[4:]:
    validate_qstat_absence_text(text, job_id, status)
root = ElementTree.fromstring(text)
for job in root.iter():
    values = {str(child.tag).rsplit("}", 1)[-1]: (child.text or "").strip() for child in job}
    if values.get("JB_name", "").endswith("_" + sys.argv[3]):
        raise RuntimeError("rollback left a setup-tagged job in scheduler state")
PY
  then
    printf '%s\n' "cleanup_uncertain" > "${SETUP_ROOT}/rollback/status"
    exit 97
  fi
  printf '%s\n' "all_submitted_jobs_absent" > "${SETUP_ROOT}/rollback/status"
  exit "${original_status}"
}
trap 'rollback $?' ERR
trap 'rollback 130' INT TERM

submit() {
  local role="$1"
  local job_file="$2"
  local job_name="$3"
  local raw_file="${SETUP_ROOT}/qsub_raw/${role}.txt"
  local before="${SETUP_ROOT}/rollback/${role}.before.xml"
  local after="${SETUP_ROOT}/rollback/${role}.after.xml"
  set +e
  "${QSTAT_BIN}" -xml -u "$(id -un)" > "${before}" 2>&1
  local before_status=$?
  set -e
  if [[ "${before_status}" -ne 0 ]]; then
    CLEANUP_UNCERTAIN=1
    return 1
  fi
  if ! "${PYTHON_BIN}" - "${before}" <<'PY'
from pathlib import Path
import sys
from src.experiments.terminal_plan_diagnostics import validate_qstat_absence_text
validate_qstat_absence_text(Path(sys.argv[1]).read_text(encoding="utf-8"), "__pre_submit_probe__", 0)
PY
  then
    CLEANUP_UNCERTAIN=1
    return 1
  fi
  set +e
  "${QSUB_BIN}" -terse "${job_file}" > "${raw_file}" 2>&1
  local qsub_status=$?
  set -e
  printf '%s\n' "${qsub_status}" > "${SETUP_ROOT}/qsub_raw/${role}.status"
  local raw job_id
  raw="$(tr -d '\r\n' < "${raw_file}")"
  job_id="$(printf '%s' "${raw}" | sed -E 's/^([0-9]+).*/\1/')"
  if [[ "${qsub_status}" -ne 0 || ! "${job_id}" =~ ^[0-9]+$ ]]; then
    set +e
    "${QSTAT_BIN}" -xml -u "$(id -un)" > "${after}" 2>&1
    local after_status=$?
    set -e
    if [[ "${after_status}" -ne 0 ]]; then
      CLEANUP_UNCERTAIN=1
      return 1
    fi
    local recovered_path="${SETUP_ROOT}/rollback/${role}.recovered_candidates"
    set +e
    "${PYTHON_BIN}" - "${before}" "${after}" "${job_name}" > "${recovered_path}" <<'PY'
from pathlib import Path
import sys
from xml.etree import ElementTree
def matching(path, name):
    root = ElementTree.fromstring(Path(path).read_text(encoding="utf-8"))
    result = set()
    for job in root.iter():
        values = {str(child.tag).rsplit("}", 1)[-1]: (child.text or "").strip() for child in job}
        if values.get("JB_name") == name and values.get("JB_job_number", "").isdigit():
            result.add(values["JB_job_number"])
    return result
for item in sorted(matching(sys.argv[2], sys.argv[3]) - matching(sys.argv[1], sys.argv[3]), key=int):
    print(item)
PY
    local recovery_status=$?
    set -e
    local -a recovered=()
    if [[ "${recovery_status}" -eq 0 ]]; then
      while IFS= read -r recovered_id; do
        [[ -n "${recovered_id}" ]] && recovered+=("${recovered_id}")
      done < "${recovered_path}"
    else
      CLEANUP_UNCERTAIN=1
    fi
    if [[ "${#recovered[@]}" -gt 0 ]]; then
      submitted_jobs+=("${recovered[@]}")
      printf '%s\n' "${recovered[@]}" > "${SETUP_ROOT}/rollback/${role}.recovered_job_ids"
    fi
    if [[ "${qsub_status}" -eq 0 && "${#recovered[@]}" -ne 1 ]]; then
      CLEANUP_UNCERTAIN=1
    fi
    echo "Unable to parse qsub job ID: ${raw}" >&2
    return 1
  fi
  submitted_jobs+=("${job_id}")
  LAST_JOB_ID="${job_id}"
  printf '%s\t%s\t%s\t%s\n' \
    "${role}" "${job_id}" "${job_file}" "${raw_file}" >> "${submission_file}"
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
    job_name="tv_${STAGE:0:1}${replicate}${segment_index}_${run_tag}"
    cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N ${job_name}
#$ -q ${QUEUE}
#$ -j y
#$ -o ${SETUP_ROOT}/logs/${role}.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=${fragment_h_rt}
#$ -l h_data=2G
#$ -t ${start}-${end}
#$ -tc ${segment_size}
set -euo pipefail
export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/terminal_validation_array.py freeze-plan-fragment \\
  --stage "${STAGE}" --shard-index "\${SGE_TASK_ID}" --shard-count "${shard_count}" \\
  --output "${SETUP_ROOT}/plan_${replicate}/fragment_\$(printf '%03d' \${SGE_TASK_ID}).json" \\
  --profile-output "${SETUP_ROOT}/profiles_${replicate}/fragment_\$(printf '%03d' \${SGE_TASK_ID}).json"
EOF
    chmod 500 "${job_file}"
    if ! submit "${role}" "${job_file}" "${job_name}" >/dev/null; then
      rollback 1
    fi
    start=$((end + 1))
  done
done

hold_ids="$(IFS=,; echo "${submitted_jobs[*]}")"
merge_job_file="${SETUP_ROOT}/jobs/plan_merge.job"
merge_job_name="tv_${STAGE:0:1}merge_${run_tag}"
cat > "${merge_job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N ${merge_job_name}
#$ -q ${QUEUE}
#$ -j y
#$ -o ${SETUP_ROOT}/logs/plan_merge.\$JOB_ID.log
#$ -l h_rt=${merge_h_rt}
#$ -l h_data=4G
#$ -hold_jid ${hold_ids}
set -euo pipefail
export LANG=C LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/terminal_validation_array.py merge-plan-fragments \\
  --stage "${STAGE}" --replicate-a-dir "${SETUP_ROOT}/plan_a" \\
  --replicate-b-dir "${SETUP_ROOT}/plan_b" --shard-count "${shard_count}" \\
  --output "${manifest}" --assembly-output "${SETUP_ROOT}/manifest_plan_assembly.json" \\
  --compute-ceiling "${COMPUTE_CEILING}" --max-descriptors-per-subshard 450 \\
  --queue "${QUEUE}" --h-rt-seconds "${formal_h_rt_seconds}" --memory-bytes 8589934592 \\
  --throttle "${formal_throttle}" \\
  --profile-output "${SETUP_ROOT}/profiles_merge/merge.json"
EOF
chmod 500 "${merge_job_file}"
if ! submit plan_merge "${merge_job_file}" "${merge_job_name}"; then
  rollback 1
fi
merge_job="${LAST_JOB_ID}"

trap - ERR INT TERM

printf 'Submitted %s setup as %s planning arrays followed by merge job %s.\n' \
  "${STAGE}" "${#submitted_jobs[@]}" "${merge_job}"
