#!/usr/bin/env bash
# Submit immutable one-slot terminal smoke tasks or the 90-owner validation array.

set -euo pipefail

: "${MANIFEST:?Set MANIFEST to an accepted frozen execution manifest}"
: "${OUTPUT_ROOT:?Set a new immutable OUTPUT_ROOT}"
: "${REVIEW_VERDICT_FILE:?Set the stage-specific Reviewer verdict file}"
: "${APPROVED_REVIEW_VERDICT_HASH:?Set the externally approved verdict-file hash}"
: "${EXECUTION_AUTHORIZATION:?Set the manifest-specific execution authorization}"
: "${APPROVED_EXECUTION_AUTHORIZATION_HASH:?Set its externally approved file hash}"
: "${COMPUTE_CEILING:?Set the fresh Compute Ceiling Report bound by the manifest}"

PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
QSUB_BIN="${QSUB_BIN:-qsub}"
QDEL_BIN="${QDEL_BIN:-qdel}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$(cd "$(dirname "${MANIFEST}")" && pwd)/$(basename "${MANIFEST}")"
OUTPUT_ROOT="$(cd "$(dirname "${OUTPUT_ROOT}")" && pwd)/$(basename "${OUTPUT_ROOT}")"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite terminal validation run: ${OUTPUT_ROOT}" >&2
  exit 1
fi

actual_verdict_hash="$(shasum -a 256 "${REVIEW_VERDICT_FILE}" | awk '{print $1}')"
if [[ "${actual_verdict_hash}" != "${APPROVED_REVIEW_VERDICT_HASH}" ]]; then
  echo "Reviewer verdict file hash is not externally approved." >&2
  exit 1
fi

read -r stage task_count queue h_rt memory throttle manifest_hash < <(
  "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1], encoding="utf-8"))
def decode(v):
    if isinstance(v, dict) and set(v) == {"float_hex"}: return float.fromhex(v["float_hex"])
    if isinstance(v, dict): return {k: decode(x) for k, x in v.items()}
    if isinstance(v, list): return [decode(x) for x in v]
    return v
m = decode(raw)
r = m["resources"]
print(m["stage"], m["task_count"], r["queue"], r["h_rt_seconds"],
      r["memory_bytes"], r["throttle"], m["manifest_hash"])
PY
)

if [[ "${stage}" == "smoke" ]]; then
  required_verdict='ACCEPT TERMINAL IMPLEMENTATION FOR SCHEDULED SMOKE'
elif [[ "${stage}" == "full" ]]; then
  required_verdict='ACCEPT TERMINAL SMOKE / AUTHORIZE TERMINAL VALIDATION ARRAY'
else
  echo "Unknown manifest stage: ${stage}" >&2
  exit 1
fi
if [[ "$(tr -d '\r\n' < "${REVIEW_VERDICT_FILE}")" != "${required_verdict}" ]]; then
  echo "Reviewer verdict text does not authorize ${stage}." >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/terminal_validation_array.py validate-manifest \
  --manifest "${MANIFEST}" \
  --structural-only

"${PYTHON_BIN}" scripts/terminal_validation_array.py validate-authorization \
  --manifest "${MANIFEST}" \
  --authorization "${EXECUTION_AUTHORIZATION}" \
  --approved-authorization-hash "${APPROVED_EXECUTION_AUTHORIZATION_HASH}"

"${PYTHON_BIN}" scripts/terminal_validation_array.py validate-compute-ceiling \
  --manifest "${MANIFEST}" \
  --compute-ceiling "${COMPUTE_CEILING}"

if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Terminal validation submission requires a clean committed worktree." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/scheduler/qsub_raw" "${OUTPUT_ROOT}/scheduler/jobs" "${OUTPUT_ROOT}/logs"

h="$((${h_rt} / 3600))"
m="$(((${h_rt} % 3600) / 60))"
s="$((${h_rt} % 60))"
h_rt_text="$(printf '%02d:%02d:%02d' "${h}" "${m}" "${s}")"
submission_tsv="${OUTPUT_ROOT}/scheduler/submissions.tsv"
: > "${submission_tsv}"
declare -a submitted_jobs=()

rollback() {
  if [[ "${#submitted_jobs[@]}" -gt 0 ]]; then
    "${QDEL_BIN}" "${submitted_jobs[@]}" >/dev/null 2>&1 || true
  fi
}
trap rollback ERR

write_job() {
  local job_file="$1" job_name="$2" array_directive="$3" task_expression="$4"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N ${job_name}
#$ -q ${queue}
#$ -j y
#$ -o ${OUTPUT_ROOT}/logs/${job_name}.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=${h_rt_text}
#$ -l h_data=${memory}
${array_directive}
set -euo pipefail
export LANG=C
export LC_ALL=C
cd "${PROJECT_ROOT}"
task_id=${task_expression}
"${PYTHON_BIN}" scripts/terminal_validation_array.py run-task \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --task-id "\${task_id}"
EOF
  chmod 500 "${job_file}"
}

submit_one() {
  local task_id="$1" job_file="$2" job_name="$3" array_flag="$4" task_ids="$5"
  local raw_file="${OUTPUT_ROOT}/scheduler/qsub_raw/${job_name}.txt"
  "${QSUB_BIN}" -terse "${job_file}" > "${raw_file}"
  local raw job_id
  raw="$(tr -d '\r\n' < "${raw_file}")"
  job_id="$(printf '%s' "${raw}" | sed -E 's/^([0-9]+).*/\1/')"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Unable to parse qsub job ID: ${raw}" >&2
    return 1
  fi
  submitted_jobs+=("${job_id}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${job_name}" "${queue}" "${array_flag}" "${task_ids}" \
    "${raw_file}" "${job_file}" >> "${submission_tsv}"
}

token="$(date -u +%Y%m%dT%H%M%SZ)_${manifest_hash:0:8}"
if [[ "${stage}" == "smoke" ]]; then
  for task_id in $(seq 1 "${task_count}"); do
    job_name="tvsmk_${task_id}_${token}"
    job_file="${OUTPUT_ROOT}/scheduler/jobs/${job_name}.job"
    write_job "${job_file}" "${job_name}" "" "${task_id}"
    submit_one "${task_id}" "${job_file}" "${job_name}" false "${task_id}"
  done
else
  job_name="tvfull_${token}"
  job_file="${OUTPUT_ROOT}/scheduler/jobs/${job_name}.job"
  array_directive="$(printf '#$ -t 1-%s\n#$ -tc %s' "${task_count}" "${throttle}")"
  write_job "${job_file}" "${job_name}" "${array_directive}" '${SGE_TASK_ID}'
  task_ids="$(seq -s, 1 "${task_count}")"
  submit_one 0 "${job_file}" "${job_name}" true "${task_ids}"
fi

"${PYTHON_BIN}" - "${submission_tsv}" "${OUTPUT_ROOT}/scheduler/submissions_input.json" <<'PY'
import json, sys
rows=[]
for line in open(sys.argv[1], encoding="utf-8"):
    job_id,name,queue,array_flag,tasks,qsub,job_file=line.rstrip("\n").split("\t")
    rows.append({"job_id":job_id,"job_name":name,"queue":queue,
                 "array_job":array_flag=="true",
                 "manifest_task_ids":[int(x) for x in tasks.split(",")],
                 "qsub_raw_path":qsub,"job_script_path":job_file})
with open(sys.argv[2],"x",encoding="utf-8") as f:
    json.dump({"submissions":rows},f,indent=2,sort_keys=True); f.write("\n")
PY
"${PYTHON_BIN}" scripts/terminal_validation_array.py record-scheduler \
  --manifest "${MANIFEST}" \
  --submissions "${OUTPUT_ROOT}/scheduler/submissions_input.json" \
  --evidence-root "${OUTPUT_ROOT}/scheduler" \
  --output "${OUTPUT_ROOT}/scheduler/scheduler_evidence.json"

trap - ERR
printf 'Submitted %s terminal stage with %s one-slot tasks.\n' "${stage}" "${task_count}"
printf 'Run provisional collection only after every task has completed; capture exact qacct before finalization.\n'
