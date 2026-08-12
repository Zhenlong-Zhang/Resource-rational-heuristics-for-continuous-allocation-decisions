#!/usr/bin/env bash
# Submit immutable one-slot terminal smoke tasks or the 90-owner validation array.

set -Eeuo pipefail

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
QSTAT_BIN="${QSTAT_BIN:-qstat}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEDULER_USER="$(id -un)"
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
  --approved-authorization-hash "${APPROVED_EXECUTION_AUTHORIZATION_HASH}" \
  --approved-python-bin "${PYTHON_BIN}" \
  --authorized-manifest-path "${MANIFEST}" \
  --approved-scheduler-user "${SCHEDULER_USER}"

"${PYTHON_BIN}" scripts/terminal_validation_array.py validate-compute-ceiling \
  --manifest "${MANIFEST}" \
  --compute-ceiling "${COMPUTE_CEILING}"

if [[ -n "$(cd "${PROJECT_ROOT}" && git status --porcelain --untracked-files=all)" ]]; then
  echo "Terminal validation submission requires a clean committed worktree." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/scheduler/qsub_raw" \
  "${OUTPUT_ROOT}/scheduler/jobs" \
  "${OUTPUT_ROOT}/scheduler/rollback" \
  "${OUTPUT_ROOT}/logs"

h="$((${h_rt} / 3600))"
m="$(((${h_rt} % 3600) / 60))"
s="$((${h_rt} % 60))"
h_rt_text="$(printf '%02d:%02d:%02d' "${h}" "${m}" "${s}")"
submission_tsv="${OUTPUT_ROOT}/scheduler/submissions.tsv"
: > "${submission_tsv}"
token="$("${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${manifest_hash}" <<'PY'
import hashlib, os, sys
print(hashlib.sha256(os.fsencode(sys.argv[1]) + b"\0" + sys.argv[2].encode("ascii")).hexdigest()[:16])
PY
)"
if [[ ! "${token}" =~ ^[0-9a-f]{16}$ ]]; then
  echo "Unable to establish a collision-resistant validation run tag." >&2
  exit 97
fi
RUN_TAG="${token}"
declare -a submitted_jobs=()
declare -a preexisting_tagged_jobs=()
submitted_job_set=" "
preexisting_tagged_job_set=" "
cleanup_uncertain=0

rollback() {
  local original_status="${1:-1}"
  trap - ERR INT TERM
  set +e
  "${QSTAT_BIN}" -xml -u "${SCHEDULER_USER}" > "${OUTPUT_ROOT}/scheduler/rollback/discovery.xml" 2>&1
  local discovery_status=$?
  set -e
  printf '%s\n' "${discovery_status}" > "${OUTPUT_ROOT}/scheduler/rollback/discovery.status"
  if [[ "${discovery_status}" -eq 0 ]]; then
    local discovered_ids="${OUTPUT_ROOT}/scheduler/rollback/discovered_job_ids"
    set +e
    "${PYTHON_BIN}" - "${OUTPUT_ROOT}/scheduler/rollback/discovery.xml" "${RUN_TAG}" > "${discovered_ids}" <<'PY'
from pathlib import Path
import sys
from xml.etree import ElementTree
from src.experiments.terminal_execution import validate_qstat_snapshot_text
snapshot = Path(sys.argv[1]).read_text(encoding="utf-8")
validate_qstat_snapshot_text(snapshot, 0)
root = ElementTree.fromstring(snapshot)
for job in root.iter():
    values = {str(child.tag).rsplit("}", 1)[-1]: (child.text or "").strip() for child in job}
    if values.get("JB_name", "").endswith("_" + sys.argv[2]) and values.get("JB_job_number", "").isdigit():
        print(values["JB_job_number"])
PY
    local discovery_parse_status=$?
    set -e
    if [[ "${discovery_parse_status}" -eq 0 ]]; then
      while IFS= read -r discovered_id; do
        if [[ -n "${discovered_id}" \
            && "${submitted_job_set}" != *" ${discovered_id} "* \
            && "${preexisting_tagged_job_set}" != *" ${discovered_id} "* ]]; then
          submitted_jobs+=("${discovered_id}")
          submitted_job_set+="${discovered_id} "
        fi
      done < "${discovered_ids}"
    else
      cleanup_uncertain=1
    fi
  else
    cleanup_uncertain=1
  fi
  set +e
  if [[ "${#submitted_jobs[@]}" -gt 0 ]]; then
    "${QDEL_BIN}" "${submitted_jobs[@]}" > "${OUTPUT_ROOT}/scheduler/rollback/qdel.raw" 2>&1
    local qdel_status=$?
  else
    printf '%s\n' "no_discovered_jobs" > "${OUTPUT_ROOT}/scheduler/rollback/qdel.raw"
    local qdel_status=0
  fi
  "${QSTAT_BIN}" -xml -u "${SCHEDULER_USER}" > "${OUTPUT_ROOT}/scheduler/rollback/qstat.xml" 2>&1
  local qstat_status=$?
  set -e
  printf '%s\n' "${qdel_status}" > "${OUTPUT_ROOT}/scheduler/rollback/qdel.status"
  printf '%s\n' "${qstat_status}" > "${OUTPUT_ROOT}/scheduler/rollback/qstat.status"
  if [[ "${cleanup_uncertain}" -ne 0 || "${qstat_status}" -ne 0 ]] || ! "${PYTHON_BIN}" - \
      "${OUTPUT_ROOT}/scheduler/rollback/qstat.xml" "${RUN_TAG}" \
      ${submitted_jobs[@]+"${submitted_jobs[@]}"} <<'PY'
from pathlib import Path
import sys
from xml.etree import ElementTree
from src.experiments.terminal_execution import validate_qstat_snapshot_text
snapshot = Path(sys.argv[1]).read_text(encoding="utf-8")
validate_qstat_snapshot_text(
    snapshot, 0, absent_job_ids=sys.argv[3:], absent_run_tag=sys.argv[2]
)
root = ElementTree.fromstring(snapshot)
jobs = []
for job in root.iter():
    values = {str(child.tag).rsplit("}", 1)[-1]: (child.text or "").strip() for child in job}
    jobs.append(values)
known = set(sys.argv[3:])
for values in jobs:
    if values.get("JB_job_number") in known or values.get("JB_name", "").endswith("_" + sys.argv[2]):
        raise RuntimeError("rollback left a validation-tagged job in scheduler state")
PY
  then
    printf '%s\n' "cleanup_uncertain" > "${OUTPUT_ROOT}/scheduler/rollback/status"
    exit 97
  fi
  printf '%s\n' "all_submitted_jobs_absent" > "${OUTPUT_ROOT}/scheduler/rollback/status"
  exit "${original_status}"
}
trap 'rollback $?' ERR
trap 'rollback 130' INT TERM

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
"${PYTHON_BIN}" scripts/terminal_validation_array.py run-task \\
  --manifest "${MANIFEST}" \\
  --output-root "${OUTPUT_ROOT}" \\
  --task-id "\${task_id}"
EOF
  chmod 500 "${job_file}"
}

submit_array() {
  local job_file="$1" job_name="$2" task_ids="$3"
  local raw_file="${OUTPUT_ROOT}/scheduler/qsub_raw/${job_name}.txt"
  local status_file="${OUTPUT_ROOT}/scheduler/qsub_raw/${job_name}.status"
  local before="${OUTPUT_ROOT}/scheduler/rollback/qsub.before.xml"
  local after="${OUTPUT_ROOT}/scheduler/rollback/qsub.after.xml"
  "${QSTAT_BIN}" -xml -u "${SCHEDULER_USER}" > "${before}"
  "${PYTHON_BIN}" - "${before}" <<'PY'
from pathlib import Path
import sys
from src.experiments.terminal_execution import validate_qstat_snapshot_text
validate_qstat_snapshot_text(Path(sys.argv[1]).read_text(encoding="utf-8"), 0)
PY
  mapfile -t preexisting_tagged_jobs < <("${PYTHON_BIN}" - "${before}" "${RUN_TAG}" <<'PY'
from pathlib import Path
import sys
from xml.etree import ElementTree
from src.experiments.terminal_execution import validate_qstat_snapshot_text
snapshot = Path(sys.argv[1]).read_text(encoding="utf-8")
validate_qstat_snapshot_text(snapshot, 0)
root = ElementTree.fromstring(snapshot)
for job in root.iter():
    values = {str(child.tag).rsplit("}", 1)[-1]: (child.text or "").strip() for child in job}
    if values.get("JB_name", "").endswith("_" + sys.argv[2]) and values.get("JB_job_number", "").isdigit():
        print(values["JB_job_number"])
PY
)
  for preexisting_id in ${preexisting_tagged_jobs[@]+"${preexisting_tagged_jobs[@]}"}; do
    preexisting_tagged_job_set+="${preexisting_id} "
  done
  if [[ "${#preexisting_tagged_jobs[@]}" -gt 0 ]]; then
    cleanup_uncertain=1
    echo "Validation run tag already exists in scheduler state; refusing submission." >&2
    return 96
  fi
  set +e
  "${QSUB_BIN}" -terse "${job_file}" > "${raw_file}" 2>&1
  local qsub_status=$?
  set -e
  printf '%s\n' "${qsub_status}" > "${status_file}"
  local raw job_id
  raw="$(tr -d '\r\n' < "${raw_file}")"
  job_id="$(printf '%s' "${raw}" | sed -E 's/^([0-9]+).*/\1/')"
  if [[ "${qsub_status}" -ne 0 || ! "${job_id}" =~ ^[0-9]+$ ]]; then
    "${QSTAT_BIN}" -xml -u "${SCHEDULER_USER}" > "${after}"
    mapfile -t recovered < <("${PYTHON_BIN}" - "${before}" "${after}" "${job_name}" <<'PY'
from pathlib import Path
import sys
from xml.etree import ElementTree
from src.experiments.terminal_execution import validate_qstat_snapshot_text
def matching(path, name):
    snapshot = Path(path).read_text(encoding="utf-8")
    validate_qstat_snapshot_text(snapshot, 0)
    root = ElementTree.fromstring(snapshot)
    result = set()
    for job in root.iter():
        values = {str(child.tag).rsplit("}", 1)[-1]: (child.text or "").strip() for child in job}
        if values.get("JB_name") == name and values.get("JB_job_number", "").isdigit():
            result.add(values["JB_job_number"])
    return result
for item in sorted(matching(sys.argv[2], sys.argv[3]) - matching(sys.argv[1], sys.argv[3]), key=int):
    print(item)
PY
)
    if [[ "${#recovered[@]}" -gt 0 ]]; then
      submitted_jobs+=("${recovered[@]}")
      for recovered_id in "${recovered[@]}"; do
        submitted_job_set+="${recovered_id} "
      done
      printf '%s\n' "${recovered[@]}" > "${OUTPUT_ROOT}/scheduler/rollback/recovered_job_ids"
    fi
    if [[ "${qsub_status}" -eq 0 && "${#recovered[@]}" -ne 1 ]]; then
      cleanup_uncertain=1
    fi
    echo "Unable to parse qsub job ID: ${raw}" >&2
    return 1
  fi
  submitted_jobs+=("${job_id}")
  submitted_job_set+="${job_id} "
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${job_name}" "${queue}" "${task_ids}" \
    "${raw_file}" "${status_file}" "${job_file}" >> "${submission_tsv}"
}

job_name="tv${stage}_${token}"
job_file="${OUTPUT_ROOT}/scheduler/jobs/${job_name}.job"
array_directive="$(printf '#$ -t 1-%s\n#$ -tc %s' "${task_count}" "${throttle}")"
write_job "${job_file}" "${job_name}" "${array_directive}" '${SGE_TASK_ID}'
task_ids="$(seq -s, 1 "${task_count}")"
submit_array "${job_file}" "${job_name}" "${task_ids}"

"${PYTHON_BIN}" - "${submission_tsv}" "${OUTPUT_ROOT}/scheduler/submissions_input.json" <<'PY'
import json, sys
rows=[]
for line in open(sys.argv[1], encoding="utf-8"):
    job_id,name,queue,tasks,qsub,qsub_status,job_file=line.rstrip("\n").split("\t")
    rows.append({"job_id":job_id,"job_name":name,"queue":queue,
                 "array_job":True,
                 "manifest_task_ids":[int(x) for x in tasks.split(",")],
                 "qsub_raw_path":qsub,"qsub_status_path":qsub_status,
                 "job_script_path":job_file})
with open(sys.argv[2],"x",encoding="utf-8") as f:
    json.dump({"submissions":rows},f,indent=2,sort_keys=True); f.write("\n")
PY
"${PYTHON_BIN}" scripts/terminal_validation_array.py record-scheduler \
  --manifest "${MANIFEST}" \
  --submissions "${OUTPUT_ROOT}/scheduler/submissions_input.json" \
  --evidence-root "${OUTPUT_ROOT}/scheduler" \
  --execution-project-root "${PROJECT_ROOT}" \
  --approved-python-bin "${PYTHON_BIN}" \
  --scheduler-user "${SCHEDULER_USER}" \
  --run-tag "${RUN_TAG}" \
  --output "${OUTPUT_ROOT}/scheduler/scheduler_evidence.json"

trap - ERR INT TERM
printf 'Submitted %s terminal stage with %s one-slot tasks.\n' "${stage}" "${task_count}"
printf 'Run provisional collection only after every task has completed; capture exact qacct before finalization.\n'
