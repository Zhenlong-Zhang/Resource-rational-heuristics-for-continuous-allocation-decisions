#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:?Set MANIFEST to the frozen R5 manifest path}"
RUN_PROJECT_ROOT="${RUN_PROJECT_ROOT:-${PROJECT_ROOT}}"
REPLACED_ARRAY_JOB_ID="${REPLACED_ARRAY_JOB_ID:?Set the quiesced array job ID}"
LANE_COUNT="${LANE_COUNT:-3}"
LANE_CONCURRENCY="${LANE_CONCURRENCY:-100}"
TASK_TIME="${TASK_TIME:-04:00:00}"
TASK_MEMORY="${TASK_MEMORY:-2G}"
QUEUE="${QUEUE:-campus2.q}"
SUBMIT="${SUBMIT:-1}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi
if [[ "${LANE_COUNT}" -lt 1 || "${LANE_COUNT}" -gt 4 ]]; then
  echo "LANE_COUNT must be between 1 and 4" >&2
  exit 1
fi
if [[ "${LANE_CONCURRENCY}" -lt 1 || "${LANE_CONCURRENCY}" -gt 100 ]]; then
  echo "LANE_CONCURRENCY must be between 1 and 100" >&2
  exit 1
fi
if env LC_ALL=C LANG=C qstat -j "${REPLACED_ARRAY_JOB_ID}" >/dev/null 2>&1; then
  echo "Refusing to resume while replaced array ${REPLACED_ARRAY_JOB_ID} still exists" >&2
  exit 1
fi

source /u/local/Modules/default/init/bash
module load python/3.9.6
export LANG=C
export LC_ALL=C

MANIFEST="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${MANIFEST}")"
OUTPUT_DIR="$(dirname "${MANIFEST}")"
RESUME_DIR="${OUTPUT_DIR}/resume_lanes"
mkdir -p "${RESUME_DIR}" "${RUN_PROJECT_ROOT}/logs"

python3 - "${MANIFEST}" "${RUN_PROJECT_ROOT}" "${RESUME_DIR}" "${LANE_COUNT}" <<'PY'
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
resume_dir = Path(sys.argv[3])
lane_count = int(sys.argv[4])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=run_root, text=True
).strip()
if commit != manifest["git_commit"]:
    raise RuntimeError(
        f"run checkout {commit} does not match frozen manifest {manifest['git_commit']}"
    )

missing = []
completed = []
for task in manifest["tasks"]:
    index = int(task["task_index"])
    task_dir = manifest_path.parent / "tasks" / f"task_{index:06d}"
    rows_path = task_dir / "rows.csv"
    status_path = task_dir / "status.json"
    if not rows_path.exists() and not status_path.exists():
        missing.append(index)
        continue
    if rows_path.exists() != status_path.exists():
        missing.append(index)
        continue
    status = json.loads(status_path.read_text(encoding="utf-8"))
    required = {
        "status": "ok",
        "task_index": index,
        "environment_index": int(task["environment_index"]),
        "episode_start": int(task["episode_start"]),
        "episode_count": int(task["episode_count"]),
        "manifest_hash": manifest["manifest_hash"],
        "git_commit": manifest["git_commit"],
    }
    mismatches = {
        key: (status.get(key), value)
        for key, value in required.items()
        if status.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"invalid completed task {index}: {mismatches}")
    if hashlib.sha256(rows_path.read_bytes()).hexdigest() != status.get("row_hash"):
        raise RuntimeError(f"row hash mismatch for task {index}")
    with rows_path.open(newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    multiplier = len(manifest.get("sample_budgets", [])) if manifest["family"] == "fixed_budget" else 1
    expected_rows = int(task["episode_count"]) * multiplier
    if row_count != expected_rows or int(status.get("row_count", -1)) != expected_rows:
        raise RuntimeError(f"row count mismatch for task {index}")
    completed.append(index)

lanes = [[] for _ in range(lane_count)]
for position, task_index in enumerate(missing):
    lanes[position % lane_count].append(task_index)

lane_records = []
for lane_index, task_indices in enumerate(lanes, start=1):
    path = resume_dir / f"lane_{lane_index}.tsv"
    path.write_text("".join(f"{value}\n" for value in task_indices), encoding="utf-8")
    lane_records.append(
        {
            "lane": lane_index,
            "task_count": len(task_indices),
            "first_task": task_indices[0] if task_indices else None,
            "last_task": task_indices[-1] if task_indices else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )

plan = {
    "manifest": str(manifest_path),
    "manifest_hash": manifest["manifest_hash"],
    "git_commit": manifest["git_commit"],
    "total_tasks": len(manifest["tasks"]),
    "completed_tasks_before_resume": len(completed),
    "missing_tasks_before_resume": len(missing),
    "lanes": lane_records,
}
(resume_dir / "resume_plan.json").write_text(
    json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(plan, sort_keys=True))
PY

if [[ "${SUBMIT}" == "0" ]]; then
  echo "Dry run complete; no jobs submitted."
  exit 0
fi

lane_job_ids=()
for lane in $(seq 1 "${LANE_COUNT}"); do
  lane_file="${RESUME_DIR}/lane_${lane}.tsv"
  task_count="$(wc -l < "${lane_file}" | tr -d ' ')"
  if [[ "${task_count}" -eq 0 ]]; then
    continue
  fi
  job_file="${RESUME_DIR}/lane_${lane}.job"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N r5_six_lane_${lane}
#$ -q ${QUEUE}
#$ -j y
#$ -o ${RUN_PROJECT_ROOT}/logs/r5_six_lane_${lane}.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=${TASK_TIME}
#$ -l h_data=${TASK_MEMORY}
#$ -t 1-${task_count}
#$ -tc ${LANE_CONCURRENCY}
set -euo pipefail
source /u/local/Modules/default/init/bash
module load python/3.9.6
export LANG=C
export LC_ALL=C
task_index="\$(sed -n "\${SGE_TASK_ID}p" "${lane_file}")"
if [[ -z "\${task_index}" ]]; then
  echo "Missing task index for lane ${lane}" >&2
  exit 1
fi
cd "${RUN_PROJECT_ROOT}"
python3 scripts/r5_array_workflow.py run-task \
  --manifest "${MANIFEST}" \
  --task-index "\${task_index}"
EOF
  submit_output="$(env LC_ALL=C LANG=C qsub "${job_file}")"
  job_id="$(printf '%s\n' "${submit_output}" | awk '{print $3}' | cut -d. -f1)"
  if [[ -z "${job_id}" ]]; then
    echo "Could not parse lane ${lane} job ID: ${submit_output}" >&2
    exit 1
  fi
  lane_job_ids+=("${job_id}")
done

if [[ "${#lane_job_ids[@]}" -eq 0 ]]; then
  echo "No missing tasks; running strict collection directly."
  cd "${RUN_PROJECT_ROOT}"
  python3 scripts/r5_array_workflow.py collect --manifest "${MANIFEST}"
  exit 0
fi

hold_ids="$(IFS=,; echo "${lane_job_ids[*]}")"
collector_file="${RESUME_DIR}/collector.job"
cat > "${collector_file}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N r5_six_lane_collect
#$ -q ${QUEUE}
#$ -j y
#$ -o ${RUN_PROJECT_ROOT}/logs/r5_six_lane_collect.\$JOB_ID.log
#$ -l h_rt=01:00:00
#$ -l h_data=4G
#$ -hold_jid ${hold_ids}
set -euo pipefail
source /u/local/Modules/default/init/bash
module load python/3.9.6
export LANG=C
export LC_ALL=C
cd "${RUN_PROJECT_ROOT}"
python3 scripts/r5_array_workflow.py collect --manifest "${MANIFEST}"
EOF

collector_output="$(env LC_ALL=C LANG=C qsub "${collector_file}")"
collector_job_id="$(printf '%s\n' "${collector_output}" | awk '{print $3}' | cut -d. -f1)"
if [[ -z "${collector_job_id}" ]]; then
  echo "Could not parse collector job ID: ${collector_output}" >&2
  exit 1
fi

python3 - "${RESUME_DIR}/resume_plan.json" "${hold_ids}" "${collector_job_id}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["lane_job_ids"] = sys.argv[2].split(",")
payload["collector_job_id"] = sys.argv[3]
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf 'lane_job_ids=%s\ncollector_job_id=%s\nplan=%s\n' \
  "${hold_ids}" "${collector_job_id}" "${RESUME_DIR}/resume_plan.json"
