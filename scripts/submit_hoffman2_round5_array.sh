#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Hoffman2's system Python is 3.6. Load the project runtime explicitly rather
# than inheriting the login environment with `-V` (which can also copy an
# unsupported locale to compute nodes).
source /u/local/Modules/default/init/bash
module load python/3.9.6

FAMILY="${FAMILY:-oracle}"
OUTPUT_DIR="${OUTPUT_DIR:-results/r5_${FAMILY}_$(date +%Y%m%d_%H%M%S)}"
EPISODES="${EPISODES:-120}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
OBSERVATION_DRAWS="${OBSERVATION_DRAWS:-500}"
ORACLE_GRID_SIZE="${ORACLE_GRID_SIZE:-4001}"
SEED_NAMESPACE_OFFSET="${SEED_NAMESPACE_OFFSET:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-500}"
TASK_TIME="${TASK_TIME:-04:00:00}"
TASK_MEMORY="${TASK_MEMORY:-2G}"
CONFIGS_JSON="${CONFIGS_JSON:-}"
SAMPLE_BUDGETS="${SAMPLE_BUDGETS:-0,2,4,6,8,10,12}"
RR_POLICY="${RR_POLICY:-myopic_voi}"
DP_MAX_SAMPLES="${DP_MAX_SAMPLES:-10}"
DP_MEAN_GRID_SIZE="${DP_MEAN_GRID_SIZE:-50}"
DP_MEAN_GRID_RADIUS_SD="${DP_MEAN_GRID_RADIUS_SD:-3.0}"
DP_OBSERVATION_BRANCHES="${DP_OBSERVATION_BRANCHES:-7}"
DP_OBSERVATION_INTEGRATION="${DP_OBSERVATION_INTEGRATION:-gauss_hermite}"
QUEUE="${QUEUE:-campus2.q}"

mkdir -p "${OUTPUT_DIR}" jobs logs

CREATE_ARGS=(
  scripts/r5_array_workflow.py create
  --family "${FAMILY}"
  --output-dir "${OUTPUT_DIR}"
  --episodes "${EPISODES}"
  --episodes-per-task "${EPISODES_PER_TASK}"
  --observation-draws "${OBSERVATION_DRAWS}"
  --oracle-grid-size "${ORACLE_GRID_SIZE}"
  --seed-namespace-offset "${SEED_NAMESPACE_OFFSET}"
  --sample-budgets "${SAMPLE_BUDGETS}"
  --rr-policy "${RR_POLICY}"
  --dp-max-samples "${DP_MAX_SAMPLES}"
  --dp-mean-grid-size "${DP_MEAN_GRID_SIZE}"
  --dp-mean-grid-radius-sd "${DP_MEAN_GRID_RADIUS_SD}"
  --dp-observation-branches "${DP_OBSERVATION_BRANCHES}"
  --dp-observation-integration "${DP_OBSERVATION_INTEGRATION}"
)
if [[ -n "${CONFIGS_JSON}" ]]; then
  CREATE_ARGS+=(--configs-json "${CONFIGS_JSON}")
fi
python3 "${CREATE_ARGS[@]}"

MANIFEST="${OUTPUT_DIR}/r5_manifest.json"
TASK_COUNT="$(python3 - "${MANIFEST}" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]))
PY
)"

ARRAY_JOB="jobs/r5_${FAMILY}_array.job"
cat > "${ARRAY_JOB}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N r5_${FAMILY}
#$ -q ${QUEUE}
#$ -j y
#$ -o logs/r5_${FAMILY}.\$JOB_ID.\$TASK_ID.log
#$ -l h_rt=${TASK_TIME}
#$ -l h_data=${TASK_MEMORY}
#$ -t 1-${TASK_COUNT}
#$ -tc ${MAX_CONCURRENT}
set -euo pipefail
source /u/local/Modules/default/init/bash
module load python/3.9.6
export LANG=C
export LC_ALL=C
cd "${PROJECT_ROOT}"
python3 scripts/r5_array_workflow.py run-task \
  --manifest "${MANIFEST}" \
  --task-index "\$((SGE_TASK_ID - 1))"
EOF

ARRAY_JOB_ID="$(qsub "${ARRAY_JOB}" | awk '{print $3}' | cut -d. -f1)"

COLLECT_JOB="jobs/r5_${FAMILY}_collect.job"
cat > "${COLLECT_JOB}" <<EOF
#!/usr/bin/env bash
#$ -cwd
#$ -N r5_${FAMILY}_collect
#$ -q ${QUEUE}
#$ -j y
#$ -o logs/r5_${FAMILY}_collect.\$JOB_ID.log
#$ -l h_rt=01:00:00
#$ -l h_data=4G
#$ -hold_jid ${ARRAY_JOB_ID}
set -euo pipefail
source /u/local/Modules/default/init/bash
module load python/3.9.6
export LANG=C
export LC_ALL=C
cd "${PROJECT_ROOT}"
python3 scripts/r5_array_workflow.py collect --manifest "${MANIFEST}"
EOF

COLLECT_JOB_ID="$(qsub "${COLLECT_JOB}" | awk '{print $3}')"

printf 'family=%s\noutput_dir=%s\ntasks=%s\narray_job_id=%s\ncollect_job_id=%s\n' \
  "${FAMILY}" "${OUTPUT_DIR}" "${TASK_COUNT}" "${ARRAY_JOB_ID}" "${COLLECT_JOB_ID}"
