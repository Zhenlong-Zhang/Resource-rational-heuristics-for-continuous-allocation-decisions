#!/usr/bin/env bash
set -euo pipefail

# Submit the Round 3 1200-episode approximation-method comparison as a
# Hoffman2 array job, one environment/policy/config per task.

export LC_ALL=C
export LANG=C

PROJECT_ROOT="${PROJECT_ROOT:-/u/home/z/zzl/Resource-rational-heuristics-for-continuous-allocation-decisions}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-results/r3_approximation_methods_checkpointed_1200ep}"
EPISODES="${EPISODES:-1200}"
VOI_SAMPLES="${VOI_SAMPLES:-500}"
BLINKERED_SAMPLES="${BLINKERED_SAMPLES:-250}"
OBSERVATIONS_PER_PERSON="${OBSERVATIONS_PER_PERSON:-500}"
TERMINAL_INTEGRATION="${TERMINAL_INTEGRATION:-}"
GAUSS_HERMITE_ORDER="${GAUSS_HERMITE_ORDER:-15}"
FLUSH_EVERY="${FLUSH_EVERY:-1}"
DP_MAX_SAMPLES_VALUES="${DP_MAX_SAMPLES_VALUES:-2,4,6,10}"
DP_MEAN_GRID_SIZES="${DP_MEAN_GRID_SIZES:-7,11,21,50}"
DP_OBSERVATION_BRANCHES="${DP_OBSERVATION_BRANCHES:-3,5,7}"
TASK_H_RT="${TASK_H_RT:-43200}"
TASK_H_DATA="${TASK_H_DATA:-3G}"
COMBINE_H_RT="${COMBINE_H_RT:-43200}"
COMBINE_H_DATA="${COMBINE_H_DATA:-3G}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-81}"
DEFAULT_ENVIRONMENTS="baseline,high_need_variability,noisy_information,scarce_time,diminishing_marginal_utility,high_shortfall_penalty,high_sampling_cost,prior_knowledge_symmetric,initial_belief_asymmetry,lower_average_need,unequal_learning_efficiency,near_zero_utility,positive_utility_low_need,positive_utility_high_efficiency"
ENVIRONMENTS="${ENVIRONMENTS:-${DEFAULT_ENVIRONMENTS}}"

cd "${PROJECT_ROOT}"
mkdir -p jobs logs "${OUTPUT_DIR}"

if [[ ! -f scripts/run_method_comparison_task.py ]]; then
  echo "Missing scripts/run_method_comparison_task.py" >&2
  exit 1
fi
if [[ ! -f scripts/combine_method_comparison_results.py ]]; then
  echo "Missing scripts/combine_method_comparison_results.py" >&2
  exit 1
fi

split_csv() {
  local raw="${1//,/ }"
  local item
  for item in ${raw}; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      printf '%s\n' "${item}"
    fi
  done
}

env_file="${PROJECT_ROOT}/jobs/r3_approx_methods_envs.txt"
split_csv "${ENVIRONMENTS}" > "${env_file}"

task_file="${PROJECT_ROOT}/jobs/r3_approx_methods_tasks.tsv"
: > "${task_file}"
while IFS= read -r environment_name; do
  if [[ -z "${environment_name}" ]]; then
    continue
  fi

  printf '%s\t%s\t%s\t\t\t\n' \
    "${environment_name}" \
    "myopic_voi" \
    "myopic_voi_samples${VOI_SAMPLES}" >> "${task_file}"
  printf '%s\t%s\t%s\t\t\t\n' \
    "${environment_name}" \
    "blinkered" \
    "blinkered_samples${BLINKERED_SAMPLES}" >> "${task_file}"

  while IFS= read -r dp_max_samples; do
    while IFS= read -r dp_mean_grid_size; do
      while IFS= read -r dp_observation_branches; do
        policy_label="discretized_dp_max${dp_max_samples}_grid${dp_mean_grid_size}_branches${dp_observation_branches}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${environment_name}" \
          "discretized_dp" \
          "${policy_label}" \
          "${dp_max_samples}" \
          "${dp_mean_grid_size}" \
          "${dp_observation_branches}" >> "${task_file}"
      done < <(split_csv "${DP_OBSERVATION_BRANCHES}")
    done < <(split_csv "${DP_MEAN_GRID_SIZES}")
  done < <(split_csv "${DP_MAX_SAMPLES_VALUES}")
done < "${env_file}"

task_count=$(wc -l < "${task_file}" | tr -d ' ')
if [[ "${task_count}" == "0" ]]; then
  echo "No method-comparison tasks were generated" >&2
  exit 1
fi

cat > "${OUTPUT_DIR}/run_config.txt" <<EOF
project_root=${PROJECT_ROOT}
python_bin=${PYTHON_BIN}
output_dir=${OUTPUT_DIR}
episodes=${EPISODES}
voi_samples=${VOI_SAMPLES}
blinkered_samples=${BLINKERED_SAMPLES}
observations_per_person=${OBSERVATIONS_PER_PERSON}
terminal_integration=${TERMINAL_INTEGRATION}
gauss_hermite_order=${GAUSS_HERMITE_ORDER}
flush_every=${FLUSH_EVERY}
dp_max_samples_values=${DP_MAX_SAMPLES_VALUES}
dp_mean_grid_sizes=${DP_MEAN_GRID_SIZES}
dp_observation_branches=${DP_OBSERVATION_BRANCHES}
task_h_rt=${TASK_H_RT}
task_h_data=${TASK_H_DATA}
combine_h_rt=${COMBINE_H_RT}
combine_h_data=${COMBINE_H_DATA}
max_concurrent_tasks=${MAX_CONCURRENT_TASKS}
environments=${ENVIRONMENTS}
task_count=${task_count}
EOF

array_job_file="${PROJECT_ROOT}/jobs/r3_approximation_methods_1200ep_array.job"
cat > "${array_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_approximation_methods_1200ep_array.\$JOB_ID.\$TASK_ID.log
#$ -j y
#$ -l h_rt=${TASK_H_RT},h_data=${TASK_H_DATA}
#$ -N rr_r3_marr

export LC_ALL=C
export LANG=C
set -euo pipefail

PYTHON=${PYTHON_BIN}
TASK_LINE=\$(sed -n "\${SGE_TASK_ID}p" jobs/r3_approx_methods_tasks.tsv)
if [[ -z "\${TASK_LINE}" ]]; then
  echo "No task manifest row for SGE_TASK_ID=\${SGE_TASK_ID}" >&2
  exit 1
fi

IFS=\$'\t' read -r ENV_NAME POLICY POLICY_LABEL DP_MAX_SAMPLES DP_MEAN_GRID_SIZE DP_OBSERVATION_BRANCHES <<< "\${TASK_LINE}"
OUT="${OUTPUT_DIR}/tasks/methods/\${ENV_NAME}/\${POLICY_LABEL}"
mkdir -p "\${OUT}"
echo "Task \${SGE_TASK_ID}/${task_count}: environment=\${ENV_NAME}; policy=\${POLICY}; label=\${POLICY_LABEL}"
\$PYTHON --version
TERMINAL_INTEGRATION="${TERMINAL_INTEGRATION}"
command=(\$PYTHON scripts/run_method_comparison_task.py
  --preset server
  --output-dir "\${OUT}"
  --environment "\${ENV_NAME}"
  --episodes "${EPISODES}"
  --voi-samples "${VOI_SAMPLES}"
  --blinkered-samples "${BLINKERED_SAMPLES}"
  --common-observations on
  --observations-per-person "${OBSERVATIONS_PER_PERSON}"
  --gauss-hermite-order "${GAUSS_HERMITE_ORDER}"
  --policy "\${POLICY}"
  --policy-label "\${POLICY_LABEL}"
  --dp-max-samples "\${DP_MAX_SAMPLES:-2}"
  --dp-mean-grid-size "\${DP_MEAN_GRID_SIZE:-7}"
  --dp-observation-branches "\${DP_OBSERVATION_BRANCHES:-3}"
  --resume
  --flush-every "${FLUSH_EVERY}")
if [[ -n "\${TERMINAL_INTEGRATION}" ]]; then
  command+=(--terminal-integration "\${TERMINAL_INTEGRATION}")
fi
"\${command[@]}"
EOF
chmod +x "${array_job_file}"

array_submit_output=$(env LC_ALL=C LANG=C qsub -t "1-${task_count}" -tc "${MAX_CONCURRENT_TASKS}" "${array_job_file}")
echo "${array_submit_output}"
array_job_id=$(echo "${array_submit_output}" | awk '/Your job-array/ {print $3; found=1} /Your job / && !found {print $3}' | tr -d '"' | sed 's/[.:].*$//')
if [[ -z "${array_job_id}" ]]; then
  echo "Could not parse approximation-method array job id" >&2
  exit 1
fi

combine_job_file="${PROJECT_ROOT}/jobs/r3_approximation_methods_1200ep_array_combine.job"
cat > "${combine_job_file}" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/r3_approximation_methods_1200ep_array_combine.\$JOB_ID.log
#$ -j y
#$ -l h_rt=${COMBINE_H_RT},h_data=${COMBINE_H_DATA}
#$ -N rr_r3_mcomb

export LC_ALL=C
export LANG=C
set -euo pipefail

PYTHON=${PYTHON_BIN}
\$PYTHON --version
\$PYTHON scripts/combine_method_comparison_results.py \\
  --input-dir "${OUTPUT_DIR}" \\
  --output-dir "${OUTPUT_DIR}" \\
  --task-manifest jobs/r3_approx_methods_tasks.tsv \\
  --require-complete

tar -czf "${OUTPUT_DIR}.tar.gz" "${OUTPUT_DIR}" \\
  jobs/r3_approx_methods_envs.txt \\
  jobs/r3_approx_methods_tasks.tsv \\
  "${OUTPUT_DIR}/run_config.txt" \\
  logs/r3_approximation_methods_1200ep_array.*.log \\
  logs/r3_approximation_methods_1200ep_array_combine.*.log
ls -lh "${OUTPUT_DIR}.tar.gz"
EOF
chmod +x "${combine_job_file}"

combine_submit_output=$(env LC_ALL=C LANG=C qsub -hold_jid "${array_job_id}" "${combine_job_file}")
echo "${combine_submit_output}"

echo
echo "Submitted approximation-method array job ${array_job_id} with ${task_count} method/config tasks."
