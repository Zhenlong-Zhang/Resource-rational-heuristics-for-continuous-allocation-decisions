#!/usr/bin/env bash
set -euo pipefail

# Submit the Round 3 1200-episode approximation-method comparison as a
# Hoffman2 array job, one Step 7 environment per task.

export LC_ALL=C
export LANG=C

PROJECT_ROOT="${PROJECT_ROOT:-/u/home/z/zzl/Resource-rational-heuristics-for-continuous-allocation-decisions}"
PYTHON_BIN="${PYTHON_BIN:-/u/home/z/zzl/.conda/envs/rr-allocation/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-results/r3_approximation_methods_1200ep}"
EPISODES="${EPISODES:-1200}"
VOI_SAMPLES="${VOI_SAMPLES:-500}"
BLINKERED_SAMPLES="${BLINKERED_SAMPLES:-250}"
OBSERVATIONS_PER_PERSON="${OBSERVATIONS_PER_PERSON:-500}"
TASK_H_RT="${TASK_H_RT:-43200}"
TASK_H_DATA="${TASK_H_DATA:-3G}"
COMBINE_H_RT="${COMBINE_H_RT:-43200}"
COMBINE_H_DATA="${COMBINE_H_DATA:-3G}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-14}"

cd "${PROJECT_ROOT}"
mkdir -p jobs logs "${OUTPUT_DIR}"

env_file="${PROJECT_ROOT}/jobs/r3_approx_methods_envs.txt"
cat > "${env_file}" <<'EOF'
baseline
high_need_variability
noisy_information
scarce_time
diminishing_marginal_utility
high_shortfall_penalty
high_sampling_cost
prior_knowledge_symmetric
initial_belief_asymmetry
lower_average_need
unequal_learning_efficiency
near_zero_utility
positive_utility_low_need
positive_utility_high_efficiency
EOF

task_count=$(wc -l < "${env_file}" | tr -d ' ')

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
ENV_NAME=\$(sed -n "\${SGE_TASK_ID}p" jobs/r3_approx_methods_envs.txt)
OUT="${OUTPUT_DIR}/tasks/step7/\${ENV_NAME}"
mkdir -p "\${OUT}"
echo "Task \${SGE_TASK_ID}/${task_count}: \${ENV_NAME}"
\$PYTHON --version
\$PYTHON scripts/generate_results.py \\
  --preset server \\
  --sections step7 \\
  --output-dir "\${OUT}" \\
  --environment "\${ENV_NAME}" \\
  --episodes "${EPISODES}" \\
  --voi-samples "${VOI_SAMPLES}" \\
  --blinkered-samples "${BLINKERED_SAMPLES}" \\
  --common-observations on \\
  --observations-per-person "${OBSERVATIONS_PER_PERSON}" \\
  --gauss-hermite-order 15 \\
  --dp-max-samples-values 2,4,6,10 \\
  --dp-mean-grid-sizes 7,11,21,50 \\
  --dp-observation-branches 3,5
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
\$PYTHON scripts/run_parallel_r2.py \\
  --preset server \\
  --sections step7 \\
  --episodes "${EPISODES}" \\
  --voi-samples "${VOI_SAMPLES}" \\
  --blinkered-samples "${BLINKERED_SAMPLES}" \\
  --common-observations on \\
  --observations-per-person "${OBSERVATIONS_PER_PERSON}" \\
  --resume \\
  --max-workers 1 \\
  --output-dir "${OUTPUT_DIR}"

tar -czf "${OUTPUT_DIR}.tar.gz" "${OUTPUT_DIR}" \\
  logs/r3_approximation_methods_1200ep_array.*.log \\
  logs/r3_approximation_methods_1200ep_array_combine.*.log
ls -lh "${OUTPUT_DIR}.tar.gz"
EOF
chmod +x "${combine_job_file}"

combine_submit_output=$(env LC_ALL=C LANG=C qsub -hold_jid "${array_job_id}" "${combine_job_file}")
echo "${combine_submit_output}"

echo
echo "Submitted approximation-method array job ${array_job_id} with ${task_count} tasks."
