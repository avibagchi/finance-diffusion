#!/bin/bash
#
# Submit multiple training jobs, one per num_factors value.
# Usage: ./submit_num_factors_sweep.sh
#        NUM_FACTORS_ARR="0 10 50 100 200 350" ./submit_num_factors_sweep.sh
#
# Edit NUM_FACTORS_ARR below or set it when running.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_train.sh"

# Array of num_factors values to sweep (edit or override with env var)
NUM_FACTORS_ARR=(${NUM_FACTORS_ARR:-0 10 25 50 75 100 150 200 250 300 350})

# Base settings (override with env when running)
IMPLICIT=${IMPLICIT:-1}
USE_SCORE_DECOMP=${USE_SCORE_DECOMP:-1}
DATA_PT=${DATA_PT:-/work/nvme/bemc/abagchi2/finance-diffusion/data/data_clean.pt}
RESULTS_BASE=${RESULTS_BASE:-${SCRIPT_DIR}/sweep_results}

mkdir -p "$RESULTS_BASE"

for K in "${NUM_FACTORS_ARR[@]}"; do
    JOB_NAME="finance_diffac_k${K}"
    LOG_DIR="${RESULTS_BASE}/k${K}"
    mkdir -p "$LOG_DIR"

    sbatch \
        --job-name="$JOB_NAME" \
        --output="${LOG_DIR}/train.log" \
        --error="${LOG_DIR}/train.err" \
        --export=ALL,"NUM_FACTORS=${K}","IMPLICIT=${IMPLICIT}","USE_SCORE_DECOMP=${USE_SCORE_DECOMP}","DATA_PT=${DATA_PT}","OUTPUT_DIR=${LOG_DIR}","SAVE=${LOG_DIR}/checkpoint_k${K}.pt" \
        "$RUN_SCRIPT"

    echo "Submitted job for num_factors=${K} -> ${LOG_DIR}"
done

echo "Submitted ${#NUM_FACTORS_ARR[@]} jobs. Results in ${RESULTS_BASE}/"
