#!/bin/bash
#SBATCH --job-name=finance_diffusion_train
#SBATCH --output=finance_diffusion_train.log
#SBATCH --error=finance_diffusion_train.err
#SBATCH --partition=gpuA100x4
#SBATCH --account=bemc-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Load modules
# module load gcc/11.4.0
# module load cuda/12.3.0
# module load cray-python/3.11.5

# Under Slurm, use submit dir (where sbatch was run); otherwise use script location
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$PROJECT_DIR/.venv/bin/activate"

# Training parameters (override via sbatch --export or edit here)
# When DATA_PT is set: num_months, num_assets, num_factors omitted (use all from file)
NUM_MONTHS=${NUM_MONTHS:-500}
NUM_ASSETS=${NUM_ASSETS:-200}
NUM_FACTORS=${NUM_FACTORS:-350}
HIDDEN_SIZE=${HIDDEN_SIZE:-256}
DEPTH=${DEPTH:-6}
NUM_HEADS=${NUM_HEADS:-8}
TIMESTEPS=${TIMESTEPS:-1000}
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-4}
GAMMA=${GAMMA:-100.0}
N_GEN_SAMPLES=${N_GEN_SAMPLES:-200}
SEED=${SEED:-42}
SAVE=${SAVE:-checkpoint.pt}
OUTPUT_DIR=${OUTPUT_DIR:-}
DATA_PT=${DATA_PT:-/work/nvme/bemc/abagchi2/finance-diffusion/data/data.pt}
IMPLICIT=${IMPLICIT:-0}

# Optional args: --data_pt when DATA_PT set, --implicit when IMPLICIT=1
EXTRA_ARGS=()
[ -n "$DATA_PT" ] && EXTRA_ARGS+=(--data_pt "$DATA_PT")
[ -n "$OUTPUT_DIR" ] && EXTRA_ARGS+=(--output_dir "$OUTPUT_DIR")
[ "$IMPLICIT" = "1" ] || [ "$IMPLICIT" = "true" ] || [ "$IMPLICIT" = "yes" ] && EXTRA_ARGS+=(--implicit)

# Omit num_months/num_assets/num_factors when using data file (use all from file)
DATA_ARGS=()
[ -z "$DATA_PT" ] && DATA_ARGS+=(--num_months "$NUM_MONTHS" --num_assets "$NUM_ASSETS" --num_factors "$NUM_FACTORS")

SRC_DIR="${PROJECT_DIR}/src"
cd "$SRC_DIR"

python3 train.py \
    "${DATA_ARGS[@]}" \
    --hidden_size "$HIDDEN_SIZE" \
    --depth "$DEPTH" \
    --num_heads "$NUM_HEADS" \
    --timesteps "$TIMESTEPS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --gamma "$GAMMA" \
    --n_gen_samples "$N_GEN_SAMPLES" \
    --seed "$SEED" \
    --save "$SAVE" \
    "${EXTRA_ARGS[@]}" \
    "$@"
