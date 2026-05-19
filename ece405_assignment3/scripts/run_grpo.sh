#!/bin/bash
#SBATCH --job-name=grpo-train
#SBATCH --partition=gpu
#SBATCH --account=uh
#SBATCH --exclude=gn-09-17-00
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# GRPO training on Qwen2.5-Math-1.5B (problem: grpo).
# Submit with:
#   koa submit scripts/run_grpo.sh
# Override knobs via --env, e.g.:
#   koa submit scripts/run_grpo.sh --env LOSS_TYPE=grpo_clip --env N_GRPO_STEPS=100

set -euo pipefail

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"

if [[ -n "${KOA_RUN_DIR:-}" ]]; then
  JOB_DIR="${KOA_RUN_DIR}"
  REPO_DIR="${JOB_DIR}/repo"
  RESULTS_DIR="${JOB_DIR}/results"
  mkdir -p "${REPO_DIR}" "${RESULTS_DIR}"
  cd "${REPO_DIR}"
else
  JOB_DIR="${RESULTS_ROOT}/${SLURM_JOB_ID}"
  REPO_DIR="$(pwd)"
  RESULTS_DIR="${JOB_DIR}/results"
  mkdir -p "${RESULTS_DIR}"
fi
export RESULTS_DIR REPO_DIR

echo "Writing outputs to ${RESULTS_DIR}"
echo "==== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo

echo "==== GPU Info ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi

if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ MIG- ]]; then
  echo "Detected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (MIG slice)."
  echo "vLLM cannot run on MIG slices; exiting so the job can be resubmitted."
  exit 2
fi

source "scripts/setup_env.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-Math-1.5B}"
SCRATCH_DATA_ROOT="${SCRATCH_DATA_ROOT:-/mnt/lustre/koa/scratch/pavelb/ece405_data}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SCRATCH_DATA_ROOT}/math/train.jsonl}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${SCRATCH_DATA_ROOT}/math/validation.jsonl}"
OUTPUT_DIR="${RESULTS_DIR}/grpo"

LOSS_TYPE="${LOSS_TYPE:-reinforce_with_baseline}"
N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
GROUP_SIZE="${GROUP_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"

WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo-${LOSS_TYPE}-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"

srun uv run python -m cs336_alignment.train_grpo \
  --model "${MODEL}" \
  --train-data-path "${TRAIN_DATA_PATH}" \
  --val-data-path "${VAL_DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --loss-type "${LOSS_TYPE}" \
  --n-grpo-steps "${N_GRPO_STEPS}" \
  --group-size "${GROUP_SIZE}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --wandb-project "ece405-grpo" \
  --wandb-run-name "${WANDB_RUN_NAME}"
