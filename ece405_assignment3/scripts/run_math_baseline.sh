#!/bin/bash
#SBATCH --job-name=math-baseline
#SBATCH --partition=ece405
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Zero-shot MATH evaluation on Qwen2.5-Math-1.5B (problem: math_baseline).
# Submit with:
#   koa submit scripts/run_math_baseline.sh
# Override knobs via --env, e.g.:
#   koa submit scripts/run_math_baseline.sh --env LIMIT=64

set -euo pipefail

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"

if [[ -n "${KOA_RUN_DIR:-}" ]]; then
  # Launched via koa submit — repo snapshot is in KOA_RUN_DIR/repo
  JOB_DIR="${KOA_RUN_DIR}"
  REPO_DIR="${JOB_DIR}/repo"
  RESULTS_DIR="${JOB_DIR}/results"
  mkdir -p "${REPO_DIR}" "${RESULTS_DIR}"
  cd "${REPO_DIR}"
else
  # Direct sbatch — run from --chdir working directory
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
DATA_PATH="${DATA_PATH:-data/math/validation.jsonl}"

# Download MATH dataset if not already present.
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "==== Downloading MATH dataset ===="
  uv run python -c "
from pathlib import Path
from datasets import load_dataset
math_dir = Path('${DATA_PATH}').parent
math_dir.mkdir(parents=True, exist_ok=True)
ds = load_dataset('qwedsacf/competition_math')
full = ds['train'].shuffle(seed=42)
full.select(range(7500)).to_json(math_dir / 'train.jsonl')
full.select(range(7500, len(full))).to_json(math_dir / 'validation.jsonl')
print(f'Downloaded {len(full)} examples')
"
fi
OUTPUT_DIR="${RESULTS_DIR}/math_baseline"
LIMIT="${LIMIT:-}"

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi

mkdir -p "${OUTPUT_DIR}"

srun uv run python -m cs336_alignment.math_baseline \
  --model "${MODEL}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
