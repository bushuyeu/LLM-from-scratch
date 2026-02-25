#!/bin/bash
set -euo pipefail

# --- Environment setup ---
module load system/CUDA/12.9.1
module load numlib/cuDNN/9.15.0.57-CUDA-12.9.1

# Install uv if not present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install project dependencies
uv sync

# Download tokenized data from HF
uv run huggingface-cli download bushuyeu/ece405-tokenized-data \
    --repo-type dataset --local-dir outputs/

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Learning rate sweep (Problem 7.2a) ---
# Budget: ~4 H100 hours total. Each run: 5000 steps × ~0.3s/step ≈ 25 min
# Sweep: 7 learning rates ≈ 3 hours
COMMON_ARGS="--dataset tinystories --device cuda:0 --batch_size 64 \
    --max_iters 5000 --eval_interval 100 --checkpoint_interval 5000 \
    --generate_interval 1000 --compile --wandb --wandb_project ece496b-lm"

for LR in 3e-4 5e-4 1e-3 2e-3 3e-3 5e-3 1e-2; do
    echo ""
    echo "============================================"
    echo "  Learning rate: $LR"
    echo "============================================"
    uv run python train.py $COMMON_ARGS \
        --max_learning_rate $LR \
        --min_learning_rate $(python3 -c "print(float('$LR') * 0.1)") \
        --wandb_run_name "ts-lr-${LR}" \
        --checkpoint_dir "checkpoints/lr-${LR}" \
    || echo "WARNING: lr=$LR diverged or failed"
done

echo ""
echo "=== Sweep complete ==="
