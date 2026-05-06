"""Generate ECE405_Assignment3.ipynb with the section/problem skeleton from the
CS336 Spring 2025 assignment 5 (alignment) handout. Mirrors the layout of
ece405_assignment2/notebooks/ECE405_Assignment2.ipynb: markdown headers per
problem followed by empty code cells for implementation.
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "ECE405_Assignment3.ipynb"


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(lines),
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "\n".join(lines),
    }


cells: list[dict] = []

cells.append(md(
    "# ECE405 Assignment 3 — Alignment and Reasoning RL",
    "",
    "Based on CS336 Assignment 5 (Stanford, Spring 2025).",
    "",
    "Goal: train a small LM to reason on MATH problems via",
    "(1) zero-shot baseline, (2) SFT on R1 reasoning traces,",
    "(3) Expert Iteration with verified rewards, and (4) GRPO.",
    "",
    "Original (2025) repo: https://github.com/stanford-cs336/assignment5-alignment  ",
    "Older (2024) repo referenced in ECE405 deviations: https://github.com/igormolybog/s2025-assignment3-alignment  ",
    "ECE405 fork: https://github.com/igormolybog/ece405-assignment3-alignment",
    "",
    "## ECE405 deviations from CS336",
    "",
    "(From `README.md` in this directory.)",
    "",
    "1. **Implementation flexibility.** From-scratch preferred, but if stuck use HF/PyTorch helpers (e.g. `Trainer`) and proceed to experiments. Submit attempted code for partial credit.",
    "2. **Submission.** Submit the report via the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLScJg_QkwjKux3xKeM-EOmZyvA6zlbVIrf_lxN_qoCFoxdqTrg/viewform). Code goes in your GitHub branch. **No leaderboard submission required.**",
    "3. **Models.** Use Qwen2.5 (not Llama 3.1):",
    "   - `Qwen2.5-0.5B` replaces `Llama-3 8B Base`",
    "   - `Qwen2.5-3B-Instruct` replaces `Llama-3 70B Instruct`",
    "   - The 2025 main body uses `Qwen2.5-Math-1.5B` as the math-RL base.",
    "4. **Optional supplement edits** (`alpaca_eval_baseline (c)` and `sst_baseline (c)`):",
    "   - Edit `scripts/alpaca_eval_vllm_llama3_70b_fn` so `model_name` points at your local Qwen2.5-3B-Instruct dir.",
    "   - Pass the local Qwen2.5-3B-Instruct path for `sst_baseline (c)`.",
    "5. **FlashAttention-2 / dtype.** Use `attn_implementation=\"flash_attention_2\"` and `bfloat16` only when the partition GPU is A40 or better; otherwise drop the flag and use `float32`.",
    "6. **SFT (Problem `sft`).** Substantially reduce training time — the point is to see the loss decrease, not converge. ECE405 sets a ~30 min budget vs the 24 H100h upstream estimate. Use `per_device_train_batch_size=1` with gradient accumulation and disable activation checkpointing if needed to fit memory.",
    "7. **DPO (Problem `dpo_training`).** Train ~30 min instead of a full HH epoch. Single GPU for both reference and trained model (consecutive forward passes).",
    "",
    "**Compute target: Koa cluster** (UHM HPC). Items 3 and 5 apply across the whole assignment; items 4, 6, 7 reference the **2024 supplement** problems (alpaca/MMLU/GSM8K/SST/DPO).",
))

cells.append(md(
    "## Setup",
    "",
    "Install dependencies, download Qwen2.5 models, and fetch the MATH dataset from HuggingFace.",
    "",
    "**Per ECE405 deviation #5:** the cells below auto-detect whether the Koa partition GPU is A40+",
    "and select `bfloat16 + flash_attention_2` if so, otherwise `float32` with default attention.",
))
cells.append(code(
    "# uv sync --no-install-package flash-attn",
    "# uv sync",
    "# uv run pytest  # initially all tests fail with NotImplementedError",
))
cells.append(code(
    "import torch",
    "from transformers import AutoModelForCausalLM, AutoTokenizer",
    "",
    "# ECE405 model choices (deviation #3):",
    "TINY_MODEL = \"Qwen/Qwen2.5-0.5B\"           # replaces Llama-3 8B Base",
    "MEDIUM_MODEL = \"Qwen/Qwen2.5-3B-Instruct\"  # replaces Llama-3 70B Instruct",
    "MATH_MODEL = \"Qwen/Qwen2.5-Math-1.5B\"      # 2025 main-body base for Sections 3-9",
    "",
    "# Per ECE405 deviation #5: float32 + no flash-attn-2 unless on A40+",
    "GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\"",
    "HIGH_END = any(s in GPU_NAME for s in (\"A40\", \"A100\", \"H100\", \"L40\"))",
    "DTYPE = torch.bfloat16 if HIGH_END else torch.float32",
    "ATTN_IMPL = \"flash_attention_2\" if HIGH_END else None  # pass `attn_implementation=ATTN_IMPL` only when truthy",
    "print(f\"GPU={GPU_NAME!r}  HIGH_END={HIGH_END}  DTYPE={DTYPE}  ATTN_IMPL={ATTN_IMPL}\")",
    "",
    "MODELS_TO_DOWNLOAD = [TINY_MODEL, MEDIUM_MODEL, MATH_MODEL]",
    "for name in MODELS_TO_DOWNLOAD:",
    "    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)",
    "    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)",
    "    out = f\"../../{name}\"",
    "    tokenizer.save_pretrained(out)",
    "    model.save_pretrained(out)",
))

cells.append(md(
    "### Dataset: MATH (Hendrycks et al. 2021)",
    "",
    "The CS336 handout points to `/data/a5-alignment/MATH/{train,validation}.jsonl` on the Together cluster.",
    "ECE405 doesn't have that — instead, pull the public mirror from HuggingFace:",
    "[`qwedsacf/competition_math`](https://huggingface.co/datasets/qwedsacf/competition_math).",
    "",
    "Each row has `problem`, `level`, `type`, `solution`. The final boxed answer can be parsed via",
    "`cs336_alignment.drgrpo_grader` or `Math-Verify`. Split into 7.5K train / 5K test as in the original.",
    "",
    "**SFT note:** This dataset has no R1 reasoning traces. For Section 4 (SFT) we need a separate",
    "source of (question, CoT reasoning + answer) pairs — e.g., generate them with a stronger model,",
    "or substitute a public R1-style trace dataset. EI (Section 5) and GRPO (Sections 7–9) only need",
    "questions + ground-truth answers, so this dataset is sufficient on its own.",
))
cells.append(code(
    "from pathlib import Path",
    "from datasets import load_dataset",
    "",
    "MATH_DIR = Path(\"../data/math\")",
    "MATH_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "ds = load_dataset(\"qwedsacf/competition_math\")",
    "# Only one split is provided ('train', ~12.5K). Resplit into 7.5K train / 5K validation.",
    "full = ds[\"train\"].shuffle(seed=42)",
    "train = full.select(range(7500))",
    "validation = full.select(range(7500, len(full)))",
    "",
    "train.to_json(MATH_DIR / \"train.jsonl\")",
    "validation.to_json(MATH_DIR / \"validation.jsonl\")",
    "print(f\"train: {len(train)}, validation: {len(validation)}\")",
))
cells.append(code(
    "# Quick peek at one example so the schema is clear:",
    "import json",
    "with open(MATH_DIR / \"validation.jsonl\") as f:",
    "    ex = json.loads(f.readline())",
    "print(ex)",
))

cells.append(md(
    "## Section 3: Measuring Zero-Shot MATH Performance",
    "",
    "Establish a baseline on the MATH 5K test set with Qwen2.5-Math-1.5B using the `r1_zero` prompt.",
    "Reward function: `cs336_alignment.drgrpo_grader.r1_zero_reward_fn`.",
    "Generation: temperature 1.0, top-p 1.0, max_tokens 1024, stop=`['</answer>']`.",
))

cells.append(md(
    "### Problem (math_baseline) — 4 points",
    "",
    "**(a)** Write `evaluate_vllm(vllm_model, reward_fn, prompts, eval_sampling_params)`:",
    "load MATH validation, format with `r1_zero` prompt, generate, score, serialize results.  ",
    "**Deliverable:** evaluation script.",
    "",
    "**(b)** Categorize generations into:",
    "(1) format=1 & answer=1, (2) format=1 & answer=0, (3) format=0 & answer=0.  ",
    "Inspect ≥10 examples per failing category. Is failure due to model or parser?  ",
    "**Deliverable:** commentary + 10 examples per category.",
    "",
    "**(c)** Report zero-shot baseline accuracy.",
    "",
    "Adapter / file: `cs336_alignment/math_baseline.py`.",
))
cells.append(code(
    "# evaluate_vllm: runs the model on prompts, scores with reward_fn, writes results to disk.",
    "from typing import Callable",
    "from vllm import LLM, SamplingParams",
    "",
    "def evaluate_vllm(",
    "    vllm_model: LLM,",
    "    reward_fn: Callable[[str, str], dict[str, float]],",
    "    prompts: list[str],",
    "    eval_sampling_params: SamplingParams,",
    ") -> None:",
    "    \"\"\"Generate, score, and serialize results to disk.\"\"\"",
    "    raise NotImplementedError",
))
cells.append(code(
    "# (b) Categorize generations and inspect failure modes.",
))
cells.append(code(
    "# (c) Report baseline metrics.",
))

cells.append(md(
    "## Section 4: Supervised Finetuning for MATH",
    "",
    "SFT Qwen2.5-Math-1.5B Base on R1 reasoning traces (`/data/a5-alignment/MATH/sft.jsonl`).",
    "",
    "All adapters are wired in `tests/adapters.py`. Tests live in `tests/test_sft.py`.",
))

cells.append(md(
    "### Problem (tokenize_prompt_and_output) — 2 points",
    "",
    "Implement `tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)`.",
    "",
    "Returns dict with:",
    "- `input_ids`: shape `(B, max(len) - 1)` — tokenized prompt+output, last token sliced off",
    "- `labels`: shape `(B, max(len) - 1)` — shifted input_ids",
    "- `response_mask`: 1 on response tokens, 0 on prompt/padding",
    "",
    "Test: `uv run pytest -k test_tokenize_prompt_and_output`",
))
cells.append(code(
    "# Implement in cs336_alignment/sft.py and wire to tests/adapters.py:run_tokenize_prompt_and_output",
))

cells.append(md(
    "### Problem (compute_entropy) — 1 point",
    "",
    "`compute_entropy(logits) -> Tensor` of per-token entropies over vocab dim.",
    "Use a numerically stable form (e.g., logsumexp).",
    "",
    "Test: `uv run pytest -k test_compute_entropy`",
))
cells.append(code(
    "# Implement compute_entropy with logsumexp for stability",
))

cells.append(md(
    "### Problem (get_response_log_probs) — 2 points",
    "",
    "`get_response_log_probs(model, input_ids, labels, return_token_entropy=False)`",
    "returns `{\"log_probs\": ..., \"token_entropy\": ...}`.",
    "",
    "Test: `uv run pytest -k test_get_response_log_probs`",
))
cells.append(code(
    "# Implement get_response_log_probs",
))

cells.append(md(
    "### Problem (masked_normalize) — 1 point",
    "",
    "`masked_normalize(tensor, mask, normalize_constant, dim=None)`:",
    "sum masked elements along `dim`, divide by `normalize_constant`.",
    "",
    "Test: `uv run pytest -k test_masked_normalize`",
))
cells.append(code(
    "# Implement masked_normalize",
))

cells.append(md(
    "### Problem (sft_microbatch_train_step) — 3 points",
    "",
    "`sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant=1.0)`:",
    "computes negative-log-likelihood loss on response tokens, divides by `gradient_accumulation_steps`,",
    "calls `loss.backward()`, returns `(loss, metadata)`.",
    "",
    "Test: `uv run pytest -k test_sft_microbatch_train_step`",
))
cells.append(code(
    "# Implement sft_microbatch_train_step (call loss.backward() inside)",
))

cells.append(md(
    "### Problem (log_generations) — 1 point",
    "",
    "`log_generations(...)`: log per-example prompt, response, ground truth, reward (format/answer/total),",
    "average token entropy, and avg response length (overall, correct, incorrect).",
))
cells.append(code(
    "# Implement log_generations",
))

cells.append(md(
    "### Problem (sft_experiment) — 2 points (2 H100 hrs upstream; ECE405 #6 allows substantial reduction)",
    "",
    "1. Run SFT on `sft.jsonl` with dataset sizes `{128, 256, 512, 1024, full}`.",
    "   Tune lr/batch to ≥15% validation accuracy on full data.  ",
    "   **Deliverable:** validation accuracy curves vs dataset size.",
    "",
    "2. Filter SFT examples to only correct-answer ones; rerun SFT on the filtered full set.  ",
    "   **Deliverable:** filtered dataset size + validation accuracy curve. Compare with (1).",
    "",
    "**ECE405 deviation #6:** OK to compress training to ~30 min — goal is showing the loss going down,",
    "not converging. If memory is tight on the chosen Koa partition, use `per_device_train_batch_size=1`",
    "with gradient accumulation and disable activation checkpointing.",
    "",
    "Run on Koa with 2 GPUs (one for policy, one for vLLM eval). See `scripts/launch_sft.sh` (TODO).",
    "Helpers: `init_vllm`, `load_policy_into_vllm_instance` (provided in the handout).",
    "Use `wandb.define_metric(\"train_step\")` / `eval_step` and gradient clipping (clip=1.0).",
))
cells.append(code(
    "# Driver script: launches an SFT run on Koa via Slurm.",
    "# Sweep over dataset sizes {128, 256, 512, 1024, full} per (1).",
))

cells.append(md(
    "## Section 5: Expert Iteration for MATH",
    "",
    "EI = sample G rollouts per question, keep correct ones, SFT on them, repeat for `n_ei_steps`.",
    "Use `min_tokens >= 4` in `SamplingParams` to avoid empty generations.",
))

cells.append(md(
    "### Problem (expert_iteration_experiment) — 2 points (6 H100 hrs)",
    "",
    "Run EI on MATH (`train.jsonl`) with `n_ei_steps=5`. Vary G ∈ rollouts and epochs in SFT step.  ",
    "Try at least 2 rollout counts and 2 epoch counts. Batch size for `D_b` ∈ `{512, 1024, 2048}`.",
    "",
    "**Deliverables:**",
    "- Validation accuracy curves per rollout config.",
    "- Model achieving ≥15% MATH validation accuracy.",
    "- 2-sentence comparison vs SFT and across EI steps.",
    "- Plot of model response entropy over training.",
))
cells.append(code(
    "# Run expert iteration",
))

cells.append(md(
    "## Section 6: Primer on Policy Gradients",
    "",
    "(Reading section — no problems.) Notes:",
    "",
    "- LM acts as categorical policy: $a_t \\sim \\pi_\\theta(\\cdot|s_t) = \\mathrm{softmax}(f_\\theta(s_t))$.",
    "- Trajectory $\\tau = (s_0, a_0, ..., s_T, a_T)$; reward only at terminal token.",
    "- REINFORCE: $\\nabla_\\theta J(\\pi_\\theta) = \\mathbb{E}_\\tau\\left[\\sum_t \\nabla_\\theta \\log \\pi_\\theta(a_t|s_t) R(\\tau)\\right]$.",
    "- Baseline: subtract $b(s_t)$ to reduce variance, unbiased as long as $b$ depends only on state.",
    "- Off-policy correction: importance ratio $\\pi_\\theta(a|s) / \\pi_{\\theta_{\\mathrm{old}}}(a|s)$.",
))

cells.append(md(
    "## Section 7: Group Relative Policy Optimization",
    "",
    "GRPO: sample G outputs per question, advantage $A^{(i)} = (r^{(i)} - \\mathrm{mean}(r)) / (\\mathrm{std}(r) + \\varepsilon)$,",
    "off-policy update with PPO-style clipping.",
    "",
    "Tests live in `tests/test_grpo.py`.",
))

cells.append(md(
    "### Problem (compute_group_normalized_rewards) — 2 points",
    "",
    "`compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std)`",
    "→ `(advantages, raw_rewards, metadata)`.",
    "",
    "If `normalize_by_std=False`, advantage is just $r - \\bar r$ (Dr. GRPO).",
    "",
    "Test: `uv run pytest -k test_compute_group_normalized_rewards`",
))
cells.append(code(
    "# Implement compute_group_normalized_rewards",
))

cells.append(md(
    "### Problem (compute_naive_policy_gradient_loss) — 1 point",
    "",
    "`compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs)` →",
    "per-token loss `-A_t * log pi(o_t|...)`. Broadcast advantages over sequence.",
    "",
    "Test: `uv run pytest -k test_compute_naive_policy_gradient_loss`",
))
cells.append(code(
    "# Implement naive PG loss",
))

cells.append(md(
    "### Problem (compute_grpo_clip_loss) — 2 points",
    "",
    "`compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)` →",
    "$-\\min(r_t A_t, \\mathrm{clip}(r_t, 1-\\varepsilon, 1+\\varepsilon) A_t)$ where $r_t = \\exp(\\log\\pi_\\theta - \\log\\pi_{old})$.",
    "Return `(loss, metadata)`; log clip fraction.",
    "",
    "Test: `uv run pytest -k test_compute_grpo_clip_loss`",
))
cells.append(code(
    "# Implement GRPO-Clip loss",
))

cells.append(md(
    "### Problem (compute_policy_gradient_loss) — 1 point",
    "",
    "Wrapper dispatching on `loss_type ∈ {no_baseline, reinforce_with_baseline, grpo_clip}`.",
    "",
    "Test: `uv run pytest -k test_compute_policy_gradient_loss`",
))
cells.append(code(
    "# Implement compute_policy_gradient_loss wrapper",
))

cells.append(md(
    "### Problem (masked_mean) — 1 point",
    "",
    "`masked_mean(tensor, mask, dim=None)` → mean over masked positions.",
    "",
    "Test: `uv run pytest -k test_masked_mean`",
))
cells.append(code(
    "# Implement masked_mean",
))

cells.append(md(
    "### Problem (grpo_microbatch_train_step) — 3 points",
    "",
    "`grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards=None, advantages=None, old_log_probs=None, cliprange=None)`:",
    "compute per-token loss → `masked_mean` over response tokens → mean over batch → divide by `gradient_accumulation_steps` → backward.",
    "",
    "Test: `uv run pytest -k test_grpo_microbatch_train_step`",
))
cells.append(code(
    "# Implement grpo_microbatch_train_step",
))

cells.append(md(
    "### Problem (grpo_train_loop) — 5 points",
    "",
    "Full GRPO loop. Hyperparameters (start with these):",
    "",
    "```",
    "n_grpo_steps = 200",
    "learning_rate = 1e-5",
    "advantage_eps = 1e-6",
    "rollout_batch_size = 256",
    "group_size = 8",
    "sampling_temperature = 1.0",
    "sampling_min_tokens = 4",
    "sampling_max_tokens = 1024",
    "epochs_per_rollout_batch = 1   # on-policy",
    "train_batch_size = 256         # on-policy",
    "gradient_accumulation_steps = 128  # microbatch=2 fits H100",
    "gpu_memory_utilization = 0.85",
    "loss_type = \"reinforce_with_baseline\"",
    "use_std_normalization = True",
    "optimizer = AdamW(lr=lr, weight_decay=0.0, betas=(0.9, 0.95))",
    "```",
    "",
    "**Deliverable:** validation reward curve + sample rollouts over time.",
))
cells.append(code(
    "# GRPO train loop — driver script. Likely runs as a job; cell can submit it.",
))

cells.append(md(
    "## Section 8: GRPO Experiments",
    "",
    "All runs use 2 GPUs (vLLM + policy). Stop early if a config diverges.",
))

cells.append(md(
    "### Problem (grpo_learning_rate) — 2 points (6 H100 hrs)",
    "",
    "Sweep learning rate. Report validation reward curves.  ",
    "**Deliverable:** model with ≥25% MATH validation accuracy on at least one LR.",
))
cells.append(code(
    "# LR sweep",
))

cells.append(md(
    "### Problem (grpo_baselines) — 2 points (2 H100 hrs)",
    "",
    "Compare `no_baseline` vs `reinforce_with_baseline`. Report reward curves and other trends.",
))
cells.append(code(
    "# baseline ablation",
))

cells.append(md(
    "### Problem (think_about_length_normalization) — 1 point",
    "",
    "**Conceptual.** Compare `masked_mean` vs `masked_normalize` (with `constant_normalizer=max_gen_len`).",
    "Pros/cons of each? Settings where one beats the other?",
    "",
    "**Deliverable:** written analysis (no experiments yet).",
))
cells.append(md(
    "_Written response:_",
    "",
    "TODO",
))

cells.append(md(
    "### Problem (grpo_length_normalization) — 2 points (2 H100 hrs)",
    "",
    "Empirical: GRPO with `masked_mean` vs `masked_normalize`. Report curves; comment on stability (gradient norm).",
))
cells.append(code(
    "# length normalization ablation",
))

cells.append(md(
    "### Problem (grpo_group_standard_deviation) — 2 points (2 H100 hrs)",
    "",
    "Compare `use_std_normalization=True` vs `False`. Report curves; comment on stability.",
))
cells.append(code(
    "# std normalization ablation",
))

cells.append(md(
    "### Problem (grpo_off_policy)",
    "",
    "Implement off-policy GRPO:",
    "- multiple epochs/optimizer steps per rollout batch (controlled by `rollout_batch_size`, `epochs_per_rollout_batch`, `train_batch_size`)",
    "- compute `old_log_probs` from policy after rollout (use `torch.inference_mode()`)",
    "- use `loss_type=\"grpo_clip\"`",
))
cells.append(code(
    "# off-policy GRPO support",
))

cells.append(md(
    "### Problem (grpo_off_policy_sweep) — 4 points (12 H100 hrs)",
    "",
    "Fix `rollout_batch_size=256`. Sweep `epochs_per_rollout_batch` × `train_batch_size`.",
    "Broad sweep at <50 GRPO steps, then focused sweep at ~200 steps.",
    "Compare to on-policy (`epochs=1, train_batch=256`) by validation step and wall-clock.",
    "**Note:** adjust `gradient_accumulation_steps` to keep memory constant.",
))
cells.append(code(
    "# off-policy sweep",
))

cells.append(md(
    "### Problem (grpo_off_policy_clip_ablation) — 2 points (2 H100 hrs)",
    "",
    "Add a new loss type `\"GRPO-No-Clip\"` (`-π/π_old · A_t`). Run with best off-policy hyperparams.",
    "Compare to clipped run: entropy, response length, gradient norm.",
))
cells.append(code(
    "# GRPO-No-Clip loss",
))

cells.append(md(
    "### Problem (grpo_prompt_ablation) — 2 points (2 H100 hrs)",
    "",
    "Train with `cs336_alignment/prompts/question_only.prompt` and `question_only_reward_fn`.",
    "Compare to R1-Zero prompt run. Discuss entropy, response length, gradient norm.",
))
cells.append(code(
    "# prompt ablation",
))

cells.append(md(
    "## Section 9: Leaderboard — GRPO on MATH",
    "",
    "Free-form: maximize MATH validation accuracy in 4 hours of training on 2 H100s.",
    "Constraints: no extra data, no extra models; full 5K validation set;",
    "use R1-Zero prompt and `r1_zero_reward_fn` for validation; temperature 1.0, max_tokens 1024.",
))

cells.append(md(
    "### Problem (leaderboard) — 16 points (16 H100 hrs)",
    "",
    "**Deliverable:** validation accuracy + screenshot of accuracy vs wall-clock (≤4 hours).",
))
cells.append(code(
    "# Leaderboard run",
))

cells.append(md(
    "## Optional Supplement (Instruction Tuning, MMLU/GSM8K, DPO, Safety)",
    "",
    "Covers the 2024-style problems referenced by ECE405 deviations #4, #6, #7:",
    "see `cs336_spring2024_assignment5_alignment.pdf` and",
    "`cs336_spring2025_assignment5_supplement_safety_rlhf.pdf` in this directory.",
    "",
    "Adapters live in `tests/adapters.py`:",
    "`get_packed_sft_dataset`, `run_iterate_batches`, `run_parse_mmlu_response`,",
    "`run_parse_gsm8k_response`, `run_compute_per_instance_dpo_loss`.",
    "",
    "**Models (deviation #3):** `Qwen2.5-0.5B` as base, `Qwen2.5-3B-Instruct` as the larger judge.",
))

cells.append(md(
    "### Problem (sft) — Instruction Tuning",
    "",
    "Pack Alpaca-style instruction tuning examples into fixed-length sequences and train Qwen2.5-0.5B.",
    "",
    "**ECE405 deviation #6:** train for ~30 min instead of the upstream 24 H100h estimate.",
    "If memory tight: `per_device_train_batch_size=1`, gradient accumulation, no activation checkpointing.",
))
cells.append(code(
    "# Implement get_packed_sft_dataset and run_iterate_batches.",
    "# Slurm driver trains Qwen2.5-0.5B on Alpaca for ~30 min on Koa.",
))

cells.append(md(
    "### Problem (mmlu_baseline)",
    "",
    "Zero-shot MMLU evaluation. Implement `run_parse_mmlu_response` to extract `A/B/C/D` from the model's output.",
))
cells.append(code(
    "# Implement run_parse_mmlu_response and the eval loop using vLLM + Qwen2.5-0.5B.",
))

cells.append(md(
    "### Problem (gsm8k_baseline)",
    "",
    "Zero-shot GSM8K evaluation. `run_parse_gsm8k_response` returns the last numeric token in the output.",
))
cells.append(code(
    "# Implement run_parse_gsm8k_response and the eval loop.",
))

cells.append(md(
    "### Problem (alpaca_eval_baseline)",
    "",
    "**(c) ECE405 deviation #4:** edit `scripts/alpaca_eval_vllm_llama3_70b_fn`",
    "so that `model_name` is the local path to Qwen2.5-3B-Instruct (e.g. `../../Qwen/Qwen2.5-3B-Instruct`).",
))
cells.append(code(
    "# Run alpaca_eval against Qwen2.5-3B-Instruct (judge) over your SFT model's outputs.",
))

cells.append(md(
    "### Problem (sst_baseline)",
    "",
    "**(c) ECE405 deviation #4:** provide the local Qwen2.5-3B-Instruct path.",
    "Use `scripts/evaluate_safety.py` with the SimpleSafetyTests dataset in `data/simple_safety_tests/`.",
))
cells.append(code(
    "# Run evaluate_safety.py with model=local Qwen2.5-3B-Instruct path.",
))

cells.append(md(
    "### Problem (dpo_training)",
    "",
    "DPO fine-tuning on the Anthropic HH preference dataset.",
    "",
    "**ECE405 deviation #7:** train for ~30 min instead of a full HH epoch.",
    "Use a **single GPU** for both reference and trained models — query them consecutively",
    "(load weights, forward pass, swap, forward pass) instead of holding both in memory.",
))
cells.append(code(
    "# Implement run_compute_per_instance_dpo_loss; train Qwen2.5-0.5B for ~30 min on a single GPU.",
))

cells.append(md(
    "## Submission",
    "",
    "Per ECE405 deviation #2:",
    "- Submit the report via the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLScJg_QkwjKux3xKeM-EOmZyvA6zlbVIrf_lxN_qoCFoxdqTrg/viewform).",
    "- Include a link to your GitHub branch (this repo) plus any wandb run links.",
    "- Code does **not** need to be attached.",
    "- **No leaderboard submission required.**",
))

# ----- Build notebook -----

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Convert source strings to lists of lines (Jupyter convention)
for c in notebook["cells"]:
    if isinstance(c["source"], str):
        lines = c["source"].splitlines(keepends=True)
        c["source"] = lines

NB_PATH.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {NB_PATH} with {len(cells)} cells")
