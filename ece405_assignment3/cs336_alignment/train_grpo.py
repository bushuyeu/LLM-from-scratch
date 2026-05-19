"""GRPO training loop for Qwen2.5-Math-1.5B on the MATH dataset.

Algorithm 3 from CS336 / ECE405 Assignment 5:
  for each GRPO step:
    1. Sample a batch of questions, generate G rollouts per question (vLLM).
    2. Score rollouts → group-normalize → advantages.
    3. For each training epoch over the rollout batch:
       - Gradient-accumulate GRPO loss microbatches → optimizer step.
    4. Sync updated weights back into the vLLM engine.
    5. Log to wandb; validate and checkpoint periodically.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.math_baseline import (
    evaluate_vllm,
    format_prompts,
    load_math_examples,
    load_prompt_template,
    summarize,
)
from cs336_alignment.sft import get_response_log_probs, tokenize_prompt_and_output

logger = logging.getLogger(__name__)

PROMPT_PATH_DEFAULT = Path(__file__).parent / "prompts" / "r1_zero.prompt"


# ---------------------------------------------------------------------------
# Weight sync
# ---------------------------------------------------------------------------

def sync_weights_to_vllm(policy_model: torch.nn.Module, llm: LLM) -> None:
    """Copy HF policy weights into the running vLLM engine in-place."""
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
        (name, param.data) for name, param in policy_model.named_parameters()
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(
    llm: LLM,
    val_prompts: list[str],
    val_gts: list[str],
    step: int,
    output_dir: Path,
    wandb_run=None,
) -> dict:
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    results = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        val_prompts,
        sampling,
        ground_truths=val_gts,
        output_path=output_dir / f"val_step_{step:05d}.jsonl",
    )
    metrics = summarize(results)
    logger.info(
        "Step %d | val accuracy=%.4f | format_rate=%.4f",
        step, metrics["accuracy"], metrics["format_rate"],
    )
    if wandb_run is not None:
        wandb_run.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
    return metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training on MATH")
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--train-data-path", type=Path, default=Path("data/math/train.jsonl"))
    p.add_argument("--val-data-path", type=Path, default=Path("data/math/validation.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("results/grpo"))
    p.add_argument("--prompt-template", type=Path, default=PROMPT_PATH_DEFAULT)

    # Core GRPO hyperparameters
    p.add_argument("--n-grpo-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--advantage-eps", type=float, default=1e-6)
    p.add_argument("--rollout-batch-size", type=int, default=256,
                   help="Total responses per GRPO step (= n_prompts * group_size).")
    p.add_argument("--group-size", type=int, default=8,
                   help="Number of rollouts per question (G).")
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-min-tokens", type=int, default=4)
    p.add_argument("--sampling-max-tokens", type=int, default=1024)
    p.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--gradient-accumulation-steps", type=int, default=128,
                   help="Microbatch size = train_batch_size / gradient_accumulation_steps.")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--loss-type", default="reinforce_with_baseline",
                   choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    p.add_argument("--use-std-normalization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cliprange", type=float, default=0.2)

    # Infrastructure
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45,
                   help="vLLM GPU fraction. Lower than the eval default to leave room for the "
                        "HF training model. Increase on A100/H100.")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)

    # Validation / checkpointing
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--val-limit", type=int, default=1024,
                   help="Max validation examples per eval pass.")

    # wandb
    p.add_argument("--wandb-project", default="ece405-grpo")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Derived batch-size constants
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_step = args.rollout_batch_size // args.group_size
    assert args.train_batch_size >= args.group_size
    n_microbatches = args.rollout_batch_size // micro_batch_size

    logger.info(
        "micro_batch_size=%d  n_prompts_per_step=%d  n_microbatches=%d",
        micro_batch_size, n_prompts_per_step, n_microbatches,
    )

    # wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            mode="online",
        )

    # Load data
    template = load_prompt_template(args.prompt_template)
    train_examples = load_math_examples(args.train_data_path)
    val_examples = load_math_examples(args.val_data_path)[: args.val_limit]
    train_prompts = format_prompts(train_examples, template)
    train_gts = [ex["solution"] for ex in train_examples]
    val_prompts = format_prompts(val_examples, template)
    val_gts = [ex["solution"] for ex in val_examples]
    logger.info("Train: %d  Val: %d", len(train_prompts), len(val_prompts))

    # vLLM — used for rollout generation and validation
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )

    # HF policy model — used for forward/backward during training
    torch_dtype = getattr(torch, args.dtype)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    rollout_sampling = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Initial validation before any training
    run_validation(llm, val_prompts, val_gts, step=0,
                   output_dir=args.output_dir, wandb_run=wandb_run)

    # -----------------------------------------------------------------------
    # GRPO loop
    # -----------------------------------------------------------------------
    for grpo_step in range(args.n_grpo_steps):

        # --- 1. Rollout generation ------------------------------------------
        indices = random.sample(range(len(train_prompts)), n_prompts_per_step)
        batch_prompts = [train_prompts[i] for i in indices]
        batch_gts = [train_gts[i] for i in indices]

        # Each question is repeated group_size times
        rep_prompts = [p for p in batch_prompts for _ in range(args.group_size)]
        rep_gts = [gt for gt in batch_gts for _ in range(args.group_size)]

        outputs = llm.generate(rep_prompts, rollout_sampling)
        responses = [o.outputs[0].text for o in outputs]

        # --- 2. Rewards & advantages ----------------------------------------
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=rep_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )

        # Tokenize the full rollout batch once
        tokenized = tokenize_prompt_and_output(rep_prompts, responses, tokenizer)
        input_ids = tokenized["input_ids"].cuda()
        labels = tokenized["labels"].cuda()
        response_mask = tokenized["response_mask"].cuda()
        adv = advantages.cuda().unsqueeze(-1)       # (B, 1) — broadcasts over seq dim
        raw = raw_rewards.cuda().unsqueeze(-1)       # (B, 1)

        # Compute old log-probs once per rollout batch (needed for grpo_clip)
        old_log_probs = None
        if args.loss_type == "grpo_clip":
            policy_model.eval()
            parts = []
            with torch.no_grad():
                for s in range(0, args.rollout_batch_size, micro_batch_size):
                    e = s + micro_batch_size
                    parts.append(
                        get_response_log_probs(policy_model, input_ids[s:e], labels[s:e])["log_probs"]
                    )
            old_log_probs = torch.cat(parts, dim=0)

        # --- 3. Policy gradient training ------------------------------------
        step_loss = 0.0
        step_clip_frac = 0.0

        for _epoch in range(args.epochs_per_rollout_batch):
            perm = torch.randperm(args.rollout_batch_size, device="cuda")
            ep_ids = input_ids[perm]
            ep_labels = labels[perm]
            ep_mask = response_mask[perm]
            ep_adv = adv[perm]
            ep_raw = raw[perm]
            ep_old_lp = old_log_probs[perm] if old_log_probs is not None else None

            policy_model.train()
            optimizer.zero_grad()

            for mb in range(n_microbatches):
                s, e = mb * micro_batch_size, (mb + 1) * micro_batch_size
                mb_lp = get_response_log_probs(policy_model, ep_ids[s:e], ep_labels[s:e])["log_probs"]

                loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=mb_lp,
                    response_mask=ep_mask[s:e],
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    raw_rewards=ep_raw[s:e] if args.loss_type == "no_baseline" else None,
                    advantages=ep_adv[s:e] if args.loss_type != "no_baseline" else None,
                    old_log_probs=ep_old_lp[s:e] if ep_old_lp is not None else None,
                    cliprange=args.cliprange if args.loss_type == "grpo_clip" else None,
                )
                step_loss += loss.item()
                if "clip_fraction" in meta:
                    step_clip_frac += meta["clip_fraction"].float().mean().item()

            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
            optimizer.step()

        # --- 4. Sync weights back into vLLM --------------------------------
        sync_weights_to_vllm(policy_model, llm)

        # --- 5. Logging ----------------------------------------------------
        n_iters = args.epochs_per_rollout_batch * n_microbatches
        log = {
            "train/loss": step_loss / n_iters,
            "train/mean_reward": reward_meta["mean_reward"],
            "train/frac_correct": reward_meta["frac_correct"],
            "train/std_reward": reward_meta["std_reward"],
            "train/max_reward": reward_meta["max_reward"],
            "train/min_reward": reward_meta["min_reward"],
            "train/grpo_step": grpo_step,
        }
        if args.loss_type == "grpo_clip":
            log["train/clip_fraction"] = step_clip_frac / n_iters

        logger.info(
            "step=%d  loss=%.4f  mean_reward=%.4f  frac_correct=%.4f",
            grpo_step, log["train/loss"], log["train/mean_reward"], log["train/frac_correct"],
        )
        if wandb_run is not None:
            wandb_run.log(log, step=grpo_step)

        # Validation
        if (grpo_step + 1) % args.val_every == 0:
            run_validation(llm, val_prompts, val_gts, step=grpo_step + 1,
                           output_dir=args.output_dir, wandb_run=wandb_run)

        # Checkpoint
        if (grpo_step + 1) % args.save_every == 0:
            ckpt = args.output_dir / f"checkpoint_step_{grpo_step + 1:05d}"
            policy_model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            logger.info("Saved checkpoint → %s", ckpt)

    # Final validation + checkpoint
    run_validation(llm, val_prompts, val_gts, step=args.n_grpo_steps,
                   output_dir=args.output_dir, wandb_run=wandb_run)
    final_ckpt = args.output_dir / "checkpoint_final"
    policy_model.save_pretrained(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)
    logger.info("Training complete. Final checkpoint → %s", final_ckpt)

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
