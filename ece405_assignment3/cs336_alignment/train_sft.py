"""SFT training loop for Qwen2.5-Math-1.5B on MATH SFT examples.

ECE405 deviation: batch_size=1 with gradient accumulation, ~30 min on T4.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import (
    evaluate_vllm,
    format_prompts,
    load_math_examples,
    load_prompt_template,
    summarize,
)
from cs336_alignment.sft import get_response_log_probs, sft_microbatch_train_step, tokenize_prompt_and_output

logger = logging.getLogger(__name__)

PROMPT_PATH_DEFAULT = Path(__file__).parent / "prompts" / "r1_zero.prompt"


def sync_weights_to_vllm(policy: torch.nn.Module, llm: LLM) -> None:
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
        (name, param.data) for name, param in policy.named_parameters()
    )


def run_validation(
    llm: LLM,
    val_prompts: list[str],
    val_gts: list[str],
    step: int,
    output_dir: Path,
    wandb_run=None,
) -> dict[str, Any]:
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
    logger.info("step=%d | val accuracy=%.4f | format_rate=%.4f", step, metrics["accuracy"], metrics["format_rate"])
    if wandb_run is not None:
        wandb_run.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
    return metrics


def load_sft_examples(path: Path, template: str) -> tuple[list[str], list[str]]:
    """Load SFT jsonl and return (prompt_strs, response_strs)."""
    import json

    prompts, responses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Support both field name conventions
            problem = ex.get("problem") or ex.get("prompt") or ex.get("question", "")
            solution = ex.get("solution") or ex.get("response") or ex.get("answer", "")
            if problem and solution:
                prompts.append(template.format(question=problem))
                responses.append(solution)
    return prompts, responses


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT training on MATH reasoning examples")
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--data-path", type=Path, default=Path("data/math/sft.jsonl"))
    p.add_argument("--val-data-path", type=Path, default=Path("data/math/validation.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("results/sft"))
    p.add_argument("--prompt-template", type=Path, default=PROMPT_PATH_DEFAULT)

    # Dataset size sweep
    p.add_argument("--limit", type=int, default=None, help="Cap on SFT training examples (None = full dataset)")

    # Training hyperparameters
    p.add_argument("--n-steps", type=int, default=500, help="Total number of optimizer steps")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Validation / checkpointing
    p.add_argument("--val-every", type=int, default=100, help="Validate every N optimizer steps")
    p.add_argument("--val-limit", type=int, default=256, help="Max val examples per eval pass")
    p.add_argument("--save-every", type=int, default=250)

    # vLLM / hardware
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--dtype", default="bfloat16")

    # wandb
    p.add_argument("--wandb-project", default="ece405-sft")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template = load_prompt_template(args.prompt_template)

    # Load SFT training data
    all_prompts, all_responses = load_sft_examples(args.data_path, template)
    if args.limit is not None:
        all_prompts = all_prompts[: args.limit]
        all_responses = all_responses[: args.limit]
    logger.info("SFT examples: %d", len(all_prompts))

    # Load validation data
    val_examples = load_math_examples(args.val_data_path)[: args.val_limit]
    val_prompts = format_prompts(val_examples, template)
    val_gts = [ex["solution"] for ex in val_examples]

    # vLLM for validation
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )

    # HF policy model for training
    torch_dtype = getattr(torch, args.dtype)
    policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb
        run_name = args.wandb_run_name or f"sft-limit{args.limit or 'full'}-lr{args.learning_rate}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # Initial validation
    run_validation(llm, val_prompts, val_gts, step=0, output_dir=args.output_dir, wandb_run=wandb_run)

    optimizer.zero_grad()
    microstep = 0  # counts individual forward-backward passes
    opt_step = 0   # counts optimizer steps

    while opt_step < args.n_steps:
        # Sample a microbatch (with replacement — simple but effective for small datasets)
        indices = [random.randrange(len(all_prompts)) for _ in range(args.micro_batch_size)]
        prompt_batch = [all_prompts[i] for i in indices]
        response_batch = [all_responses[i] for i in indices]

        tokenized = tokenize_prompt_and_output(prompt_batch, response_batch, tokenizer)
        input_ids = tokenized["input_ids"].cuda()
        labels = tokenized["labels"].cuda()
        response_mask = tokenized["response_mask"].cuda()

        policy.train()
        log_probs = get_response_log_probs(policy, input_ids, labels)["log_probs"]
        loss, _ = sft_microbatch_train_step(
            log_probs, response_mask, args.gradient_accumulation_steps
        )

        microstep += 1

        if microstep % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1

            logger.info("opt_step=%d  loss=%.4f", opt_step, loss.item() * args.gradient_accumulation_steps)
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item() * args.gradient_accumulation_steps, "opt_step": opt_step})

            sync_weights_to_vllm(policy, llm)

            if opt_step % args.val_every == 0:
                run_validation(llm, val_prompts, val_gts, step=opt_step,
                               output_dir=args.output_dir, wandb_run=wandb_run)

            if opt_step % args.save_every == 0:
                ckpt = args.output_dir / f"checkpoint_step_{opt_step:05d}"
                policy.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                logger.info("Saved checkpoint → %s", ckpt)

    # Final validation + checkpoint
    run_validation(llm, val_prompts, val_gts, step=opt_step, output_dir=args.output_dir, wandb_run=wandb_run)
    final_ckpt = args.output_dir / "checkpoint_final"
    policy.save_pretrained(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)
    logger.info("Done. Final checkpoint → %s", final_ckpt)

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
