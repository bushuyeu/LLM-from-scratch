"""Training script for Transformer language model.

Usage:
    uv run python train.py --dataset tinystories --device mps
    uv run python train.py --dataset owt --device cuda:0 --batch_size 64
    uv run python train.py --help
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import torch

from ece496b_basics import (
    AdamW,
    TransformerLM,
    cross_entropy,
    generate,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
    Tokenizer,
)


def swiglu_d_ff(d_model: int) -> int:
    """Compute d_ff = 8/3 * d_model, rounded up to the next multiple of 64."""
    raw = int(8 / 3 * d_model)
    return ((raw + 63) // 64) * 64


# ---------------------------------------------------------------------------
# Default hyperparameters for TinyStories (17M model)
# ---------------------------------------------------------------------------

TINYSTORIES_DEFAULTS = dict(
    vocab_size=10_000,
    context_length=256,
    d_model=512,
    num_layers=8,
    num_heads=8,
    d_ff=None,  # computed from d_model via swiglu_d_ff() if not set
    rope_theta=10000.0,
    # optimizer
    max_learning_rate=1e-3,
    min_learning_rate=1e-4,
    warmup_iters=200,
    max_iters=5000,
    batch_size=32,
    weight_decay=0.01,
    grad_clip=1.0,
    # data
    train_path="outputs/ts_train.npy",
    val_path="outputs/ts_valid.npy",
    vocab_path="outputs/ts_vocab_10k.pkl",
    merges_path="outputs/ts_merges_10k.pkl",
)

OWT_DEFAULTS = dict(
    vocab_size=32_000,
    context_length=256,
    d_model=512,
    num_layers=8,
    num_heads=8,
    d_ff=None,  # computed from d_model via swiglu_d_ff() if not set
    rope_theta=10000.0,
    # optimizer
    max_learning_rate=1e-3,
    min_learning_rate=1e-4,
    warmup_iters=200,
    max_iters=10000,
    batch_size=32,
    weight_decay=0.01,
    grad_clip=1.0,
    # data
    train_path="outputs/owt_train.npy",
    val_path="outputs/owt_valid.npy",
    vocab_path="outputs/owt_vocab_32k.pkl",
    merges_path="outputs/owt_merges_32k.pkl",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM")

    # dataset preset
    p.add_argument("--dataset", type=str, default="tinystories",
                    choices=["tinystories", "owt"],
                    help="Dataset preset (sets default hyperparams)")

    # model
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--rope_theta", type=float, default=None)

    # optimizer
    p.add_argument("--max_learning_rate", type=float, default=None)
    p.add_argument("--min_learning_rate", type=float, default=None)
    p.add_argument("--warmup_iters", type=int, default=None)
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)

    # data paths
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--val_path", type=str, default=None)
    p.add_argument("--vocab_path", type=str, default=None)
    p.add_argument("--merges_path", type=str, default=None)

    # training
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--eval_interval", type=int, default=100,
                    help="Evaluate every N steps")
    p.add_argument("--eval_batches", type=int, default=20,
                    help="Number of batches for validation loss estimate")
    p.add_argument("--checkpoint_interval", type=int, default=1000,
                    help="Save checkpoint every N steps")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    p.add_argument("--log_interval", type=int, default=10,
                    help="Log training loss every N steps")

    # generation
    p.add_argument("--generate_interval", type=int, default=500,
                    help="Generate sample text every N steps (0 to disable)")
    p.add_argument("--generate_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)

    # wandb
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb_project", type=str, default="ece496b-lm")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # ablations
    p.add_argument("--no_rope", action="store_true", help="Disable RoPE (NoPE ablation)")
    p.add_argument("--no_rmsnorm", action="store_true", help="Disable RMSNorm ablation")
    p.add_argument("--post_norm", action="store_true", help="Use post-norm instead of pre-norm")
    p.add_argument("--ffn_silu", action="store_true", help="Use FFN_SiLU instead of SwiGLU")
    p.add_argument("--weight_tying", action="store_true",
                    help="Tie input embedding and output projection weights")

    # performance
    p.add_argument("--compile", action="store_true",
                    help="JIT-compile model with torch.compile")

    return p.parse_args()


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in None values from dataset defaults."""
    defaults = TINYSTORIES_DEFAULTS if args.dataset == "tinystories" else OWT_DEFAULTS
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    # Derive d_ff from d_model if still unset
    if args.d_ff is None:
        args.d_ff = swiglu_d_ff(args.d_model)
    return args


@torch.no_grad()
def estimate_val_loss(
    model: TransformerLM,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    """Estimate validation loss over num_batches random batches."""
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        # Reshape for cross_entropy: (batch*seq, vocab) vs (batch*seq,)
        B, S, V = logits.shape
        loss = cross_entropy(logits.view(B * S, V), y.view(B * S))
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main():
    args = parse_args()
    args = apply_defaults(args)

    print(f"Training config:")
    print(json.dumps({k: v for k, v in vars(args).items()}, indent=2, default=str))

    # -----------------------------------------------------------------------
    # Data — memory-mapped for large datasets
    # -----------------------------------------------------------------------
    def load_mmap(path: str) -> np.ndarray:
        """Load a .npy file as a read-only np.memmap (skipping the .npy header)."""
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            offset = f.tell()
        return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)

    print(f"\nLoading training data from {args.train_path} ...")
    train_data = load_mmap(args.train_path)
    print(f"  Train tokens: {len(train_data):,} ({len(train_data)*2/1e9:.2f} GB)")

    val_data = None
    if args.val_path and os.path.exists(args.val_path):
        val_data = load_mmap(args.val_path)
        print(f"  Val tokens:   {len(val_data):,} ({len(val_data)*2/1e9:.2f} GB)")

    # Load tokenizer for generation
    tokenizer = None
    if args.vocab_path and os.path.exists(args.vocab_path):
        with open(args.vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(args.merges_path, "rb") as f:
            merges = pickle.load(f)
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    device = args.device
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta if not args.no_rope else 0.0,
        use_rmsnorm=not args.no_rmsnorm,
        post_norm=args.post_norm,
        use_swiglu=not args.ffn_silu,
    )

    # Weight tying: share embedding and lm_head weights
    if args.weight_tying:
        model.lm_head.weight = model.token_embeddings.weight
        print("Weight tying enabled (embedding ↔ lm_head)")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters ({num_params * 4 / 1e6:.1f} MB float32)")

    # JIT compilation
    if args.compile:
        if device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")
            model = torch.compile(model)
            print("torch.compile enabled (inductor + TF32)")
        elif device == "cpu":
            model = torch.compile(model)
            print("torch.compile enabled (cpu)")
        elif device == "mps":
            model = torch.compile(model, backend="aot_eager")
            print("torch.compile enabled (aot_eager for mps)")

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"  Resumed at iteration {start_iter}")

    # -----------------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # -----------------------------------------------------------------------
    # Checkpoint directory
    # -----------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    cosine_cycle_iters = args.max_iters
    model.train()
    t0 = time.time()
    running_loss = 0.0
    tokens_processed = 0

    print(f"\nStarting training for {args.max_iters} iterations...")
    print(f"  Batch size: {args.batch_size}, Context length: {args.context_length}")
    print(f"  Tokens per step: {args.batch_size * args.context_length:,}")
    print()

    for it in range(start_iter, args.max_iters):
        # Update learning rate
        lr = get_lr_cosine_schedule(
            it,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)

        B, S, V = logits.shape
        loss = cross_entropy(logits.view(B * S, V), y.view(B * S))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        # Tracking
        loss_val = loss.item()
        running_loss += loss_val
        tokens_processed += args.batch_size * args.context_length

        # Log training loss
        if (it + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / elapsed
            print(f"step {it+1:>6d} | loss {avg_loss:.4f} | lr {lr:.2e} "
                  f"| {tok_per_sec:,.0f} tok/s | {elapsed:.1f}s")
            if wandb_run:
                wandb_run.log({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/tokens_processed": tokens_processed,
                }, step=it + 1)
            running_loss = 0.0

        # Evaluate validation loss
        if val_data is not None and (it + 1) % args.eval_interval == 0:
            val_loss = estimate_val_loss(
                model, val_data, args.batch_size, args.context_length,
                device, args.eval_batches,
            )
            print(f"  >>> val loss: {val_loss:.4f} | perplexity: {math.exp(val_loss):.2f}")
            if wandb_run:
                wandb_run.log({
                    "val/loss": val_loss,
                    "val/perplexity": math.exp(val_loss),
                }, step=it + 1)
            model.train()

        # Generate sample text
        if (args.generate_interval > 0 and tokenizer is not None
                and (it + 1) % args.generate_interval == 0):
            eos_id = tokenizer.encode("<|endoftext|>")[0]
            prompt = "Once upon a time" if args.dataset == "tinystories" else "The"
            prompt_ids = tokenizer.encode(prompt)
            gen_ids = generate(
                model, prompt_ids,
                max_tokens=args.generate_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=eos_id,
            )
            gen_text = tokenizer.decode(gen_ids)
            print(f"  --- sample ---\n  {gen_text[:300]}\n  --- end ---")
            if wandb_run:
                wandb_run.log({"sample": wandb.Html(f"<pre>{gen_text[:500]}</pre>")},
                              step=it + 1)

        # Save checkpoint
        if (it + 1) % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{it+1}.pt")
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # -----------------------------------------------------------------------
    # Final checkpoint and cleanup
    # -----------------------------------------------------------------------
    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    total_time = time.time() - t0
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"Total tokens: {tokens_processed:,}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
