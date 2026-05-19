"""GRPO helper functions for ECE405 Assignment 3 (CS336 Assignment 5)."""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch

from cs336_alignment.sft import masked_normalize


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Mean of tensor over masked positions.

    Args:
        tensor: values to average.
        mask: 1 = include, 0 = exclude (same shape as tensor).
        dim: dimension to average over (None = global mean over all masked).

    Returns:
        Masked mean with matching semantics to tensor.mean(dim).
    """
    masked_sum = (tensor * mask).sum(dim=dim)
    count = mask.float().sum(dim=dim)
    return masked_sum / count


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute advantages for a batch of rollout responses.

    For each group of `group_size` responses to the same question, compute:
      A^(i) = (r^(i) - mean(r)) / (std(r) + eps)   if normalize_by_std
      A^(i) = r^(i) - mean(r)                        otherwise  (Dr. GRPO)

    Args:
        reward_fn: scores each (response, ground_truth) pair, returning a dict
            with keys "reward", "format_reward", "answer_reward".
        rollout_responses: list of length rollout_batch_size = n_prompts * group_size.
        repeated_ground_truths: list of length rollout_batch_size (each ground
            truth repeated group_size times).
        group_size: number of rollouts per question.
        advantage_eps: small constant for numerical stability.
        normalize_by_std: divide by within-group std if True.

    Returns:
        advantages  (rollout_batch_size,): group-normalized reward.
        raw_rewards (rollout_batch_size,): unnormalized reward.
        metadata: dict with mean/std/max/min of raw rewards.
    """
    raw: list[float] = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        r = reward_fn(response, gt)
        raw.append(float(r["reward"]))

    raw_rewards = torch.tensor(raw, dtype=torch.float32)
    n_groups = len(raw) // group_size

    # Reshape to (n_groups, group_size) for group-wise statistics
    grouped = raw_rewards.view(n_groups, group_size)
    group_means = grouped.mean(dim=1, keepdim=True)  # (n_groups, 1)

    if normalize_by_std:
        group_stds = grouped.std(dim=1, keepdim=True)
        advantages_grouped = (grouped - group_means) / (group_stds + advantage_eps)
    else:
        advantages_grouped = grouped - group_means

    advantages = advantages_grouped.view(-1)

    metadata: dict[str, Any] = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
        "frac_correct": (raw_rewards > 0).float().mean().item(),
    }
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Per-token policy-gradient loss: -A_t * log pi(o_t | ...).

    Args:
        raw_rewards_or_advantages: (B, 1) scalar per rollout.
        policy_log_probs: (B, T) per-token log-probs.

    Returns:
        (B, T) per-token loss (positive = loss to minimize).
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GRPO-Clip per-token loss (PPO-style clipping).

    per_token = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    where ratio = exp(log_pi - log_pi_old).

    Args:
        advantages: (B, 1) per-example advantage.
        policy_log_probs: (B, T) current policy log-probs.
        old_log_probs: (B, T) old policy log-probs.
        cliprange: ε clipping parameter.

    Returns:
        loss (B, T): per-token clipped loss.
        metadata: dict with clip_fraction tensor.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)  # (B, T)
    unclipped = ratio * advantages  # (B, T)
    clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
    loss = -torch.min(unclipped, clipped)

    is_clipped = ((ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)).float()
    metadata = {"clip_fraction": is_clipped}
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the appropriate policy-gradient loss.

    Args:
        policy_log_probs: (B, T).
        loss_type: "no_baseline" | "reinforce_with_baseline" | "grpo_clip".
        raw_rewards: (B, 1), required for "no_baseline".
        advantages: (B, 1), required for "reinforce_with_baseline" / "grpo_clip".
        old_log_probs: (B, T), required for "grpo_clip".
        cliprange: float, required for "grpo_clip".

    Returns:
        (loss (B, T), metadata dict).
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards required for no_baseline"
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}

    if loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages required for reinforce_with_baseline"
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}

    if loss_type == "grpo_clip":
        assert advantages is not None, "advantages required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs required for grpo_clip"
        assert cliprange is not None, "cliprange required for grpo_clip"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    raise ValueError(f"Unknown loss_type: {loss_type!r}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Single GRPO microbatch: policy-gradient loss → backward.

    Loss = mean_over_batch(masked_mean_over_seq(per_token_loss))
           / gradient_accumulation_steps

    Args:
        policy_log_probs: (B, T) log-probs from current policy.
        response_mask: (B, T) 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        loss_type: which policy-gradient variant to use.
        raw_rewards: (B, 1) for "no_baseline".
        advantages: (B, 1) for "reinforce_with_baseline" / "grpo_clip".
        old_log_probs: (B, T) for "grpo_clip".
        cliprange: ε for "grpo_clip".

    Returns:
        (loss, metadata).
    """
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=-1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata
