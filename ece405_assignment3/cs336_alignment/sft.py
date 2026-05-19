"""SFT helper functions for ECE405 Assignment 3 (CS336 Assignment 5)."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompts and outputs, construct response mask.

    Returns dict with:
      input_ids  (B, max_len-1): prompt+output tokens, last token dropped.
      labels     (B, max_len-1): shifted input_ids (next-token targets).
      response_mask (B, max_len-1): 1 for output tokens in labels, 0 otherwise.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    prompt_ids_list: list[list[int]] = tokenizer(
        prompt_strs, add_special_tokens=False
    )["input_ids"]
    output_ids_list: list[list[int]] = tokenizer(
        output_strs, add_special_tokens=False
    )["input_ids"]

    full_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    full_lens = [len(ids) for ids in full_ids_list]
    max_len = max(full_lens)

    batch_size = len(prompt_strs)
    input_ids = torch.full((batch_size, max_len - 1), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), pad_id, dtype=torch.long)
    response_mask = torch.zeros(batch_size, max_len - 1, dtype=torch.bool)

    for i, (p_ids, full_ids, full_len) in enumerate(
        zip(prompt_ids_list, full_ids_list, full_lens)
    ):
        padded = full_ids + [pad_id] * (max_len - full_len)
        input_ids[i] = torch.tensor(padded[:-1], dtype=torch.long)
        labels[i] = torch.tensor(padded[1:], dtype=torch.long)

        p_len = len(p_ids)
        start = p_len - 1  # first response position in labels (j s.t. labels[j]=output[0])
        end = full_len - 1  # first padding position
        if start < end:
            response_mask[i, start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-token entropy of next-token predictions.

    Args:
        logits: (batch, seq, vocab) unnormalized logits.

    Returns:
        (batch, seq) entropy values.
    """
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-token conditional log-probabilities (and optionally entropy).

    Args:
        model: causal LM (placed on device; caller controls grad mode).
        input_ids: (B, T) concatenated prompt+response tokens.
        labels: (B, T) shifted input_ids (next-token targets).
        return_token_entropy: if True, also return "token_entropy".

    Returns:
        dict with "log_probs" (B, T) and optionally "token_entropy" (B, T).
    """
    logits = model(input_ids).logits  # (B, T, V)
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result: dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum masked elements along dim, divide by normalize_constant.

    Args:
        tensor: values to sum.
        mask: boolean or float mask; 1 = include, 0 = exclude.
        normalize_constant: divisor.
        dim: dimension to reduce (None = all dims).

    Returns:
        Normalized sum with masked elements excluded.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Single SFT microbatch: NLL loss → backward.

    Loss = mean_over_batch(sum_over_seq(-log_p * mask) / normalize_constant)
           / gradient_accumulation_steps

    Args:
        policy_log_probs: (B, T) log-probs from the policy being trained.
        response_mask: (B, T) 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        normalize_constant: divisor for the per-sequence masked sum.

    Returns:
        (loss, metadata) where loss is the scalar used for backprop.
    """
    per_seq = masked_normalize(
        -policy_log_probs, response_mask, normalize_constant, dim=-1
    )
    loss = per_seq.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), {"loss": loss.detach()}


def log_generations(
    vllm_model,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params,
    tokenizer: PreTrainedTokenizerBase | None = None,
    policy_model: PreTrainedModel | None = None,
    step: int = 0,
    wandb_run=None,
    n_log: int = 4,
) -> dict[str, Any]:
    """Generate responses, compute rewards, log metrics.

    Logs:
      - Input prompt, generated response, ground-truth answer
      - Format / answer / total reward
      - Average token entropy of response (if policy_model provided)
      - Average response length (overall, correct, incorrect)

    Returns:
        dict with aggregate metrics.
    """
    import random

    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [o.outputs[0].text for o in outputs]

    rewards, format_rewards, answer_rewards = [], [], []
    correct_lengths, wrong_lengths = [], []

    for response, gt in zip(responses, ground_truths):
        r = reward_fn(response, gt)
        rewards.append(r.get("reward", 0.0))
        format_rewards.append(r.get("format_reward", 0.0))
        answer_rewards.append(r.get("answer_reward", 0.0))
        tok_len = len(response.split())
        if r.get("answer_reward", 0) > 0:
            correct_lengths.append(tok_len)
        else:
            wrong_lengths.append(tok_len)

    all_lengths = [len(r.split()) for r in responses]
    metrics: dict[str, Any] = {
        "mean_reward": sum(rewards) / len(rewards),
        "mean_format_reward": sum(format_rewards) / len(format_rewards),
        "mean_answer_reward": sum(answer_rewards) / len(answer_rewards),
        "mean_response_len": sum(all_lengths) / len(all_lengths),
        "mean_correct_len": sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0.0,
        "mean_wrong_len": sum(wrong_lengths) / len(wrong_lengths) if wrong_lengths else 0.0,
    }

    if wandb_run is not None:
        import wandb

        log_dict = {f"eval/{k}": v for k, v in metrics.items()}
        log_dict["eval_step"] = step

        sample_indices = random.sample(range(len(prompts)), min(n_log, len(prompts)))
        table = wandb.Table(
            columns=["step", "prompt", "response", "ground_truth", "reward"]
        )
        for idx in sample_indices:
            table.add_data(
                step, prompts[idx][:500], responses[idx][:1000],
                ground_truths[idx][:200], rewards[idx],
            )
        log_dict["eval/samples"] = table
        wandb_run.log(log_dict)

    for i, idx in enumerate(range(min(n_log, len(prompts)))):
        logger.info(
            "Step %d | example %d | reward=%.2f | format=%.2f | answer=%.2f\n"
            "  PROMPT: %s\n  RESPONSE: %s\n  GT: %s",
            step, idx, rewards[idx], format_rewards[idx], answer_rewards[idx],
            prompts[idx][:200], responses[idx][:500], ground_truths[idx][:200],
        )

    return metrics
