"""Optional supplement implementations: PackedSFTDataset and DPO loss."""

from __future__ import annotations

import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n### Response:\n{response}"
)


class PackedSFTDataset(Dataset):
    """Pack instruction-tuning examples into fixed-length token sequences."""

    def __init__(self, tokenizer, dataset_path, seq_length: int, shuffle: bool):
        with open(dataset_path) as f:
            examples = [json.loads(line) for line in f if line.strip()]

        if shuffle:
            random.shuffle(examples)

        all_ids: list[int] = []
        for ex in examples:
            text = _ALPACA_TEMPLATE.format(
                prompt=ex["prompt"],
                response=ex["response"],
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            all_ids.extend(ids)

        self._examples: list[dict[str, torch.Tensor]] = []
        for i in range(0, len(all_ids) - seq_length, seq_length):
            chunk = all_ids[i : i + seq_length + 1]
            if len(chunk) == seq_length + 1:
                self._examples.append(
                    {
                        "input_ids": torch.tensor(chunk[:seq_length], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._examples[idx]


def compute_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """Per-instance DPO loss.

    loss = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    """

    def _response_log_prob(model: torch.nn.Module, prompt_ids: list[int], full_ids: list[int]) -> torch.Tensor:
        input_t = torch.tensor(full_ids).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_t).logits  # (1, T, V)
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)  # (1, T-1, V)
        targets = torch.tensor(full_ids[1:]).unsqueeze(0)  # (1, T-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
        return token_lp[0, len(prompt_ids) - 1 :].sum()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_chosen = tokenizer.encode(prompt + response_chosen, add_special_tokens=False)
    full_rejected = tokenizer.encode(prompt + response_rejected, add_special_tokens=False)

    log_ratio_chosen = _response_log_prob(lm, prompt_ids, full_chosen) - _response_log_prob(lm_ref, prompt_ids, full_chosen)
    log_ratio_rejected = _response_log_prob(lm, prompt_ids, full_rejected) - _response_log_prob(lm_ref, prompt_ids, full_rejected)

    return -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
