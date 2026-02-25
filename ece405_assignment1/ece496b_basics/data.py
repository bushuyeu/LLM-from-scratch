from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random input/target pairs for language modeling.

    Picks batch_size random starting positions in the dataset and
    extracts windows of context_length tokens. Targets are the same
    windows shifted right by one — standard next-token prediction setup.

    Returns (x, y) each of shape (batch_size, context_length).
    """
    max_start = len(dataset) - context_length              # last valid start index
    starts = np.random.randint(0, max_start, size=(batch_size,))

    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])

    return (
        torch.tensor(x, dtype=torch.long, device=device),  # inputs
        torch.tensor(y, dtype=torch.long, device=device),  # targets (shifted by 1)
    )
