from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model state, optimizer state, and iteration to disk.

    Bundles everything into a single dict and uses torch.save (pickle).
    The iteration count lets us resume training from the exact step.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer from a saved checkpoint.

    Loads the state dicts into the provided model and optimizer
    (mutating them in place) and returns the saved iteration count
    so training can resume from where it left off.
    """
    checkpoint = torch.load(src, map_location="cpu")       # always load to CPU first
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
