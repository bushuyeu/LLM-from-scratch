# CLAUDE.md

## Project Overview

ECE405 Assignment 1 (adapted from Stanford CS336 Spring 2025). Implements fundamental transformer language model components from scratch: BPE tokenizer, attention, normalization, optimizers, and training infrastructure.

## Tech Stack

- Python 3.11+
- PyTorch (~2.6.0, or 2.2.2 for Intel Macs)
- Package manager: `uv`
- Linter/formatter: `ruff` (line length 120)

## Project Structure

- `ece496b_basics/` — Core implementations (BPE tokenizer, etc.)
- `cs336_basics/` — Adapter/compatibility layer
- `tests/` — Pytest suite with snapshot testing
- `tests/adapters.py` — Bridge between test suite and implementations
- `tests/fixtures/` — Reference data (GPT2 vocab, model weights)
- `tests/_snapshots/` — Snapshot files (.npz, .pkl)

## Commands

```sh
uv run pytest                    # Run all tests
uv run pytest tests/test_name.py # Run specific test file
uv run python <script.py>        # Run a script
```

## Code Conventions

- **Naming conventions must match the assignment spec exactly.** Constructor parameters, attribute names, and method signatures should use the same names as the assignment description (e.g., `in_features`/`out_features` for Linear, not `d_in`/`d_out`).
- Type annotations with `jaxtyping` for tensor shapes (e.g., `Float[Tensor, "batch seq d_model"]`)
- Google-style docstrings
- snake_case for functions/variables, SCREAMING_SNAKE_CASE for constants
- `from __future__ import annotations` in modules
- Ruff rules: `extend-select = ["UP"]`, ignore `F722`
- `__init__.py` files ignore `E402`, `F401`, `F403`, `E501`

## Testing

- Framework: pytest
- Snapshot testing via custom `NumpySnapshot` and `Snapshot` fixtures
- Memory limit testing with custom `memory_limit()` decorator
- Fixed random seeds for reproducibility
- Tests initially raise `NotImplementedError` — implement in code, connect via `tests/adapters.py`

## Data

Tokenized datasets are stored on Hugging Face Hub (private repo):
- **Repo:** `bushuyeu/ece405-tokenized-data`
- **Files:** `ts_train.npy`, `ts_valid.npy`, `owt_train.npy`, `owt_valid.npy` (uint16 numpy arrays)
- **Download:** `huggingface-cli download bushuyeu/ece405-tokenized-data --repo-type dataset --local-dir outputs/`
- BPE vocab/merges pickles are in `outputs/` locally (not on HF)

## Key Files

- `ece496b_basics/__init__.py` — Exports all implementations
- `ece496b_basics/nn.py` — All nn.Module layers, attention, transformer, generate()
- `ece496b_basics/optimizer.py` — AdamW, cosine LR schedule
- `ece496b_basics/data.py` — get_batch()
- `ece496b_basics/checkpointing.py` — save/load checkpoint
- `ece496b_basics/train_bpe.py` — BPE training
- `ece496b_basics/tokenizer.py` — Tokenizer encode/decode
- `train.py` — Main training script (CLI args, W&B, checkpointing)
- `tests/adapters.py` — Adapter functions mapping tests to implementations
- `pyproject.toml` — Project config (build, ruff, pytest settings)
- `glossary.md` — Domain terminology reference

## Submission Guidelines (from professor)

- **Format:** Single zip file containing the full project
- **Report structure:** Follow the same problem-by-problem structure as the assignment PDF
- **For each problem:** Include a brief description + point to specific file name and path where the deliverable is implemented
- **No code screenshots** — grading is done directly from the zip
- **Screenshots of experimental results** (W&B plots, loss curves, generated text) are desired
- **Emphasis:** Quality of experimental results and conclusions over code authenticity (AI tools allowed)
- **Report file:** `notebooks/writeup_draft.md` (master writeup) and `notebooks/ECE405_Assignment1_write-up_Pavel_Bushuyeu.ipynb` (notebook version)
- **Experiments notebook:** `notebooks/experiments_analysis.ipynb` — pulls W&B data, generates plots
