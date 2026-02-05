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

## Key Files

- `ece496b_basics/__init__.py` — Main implementation (train_bpe, merge_key, _iter_pretokens)
- `tests/adapters.py` — Adapter functions mapping tests to implementations
- `pyproject.toml` — Project config (build, ruff, pytest settings)
- `glossary.md` — Domain terminology reference
