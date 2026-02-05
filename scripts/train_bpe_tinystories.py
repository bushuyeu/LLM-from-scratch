#!/usr/bin/env python3
"""Train BPE tokenizer on TinyStories dataset."""

import time
import pickle
import argparse
import tracemalloc
from pathlib import Path

from ece496b_basics import train_bpe


def main():
    parser = argparse.ArgumentParser(description="Train BPE on TinyStories")
    parser.add_argument("--input", default="data/TinyStoriesV2-GPT4-train.txt", help="Input file path")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Target vocabulary size")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--profile", action="store_true", help="Run with cProfile")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Input: {args.input}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Output dir: {output_dir}")
    print()

    if args.profile:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()

    tracemalloc.start()
    start = time.time()

    vocab, merges = train_bpe(
        args.input,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.time() - start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nTraining completed in {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    print(f"Peak memory: {peak_mem / 1e9:.2f} GB")

    if args.profile:
        profiler.disable()
        print("\n" + "=" * 60)
        print("PROFILING RESULTS (top 20 by cumulative time)")
        print("=" * 60)
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats.print_stats(20)

    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    longest = max(vocab.values(), key=len)
    print(f"Longest token: {longest!r}")
    print(f"Longest token length: {len(longest)} bytes")
    print(f"Longest token decoded: {longest.decode('utf-8', errors='replace')}")

    # Save results
    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"\nSaved vocab to: {vocab_path}")
    print(f"Saved merges to: {merges_path}")


if __name__ == "__main__":
    main()
