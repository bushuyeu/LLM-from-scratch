#!/usr/bin/env python3

"""Train BPE tokenizer on TinyStories dataset."""

import time                              # For measuring training duration
import pickle                            # For serializing vocab and merges to disk
import argparse                          # For parsing command-line arguments
from pathlib import Path                 # For cross-platform file path handling

from ece496b_basics import train_bpe     # Import our BPE training function from the package


def main():                              # Main entry point function
    parser = argparse.ArgumentParser(description="Train BPE on TinyStories")  # Create argument parser with description
    parser.add_argument("--input", default="data/TinyStoriesV2-GPT4-train.txt", help="Input file path")  # Input file argument with default
    parser.add_argument("--vocab-size", type=int, default=10000, help="Target vocabulary size")  # Vocab size as integer, default 10k
    parser.add_argument("--output-dir", default="outputs", help="Output directory")  # Where to save results
    parser.add_argument("--profile", action="store_true", help="Run with profiling")  # Boolean flag for profiling (no value needed)
    args = parser.parse_args()           # Parse command-line arguments into args object

    # Create output directory
    output_dir = Path(args.output_dir)   # Convert string path to Path object
    output_dir.mkdir(exist_ok=True)      # Create directory if it doesn't exist, don't error if it does

    print(f"Input: {args.input}")        # Display input file path
    print(f"Vocab size: {args.vocab_size}")  # Display target vocab size
    print(f"Output dir: {output_dir}")   # Display output directory
    print()                              # Blank line for readability

    if args.profile:                     # If --profile flag was passed
        import cProfile                  # Import profiler (lazy import since not always needed)
        import pstats                    # Import stats formatter for profiler output
        
        profiler = cProfile.Profile()    # Create a new profiler instance
        profiler.enable()                # Start collecting profiling data

    # Train BPE
    print("Starting BPE training...")    # Status message
    start = time.time()                  # Record start time

    vocab, merges = train_bpe(           # Call our BPE training function
        args.input,                      # Path to training corpus
        vocab_size=args.vocab_size,      # Target vocabulary size
        special_tokens=["<|endoftext|>"] # TinyStories document delimiter token
    )

    elapsed = time.time() - start        # Calculate elapsed time in seconds
    print(f"\nTraining completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")  # Display duration

    if args.profile:                     # If profiling was enabled
        profiler.disable()               # Stop collecting profiling data
        print("\n" + "="*60)             # Print separator line
        print("PROFILING RESULTS (top 20 by cumulative time)")  # Header
        print("="*60)                    # Print separator line
        stats = pstats.Stats(profiler)   # Create stats object from profiler data
        stats.sort_stats('cumulative')   # Sort by cumulative time (total time in function + callees)
        stats.print_stats(20)            # Print top 20 time-consuming functions

    # Analyze results
    print("\n" + "="*60)                 # Print separator line
    print("RESULTS")                     # Header
    print("="*60)                        # Print separator line
    print(f"Vocab size: {len(vocab)}")   # Display actual vocab size (should match target)
    print(f"Number of merges: {len(merges)}")  # Display number of learned merges

    # Find longest token
    longest = max(vocab.values(), key=len)  # Find token with most bytes
    print(f"Longest token: {longest!r}") # Display raw bytes representation
    print(f"Longest token length: {len(longest)} bytes")  # Display byte length
    print(f"Longest token decoded: {longest.decode('utf-8', errors='replace')}")  # Decode to string, replace invalid UTF-8

    # Save results
    vocab_path = output_dir / "vocab_10k.pkl"   # Construct path for vocab file
    merges_path = output_dir / "merges_10k.pkl" # Construct path for merges file
    
    with open(vocab_path, "wb") as f:    # Open vocab file for binary writing
        pickle.dump(vocab, f)            # Serialize vocab dict to file
    with open(merges_path, "wb") as f:   # Open merges file for binary writing
        pickle.dump(merges, f)           # Serialize merges list to file
    
    print(f"\nSaved vocab to: {vocab_path}")   # Confirm vocab saved
    print(f"Saved merges to: {merges_path}")   # Confirm merges saved


if __name__ == "__main__":                
    main()                