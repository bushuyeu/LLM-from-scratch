# ece496b_basics/__init__.py

from typing import List, Dict, Tuple, Iterable  # Typing helpers for annotations
import os  # PathLike typing support
import regex as re  # Regex module with Unicode property support
from collections import defaultdict  # lets increment counts without checking for missing keys 

# GPT-2 pre-tokenization pattern - splits text into word-like chunks
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _iter_pretokens(text: str, special_tokens: List[str], pat: re.Pattern) -> Iterable[bytes]:  # Yield pre-tokens from text
    if special_tokens:  # Split on special tokens first
        delimiter = "|".join(re.escape(tok) for tok in special_tokens)  # Build escaped split pattern
        segments = re.split(delimiter, text)  # Split text around special tokens
    else:  # No special tokens provided
        segments = [text]  # Use full text as one segment

    for segment in segments:  # Process each segment separately
        if not segment:  # Skip empty segments
            continue  # Nothing to tokenize
        for match in pat.finditer(segment):  # Find regex matches
            token = match.group(0)  # Extract matched substring
            if token:  # Ensure token is non-empty
                yield token.encode("utf-8")  # Return UTF-8 bytes

# Apply a BPE merge to a token sequence, return new sequence and whether it changed
def merge_key(ids: Tuple[int, ...], pair: Tuple[int, int], idx: int) -> Tuple[Tuple[int, ...], bool]:
    new_ids: List[int] = []                      # Output list of token ids after merge
    i: int = 0                                   # Position pointer over input
    changed: bool = False                        # Track if any merge happened
    while i < len(ids):                          # Continue until end of sequence
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:  # Check if next two tokens match pair
            new_ids.append(idx)                  # Replace pair with new merged token
            i += 2                               # Skip both merged tokens
            changed = True                       # Mark that we made a change
        else:                                    # No match at this position
            new_ids.append(ids[i])               # Keep current token as-is
            i += 1                               # Advance by one
    return tuple(new_ids), changed               # Return new sequence and change flag


# Train a byte-level BPE tokenizer from a text file
def train_bpe(  
    input_path: str | os.PathLike,               # Path to training text file
    vocab_size: int,                             # Maximum final vocabulary size (bytes + merges + specials)
    special_tokens: List[str],                   # Special tokens to append to the vocabulary
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:  # Return vocab and merges

    assert vocab_size > 0, "vocab_size must be positive"  # Validate vocab_size is positive

    num_special: int = len(special_tokens)       # Count how many special tokens will be added
    num_merges: int = vocab_size - 256 - num_special  # Number of merges allowed after reserving space
    assert num_merges >= 0, f"vocab_size={vocab_size} is too small for 256 byte tokens + {num_special} special tokens"

    # read input
    with open(input_path, "r", encoding="utf-8") as f:  # Open file as UTF-8 text (regex operates on Unicode)
        text: str = f.read()                     # Read entire file as string

    # pre-tokenize and count unique chunks (deduplication with frequency counts)
    pre_token_counts: Dict[Tuple[int, ...], int] = defaultdict(int)  # Pre-token -> frequency
    pat = re.compile(GPT2_SPLIT_PATTERN)  # Compile GPT-2 split regex
    for token_bytes in _iter_pretokens(text, special_tokens, pat):  # Iterate pre-tokens
        key = tuple(token_bytes)  # Represent token as tuple of byte values
        pre_token_counts[key] += 1  # Increment count for this pre-token

    # build vocab incrementally (needed for tie-breaking by byte representation)
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}  # Start with 256 single-byte tokens

    merges_order: List[Tuple[int, int]] = []     # Ordered list of merges as (id0, id1) pairs

    # BPE training loop: learn merges one at a time
    for i in range(num_merges):                  # Repeat until we learn enough merges
        # Count all pairs across all unique pre-tokens (weighted by pre-token frequency)
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)  # Pair -> total frequency
        for token, count in pre_token_counts.items():  # Iterate over unique pre-tokens
            for j in range(len(token) - 1):      # For each adjacent pair in this pre-token
                pair_counts[(token[j], token[j + 1])] += count  # Add pre-token frequency to pair count

        if not pair_counts:                      # No pairs left to merge
            break                                # Stop training early

        # Find most frequent pair, ties broken by largest byte tuple representation
        best: Tuple[int, int] | None = None      # Best pair found so far
        best_count: int = -1                     # Count of best pair
        best_bytes: Tuple[bytes, bytes] | None = None  # Byte representation for tie-breaking
        for p, count in pair_counts.items():     # Iterate through all pairs
            if count > best_count:               # New highest count found
                best = p                         # Update best pair
                best_count = count               # Update best count
                best_bytes = (vocab[p[0]], vocab[p[1]])  # Update byte representation
            elif count == best_count:            # Tie in count
                p_bytes: Tuple[bytes, bytes] = (vocab[p[0]], vocab[p[1]])  # Get bytes of current pair
                if best_bytes is None or p_bytes > best_bytes:  # Ensure best_bytes is not None before comparing
                    best = p                     # Update best pair
                    best_bytes = p_bytes         # Update byte representation

        idx: int = 256 + i                       # Assign next token id after base 0..255 byte ids

        # Apply merge to all pre-tokens and rebuild counts
        new_pre_token_counts: Dict[Tuple[int, ...], int] = defaultdict(int)  # New pre-token -> frequency
        for token, count in pre_token_counts.items():  # Iterate over current pre-tokens
            new_token, changed = merge_key(token, best, idx)  # Apply merge to this pre-token
            new_pre_token_counts[new_token] += count  # Add count to (possibly merged) pre-token

        pre_token_counts = new_pre_token_counts  # Replace with updated pre-token counts
        merges_order.append(best)                # Append to ordered list of merges
        vocab[idx] = vocab[best[0]] + vocab[best[1]]  # Add new merged token to vocab

    # add special tokens to vocab
    next_id: int = 256 + len(merges_order)       # Compute next available token id after learned merges

    for sp in special_tokens:                    # Add special tokens after training
        vocab[next_id] = sp.encode("utf-8")      # Store special token bytes in vocab
        next_id += 1                             # Advance to next free token id

    merges_bytes: List[Tuple[bytes, bytes]] = [  # Convert merges from id-pairs to bytes-pairs
        (vocab[a], vocab[b])                     # Each merge represented by bytes of its two tokens
        for (a, b) in merges_order               # Preserve the exact order merges were created
    ]

    return vocab, merges_bytes                   # Return the final vocabulary and ordered list of merges