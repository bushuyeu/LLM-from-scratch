# ece496b_basics/__init__.py

from typing import List, Dict, Tuple, Iterable  # Typing helpers for annotations
import os  # PathLike typing support
import regex as re  # Regex module with Unicode property support
from collections import defaultdict  # lets increment counts without checking for missing keys
import multiprocessing  # For parallel pre-tokenization

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


def _pretokenize_chunk(args: Tuple) -> Dict[Tuple[int, ...], int]:
    """Pre-tokenize a byte range of a file and return token counts."""
    input_path, start, end, special_tokens = args
    pat = re.compile(GPT2_SPLIT_PATTERN)
    counts: Dict[Tuple[int, ...], int] = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    text = raw.decode("utf-8", errors="ignore")
    for token_bytes in _iter_pretokens(text, special_tokens, pat):
        key = tuple(token_bytes)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _find_chunk_boundaries(
    file_path: str | os.PathLike,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """Split file into chunks on special token boundaries."""
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        mini_chunk_size = 4096
        for bi in range(1, len(boundaries) - 1):
            initial_position = boundaries[bi]
            f.seek(initial_position)
            while True:
                mini_chunk = f.read(mini_chunk_size)
                if mini_chunk == b"":
                    boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    return sorted(set(boundaries))


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

    # --- Phase 1: Pre-tokenize with multiprocessing ---
    input_path = str(input_path)
    file_size = os.path.getsize(input_path)

    # Use multiprocessing for files larger than 1MB
    num_workers = max(1, os.cpu_count() or 1)
    if file_size > 1_000_000 and num_workers > 1:
        # Determine split token for chunk boundaries
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        boundaries = _find_chunk_boundaries(input_path, num_workers, split_token)

        chunk_args = [
            (input_path, boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
        ]

        with multiprocessing.Pool(num_workers) as pool:
            chunk_results = pool.map(_pretokenize_chunk, chunk_args)

        # Merge all chunk counts
        pre_token_counts: Dict[Tuple[int, ...], int] = {}
        for chunk_counts in chunk_results:
            for key, count in chunk_counts.items():
                pre_token_counts[key] = pre_token_counts.get(key, 0) + count
    else:
        # Small file: single-threaded (avoids multiprocessing overhead)
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        pre_token_counts = {}
        pat = re.compile(GPT2_SPLIT_PATTERN)
        for token_bytes in _iter_pretokens(text, special_tokens, pat):
            key = tuple(token_bytes)
            pre_token_counts[key] = pre_token_counts.get(key, 0) + 1

    # --- Phase 2: Build data structures for incremental BPE ---
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Convert to indexed mutable pre-token lists with weights
    pt_tokens: List[List[int]] = []       # mutable token sequences
    pt_weights: List[int] = []            # frequency weights
    for token_tuple, weight in pre_token_counts.items():
        pt_tokens.append(list(token_tuple))
        pt_weights.append(weight)
    del pre_token_counts

    # Build initial pair counts and reverse index
    pair_counts: Dict[Tuple[int, int], int] = {}
    pair_to_pts: Dict[Tuple[int, int], set] = defaultdict(set)

    for idx in range(len(pt_tokens)):
        tokens = pt_tokens[idx]
        w = pt_weights[idx]
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + w
            pair_to_pts[pair].add(idx)

    merges_order: List[Tuple[int, int]] = []

    # --- Phase 3: Incremental BPE merge loop ---
    for i in range(num_merges):
        if not pair_counts:
            break

        # Find most frequent pair with tie-breaking by largest byte representation
        best: Tuple[int, int] = None
        best_count: int = -1
        best_bytes: Tuple[bytes, bytes] = None
        for p, count in pair_counts.items():
            if count > best_count:
                best = p
                best_count = count
                best_bytes = (vocab[p[0]], vocab[p[1]])
            elif count == best_count:
                p_bytes = (vocab[p[0]], vocab[p[1]])
                if p_bytes > best_bytes:
                    best = p
                    best_bytes = p_bytes

        new_id: int = 256 + i
        a, b = best

        # Update vocab
        vocab[new_id] = vocab[a] + vocab[b]
        merges_order.append(best)

        # Get affected pre-tokens and remove the merged pair from tracking
        affected = pair_to_pts.pop((a, b), set())
        pair_counts.pop((a, b), None)

        # Apply merge incrementally to each affected pre-token
        for pt_idx in affected:
            tokens = pt_tokens[pt_idx]
            w = pt_weights[pt_idx]

            j = 0
            while j < len(tokens) - 1:
                if tokens[j] == a and tokens[j + 1] == b:
                    # Remove old left neighbor pair count
                    if j > 0:
                        old_left = (tokens[j - 1], a)
                        pair_counts[old_left] = pair_counts.get(old_left, 0) - w
                        if pair_counts[old_left] <= 0:
                            pair_counts.pop(old_left, None)

                    # Remove old right neighbor pair count
                    if j + 2 < len(tokens):
                        old_right = (b, tokens[j + 2])
                        pair_counts[old_right] = pair_counts.get(old_right, 0) - w
                        if pair_counts[old_right] <= 0:
                            pair_counts.pop(old_right, None)

                    # Apply merge in-place
                    tokens[j] = new_id
                    del tokens[j + 1]

                    # Add new left neighbor pair
                    if j > 0:
                        new_left = (tokens[j - 1], new_id)
                        pair_counts[new_left] = pair_counts.get(new_left, 0) + w
                        pair_to_pts.setdefault(new_left, set()).add(pt_idx)

                    # Add new right neighbor pair
                    if j + 1 < len(tokens):
                        new_right = (new_id, tokens[j + 1])
                        pair_counts[new_right] = pair_counts.get(new_right, 0) + w
                        pair_to_pts.setdefault(new_right, set()).add(pt_idx)

                    # Don't increment j: after merge, tokens[j]=new_id.
                    # Since new_id != a (new_id >= 256+i, a < 256+i for previously
                    # existing tokens), the next check will fall to else and increment.
                else:
                    j += 1

    # --- Phase 4: Add special tokens ---
    next_id: int = 256 + len(merges_order)
    for sp in special_tokens:
        vocab[next_id] = sp.encode("utf-8")
        next_id += 1

    merges_bytes: List[Tuple[bytes, bytes]] = [
        (vocab[a_id], vocab[b_id])
        for (a_id, b_id) in merges_order
    ]

    return vocab, merges_bytes
