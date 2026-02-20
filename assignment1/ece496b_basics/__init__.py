from typing import List, Dict, Tuple, Iterable
import os
import regex as re
from collections import defaultdict
import multiprocessing

# GPT-2 pre-tokenization pattern — splits text into word-like chunks
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _iter_pretokens(text: str, special_tokens: List[str], pat: re.Pattern) -> Iterable[bytes]:
    """Pre-tokenize text into UTF-8 byte chunks using the GPT-2 regex.

    Special tokens act as hard boundaries: the text is first split on them
    so they never get merged with surrounding content, then the GPT-2 regex
    is applied to each segment independently.
    """
    if special_tokens:
        delimiter = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(delimiter, text)           # split around special tokens
    else:
        segments = [text]

    for segment in segments:
        if not segment:
            continue
        for match in pat.finditer(segment):            # GPT-2 regex matches
            token = match.group(0)
            if token:
                yield token.encode("utf-8")            # yield as raw bytes

def _trim_incomplete_utf8(raw: bytes) -> bytes:
    """Trim a trailing incomplete UTF-8 multi-byte sequence.

    When a file is split into byte-range chunks for multiprocessing,
    a multi-byte character (e.g. é = 2 bytes) may straddle the boundary.
    This trims the incomplete tail; the next chunk picks up those bytes.
    """
    if not raw:
        return raw
    # Walk backwards past continuation bytes (10xxxxxx) to find the leading byte
    i = len(raw) - 1
    while i >= 0 and (raw[i] & 0xC0) == 0x80:
        i -= 1
    if i < 0:
        return raw
    # Determine how many bytes the leading byte expects
    lead = raw[i]
    expected = 4 if lead >= 0xF0 else 3 if lead >= 0xE0 else 2 if lead >= 0xC0 else 1
    if len(raw) - i < expected:                        # sequence extends past our slice
        return raw[:i]                                 # trim — next chunk will have it
    return raw


def _pretokenize_chunk(args: Tuple) -> Dict[Tuple[int, ...], int]:
    """Pre-tokenize one byte range of a file, returning token frequency counts.

    Each multiprocessing worker runs this on its own chunk.
    """
    input_path, start, end, special_tokens = args
    pat = re.compile(GPT2_SPLIT_PATTERN)
    counts: Dict[Tuple[int, ...], int] = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    raw = _trim_incomplete_utf8(raw)                   # safe UTF-8 boundary
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
    """Find byte offsets that split a file into chunks aligned to special-token boundaries.

    Starting from evenly spaced positions, each boundary is moved forward to
    the next occurrence of split_special_token so no document is cut in half.
    """
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        mini_chunk_size = 4096
        # Overlap reads by token length - 1 so a special token straddling
        # two 4096-byte windows is still found
        overlap = len(split_special_token) - 1
        for bi in range(1, len(boundaries) - 1):       # first and last are fixed
            initial_position = boundaries[bi]
            f.seek(initial_position)
            while True:
                mini_chunk = f.read(mini_chunk_size)
                if mini_chunk == b"":                  # reached EOF
                    boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:                     # found the token
                    boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size - overlap

    return sorted(set(boundaries))


def _adjust_pair(pair_counts, pair, delta):
    """Increment/decrement a pair count; remove the entry if it hits zero."""
    new_val = pair_counts.get(pair, 0) + delta
    if new_val <= 0:
        pair_counts.pop(pair, None)
    else:
        pair_counts[pair] = new_val


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer from a text file.

    Algorithm:
      1. Pre-tokenize the file into word-like chunks (parallel for large files)
      2. Build pair frequency counts and a reverse index once
      3. Iteratively merge the most frequent pair, updating counts incrementally
         (only touching pre-tokens that contain the merged pair)

    Returns:
        vocab:  dict mapping token id -> bytes
        merges: ordered list of (bytes, bytes) merge pairs
    """
    assert vocab_size > 0, "vocab_size must be positive"
    num_special = len(special_tokens)
    num_merges = vocab_size - 256 - num_special        # 256 byte tokens are always present
    assert num_merges >= 0, f"vocab_size={vocab_size} too small for 256 bytes + {num_special} special tokens"

    # --- Phase 1: Pre-tokenize (parallel for large files) ---
    input_path = str(input_path)
    file_size = os.path.getsize(input_path)
    num_workers = max(1, os.cpu_count() or 1)

    if file_size > 1_000_000 and num_workers > 1:
        # Split file into chunks aligned to special-token boundaries
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        boundaries = _find_chunk_boundaries(input_path, num_workers, split_token)
        chunk_args = [
            (input_path, boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
        ]
        # Each worker pre-tokenizes its chunk and returns a count dict
        with multiprocessing.Pool(num_workers) as pool:
            chunk_results = pool.map(_pretokenize_chunk, chunk_args)
        # Merge per-worker counts into a single global count dict
        pre_token_counts: Dict[Tuple[int, ...], int] = {}
        for chunk_counts in chunk_results:
            for key, count in chunk_counts.items():
                pre_token_counts[key] = pre_token_counts.get(key, 0) + count
    else:
        # Small file: single-threaded to avoid multiprocessing overhead
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        pre_token_counts = {}
        pat = re.compile(GPT2_SPLIT_PATTERN)
        for token_bytes in _iter_pretokens(text, special_tokens, pat):
            key = tuple(token_bytes)
            pre_token_counts[key] = pre_token_counts.get(key, 0) + 1

    # --- Phase 2: Build data structures for incremental merging ---
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}  # 256 single-byte tokens

    # Store pre-tokens as mutable lists (for in-place merging) with frequency weights.
    # Identical pre-tokens are already deduplicated; the weight tracks how many times
    # each one appeared in the corpus.
    pt_tokens: List[List[int]] = []                    # mutable token sequences
    pt_weights: List[int] = []                         # frequency weight per sequence
    for token_tuple, weight in pre_token_counts.items():
        pt_tokens.append(list(token_tuple))
        pt_weights.append(weight)
    del pre_token_counts                               # free memory

    # pair_counts:  total weighted frequency of each adjacent pair across all pre-tokens
    # pair_to_pts:  reverse index — which pre-token indices contain each pair.
    #               This is a superset (may have stale entries after merges) but that's
    #               safe: stale entries just cause a no-op scan on that pre-token.
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
    # Instead of recomputing all pair counts from scratch each iteration (O(total_tokens)
    # per merge), we update only the counts affected by each merge (O(affected_pretokens)).
    for i in range(num_merges):
        if not pair_counts:
            break

        # Find most frequent pair; ties broken by largest byte representation
        best = None
        best_count = -1
        best_bytes = None
        for p, count in pair_counts.items():
            if count > best_count:
                best, best_count = p, count
                best_bytes = (vocab[p[0]], vocab[p[1]])
            elif count == best_count:
                p_bytes = (vocab[p[0]], vocab[p[1]])
                if p_bytes > best_bytes:
                    best, best_bytes = p, p_bytes

        new_id = 256 + i                               # new merged token id
        a, b = best
        vocab[new_id] = vocab[a] + vocab[b]            # register merged token
        merges_order.append(best)

        # Only process pre-tokens that actually contain (a, b)
        affected = pair_to_pts.pop((a, b), set())
        pair_counts.pop((a, b), None)

        for pt_idx in affected:
            tokens = pt_tokens[pt_idx]
            w = pt_weights[pt_idx]
            j = 0
            while j < len(tokens) - 1:
                if tokens[j] == a and tokens[j + 1] == b:
                    # Subtract counts for the neighbor pairs being destroyed
                    if j > 0:
                        _adjust_pair(pair_counts, (tokens[j - 1], a), -w)
                    if j + 2 < len(tokens):
                        _adjust_pair(pair_counts, (b, tokens[j + 2]), -w)

                    # Merge in place: replace (a, b) with new_id
                    tokens[j] = new_id
                    del tokens[j + 1]

                    # Add counts for the new neighbor pairs created by the merge
                    if j > 0:
                        new_left = (tokens[j - 1], new_id)
                        _adjust_pair(pair_counts, new_left, w)
                        pair_to_pts.setdefault(new_left, set()).add(pt_idx)
                    if j + 1 < len(tokens):
                        new_right = (new_id, tokens[j + 1])
                        _adjust_pair(pair_counts, new_right, w)
                        pair_to_pts.setdefault(new_right, set()).add(pt_idx)

                    # Don't advance j — tokens[j] is now new_id (which != a),
                    # so the next loop iteration will hit else and advance j.
                else:
                    j += 1

    # --- Phase 4: Append special tokens after all learned merges ---
    next_id = 256 + len(merges_order)
    for sp in special_tokens:
        vocab[next_id] = sp.encode("utf-8")
        next_id += 1

    merges_bytes = [(vocab[a_id], vocab[b_id]) for (a_id, b_id) in merges_order]
    return vocab, merges_bytes
