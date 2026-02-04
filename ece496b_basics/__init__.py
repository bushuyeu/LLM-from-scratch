# ece496b_basics/__init__.py (production-grade optimized)

from typing import List, Dict, Tuple, Iterable  # Typing helpers for annotations
import os                                       # PathLike typing support
import regex as re                              # Regex module with Unicode property support
from collections import defaultdict             # Lets increment counts without checking for missing keys
import heapq                                    # Priority queue for O(log n) best pair finding
from multiprocessing import Pool, cpu_count    # Parallel processing for pre-tokenization

# GPT-2 pre-tokenization pattern - splits text into word-like chunks
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Compile pattern once at module level (avoid recompiling)
_COMPILED_PATTERN = re.compile(GPT2_SPLIT_PATTERN)


def _process_chunk(args: Tuple[str, List[str]]) -> Dict[Tuple[int, ...], int]:
    """Process a single text chunk - used for multiprocessing."""
    chunk, special_tokens = args                # Unpack arguments
    local_counts: Dict[Tuple[int, ...], int] = defaultdict(int)  # Local pre-token counts
    
    if special_tokens:                          # Split on special tokens first
        delimiter = "|".join(re.escape(tok) for tok in special_tokens)  # Build escaped pattern
        segments = re.split(delimiter, chunk)   # Split around special tokens
    else:                                       # No special tokens
        segments = [chunk]                      # Use full chunk
    
    for segment in segments:                    # Process each segment
        if not segment:                         # Skip empty
            continue
        for match in _COMPILED_PATTERN.finditer(segment):  # Find matches
            token = match.group(0)              # Get matched text
            if token:                           # Non-empty
                key = tuple(token.encode("utf-8"))  # Convert to byte tuple
                local_counts[key] += 1          # Count it
    
    return dict(local_counts)                   # Return as regular dict for pickling


def _parallel_pretokenize(text: str, special_tokens: List[str], num_workers: int = None) -> Dict[Tuple[int, ...], int]:
    """Pre-tokenize text using multiple CPU cores."""
    if num_workers is None:                     # Auto-detect workers
        num_workers = max(1, cpu_count() - 1)   # Leave one core free
    
    # Split text into chunks at document boundaries (special tokens)
    if special_tokens:                          # Have special tokens to split on
        delimiter = special_tokens[0]           # Use first special token as delimiter
        chunks = text.split(delimiter)          # Split into documents
        # Re-add delimiter to chunks (except last) for proper processing
        chunks = [c + delimiter if i < len(chunks) - 1 else c for i, c in enumerate(chunks)]
    else:                                       # No special tokens
        # Split into roughly equal chunks
        chunk_size = len(text) // num_workers + 1  # Size per worker
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Filter empty chunks
    chunks = [c for c in chunks if c.strip()]   # Remove empty/whitespace chunks
    
    if len(chunks) <= 1 or num_workers <= 1:    # Not worth parallelizing
        return _process_chunk((text, special_tokens))  # Process directly
    
    # Process chunks in parallel
    combined_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    
    with Pool(num_workers) as pool:             # Create process pool
        args = [(chunk, special_tokens) for chunk in chunks]  # Prepare arguments
        results = pool.map(_process_chunk, args)  # Map across workers
        
        for local_counts in results:            # Combine results
            for token, count in local_counts.items():
                combined_counts[token] += count # Aggregate counts
    
    return dict(combined_counts)                # Return combined counts


class MaxHeap:
    """Max-heap for efficiently finding the best pair."""
    
    def __init__(self):                         # Initialize empty heap
        self._heap: List[Tuple[int, bytes, bytes, Tuple[int, int]]] = []  # (neg_count, byte1, byte2, pair)
        self._valid: Dict[Tuple[int, int], int] = {}  # Track current valid counts
    
    def push(self, pair: Tuple[int, int], count: int, vocab: Dict[int, bytes]):
        """Add or update a pair in the heap."""
        self._valid[pair] = count               # Mark current count as valid
        if count > 0:                           # Only add if positive count
            # Use negative count for max-heap behavior (heapq is min-heap)
            # Include bytes for tie-breaking (larger bytes = higher priority)
            heapq.heappush(self._heap, (-count, vocab[pair[0]], vocab[pair[1]], pair))
    
    def pop_best(self, vocab: Dict[int, bytes]) -> Tuple[Tuple[int, int], int]:
        """Get the best (most frequent) pair."""
        while self._heap:                       # While heap not empty
            neg_count, b1, b2, pair = heapq.heappop(self._heap)  # Get top element
            count = -neg_count                  # Convert back to positive
            
            # Check if this entry is still valid (count matches current)
            if pair in self._valid and self._valid[pair] == count and count > 0:
                return pair, count              # Valid best pair found
            # Otherwise entry is stale, continue to next
        
        return None, 0                          # No valid pairs left
    
    def update(self, pair: Tuple[int, int], count: int, vocab: Dict[int, bytes]):
        """Update count for a pair (lazy deletion - just add new entry)."""
        self._valid[pair] = count               # Update valid count
        if count > 0:                           # Only add if positive
            heapq.heappush(self._heap, (-count, vocab[pair[0]], vocab[pair[1]], pair))
    
    def remove(self, pair: Tuple[int, int]):
        """Mark a pair as removed."""
        self._valid[pair] = 0                   # Mark as invalid


def _iter_pretokens(text: str, special_tokens: List[str], pat: re.Pattern) -> Iterable[bytes]:
    """Yield pre-tokens from text (kept for compatibility)."""
    if special_tokens:                          # Split on special tokens first
        delimiter = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(delimiter, text)
    else:
        segments = [text]

    for segment in segments:
        if not segment:
            continue
        for match in pat.finditer(segment):
            token = match.group(0)
            if token:
                yield token.encode("utf-8")


def train_bpe(
    input_path: str | os.PathLike,              # Path to training text file
    vocab_size: int,                            # Maximum final vocabulary size
    special_tokens: List[str],                  # Special tokens to append
    use_multiprocessing: bool = True,           # Enable parallel pre-tokenization
    num_workers: int = None,                    # Number of workers (None = auto)
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    assert vocab_size > 0                       # Validate vocab_size

    num_special = len(special_tokens)           # Count special tokens
    num_merges = vocab_size - 256 - num_special # Merges to learn
    assert num_merges >= 0                      # Ensure room for merges

    # Read input file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Pre-tokenize (parallel or single-threaded)
    if use_multiprocessing:                     # Use parallel processing
        pre_token_counts = _parallel_pretokenize(text, special_tokens, num_workers)
    else:                                       # Single-threaded fallback
        pre_token_counts = defaultdict(int)
        for token_bytes in _iter_pretokens(text, special_tokens, _COMPILED_PATTERN):
            pre_token_counts[tuple(token_bytes)] += 1
        pre_token_counts = dict(pre_token_counts)

    # Initialize vocabulary with 256 byte tokens
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges_order: List[Tuple[int, int]] = []

    # Build initial pair counts and index
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    pair_to_pretokens: Dict[Tuple[int, int], set] = defaultdict(set)

    for token, count in pre_token_counts.items():
        for j in range(len(token) - 1):
            pair = (token[j], token[j + 1])
            pair_counts[pair] += count
            pair_to_pretokens[pair].add(token)

    # Initialize max-heap for O(log n) best pair finding
    heap = MaxHeap()
    for pair, count in pair_counts.items():
        heap.push(pair, count, vocab)

    # BPE training loop
    for i in range(num_merges):
        # Find best pair using heap (O(log n) amortized)
        best, best_count = heap.pop_best(vocab)
        
        if best is None or best_count == 0:     # No valid pairs left
            break

        idx = 256 + i                           # New token id

        # Get affected pre-tokens
        affected_pretokens = pair_to_pretokens.get(best, set()).copy()
        
        if not affected_pretokens:              # No tokens to merge
            continue

        # Process only affected pre-tokens
        tokens_to_remove = []                   # Track tokens to remove
        tokens_to_add = {}                      # Track tokens to add

        for token in affected_pretokens:
            if token not in pre_token_counts:   # Token already processed
                continue
                
            count = pre_token_counts[token]
            tokens_to_remove.append(token)      # Mark for removal

            # Decrement old pair counts
            for j in range(len(token) - 1):
                old_pair = (token[j], token[j + 1])
                pair_counts[old_pair] -= count
                heap.update(old_pair, pair_counts[old_pair], vocab)
                pair_to_pretokens[old_pair].discard(token)
                if pair_counts[old_pair] <= 0:
                    pair_counts.pop(old_pair, None)
                    heap.remove(old_pair)

            # Apply merge
            new_token = []
            j = 0
            while j < len(token):
                if j < len(token) - 1 and token[j] == best[0] and token[j + 1] == best[1]:
                    new_token.append(idx)
                    j += 2
                else:
                    new_token.append(token[j])
                    j += 1
            new_token = tuple(new_token)

            # Track new token
            if new_token in tokens_to_add:
                tokens_to_add[new_token] += count
            else:
                tokens_to_add[new_token] = count

        # Update pre_token_counts
        for token in tokens_to_remove:
            del pre_token_counts[token]
        
        for new_token, count in tokens_to_add.items():
            if new_token in pre_token_counts:   # Merge with existing
                pre_token_counts[new_token] += count
            else:
                pre_token_counts[new_token] = count

            # Increment new pair counts
            for j in range(len(new_token) - 1):
                new_pair = (new_token[j], new_token[j + 1])
                old_count = pair_counts.get(new_pair, 0)
                pair_counts[new_pair] = old_count + count
                pair_to_pretokens[new_pair].add(new_token)
                heap.update(new_pair, pair_counts[new_pair], vocab)

        # Record merge
        merges_order.append(best)
        vocab[idx] = vocab[best[0]] + vocab[best[1]]

    # Add special tokens
    next_id = 256 + len(merges_order)
    for sp in special_tokens:
        vocab[next_id] = sp.encode("utf-8")
        next_id += 1

    # Convert merges to bytes format
    merges_bytes = [(vocab[a], vocab[b]) for (a, b) in merges_order]

    return vocab, merges_bytes