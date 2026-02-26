from __future__ import annotations

import hashlib
import os
import pathlib
import unicodedata
from collections import defaultdict

import mmh3


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
) -> None:
    """Remove lines that appear more than once across all input files."""
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Pass 1: count line frequencies using hashes for memory efficiency
    line_counts: dict[str, int] = defaultdict(int)
    for filepath in input_files:
        with open(filepath) as f:
            for line in f:
                h = hashlib.md5(line.encode()).hexdigest()
                line_counts[h] += 1

    # Pass 2: rewrite files keeping only unique lines
    for filepath in input_files:
        filepath = pathlib.Path(filepath)
        out_path = output_directory / filepath.name
        with open(filepath) as fin, open(out_path, "w") as fout:
            for line in fin:
                h = hashlib.md5(line.encode()).hexdigest()
                if line_counts[h] == 1:
                    fout.write(line)


def _normalize_text(text: str) -> str:
    """Normalize text for minhash: lowercase, NFD, remove accents and punctuation."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    # Remove accent marks (combining characters)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Remove punctuation, keep alphanumeric and whitespace
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def _get_word_ngrams(text: str, n: int) -> set[str]:
    """Get set of word n-grams from text."""
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _compute_minhash_signature(ngrams: set[str], num_hashes: int) -> list[int]:
    """Compute minhash signature using mmh3 with different seeds."""
    if not ngrams:
        return [0] * num_hashes
    signature = []
    for seed in range(num_hashes):
        min_hash = min(mmh3.hash(ng, seed, signed=False) for ng in ngrams)
        signature.append(min_hash)
    return signature


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute exact Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    """Remove fuzzy duplicate documents using MinHash + LSH."""
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    rows_per_band = num_hashes // num_bands

    # Read documents and compute signatures
    documents: list[tuple[pathlib.Path, str, str, set[str], list[int]]] = []
    for filepath in input_files:
        filepath = pathlib.Path(filepath)
        with open(filepath) as f:
            raw_text = f.read()
        normalized = _normalize_text(raw_text)
        word_ngrams = _get_word_ngrams(normalized, ngrams)
        signature = _compute_minhash_signature(word_ngrams, num_hashes)
        documents.append((filepath, raw_text, normalized, word_ngrams, signature))

    # LSH: bucket documents by band
    candidate_pairs: set[tuple[int, int]] = set()
    for band_idx in range(num_bands):
        buckets: dict[tuple, list[int]] = defaultdict(list)
        start = band_idx * rows_per_band
        end = start + rows_per_band
        for doc_idx, (_, _, _, _, sig) in enumerate(documents):
            band_hash = tuple(sig[start:end])
            buckets[band_hash].append(doc_idx)
        # All pairs in the same bucket are candidates
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        candidate_pairs.add((bucket_docs[i], bucket_docs[j]))

    # Verify candidates with exact Jaccard similarity
    # Use union-find to cluster duplicates
    parent = list(range(len(documents)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in candidate_pairs:
        ngrams_i = documents[i][3]
        ngrams_j = documents[j][3]
        sim = _jaccard_similarity(ngrams_i, ngrams_j)
        if sim >= jaccard_threshold:
            union(i, j)

    # For each cluster, keep one document (the first by index)
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(documents)):
        clusters[find(idx)].append(idx)

    keep_indices = set()
    for members in clusters.values():
        keep_indices.add(min(members))

    # Write kept documents
    for idx in keep_indices:
        filepath, raw_text, _, _, _ = documents[idx]
        out_path = output_directory / filepath.name
        with open(out_path, "w") as f:
            f.write(raw_text)
