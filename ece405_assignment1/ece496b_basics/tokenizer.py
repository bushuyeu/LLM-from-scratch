from __future__ import annotations

import json
from typing import Dict, Iterable, List, Tuple

import regex as re

from .train_bpe import GPT2_SPLIT_PATTERN


def _gpt2_byte_decoder() -> Dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode — maps unicode chars back to byte values.

    GPT-2 serializes byte-level tokens as printable unicode strings (e.g. space
    byte 0x20 becomes 'Ġ'). This decoder reverses that mapping so we can read
    vocab/merge files in GPT-2 format back into raw bytes.
    """
    # 188 bytes that are already printable — kept as-is
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # Remaining 68 bytes get shifted to chr(256 + n) for readability
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}              # unicode char -> byte value


class Tokenizer:
    """Byte-level BPE tokenizer compatible with GPT-2 pre-tokenization.

    Given a trained vocabulary and merge list (from train_bpe), encodes text
    into token IDs and decodes token IDs back to text. Matches tiktoken's
    GPT-2 encoding exactly.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        self.vocab = vocab                                 # id -> bytes
        self.merges = merges                               # ordered merge pairs

        # Sort special tokens longest-first so regex prefers longer matches
        # e.g. "<|endoftext|><|endoftext|>" beats "<|endoftext|>"
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        # Reverse vocab: bytes -> token id (for encoding)
        self._bytes_to_id: Dict[bytes, int] = {v: k for k, v in vocab.items()}

        # Merge priority table — lower index = merge first
        self._merge_rank: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        self._pat = re.compile(GPT2_SPLIT_PATTERN)        # pre-tokenization regex

        # Special-token splitting regex (longest first for greedy matching)
        if self.special_tokens:
            self._special_pat = re.compile(
                "|".join(re.escape(t) for t in self.special_tokens)
            )
        else:
            self._special_pat = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> Tokenizer:
        """Construct a Tokenizer from serialized vocab and merges files.

        Reads GPT-2 format: vocab is JSON {unicode_token: id}, merges is a
        text file with one space-separated pair per line. Both use GPT-2's
        bytes-to-unicode encoding for representing raw bytes as printable chars.
        """
        byte_decoder = _gpt2_byte_decoder()

        # Vocab: {unicode_string: token_id} → {token_id: raw_bytes}
        with open(vocab_filepath) as f:
            raw_vocab = json.load(f)
        vocab: Dict[int, bytes] = {
            token_id: bytes(byte_decoder[ch] for ch in token_str)
            for token_str, token_id in raw_vocab.items()
        }

        # Append any special tokens missing from the vocab
        if special_tokens:
            for sp in special_tokens:
                sp_bytes = sp.encode("utf-8")
                if sp_bytes not in {v for v in vocab.values()}:
                    vocab[len(vocab)] = sp_bytes

        # Merges: lines of "token_a token_b" → [(bytes_a, bytes_b), ...]
        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath) as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) == 2:                          # skip blank / header lines
                    a = bytes(byte_decoder[ch] for ch in parts[0])
                    b = bytes(byte_decoder[ch] for ch in parts[1])
                    merges.append((a, b))

        return cls(vocab, merges, special_tokens)

    # --- BPE merge application (encoding direction) ---

    def _bpe_encode_word(self, word_bytes: bytes) -> List[int]:
        """Apply BPE merges to a single pre-tokenized word.

        Starts with individual bytes, then iteratively merges the
        highest-priority adjacent pair until no more merges apply.
        Returns a list of token IDs.
        """
        if not word_bytes:
            return []

        tokens = [bytes([b]) for b in word_bytes]          # start as single bytes

        while len(tokens) >= 2:
            # Find the adjacent pair with the lowest merge rank
            best_rank = float("inf")
            best_pair = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:                          # no applicable merges left
                break

            # Merge all occurrences of the best pair in one pass
            merged = best_pair[0] + best_pair[1]
            new_tokens: List[bytes] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(merged)              # replace pair with merged
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self._bytes_to_id[t] for t in tokens]      # map bytes -> IDs

    def _encode_segment(self, text: str) -> List[int]:
        """Encode a text segment (no special tokens) using GPT-2 regex + BPE.

        Splits text into word-like chunks via the regex, then applies BPE
        merges to each chunk independently.
        """
        ids: List[int] = []
        for match in self._pat.finditer(text):
            word_bytes = match.group(0).encode("utf-8")
            ids.extend(self._bpe_encode_word(word_bytes))
        return ids

    # --- Public API ---

    def encode(self, text: str) -> List[int]:
        """Encode a string to a list of token IDs.

        Special tokens are split out first (longest match wins), then each
        non-special segment is pre-tokenized and BPE-encoded.
        """
        if not text:
            return []

        ids: List[int] = []

        if self._special_pat:
            # Interleave: [part, special, part, special, ..., part]
            parts = self._special_pat.split(text)          # text between specials
            specials = self._special_pat.findall(text)     # the matched specials

            for i, part in enumerate(parts):
                if part:
                    ids.extend(self._encode_segment(part))
                if i < len(specials):                      # append special token ID
                    sp_bytes = specials[i].encode("utf-8")
                    ids.append(self._bytes_to_id[sp_bytes])
        else:
            ids = self._encode_segment(text)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Encode an iterable of text chunks, yielding token IDs one at a time.

        Memory-efficient: processes one chunk (line) at a time without loading
        the entire input into memory. Trailing whitespace is buffered across
        chunks so that e.g. consecutive "\\n" from separate lines merge into
        a single whitespace token — matching the result of encode() on the
        full concatenated text.
        """
        buffer = ""
        for chunk in iterable:
            text = buffer + chunk
            buffer = ""

            # Buffer trailing whitespace — it may extend into the next chunk
            # (e.g. "\\n" + "\\n" should become a single "\\n\\n" token)
            stripped = text.rstrip()
            if len(stripped) < len(text):
                buffer = text[len(stripped):]               # save for next iteration
                text = stripped

            if text:
                yield from self.encode(text)

        # Flush remaining buffered whitespace
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a string.

        Concatenates the byte representation of each token and decodes
        the result as UTF-8.
        """
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")
