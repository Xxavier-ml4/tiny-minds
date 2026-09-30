"""A small byte-level BPE tokenizer, pure Python, no dependencies.

Why it exists: the byte tokenizer costs almost nothing in parameters (260 × hidden) but makes sequences ~1 token per
character, and attention cost grows with the square of that. A few hundred learned merges shorten sequences without
the parameter cost of a 32 K vocabulary (32 000 × 192 would be 6.1 M parameters on its own). ``benchmarks/tokenizer_compare.py``
measures the trade on the curriculum text.

Layout is a strict superset of ``ByteTokenizer``: ids 0-3 are PAD/BOS/EOS/UNK, 4-259 are the 256 byte values, and merge
``i`` is token ``260 + i``. With zero merges it *is* the byte tokenizer, so a byte-tokenizer model and a BPE model share
their embedding layout for the first 260 rows.

Pre-tokenisation (merges never cross these boundaries): an optional leading space plus a run of letters; an optional
leading space plus a **single digit** (so ``47`` is two tokens and arithmetic keeps digit-level structure); an optional
leading space plus a run of punctuation/underscore; or a run of whitespace. Training and encoding are deterministic:
ties between equally frequent pairs go to the smaller pair of ids.

Scope: implemented in Python only. The native runtime has no BPE reader, so a BPE model is not deployable to the
native runtime yet — it is a measured alternative, not the default.
"""
from __future__ import annotations

import heapq
import re
from collections import Counter
from typing import Any, Iterable, Sequence

from tinymind.model.tokenizer import Tokenizer

_PRETOKEN = re.compile(r" ?[^\W\d_]+| ?\d| ?(?:[^\s\w]|_)+|\s+")
PRETOKENIZER_ID = "letters-digit-punct-space-v1"
NUM_SPECIALS = 4
BASE = 256 + NUM_SPECIALS


class BPETokenizer(Tokenizer):
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, merges: Sequence[tuple[int, int]] = ()) -> None:
        self._merges = [(int(a), int(b)) for a, b in merges]
        self._rank = {pair: i for i, pair in enumerate(self._merges)}
        self._bytes: list[bytes] = [b""] * NUM_SPECIALS + [bytes([b]) for b in range(256)]
        for a, b in self._merges:
            if not (NUM_SPECIALS <= a < len(self._bytes) and NUM_SPECIALS <= b < len(self._bytes)):
                raise ValueError(f"merge ({a}, {b}) refers to a token that does not exist yet")
            self._bytes.append(self._bytes[a] + self._bytes[b])
        self._cache: dict[str, list[int]] = {}

    # ---- Tokenizer interface -------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self._bytes)

    @property
    def pad_token_id(self) -> int:
        return self.PAD

    @property
    def bos_token_id(self) -> int:
        return self.BOS

    @property
    def eos_token_id(self) -> int:
        return self.EOS

    def _encode_piece(self, piece: str) -> list[int]:
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        ids = [b + NUM_SPECIALS for b in piece.encode("utf-8")]
        while len(ids) > 1:
            best_rank, best_at = None, -1
            for i in range(len(ids) - 1):
                rank = self._rank.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_at = rank, i
            if best_rank is None:
                break
            ids[best_at:best_at + 2] = [BASE + best_rank]
        if len(self._cache) < 200_000:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode() expects str, got {type(text).__name__}")
        ids: list[int] = [self.BOS] if add_bos else []
        for piece in _PRETOKEN.findall(text):
            ids.extend(self._encode_piece(piece))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        raw = bytearray()
        for token_id in ids:
            if token_id in (self.PAD, self.BOS, self.EOS, self.UNK):
                continue
            if not (NUM_SPECIALS <= token_id < len(self._bytes)):
                raise ValueError(f"token id {token_id} is out of range for this BPE tokenizer (vocab_size={self.vocab_size})")
            raw += self._bytes[token_id]
        return raw.decode("utf-8", errors="replace")

    def spec(self) -> dict[str, Any]:
        return {"type": "bpe", "version": 1, "vocab_size": self.vocab_size, "pad_token_id": self.PAD, "bos_token_id": self.BOS,
                "eos_token_id": self.EOS, "unk_token_id": self.UNK, "byte_offset": NUM_SPECIALS,
                "pretokenizer": PRETOKENIZER_ID, "merges": [[a, b] for a, b in self._merges]}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "BPETokenizer":
        if spec.get("type") != "bpe" or spec.get("version") != 1 or spec.get("pretokenizer") != PRETOKENIZER_ID:
            raise ValueError(f"unsupported BPE spec (type/version/pretokenizer): {spec.get('type')!r}/{spec.get('version')!r}/"
                             f"{spec.get('pretokenizer')!r}")
        tok = cls([tuple(m) for m in spec["merges"]])
        if tok.spec() != spec:
            raise ValueError("BPE spec is inconsistent (vocab_size or special ids do not match its merges)")
        return tok

    # ---- training ------------------------------------------------------------------------
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "BPETokenizer":
        """Learn ``vocab_size - 260`` merges from ``texts``. Deterministic for a given corpus order-independent input
        (word frequencies only), so two runs on the same texts produce the same tokenizer.

        Each step merges the most frequent adjacent pair (pair frequency = the sum, over the distinct pieces, of the
        piece's frequency times the pair's occurrences in it, overlapping ones included; ties go to the smaller pair
        of ids), rewriting every occurrence left to right, and stops early only when no pair is left. The pair
        counts are kept up to date INCREMENTALLY — only the pieces that contain the merged pair are rewritten and
        re-counted — with a lazily invalidated heap for the maximum. That is the same procedure as recounting every
        pair of every piece at every step (tests/model/test_bpe.py checks the merge lists are identical) at a small
        fraction of the cost: a 16k vocabulary learned from megabytes of real text takes minutes, not an hour of the
        CI job's training budget."""
        if vocab_size < BASE:
            raise ValueError(f"vocab_size must be >= {BASE} (the byte tokenizer's size)")
        piece_freq: Counter = Counter()
        for text in texts:
            piece_freq.update(_PRETOKEN.findall(text))   # the pattern never matches an empty string
        words: list[list[int]] = [[b + NUM_SPECIALS for b in piece.encode("utf-8")] for piece in piece_freq]
        freqs: list[int] = list(piece_freq.values())
        del piece_freq
        counts: dict[tuple[int, int], int] = {}
        where: dict[tuple[int, int], set[int]] = {}   # pair -> pieces that contain (or once contained) it
        for index, (word, freq) in enumerate(zip(words, freqs)):
            for pair in zip(word, word[1:]):
                counts[pair] = counts.get(pair, 0) + freq
                where.setdefault(pair, set()).add(index)
        # Max-heap on (count, -a, -b) as a min-heap of (-count, a, b). An entry is current only while its count
        # equals the pair's count; every count change pushes a fresh entry, and stale ones are dropped on sight.
        heap = [(-count, a, b) for (a, b), count in counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        for i in range(vocab_size - BASE):
            while heap and counts.get((heap[0][1], heap[0][2]), 0) != -heap[0][0]:
                heapq.heappop(heap)
            if not heap:
                break   # no pair left anywhere: every piece is a single token
            _, a, b = heapq.heappop(heap)
            best = (a, b)
            merges.append(best)
            new_id = BASE + i
            delta: dict[tuple[int, int], int] = {}
            for index in where.pop(best, ()):
                word = words[index]
                out: list[int] = []
                j, n, hit = 0, len(word), False
                while j < n:
                    if j < n - 1 and word[j] == a and word[j + 1] == b:
                        out.append(new_id)
                        j += 2
                        hit = True
                    else:
                        out.append(word[j])
                        j += 1
                if not hit:
                    continue   # an index entry left over from before an earlier merge rewrote this piece
                freq = freqs[index]
                for pair in zip(word, word[1:]):
                    delta[pair] = delta.get(pair, 0) - freq
                for pair in zip(out, out[1:]):
                    delta[pair] = delta.get(pair, 0) + freq
                    if new_id in pair:
                        where.setdefault(pair, set()).add(index)
                words[index] = out
            for pair, change in delta.items():
                if change:
                    count = counts.get(pair, 0) + change
                    if count:
                        counts[pair] = count
                        heapq.heappush(heap, (-count, pair[0], pair[1]))
                    else:
                        del counts[pair]
        return cls(merges)
