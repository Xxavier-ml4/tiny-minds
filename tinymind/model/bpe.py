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
        (word frequencies only), so two runs on the same texts produce the same tokenizer."""
        if vocab_size < BASE:
            raise ValueError(f"vocab_size must be >= {BASE} (the byte tokenizer's size)")
        words: Counter = Counter()
        for text in texts:
            for piece in _PRETOKEN.findall(text):
                if len(piece) > 0:
                    words[tuple(b + NUM_SPECIALS for b in piece.encode("utf-8"))] += 1
        merges: list[tuple[int, int]] = []
        table = dict(words)
        for i in range(vocab_size - BASE):
            pairs: Counter = Counter()
            for word, freq in table.items():
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            best = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            merges.append(best)
            new_id = BASE + i
            merged: dict[tuple[int, ...], int] = {}
            for word, freq in table.items():
                if len(word) > 1:
                    out, j = [], 0
                    while j < len(word):
                        if j < len(word) - 1 and (word[j], word[j + 1]) == best:
                            out.append(new_id)
                            j += 2
                        else:
                            out.append(word[j])
                            j += 1
                    word = tuple(out)
                merged[word] = merged.get(word, 0) + freq
            table = merged
        return cls(merges)
