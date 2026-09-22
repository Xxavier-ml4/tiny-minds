"""Tokenizer interface, plus a working byte-level reference implementation.

TinyMind does not ship a trained subword tokenizer in this delivery — that
requires a trained SentencePiece/BPE model, which in turn wants a text
corpus and a training run neither of which exist yet (see
docs/architecture/tinymind-design.md section 7 for the same reasoning
applied to the model backend). ``ByteTokenizer`` below is not a placeholder
in the "raises NotImplementedError" sense: it is a complete, correct,
tested tokenizer over raw UTF-8 bytes plus a handful of special tokens, and
every other subsystem that needs "a tokenizer" to exercise its code path
(the .tm format's embedded tokenizer blob, the training dataset loader) can
use it for real today. A trained subword ``Tokenizer`` implementation is
future work; nothing in this module's interface would need to change to add
one.
"""
from __future__ import annotations

import abc
import hashlib
import json
from typing import Any, Sequence


class Tokenizer(abc.ABC):
    """Common interface every TinyMind tokenizer implements."""

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def pad_token_id(self) -> int: ...

    @property
    @abc.abstractmethod
    def bos_token_id(self) -> int: ...

    @property
    @abc.abstractmethod
    def eos_token_id(self) -> int: ...

    @abc.abstractmethod
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str: ...

    @abc.abstractmethod
    def spec(self) -> dict[str, Any]:
        """A JSON-serialisable description that fully determines this
        tokenizer's behaviour (type, version, vocabulary, special ids). It is
        what checkpoints and exported model packages store, and what
        ``tokenizer_from_spec`` rebuilds from — so a checkpoint can be
        validated against, and a model package loaded without, the code that
        trained it."""

    def spec_hash(self) -> str:
        """SHA-256 of the canonical JSON of ``spec()``: the tokenizer identity
        compared on resume and on stage promotion."""
        return hash_spec(self.spec())

    def __call__(self, texts: str | list[str], truncation: bool = True,
                 max_length: int | None = None, add_bos: bool = False,
                 add_eos: bool = False) -> dict[str, list]:
        """HF-tokenizer-shaped convenience call, batched or single."""
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        out_ids = []
        for text in batch:
            ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            out_ids.append(ids)
        return {"input_ids": out_ids[0] if single else out_ids}


class ByteTokenizer(Tokenizer):
    """UTF-8 byte-level tokenizer: 256 byte tokens + 4 special tokens.

    Vocabulary layout (0-259):
      0        PAD
      1        BOS
      2        EOS
      3        UNK  (reserved; byte-level encoding never actually emits it,
                      since every byte 0-255 has a token, but downstream
                      code that expects an UNK id — e.g. a from-scratch
                      subword tokenizer swapped in later — can rely on the
                      id being stable)
      4-259    byte values 0x00-0xFF

    This never fails to encode any string (UTF-8 covers all of Unicode) and
    round-trips exactly, which makes it a genuinely useful default for
    testing everything downstream of "a tokenizer exists," at the cost of
    long sequences relative to a trained subword vocabulary.
    """

    _NUM_SPECIALS = 4
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self) -> None:
        self._vocab_size = 256 + self._NUM_SPECIALS

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.PAD

    @property
    def bos_token_id(self) -> int:
        return self.BOS

    @property
    def eos_token_id(self) -> int:
        return self.EOS

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode() expects str, got {type(text).__name__}")
        ids = [b + self._NUM_SPECIALS for b in text.encode("utf-8")]
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        return ids

    def spec(self) -> dict[str, Any]:
        return {"type": "byte", "version": 1, "vocab_size": self._vocab_size,
                "pad_token_id": self.PAD, "bos_token_id": self.BOS, "eos_token_id": self.EOS,
                "unk_token_id": self.UNK, "byte_offset": self._NUM_SPECIALS}

    def decode(self, ids: Sequence[int]) -> str:
        raw = bytearray()
        for token_id in ids:
            if token_id in (self.PAD, self.BOS, self.EOS, self.UNK):
                continue
            byte_value = token_id - self._NUM_SPECIALS
            if not (0 <= byte_value <= 255):
                raise ValueError(f"token id {token_id} is out of range for ByteTokenizer "
                                 f"(vocab_size={self.vocab_size})")
            raw.append(byte_value)
        return raw.decode("utf-8", errors="replace")


def hash_spec(spec: dict[str, Any]) -> str:
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def tokenizer_from_spec(spec: dict[str, Any]) -> Tokenizer:
    """Rebuild a tokenizer from ``Tokenizer.spec()`` output. Unknown types or
    versions raise: a checkpoint from a tokenizer this code cannot reproduce
    must not be resumed or served with a guess."""
    kind = spec.get("type")
    if kind == "byte":
        if spec.get("version") != 1:
            raise ValueError(f"unsupported byte-tokenizer version {spec.get('version')!r}")
        tok = ByteTokenizer()
        if tok.spec() != spec:
            raise ValueError(f"byte-tokenizer spec does not match this build's ByteTokenizer: {spec}")
        return tok
    if kind == "bpe":
        try:
            from tinymind.model.bpe import BPETokenizer  # optional subword tokenizer
        except ImportError as exc:  # pragma: no cover - exercised only if the module is absent
            raise ValueError("this build has no BPE tokenizer implementation, so a checkpoint or "
                             "package that uses one cannot be loaded here") from exc
        return BPETokenizer.from_spec(spec)
    raise ValueError(f"unknown tokenizer type {kind!r}")
