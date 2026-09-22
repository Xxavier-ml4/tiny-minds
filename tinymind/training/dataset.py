"""Load JSONL training data (the format ``tinymind.data.validate`` checks)
into tokenized examples.

Real and testable today: it needs a tokenizer (``tinymind.model.tokenizer.
ByteTokenizer`` is real — see its module docstring) and validated JSONL
(``tinymind.data.validate`` is real), not a trained model. What a real
training *run* additionally needs — a model to compute gradients through —
is exactly the piece that doesn't exist yet (see
``tinymind/training/__init__.py`` for ``Trainer``'s status).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterator

from tinymind.model.tokenizer import Tokenizer


@dataclasses.dataclass
class TokenizedExample:
    example_id: str
    input_ids: list[int]
    target_type: str
    target_text: str
    """The target rendered as text (an answer string, or a tool call
    serialized as JSON) — rendering a *target* into training-ready label
    ids (as opposed to input ids) is architecture-specific (does the model
    predict the whole sequence, or just the completion?) and is deferred to
    ``tinymind.training.collator`` once a real model architecture exists to
    collate a batch for."""


class TrainingDataset:
    """Iterates ``TokenizedExample``s from a validated JSONL file.
    Deliberately does not cache the whole file in memory — a training
    corpus can be far larger than RAM, and streaming is the only version of
    this class that would still work at real scale, so this delivery
    doesn't build a second, in-memory-only version that would need
    rewriting later."""

    def __init__(self, path: str | Path, tokenizer: Tokenizer, *, max_length: int | None = None) -> None:
        self._path = Path(path)
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __iter__(self) -> Iterator[TokenizedExample]:
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                yield self._tokenize(example)

    def _tokenize(self, example: dict) -> TokenizedExample:
        prompt_text = " ".join(
            m.get("content", "") for m in example.get("messages", []) if isinstance(m, dict))
        input_ids = self._tokenizer.encode(prompt_text, add_bos=True)
        if self._max_length is not None:
            input_ids = input_ids[: self._max_length]

        target = example.get("target", {})
        target_type = target.get("type", "answer")
        if target_type == "tool_call":
            target_text = json.dumps({"name": target.get("name"), "arguments": target.get("arguments", {})})
        else:
            target_text = target.get("content", "")

        return TokenizedExample(example_id=example.get("id", ""), input_ids=input_ids,
                                target_type=target_type, target_text=target_text)

    def count(self) -> int:
        """Number of non-empty lines. Reads the file a second time rather
        than caching — see the class docstring on why this stays a stream."""
        with self._path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
