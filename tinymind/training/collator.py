"""``CausalLMCollator``: pads a list of ``TokenizedExample``s to the same
length and builds the matching attention mask — the first genuinely
model-dependent piece of ``tinymind.training`` (see that package's
``__init__.py`` docstring: this needed a real model's input layout to
implement meaningfully, which now exists).
"""
from __future__ import annotations

import numpy as np

from tinymind.training.dataset import TokenizedExample


class CausalLMCollator:
    def __init__(self, pad_token_id: int = 0, max_length: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def collate(self, examples: list[TokenizedExample]) -> dict[str, np.ndarray]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        sequences = [ex.input_ids for ex in examples]
        max_len = max(len(s) for s in sequences)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        input_ids = np.full((len(examples), max_len), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(examples), max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            truncated = seq[:max_len]
            input_ids[i, :len(truncated)] = truncated
            attention_mask[i, :len(truncated)] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.copy()}
