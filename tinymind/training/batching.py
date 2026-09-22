"""Turn rendered examples into model-ready batches: padded, or packed.

*Padded* rows hold one example, right-padded; padding needs no attention mask
because attention is causal (a real token never sees the pad tokens after
it) and padded positions carry label -100.

*Packed* rows hold several examples back to back (greedy, in the order given,
so the data order stays a pure function of the epoch plan). Correctness rules,
each covered by ``tests/training/test_batching.py``:

* ``segment_ids`` (1, 2, ... per example; 0 = tail padding) make attention
  block-diagonal causal, so an example can never attend to a neighbour;
* RoPE positions restart at 0 in every segment (the model derives them from
  ``segment_ids``), so a packed example sees the positions it would see alone;
* every example's first label is -100 (the renderer guarantees it), so the
  last token of one example is never trained to predict the first token of
  the next.

Every batch carries ``num_loss_tokens``, the count of targets that receive
loss after the model's internal shift; the trainer sums it over the
micro-batches of an optimizer step and passes the total to the loss as its
divisor, which is what makes ``batch x accumulation`` exactly equivalent to
one bigger batch.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from tinymind.training.render import IGNORE_INDEX, RenderedExample


@dataclasses.dataclass
class Batch:
    input_ids: np.ndarray                 # int64 [B, T]
    labels: np.ndarray                    # int64 [B, T], -100 = no loss
    segment_ids: np.ndarray | None        # int64 [B, T] when packed, else None
    num_examples: int
    num_real_tokens: int                  # tokens that are not padding
    num_loss_tokens: int                  # targets with loss, after the shift

    @property
    def shape(self) -> tuple[int, int]:
        return self.input_ids.shape  # type: ignore[return-value]

    @property
    def padded_tokens(self) -> int:
        return int(self.input_ids.size)


def _count_loss_tokens(labels: np.ndarray) -> int:
    return int((labels[:, 1:] != IGNORE_INDEX).sum())


def collate_padded(examples: Sequence[RenderedExample], pad_id: int, max_seq_len: int) -> Batch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    longest = max(len(e) for e in examples)
    if longest > max_seq_len:
        raise ValueError(f"example {[e.example_id for e in examples if len(e) == longest][0]!r} has {longest} "
                         f"tokens > max_seq_len {max_seq_len}; fix the overflow policy when building the dataset")
    ids = np.full((len(examples), longest), pad_id, dtype=np.int64)
    labels = np.full((len(examples), longest), IGNORE_INDEX, dtype=np.int64)
    for row, e in enumerate(examples):
        ids[row, :len(e)] = e.ids
        labels[row, :len(e)] = e.labels
    return Batch(ids, labels, None, len(examples), sum(len(e) for e in examples), _count_loss_tokens(labels))


def plan_packed_rows(lengths: Sequence[int], max_seq_len: int) -> list[list[int]]:
    """Greedy sequential packing: positions ``0..n-1`` into rows of at most
    ``max_seq_len`` tokens, never reordering (next-fit). Deterministic."""
    rows: list[list[int]] = []
    used = 0
    for position, length in enumerate(lengths):
        if length > max_seq_len:
            raise ValueError(f"example at position {position} has {length} tokens > max_seq_len {max_seq_len}")
        if rows and used + length <= max_seq_len:
            rows[-1].append(position)
            used += length
        else:
            rows.append([position])
            used = length
    return rows


def collate_packed(row_examples: Sequence[Sequence[RenderedExample]], pad_id: int, max_seq_len: int) -> Batch:
    """``row_examples[r]`` are the examples packed (in order) into row ``r``."""
    if not row_examples:
        raise ValueError("cannot collate an empty batch")
    widths = [sum(len(e) for e in row) for row in row_examples]
    if max(widths) > max_seq_len:
        raise ValueError(f"a packed row has {max(widths)} tokens > max_seq_len {max_seq_len}")
    width = max(widths)
    ids = np.full((len(row_examples), width), pad_id, dtype=np.int64)
    labels = np.full((len(row_examples), width), IGNORE_INDEX, dtype=np.int64)
    segments = np.zeros((len(row_examples), width), dtype=np.int64)
    real = 0
    count = 0
    for r, row in enumerate(row_examples):
        cursor = 0
        for seg_no, e in enumerate(row, start=1):
            if e.labels[0] != IGNORE_INDEX:
                raise ValueError(f"example {e.example_id!r}: labels[0] must be -100 to pack safely")
            n = len(e)
            ids[r, cursor:cursor + n] = e.ids
            labels[r, cursor:cursor + n] = e.labels
            segments[r, cursor:cursor + n] = seg_no
            cursor += n
            count += 1
        real += cursor
    return Batch(ids, labels, segments, count, real, _count_loss_tokens(labels))
