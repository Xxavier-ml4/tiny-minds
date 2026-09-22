"""Causal language-modeling loss: shift-by-one next-token cross entropy,
built on ``tinymind.model.tensor.cross_entropy`` (see that function for the
actual gradient formula and its numerical-gradient test).

Given ``input_ids = [x0, x1, ..., x_{T-1}]``, the model's logits at
position ``i`` predict ``x_{i+1}``; position ``T-1``'s prediction has no
target and is dropped, matching the brief section 10 exactly:
``targets = [x1, x2, ..., xT]`` is a description of the same shift.
"""
from __future__ import annotations

import numpy as np

from tinymind.model.tensor import Tensor, cross_entropy

_IGNORE_INDEX = -100


def causal_lm_loss(logits: Tensor, input_ids: np.ndarray,
                   attention_mask: np.ndarray | None = None,
                   ignore_index: int = _IGNORE_INDEX,
                   normalizer: float | None = None) -> Tensor:
    """``logits``: ``[B, T, vocab_size]``. ``input_ids``: ``[B, T]``.
    ``attention_mask``: ``[B, T]`` of 0/1, 0 marking a padding position to
    exclude from the loss (a padding position's *target* is excluded — its
    input is still seen by attention, since padding is conventionally
    causal-masked separately if needed, which is out of scope for the
    single-sequence-per-row case this reference model targets).

    ``input_ids`` may itself contain ``ignore_index`` (-100) entries: that is
    how completion-only training masks the prompt (see
    ``tinymind/training/render.py``). ``normalizer`` overrides the divisor
    (default: number of non-ignored targets in this call) — used by gradient
    accumulation to divide every micro-batch by the whole effective batch's
    token count.
    """
    shifted_logits = logits[:, :-1, :]
    targets = np.array(input_ids[:, 1:])

    if attention_mask is not None:
        target_mask = np.array(attention_mask[:, 1:])
        targets = np.where(target_mask == 0, ignore_index, targets)

    return cross_entropy(shifted_logits, targets, ignore_index=ignore_index, normalizer=normalizer)
