"""Datasets, deterministic data order, mixtures.

Three pieces:

``TokenizedDataset``
    Rendered examples plus an identity: ``content_hash`` covers the renderer
    (template + tokenizer identity) and every example's id, category, token
    ids and label mask, so a change to the data, the tokenizer or the
    template changes the hash. Checkpoints store it; resume compares it.

``DataPlan``
    A pure function ``(seed, epoch) -> ordered micro-batches``. There is no
    stateful RNG to restore: the position in the data is just
    ``(epoch, micro_batch_index)``, and the same plan rebuilt after a restart
    yields the identical batch at that position. The algorithm (named
    ``ALGORITHM``, recorded in checkpoints) is:

    1. per source ``i`` with weight ``w_i`` an epoch takes
       ``quota_i = round(w_i / sum(w) * epoch_examples)`` examples (largest
       remainder, so the quotas sum exactly), drawn from consecutive
       permutations of that source made with
       ``SeedSequence([seed, epoch, i, 1])`` (a small source is cycled);
    2. the concatenation is shuffled with ``SeedSequence([seed, epoch, 65535, 2])``;
    3. examples are packed greedily in that order (or one per row when not
       packing) and cut into micro-batches of ``batch_size`` rows; a final
       group too small to fill one optimizer step is dropped (``drop_last``).

    Because each source's permutation has its own seed, adding or removing
    another source does not reorder this one.

``split_records``
    A deterministic train/validation split keyed on a hash of the example id
    (stable under re-ordering and under appending data).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from tinymind.training.batching import Batch, collate_packed, collate_padded, plan_packed_rows
from tinymind.training.render import ChatRenderer, ExampleError, RenderedExample


class DatasetError(ValueError):
    pass


# --------------------------------------------------------------------------
# reading / splitting raw records
# --------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{number}: invalid JSON ({exc})") from exc
    return records


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_records(records: Sequence[dict[str, Any]], validation_fraction: float,
                  seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic split by ``sha256(seed:id)``; at least one validation
    example when the fraction is positive and there are two or more records."""
    if not 0.0 <= validation_fraction < 1.0:
        raise DatasetError(f"validation_fraction must be in [0, 1), got {validation_fraction}")
    keyed = []
    for record in records:
        rid = record.get("id") if isinstance(record, dict) else None
        if not isinstance(rid, str):
            raise DatasetError("every record needs a string 'id' to be split deterministically")
        keyed.append((int(hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest()[:12], 16), record))
    if validation_fraction == 0.0 or len(records) < 2:
        return [r for _, r in keyed], []
    n_val = max(1, int(round(validation_fraction * len(records))))
    ranked = sorted(range(len(keyed)), key=lambda i: (keyed[i][0], i))
    val_idx = set(ranked[:n_val])
    train = [r for i, (_, r) in enumerate(keyed) if i not in val_idx]
    val = [r for i, (_, r) in enumerate(keyed) if i in val_idx]
    return train, val


# --------------------------------------------------------------------------
# TokenizedDataset
# --------------------------------------------------------------------------
class TokenizedDataset:
    def __init__(self, examples: Sequence[RenderedExample], renderer: ChatRenderer, name: str = "dataset",
                 source_sha256: str | None = None, dropped: dict[str, int] | None = None) -> None:
        if not examples:
            raise DatasetError(f"{name}: no examples")
        ids = [e.example_id for e in examples]
        if len(set(ids)) != len(ids):
            dup = next(i for i in ids if ids.count(i) > 1)
            raise DatasetError(f"{name}: duplicate example id {dup!r}")
        self.examples = list(examples)
        self.renderer = renderer
        self.name = name
        self.source_sha256 = source_sha256
        self.dropped = dict(dropped or {})
        self._hash: str | None = None

    def __len__(self) -> int:
        return len(self.examples)

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_records(cls, records: Iterable[Any], renderer: ChatRenderer, max_seq_len: int, *,
                     overflow: str = "error", name: str = "dataset",
                     source_sha256: str | None = None) -> "TokenizedDataset":
        """``overflow``: what to do with an example longer than ``max_seq_len``
        — ``error`` (default), ``drop`` (counted in ``dropped``; the caller
        should surface it) or ``truncate`` (right-truncate; a truncated chat
        example loses its EOS, so this is meant for ``text`` data)."""
        if overflow not in ("error", "drop", "truncate"):
            raise ValueError("overflow must be error|drop|truncate")
        rendered: list[RenderedExample] = []
        dropped = {"too_long": 0, "truncated": 0}
        for position, record in enumerate(records):
            try:
                ex = renderer.render(record)
            except ExampleError as exc:
                raise DatasetError(f"{name}: record {position}: {exc}") from exc
            if len(ex) > max_seq_len:
                if overflow == "error":
                    raise DatasetError(f"{name}: example {ex.example_id!r} renders to {len(ex)} tokens "
                                       f"> max_seq_len {max_seq_len} (use overflow='drop' or 'truncate' explicitly)")
                if overflow == "drop":
                    dropped["too_long"] += 1
                    continue
                labels = ex.labels[:max_seq_len].copy()
                if not (labels[1:] != -100).any():
                    dropped["too_long"] += 1
                    continue
                ex = RenderedExample(ex.example_id, ex.category, ex.kind, ex.ids[:max_seq_len].copy(), labels)
                dropped["truncated"] += 1
            rendered.append(ex)
        return cls(rendered, renderer, name=name, source_sha256=source_sha256, dropped=dropped)

    @classmethod
    def from_jsonl(cls, path: str | Path, renderer: ChatRenderer, max_seq_len: int, *,
                   overflow: str = "error", name: str | None = None) -> "TokenizedDataset":
        return cls.from_records(read_jsonl(path), renderer, max_seq_len, overflow=overflow,
                                name=name or Path(path).name, source_sha256=file_sha256(path))

    # ---- identity -------------------------------------------------------
    @property
    def content_hash(self) -> str:
        if self._hash is None:
            digest = hashlib.sha256()
            digest.update(json.dumps(self.renderer.spec(), sort_keys=True).encode())
            for e in self.examples:
                meta = f"{e.example_id}\x00{e.category}\x00{e.kind}\x00{len(e)}".encode()
                digest.update(len(meta).to_bytes(4, "little") + meta)
                digest.update(np.ascontiguousarray(e.ids, dtype="<i4").tobytes())
                digest.update(np.ascontiguousarray(e.labels, dtype="<i4").tobytes())
            self._hash = digest.hexdigest()
        return self._hash

    def stats(self) -> dict[str, Any]:
        lengths = np.array([len(e) for e in self.examples])
        loss_tokens = sum(e.num_loss_tokens for e in self.examples)
        by_category: dict[str, int] = {}
        for e in self.examples:
            by_category[e.category or "(none)"] = by_category.get(e.category or "(none)", 0) + 1
        return {"name": self.name, "num_examples": len(self), "total_tokens": int(lengths.sum()),
                "loss_tokens": int(loss_tokens), "min_len": int(lengths.min()), "max_len": int(lengths.max()),
                "mean_len": round(float(lengths.mean()), 2), "by_category": by_category,
                "dropped": self.dropped, "content_hash": self.content_hash, "source_sha256": self.source_sha256}


# --------------------------------------------------------------------------
# DataPlan
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DataSource:
    name: str
    dataset: TokenizedDataset
    weight: float = 1.0


def _largest_remainder(weights: Sequence[float], total: int) -> list[int]:
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0).any() or w.sum() <= 0:
        raise DatasetError(f"mixture weights must be non-negative with a positive sum, got {list(weights)}")
    exact = w / w.sum() * total
    floors = np.floor(exact).astype(int)
    remainder = total - int(floors.sum())
    order = sorted(range(len(w)), key=lambda i: (-(exact[i] - floors[i]), i))
    for i in order[:remainder]:
        floors[i] += 1
    return [int(x) for x in floors]


class DataPlan:
    ALGORITHM = "tinymind-dataplan-v1"

    def __init__(self, sources: Sequence[DataSource], *, seed: int, batch_size: int, max_seq_len: int,
                 packing: bool, pad_id: int, epoch_examples: int | None = None) -> None:
        if not sources:
            raise DatasetError("a DataPlan needs at least one source")
        if len({s.name for s in sources}) != len(sources):
            raise DatasetError("source names must be unique")
        if batch_size < 1 or max_seq_len < 2:
            raise DatasetError("batch_size >= 1 and max_seq_len >= 2 required")
        self.sources = list(sources)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.max_seq_len = int(max_seq_len)
        self.packing = bool(packing)
        self.pad_id = int(pad_id)
        self.epoch_examples = int(epoch_examples) if epoch_examples else sum(len(s.dataset) for s in sources)
        self._quotas = _largest_remainder([s.weight for s in sources], self.epoch_examples)
        self._cache: tuple[int, list[list[tuple[int, int]]]] | None = None

    # ---- identity -------------------------------------------------------
    def identity(self) -> dict[str, Any]:
        return {"algorithm": self.ALGORITHM, "seed": self.seed, "batch_size": self.batch_size,
                "max_seq_len": self.max_seq_len, "packing": self.packing, "epoch_examples": self.epoch_examples,
                "sources": [{"name": s.name, "weight": s.weight, "num_examples": len(s.dataset),
                             "content_hash": s.dataset.content_hash} for s in self.sources]}

    def dataset_hash(self) -> str:
        """The data identity a checkpoint pins: what the examples are and how
        they are mixed, but not the batch size (which the training-config
        hash covers)."""
        ident = self.identity()
        core = {"sources": ident["sources"], "epoch_examples": ident["epoch_examples"],
                "algorithm": ident["algorithm"], "packing": ident["packing"], "max_seq_len": ident["max_seq_len"]}
        return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()

    # ---- epoch structure --------------------------------------------------
    def _epoch_order(self, epoch: int) -> np.ndarray:
        pieces = []
        for i, (source, quota) in enumerate(zip(self.sources, self._quotas)):
            if quota == 0:
                continue
            n = len(source.dataset)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, i, 1]))
            reps = -(-quota // n)
            idx = np.concatenate([rng.permutation(n) for _ in range(reps)])[:quota]
            pieces.append(np.stack([np.full(quota, i, dtype=np.int64), idx.astype(np.int64)], axis=1))
        order = np.concatenate(pieces)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 65535, 2]))
        return order[rng.permutation(len(order))]

    def epoch_rows(self, epoch: int) -> list[list[tuple[int, int]]]:
        """Rows of ``(source_index, example_index)``: one per padded example, or
        several per packed row."""
        if self._cache is not None and self._cache[0] == epoch:
            return self._cache[1]
        order = self._epoch_order(epoch)
        pairs = [(int(s), int(i)) for s, i in order]
        if self.packing:
            lengths = [len(self.sources[s].dataset.examples[i]) for s, i in pairs]
            rows = [[pairs[p] for p in row] for row in plan_packed_rows(lengths, self.max_seq_len)]
        else:
            rows = [[p] for p in pairs]
        self._cache = (epoch, rows)
        return rows

    def num_micro_batches(self, epoch: int) -> int:
        return len(self.epoch_rows(epoch)) // self.batch_size

    def micro_batch(self, epoch: int, index: int) -> Batch:
        rows = self.epoch_rows(epoch)
        if not 0 <= index < len(rows) // self.batch_size:
            raise IndexError(f"micro-batch {index} out of range for epoch {epoch} "
                             f"({len(rows) // self.batch_size} micro-batches)")
        chunk = rows[index * self.batch_size:(index + 1) * self.batch_size]
        resolved = [[self.sources[s].dataset.examples[i] for s, i in row] for row in chunk]
        if self.packing:
            return collate_packed(resolved, self.pad_id, self.max_seq_len)
        return collate_padded([row[0] for row in resolved], self.pad_id, self.max_seq_len)

    def steps_in_epoch(self, epoch: int, accumulation: int) -> int:
        return self.num_micro_batches(epoch) // accumulation

    def repeat_factors(self) -> dict[str, float]:
        """How many times each source's examples are seen per epoch (>10 is an
        overfitting risk worth a look)."""
        return {s.name: round(q / len(s.dataset), 3) for s, q in zip(self.sources, self._quotas)}


# --------------------------------------------------------------------------
# sequential evaluation batches
# --------------------------------------------------------------------------
def sequential_batches(dataset: TokenizedDataset, batch_size: int, max_seq_len: int, pad_id: int,
                       packing: bool = False, limit: int | None = None) -> Iterator[Batch]:
    """One pass over ``dataset`` in file order (no shuffling): what validation
    uses, so the number is comparable across runs and checkpoints."""
    examples = dataset.examples
    if packing:
        rows = [[examples[p] for p in row] for row in plan_packed_rows([len(e) for e in examples], max_seq_len)]
    else:
        rows = [[e] for e in examples]
    n_batches = math.ceil(len(rows) / batch_size)
    if limit is not None:
        n_batches = min(n_batches, limit)
    for b in range(n_batches):
        chunk = rows[b * batch_size:(b + 1) * batch_size]
        yield collate_packed(chunk, pad_id, max_seq_len) if packing else collate_padded([r[0] for r in chunk], pad_id, max_seq_len)
