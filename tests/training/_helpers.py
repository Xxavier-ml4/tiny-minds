"""Shared fixtures for the Phase 3B training tests: a tiny model, a tiny
varied-length instruction dataset, and a fake clock."""
from __future__ import annotations

import tempfile
from pathlib import Path

from tinymind.model import ModelConfig
from tinymind.model.tokenizer import ByteTokenizer
from tinymind.training.config import TrainingConfig
from tinymind.training.data import DataSource, TokenizedDataset
from tinymind.training.engine import TrainingEngine
from tinymind.training.render import ChatRenderer

TOK = ByteTokenizer()
RENDERER = ChatRenderer(TOK)


def records(n: int, offset: int = 0, prefix: str = "e"):
    out = []
    for i in range(offset, offset + n):
        a = " ".join(str((i * 7 + j) % 10) for j in range(1 + i % 5))  # completions of 1..5 tokens-ish
        out.append({"id": f"{prefix}{i}", "category": "toy" if i % 2 else "toy2",
                    "messages": [{"role": "user", "content": f"repeat {i % 10}"},
                                 {"role": "assistant", "content": f"{i % 10} {a}"}]})
    return out


def dataset(n=40, offset=0, prefix="e", name="train", max_len=96):
    return TokenizedDataset.from_records(records(n, offset, prefix), RENDERER, max_len, name=name)


def model_config(**over):
    base = dict(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64,
                max_seq_len=96, vocab_size=TOK.vocab_size)
    base.update(over)
    return ModelConfig(**base)


def train_config(**over):
    base = dict(stage="stage0", seed=3, batch_size=4, max_steps=20, warmup_steps=3, learning_rate=3e-3,
                min_learning_rate=3e-4, max_seq_len=96, eval_interval=0, checkpoint_interval=10, log_interval=0,
                packing=False, keep_checkpoints=3)
    base.update(over)
    return TrainingConfig(**base)


class FakeClock:
    """Advances ``tick`` seconds every time it is read."""

    def __init__(self, tick: float = 1.0) -> None:
        self.now, self.tick = 0.0, tick

    def __call__(self) -> float:
        value = self.now
        self.now += self.tick
        return value


def make_engine(out, *, model_over=None, train_over=None, train_data=None, val_data=None, tokenizer=None,
                resume=None, init_from=None, clock=None, fault_hook=None, exporter=None, **kw):
    train_data = train_data or dataset()
    val_data = val_data or dataset(8, offset=1000, prefix="v", name="val")
    return TrainingEngine(model_config=model_config(**(model_over or {})), tokenizer=tokenizer or TOK,
                          config=train_config(**(train_over or {})), sources=[DataSource("train", train_data)],
                          validation=val_data, output_dir=out, resume=resume, init_from=init_from,
                          clock=clock or (lambda: 0.0), log=None, fault_hook=fault_hook, exporter=exporter, **kw)


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="tm-test-"))
