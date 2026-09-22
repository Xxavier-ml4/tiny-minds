"""Model checkpoint save/load — independent from the ``.tm`` deployment
container (``tinymind.runtime.format``; the bridge between real weights and
that format is ``tinymind/model/tm_export.py``). A checkpoint is for
iterating during training (it can hold optimizer state, full float32
weights, step count); a `.tm` file is for shipping a finished model to a
device. Conflating the two seemed like the wrong call — see
``docs/architecture/model-implementation.md``'s checkpoint-format section.

**No pickle.** Per the brief section 15 ("do not use pickle as the primary
persistent model format") and section 28 ("do not introduce... pickle-based
model loading"): weights are stored via ``numpy.savez`` with
``allow_pickle=False`` on load, which is NumPy's own binary ``.npy``
format per array (a documented, simple, non-pickle binary layout — pickle
is only ever invoked by NumPy for ``dtype=object`` arrays, which nothing
here ever creates), not Python's ``pickle`` module. Config and metadata are
plain JSON.

A checkpoint is a directory:

    <path>/
        config.json            # ModelConfig.to_dict()
        tokenizer_meta.json     # {"class": ..., "vocab_size": ...}
        training_meta.json      # optional: step, optimizer name, etc.
        weights.npz             # every named parameter, via np.savez
        optimizer_state.npz     # optional: AdamW's m/v moments, if given
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.model import TinyMindTransformer
from tinymind.model.optim import AdamW


class CheckpointError(Exception):
    pass


def save_pretrained(model: TinyMindTransformer, path: str | Path, *,
                    tokenizer_meta: dict[str, Any] | None = None,
                    training_meta: dict[str, Any] | None = None,
                    optimizer: AdamW | None = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    (path / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2), encoding="utf-8")
    (path / "tokenizer_meta.json").write_text(
        json.dumps(tokenizer_meta or {"class": "ByteTokenizer", "vocab_size": model.config.vocab_size},
                  indent=2), encoding="utf-8")
    if training_meta is not None:
        (path / "training_meta.json").write_text(json.dumps(training_meta, indent=2), encoding="utf-8")

    weights = {name: param.data for name, param in model.named_parameters()}
    np.savez(path / "weights.npz", **weights)

    if optimizer is not None:
        opt_state = {"step_count": np.array(optimizer.step_count)}
        for i, (m, v) in enumerate(zip(optimizer._m, optimizer._v)):
            opt_state[f"m_{i}"] = m
            opt_state[f"v_{i}"] = v
        np.savez(path / "optimizer_state.npz", **opt_state)


def from_pretrained(path: str | Path, *, seed: int | None = None) -> TinyMindTransformer:
    path = Path(path)
    config_path = path / "config.json"
    weights_path = path / "weights.npz"
    if not config_path.is_file():
        raise CheckpointError(f"no config.json in checkpoint directory {path}")
    if not weights_path.is_file():
        raise CheckpointError(f"no weights.npz in checkpoint directory {path}")

    config = ModelConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    model = TinyMindTransformer(config, seed=seed)  # random init, immediately overwritten below

    # allow_pickle=False is deliberate and load-bearing, not a default left
    # alone — see this module's docstring. Numeric .npz arrays never need
    # pickle; a checkpoint that does is untrusted/malformed input, not
    # something this loader will execute.
    with np.load(weights_path, allow_pickle=False) as loaded:
        loaded_names = set(loaded.files)
        model_names = {name for name, _ in model.named_parameters()}
        missing = model_names - loaded_names
        unexpected = loaded_names - model_names
        if missing:
            raise CheckpointError(f"checkpoint is missing parameter(s): {sorted(missing)}")
        if unexpected:
            raise CheckpointError(f"checkpoint has unexpected parameter(s) not in this "
                                  f"config's model: {sorted(unexpected)}")
        for name, param in model.named_parameters():
            array = loaded[name]
            if array.shape != param.data.shape:
                raise CheckpointError(
                    f"parameter {name!r} shape mismatch: checkpoint has {array.shape}, "
                    f"model expects {param.data.shape}")
            param.data[...] = array

    return model


def load_optimizer_state(path: str | Path, optimizer: AdamW) -> None:
    opt_path = Path(path) / "optimizer_state.npz"
    if not opt_path.is_file():
        raise CheckpointError(f"no optimizer_state.npz in checkpoint directory {path}")
    with np.load(opt_path, allow_pickle=False) as loaded:
        optimizer.step_count = int(loaded["step_count"])
        for i in range(len(optimizer.parameters)):
            optimizer._m[i] = loaded[f"m_{i}"]
            optimizer._v[i] = loaded[f"v_{i}"]
