"""Bridge between a real, trained ``TinyMindTransformer`` and the existing
``.tm`` deployment container (``tinymind.runtime.format`` — unchanged
format, hardened further this phase; see that module).

Deliberately thin: this module's only job is mapping
``model.named_parameters()`` (dotted names, ``Tensor`` objects) to the
``.tm`` writer's expected ``{name: (dtype, shape, raw_bytes)}`` shape, and
back. All of the actual safety work (bounds checking, overlap/ header-
region checks, checksum verification) already lives in
``tinymind.runtime.format`` and is not duplicated here.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.model import TinyMindTransformer
from tinymind.runtime.format import ModelFormatError, read_model, write_model

_ARCHITECTURE_TAG = "tinymind-transformer-v1"


class TmExportError(ModelFormatError):
    pass


_RESERVED_METADATA = ("architecture", "container_kind", "model_config")


def export_to_tm(model: TinyMindTransformer, path: str | Path, extra_metadata: dict | None = None) -> None:
    """Write ``model`` as a float32 ``.tm`` deployment file. ``extra_metadata``
    (tokenizer spec, prompt-renderer spec, provenance, ...) is stored alongside
    the three core keys; readers that do not know a key ignore it, and the
    native runtime reads only ``architecture`` and ``model_config``. The file is
    written to a temporary name and renamed, so an interrupted export never
    leaves a truncated model where a good one used to be."""
    metadata = {
        "architecture": _ARCHITECTURE_TAG,
        "container_kind": "deployment",  # distinguishes from a training checkpoint — see checkpoint.py
        "model_config": model.config.to_dict(),
    }
    for key, value in (extra_metadata or {}).items():
        if key in _RESERVED_METADATA:
            raise TmExportError(f"extra_metadata may not override the reserved key {key!r}")
        metadata[key] = value
    tensors = {name: ("float32", tuple(param.data.shape), param.data.tobytes())
              for name, param in model.named_parameters()}
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        write_model(tmp, metadata=metadata, tensors=tensors)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_tm_metadata(path: str | Path) -> dict:
    """The metadata block of a ``.tm`` file (checksums of tensors are not read)."""
    return dict(read_model(path).metadata)


def import_from_tm(path: str | Path, *, seed: int | None = None) -> TinyMindTransformer:
    model_file = read_model(path)

    architecture = model_file.metadata.get("architecture")
    if architecture != _ARCHITECTURE_TAG:
        raise TmExportError(
            f"this .tm file declares architecture {architecture!r}, expected "
            f"{_ARCHITECTURE_TAG!r} — it was not exported by "
            "tinymind.model.tm_export.export_to_tm, or is a different format version")

    config_dict = model_file.metadata.get("model_config")
    if config_dict is None:
        raise TmExportError("this .tm file has no 'model_config' in its metadata; cannot "
                            "reconstruct the architecture to load weights into")
    config = ModelConfig.from_dict(config_dict)
    model = TinyMindTransformer(config, seed=seed)

    model_param_names = {name for name, _ in model.named_parameters()}
    file_tensor_names = set(model_file.tensors)
    missing = model_param_names - file_tensor_names
    unexpected = file_tensor_names - model_param_names
    if missing:
        raise TmExportError(f".tm file is missing parameter(s) this config's model needs: {sorted(missing)}")
    if unexpected:
        raise TmExportError(f".tm file has tensor(s) not used by this config's model: {sorted(unexpected)}")

    for name, param in model.named_parameters():
        entry = model_file.tensors[name]
        if tuple(entry.shape) != tuple(param.data.shape):
            raise TmExportError(
                f"tensor {name!r} shape mismatch: .tm file has {entry.shape}, "
                f"model expects {param.data.shape}")
        raw = model_file.read_tensor(name)  # checksum-verified inside read_tensor
        array = np.frombuffer(raw, dtype=np.float32).reshape(entry.shape)
        param.data[...] = array

    return model
