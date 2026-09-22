"""The inference package — what ships to a device.

A package is a directory of plain files (no pickle, no code)::

    model.tm       canonical FP32 weights (the ``.tm`` container: per-tensor CRC32);
                   its metadata also carries the model config, the tokenizer spec,
                   the prompt-template spec and provenance, so the single file is
                   self-describing
    tokenizer.json the tokenizer spec (type, version, special ids; vocabulary for
                   learned tokenizers) — enough to rebuild the tokenizer
    package.json   manifest: format version, architecture id, model config + hash,
                   parameter count, tokenizer/template identity, provenance, and
                   SHA-256 + size of every file above

It contains **no optimizer state, no training data, no training framework**,
and loading it needs nothing but this package's reader (or the native
runtime for ``model.tm``): no Python training code, no GitHub Actions, no
network. ``tests/export/test_package.py`` loads one in a subprocess where
``tinymind.training`` is made unimportable, then compares logits with the
in-memory model.

Packages are written to a temporary directory and renamed into place.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tinymind.data.render import TEMPLATE_ID, ChatRenderer
from tinymind.model.config import ModelConfig
from tinymind.model.generation import ModelGenerationConfig, generate_with_cache_ids
from tinymind.model.model import TinyMindTransformer
from tinymind.model.tm_export import export_to_tm, import_from_tm, read_tm_metadata
from tinymind.model.tokenizer import Tokenizer, hash_spec, tokenizer_from_spec

PACKAGE_KIND = "tinymind-inference-package"
PACKAGE_FORMAT_VERSION = 1
ARCHITECTURE_ID = "tinymind-transformer-v1"
_FILES = ("model.tm", "tokenizer.json")


class PackageError(Exception):
    """The package is incomplete, altered, or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def export_package(model: TinyMindTransformer, tokenizer: Tokenizer, out_dir: str | Path, *,
                   provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write ``model`` + ``tokenizer`` as an inference package at ``out_dir``
    (replacing any previous package there) and return its manifest."""
    if model.config.vocab_size != tokenizer.vocab_size:
        raise PackageError(f"model vocab_size {model.config.vocab_size} != tokenizer vocab_size {tokenizer.vocab_size}")
    out_dir = Path(out_dir)
    renderer = ChatRenderer(tokenizer)
    tok_spec = tokenizer.spec()
    prov = dict(provenance or {})
    prov.setdefault("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    tmp.mkdir()
    try:
        export_to_tm(model, tmp / "model.tm", extra_metadata={
            "package_format_version": PACKAGE_FORMAT_VERSION, "tokenizer": tok_spec, "renderer": renderer.spec(),
            "provenance": prov})
        _write_json(tmp / "tokenizer.json", tok_spec)
        cfg = model.config
        manifest = {
            "kind": PACKAGE_KIND, "format_version": PACKAGE_FORMAT_VERSION, "architecture": ARCHITECTURE_ID,
            "model_config": cfg.to_dict(), "model_config_hash": cfg.stable_hash(),
            "parameter_count": model.count_parameters(),
            "weights": {"file": "model.tm", "dtype": "float32", "layout": "docs/architecture/native-model-contract.md"},
            "tokenizer": {"file": "tokenizer.json", "type": tok_spec["type"], "spec_hash": hash_spec(tok_spec),
                          "vocab_size": tokenizer.vocab_size},
            "renderer": renderer.spec(), "provenance": prov,
            "requires": {"optimizer_state": False, "training_data": False, "training_framework": False, "network": False},
            "files": {name: {"sha256": _sha256(tmp / name), "size": (tmp / name).stat().st_size} for name in _FILES},
        }
        _write_json(tmp / "package.json", manifest)
        if out_dir.exists():
            old = out_dir.with_name(f".{out_dir.name}.old-{uuid.uuid4().hex[:6]}")
            os.rename(out_dir, old)
            os.rename(tmp, out_dir)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.rename(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


@dataclasses.dataclass
class PackageReport:
    path: Path
    ok: bool
    errors: list[str]
    manifest: dict[str, Any] | None = None

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise PackageError(f"{self.path}: " + "; ".join(self.errors))


def verify_package(path: str | Path) -> PackageReport:
    """Checksums of every file, manifest/`.tm`/tokenizer.json agreement, and
    that the weights load into the declared architecture."""
    path = Path(path)
    errors: list[str] = []
    mpath = path / "package.json"
    if not mpath.is_file():
        return PackageReport(path, False, ["package.json missing"])
    try:
        manifest = json.loads(mpath.read_text())
    except ValueError as exc:
        return PackageReport(path, False, [f"package.json unreadable: {exc}"])
    if manifest.get("kind") != PACKAGE_KIND or manifest.get("format_version") != PACKAGE_FORMAT_VERSION:
        return PackageReport(path, False, [f"unsupported package kind/version {manifest.get('kind')!r}/{manifest.get('format_version')!r}"], manifest)
    for name in _FILES:
        entry, f = manifest.get("files", {}).get(name), path / name
        if entry is None or not f.is_file():
            errors.append(f"{name} missing")
        elif f.stat().st_size != entry["size"] or _sha256(f) != entry["sha256"]:
            errors.append(f"{name} does not match its recorded size/SHA-256")
    if errors:
        return PackageReport(path, False, errors, manifest)
    try:
        spec = json.loads((path / "tokenizer.json").read_text())
        meta = read_tm_metadata(path / "model.tm")
        if hash_spec(spec) != manifest["tokenizer"]["spec_hash"]:
            errors.append("tokenizer.json does not match the manifest's tokenizer hash")
        if meta.get("tokenizer") != spec:
            errors.append("model.tm's embedded tokenizer spec differs from tokenizer.json")
        if meta.get("model_config") != manifest["model_config"]:
            errors.append("model.tm's model_config differs from the manifest's")
        # redundant manifest fields must agree with the files they summarise
        if manifest["tokenizer"].get("type") != spec.get("type") or manifest["tokenizer"].get("vocab_size") != spec.get("vocab_size"):
            errors.append("manifest tokenizer type/vocab_size disagree with tokenizer.json")
        if manifest.get("architecture") != ARCHITECTURE_ID or meta.get("architecture") != ARCHITECTURE_ID:
            errors.append(f"architecture is not {ARCHITECTURE_ID!r}")
        config = ModelConfig.from_dict(manifest["model_config"])
        if config.stable_hash() != manifest.get("model_config_hash") or config.count_parameters() != manifest.get("parameter_count"):
            errors.append("manifest model_config_hash / parameter_count disagree with its model_config")
        if manifest["renderer"].get("template") != TEMPLATE_ID:
            errors.append(f"unsupported prompt template {manifest['renderer'].get('template')!r}")
        if manifest["renderer"] != meta.get("renderer"):
            errors.append("model.tm's embedded renderer spec differs from the manifest's")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cannot read package contents: {type(exc).__name__}: {exc}")
    return PackageReport(path, not errors, errors, manifest)


class InferencePackage:
    """A loaded package: model + tokenizer + renderer, ready to generate."""

    def __init__(self, model: TinyMindTransformer, tokenizer: Tokenizer, manifest: dict[str, Any], path: Path) -> None:
        self.model, self.tokenizer, self.manifest, self.path = model, tokenizer, manifest, path
        self.renderer = ChatRenderer(tokenizer)

    def prompt_ids(self, messages: str | Sequence[dict[str, Any]], tools: Sequence[str] = ()) -> list[int]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.renderer.render_prompt(messages, tools)

    def generate_ids(self, messages: str | Sequence[dict[str, Any]], *, tools: Sequence[str] = (), max_new_tokens: int = 64,
                     temperature: float = 0.0, top_k: int | None = None, repetition_penalty: float = 1.0,
                     seed: int | None = None) -> tuple[list[int], str]:
        """``(new token ids, finish_reason)``; ``finish_reason`` is ``stop`` (EOS) or ``length``."""
        ids = self.prompt_ids(messages, tools)
        cfg = ModelGenerationConfig(max_new_tokens=max_new_tokens, do_sample=temperature > 0.0,
                                    temperature=max(temperature, 1e-6), top_k=top_k, repetition_penalty=repetition_penalty,
                                    eos_token_id=self.tokenizer.eos_token_id, seed=seed)
        out = generate_with_cache_ids(self.model, np.array([ids]), cfg)
        new = out[0][len(ids):].tolist()
        return new, ("stop" if new and new[-1] == self.tokenizer.eos_token_id else "length")

    def generate(self, messages: str | Sequence[dict[str, Any]], **kwargs: Any) -> str:
        new, _ = self.generate_ids(messages, **kwargs)
        return self.renderer.decode_completion(new)


def load_package(path: str | Path) -> InferencePackage:
    """Verify and load. Raises ``PackageError`` on any problem — never guesses
    a tokenizer or template."""
    path = Path(path)
    report = verify_package(path)
    report.raise_if_bad()
    manifest = report.manifest
    assert manifest is not None
    tokenizer = tokenizer_from_spec(json.loads((path / "tokenizer.json").read_text()))
    model = import_from_tm(path / "model.tm")
    if ModelConfig.from_dict(manifest["model_config"]).stable_hash() != manifest["model_config_hash"]:
        raise PackageError("manifest model_config_hash does not match its model_config")
    model.check_tokenizer_compatibility(tokenizer)
    return InferencePackage(model, tokenizer, manifest, path)
