"""Post-training INT8 for an inference package: a *storage* format with a reference loader.

Scheme (``tinymind.quantization.Int8Scheme``): per-output-row symmetric, ``scale = max|w| / 127``, weights
rounded to int8, scales kept as float32; norm gains stay float32; the token-embedding table (also the tied output
head) stays float32 unless ``quantize_embeddings=True``. The int8 weights are written to ``model.int8.tm`` next to
``model.tm`` and listed, with SHA-256, in ``package.json``.

Honest scope: the Python reference runtime *dequantizes to float32 before computing*, so this path reduces file
size (measured by ``benchmarks/quantization_eval.py``) but not RAM or latency; and the native runtime cannot yet
read the int8 file (it loads only the float32 ``model.tm``). No claim of an int8 speed-up is made anywhere.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tinymind.export.package import InferencePackage, PackageError, _sha256, load_package, verify_package
from tinymind.model.tokenizer import tokenizer_from_spec
from tinymind.quantization.model_quantizer import (dequantize_to_model, export_quantized_tm, import_quantized_tm,
                                                    quantize_model)

INT8_FILE = "model.int8.tm"


def add_int8(package_dir: str | Path, *, quantize_embeddings: bool = False) -> dict[str, Any]:
    """Quantize the package's float32 model and add ``model.int8.tm``; returns the manifest entry."""
    d = Path(package_dir)
    pkg = load_package(d)  # verifies first
    q = quantize_model(pkg.model, "int8", quantize_embeddings=quantize_embeddings)
    tmp = d / f".{INT8_FILE}.tmp-{os.getpid()}"
    try:
        export_quantized_tm(q, tmp)
        os.replace(tmp, d / INT8_FILE)
    finally:
        if tmp.exists():
            tmp.unlink()
    manifest = json.loads((d / "package.json").read_text())
    entry = {"file": INT8_FILE, "scheme": "int8-per-row-symmetric", "quantize_embeddings": quantize_embeddings,
             "runtime": "reference: dequantized to float32 before use; not readable by the native runtime",
             "sha256": _sha256(d / INT8_FILE), "size": (d / INT8_FILE).stat().st_size}
    manifest["weights_int8"] = entry
    manifest["files"][INT8_FILE] = {"sha256": entry["sha256"], "size": entry["size"]}
    tmp_manifest = d / f".package.json.tmp-{os.getpid()}"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_manifest, d / "package.json")
    return entry


def load_int8_package(package_dir: str | Path) -> InferencePackage:
    """The int8 weights dequantized into a runnable model, with the package's own tokenizer and template."""
    d = Path(package_dir)
    report = verify_package(d)
    report.raise_if_bad()
    manifest = report.manifest
    assert manifest is not None
    entry = manifest.get("weights_int8")
    if not entry:
        raise PackageError(f"{d} has no int8 weights (run add_int8 first)")
    f = d / entry["file"]
    if not f.is_file() or f.stat().st_size != entry["size"] or _sha256(f) != entry["sha256"]:
        raise PackageError(f"{entry['file']} does not match its recorded size/SHA-256")
    model = dequantize_to_model(import_quantized_tm(f))
    tokenizer = tokenizer_from_spec(json.loads((d / "tokenizer.json").read_text()))
    model.check_tokenizer_compatibility(tokenizer)
    return InferencePackage(model, tokenizer, manifest, d)
