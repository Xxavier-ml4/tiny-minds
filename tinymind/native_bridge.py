"""Compile the native C++ runtime for equivalence checks and reporting.

Factored out of `tests/model/test_native_equivalence.py` so the trained-model
report (`tinymind/ci/trained_model_report.py`) can build and run the same
binary that test suite uses, instead of re-implementing it. Skips cleanly
(`HAS_CXX = False`) wherever no C++ compiler is available — nothing here
requires one to exist.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

NATIVE_DIR = Path(__file__).resolve().parents[1] / "native"
HAS_CXX = which("g++") is not None


class NativeBuildError(RuntimeError):
    """`g++` ran and failed. Distinct from `HAS_CXX = False` (no compiler at
    all) so callers can tell "not supported here" from "supported but broken"."""


def compile_sources(build_dir: Path, sources: list[Path], out_name: str) -> Path:
    binary = build_dir / out_name
    cmd = ["g++", "-std=c++17", "-O2", "-I", str(NATIVE_DIR / "include"), "-I", str(NATIVE_DIR / "src"),
          *[str(s) for s in sources], "-o", str(binary)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise NativeBuildError(f"g++ failed:\n{result.stderr}")
    return binary


def build_equivalence_binary(build_dir: Path) -> Path:
    """The `equiv` binary: `equiv <model.tm> <logits_out.bin> <id0> <id1> ...`
    — loads a `.tm` file, runs one forward pass, writes float32 logits."""
    return compile_sources(build_dir, [
        NATIVE_DIR / "tests" / "test_model_equivalence.cpp",
        NATIVE_DIR / "src" / "model.cpp",
        NATIVE_DIR / "src" / "tensor.cpp",
    ], "equiv")


def compare_logits(binary: Path, tm_path: Path, python_logits, input_ids: list[int], work_dir: Path) -> dict:
    """Run the native binary on `tm_path` and compare against `python_logits`
    (a NumPy array, same shape the native binary is expected to produce).
    Returns a plain dict, never raises for a numeric mismatch (the caller
    decides what to do with the numbers); raises only if the binary itself
    fails to run."""
    import numpy as np

    logits_path = work_dir / "logits.bin"
    result = subprocess.run([str(binary), str(tm_path), str(logits_path), *[str(i) for i in input_ids]],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise NativeBuildError(f"native binary failed:\n{result.stderr}")
    native_logits = np.fromfile(logits_path, dtype=np.float32).reshape(python_logits.shape)
    max_abs_diff = float(np.max(np.abs(python_logits - native_logits)))
    mismatches = int((np.argmax(python_logits, axis=-1) != np.argmax(native_logits, axis=-1)).sum())
    return {"max_abs_logit_diff": max_abs_diff, "greedy_token_mismatches": mismatches,
           "positions_compared": int(python_logits.shape[0])}
