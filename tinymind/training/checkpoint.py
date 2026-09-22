"""Training checkpoints: complete, atomic, verifiable.

A *training checkpoint* is everything needed to continue a run as if it had
never stopped. It is **not** the same thing as an *inference model* (weights
+ architecture + tokenizer, see ``tinymind/export``); a weights-only file is
never called a checkpoint here.

Layout (one directory per checkpoint, several per run)::

    <root>/
      step-00000500/
        model.npz        float32 weights, keyed by parameter name
        optimizer.npz    AdamW first/second moments, keyed "m.<name>" / "v.<name>"
        state.json       model config, tokenizer + renderer spec, dataset
                         identity, training config, optimizer hyper-parameters,
                         scheduler, progress (step/epoch/data cursor/token
                         counters), gradient-accumulation state, RNG state,
                         architecture id, environment
        manifest.json    written LAST: identities + SHA-256 and size of every
                         other file
      latest.json        {"checkpoint": "step-00000500", "manifest_sha256": ...}
      previous.json      the pointer that ``latest`` replaced
      .tmp-*             a checkpoint being written (never read, never trusted)

Write protocol (single writer per root): build everything inside ``.tmp-*``
with fsync after each file -> write the manifest last -> fsync the directory
-> ``rename`` to its final name (atomic on POSIX) -> replace ``previous.json``
then ``latest.json`` (each via temp file + ``os.replace``) -> only then delete
checkpoints beyond ``keep_last``. A process killed at any point leaves either
the old state or the new one; the pointers always name a complete checkpoint.
Tests inject a failure (and a real ``os._exit``) at every step.

Files are NumPy ``.npz`` (never pickle: ``allow_pickle=False``) and JSON.
The RNG state is stored as the JSON form of NumPy's ``PCG64`` state.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tinymind.model.config import ModelConfig, parameter_shapes

FORMAT_VERSION = 1
KIND = "tinymind-training-checkpoint"
ARCHITECTURE_ID = "tinymind-transformer-v1"
_FILES = ("model.npz", "optimizer.npz", "state.json")


class CheckpointError(Exception):
    """Base class for every checkpoint failure."""


class CheckpointCorruptError(CheckpointError):
    """The checkpoint is incomplete, altered or internally inconsistent."""


class ResumeMismatchError(CheckpointError):
    """The checkpoint is valid but does not match what this run is configured to do."""


# --------------------------------------------------------------------------
# small durable-IO helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:  # e.g. Windows: directories cannot be opened
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_durable(path: Path, obj: Any) -> None:
    _write_bytes(path, json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))


def _replace_json(path: Path, obj: Any) -> None:
    """Crash-safe replacement of a small JSON file (temp + fsync + os.replace)."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    _write_json_durable(tmp, obj)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _savez_durable(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def git_commit() -> str | None:
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict[str, Any]:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "git_commit": git_commit()}


def rng_state_to_json(rng: np.random.Generator) -> dict[str, Any]:
    return json.loads(json.dumps(rng.bit_generator.state))


def rng_from_json(state: dict[str, Any]) -> np.random.Generator:
    if state.get("bit_generator") != "PCG64":
        raise CheckpointCorruptError(f"unsupported RNG state {state.get('bit_generator')!r}")
    bit_generator = np.random.PCG64()
    bit_generator.state = state
    return np.random.Generator(bit_generator)


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------
# what a checkpoint holds
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Snapshot:
    """Everything ``save_checkpoint`` records. Built by the trainer at an
    optimizer-step boundary — so the gradient-accumulation buffer is empty by
    construction and ``accumulation.pending_micro_batches`` is always 0."""
    stage: str
    stage_complete: bool
    weights: dict[str, np.ndarray]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    progress: dict[str, Any]          # global_step, epoch, cursor, cumulative_steps, token counters, ...
    rng_state: dict[str, Any]
    model_config: dict[str, Any]
    tokenizer_spec: dict[str, Any]
    renderer_spec: dict[str, Any]
    dataset: dict[str, Any]           # {"dataset_hash", "identity", "validation_hash"}
    training_config: dict[str, Any]
    training_config_compat: dict[str, Any]
    parent: dict[str, Any] | None = None
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LoadedCheckpoint:
    path: Path
    manifest: dict[str, Any]
    state: dict[str, Any]
    weights: dict[str, np.ndarray]
    optimizer_arrays: dict[str, np.ndarray]

    @property
    def optimizer_state(self) -> dict[str, Any]:
        opt = self.state["optimizer"]
        return {"hyper": opt["hyper"], "lr": opt["lr"], "step_count": opt["step_count"],
                "names": opt["names"], "arrays": self.optimizer_arrays}

    @property
    def step(self) -> int:
        return int(self.state["progress"]["global_step"])


def checkpoint_name(step: int) -> str:
    return f"step-{step:08d}"


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------
def save_checkpoint(root: str | Path, snap: Snapshot, *, keep_last: int = 3,
                    fault_hook: Callable[[str], None] | None = None) -> Path:
    """Write ``snap`` as the new latest checkpoint under ``root``; returns its
    directory. ``fault_hook(point)`` is called at each stage of the protocol
    (tests use it to crash the writer)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    hook = fault_hook or (lambda point: None)
    step = int(snap.progress["global_step"])
    final = root / checkpoint_name(step)

    for stale in root.glob(".tmp-*"):  # leftovers of a writer that died; never valid, safe to drop
        shutil.rmtree(stale, ignore_errors=True)

    if final.exists():
        if verify_checkpoint(final).ok:
            _update_pointers(root, final)  # this exact step is already durably on disk
            return final
        os.rename(final, root / f".corrupt-{final.name}-{uuid.uuid4().hex[:6]}")

    tmp = root / f".tmp-{final.name}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    tmp.mkdir()
    try:
        _savez_durable(tmp / "model.npz", {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in snap.weights.items()})
        hook("after_model")
        _savez_durable(tmp / "optimizer.npz", snap.optimizer_state["arrays"])
        hook("after_optimizer")
        state = {
            "format_version": FORMAT_VERSION,
            "model_config": snap.model_config,
            "tokenizer": snap.tokenizer_spec,
            "renderer": snap.renderer_spec,
            "dataset": snap.dataset,
            "training_config": snap.training_config,
            "training_config_compat": snap.training_config_compat,
            "optimizer": {"hyper": snap.optimizer_state["hyper"], "lr": snap.optimizer_state["lr"],
                          "step_count": snap.optimizer_state["step_count"], "names": snap.optimizer_state["names"]},
            "scheduler": snap.scheduler_state,
            "progress": snap.progress,
            "accumulation": {"pending_micro_batches": 0,
                             "gradient_accumulation_steps": snap.training_config_compat.get("gradient_accumulation_steps"),
                             "note": "checkpoints are taken only at optimizer-step boundaries, where the "
                                     "gradient buffer is empty by construction"},
            "rng": snap.rng_state,
            "architecture": {"id": ARCHITECTURE_ID,
                             "parameter_count": int(sum(int(np.prod(v.shape)) for v in snap.weights.values())),
                             "parameter_layout_hash": _sha_json({k: list(v.shape) for k, v in snap.weights.items()})},
            "metrics": snap.metrics,
            "environment": environment_info(),
        }
        _write_json_durable(tmp / "state.json", state)
        hook("after_state")

        manifest = {
            "format_version": FORMAT_VERSION, "kind": KIND, "stage": snap.stage,
            "global_step": step, "epoch": int(snap.progress["epoch"]), "stage_complete": bool(snap.stage_complete),
            "model_config_hash": ModelConfig.from_dict(snap.model_config).stable_hash(),
            "tokenizer_hash": _sha_json(snap.tokenizer_spec),
            "renderer_hash": _sha_json(snap.renderer_spec),
            "dataset_hash": snap.dataset["dataset_hash"],
            "training_config_hash": _sha_json(snap.training_config_compat),
            "model_parameter_count": state["architecture"]["parameter_count"],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": state["environment"]["git_commit"],
            "parent": snap.parent,
            "files": {name: {"sha256": sha256_file(tmp / name), "size": (tmp / name).stat().st_size} for name in _FILES},
        }
        _write_json_durable(tmp / "manifest.json", manifest)
        hook("after_manifest")
        _fsync_dir(tmp)
        hook("before_rename")
        os.rename(tmp, final)
        _fsync_dir(root)
        hook("after_rename")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)  # best effort; a hard kill leaves it for the next save to sweep
        raise

    _update_pointers(root, final)
    hook("after_latest")
    _prune(root, keep_last)
    return final


def _pointer(final: Path) -> dict[str, Any]:
    return {"checkpoint": final.name, "manifest_sha256": sha256_file(final / "manifest.json"),
            "global_step": int(final.name.split("-")[1])}


def _update_pointers(root: Path, final: Path) -> None:
    latest = root / "latest.json"
    new = _pointer(final)
    if latest.exists():
        try:
            old = json.loads(latest.read_text())
            if old.get("checkpoint") != new["checkpoint"]:
                _replace_json(root / "previous.json", old)
        except (OSError, json.JSONDecodeError):
            pass
    _replace_json(latest, new)


def _prune(root: Path, keep_last: int) -> None:
    protected = set()
    for name in ("latest.json", "previous.json"):
        try:
            protected.add(json.loads((root / name).read_text())["checkpoint"])
        except (OSError, ValueError, KeyError):
            pass
    dirs = sorted(d for d in root.glob("step-*") if d.is_dir())
    for d in dirs[:-keep_last] if keep_last > 0 else []:
        if d.name not in protected:
            shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------
# verification and loading
# --------------------------------------------------------------------------
@dataclasses.dataclass
class VerificationReport:
    path: Path
    ok: bool
    errors: list[str]
    manifest: dict[str, Any] | None = None

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise CheckpointCorruptError(f"{self.path}: " + "; ".join(self.errors))


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def verify_checkpoint(path: str | Path) -> VerificationReport:
    """Full integrity check: manifest present and well-formed, every file's
    size and SHA-256, arrays loadable without pickle, every weight present with
    the shape the stored model config implies, float32 and finite, optimizer
    arrays matching, and state/manifest agreeing on step and identities."""
    path = Path(path)
    errors: list[str] = []
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return VerificationReport(path, False, ["manifest.json missing (incomplete checkpoint)"])
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        return VerificationReport(path, False, [f"manifest.json unreadable: {exc}"])
    if manifest.get("kind") != KIND or manifest.get("format_version") != FORMAT_VERSION:
        return VerificationReport(path, False, [f"unsupported kind/format_version "
                                                f"{manifest.get('kind')!r}/{manifest.get('format_version')!r}"], manifest)
    files = manifest.get("files", {})
    for name in _FILES:
        entry = files.get(name)
        f = path / name
        if entry is None:
            errors.append(f"{name} not listed in manifest")
        elif not f.is_file():
            errors.append(f"{name} missing")
        elif f.stat().st_size != entry["size"]:
            errors.append(f"{name} size {f.stat().st_size} != manifest {entry['size']}")
        elif sha256_file(f) != entry["sha256"]:
            errors.append(f"{name} SHA-256 mismatch")
    if errors:
        return VerificationReport(path, False, errors, manifest)

    try:
        state = json.loads((path / "state.json").read_text())
        weights = _npz_arrays(path / "model.npz")
        opt_arrays = _npz_arrays(path / "optimizer.npz")
        config = ModelConfig.from_dict(state["model_config"])
    except Exception as exc:  # noqa: BLE001 - any failure to parse means unusable
        return VerificationReport(path, False, [f"cannot parse contents: {type(exc).__name__}: {exc}"], manifest)

    expected = parameter_shapes(config)
    for name, shape in expected.items():
        arr = weights.get(name)
        if arr is None:
            errors.append(f"weight {name} missing")
        elif tuple(arr.shape) != tuple(shape):
            errors.append(f"weight {name} shape {tuple(arr.shape)} != {tuple(shape)} implied by model config")
        elif arr.dtype != np.float32:
            errors.append(f"weight {name} dtype {arr.dtype} != float32")
        elif not np.isfinite(arr).all():
            errors.append(f"weight {name} contains NaN/Inf")
    extra = sorted(set(weights) - set(expected))
    if extra:
        errors.append(f"unexpected weights {extra[:3]}")
    names = state["optimizer"]["names"]
    if names != list(expected):
        errors.append("optimizer parameter names differ from the model's parameter list")
    for name in names:
        for prefix in ("m", "v"):
            arr = opt_arrays.get(f"{prefix}.{name}")
            if arr is None or tuple(arr.shape) != tuple(expected.get(name, ())) or not np.isfinite(arr).all():
                errors.append(f"optimizer array {prefix}.{name} missing/mis-shaped/non-finite")
                break
    progress = state.get("progress", {})
    if int(progress.get("global_step", -1)) != manifest["global_step"]:
        errors.append("state.json global_step disagrees with manifest")
    if manifest["model_config_hash"] != config.stable_hash():
        errors.append("manifest model_config_hash disagrees with state.json")
    if manifest["tokenizer_hash"] != _sha_json(state["tokenizer"]):
        errors.append("manifest tokenizer_hash disagrees with state.json")
    if manifest["dataset_hash"] != state["dataset"]["dataset_hash"]:
        errors.append("manifest dataset_hash disagrees with state.json")
    if state.get("rng", {}).get("bit_generator") != "PCG64":
        errors.append("RNG state missing or unsupported")
    return VerificationReport(path, not errors, errors, manifest)


def load_checkpoint(path: str | Path, verify: bool = True) -> LoadedCheckpoint:
    """Load one checkpoint directory (verified unless ``verify=False``)."""
    path = Path(path)
    if verify:
        verify_checkpoint(path).raise_if_bad()
    manifest = json.loads((path / "manifest.json").read_text())
    return LoadedCheckpoint(path, manifest, json.loads((path / "state.json").read_text()),
                            _npz_arrays(path / "model.npz"), _npz_arrays(path / "optimizer.npz"))


def find_latest_valid(root: str | Path) -> tuple[Path | None, list[str]]:
    """Newest checkpoint under ``root`` that passes ``verify_checkpoint``,
    trying ``latest`` then ``previous`` and then every ``step-*`` newest
    first. Returns ``(path or None, notes)``; ``notes`` lists each candidate
    that was rejected and why, so falling back is never silent."""
    root = Path(root)
    notes: list[str] = []
    candidates: list[Path] = []
    for pointer in ("latest.json", "previous.json"):
        try:
            candidates.append(root / json.loads((root / pointer).read_text())["checkpoint"])
        except (OSError, ValueError, KeyError) as exc:
            if (root / pointer).exists():
                notes.append(f"{pointer} unreadable: {exc}")
    candidates += sorted((d for d in root.glob("step-*") if d.is_dir()), reverse=True)
    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        report = verify_checkpoint(cand)
        if report.ok:
            return cand, notes
        notes.append(f"{cand.name} rejected: " + "; ".join(report.errors))
    return None, notes


def resolve_checkpoint(path: str | Path) -> tuple[Path, list[str]]:
    """``path`` may be a checkpoint directory or a run's checkpoint root. A
    named checkpoint is verified and used as-is (no fallback); a root resolves
    to its newest valid checkpoint. Raises ``CheckpointCorruptError`` rather than
    returning nothing."""
    path = Path(path)
    if (path / "manifest.json").exists():
        verify_checkpoint(path).raise_if_bad()
        return path, []
    if not path.exists():
        raise CheckpointCorruptError(f"{path}: does not exist")
    found, notes = find_latest_valid(path)
    if found is None:
        raise CheckpointCorruptError(f"{path}: no valid checkpoint found" + (" (" + "; ".join(notes) + ")" if notes else ""))
    return found, notes


# --------------------------------------------------------------------------
# resume / promotion validation
# --------------------------------------------------------------------------
def _diff(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    return [f"{k}: checkpoint={stored.get(k)!r} now={current.get(k)!r}" for k in sorted(set(stored) | set(current))
            if stored.get(k) != current.get(k)]


def validate_resume(ckpt: LoadedCheckpoint, *, stage: str, model_config: dict[str, Any], tokenizer_spec: dict[str, Any],
                    renderer_spec: dict[str, Any], dataset_hash: str, dataset_identity: dict[str, Any],
                    training_compat: dict[str, Any]) -> None:
    """Exact-resume gate: raises ``ResumeMismatchError`` listing EVERY mismatch
    (architecture, tokenizer, renderer, dataset, training config, stage)."""
    problems: list[str] = []
    st = ckpt.state
    if ckpt.manifest.get("stage_complete"):
        problems.append("the checkpoint is a COMPLETED stage; nothing to resume (use --init-from to start the next stage)")
    if ckpt.manifest["stage"] != stage:
        problems.append(f"stage differs: checkpoint={ckpt.manifest['stage']!r}, this run={stage!r} "
                        "(--resume continues a stage; --init-from starts a new one)")
    if st["model_config"] != model_config:
        problems.append("model architecture differs: " + "; ".join(_diff(st["model_config"], model_config)))
    if st["tokenizer"] != tokenizer_spec:
        problems.append("tokenizer differs: " + "; ".join(_diff(st["tokenizer"], tokenizer_spec)))
    if st["renderer"] != renderer_spec:
        problems.append("prompt renderer/template differs: " + "; ".join(_diff(st["renderer"], renderer_spec)))
    if st["dataset"]["dataset_hash"] != dataset_hash:
        old, new = st["dataset"]["identity"], dataset_identity
        detail = _diff({"sources": [(s["name"], s["content_hash"][:12], s["weight"]) for s in old["sources"]],
                        "epoch_examples": old["epoch_examples"]},
                       {"sources": [(s["name"], s["content_hash"][:12], s["weight"]) for s in new["sources"]],
                        "epoch_examples": new["epoch_examples"]})
        problems.append("dataset identity differs: " + ("; ".join(detail) or "hash differs"))
    if st["training_config_compat"] != training_compat:
        problems.append("training configuration differs: " + "; ".join(_diff(st["training_config_compat"], training_compat)))
    if problems:
        raise ResumeMismatchError("cannot resume from " + str(ckpt.path) + ":\n  - " + "\n  - ".join(problems))


def validate_init_from(ckpt: LoadedCheckpoint, *, model_config: dict[str, Any], tokenizer_spec: dict[str, Any],
                       renderer_spec: dict[str, Any]) -> None:
    """Stage-promotion gate: the weights may seed a new stage only if the
    architecture, tokenizer and prompt template are identical."""
    problems = []
    st = ckpt.state
    if st["model_config"] != model_config:
        problems.append("model architecture differs: " + "; ".join(_diff(st["model_config"], model_config)))
    if st["tokenizer"] != tokenizer_spec:
        problems.append("tokenizer differs: " + "; ".join(_diff(st["tokenizer"], tokenizer_spec)))
    if st["renderer"] != renderer_spec:
        problems.append("prompt renderer/template differs: " + "; ".join(_diff(st["renderer"], renderer_spec)))
    if problems:
        raise ResumeMismatchError("cannot initialise from " + str(ckpt.path) + ":\n  - " + "\n  - ".join(problems))


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    """What ``tinymind checkpoint-info`` prints (verification included)."""
    report = verify_checkpoint(path)
    out: dict[str, Any] = {"path": str(path), "valid": report.ok, "errors": report.errors}
    if report.manifest:
        out["manifest"] = {k: v for k, v in report.manifest.items() if k != "files"}
        out["files"] = report.manifest.get("files")
    if report.ok:
        state = json.loads((Path(path) / "state.json").read_text())
        out["progress"] = state["progress"]
        out["scheduler"] = state["scheduler"]
        out["optimizer"] = {k: state["optimizer"][k] for k in ("hyper", "lr", "step_count")}
        out["model_config"] = state["model_config"]
        out["tokenizer"] = state["tokenizer"]
        out["dataset"] = {"dataset_hash": state["dataset"]["dataset_hash"]}
        out["metrics"] = state.get("metrics", {})
    return out
