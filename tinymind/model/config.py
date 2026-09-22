"""Model sizing and architecture configuration.

A single ``ModelConfig`` covers every size in the 50M-1B family described in
the engineering brief. Nothing downstream of this module hard-codes a
dimension: a consumer asks for ``hidden_size`` etc. off the config object,
never a literal.

This mirrors what the Needle analysis flagged as a real strength of the
`.cact` format (docs/architecture/needle-analysis.md section 15): geometry
lives in one place and everything else reads it, so one code path serves the
whole size family.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

_VALID_NORM_TYPES = ("rmsnorm", "layernorm")
_VALID_MLP_TYPES = ("swiglu", "gelu_mlp")
_VALID_ATTENTION_TYPES = ("mha", "gqa", "mqa")
_VALID_DTYPES = ("float32", "float16", "bfloat16")


class ModelConfigError(ValueError):
    """A ``ModelConfig`` field failed validation."""


@dataclasses.dataclass
class ModelConfig:
    # Core sizing — the fields that actually change with model size.
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: int = 8
    """1 = MQA, < num_heads = GQA, == num_heads = plain MHA."""
    intermediate_size: int = 1376
    max_seq_len: int = 2048
    vocab_size: int = 32000

    # Architectural knobs — brief section 14 ("architectural experiments").
    # Each has a sane default; nothing in tinymind.model consumes these as
    # literals, always through this object, so swapping one is a config
    # change, not a code change.
    attention_type: str = "gqa"
    norm_type: str = "rmsnorm"
    mlp_type: str = "swiglu"
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    sliding_window: int = 0  # 0 = full attention

    # Added in Phase 3A for the real transformer implementation
    # (tinymind/model/model.py) — not used by anything in Phase 1, which is
    # why they weren't here yet. ``norm_type``/``mlp_type`` above already
    # cover "which formula"; these three cover "what numeric knobs that
    # formula needs."
    norm_epsilon: float = 1e-6
    dropout: float = 0.0
    dtype: str = "float32"
    """Kept as a config field per the brief's request even though the
    autograd engine (tinymind/model/tensor.py) only actually computes in
    float32 today — see that module's docstring on why lower precision
    isn't implemented in this delivery. Validated against a fixed set so a
    config file can't silently request a precision nothing reads."""

    # Bookkeeping
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.hidden_size % max(self.num_heads, 1) != 0:
            raise ModelConfigError(
                f"hidden_size ({self.hidden_size}) must be positive and divisible "
                f"by num_heads ({self.num_heads})")
        if self.num_layers <= 0:
            raise ModelConfigError(f"num_layers must be positive, got {self.num_layers}")
        if not (0 < self.num_kv_heads <= self.num_heads):
            raise ModelConfigError(
                f"num_kv_heads ({self.num_kv_heads}) must be in (0, num_heads="
                f"{self.num_heads}]")
        if self.num_heads % self.num_kv_heads != 0:
            raise ModelConfigError(
                f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads "
                f"({self.num_kv_heads})")
        if self.vocab_size <= 0:
            raise ModelConfigError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ModelConfigError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.norm_type not in _VALID_NORM_TYPES:
            raise ModelConfigError(
                f"norm_type must be one of {_VALID_NORM_TYPES}, got {self.norm_type!r}")
        if self.mlp_type not in _VALID_MLP_TYPES:
            raise ModelConfigError(
                f"mlp_type must be one of {_VALID_MLP_TYPES}, got {self.mlp_type!r}")
        if self.attention_type not in _VALID_ATTENTION_TYPES:
            raise ModelConfigError(
                f"attention_type must be one of {_VALID_ATTENTION_TYPES}, "
                f"got {self.attention_type!r}")
        if self.dtype not in _VALID_DTYPES:
            raise ModelConfigError(f"dtype must be one of {_VALID_DTYPES}, got {self.dtype!r}")
        if not (0.0 <= self.dropout < 1.0):
            raise ModelConfigError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.norm_epsilon <= 0:
            raise ModelConfigError(f"norm_epsilon must be positive, got {self.norm_epsilon}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def approx_param_count(self) -> int:
        """Kept for backward compatibility (Phase 1 presets and the CLI call
        it). In Phase 3A this was an estimate that omitted norm weights and,
        for ``tie_embeddings=False``, the LM head; it is now the exact count
        (``count_parameters``), so the name is historical only."""
        return count_parameters(self)

    def count_parameters(self) -> int:
        return count_parameters(self)

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return parameter_shapes(self)

    def unsupported_settings(self) -> list[str]:
        """Settings this implementation accepts in a config file but would
        silently ignore. Phase 3A built such models without complaint (audit
        experiment ``silent_config_knobs``: every one produced bit-identical
        logits to the default), so a config could describe an architecture
        that does not exist. ``TinyMindTransformer`` refuses to build from a
        config that lists any of these."""
        problems = []
        if self.norm_type != "rmsnorm":
            problems.append(f"norm_type={self.norm_type!r} is not implemented (only 'rmsnorm')")
        if self.mlp_type != "swiglu":
            problems.append(f"mlp_type={self.mlp_type!r} is not implemented (only 'swiglu')")
        if self.sliding_window != 0:
            problems.append(f"sliding_window={self.sliding_window} is not implemented (only 0 = full attention)")
        if self.dropout != 0.0:
            problems.append(f"dropout={self.dropout} is not implemented (only 0.0)")
        if self.dtype != "float32":
            problems.append(f"dtype={self.dtype!r} is not implemented (compute is float32 only)")
        h, kv = self.num_heads, self.num_kv_heads
        contradiction = {"mha": kv != h, "mqa": kv != 1}.get(self.attention_type, False)
        if contradiction:
            problems.append(f"attention_type={self.attention_type!r} contradicts num_heads={h}, "
                            f"num_kv_heads={kv} (mha needs kv==heads, mqa needs kv==1; 'gqa' is "
                            "accepted as the generic label)")
        return problems

    def require_supported(self) -> None:
        problems = self.unsupported_settings()
        if problems:
            raise ModelConfigError("this config asks for behaviour the model does not implement: "
                                   + "; ".join(problems))

    def stable_hash(self) -> str:
        """SHA-256 of the canonical JSON of the architecture (used in
        checkpoints and manifests to detect any architecture change)."""
        import hashlib
        import json
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ModelConfigError(f"unknown ModelConfig field(s): {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        # Named presets may nest under a `model:` key (matches the example
        # in the engineering brief section 41); accept either shape.
        if "model" in data and isinstance(data["model"], dict):
            data = data["model"]
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def parameter_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Every learnable tensor's name and shape, in the exact order
    ``TinyMindTransformer.named_parameters()`` yields them — derived from the
    config alone, without building the model. ``tests/model/test_config_count.py``
    asserts it matches an instantiated model for MHA/GQA/MQA, tied/untied and
    several sizes; checkpoint verification uses it to validate stored arrays
    without instantiating anything."""
    h, kv_dim, q_dim = config.hidden_size, config.num_kv_heads * config.head_dim, config.num_heads * config.head_dim
    i, v = config.intermediate_size, config.vocab_size
    shapes: dict[str, tuple[int, ...]] = {"embed_tokens": (v, h)}
    for layer in range(config.num_layers):
        p = f"block_{layer}."
        shapes[p + "attn_norm.weight"] = (h,)
        shapes[p + "attention.q_proj.weight"] = (q_dim, h)
        shapes[p + "attention.k_proj.weight"] = (kv_dim, h)
        shapes[p + "attention.v_proj.weight"] = (kv_dim, h)
        shapes[p + "attention.o_proj.weight"] = (h, q_dim)
        shapes[p + "mlp_norm.weight"] = (h,)
        shapes[p + "mlp.gate_proj.weight"] = (i, h)
        shapes[p + "mlp.up_proj.weight"] = (i, h)
        shapes[p + "mlp.down_proj.weight"] = (h, i)
    shapes["final_norm.weight"] = (h,)
    if not config.tie_embeddings:
        shapes["lm_head.weight"] = (v, h)
    return shapes


def count_parameters(config: ModelConfig) -> int:
    """Exact number of learnable parameters for ``config`` (norm weights and,
    when untied, the LM head included)."""
    total = 0
    for shape in parameter_shapes(config).values():
        n = 1
        for dim in shape:
            n *= dim
        total += n
    return total


def load_preset(name: str) -> ModelConfig:
    """Load one of the named size presets shipped in ``configs/``.

    ``name`` is a bare preset name (``"150m"``) or a path to any YAML file
    shaped like one.
    """
    candidate = Path(name)
    if candidate.is_file():
        return ModelConfig.from_yaml(candidate)
    here = Path(__file__).resolve().parents[2] / "configs" / f"{name}.yaml"
    if not here.is_file():
        available = sorted(p.stem for p in here.parent.glob("*.yaml")) if here.parent.is_dir() else []
        raise ModelConfigError(f"no preset {name!r}; available: {available}")
    return ModelConfig.from_yaml(here)
