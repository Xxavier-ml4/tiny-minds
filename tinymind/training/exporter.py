"""The exporter the training engine calls at the end of every run (stage
complete, time budget, stop request): writes the FP32 inference package."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tinymind.export.package import export_package


def package_exporter(engine: Any, out_dir: Path) -> dict[str, Any]:
    ckpt = engine.last_checkpoint
    prov = {"stage": engine.config.stage, "global_step": engine.step, "cumulative_steps": engine.cumulative_steps,
            "stage_complete": engine.step >= engine.total_steps,
            "dataset_hash": engine.plan.dataset_hash(), "training_config_hash": engine.config.compat_hash(engine.total_steps),
            "checkpoint": ckpt.name if ckpt else None,
            "final_validation": engine.val_history[-1] if engine.val_history else None}
    from tinymind.training.checkpoint import environment_info, sha256_file
    prov["git_commit"] = environment_info()["git_commit"]
    if ckpt:
        prov["checkpoint_manifest_sha256"] = sha256_file(ckpt / "manifest.json")
    pkg = Path(out_dir) / "export"
    manifest = export_package(engine.model, engine.tokenizer, pkg, provenance=prov)
    return {"package": str(pkg), "tm": str(pkg / "model.tm"), "parameter_count": manifest["parameter_count"],
            "fp32_bytes": manifest["files"]["model.tm"]["size"]}
