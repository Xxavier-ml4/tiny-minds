"""Runtime configuration: the YAML/JSON shape from the engineering brief
section 41 —

.. code-block:: yaml

    runtime:
      max_tokens: 256
      temperature: 0.0
    reasoning:
      adaptive: true
      max_steps: 4
    tools:
      retrieval:
        enabled: true
        top_k: 5
    confidence:
      enabled: true
      execute_threshold: 0.90
      verify_threshold: 0.70
    memory:
      enabled: true

Every section is optional in the file — a missing section falls back to
its dataclass defaults, so a config file only needs to state what it's
overriding.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class RuntimeSection:
    max_tokens: int = 256
    temperature: float = 0.0


@dataclasses.dataclass
class ReasoningSection:
    adaptive: bool = True
    max_steps: int = 4


@dataclasses.dataclass
class ToolRetrievalSection:
    enabled: bool = True
    top_k: int = 5


@dataclasses.dataclass
class ToolsSection:
    retrieval: ToolRetrievalSection = dataclasses.field(default_factory=ToolRetrievalSection)


@dataclasses.dataclass
class ConfidenceSection:
    enabled: bool = True
    execute_threshold: float = 0.90
    verify_threshold: float = 0.70


@dataclasses.dataclass
class MemorySection:
    enabled: bool = True


@dataclasses.dataclass
class TinyMindConfig:
    runtime: RuntimeSection = dataclasses.field(default_factory=RuntimeSection)
    reasoning: ReasoningSection = dataclasses.field(default_factory=ReasoningSection)
    tools: ToolsSection = dataclasses.field(default_factory=ToolsSection)
    confidence: ConfidenceSection = dataclasses.field(default_factory=ConfidenceSection)
    memory: MemorySection = dataclasses.field(default_factory=MemorySection)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyMindConfig":
        return cls(
            runtime=RuntimeSection(**data.get("runtime", {})),
            reasoning=ReasoningSection(**data.get("reasoning", {})),
            tools=ToolsSection(retrieval=ToolRetrievalSection(**data.get("tools", {}).get("retrieval", {}))),
            confidence=ConfidenceSection(**data.get("confidence", {})),
            memory=MemorySection(**data.get("memory", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TinyMindConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    @classmethod
    def default(cls) -> "TinyMindConfig":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": dataclasses.asdict(self.runtime),
            "reasoning": dataclasses.asdict(self.reasoning),
            "tools": {"retrieval": dataclasses.asdict(self.tools.retrieval)},
            "confidence": dataclasses.asdict(self.confidence),
            "memory": dataclasses.asdict(self.memory),
        }

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
