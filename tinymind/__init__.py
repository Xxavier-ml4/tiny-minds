"""TinyMind: a mobile-first, tool-aware, structured-output runtime and
model stack for tiny local reasoning agents.

This is Phase 1 (see ``STATUS.md``): the architecture, the reference
runtime, and the tool/routing/confidence/grounding/memory subsystems are
real and tested. There is no trained TinyMind model in this delivery —
``Model`` below defaults to ``EchoBackend``, a deterministic non-neural
stand-in (see ``tinymind.model.backends.echo`` for exactly what it is and
isn't), so that the API surface described in the engineering brief section
43 is genuinely usable and testable today, honestly, rather than only
sketched.

    from tinymind import Model

    model = Model("models/tinymind-150m-q4.tm")
    response = model.generate("Explain photosynthesis in one sentence.")
    # -> with the default EchoBackend, `response` is an echo, not an
    #    explanation; see the module docstring above and STATUS.md.

    model = Model(tools=[my_tool])
    result = model.run("do something my_tool can help with")
"""
from __future__ import annotations

from typing import Any, Callable, Iterator

from tinymind.config import TinyMindConfig
from tinymind.model.backend import GenerationResult, ModelBackend
from tinymind.model.backends.echo import EchoBackend
from tinymind.runtime.constraints.json_schema import StructuredOutputResult
from tinymind.runtime.engine import Engine
from tinymind.runtime.session import SessionResult
from tinymind.tools.permissions import ToolPermissions
from tinymind.tools.registry import ToolRegistry
from tinymind.tools.schema import Constraint, schema_of, tool

__version__ = "0.1.0-dev"

__all__ = [
    "Model", "Constraint", "tool", "ToolPermissions", "TinyMindConfig",
    "ModelBackend", "GenerationResult", "SessionResult", "StructuredOutputResult",
]


class Model:
    """The top-level facade: one model, its tools, and one implicit
    conversation session. For multiple independent sessions against the
    same backend, use ``tinymind.runtime.Engine`` directly instead of this
    class — ``Model`` is deliberately the simple, single-session surface
    the brief's own API examples show.
    """

    def __init__(self, path: str | None = None, *, tools: list[Callable] | None = None,
                backend: ModelBackend | None = None,
                registry: ToolRegistry | None = None,
                config: TinyMindConfig | None = None) -> None:
        self._backend = backend or EchoBackend()
        if path is not None:
            self._backend.load(path)
        self._config = config or TinyMindConfig.default()

        if registry is not None and tools:
            raise ValueError("pass either registry= or tools=, not both")
        self._registry = registry if registry is not None else ToolRegistry()
        for source in tools or []:
            self._registry.register(source)

        self._engine = Engine(self._backend, self._registry)
        self._session = self._engine.create_session("default")

    @property
    def tools(self) -> ToolRegistry:
        return self._registry

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        result = self._session.chat(prompt, max_new_tokens=max_new_tokens or self._config.runtime.max_tokens)
        return result.text

    def run(self, text: str) -> SessionResult:
        """Route ``text`` and act on it: a direct answer for CHAT/REASON, or
        the routing outcome for TOOL_CALL/PLAN — see
        ``tinymind.runtime.session.Session.run`` for exactly what each mode
        does with the configured backend, honestly, including what it
        cannot do without a real trained model.
        """
        return self._session.run(text)

    def extract(self, text: str, schema: Any) -> StructuredOutputResult:
        """Extract structured data matching ``schema`` (a dataclass, a
        Pydantic model, or a raw JSON-Schema dict) from ``text``."""
        schema_dict = schema if isinstance(schema, dict) else schema_of(schema).parameters
        result = self._session.run(text, output_schema=schema_dict)
        return result.structured or StructuredOutputResult(valid=False, value=None, errors=[result.reason])

    def stream(self, prompt: str, max_new_tokens: int | None = None) -> Iterator[str]:
        yield from self._backend.stream(prompt, max_new_tokens=max_new_tokens or self._config.runtime.max_tokens)

    def reset(self) -> None:
        self._backend.reset()
        self._session.history.clear()
