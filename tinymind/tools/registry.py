"""``ToolRegistry``: register / unregister / get / list, per brief section 4.

A registered tool bundles the callable that actually runs, its derived
``ToolSchema``, and its ``ToolPermissions``. The registry is the single
place the runtime is allowed to look up "what can I call and how" — nothing
in ``tinymind`` calls a tool function it did not get from a registry lookup
(see ``tinymind/tools/executor.py``).
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, Iterator

from tinymind.tools.permissions import ToolPermissions
from tinymind.tools.schema import ToolSchema, schema_of


class ToolRegistryError(ValueError):
    pass


@dataclasses.dataclass
class RegisteredTool:
    schema: ToolSchema
    fn: Callable
    permissions: ToolPermissions

    @property
    def name(self) -> str:
        return self.schema.name


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, source: Callable | type | dict, *, fn: Callable | None = None,
                 permissions: ToolPermissions | None = None,
                 name: str | None = None) -> RegisteredTool:
        """Register a tool.

        ``source`` is a decorated function, a dataclass, a Pydantic model, or
        a raw JSON-Schema dict. When ``source`` is not itself callable
        (a dataclass, a model, or a dict), pass the function that should
        actually run via ``fn=``. ``name`` overrides the derived name, which
        is useful for two tools that would otherwise derive the same name
        from two different functions.
        """
        schema = schema_of(source)
        if name:
            schema = dataclasses.replace(schema, name=name)
        runner = fn if fn is not None else source
        if not callable(runner):
            raise ToolRegistryError(
                f"tool {schema.name!r} has no callable to run; pass fn=... "
                "when registering a dataclass, Pydantic model, or raw schema dict")
        if schema.name in self._tools:
            raise ToolRegistryError(
                f"a tool named {schema.name!r} is already registered; "
                f"unregister it first or pass name=... to disambiguate")
        entry = RegisteredTool(schema=schema, fn=runner,
                               permissions=permissions or ToolPermissions())
        self._tools[schema.name] = entry
        return entry

    def unregister(self, name: str) -> None:
        if name not in self._tools:
            raise ToolRegistryError(f"no tool named {name!r} is registered")
        del self._tools[name]

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolRegistryError(f"no tool named {name!r} is registered") from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def __iter__(self) -> Iterator[RegisteredTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def schemas(self) -> list[dict]:
        """The declared-tools JSON any model context needs — schema only,
        never the callable, so this is safe to hand to a prompt builder."""
        return [entry.schema.to_dict() for entry in self._tools.values()]

    @classmethod
    def from_iterable(cls, sources: Iterable) -> "ToolRegistry":
        registry = cls()
        for source in sources:
            registry.register(source)
        return registry
