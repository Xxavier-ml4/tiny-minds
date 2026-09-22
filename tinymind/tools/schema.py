"""Turn a Python callable, dataclass, or Pydantic model into a ``ToolSchema``.

Three sources are supported, per the engineering brief section 4:

- a plain function, via its signature, type hints, and a Google-style
  ``Args:`` docstring block
- a ``@dataclasses.dataclass``, via its fields
- a Pydantic ``BaseModel``, via ``model_json_schema()`` — only if pydantic
  is installed; TinyMind does not require it (same soft-dependency choice
  needle-analysis.md noted as worth keeping, reimplemented independently
  here)

A raw JSON Schema dict is always accepted as-is by ``ToolRegistry.register``
without going through this module at all.
"""
from __future__ import annotations

import dataclasses
import enum
import inspect
import re
import types
import typing
from typing import Any, Callable

_JSON_TYPE_MAP = {str: "string", int: "integer", float: "number",
                  bool: "boolean", list: "array", dict: "object"}
_UNION_ORIGINS = (typing.Union, getattr(types, "UnionType", typing.Union))
_MISSING = object()


class SchemaError(ValueError):
    """A schema could not be derived from the given source."""


@dataclasses.dataclass
class Constraint:
    """Per-argument constraints, attached via ``typing.Annotated`` or as a
    function parameter's default value.

    Mirrors the shape of JSON Schema's own constraint keywords directly —
    each non-None field here maps to exactly one JSON Schema keyword in
    ``apply()`` — rather than inventing a parallel vocabulary.
    """
    default: Any = _MISSING
    description: str | None = None
    enum: list | None = None
    const: Any = _MISSING
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    exclusive_maximum: float | None = None
    multiple_of: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    format: str | None = None
    min_items: int | None = None
    max_items: int | None = None
    unique_items: bool | None = None

    def has_default(self) -> bool:
        return self.default is not _MISSING

    def apply(self, schema: dict) -> dict:
        mapping = (
            ("description", self.description), ("enum", self.enum),
            ("minimum", self.minimum), ("maximum", self.maximum),
            ("exclusiveMinimum", self.exclusive_minimum),
            ("exclusiveMaximum", self.exclusive_maximum),
            ("multipleOf", self.multiple_of), ("minLength", self.min_length),
            ("maxLength", self.max_length), ("pattern", self.pattern),
            ("format", self.format), ("minItems", self.min_items),
            ("maxItems", self.max_items), ("uniqueItems", self.unique_items),
        )
        for key, value in mapping:
            if value is not None:
                schema[key] = list(value) if key == "enum" else value
        if self.const is not _MISSING:
            schema["const"] = self.const
        return schema


@dataclasses.dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict  # a JSON Schema object: {"type": "object", "properties": {...}, "required": [...]}

    def to_dict(self) -> dict:
        out = {"name": self.name, "parameters": self.parameters}
        if self.description:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ToolSchema":
        if "name" not in data:
            raise SchemaError("tool schema dict is missing required key 'name'")
        params = data.get("parameters", {"type": "object", "properties": {}})
        return cls(name=data["name"], description=data.get("description", ""),
                   parameters=params)


def _is_optional(annotation) -> bool:
    return (typing.get_origin(annotation) in _UNION_ORIGINS
            and type(None) in typing.get_args(annotation))


def _is_pydantic_model(obj) -> bool:
    return isinstance(obj, type) and any(
        base.__module__.startswith("pydantic") and base.__name__ == "BaseModel"
        for base in obj.__mro__)


def _annotation_to_json_type(annotation) -> dict:
    if annotation is inspect.Parameter.empty or annotation is None:
        return {"type": "string"}
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _annotation_to_json_type(typing.get_args(annotation)[0])
    if origin is None:
        if annotation in _JSON_TYPE_MAP:
            return {"type": _JSON_TYPE_MAP[annotation]}
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            values = [member.value for member in annotation]
            value_type = _JSON_TYPE_MAP.get(type(values[0]), "string") if values else "string"
            return {"type": value_type, "enum": values}
        if _is_pydantic_model(annotation):
            return schema_from_pydantic(annotation).parameters
        if dataclasses.is_dataclass(annotation):
            return schema_from_dataclass(annotation).parameters
        return {"type": "string"}
    if origin is typing.Literal:
        args = list(typing.get_args(annotation))
        value_type = _JSON_TYPE_MAP.get(type(args[0]), "string") if args else "string"
        return {"type": value_type, "enum": args}
    if origin in (list, typing.List):
        args = typing.get_args(annotation)
        item_schema = _annotation_to_json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    if origin in _UNION_ORIGINS:
        rest = [a for a in typing.get_args(annotation) if a is not type(None)]
        if rest:
            return _annotation_to_json_type(rest[0])
    return {"type": "string"}


def _constraint_from_annotation(annotation, default) -> Constraint | None:
    if isinstance(default, Constraint):
        return default
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        for meta in typing.get_args(annotation)[1:]:
            if isinstance(meta, Constraint):
                return meta
    if origin in _UNION_ORIGINS:
        rest = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(rest) == 1:
            return _constraint_from_annotation(rest[0], _MISSING)
    return None


_ARGS_HEADS = ("args:", "arguments:", "parameters:", "params:")


def _parse_google_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    if not doc:
        return "", {}
    lines = doc.strip("\n").splitlines()
    summary_lines: list[str] = []
    i = 0
    while i < len(lines) and lines[i].strip().lower() not in _ARGS_HEADS:
        summary_lines.append(lines[i].strip())
        i += 1
    arg_docs: dict[str, str] = {}
    for line in lines[i + 1:]:
        match = re.match(r"\s+(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", line)
        if match:
            arg_docs[match.group(1)] = match.group(2).strip()
    summary = " ".join(part for part in summary_lines if part).strip()
    return summary, arg_docs


def schema_from_function(fn: Callable) -> ToolSchema:
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    description, arg_docs = _parse_google_docstring(fn.__doc__)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name in ("self", "cls") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        prop_schema = _annotation_to_json_type(annotation)
        if name in arg_docs and "description" not in prop_schema:
            prop_schema["description"] = arg_docs[name]
        constraint = _constraint_from_annotation(annotation, param.default)
        if constraint:
            constraint.apply(prop_schema)
        properties[name] = prop_schema
        has_default = param.default is not param.empty and not isinstance(param.default, Constraint)
        if constraint and constraint.has_default():
            has_default = True
        if not has_default and not _is_optional(annotation):
            required.append(name)
    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return ToolSchema(name=fn.__name__, description=description, parameters=parameters)


def schema_from_dataclass(cls: type) -> ToolSchema:
    if not dataclasses.is_dataclass(cls):
        raise SchemaError(f"{cls!r} is not a dataclass")
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:
        hints = {f.name: f.type for f in dataclasses.fields(cls)}
    properties: dict[str, dict] = {}
    required: list[str] = []
    for field in dataclasses.fields(cls):
        annotation = hints.get(field.name, field.type)
        prop_schema = _annotation_to_json_type(annotation)
        constraint = _constraint_from_annotation(annotation, field.default)
        if constraint:
            constraint.apply(prop_schema)
        properties[field.name] = prop_schema
        has_default = (field.default is not dataclasses.MISSING
                       or field.default_factory is not dataclasses.MISSING)  # type: ignore[misc]
        if not has_default and not _is_optional(annotation):
            required.append(field.name)
    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    doc = (cls.__doc__ or "").strip()
    # dataclass auto-docstrings look like "ClassName(field: type = ...)" —
    # not useful as a tool description, so only keep hand-written docstrings.
    if doc.startswith(f"{cls.__name__}("):
        doc = ""
    return ToolSchema(name=cls.__name__, description=doc, parameters=parameters)


def schema_from_pydantic(model: type) -> ToolSchema:
    if not _is_pydantic_model(model):
        raise SchemaError(f"{model!r} is not a Pydantic BaseModel")
    raw = model.model_json_schema() if hasattr(model, "model_json_schema") else model.schema()
    parameters = {"type": "object", "properties": raw.get("properties", {})}
    for key in ("required", "$defs", "definitions"):
        if key in raw:
            parameters[key] = raw[key]
    description = (model.__doc__ or "").strip()
    return ToolSchema(name=raw.get("title", model.__name__), description=description,
                      parameters=parameters)


def schema_of(source: Callable | type | dict) -> ToolSchema:
    """Dispatch to the right ``schema_from_*`` for whatever ``source`` is."""
    if isinstance(source, dict):
        return ToolSchema.from_dict(source)
    if isinstance(source, type):
        if _is_pydantic_model(source):
            return schema_from_pydantic(source)
        if dataclasses.is_dataclass(source):
            return schema_from_dataclass(source)
        raise SchemaError(
            f"{source!r} is a class but neither a dataclass nor a Pydantic BaseModel")
    if callable(source):
        cached = getattr(source, "_tinymind_schema", None)
        return cached if cached is not None else schema_from_function(source)
    raise SchemaError(f"cannot derive a tool schema from {source!r}")


def tool(fn: Callable) -> Callable:
    """Decorator: attach a derived ``ToolSchema`` to ``fn`` as
    ``fn._tinymind_schema``, and return ``fn`` unchanged otherwise."""
    fn._tinymind_schema = schema_from_function(fn)  # type: ignore[attr-defined]
    return fn
