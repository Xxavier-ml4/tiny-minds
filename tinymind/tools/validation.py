"""A standalone JSON-Schema-subset validator.

Deliberately dependency-free (no ``jsonschema`` package) so it works
identically for tool-argument validation here and for the structured-output
constraint checking in ``tinymind.runtime.constraints.json_schema``, which
imports ``validate()`` from this module rather than duplicating it — one
validator, two call sites, per the brief's own instruction not to build a
model-coupled and a tools-coupled version of the same logic separately.

Supports: ``type`` (including a list of types), ``enum``, ``const``,
``required``, ``properties``/``additionalProperties``, ``items``,
numeric bounds (``minimum``/``maximum``/``exclusiveMinimum``/
``exclusiveMaximum``/``multipleOf``), string bounds (``minLength``/
``maxLength``/``pattern``/``format`` for a small set of common formats),
and array bounds (``minItems``/``maxItems``/``uniqueItems``).

Not supported (raises ``SchemaError`` if present, rather than silently
ignoring): ``$ref``/``$defs`` resolution, ``oneOf``/``anyOf``/``allOf``,
``if``/``then``/``else``. These are real JSON Schema features TinyMind's
tool schemas do not currently need; adding them is a matter of extending
``_validate_node``, not a redesign.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any

from tinymind.tools.schema import SchemaError

_JSON_TYPE_CHECKERS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}

_FORMAT_PATTERNS = {
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    "date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "date-time": re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"),
    "uuid": re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                       r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
}

_UNSUPPORTED_KEYWORDS = ("$ref", "oneOf", "anyOf", "allOf", "if", "then", "else")


@dataclasses.dataclass
class ValidationError:
    path: str  # JSON-pointer-ish, e.g. "brightness" or "items[2].room"
    message: str

    def __str__(self) -> str:
        return f"{self.path or '<root>'}: {self.message}"


@dataclasses.dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError]

    def __bool__(self) -> bool:
        return self.valid

    def raise_if_invalid(self) -> None:
        if not self.valid:
            detail = "; ".join(str(e) for e in self.errors)
            raise SchemaValidationError(f"validation failed: {detail}", self.errors)


class SchemaValidationError(ValueError):
    def __init__(self, message: str, errors: list[ValidationError]):
        super().__init__(message)
        self.errors = errors


def validate(value: Any, schema: dict) -> ValidationResult:
    errors: list[ValidationError] = []
    _validate_node(value, schema, "", errors)
    return ValidationResult(valid=not errors, errors=errors)


def _check_unsupported(schema: dict) -> None:
    found = [kw for kw in _UNSUPPORTED_KEYWORDS if kw in schema]
    if found:
        raise SchemaError(
            f"schema uses unsupported keyword(s) {found}; this validator handles a "
            "JSON-Schema subset (see tinymind/tools/validation.py module docstring)")


def _validate_node(value: Any, schema: dict, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(schema, dict):
        return
    _check_unsupported(schema)

    if "const" in schema and value != schema["const"]:
        errors.append(ValidationError(path, f"must equal const {schema['const']!r}"))
        return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(ValidationError(path, f"must be one of {schema['enum']!r}, got {value!r}"))
        return

    schema_type = schema.get("type")
    if schema_type is not None:
        types_ok = schema_type if isinstance(schema_type, list) else [schema_type]
        if not any(_JSON_TYPE_CHECKERS.get(t, lambda v: True)(value) for t in types_ok):
            errors.append(ValidationError(path, f"expected type {schema_type!r}, got "
                                          f"{type(value).__name__}"))
            return

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_number(value, schema, path, errors)
    elif isinstance(value, str):
        _validate_string(value, schema, path, errors)
    elif isinstance(value, list):
        _validate_array(value, schema, path, errors)
    elif isinstance(value, dict):
        _validate_object(value, schema, path, errors)


def _validate_number(value, schema, path, errors) -> None:
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(ValidationError(path, f"{value} < minimum {schema['minimum']}"))
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(ValidationError(path, f"{value} > maximum {schema['maximum']}"))
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        errors.append(ValidationError(path, f"{value} <= exclusiveMinimum {schema['exclusiveMinimum']}"))
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        errors.append(ValidationError(path, f"{value} >= exclusiveMaximum {schema['exclusiveMaximum']}"))
    if "multipleOf" in schema:
        multiple_of = schema["multipleOf"]
        if multiple_of and abs((value / multiple_of) - round(value / multiple_of)) > 1e-9:
            errors.append(ValidationError(path, f"{value} is not a multiple of {multiple_of}"))


def _validate_string(value: str, schema, path, errors) -> None:
    if "minLength" in schema and len(value) < schema["minLength"]:
        errors.append(ValidationError(path, f"length {len(value)} < minLength {schema['minLength']}"))
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(ValidationError(path, f"length {len(value)} > maxLength {schema['maxLength']}"))
    if "pattern" in schema and not re.search(schema["pattern"], value):
        errors.append(ValidationError(path, f"does not match pattern {schema['pattern']!r}"))
    fmt = schema.get("format")
    if fmt in _FORMAT_PATTERNS and not _FORMAT_PATTERNS[fmt].match(value):
        errors.append(ValidationError(path, f"does not match format {fmt!r}"))


def _validate_array(value: list, schema, path, errors) -> None:
    if "minItems" in schema and len(value) < schema["minItems"]:
        errors.append(ValidationError(path, f"has {len(value)} items < minItems {schema['minItems']}"))
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        errors.append(ValidationError(path, f"has {len(value)} items > maxItems {schema['maxItems']}"))
    if schema.get("uniqueItems"):
        seen = []
        for item in value:
            if item in seen:
                errors.append(ValidationError(path, f"items must be unique; {item!r} repeats"))
                break
            seen.append(item)
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            _validate_node(item, item_schema, f"{path}[{index}]" if path else f"[{index}]", errors)


def _validate_object(value: dict, schema, path, errors) -> None:
    required = schema.get("required", [])
    for key in required:
        if key not in value:
            errors.append(ValidationError(path, f"missing required property {key!r}"))
    properties = schema.get("properties", {})
    for key, item in value.items():
        if key in properties:
            child_path = f"{path}.{key}" if path else key
            _validate_node(item, properties[key], child_path, errors)
        elif schema.get("additionalProperties") is False:
            errors.append(ValidationError(path, f"additional property {key!r} is not allowed"))
