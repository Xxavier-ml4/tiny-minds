"""Validate training-data JSONL against the format the engineering brief
defines in section 22: one JSON object per line, with ``id``, ``messages``,
``tools``, and a ``target`` of type ``answer`` / ``tool_call`` /
``multi_tool`` / ``structured`` / ``clarification`` / ``refusal``.

This is real, complete validation — every example in
``datasets/seed/*.jsonl`` (once such a file exists; none ships in this
delivery, see ``STATUS.md``) can be checked against it today. It does not
depend on a trained model, a tokenizer, or anything else not yet built.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterator

from tinymind.tools.validation import validate as validate_json_schema

_VALID_TARGET_TYPES = ("answer", "tool_call", "multi_tool", "structured", "clarification", "refusal")
_VALID_ROLES = ("user", "assistant", "system", "tool")

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "parameters": {"type": "object"},
    },
    "required": ["name", "parameters"],
}

_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": list(_VALID_ROLES)},
        "content": {"type": "string"},
    },
    "required": ["role", "content"],
}


@dataclasses.dataclass
class ExampleError:
    line_number: int
    example_id: str | None
    message: str

    def __str__(self) -> str:
        prefix = f"line {self.line_number}"
        if self.example_id:
            prefix += f" (id={self.example_id!r})"
        return f"{prefix}: {self.message}"


@dataclasses.dataclass
class ValidationReport:
    total: int
    valid: int
    errors: list[ExampleError]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return f"{self.valid}/{self.total} valid examples, {len(self.errors)} error(s)"


def _validate_example(example: Any, line_number: int) -> list[ExampleError]:
    errors: list[ExampleError] = []
    example_id = example.get("id") if isinstance(example, dict) else None

    if not isinstance(example, dict):
        return [ExampleError(line_number, None, f"top-level value must be an object, got {type(example).__name__}")]

    if "id" not in example or not isinstance(example.get("id"), str):
        errors.append(ExampleError(line_number, example_id, "missing or non-string 'id'"))

    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append(ExampleError(line_number, example_id, "'messages' must be a non-empty array"))
    else:
        for i, message in enumerate(messages):
            result = validate_json_schema(message, _MESSAGE_SCHEMA)
            for err in result.errors:
                errors.append(ExampleError(line_number, example_id, f"messages[{i}].{err}"))

    tools = example.get("tools", [])
    if not isinstance(tools, list):
        errors.append(ExampleError(line_number, example_id, "'tools' must be an array"))
    else:
        for i, tool in enumerate(tools):
            result = validate_json_schema(tool, _TOOL_SCHEMA)
            for err in result.errors:
                errors.append(ExampleError(line_number, example_id, f"tools[{i}].{err}"))

    target = example.get("target")
    if not isinstance(target, dict):
        errors.append(ExampleError(line_number, example_id, "'target' must be an object"))
    else:
        target_type = target.get("type")
        if target_type not in _VALID_TARGET_TYPES:
            errors.append(ExampleError(
                line_number, example_id,
                f"target.type must be one of {_VALID_TARGET_TYPES}, got {target_type!r}"))
        elif target_type == "tool_call":
            if "name" not in target:
                errors.append(ExampleError(line_number, example_id, "target.type=tool_call requires 'name'"))
            if "arguments" not in target or not isinstance(target.get("arguments"), dict):
                errors.append(ExampleError(line_number, example_id,
                                           "target.type=tool_call requires an 'arguments' object"))
            tool_names = {t.get("name") for t in tools if isinstance(t, dict)}
            if target.get("name") not in tool_names:
                errors.append(ExampleError(
                    line_number, example_id,
                    f"target.name {target.get('name')!r} is not declared in this example's 'tools'"))
        elif target_type == "answer" and "content" not in target:
            errors.append(ExampleError(line_number, example_id, "target.type=answer requires 'content'"))

    return errors


def iter_examples(path: str | Path) -> Iterator[tuple[int, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, _MalformedLine(str(exc))


@dataclasses.dataclass
class _MalformedLine:
    detail: str


def validate_file(path: str | Path) -> ValidationReport:
    total = 0
    valid = 0
    errors: list[ExampleError] = []
    for line_number, example in iter_examples(path):
        total += 1
        if isinstance(example, _MalformedLine):
            errors.append(ExampleError(line_number, None, f"invalid JSON: {example.detail}"))
            continue
        line_errors = _validate_example(example, line_number)
        if line_errors:
            errors.extend(line_errors)
        else:
            valid += 1
    return ValidationReport(total=total, valid=valid, errors=errors)
