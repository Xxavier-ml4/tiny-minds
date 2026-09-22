"""Canonical training-example format and the one renderer that turns it into
tokens — used by training, evaluation, inference and benchmark generation.

**Why this exists.** Phase 3A's ``TrainingDataset`` joined the user messages,
encoded them, and threw the target away (audit experiment ``data_pipeline``),
so nothing ever taught prompt -> response. Here the sequence the model is
trained on is defined in exactly one place, with an explicit per-token
loss mask, and ``render_prompt`` is by construction a *prefix* of
``render``'s output (tested), so a model is prompted the way it was trained.

Example formats accepted (one JSON object per line):

``text``       ``{"id": "...", "text": "..."}`` — plain continuation.
``chat``       ``{"id": "...", "messages": [{"role": "user"|"assistant"|"system"|"tool",
               "content": "..."}, ...], "tools": [<names or {"name":..}>], "category": "..."}``
``legacy``     the Phase 3A/brief-section-22 shape: ``messages`` (no trailing
               assistant) plus ``target`` = ``{"type": answer|tool_call|multi_tool|
               structured|clarification|refusal, ...}``; the target becomes the
               final assistant message.

Rendered sequence (template ``tinymind-chat-v1``; ``[X]`` = special token)::

    [BOS]
    tools: calculator, lookup\\n          <- only if tools are declared      (no loss)
    system:\\n<content>\\n                 <- only if present                 (no loss)
    user:\\n<prompt>\\n                                                       (no loss)
    assistant:\\n                                                             (no loss)
    <response>[EOS]                                                          (LOSS)
    tool:\\n<tool result>\\n                                                  (no loss)
    assistant:\\n<final answer>[EOS]                                          (LOSS)

so the model learns the response *and* when to stop, and never learns to
generate the user's side. ``text`` examples are ``[BOS]<text>[EOS]`` with loss
everywhere except that BOS is never a target. Tool calls are ordinary
assistant text: ``{"name":"calculator","arguments":{"expr":"12*7"}}`` — the
model's output is untrusted text until the runtime parses and validates it.

Labels use the Hugging Face convention the model already implements:
``labels`` has the same length as ``input_ids``, is shifted inside the loss,
and ``-100`` means "no loss for this token". ``labels[0]`` is always -100.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Sequence

import numpy as np

from tinymind.model.tokenizer import Tokenizer

TEMPLATE_ID = "tinymind-chat-v1"
IGNORE_INDEX = -100
ROLES = ("system", "user", "assistant", "tool")


class ExampleError(ValueError):
    """A training example that cannot be rendered. Always names the example."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tool_call_text(name: str, arguments: dict[str, Any]) -> str:
    """The exact text a tool call has in an assistant message: fixed key
    order (name first, so the model commits to a tool before its arguments),
    sorted argument keys, no whitespace."""
    return '{"name":' + json.dumps(name, ensure_ascii=False) + ',"arguments":' + canonical_json(arguments) + "}"


def _target_to_text(target: dict[str, Any], example_id: str) -> str:
    kind = target.get("type")
    if kind in ("answer", "clarification", "refusal"):
        content = target.get("content")
        if not isinstance(content, str):
            raise ExampleError(f"{example_id}: target.type={kind} needs a string 'content'")
        return content
    if kind == "structured":
        content = target.get("content")
        return content if isinstance(content, str) else canonical_json(content)
    if kind == "tool_call":
        if not isinstance(target.get("name"), str) or not isinstance(target.get("arguments"), dict):
            raise ExampleError(f"{example_id}: tool_call target needs 'name' and an 'arguments' object")
        return tool_call_text(target["name"], target["arguments"])
    if kind == "multi_tool":
        calls = target.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ExampleError(f"{example_id}: multi_tool target needs a non-empty 'calls' list")
        return "[" + ",".join(tool_call_text(c["name"], c["arguments"]) for c in calls) + "]"
    raise ExampleError(f"{example_id}: unknown target.type {kind!r}")


@dataclasses.dataclass(frozen=True)
class NormalizedExample:
    id: str
    kind: str  # "text" | "chat"
    category: str
    text: str | None
    messages: tuple[dict[str, Any], ...]  # role, content, train
    tools: tuple[str, ...]


def normalize_example(raw: Any) -> NormalizedExample:
    """Validate a raw JSON example and reduce every accepted shape to one."""
    if not isinstance(raw, dict):
        raise ExampleError(f"example must be a JSON object, got {type(raw).__name__}")
    example_id = raw.get("id")
    if not isinstance(example_id, str) or not example_id:
        raise ExampleError("example needs a non-empty string 'id'")
    category = raw.get("category", "")
    if not isinstance(category, str):
        raise ExampleError(f"{example_id}: 'category' must be a string")
    if "text" in raw and "messages" in raw:
        raise ExampleError(f"{example_id}: has both 'text' and 'messages'")

    if "text" in raw:
        text = raw["text"]
        if not isinstance(text, str) or not text.strip():
            raise ExampleError(f"{example_id}: 'text' must be a non-empty string")
        return NormalizedExample(example_id, "text", category, text, (), ())

    messages_raw = raw.get("messages")
    if not isinstance(messages_raw, list) or not messages_raw:
        raise ExampleError(f"{example_id}: needs 'text' or a non-empty 'messages' list")
    messages: list[dict[str, Any]] = []
    for i, message in enumerate(messages_raw):
        if not isinstance(message, dict) or message.get("role") not in ROLES \
                or not isinstance(message.get("content"), str):
            raise ExampleError(f"{example_id}: messages[{i}] needs role in {ROLES} and string content")
        role = message["role"]
        train = message.get("train", role == "assistant")
        if not isinstance(train, bool) or (train and role != "assistant"):
            raise ExampleError(f"{example_id}: messages[{i}] 'train' must be a bool and only assistant turns can be trained")
        messages.append({"role": role, "content": message["content"], "train": train})

    tools_raw = raw.get("tools", [])
    if not isinstance(tools_raw, list):
        raise ExampleError(f"{example_id}: 'tools' must be a list")
    tools = tuple(t if isinstance(t, str) else t.get("name") for t in tools_raw if isinstance(t, (str, dict)))
    if any(not isinstance(t, str) or not t for t in tools) or len(tools) != len(tools_raw):
        raise ExampleError(f"{example_id}: every tool needs a name")

    if "target" in raw:  # legacy shape
        if messages[-1]["role"] == "assistant":
            raise ExampleError(f"{example_id}: has a 'target' but 'messages' already ends with an assistant turn")
        target = raw["target"]
        if not isinstance(target, dict):
            raise ExampleError(f"{example_id}: 'target' must be an object")
        if target.get("type") == "tool_call" and tools and target.get("name") not in tools:
            raise ExampleError(f"{example_id}: target calls {target.get('name')!r}, not among declared tools {list(tools)}")
        messages.append({"role": "assistant", "content": _target_to_text(target, example_id), "train": True})
        category = category or str(target.get("type", ""))

    trained = [m for m in messages if m["train"]]
    if not trained:
        raise ExampleError(f"{example_id}: no trainable assistant message — nothing to learn from")
    for m in trained:
        if not m["content"].strip():
            raise ExampleError(f"{example_id}: a trainable assistant message is empty "
                               "(Phase 3A demos used empty targets; that trains nothing)")
    return NormalizedExample(example_id, "chat", category, None, tuple(messages), tools)


@dataclasses.dataclass(frozen=True)
class Segment:
    """A run of tokens that is either all-loss or all-no-loss."""
    text: str          # tokenized normally ('' for a special token)
    special: str | None  # "bos" | "eos" | None
    trainable: bool


@dataclasses.dataclass(frozen=True)
class RenderedExample:
    example_id: str
    category: str
    kind: str
    ids: np.ndarray     # int32 [T]
    labels: np.ndarray  # int32 [T], -100 where no loss

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    @property
    def num_loss_tokens(self) -> int:
        """Targets that actually receive loss (position 0 is never a target)."""
        return int((self.labels[1:] != IGNORE_INDEX).sum())


class ChatRenderer:
    """Segments -> ids + labels. Stateless apart from the tokenizer it wraps."""

    def __init__(self, tokenizer: Tokenizer, template: str = TEMPLATE_ID) -> None:
        if template != TEMPLATE_ID:
            raise ValueError(f"unknown template {template!r}; this build implements {TEMPLATE_ID!r}")
        self.tokenizer = tokenizer
        self.template = template

    def spec(self) -> dict[str, Any]:
        return {"template": self.template, "tokenizer_spec_hash": self.tokenizer.spec_hash()}

    # ---- segmentation -------------------------------------------------
    def _prefix_segments(self, ex_tools: Sequence[str], messages: Sequence[dict[str, Any]]) -> list[Segment]:
        segs = [Segment("", "bos", False)]
        if ex_tools:
            segs.append(Segment("tools: " + ", ".join(ex_tools) + "\n", None, False))
        for m in messages:
            if m["role"] == "system":
                segs.append(Segment("system:\n" + m["content"] + "\n", None, False))
        return segs

    def segments(self, ex: NormalizedExample) -> list[Segment]:
        if ex.kind == "text":
            return [Segment("", "bos", False), Segment(ex.text or "", None, True), Segment("", "eos", True)]
        segs = self._prefix_segments(ex.tools, ex.messages)
        for m in ex.messages:
            role = m["role"]
            if role == "system":
                continue
            if role == "assistant":
                segs.append(Segment("assistant:\n", None, False))
                segs.append(Segment(m["content"], None, m["train"]))
                segs.append(Segment("", "eos", m["train"]))
            else:
                segs.append(Segment(f"{role}:\n{m['content']}\n", None, False))
        return segs

    def _tokenize(self, segs: Sequence[Segment]) -> tuple[np.ndarray, np.ndarray]:
        ids: list[int] = []
        train: list[bool] = []
        for seg in segs:
            if seg.special == "bos":
                piece = [self.tokenizer.bos_token_id]
            elif seg.special == "eos":
                piece = [self.tokenizer.eos_token_id]
            else:
                piece = self.tokenizer.encode(seg.text)
            ids.extend(piece)
            train.extend([seg.trainable] * len(piece))
        ids_arr = np.asarray(ids, dtype=np.int32)
        labels = np.where(np.asarray(train, dtype=bool), ids_arr, IGNORE_INDEX).astype(np.int32)
        labels[0] = IGNORE_INDEX  # the first token is only ever an input
        return ids_arr, labels

    # ---- public API ---------------------------------------------------
    def render(self, raw: Any) -> RenderedExample:
        ex = raw if isinstance(raw, NormalizedExample) else normalize_example(raw)
        ids, labels = self._tokenize(self.segments(ex))
        if not (labels[1:] != IGNORE_INDEX).any():
            raise ExampleError(f"{ex.id}: rendered sequence has no loss tokens")
        return RenderedExample(ex.id, ex.category, ex.kind, ids, labels)

    def render_prompt(self, messages: Sequence[dict[str, Any]], tools: Sequence[str] = ()) -> list[int]:
        """Token ids to feed the model so it writes the next assistant turn:
        exactly the training sequence up to (and including) ``assistant:\\n``."""
        if not messages:
            raise ValueError("render_prompt needs at least one message")
        if messages[-1]["role"] == "assistant":
            raise ValueError("render_prompt: the last message must not be an assistant turn")
        norm = [{"role": m["role"], "content": m["content"], "train": False} for m in messages]
        ex = NormalizedExample("<prompt>", "chat", "", None, tuple(norm), tuple(tools))
        segs = self.segments(ex) + [Segment("assistant:\n", None, False)]
        return [int(t) for t in self._tokenize(segs)[0]]

    def decode_completion(self, ids: Sequence[int]) -> str:
        """Text of a generated continuation: everything before the first EOS."""
        out: list[int] = []
        for token in ids:
            if int(token) == self.tokenizer.eos_token_id:
                break
            out.append(int(token))
        return self.tokenizer.decode(out)

    def explain(self, raw: Any) -> list[tuple[str, bool]]:
        """(text, trained?) runs of the exact training sequence — for docs and
        for eyeballing a dataset; specials are shown as ``<BOS>``/``<EOS>``."""
        ex = raw if isinstance(raw, NormalizedExample) else normalize_example(raw)
        runs = []
        for seg in self.segments(ex):
            runs.append(({"bos": "<BOS>", "eos": "<EOS>"}.get(seg.special or "", seg.text), seg.trainable))
        return runs
