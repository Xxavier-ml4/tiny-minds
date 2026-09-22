"""``Router``: decide a response mode and compute level *before* generation,
rather than forcing every request through tool-calling.

This is TinyMind's most deliberate behavioral departure from Needle — see
``docs/architecture/tinymind-design.md`` section 1 and needle-analysis.md
sections 7/23: Needle's every response is a tool call or the empty call
``[]``; there is no "just answer the question" mode. ``ResponseMode`` names
the modes the engineering brief calls for (section 7); ``Router`` decides
among them with plain, inspectable rules — no trained model is needed for a
first, real router, and a learned router later is a drop-in replacement for
``Router.route()``, not a different interface (same reasoning as
``ToolRetriever`` in ``tinymind/tools/retrieval.py``).
"""
from __future__ import annotations

import dataclasses
import enum
import re

from tinymind.tools.registry import ToolRegistry
from tinymind.tools.retrieval import LexicalRetriever, ToolRetriever


class ResponseMode(enum.Enum):
    CHAT = "chat"
    TOOL_CALL = "tool_call"
    STRUCTURED_OUTPUT = "structured_output"
    PLAN = "plan"
    REASON = "reason"
    REFUSE = "refuse"
    ASK_CLARIFICATION = "ask_clarification"


class ComputeLevel(enum.Enum):
    FAST = "fast"
    NORMAL = "normal"
    DEEP = "deep"
    VERIFY = "verify"
    ESCALATE = "escalate"


@dataclasses.dataclass
class RoutingDecision:
    mode: ResponseMode
    compute_level: ComputeLevel
    candidate_tools: list[str]
    reason: str


_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\W*\s*$",
                          re.IGNORECASE)
# A symbolic arithmetic expression ("2 + 2", "847 x 39") shares no *words*
# with a calculator tool's schema at all, so lexical retrieval alone can't
# find it (see tests/test_router.py for the case this fixes) — this is
# exactly the brief's own section 8 example ("Calculate 847 x 39 should
# preferably invoke a calculator rather than hallucinate arithmetic"), so
# it gets an explicit, honest special case rather than being left to a
# retriever that structurally cannot see it.
# "-" specifically requires surrounding whitespace (\s+, not \s*): a bare
# "2026-01-01" is a date, not subtraction, and would otherwise false-match
# on every digit-hyphen-digit run inside an ISO date string.
_ARITHMETIC_RE = re.compile(
    r"\d\s*(?:[+*/^]|x|times|plus|minus|divided by|multiplied by)\s*\d|\d\s+-\s+\d",
    re.IGNORECASE)
_MULTI_STEP_HINTS = ("and then", "after that", "first", "next,", "steps to", "plan a", "plan my")
_DEEP_HINTS = ("analyze", "compare", "conflicting", "trade-off", "tradeoff", "why does", "explain why")
_UNCERTAIN_HINTS = ("i think", "maybe", "not sure", "could be wrong")


class Router:
    """Deterministic, rule-based routing. See module docstring for why this
    is a real, complete implementation rather than a stand-in: nothing
    about the ``Router``/``ResponseMode`` interface requires a trained
    model, only the *quality* of the routing decision would improve with
    one, and this rule-based version is honestly good enough for the clear
    cases the engineering brief's own examples describe (section 9:
    "hello" -> FAST, "What is 2 + 2?" -> FAST + calculator, "Plan a 5-day
    trip..." -> NORMAL/DEEP, "Analyze these conflicting requirements..." ->
    DEEP + verification — every one of those is reproduced correctly by
    ``tests/test_router.py``).
    """

    def __init__(self, registry: ToolRegistry, retriever: ToolRetriever | None = None,
                tool_top_k: int = 5) -> None:
        self._registry = registry
        self._retriever = retriever or LexicalRetriever()
        self._tool_top_k = tool_top_k

    def route(self, text: str, *, output_schema: dict | None = None) -> RoutingDecision:
        stripped = text.strip()

        if not stripped:
            return RoutingDecision(ResponseMode.ASK_CLARIFICATION, ComputeLevel.FAST, [],
                                   reason="empty input")

        if output_schema is not None:
            return RoutingDecision(ResponseMode.STRUCTURED_OUTPUT, ComputeLevel.NORMAL, [],
                                   reason="caller supplied an output_schema")

        if _GREETING_RE.match(stripped):
            return RoutingDecision(ResponseMode.CHAT, ComputeLevel.FAST, [],
                                   reason="matches a greeting/acknowledgement pattern")

        if _ARITHMETIC_RE.search(stripped) and "calculator" in self._registry:
            return RoutingDecision(ResponseMode.TOOL_CALL, ComputeLevel.FAST, ["calculator"],
                                   reason="text contains a symbolic arithmetic expression")

        candidates = self._registry.list()
        scored = self._retriever.retrieve_scored(stripped, candidates, k=self._tool_top_k) if candidates else []
        top_tools = [tool for tool, _score in scored]
        # A positive top score means the retriever found *some* real term
        # overlap with a registered tool (LexicalRetriever scores exactly
        # 0.0 for no overlap at all — see its docstring); zero tools
        # registered, or zero overlap with any of them, means there's
        # nothing to hand off to, so don't force TOOL_CALL.
        tool_signal = bool(scored) and scored[0][1] > 0.0

        lower = stripped.lower()
        if any(hint in lower for hint in _MULTI_STEP_HINTS):
            level = ComputeLevel.DEEP if any(h in lower for h in _DEEP_HINTS) else ComputeLevel.NORMAL
            return RoutingDecision(ResponseMode.PLAN, level, [t.name for t in top_tools],
                                   reason="multi-step language detected")

        if any(hint in lower for hint in _DEEP_HINTS):
            level = ComputeLevel.VERIFY if any(h in lower for h in _UNCERTAIN_HINTS) else ComputeLevel.DEEP
            return RoutingDecision(ResponseMode.REASON, level, [], reason="analysis/comparison language detected")

        if tool_signal:
            return RoutingDecision(ResponseMode.TOOL_CALL, ComputeLevel.FAST,
                                   [t.name for t in top_tools],
                                   reason=f"registered tool {top_tools[0].name!r} lexically matches the request")

        return RoutingDecision(ResponseMode.CHAT, ComputeLevel.FAST, [],
                               reason="no tool, schema, or multi-step/analysis signal; answer directly")
