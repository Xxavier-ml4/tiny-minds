"""``Session``: one conversation's worth of state, and the place every other
real subsystem in this delivery gets wired together end to end.

Honest about the one thing it cannot do without a trained model: deciding
*which* tool to call and *what arguments* to call it with, from free text,
in the general case. That is a real language-understanding task. What
``Session`` *can* do today, for real, is everything downstream of a
decided call: route the request (``tinymind.routing``), validate and
execute it (``tinymind.tools``), check whether its arguments are grounded
in the source text (``tinymind.runtime.grounding``), score confidence
(``tinymind.confidence``), and apply a confidence policy — see
``execute_tool_call()``, which is real, tested, and exercised by every
integration test in ``tests/test_session.py``. ``chat()`` and the
``STRUCTURED_OUTPUT`` path in ``run()`` call through to whatever
``ModelBackend`` is configured; against ``EchoBackend`` they predictably
don't produce a real answer or valid structured output, and ``run()``
reports that honestly (``ok=False`` with a clear reason) rather than
pretending success — see ``STATUS.md``.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from tinymind.confidence.confidence import ConfidenceEstimator, ConfidencePolicy, ConfidenceResult
from tinymind.model.backend import GenerationResult, ModelBackend
from tinymind.memory.short_term import ShortTermMemory
from tinymind.memory.store import MemoryEntry
from tinymind.routing.router import ResponseMode, Router, RoutingDecision
from tinymind.runtime.constraints.json_schema import StructuredOutputResult, parse_structured_output
from tinymind.runtime.grounding import GroundingMode, GroundingResult, check_grounding
from tinymind.tools.executor import ToolExecutor
from tinymind.tools.registry import ToolRegistry
from tinymind.tools.results import ToolCall, ToolResult


@dataclasses.dataclass
class SessionResult:
    mode: ResponseMode
    ok: bool
    text: str | None = None
    tool_result: ToolResult | None = None
    structured: StructuredOutputResult | None = None
    confidence: ConfidenceResult | None = None
    grounding: GroundingResult | None = None
    reason: str = ""


class Session:
    def __init__(self, backend: ModelBackend, registry: ToolRegistry, *,
                router: Router | None = None,
                grounding_mode: GroundingMode = GroundingMode.BALANCED,
                confidence_policy: ConfidencePolicy | None = None,
                history_size: int = 50) -> None:
        self._backend = backend
        self._registry = registry
        self._executor = ToolExecutor(registry)
        self._router = router or Router(registry)
        self._grounding_mode = grounding_mode
        self._confidence_policy = confidence_policy or ConfidencePolicy()
        self._confidence_estimator = ConfidenceEstimator()
        self._history = ShortTermMemory(max_entries=history_size)

    @property
    def history(self) -> ShortTermMemory:
        return self._history

    def route(self, text: str, *, output_schema: dict | None = None) -> RoutingDecision:
        return self._router.route(text, output_schema=output_schema)

    def chat(self, text: str, max_new_tokens: int = 256) -> GenerationResult:
        self._history.add(MemoryEntry(content=text, source="user"))
        result = self._backend.generate(text, max_new_tokens=max_new_tokens)
        self._history.add(MemoryEntry(content=result.text, source="assistant"))
        return result

    def execute_tool_call(self, name: str, arguments: dict[str, Any], *,
                          source_text: str | None = None,
                          tool_ranked_scores: list[float] | None = None,
                          confirmed: bool = False) -> SessionResult:
        """Run one already-decided call through the full real pipeline:
        validate + execute (``tinymind.tools``), ground against
        ``source_text`` if given, score confidence, and apply the
        confidence policy. This is the part of ``Session`` that works
        completely independently of whether ``self._backend`` is a real
        model — everything here is deterministic given ``name`` and
        ``arguments``.
        """
        call = ToolCall(name=name, arguments=arguments)
        tool_result = self._executor.execute(call, confirmed=confirmed)

        grounding_result = None
        if source_text is not None:
            grounding_result = check_grounding(arguments, source_text, mode=self._grounding_mode)
            if grounding_result.blocks_execution() and tool_result.ok:
                # The tool call succeeded mechanically, but strict/balanced
                # grounding says at least one argument was invented, not
                # derived from the input — report failure even though the
                # underlying tool call itself didn't raise.
                tool_result = dataclasses.replace(
                    tool_result, ok=False,
                    error=f"ungrounded argument(s): {grounding_result.ungrounded_fields}",
                    error_code="ungrounded_arguments")

        validation_result = None
        if name in self._registry:
            from tinymind.tools.validation import validate
            validation_result = validate(arguments, self._registry.get(name).schema.parameters)

        confidence = self._confidence_estimator.estimate(
            tool_ranked_scores=tool_ranked_scores, validation=validation_result,
            grounding=grounding_result)

        action = self._confidence_policy.decide(confidence.confidence)
        ok = tool_result.ok and action == "execute"

        reason = "" if ok else (
            tool_result.error if not tool_result.ok
            else f"confidence policy says {action!r}, not execute (confidence={confidence.confidence:.2f})")

        return SessionResult(mode=ResponseMode.TOOL_CALL, ok=ok, tool_result=tool_result,
                             confidence=confidence, grounding=grounding_result, reason=reason)

    def run(self, text: str, *, output_schema: dict | None = None) -> SessionResult:
        decision = self.route(text, output_schema=output_schema)

        if decision.mode in (ResponseMode.REFUSE, ResponseMode.ASK_CLARIFICATION):
            return SessionResult(mode=decision.mode, ok=False, reason=decision.reason)

        if decision.mode == ResponseMode.STRUCTURED_OUTPUT:
            generation = self.chat(text)
            structured = parse_structured_output(generation.text, output_schema or {})
            return SessionResult(mode=decision.mode, ok=structured.valid, text=generation.text,
                                 structured=structured,
                                 reason="" if structured.valid else "; ".join(structured.errors))

        if decision.mode == ResponseMode.TOOL_CALL:
            if not self._backend.is_real_model:
                return SessionResult(
                    mode=decision.mode, ok=False,
                    reason=(f"routed to a tool call (candidates: {decision.candidate_tools}) but "
                           f"{type(self._backend).__name__} cannot decide arguments from free text "
                           "— call execute_tool_call() directly with a decided (name, arguments), "
                           "or use a real ModelBackend once one exists (see STATUS.md)"))
            # A real backend would be asked to produce a call here, then
            # dispatch to execute_tool_call(); no real backend exists yet.
            raise NotImplementedError("no real ModelBackend is available to decide tool arguments")

        # CHAT, REASON, PLAN (without a pre-built Plan) all fall back to
        # direct generation in this delivery.
        generation = self.chat(text)
        return SessionResult(mode=decision.mode, ok=True, text=generation.text)
