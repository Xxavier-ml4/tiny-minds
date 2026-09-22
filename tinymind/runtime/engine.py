"""``Engine``: creates and owns ``Session`` objects.

Directly addresses needle-analysis.md section 17 / 23: Needle allows only
one *active* base-model session per process (a second silently displaces
the first) and works around it for tuned models with a whole subprocess per
agent. TinyMind's ``Engine`` holds a dict of independent ``Session``
objects from the start — each with its own tool registry view, router,
grounding mode, and conversation history — so N concurrent sessions is the
normal case, not a special one. What it does *not* solve, because it's a
question about the native runtime rather than this Python layer: whether
two sessions can run inference *concurrently* on one loaded set of model
weights without contention. That's tracked as a native-runtime question in
``STATUS.md`` (see ``docs/architecture/tinymind-design.md`` section 10) —
today's only ``ModelBackend`` (``EchoBackend``) is stateless per call and
has no contention to manage, so this Python layer's job (independent
per-session *state*) is fully real and tested even though the deeper
"shared weights, concurrent native inference" question isn't reachable yet.
"""
from __future__ import annotations

from tinymind.confidence.confidence import ConfidencePolicy
from tinymind.model.backend import ModelBackend
from tinymind.routing.router import Router
from tinymind.runtime.grounding import GroundingMode
from tinymind.runtime.session import Session
from tinymind.tools.registry import ToolRegistry


class EngineError(ValueError):
    pass


class Engine:
    def __init__(self, backend: ModelBackend, registry: ToolRegistry | None = None) -> None:
        self._backend = backend
        self._default_registry = registry or ToolRegistry()
        self._sessions: dict[str, Session] = {}

    def create_session(self, session_id: str, *, registry: ToolRegistry | None = None,
                       grounding_mode: GroundingMode = GroundingMode.BALANCED,
                       confidence_policy: ConfidencePolicy | None = None) -> Session:
        if session_id in self._sessions:
            raise EngineError(f"session {session_id!r} already exists; call close_session() first "
                              "or choose a different id — Engine does not silently displace a session "
                              "the way Needle's single global engine handle does (see this module's "
                              "docstring)")
        session = Session(self._backend, registry or self._default_registry,
                          router=Router(registry or self._default_registry),
                          grounding_mode=grounding_mode, confidence_policy=confidence_policy)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise EngineError(f"no session {session_id!r}; call create_session() first") from None

    def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.history.clear()

    def session_ids(self) -> list[str]:
        return list(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)
