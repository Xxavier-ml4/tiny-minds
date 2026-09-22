"""``tinymind serve``: a local HTTP API server, stdlib-only (no Flask/
FastAPI dependency — consistent with this delivery's minimal-dependency
posture; see ``pyproject.toml``), binding ``127.0.0.1`` unless the caller
passes a different host explicitly (brief section 29 and
``docs/architecture/tinymind-design.md`` section 12).

Endpoints, matching the brief's section 29 list where they're implementable
without a real model:

- ``GET  /v1/health`` — real.
- ``GET  /v1/model`` — real: reports which ``ModelBackend`` is loaded and
  whether it's a real trained model (``is_real_model``), rather than
  reporting nothing or faking a model card.
- ``POST /v1/chat/completions`` — an OpenAI-compatible-*shaped* endpoint
  (brief: "Use an OpenAI-compatible endpoint where practical"), real
  end to end against whatever ``ModelBackend`` is configured; against the
  default ``EchoBackend`` it returns an honest echo, not a real chat
  completion — see that class's docstring.
- ``POST /v1/reset`` — real.
- ``POST /v1/tools`` — real: lists the server's registered tools.

Not implemented: ``/v1/generate``/``/v1/complete`` as *separate* endpoints
from ``/v1/chat/completions`` (the brief lists them separately; this
delivery has one generation path, not several, since splitting them apart
would just be three thin wrappers around the same ``Session.chat()`` call
with no behavioral difference to justify three endpoints).
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tinymind import Model, __version__
from tinymind.tools.builtins import register_builtins
from tinymind.tools.registry import ToolRegistry


def _model_with_registry(registry: ToolRegistry) -> Model:
    return Model(registry=registry)


class _Handler(BaseHTTPRequestHandler):
    model: Model  # set per-server via make_handler_class

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass  # quiet by default; a real deployment would wire this to logging

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        if self.path == "/v1/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/model":
            self._send_json(200, {
                "backend": type(self.model._backend).__name__,
                "is_real_model": self.model._backend.is_real_model,
                "loaded": self.model._backend.is_loaded,
            })
        else:
            self._send_json(404, {"error": f"no such endpoint: GET {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        if self.path == "/v1/chat/completions":
            messages = body.get("messages", [])
            prompt = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
            if not prompt and "prompt" in body:
                prompt = body["prompt"]
            text = self.model.generate(prompt)
            self._send_json(200, {
                "id": "cmpl-tinymind",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop"}],
            })
        elif self.path == "/v1/tools":
            self._send_json(200, {"tools": self.model.tools.schemas()})
        elif self.path == "/v1/reset":
            self.model.reset()
            self._send_json(200, {"status": "reset"})
        else:
            self._send_json(404, {"error": f"no such endpoint: POST {self.path}"})


def make_handler_class(model: Model) -> type:
    return type("BoundHandler", (_Handler,), {"model": model})


def serve(host: str = "127.0.0.1", port: int = 8420, model: Model | None = None) -> None:
    if model is None:
        registry = ToolRegistry()
        register_builtins(registry)
        model = _model_with_registry(registry)

    handler_class = make_handler_class(model)
    server = ThreadingHTTPServer((host, port), handler_class)
    print(f"tinymind {__version__} serving on http://{host}:{port} "
         f"(backend: {type(model._backend).__name__}, real model: {model._backend.is_real_model})")
    if not model._backend.is_real_model:
        print("note: this backend is a deterministic stand-in, not a trained model — see STATUS.md")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
