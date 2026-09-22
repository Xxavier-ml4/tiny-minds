"""``TransformerBackend``: the real ``ModelBackend`` implementation, using
``TinyMindTransformer`` — what the brief section 17 asks for. Does not
replace or delete ``EchoBackend`` (still real and useful for runtime tests
that don't want the cost or randomness of an actual, if untrained, forward
pass — see that class's own docstring, unchanged by this phase).

Loads weights via ``tinymind.model.tm_export.import_from_tm`` (the
hardened, checksum-verified `.tm` path) — never via ``pickle`` (brief
section 28's explicit prohibition), and never executes anything from a
model file beyond reading declared tensors into declared parameter slots.
"""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from tinymind.model.backend import GenerationResult, ModelBackend
from tinymind.model.generation import ModelGenerationConfig, generate_with_cache_ids
from tinymind.model.model import TinyMindTransformer
from tinymind.model.tensor import no_grad
from tinymind.model.tm_export import import_from_tm
from tinymind.model.tokenizer import ByteTokenizer, Tokenizer


class TransformerBackend(ModelBackend):
    """A real, untrained-unless-you-load-real-weights neural backend. See
    ``STATUS.md`` for exactly what "real" means here: the architecture,
    forward pass, and generation loop are genuine; whether the *weights*
    are any good depends entirely on what was trained and loaded — this
    class makes no claim about that on its own, and ``is_real_model``
    returns ``True`` regardless of training quality, matching its actual
    meaning (see ``tinymind.model.backend.ModelBackend.is_real_model``'s
    docstring: "real" distinguishes an actual neural forward pass from a
    stand-in like ``EchoBackend``, not "well-trained" from "poorly
    trained").
    """

    def __init__(self, tokenizer: Tokenizer | None = None, renderer=None) -> None:
        self._model: TinyMindTransformer | None = None
        self._tokenizer: Tokenizer = tokenizer or ByteTokenizer()
        # A prompt renderer (tinymind.data.render.ChatRenderer). With one, ``generate(text)`` sends the text
        # as a user turn in exactly the format the model was trained on; without one (Phase 3A models, which
        # were trained on raw text) the prompt is encoded verbatim, as before.
        self._renderer = renderer
        self._loaded_path: str | None = None

    def load(self, path: str) -> None:
        """``path`` is a ``.tm`` file or an inference-package directory. A package (or a ``.tm`` whose metadata
        carries a tokenizer and prompt template) configures the tokenizer and renderer itself."""
        from pathlib import Path

        if Path(path).is_dir():
            from tinymind.export.package import load_package
            pkg = load_package(path)
            self._model, self._tokenizer, self._renderer = pkg.model, pkg.tokenizer, pkg.renderer
        else:
            from tinymind.data.render import ChatRenderer
            from tinymind.model.tm_export import read_tm_metadata
            from tinymind.model.tokenizer import tokenizer_from_spec
            self._model = import_from_tm(path)
            meta = read_tm_metadata(path)
            if "tokenizer" in meta:
                self._tokenizer = tokenizer_from_spec(meta["tokenizer"])
                if meta.get("renderer") is not None:
                    self._renderer = ChatRenderer(self._tokenizer, meta["renderer"]["template"])
        self._model.check_tokenizer_compatibility(self._tokenizer)
        self._loaded_path = str(path)

    def _prompt_ids(self, prompt: str) -> list[int]:
        if self._renderer is not None:
            return self._renderer.render_prompt([{"role": "user", "content": prompt}])
        return self._tokenizer.encode(prompt, add_bos=True)

    def _decode_new(self, new_ids: list[int]) -> str:
        return self._renderer.decode_completion(new_ids) if self._renderer is not None else self._tokenizer.decode(new_ids)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_real_model(self) -> bool:
        return True

    def _require_model(self) -> TinyMindTransformer:
        if self._model is None:
            raise RuntimeError(
                "TransformerBackend.load(path) must be called before generate()/embed() — "
                "no model is loaded")
        return self._model

    def generate(self, prompt: str, max_new_tokens: int = 256,
                temperature: float = 0.0) -> GenerationResult:
        model = self._require_model()
        start = time.monotonic()
        input_ids = np.array([self._prompt_ids(prompt)])
        config = ModelGenerationConfig(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                       temperature=max(temperature, 1e-6),
                                       eos_token_id=self._tokenizer.eos_token_id)
        output_ids = generate_with_cache_ids(model, input_ids, config)
        new_ids = output_ids[0][input_ids.shape[1]:].tolist()
        text = self._decode_new(new_ids)
        finish_reason = "stop" if (new_ids and new_ids[-1] == self._tokenizer.eos_token_id) else "length"
        return GenerationResult(text=text, tokens_generated=len(new_ids), finish_reason=finish_reason,
                                latency_ms=(time.monotonic() - start) * 1000.0)

    def stream(self, prompt: str, max_new_tokens: int = 256,
              temperature: float = 0.0) -> Iterator[str]:
        # A real token-by-token streaming implementation would drive the
        # same prefill+decode loop as generate_with_cache_ids one step at a
        # time; today this yields the complete result as a single chunk,
        # which is honest about latency (no partial output appears early)
        # while still satisfying the Iterator[str] interface every caller
        # already uses. See STATUS.md.
        result = self.generate(prompt, max_new_tokens, temperature)
        yield result.text

    def reset(self) -> None:
        # No persistent per-call state to clear: generate()/embed() each
        # build a fresh KVCache internally (see tinymind/model/generation.py)
        # rather than holding one across calls — the same reasoning
        # EchoBackend's reset() documents for itself.
        pass

    def embed(self, text: str) -> list[float]:
        model = self._require_model()
        input_ids = np.array([self._tokenizer.encode(text, add_bos=True, add_eos=True)])
        with no_grad():
            out = model(input_ids, output_hidden_states=True)
        pooled = out.hidden_states.data[0].mean(axis=0)  # mean-pool over sequence positions
        norm = float(np.linalg.norm(pooled)) or 1.0
        return (pooled / norm).tolist()
