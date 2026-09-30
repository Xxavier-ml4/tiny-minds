"""Reading, writing and hashing a BPE ``tokenizer.json`` (brief section 2).

``BPETokenizer`` (``tinymind/model/bpe.py``) already knows how to train from
texts, serialise to a ``spec()`` (the complete merge list plus special ids)
and rebuild from one. This module is the file/CLI layer around it, and the one
place the on-disk ``tokenizer.json`` format is defined, so a checkpoint, an
exported package and the ``tinymind tokenizer train-bpe`` command all agree:

* ``train_bpe`` — deterministic training from a corpus (plain-text files, a
  directory of them, or JSONL records). Two runs on the same corpus produce
  byte-identical specs and therefore the same hash.
* ``save_tokenizer`` / ``load_tokenizer`` — canonical JSON on disk (sorted
  keys, ``\\n`` terminated) so the file bytes are reproducible, plus a
  ``spec_hash`` (SHA-256 of the canonical spec JSON — the same function
  checkpoints and packages compare on resume/promotion).
* ``read_corpus`` — the corpus reader the CLI uses; kept here so tests can
  exercise it directly.

The tokenizer can be reconstructed *solely* from ``tokenizer.json``: nothing
else about the training run is needed to encode/decode with it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from tinymind.model.bpe import BPETokenizer
from tinymind.model.tokenizer import Tokenizer, hash_spec, tokenizer_from_spec

_TEXT_SUFFIXES = (".txt", ".text", ".md")


def read_corpus(inputs: str | Path | Iterable[str | Path]) -> list[str]:
    """Corpus text from ``inputs`` (a path, or several). Each path may be:

    * a ``.jsonl`` file — every record contributes its text: a ``"text"``
      field if present, otherwise the concatenation of its ``messages``
      contents (so a curriculum file can train the tokenizer that will encode
      it);
    * a ``.txt``/``.md``/other text file — split into lines (blank lines
      dropped);
    * a directory — every text/JSONL file under it, in sorted path order.

    Order is deterministic (sorted paths, file order within a file), though BPE
    training itself depends only on piece frequencies, not order.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    out: list[str] = []
    for path in _iter_paths(inputs):
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out.extend(_record_texts(rec))
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(line)
    return out


def _iter_paths(inputs: Iterable[str | Path]) -> Iterator[Path]:
    for entry in inputs:
        p = Path(entry)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and (f.suffix == ".jsonl" or f.suffix in _TEXT_SUFFIXES):
                    yield f
        elif p.is_file():
            yield p
        else:
            raise FileNotFoundError(f"corpus path {entry!r} does not exist")


def _record_texts(rec: dict[str, Any]) -> list[str]:
    if "text" in rec:
        return [str(rec["text"])]
    if "messages" in rec:
        return [str(m.get("content", "")) for m in rec["messages"] if str(m.get("content", "")).strip()]
    return []


def train_bpe(inputs: str | Path | Iterable[str | Path], vocab_size: int) -> BPETokenizer:
    """Train a byte-level BPE tokenizer of ``vocab_size`` from ``inputs``.

    Deterministic: the same corpus yields the same merges (BPETokenizer.train's
    frequency-only, fixed tie-break procedure), so ``save_tokenizer`` writes the
    same bytes and ``spec_hash`` is stable across runs.
    """
    texts = read_corpus(inputs)
    if not texts:
        raise ValueError("corpus is empty: no text found in the given input(s)")
    return BPETokenizer.train(texts, vocab_size=vocab_size)


def spec_hash(tok: Tokenizer) -> str:
    """SHA-256 of the tokenizer's canonical spec JSON — its stored identity."""
    return hash_spec(tok.spec())


def _canonical(spec: dict[str, Any]) -> str:
    # Human-diffable but reproducible: sorted keys, no trailing spaces, newline
    # terminated. hash_spec() hashes the compact form of the same object, so the
    # hash does not depend on this indentation.
    return json.dumps(spec, sort_keys=True, indent=2) + "\n"


def save_tokenizer(tok: Tokenizer, path: str | Path) -> str:
    """Write ``tok``'s spec to ``path`` as canonical ``tokenizer.json``.

    Returns the ``spec_hash``. Creates parent directories as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(tok.spec()), encoding="utf-8")
    return spec_hash(tok)


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Rebuild a tokenizer solely from a ``tokenizer.json`` written by
    ``save_tokenizer`` (or embedded in a package/checkpoint). Unknown types or
    inconsistent specs raise (via ``tokenizer_from_spec``)."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return tokenizer_from_spec(spec)
