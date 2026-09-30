"""Real pretraining-corpus ingestion: sharded, streaming, verifiable,
deduplicated, deterministically split, decontaminated, with full provenance.

This replaces the Stage-1 synthetic placeholder as the *primary* language data
(synthetic generators stay, as supplemental sources). Nothing here is tied to a
model size: the same pipeline feeds a 50M run and a 1B run; only the manifest
(sources and shard lists), the fractions and the thresholds differ.

A corpus source is a dataset-manifest entry of type ``local`` or ``url``::

    source          stable id
    shards          list of shard paths / file:// / http(s) URLs (or a single ``path``/``url``)
    sha256          expected SHA-256 per shard (list, same order) — verified when given;
                    the prepared manifest records every shard's observed hash so you can pin it
    format          "text" (plain prose; paragraphs separated by blank lines) or "jsonl"
                    (one document per line, ``text_field``); ``.gz`` shards are decompressed
    text_field      JSONL field holding the document text (default "text")
    auth_env        name of an environment variable holding a bearer token for private URLs;
                    the token is read at download time, sent only to the shard's own host
                    (never forwarded on a redirect), and never logged or stored
    license         recorded verbatim in the provenance manifest
    chunk_chars     target chunk size; keep chunk_chars x tokens-per-char + 2 <= max_seq_len
    min_chars       drop chunks shorter than this
    group_chunks    consecutive chunks of a text shard that share a split (limits leakage of
                    adjacent context between train and held-out splits)
    max_records     cap on kept chunks from this source, spread evenly over its shards
                    (at most ceil(max_records / shards) from each, so every shard is represented)
    supplemental    true for non-natural text (e.g. generated prose); excluded from natural_*

Pipeline (:func:`prepare_corpus`): stream every shard in manifest order ->
verify its hash -> split into paragraphs/documents -> chunk -> exact dedup
(normalised text) and near dedup (MinHash LSH over 5-word shingles) across the
whole corpus -> optional n-gram decontamination against evaluation files ->
deterministic group-level train/val/test assignment keyed on a seeded hash ->
``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` of ``{"id", "text", "category",
"source"}`` records + ``corpus_manifest.json`` (sources, licenses, redacted URLs,
shard hashes, token counts per split, tokenizer hash, dataset-manifest hash,
dedup/decontamination/contamination reports). Downloads land in a cache under
the output directory; neither is meant for git (see ``.gitignore``).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

import numpy as np

from tinymind.data.contamination import normalize

CORPUS_KIND = "tinymind-corpus"
CORPUS_FORMAT_VERSION = 1
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MERSENNE = np.uint64(4294967291)  # largest prime below 2**32: MinHash values fit uint32


class CorpusError(ValueError):
    pass


# --------------------------------------------------------------------------- shards
def redact_url(url: str) -> str:
    """A URL safe to log and record: no user-info, query string or fragment
    (pre-signed URLs carry credentials there)."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return url
    host = p.hostname or ""
    if p.port:
        host = f"{host}:{p.port}"
    return urlunparse((p.scheme, host, p.path, "", "", ""))


def shard_needs_network(shard: str) -> bool:
    return urlparse(str(shard)).scheme in ("http", "https")


def _local_path(shard: str, base_dir: Path) -> Path:
    p = urlparse(str(shard))
    if p.scheme == "file":
        return Path(urllib.request.url2pathname(p.path))
    path = Path(shard)
    return path if path.is_absolute() else (base_dir / path)


def _hash_file(path: Path) -> tuple[str, int]:
    digest, n = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            n += len(block)
    return digest.hexdigest(), n


def fetch_shard(shard: str, *, base_dir: Path, cache_dir: Path, allow_download: bool, auth_env: str = "",
                expected_sha256: str = "", timeout: float = 120.0) -> tuple[Path, str, int]:
    """``(local file, sha256, bytes)`` for one shard. Local and ``file://`` shards
    are read in place; ``http(s)`` shards are streamed into ``cache_dir`` in 1 MiB
    blocks with the hash computed on the fly, and only when ``allow_download``.
    A given ``expected_sha256`` must match or the shard is rejected."""
    if shard_needs_network(shard):
        shown = redact_url(shard)
        if not allow_download:
            raise CorpusError(f"shard {shown} needs a network: re-run with --allow-download in an environment that "
                              "permits it (preparation is hermetic by default and never downloads silently)")
        request = urllib.request.Request(shard, headers={"User-Agent": "tinymind-corpus/1"})
        if auth_env:
            token = os.environ.get(auth_env, "")
            if not token:
                raise CorpusError(f"shard {shown} needs a bearer token in the environment variable {auth_env}, "
                                  "which is not set")
            # unredirected: the token goes to this host only and is not forwarded on a redirect
            request.add_unredirected_header("Authorization", f"Bearer {token}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = Path(urlparse(shard).path).name or "shard"
        dest = cache_dir / f"{hashlib.sha256(shown.encode()).hexdigest()[:16]}-{name}"
        if dest.is_file():  # fetched earlier in this job (e.g. by prepare-external): reuse it if it still verifies
            got, n = _hash_file(dest)
            if not expected_sha256 or got == expected_sha256:
                return dest, got, n
            dest.unlink()
        part = dest.with_name(dest.name + ".part")
        digest, n = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    digest.update(block)
                    out.write(block)
                    n += len(block)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            part.unlink(missing_ok=True)
            reason = getattr(exc, "reason", None) or getattr(exc, "code", None) or type(exc).__name__
            raise CorpusError(f"download of {shown} failed: {reason}") from None
        got = digest.hexdigest()
        if expected_sha256 and got != expected_sha256:
            part.unlink(missing_ok=True)
            raise CorpusError(f"shard {shown}: sha256 {got} != expected {expected_sha256} (the data changed; "
                              "update the manifest deliberately)")
        part.replace(dest)
        return dest, got, n
    path = _local_path(shard, base_dir)
    if not path.is_file():
        raise CorpusError(f"shard {shard} not found (looked at {path})")
    got, n = _hash_file(path)
    if expected_sha256 and got != expected_sha256:
        raise CorpusError(f"shard {shard}: sha256 {got} != expected {expected_sha256} (the data changed; update the "
                          "manifest deliberately)")
    return path, got, n


def shard_format(shard: str, declared: str = "") -> str:
    if declared:
        return declared
    name = urlparse(str(shard)).path.lower()
    name = name[:-3] if name.endswith(".gz") else name
    return "jsonl" if name.endswith((".jsonl", ".ndjson", ".json")) else "text"


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- documents and chunks
def _paragraphs(lines: Iterable[str]) -> Iterator[str]:
    """Paragraphs of prose: blank lines separate them; hard-wrapped lines inside
    a paragraph are joined with single spaces."""
    buf: list[str] = []
    for line in lines:
        s = line.strip()
        if s:
            buf.append(s)
        elif buf:
            yield " ".join(buf)
            buf = []
    if buf:
        yield " ".join(buf)


def iter_documents(path: Path, fmt: str, text_field: str = "text") -> Iterator[list[str]]:
    """Documents of a shard as lists of paragraphs. ``jsonl``: one document per
    record (its ``text_field``). ``text``: the whole shard is one continuous
    stream (chunked and grouped downstream)."""
    with _open_text(path) as handle:
        if fmt == "jsonl":
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CorpusError(f"{path.name}:{number}: invalid JSON ({exc})") from exc
                text = record.get(text_field) if isinstance(record, dict) else None
                if isinstance(text, str) and text.strip():
                    if re.search(r"\n\s*\n", text):  # blank-line paragraphs (possibly hard-wrapped inside)
                        yield list(_paragraphs(text.splitlines()))
                    else:  # one paragraph per line (web text such as C4)
                        yield [s.strip() for s in text.splitlines() if s.strip()]
        elif fmt == "text":
            yield list(_paragraphs(handle))
        else:
            raise CorpusError(f"unknown corpus format {fmt!r} (use 'text' or 'jsonl')")


def _split_long(paragraph: str, limit: int) -> list[str]:
    if len(paragraph) <= limit:
        return [paragraph]
    pieces: list[str] = []
    cur = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if len(sentence) > limit:  # no sentence boundary to use: split at whitespace
            if cur:
                pieces.append(cur)
                cur = ""
            line = ""
            for word in sentence.split():
                if line and len(line) + 1 + len(word) > limit:
                    pieces.append(line)
                    line = word
                else:
                    line = f"{line} {word}" if line else word
            if line:
                pieces.append(line)
            continue
        if cur and len(cur) + 1 + len(sentence) > limit:
            pieces.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}" if cur else sentence
    if cur:
        pieces.append(cur)
    return [p[:limit] for p in pieces]


def chunk_paragraphs(paragraphs: Iterable[str], chunk_chars: int) -> Iterator[str]:
    """Pack consecutive paragraphs into chunks of at most ``chunk_chars``
    characters (paragraphs joined by a blank line), breaking over-long
    paragraphs at sentence boundaries first and whitespace second."""
    if chunk_chars < 16:
        raise CorpusError("chunk_chars must be >= 16")
    cur: list[str] = []
    size = 0
    for para in paragraphs:
        for piece in _split_long(para, chunk_chars):
            add = len(piece) + (2 if cur else 0)
            if cur and size + add > chunk_chars:
                yield "\n\n".join(cur)
                cur, size, add = [], 0, len(piece)
            cur.append(piece)
            size += add
    if cur:
        yield "\n\n".join(cur)


# --------------------------------------------------------------------------- dedup
class NearDuplicateIndex:
    """Exact (normalised-text) and near-duplicate (MinHash + LSH banding over
    5-word shingles) detection, in stream order: the first occurrence is kept.
    Estimated Jaccard similarity >= ``threshold`` counts as a near duplicate.
    Memory is ``num_perm x 4`` bytes per kept chunk plus the band buckets; for
    corpora that are already deduplicated upstream, use ``mode='exact'`` or
    ``'none'``."""

    def __init__(self, *, mode: str = "near", threshold: float = 0.8, num_perm: int = 64, bands: int = 16,
                 shingle: int = 5, seed: int = 0) -> None:
        if mode not in ("near", "exact", "none"):
            raise CorpusError(f"dedup mode must be near|exact|none, got {mode!r}")
        if num_perm % bands:
            raise CorpusError("num_perm must be a multiple of bands")
        self.mode, self.threshold, self.num_perm, self.bands, self.shingle = mode, float(threshold), num_perm, bands, shingle
        self.rows = num_perm // bands
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5EED]))
        self._a = rng.integers(1, 2 ** 31, size=num_perm, dtype=np.uint64)
        self._b = rng.integers(0, 2 ** 31, size=num_perm, dtype=np.uint64)
        self._exact: set[bytes] = set()
        self._paragraphs: set[bytes] = set()
        self._buckets: list[dict[bytes, list[int]]] = [dict() for _ in range(bands)]
        self._signatures: list[np.ndarray] = []
        self.exact_removed = 0
        self.near_removed = 0
        self.paragraphs_removed = 0

    def new_paragraph(self, paragraph: str, min_chars: int = 40) -> bool:
        """False for a repeat of an earlier paragraph of at least ``min_chars``
        characters (boilerplate, copied passages); short lines always pass."""
        if self.mode == "none" or len(paragraph) < min_chars:
            return True
        key = hashlib.sha256(normalize(paragraph).encode("utf-8")).digest()[:16]
        if key in self._paragraphs:
            self.paragraphs_removed += 1
            return False
        self._paragraphs.add(key)
        return True

    def signature(self, norm: str) -> np.ndarray:
        words = norm.split()
        k = self.shingle
        grams = [" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))] if words else [""]
        h = np.fromiter((zlib.crc32(g.encode("utf-8")) for g in set(grams)), dtype=np.uint64)
        return ((self._a[:, None] * h[None, :] + self._b[:, None]) % _MERSENNE).min(axis=1).astype(np.uint32)

    def add(self, text: str) -> str:
        """``"kept"``, ``"exact"`` or ``"near"`` (the last two are removed)."""
        if self.mode == "none":
            return "kept"
        norm = normalize(text)
        key = hashlib.sha256(norm.encode("utf-8")).digest()[:16]
        if key in self._exact:
            self.exact_removed += 1
            return "exact"
        if self.mode == "near":
            sig = self.signature(norm)
            bands = [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]
            seen: set[int] = set()
            for band, bucket_key in zip(self._buckets, bands):
                for other in band.get(bucket_key, ()):
                    if other in seen:
                        continue
                    seen.add(other)
                    if float(np.mean(self._signatures[other] == sig)) >= self.threshold:
                        self.near_removed += 1
                        return "near"
            index = len(self._signatures)
            self._signatures.append(sig)
            for band, bucket_key in zip(self._buckets, bands):
                band.setdefault(bucket_key, []).append(index)
        self._exact.add(key)
        return "kept"

    def report(self) -> dict[str, Any]:
        return {"mode": self.mode, "threshold": self.threshold if self.mode == "near" else None,
                "num_perm": self.num_perm if self.mode == "near" else None,
                "bands": self.bands if self.mode == "near" else None, "shingle_words": self.shingle,
                "paragraphs_removed": self.paragraphs_removed,
                "exact_removed": self.exact_removed, "near_removed": self.near_removed}


def dedup_texts(texts: Sequence[str], **kw: Any) -> tuple[list[int], dict[str, Any]]:
    """Indices of the texts kept (first occurrence wins) and the dedup report."""
    index = NearDuplicateIndex(**kw)
    kept = [i for i, t in enumerate(texts) if index.add(t) == "kept"]
    return kept, index.report()


# --------------------------------------------------------------------------- decontamination
def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def eval_ngrams(records: Iterable[Mapping[str, Any]], n: int = 8) -> set[tuple[str, ...]]:
    """n-grams of evaluation prompts (``text`` records, or the non-assistant
    turns of ``messages`` records); prompts shorter than n words contribute none."""
    from tinymind.data.contamination import texts as record_texts
    grams: set[tuple[str, ...]] = set()
    for r in records:
        prompt, _ = record_texts(dict(r))
        grams |= ngrams(prompt.replace("\x1f", " "), n)
    return grams


def contaminated(text: str, grams: set[tuple[str, ...]], n: int = 8) -> bool:
    if not grams:
        return False
    words = normalize(text).split()
    return any(tuple(words[i:i + n]) in grams for i in range(len(words) - n + 1))


# --------------------------------------------------------------------------- deterministic sampling
def _rank(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()[:8], "big")


def sample_records(paths: Sequence[str | Path], k: int, *, seed: int = 0) -> tuple[list[dict[str, Any]], int]:
    """A deterministic, uniform sample without replacement of ``k`` JSONL records
    drawn from ALL of ``paths``: each record is ranked by ``sha256(seed:id)``
    (its text when it has no string ``id``) and the ``k`` lowest ranks are kept —
    a one-pass streaming bottom-k that holds only ``k`` records in memory. Unlike
    "the first ``k`` lines" it does not depend on file, shard or source order, so
    the last shard is represented as well as the first. Returns ``(records in
    rank order, records seen)``."""
    import heapq
    if k < 0:
        raise CorpusError("sample size must be >= 0")
    heap: list[tuple[int, str]] = []  # (-rank, line): the root is the worst of the k kept
    seen = 0
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rid = record.get("id") if isinstance(record, dict) else None
                rank = _rank(seed, rid if isinstance(rid, str) and rid else str(record.get("text", line)))
                seen += 1
                if len(heap) < k:
                    heapq.heappush(heap, (-rank, line))
                elif k and (-rank, line) > heap[0]:
                    heapq.heapreplace(heap, (-rank, line))
    chosen = sorted((-neg, line) for neg, line in heap)
    return [json.loads(line) for _, line in chosen], seen


def tokenizer_sample(out_dir: str | Path, *, corpus_dir: str | Path | None = None,
                     external_dir: str | Path | None = None, natural_records: int = 8000,
                     supplemental_records: int = 4000, seed: int = 0) -> dict[str, Any]:
    """Tokenizer-training input that is deterministic AND representative:

    * ``natural_sample.jsonl`` — :func:`sample_records` over the prepared
      corpus's TRAIN split (every shard and source; never val/test);
    * ``supplemental_sample.jsonl`` — :func:`sample_records` over the
      ``prepare-external`` entries that are NOT natural text (the synthetic
      supplements, whose formats the corpus does not contain; the natural
      entries there hold the corpus's held-out text too, so they are skipped);
    * ``sample_manifest.json`` — what was drawn from what (hashes, counts, seed).

    The seed is fixed by the caller (not the training seed), so every stage's
    job derives the same sample and hence the same tokenizer."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"kind": "tinymind-tokenizer-sample", "format_version": 1, "seed": seed,
                                "method": "uniform without replacement: lowest sha256(seed:id) ranks (bottom-k)"}

    def write(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        path = out / name
        with path.open("w", encoding="utf-8") as handle:
            for r in records:
                handle.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        size = sum(len(str(r.get("text", "")).encode("utf-8")) for r in records)
        return {"file": name, "records_sampled": len(records), "text_bytes": size, "sha256": _hash_file(path)[0]}

    if corpus_dir is not None:
        cm = load_corpus(corpus_dir)
        train = Path(corpus_dir) / cm["splits"]["train"]["file"]
        records, seen = sample_records([train], natural_records, seed=seed)
        manifest["natural"] = {"from": str(train), "corpus_sha256": cm["corpus_sha256"],
                               "records_available": seen, **write("natural_sample.jsonl", records)}
    if external_dir is not None:
        prepared = json.loads((Path(external_dir) / "prepared_manifest.json").read_text(encoding="utf-8"))
        files = [Path(external_dir) / e["file"] for e in prepared["entries"] if not e.get("natural")]
        records, seen = sample_records(files, supplemental_records, seed=seed)
        manifest["supplemental"] = {"from": [str(f) for f in files], "records_available": seen,
                                    **write("supplemental_sample.jsonl", records)}
    if not any(manifest.get(k, {}).get("records_sampled") for k in ("natural", "supplemental")):
        raise CorpusError("nothing to sample: give a prepared corpus (--corpus) and/or prepared external data (--external)")
    (out / "sample_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- splits
def split_groups(group_keys: Sequence[str], *, validation: float, test: float, seed: int) -> dict[str, str]:
    """Deterministic group -> split assignment: groups are ranked by
    ``sha256(seed:key)``; the first ceil(test x G) go to test, the next
    ceil(validation x G) to val (at least one each when the fraction is positive
    and there are 3+ groups), the rest to train. Stable under re-ordering; a
    given group's text always lands in one split."""
    if not (0.0 <= validation < 1.0 and 0.0 <= test < 1.0 and validation + test < 1.0):
        raise CorpusError("need 0 <= validation, test and validation + test < 1")
    ranked = sorted(set(group_keys), key=lambda k: (hashlib.sha256(f"{seed}:{k}".encode()).hexdigest(), k))
    g = len(ranked)
    n_test = math.ceil(test * g) if test > 0 else 0
    n_val = math.ceil(validation * g) if validation > 0 else 0
    if g < 3:  # too few groups to hold anything out without starving training
        n_test = n_val = 0
    out: dict[str, str] = {}
    for i, key in enumerate(ranked):
        out[key] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return out


# --------------------------------------------------------------------------- the pipeline
@dataclass
class CorpusSource:
    source: str
    shards: list[str]
    type: str = "local"
    sha256: list[str] = field(default_factory=list)
    format: str = ""
    text_field: str = "text"
    auth_env: str = ""
    license: str = ""
    description: str = ""
    weight: float = 1.0
    chunk_chars: int = 1000
    min_chars: int = 32
    group_chunks: int = 8
    max_records: int = 0
    supplemental: bool = False

    def validate(self) -> None:
        if not self.shards:
            raise CorpusError(f"corpus source {self.source!r} has no shards")
        if self.sha256 and len(self.sha256) != len(self.shards):
            raise CorpusError(f"corpus source {self.source!r}: {len(self.sha256)} sha256 values for "
                              f"{len(self.shards)} shards")
        if self.auth_env and not _ENV_NAME.match(self.auth_env):
            raise CorpusError(f"corpus source {self.source!r}: auth_env must name an environment variable")
        if self.format not in ("", "text", "jsonl"):
            raise CorpusError(f"corpus source {self.source!r}: format must be 'text' or 'jsonl'")
        if self.chunk_chars < 16 or self.min_chars < 1 or self.group_chunks < 1 or self.max_records < 0:
            raise CorpusError(f"corpus source {self.source!r}: invalid chunk_chars/min_chars/group_chunks/max_records")


def tokenizer_hash(tokenizer: Any) -> str:
    return hashlib.sha256(json.dumps(tokenizer.spec(), sort_keys=True).encode("utf-8")).hexdigest()


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def _record_id(source: str, text: str) -> str:
    return f"{source}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def prepare_corpus(sources: Sequence[CorpusSource], out_dir: str | Path, tokenizer: Any, *,
                   base_dir: str | Path = ".", manifest_path: str | Path | None = None, allow_download: bool = False,
                   seed: int = 0, validation: float = 0.02, test: float = 0.02, dedup: str = "near",
                   near_threshold: float = 0.8, eval_files: Sequence[str | Path] = (), ngram: int = 8,
                   cache_dir: str | Path | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run the whole pipeline and write ``train.jsonl``, ``val.jsonl``,
    ``test.jsonl`` and ``corpus_manifest.json`` into ``out_dir``. Returns the
    manifest. Deterministic: the same inputs, seed and settings give
    byte-identical outputs."""
    say = log or (lambda _m: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(base_dir)
    cache = Path(cache_dir) if cache_dir else out / "_download_cache"
    for s in sources:
        s.validate()
    if not sources:
        raise CorpusError("no corpus sources")
    names = [s.source for s in sources]
    if len(set(names)) != len(names):
        raise CorpusError("corpus source names must be unique")

    grams: set[tuple[str, ...]] = set()
    eval_seen: list[dict[str, Any]] = []
    for ef in eval_files:
        from tinymind.training.data import read_jsonl
        recs = read_jsonl(ef)
        grams |= eval_ngrams(recs, ngram)
        eval_seen.append({"file": str(ef), "records": len(recs)})

    index = NearDuplicateIndex(mode=dedup, threshold=near_threshold, seed=seed)
    staged = out / "_kept.jsonl"
    group_keys: list[str] = []
    per_source: list[dict[str, Any]] = []
    decontaminated = 0
    with staged.open("w", encoding="utf-8") as kept_file:
        for src in sources:
            expected = src.sha256 or [""] * len(src.shards)
            info: dict[str, Any] = {"source": src.source, "type": src.type, "license": src.license,
                                    "description": src.description, "supplemental": src.supplemental,
                                    "weight": src.weight, "auth_env": src.auth_env or None, "shards": [],
                                    "documents": 0, "chunks": 0, "chunks_kept": 0, "short_dropped": 0}
            kept_here = 0
            # max_records is spread evenly over the source's shards (at most ceil(max_records / shards) chunks from
            # each), so a capped multi-shard corpus draws from every shard instead of only the first ones.
            quota = math.ceil(src.max_records / len(src.shards)) if src.max_records else 0

            def full() -> bool:
                return bool(src.max_records) and (kept_shard >= quota or kept_here >= src.max_records)

            for shard, want in zip(src.shards, expected):
                if src.max_records and kept_here >= src.max_records:
                    break
                kept_shard = 0
                path, digest, size = fetch_shard(shard, base_dir=base, cache_dir=cache, allow_download=allow_download,
                                                 auth_env=src.auth_env, expected_sha256=want)
                fmt = shard_format(shard, src.format)
                info["shards"].append({"shard": redact_url(shard), "sha256": digest, "bytes": size, "format": fmt,
                                       "expected_sha256": want or None, "pinned": bool(want)})
                say(f"[corpus] {src.source}: {redact_url(shard)} ({size:,} bytes, sha256 {digest[:12]}"
                    f"{', verified' if want else ', unpinned'})")
                for doc_index, paragraphs in enumerate(iter_documents(path, fmt, src.text_field)):
                    info["documents"] += 1
                    paragraphs = [p for p in paragraphs if index.new_paragraph(p)]
                    for chunk_index, chunk in enumerate(chunk_paragraphs(paragraphs, src.chunk_chars)):
                        info["chunks"] += 1
                        if full():
                            break
                        if len(chunk) < src.min_chars:
                            info["short_dropped"] += 1
                            continue
                        if index.add(chunk) != "kept":
                            continue
                        if contaminated(chunk, grams, ngram):
                            decontaminated += 1
                            continue
                        group = chunk_index // src.group_chunks if fmt == "text" else 0
                        key = f"{src.source}\x00{digest}\x00{doc_index}\x00{group}"
                        group_keys.append(key)
                        kept_file.write(json.dumps({"g": key, "id": _record_id(src.source, chunk), "text": chunk,
                                                    "category": "corpus", "source": src.source},
                                                   ensure_ascii=False) + "\n")
                        kept_here += 1
                        kept_shard += 1
                    if full():
                        break
            info["chunks_kept"] = kept_here
            per_source.append(info)

    assignment = split_groups(group_keys, validation=validation, test=test, seed=seed)
    supplemental = {s.source for s in sources if s.supplemental}
    split_stats = {k: {"file": f"{k}.jsonl", "records": 0, "tokens": 0, "bytes": 0, "natural_tokens": 0,
                       "natural_bytes": 0} for k in ("train", "val", "test")}
    source_tokens: Counter[str] = Counter()
    source_bytes: Counter[str] = Counter()
    seen_ids: set[str] = set()
    handles = {k: (out / f"{k}.jsonl").open("w", encoding="utf-8") for k in split_stats}
    try:
        with staged.open("r", encoding="utf-8") as kept_file:
            for line in kept_file:
                rec = json.loads(line)
                split = assignment[rec.pop("g")]
                if rec["id"] in seen_ids:  # identical text from two sources: exact dedup already removed it
                    continue
                seen_ids.add(rec["id"])
                n_tok = count_tokens(tokenizer, rec["text"])
                n_bytes = len(rec["text"].encode("utf-8"))
                st = split_stats[split]
                st["records"] += 1
                st["tokens"] += n_tok
                st["bytes"] += n_bytes
                if rec["source"] not in supplemental:
                    st["natural_tokens"] += n_tok
                    st["natural_bytes"] += n_bytes
                source_tokens[rec["source"]] += n_tok
                source_bytes[rec["source"]] += n_bytes
                handles[split].write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        for h in handles.values():
            h.close()
        staged.unlink(missing_ok=True)
    for info in per_source:
        info["tokens"] = source_tokens[info["source"]]
        info["bytes"] = source_bytes[info["source"]]
    for k, st in split_stats.items():
        st["sha256"] = _hash_file(out / st["file"])[0]
    if split_stats["train"]["records"] == 0:
        raise CorpusError("the corpus produced no training records (check shards, min_chars and dedup settings)")

    # Cross-split contamination: exact/near duplicates were removed corpus-wide before the split and a
    # group never spans splits; verify the exact (normalised) property on the written files.
    norm_sets: dict[str, set[bytes]] = {}
    for k in ("train", "val", "test"):
        with (out / f"{k}.jsonl").open(encoding="utf-8") as handle:
            norm_sets[k] = {hashlib.sha256(normalize(json.loads(l)["text"]).encode("utf-8")).digest()[:16]
                            for l in handle}
    overlap = {"train_val": len(norm_sets["train"] & norm_sets["val"]),
               "train_test": len(norm_sets["train"] & norm_sets["test"]),
               "val_test": len(norm_sets["val"] & norm_sets["test"])}
    total_tokens = sum(st["tokens"] for st in split_stats.values())
    natural_tokens = sum(st["natural_tokens"] for st in split_stats.values())
    manifest: dict[str, Any] = {
        "kind": CORPUS_KIND, "format_version": CORPUS_FORMAT_VERSION,
        "dataset_manifest": ({"path": str(manifest_path), "sha256": _hash_file(Path(manifest_path))[0]}
                             if manifest_path else None),
        "tokenizer": {"hash": tokenizer_hash(tokenizer), "type": tokenizer.spec().get("type"),
                      "vocab_size": tokenizer.vocab_size},
        "seed": seed, "fractions": {"validation": validation, "test": test},
        "dedup": index.report(),
        "decontamination": {"ngram": ngram, "eval_files": eval_seen, "chunks_removed": decontaminated},
        "sources": per_source,
        "splits": split_stats,
        "contamination": {"exact_overlap": overlap, "contaminated": any(overlap.values())},
        "total_tokens": total_tokens, "natural_tokens": natural_tokens,
        "natural_fraction": round(natural_tokens / total_tokens, 6) if total_tokens else 0.0,
        "corpus_sha256": hashlib.sha256("".join(split_stats[k]["sha256"] for k in ("train", "val", "test"))
                                        .encode()).hexdigest(),
    }
    if manifest["contamination"]["contaminated"]:
        raise CorpusError(f"cross-split contamination detected: {overlap}")
    (out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    say(f"[corpus] {total_tokens:,} tokens ({manifest['natural_fraction']:.1%} natural): train "
        f"{split_stats['train']['records']:,} / val {split_stats['val']['records']:,} / test "
        f"{split_stats['test']['records']:,} records; dedup removed {index.paragraphs_removed} repeated paragraphs, "
        f"{index.exact_removed} exact and {index.near_removed} near-duplicate chunks")
    return manifest


def load_corpus(corpus_dir: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Read ``corpus_manifest.json`` and (by default) verify the split files
    still match the hashes it records."""
    d = Path(corpus_dir)
    path = d / "corpus_manifest.json"
    if not path.is_file():
        raise CorpusError(f"{d} is not a prepared corpus (no corpus_manifest.json; run 'tinymind data prepare-corpus')")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != CORPUS_KIND:
        raise CorpusError(f"{path} is not a {CORPUS_KIND} manifest")
    if verify:
        for name, st in manifest["splits"].items():
            f = d / st["file"]
            if not f.is_file() or _hash_file(f)[0] != st["sha256"]:
                raise CorpusError(f"prepared corpus split {f} is missing or does not match its recorded sha256")
    return manifest
