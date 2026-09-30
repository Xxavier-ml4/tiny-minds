"""External dataset support for v2 (brief section 13, revised for real pretraining data).

TinyMind's synthetic generators are honest about being generators: they show
whether the pipeline learns a behaviour, not that the model has broad language
or world knowledge. Stage 1 (language) trains primarily on a *real* natural-text
corpus supplied through a manifest — never committed into the repository — and
uses the synthetic sources as a supplement.

A manifest (``datasets/v2/manifest.json``) lists dataset entries:

    source        a stable id
    type          "local" | "url" | "synthetic"
    path          one local file (legacy single-file entries) — or use ``shards``
    url           one URL (http(s):// needs --allow-download; file:// works offline)
    shards        list of shard paths/URLs (large corpora are sharded and streamed)
    sha256        expected SHA-256: one string for a single file/shard, or a list (one per shard);
                  verified when present. Every prepared shard's observed hash is recorded.
    format        "text" (plain prose) | "jsonl" (records with a text field); inferred from the
                  extension when empty; ``.gz`` shards are decompressed
    text_field    JSONL field with the document text (default "text")
    auth_env      environment variable holding a bearer token for a private URL (never logged)
    chunk_chars / min_chars / group_chunks / max_records   chunking controls (see tinymind.data.corpus)
    supplemental  true for non-natural text (the synthetic entries are supplemental by definition)
    split         "train" | "val" | "test" (for ``prepare``; ``prepare_corpus`` makes its own splits)
    weight        relative sampling weight (advisory)
    license       the data's license (recorded in the prepared manifest)
    description   human note
    generator / examples / seed   synthetic entries only

Two preparation paths share this manifest:

* :func:`prepare` (``tinymind data prepare-external``) materialises EVERY entry
  as JSONL under an output directory — synthetic entries are generated
  deterministically, local/url entries are streamed shard by shard, verified and
  chunked into ``{"id", "text"}`` records (a legacy single local JSONL file is
  copied verbatim). Used, for example, as tokenizer-training input.
* :func:`tinymind.data.corpus.prepare_corpus` (``tinymind data prepare-corpus``)
  turns the local/url entries into the deduplicated, split, decontaminated
  Stage-1 pretraining corpus with full provenance.

Nothing downloads unless explicitly allowed (``allow_download=True``); a network
entry without it is refused loudly rather than silently skipped, so CI stays
hermetic. Every prepared entry records its ``type`` and whether it is natural
text, so downstream code and humans can tell a real corpus from a generator.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

_VALID_TYPES = ("local", "synthetic", "url")
_VALID_SPLITS = ("train", "val", "test")
_VALID_FORMATS = ("", "text", "jsonl")


class DatasetManifestError(ValueError):
    pass


@dataclasses.dataclass
class DatasetEntry:
    source: str
    type: str
    path: str = ""
    split: str = "train"
    weight: float = 1.0
    sha256: str | list[str] = ""
    license: str = ""
    description: str = ""
    # local / url corpus entries
    url: str = ""
    shards: list[str] = dataclasses.field(default_factory=list)
    format: str = ""
    text_field: str = "text"
    auth_env: str = ""
    chunk_chars: int = 1000
    min_chars: int = 32
    group_chunks: int = 8
    max_records: int = 0
    supplemental: bool = False
    # Only for type == "synthetic": which generator to use and its size.
    generator: str = ""
    examples: int = 0
    seed: int = 0

    def shard_list(self) -> list[str]:
        if self.shards:
            return [str(s) for s in self.shards]
        if self.url:
            return [self.url]
        return [self.path] if self.path else []

    def expected_hashes(self) -> list[str]:
        """One expected sha256 per shard ("" = not pinned)."""
        shards = self.shard_list()
        if isinstance(self.sha256, list):
            return [str(h) for h in self.sha256]
        if self.sha256 and len(shards) == 1:
            return [self.sha256]
        return [""] * len(shards)

    @property
    def natural(self) -> bool:
        return self.type in ("local", "url") and not self.supplemental

    def validate(self) -> None:
        if self.type not in _VALID_TYPES:
            raise DatasetManifestError(f"entry {self.source!r}: type must be one of {_VALID_TYPES}, got {self.type!r}")
        if self.split not in _VALID_SPLITS:
            raise DatasetManifestError(f"entry {self.source!r}: split must be one of {_VALID_SPLITS}, got {self.split!r}")
        if self.weight < 0:
            raise DatasetManifestError(f"entry {self.source!r}: weight must be >= 0")
        if self.type == "synthetic" and not self.generator:
            raise DatasetManifestError(f"entry {self.source!r}: a synthetic entry needs a 'generator'")
        if self.format not in _VALID_FORMATS:
            raise DatasetManifestError(f"entry {self.source!r}: format must be 'text' or 'jsonl'")
        if self.type in ("local", "url"):
            shards = self.shard_list()
            if not shards:
                raise DatasetManifestError(f"entry {self.source!r}: a {self.type} entry needs 'path', 'url' or 'shards'")
            hashes = self.expected_hashes()
            if isinstance(self.sha256, list) and len(hashes) != len(shards):
                raise DatasetManifestError(f"entry {self.source!r}: {len(hashes)} sha256 values for {len(shards)} shards")
            if isinstance(self.sha256, str) and self.sha256 and len(shards) > 1:
                raise DatasetManifestError(f"entry {self.source!r}: give one sha256 per shard (a list) for a "
                                           "sharded entry")
        if self.type in ("local", "url"):
            try:
                self.corpus_source().validate()
            except ValueError as exc:
                raise DatasetManifestError(str(exc)) from exc

    def corpus_source(self):
        """This entry as a :class:`tinymind.data.corpus.CorpusSource`."""
        from tinymind.data.corpus import CorpusSource
        hashes = self.expected_hashes()
        return CorpusSource(source=self.source, shards=self.shard_list(), type=self.type,
                            sha256=hashes if any(hashes) else [], format=self.format, text_field=self.text_field,
                            auth_env=self.auth_env, license=self.license, description=self.description,
                            weight=self.weight, chunk_chars=self.chunk_chars, min_chars=self.min_chars,
                            group_chunks=self.group_chunks, max_records=self.max_records,
                            supplemental=self.supplemental)


def load_manifest(path: str | Path) -> tuple[dict[str, Any], list[DatasetEntry]]:
    """``(header, entries)`` from a dataset manifest file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetManifestError(f"cannot read dataset manifest {path}: {exc}") from exc
    raw_entries = data.get("datasets", data.get("entries", []))
    entries = []
    known = {f.name for f in dataclasses.fields(DatasetEntry)}
    for raw in raw_entries:
        unknown = set(raw) - known
        if unknown:
            raise DatasetManifestError(f"unknown dataset entry field(s) {sorted(unknown)}")
        entry = DatasetEntry(**raw)
        entry.validate()
        entries.append(entry)
    header = {k: v for k, v in data.items() if k not in ("datasets", "entries")}
    return header, entries


def corpus_sources(entries: list[DatasetEntry]):
    """The manifest's real-corpus sources (``local``/``url`` entries), in order."""
    return [e.corpus_source() for e in entries if e.type in ("local", "url")]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _synthesize(entry: DatasetEntry, out_file: Path) -> int:
    """Deterministically generate a synthetic text corpus for a ``synthetic``
    entry, using the v2 curriculum generators. Kept deterministic (fixed seed)
    so the prepared file's hash is reproducible."""
    from tinymind.data import curriculum_v2 as C2

    if entry.generator not in C2.SOURCES:
        raise DatasetManifestError(f"entry {entry.source!r}: unknown generator {entry.generator!r}; "
                                   f"known: {sorted(C2.SOURCES)}")
    records = C2.generate_source(entry.generator, max(1, entry.examples), entry.split, entry.seed)
    with out_file.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    return len(records)


def _is_legacy_jsonl(entry: DatasetEntry) -> bool:
    """A single local JSONL file without chunking needs: copied verbatim, as the
    original single-file ``local`` entries were."""
    from tinymind.data.corpus import shard_format
    shards = entry.shard_list()
    return entry.type == "local" and len(shards) == 1 and not shards[0].lower().endswith(".gz") \
        and shard_format(shards[0], entry.format) == "jsonl"


def _count(path: Path, tokenizer: Any) -> tuple[int, int, int]:
    """``(records, tokens, bytes)`` of the text in a prepared JSONL file."""
    from tinymind.data.contamination import texts
    records = tokens = size = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            _, full = texts(json.loads(line))
            records += 1
            tokens += len(tokenizer.encode(full))
            size += len(full.encode("utf-8"))
    return records, tokens, size


def prepare(manifest_path: str | Path, out_dir: str | Path, allow_download: bool = False,
            tokenizer: Any | None = None, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Prepare every entry into ``out_dir`` and return a prepared-manifest dict.

    * ``synthetic`` — generated deterministically here (supplemental data).
    * ``local`` / ``url`` — each shard is streamed (``file://`` and local paths in
      place; ``http(s)`` only with ``allow_download=True``), verified against its
      sha256 when one is given, and chunked into ``{"id", "text"}`` records. A
      single local JSONL file is copied verbatim and verified as before.

    Records per entry: examples, tokens (counted with ``tokenizer``; the byte
    tokenizer by default), bytes, whether it is natural text, and per-shard
    provenance. The header records the dataset manifest's sha256 and the
    counting tokenizer's hash.
    """
    from tinymind.data.corpus import (CorpusError, chunk_paragraphs, fetch_shard, iter_documents, redact_url,
                                      shard_format, tokenizer_hash)
    from tinymind.model.tokenizer import ByteTokenizer

    tokenizer = tokenizer or ByteTokenizer()
    header, entries = load_manifest(manifest_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_root = Path(manifest_path).resolve().parent
    cache = Path(cache_dir) if cache_dir else out / "_download_cache"
    prepared: list[dict[str, Any]] = []
    for entry in entries:
        dest = out / f"{entry.source}.{entry.split}.jsonl"
        shards_info: list[dict[str, Any]] = []
        try:
            if entry.type == "synthetic":
                _synthesize(entry, dest)
            elif _is_legacy_jsonl(entry):
                shard = entry.shard_list()[0]
                src, digest, size = fetch_shard(shard, base_dir=manifest_root, cache_dir=cache,
                                                allow_download=allow_download, expected_sha256="")
                dest.write_bytes(src.read_bytes())
                shards_info.append({"shard": redact_url(shard), "sha256": digest, "bytes": size})
            else:
                # max_records caps the whole entry and is spread evenly over its shards (as in prepare_corpus):
                # at most ceil(max_records / shards) chunks from each shard, so every shard is represented.
                shards = entry.shard_list()
                quota = -(-entry.max_records // len(shards)) if entry.max_records else 0
                kept = 0
                with dest.open("w", encoding="utf-8") as handle:
                    for shard, want in zip(shards, entry.expected_hashes()):
                        if entry.max_records and kept >= entry.max_records:
                            break
                        kept_shard = 0
                        src, digest, size = fetch_shard(shard, base_dir=manifest_root,
                                                        cache_dir=cache,
                                                        allow_download=allow_download, auth_env=entry.auth_env,
                                                        expected_sha256=want)
                        shards_info.append({"shard": redact_url(shard), "sha256": digest, "bytes": size,
                                            "pinned": bool(want)})
                        fmt = shard_format(shard, entry.format)
                        for paragraphs in iter_documents(src, fmt, entry.text_field):
                            for chunk in chunk_paragraphs(paragraphs, entry.chunk_chars):
                                if len(chunk) < entry.min_chars:
                                    continue
                                cid = f"{entry.source}-{hashlib.sha256(chunk.encode('utf-8')).hexdigest()[:16]}"
                                handle.write(json.dumps({"id": cid, "text": chunk, "category": "corpus",
                                                         "source": entry.source}, ensure_ascii=False) + "\n")
                                kept += 1
                                kept_shard += 1
                                if entry.max_records and (kept_shard >= quota or kept >= entry.max_records):
                                    break
                            if entry.max_records and (kept_shard >= quota or kept >= entry.max_records):
                                break
        except CorpusError as exc:
            dest.unlink(missing_ok=True)
            raise DatasetManifestError(f"entry {entry.source!r}: {exc}") from exc
        got = _sha256_file(dest)
        if _is_legacy_jsonl(entry) and isinstance(entry.sha256, str) and entry.sha256 and got != entry.sha256:
            raise DatasetManifestError(
                f"entry {entry.source!r}: prepared file sha256 {got} != expected {entry.sha256} "
                "(the corpus changed; update the manifest deliberately)")
        if entry.type == "synthetic" and isinstance(entry.sha256, str) and entry.sha256 and got != entry.sha256:
            raise DatasetManifestError(
                f"entry {entry.source!r}: prepared file sha256 {got} != expected {entry.sha256} "
                "(the generator output changed; update the manifest deliberately)")
        n, tokens, size = _count(dest, tokenizer)
        prepared.append({"source": entry.source, "type": entry.type, "split": entry.split, "file": dest.name,
                         "examples": n, "tokens": tokens, "bytes": size, "sha256": got, "weight": entry.weight,
                         "natural": entry.natural, "supplemental": entry.supplemental or entry.type == "synthetic",
                         "license": entry.license, "description": entry.description, "shards": shards_info})
    total = sum(e["tokens"] for e in prepared)
    natural = sum(e["tokens"] for e in prepared if e["natural"])
    out_manifest = {"prepared_from": str(Path(manifest_path)), "dataset_manifest_sha256": _sha256_file(Path(manifest_path)),
                    "tokenizer_hash": tokenizer_hash(tokenizer), "total_tokens": total, "natural_tokens": natural,
                    "header": header, "entries": prepared}
    (out / "prepared_manifest.json").write_text(json.dumps(out_manifest, indent=2, sort_keys=True) + "\n")
    return out_manifest
