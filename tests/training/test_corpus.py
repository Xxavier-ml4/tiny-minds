"""Real pretraining-corpus pipeline (tinymind.data.corpus + tinymind.data.external):
manifest ingestion, per-shard checksums, sharded streaming (local, file://, .gz),
deduplication, deterministic splits, decontamination, provenance, and attaching
the corpus to a v2 curriculum stage. Everything runs offline: URL handling is
exercised with file:// shards; http(s) shards must be refused without
allow_download."""
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tinymind.data import curriculum_v2 as C2
from tinymind.data.corpus import (CorpusError, CorpusSource, dedup_texts, load_corpus, prepare_corpus, redact_url,
                                  split_groups)
from tinymind.data.external import DatasetManifestError, corpus_sources, load_manifest, prepare
from tinymind.model.tokenizer import ByteTokenizer

TOK = ByteTokenizer()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prose(n: int, tag: str = "p") -> list[str]:
    return [f"In village {tag}{i} the baker lit her oven before dawn, and the miller counted {i} sacks of flour "
            f"while the river ran past the old stone mill number {i}." for i in range(n)]


class CorpusFixture(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="tm-corpus-"))
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        (self.d / "books").mkdir()
        # shard 1: plain prose, paragraphs separated by blank lines (hard-wrapped lines inside)
        paras = _prose(40, "a")
        wrapped = [p[:60] + "\n" + p[60:] for p in paras]
        (self.d / "books" / "part-000.txt").write_text("\n\n".join(wrapped) + "\n", encoding="utf-8")
        # shard 2: gzipped JSONL documents, reached through a file:// URL
        with gzip.open(self.d / "docs.jsonl.gz", "wt", encoding="utf-8") as g:
            for i in range(0, 30, 3):
                g.write(json.dumps({"text": "\n".join(_prose(3, f"b{i}_")), "meta": i}) + "\n")
        self.sources = [
            CorpusSource("books", ["books/part-000.txt"], sha256=[_sha(self.d / "books" / "part-000.txt")],
                         format="text", chunk_chars=300, group_chunks=2, license="CC0-1.0"),
            CorpusSource("docs", [(self.d / "docs.jsonl.gz").as_uri()], type="url", format="jsonl",
                         chunk_chars=400, license="ODC-BY-1.0"),
        ]

    def prepare(self, name="out", **kw):
        kw.setdefault("validation", 0.15)
        kw.setdefault("test", 0.15)
        return prepare_corpus(self.sources, self.d / name, TOK, base_dir=self.d, **kw)


class TestIngestionAndProvenance(CorpusFixture):
    def test_manifest_ingestion_records_full_provenance(self):
        manifest_file = self.d / "manifest.json"
        manifest_file.write_text(json.dumps({"version": 2, "datasets": [
            {"source": "books", "type": "local", "format": "text", "shards": ["books/part-000.txt"],
             "sha256": [_sha(self.d / "books" / "part-000.txt")], "license": "CC0-1.0", "chunk_chars": 300},
            {"source": "docs", "type": "url", "format": "jsonl", "url": (self.d / "docs.jsonl.gz").as_uri(),
             "license": "ODC-BY-1.0", "chunk_chars": 400},
            {"source": "synth", "type": "synthetic", "generator": "language", "examples": 5, "supplemental": True}]}))
        _, entries = load_manifest(manifest_file)
        sources = corpus_sources(entries)
        self.assertEqual([s.source for s in sources], ["books", "docs"])  # synthetic entries are not corpus material
        m = prepare_corpus(sources, self.d / "out", TOK, base_dir=self.d, manifest_path=manifest_file,
                           validation=0.15, test=0.15)
        self.assertEqual(m["dataset_manifest"]["sha256"], _sha(manifest_file))
        self.assertEqual(m["tokenizer"]["hash"], hashlib.sha256(json.dumps(TOK.spec(), sort_keys=True).encode()).hexdigest())
        by = {s["source"]: s for s in m["sources"]}
        self.assertEqual(by["books"]["license"], "CC0-1.0")
        self.assertTrue(by["books"]["shards"][0]["pinned"])
        self.assertEqual(by["books"]["shards"][0]["sha256"], _sha(self.d / "books" / "part-000.txt"))
        self.assertEqual(by["docs"]["shards"][0]["sha256"], _sha(self.d / "docs.jsonl.gz"))  # recorded even unpinned
        self.assertFalse(by["docs"]["shards"][0]["pinned"])
        self.assertEqual(by["docs"]["documents"], 10)
        # token counts per split add up, and every record is natural text with provenance
        self.assertEqual(m["total_tokens"], sum(s["tokens"] for s in m["splits"].values()))
        self.assertEqual(m["natural_fraction"], 1.0)
        for split in ("train", "val", "test"):
            for line in (self.d / "out" / f"{split}.jsonl").read_text().splitlines():
                rec = json.loads(line)
                self.assertEqual(set(rec), {"id", "text", "category", "source"})
                self.assertIn(rec["source"], ("books", "docs"))
        self.assertIsNotNone(load_corpus(self.d / "out"))

    def test_checksum_mismatch_is_refused(self):
        self.sources[0].sha256 = ["0" * 64]
        with self.assertRaises(CorpusError) as cm:
            self.prepare()
        self.assertIn("sha256", str(cm.exception))

    def test_network_shards_need_explicit_permission(self):
        src = [CorpusSource("web", ["https://example.com/corpus/shard-000.txt"], type="url")]
        with self.assertRaises(CorpusError) as cm:
            prepare_corpus(src, self.d / "o", TOK)
        self.assertIn("--allow-download", str(cm.exception))
        manifest = self.d / "m.json"
        manifest.write_text(json.dumps({"version": 2, "datasets": [
            {"source": "web", "type": "url", "shards": ["https://example.com/a.jsonl.gz", "https://example.com/b.jsonl.gz"]}]}))
        with self.assertRaises(DatasetManifestError):
            prepare(str(manifest), str(self.d / "p"), allow_download=False)

    def test_credentials_are_never_recorded(self):
        os.environ["TM_TEST_CORPUS_TOKEN"] = "s3cr3t-value"
        self.addCleanup(os.environ.pop, "TM_TEST_CORPUS_TOKEN", None)
        self.sources[1].auth_env = "TM_TEST_CORPUS_TOKEN"
        self.prepare()
        text = (self.d / "out" / "corpus_manifest.json").read_text()
        self.assertNotIn("s3cr3t-value", text)
        self.assertIn("TM_TEST_CORPUS_TOKEN", text)  # the variable's NAME is provenance; its value is not
        self.assertEqual(redact_url("https://user:pw@host.org/data/x.jsonl.gz?sig=SECRET#f"),
                         "https://host.org/data/x.jsonl.gz")
        with self.assertRaises(CorpusError):  # a private URL whose token variable is missing fails loudly
            prepare_corpus([CorpusSource("p", ["https://example.com/p.txt"], auth_env="TM_TEST_UNSET_VAR")],
                           self.d / "o2", TOK, allow_download=True)

    def test_http_download_through_a_redirect_with_a_token_and_retries(self):
        # The http(s) path a real run takes (e.g. Hugging Face resolve URLs answer with a redirect to a CDN), served
        # from a local server: streamed into the cache, hashed, token sent to the first host only, transient failures
        # (5xx, a body cut short of its Content-Length) retried, permanent ones (404) not.
        import contextlib
        import http.server
        import io
        import threading
        from unittest import mock
        from tinymind.data.corpus import fetch_shard

        body = (self.d / "docs.jsonl.gz").read_bytes()
        seen: list[tuple[str, str]] = []
        failures = {"/flaky": 1, "/cut": 1, "/cut-always": 99}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization") or ""))
                if self.path == "/redirect/docs.jsonl.gz":
                    self.send_response(302)
                    self.send_header("Location", "/files/docs.jsonl.gz")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path == "/files/docs.jsonl.gz" or (self.path in failures and failures[self.path] <= 0):
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/flaky":
                    failures["/flaky"] -= 1
                    self.send_error(503)
                elif self.path in ("/cut", "/cut-always"):
                    failures[self.path] -= 1
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body[:len(body) // 2])
                    self.close_connection = True
                else:
                    self.send_error(404)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        env = {"no_proxy": "127.0.0.1,localhost", "NO_PROXY": "127.0.0.1,localhost", "TM_TEST_HTTP_TOKEN": "tok-123"}
        with mock.patch.dict(os.environ, env), contextlib.redirect_stderr(io.StringIO()) as log:
            src = [CorpusSource("web", [f"{base}/redirect/docs.jsonl.gz"], type="url", format="jsonl", chunk_chars=400,
                                sha256=[_sha(self.d / "docs.jsonl.gz")], auth_env="TM_TEST_HTTP_TOKEN", license="CC0-1.0")]
            man = prepare_corpus(src, self.d / "web", TOK, allow_download=True, validation=0.15, test=0.15)
            self.assertEqual(seen, [("/redirect/docs.jsonl.gz", "Bearer tok-123"), ("/files/docs.jsonl.gz", "")])
            shard = man["sources"][0]["shards"][0]
            self.assertEqual(shard["sha256"], _sha(self.d / "docs.jsonl.gz"))
            self.assertTrue((self.d / "web" / "train.jsonl").read_text().splitlines())

            kw = dict(base_dir=self.d, cache_dir=self.d / "cache", allow_download=True, backoff_seconds=0)
            for path in ("/flaky", "/cut"):   # fails once, then succeeds on the retry
                seen.clear()
                got, digest, size = fetch_shard(base + path, **kw)
                self.assertEqual((got.read_bytes(), size), (body, len(body)), path)
                self.assertEqual([p for p, _ in seen], [path, path])
            seen.clear()
            with self.assertRaises(CorpusError) as cm:   # a short body every time: never accepted as the shard
                fetch_shard(base + "/cut-always", **kw)
            self.assertIn("after 3 attempts", str(cm.exception))
            self.assertIn("connection closed after", str(cm.exception))
            self.assertEqual(len(seen), 3)
            self.assertFalse(list((self.d / "cache").glob("*cut-always*")))   # no partial file left as the shard
            seen.clear()
            with self.assertRaises(CorpusError) as cm:   # permanent: no retry
                fetch_shard(base + "/missing.jsonl.gz", **kw)
            self.assertIn("Not Found", str(cm.exception))
            self.assertEqual(len(seen), 1)
        self.assertIn("retrying (2/3)", log.getvalue())   # each retry is announced in the job log

    def test_tampered_prepared_corpus_is_detected(self):
        self.prepare()
        f = self.d / "out" / "val.jsonl"
        f.write_text(f.read_text() + "\n")
        with self.assertRaises(CorpusError):
            load_corpus(self.d / "out")


class TestDeduplication(CorpusFixture):
    def test_exact_near_and_paragraph_duplicates(self):
        words = " ".join(f"w{j}" for j in range(200))  # long enough that one changed word is clearly "near"
        kept, rep = dedup_texts([f"Doc. {words}.", f"doc {words}",  # exact after normalisation
                                 f"Doc. {words.replace('w100 ', 'zz ')}.",  # one word changed: near duplicate
                                 "an unrelated sentence about ships and harbours at dusk"])
        self.assertEqual(kept, [0, 3])
        self.assertEqual((rep["exact_removed"], rep["near_removed"]), (1, 1))
        kept, _ = dedup_texts([f"Doc. {words}.", f"Doc. {words.replace('w100 ', 'zz ')}."], mode="exact")
        self.assertEqual(kept, [0, 1])  # exact-only mode keeps near duplicates

    def test_corpus_wide_dedup_removes_repeated_documents_and_boilerplate(self):
        boiler = "Subscribe to our newsletter to receive more stories like this one every single week."
        docs = [f"Story {i}: " + " ".join(f"tale{i}_{j}" for j in range(50)) for i in range(8)]
        with (self.d / "dup.jsonl").open("w") as f:
            for t in docs + [docs[3], docs[5]]:
                f.write(json.dumps({"text": t + "\n\n" + boiler}) + "\n")
        m = prepare_corpus([CorpusSource("dup", ["dup.jsonl"], chunk_chars=2000)], self.d / "o", TOK, base_dir=self.d)
        self.assertEqual(m["dedup"]["paragraphs_removed"], 2 + 9)  # two repeated stories + nine repeated footers
        texts = [json.loads(l)["text"] for s in ("train", "val", "test") for l in (self.d / "o" / f"{s}.jsonl").read_text().splitlines()]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(sum(boiler in t for t in texts), 1)


class TestDeterministicSplits(CorpusFixture):
    def test_same_inputs_same_bytes(self):
        a, b = self.prepare("a", seed=7), self.prepare("b", seed=7)
        for f in ("train.jsonl", "val.jsonl", "test.jsonl", "corpus_manifest.json"):
            self.assertEqual((self.d / "a" / f).read_bytes(), (self.d / "b" / f).read_bytes(), f)
        self.assertEqual(a["corpus_sha256"], b["corpus_sha256"])

    def test_seed_changes_assignment_and_splits_never_overlap(self):
        a, c = self.prepare("a", seed=1), self.prepare("c", seed=2)
        self.assertNotEqual(a["corpus_sha256"], c["corpus_sha256"])
        for m, name in ((a, "a"), (c, "c")):
            self.assertFalse(m["contamination"]["contaminated"])
            sets = {s: {json.loads(l)["text"] for l in (self.d / name / f"{s}.jsonl").read_text().splitlines()} for s in ("train", "val", "test")}
            self.assertGreater(len(sets["val"]), 0)
            self.assertGreater(len(sets["test"]), 0)
            self.assertFalse(sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
            total = sum(len(v) for v in sets.values())
            self.assertEqual(total, sum(s["records"] for s in m["splits"].values()))

    def test_split_groups_is_rank_based_and_stable(self):
        keys = [f"g{i}" for i in range(50)]
        a = split_groups(keys, validation=0.1, test=0.1, seed=3)
        self.assertEqual(a, split_groups(list(reversed(keys)), validation=0.1, test=0.1, seed=3))  # order-free
        counts = {s: sum(v == s for v in a.values()) for s in ("train", "val", "test")}
        self.assertEqual(counts, {"train": 40, "val": 5, "test": 5})
        self.assertTrue(all(v == "train" for v in split_groups(["x", "y"], validation=0.1, test=0.1, seed=0).values()))
        with self.assertRaises(CorpusError):
            split_groups(keys, validation=0.6, test=0.5, seed=0)

    def test_decontamination_against_evaluation_prompts(self):
        import random
        rng = random.Random(5)
        vocab = [f"{a}{b}" for a in ("ka", "lo", "mi", "ne", "pu", "ra", "si", "to") for b in ("n", "l", "r", "s", "m")]
        paras = [" ".join(rng.choice(vocab) for _ in range(30)) + "." for _ in range(30)]
        leaked = "Which river carries the grain barges past the seven bridges of the northern town?"
        paras[11] = paras[11] + " " + leaked
        (self.d / "varied.txt").write_text("\n\n".join(paras) + "\n")
        ev = self.d / "eval.jsonl"
        ev.write_text(json.dumps({"id": "q1", "messages": [{"role": "user", "content": leaked},
                                                           {"role": "assistant", "content": "x"}]}) + "\n")
        src = [CorpusSource("varied", ["varied.txt"], chunk_chars=300)]
        clean = prepare_corpus(src, self.d / "clean", TOK, base_dir=self.d)
        dirty = prepare_corpus(src, self.d / "dirty", TOK, base_dir=self.d, eval_files=[ev])
        self.assertEqual(clean["decontamination"]["chunks_removed"], 0)
        self.assertEqual(dirty["decontamination"]["chunks_removed"], 1)  # exactly the chunk holding the question
        texts = [json.loads(l)["text"] for s in ("train", "val", "test") for l in (self.d / "dirty" / f"{s}.jsonl").read_text().splitlines()]
        self.assertFalse(any("seven bridges" in t for t in texts))
        self.assertEqual(sum(s["records"] for s in dirty["splits"].values()) + 1,
                         sum(s["records"] for s in clean["splits"].values()))


class TestRepresentativeSampling(CorpusFixture):
    """The tokenizer input and the natural held-out sets are deterministic, uniform samples over every shard —
    never "the first N records", which only covers the first shard(s)."""

    def shards(self, n=5, per=400):
        paths = []
        for i in range(n):
            p = self.d / f"part-{i}.jsonl"
            p.write_text("".join(json.dumps({"id": f"s{i}-{j}", "text": f"shard {i} record {j}"}) + "\n"
                                 for j in range(per)))
            paths.append(p)
        return paths

    def test_sample_records_is_deterministic_uniform_and_order_free(self):
        from tinymind.data.corpus import sample_records
        paths = self.shards()
        a, seen = sample_records(paths, 500, seed=0)
        self.assertEqual(seen, 2000)
        self.assertEqual(len(a), 500)
        self.assertEqual(a, sample_records(paths, 500, seed=0)[0])  # deterministic
        self.assertEqual(a, sample_records(list(reversed(paths)), 500, seed=0)[0])  # independent of file order
        self.assertNotEqual(a, sample_records(paths, 500, seed=1)[0])
        per_shard = [sum(r["id"].startswith(f"s{i}-") for r in a) for i in range(5)]
        self.assertTrue(all(60 <= c <= 140 for c in per_shard), per_shard)  # every shard, about 100 each
        self.assertEqual(len({r["id"] for r in a}), 500)  # without replacement
        self.assertEqual(len(sample_records(paths, 10_000, seed=0)[0]), 2000)  # k larger than the data: everything
        self.assertEqual(sample_records(paths, 0, seed=0)[0], [])

    def test_tokenizer_sample_covers_every_shard_and_no_held_out_text(self):
        from tinymind.data.corpus import tokenizer_sample
        m = self.prepare("corpus")
        manifest = self.d / "ext.json"
        manifest.write_text(json.dumps({"version": 2, "datasets": [
            {"source": "books", "type": "local", "format": "text", "shards": ["books/part-000.txt"], "chunk_chars": 300},
            {"source": "synth", "type": "synthetic", "generator": "language", "examples": 30, "supplemental": True}]}))
        prepare(str(manifest), str(self.d / "external"))
        s1 = tokenizer_sample(self.d / "tok1", corpus_dir=self.d / "corpus", external_dir=self.d / "external",
                              natural_records=20, supplemental_records=10)
        s2 = tokenizer_sample(self.d / "tok2", corpus_dir=self.d / "corpus", external_dir=self.d / "external",
                              natural_records=20, supplemental_records=10)
        self.assertEqual(s1["natural"]["sha256"], s2["natural"]["sha256"])  # every job derives the same sample
        natural = [json.loads(l) for l in (self.d / "tok1" / "natural_sample.jsonl").read_text().splitlines()]
        held_out = {json.loads(l)["id"] for s in ("val", "test") for l in (self.d / "corpus" / f"{s}.jsonl").read_text().splitlines()}
        train_ids = [json.loads(l)["id"] for l in (self.d / "corpus" / "train.jsonl").read_text().splitlines()]
        self.assertFalse({r["id"] for r in natural} & held_out)  # never the corpus's held-out splits
        self.assertEqual({r["source"] for r in natural}, {"books", "docs"})  # both sources, not the first one only
        self.assertNotEqual([r["id"] for r in natural], train_ids[:20])  # not the head of the file
        self.assertEqual(s1["natural"]["corpus_sha256"], m["corpus_sha256"])
        supplemental = [json.loads(l) for l in (self.d / "tok1" / "supplemental_sample.jsonl").read_text().splitlines()]
        self.assertTrue(supplemental and all(r.get("category") != "corpus" for r in supplemental))  # synthetic only
        self.assertEqual(s1["supplemental"]["from"], [str(self.d / "external" / "synth.train.jsonl")])

    def test_natural_held_out_text_is_a_sample_shared_by_every_stage(self):
        self.prepare("corpus", validation=0.3, test=0.15)
        s1 = C2.write_stage("stage1", self.d / "s1", seed=0, scale=0.1, corpus_dir=self.d / "corpus")
        s2 = C2.write_stage("stage2", self.d / "s2", seed=5, scale=0.1, corpus_dir=self.d / "corpus")
        self.assertEqual(s1["corpus"]["val_text_sha256"], s2["corpus"]["val_text_sha256"])  # comparable across stages
        val = [json.loads(l)["id"] for l in (self.d / "corpus" / "val.jsonl").read_text().splitlines()]
        chosen = [json.loads(l)["id"] for l in (self.d / "s2" / "val_text.jsonl").read_text().splitlines()]
        self.assertEqual(len(chosen), min(len(val), 200))
        self.assertTrue(set(chosen) <= set(val))
        self.assertTrue((self.d / "s1" / "corpus_manifest.json").is_file())  # provenance travels with the stage data

    def test_prepare_corpus_max_records_draws_from_every_shard(self):
        for i in range(3):
            (self.d / f"q{i}.txt").write_text("\n\n".join(_prose(30, f"quota{i}_")) + "\n")
        m = prepare_corpus([CorpusSource("q", [f"q{i}.txt" for i in range(3)], chunk_chars=200, max_records=9)],
                           self.d / "q", TOK, base_dir=self.d)
        self.assertEqual(m["sources"][0]["chunks_kept"], 9)
        self.assertEqual(len(m["sources"][0]["shards"]), 3)
        texts = [json.loads(l)["text"] for s in ("train", "val", "test") for l in (self.d / "q" / f"{s}.jsonl").read_text().splitlines()]
        self.assertEqual([sum(f"quota{i}_" in t for t in texts) for i in range(3)], [3, 3, 3])


class TestExternalPrepare(CorpusFixture):
    def test_sharded_entries_are_streamed_and_chunked(self):
        manifest = self.d / "m.json"
        manifest.write_text(json.dumps({"version": 2, "datasets": [
            {"source": "books", "type": "local", "format": "text", "shards": ["books/part-000.txt"],
             "sha256": [_sha(self.d / "books" / "part-000.txt")], "chunk_chars": 300, "license": "CC0-1.0"},
            {"source": "docs", "type": "url", "shards": [(self.d / "docs.jsonl.gz").as_uri()], "chunk_chars": 400}]}))
        m = prepare(str(manifest), str(self.d / "p"))
        by = {e["source"]: e for e in m["entries"]}
        self.assertTrue(by["books"]["natural"] and by["docs"]["natural"])
        self.assertGreater(by["books"]["examples"], 1)
        self.assertEqual(by["books"]["shards"][0]["sha256"], _sha(self.d / "books" / "part-000.txt"))
        self.assertEqual(m["natural_tokens"], m["total_tokens"])
        first = json.loads((self.d / "p" / "books.train.jsonl").read_text().splitlines()[0])
        self.assertLessEqual(len(first["text"]), 300)
        self.assertNotIn("\n", first["text"].replace("\n\n", ""))  # hard-wrapped lines joined inside paragraphs

    def test_max_records_is_spread_evenly_over_the_shards(self):
        for i in range(3):
            (self.d / f"s{i}.txt").write_text("\n\n".join(_prose(20, f"shard{i}_")) + "\n")
        manifest = self.d / "cap.json"
        manifest.write_text(json.dumps({"version": 2, "datasets": [
            {"source": "capped", "type": "local", "format": "text", "chunk_chars": 200, "max_records": 6,
             "shards": ["s0.txt", "s1.txt", "s2.txt"]}]}))
        m = prepare(str(manifest), str(self.d / "cap"))
        self.assertEqual(m["entries"][0]["examples"], 6)
        self.assertEqual(len(m["entries"][0]["shards"]), 3)  # every shard is read, not only the first
        texts = [json.loads(l)["text"] for l in (self.d / "cap" / "capped.train.jsonl").read_text().splitlines()]
        self.assertEqual([sum(f"shard{i}_" in t for t in texts) for i in range(3)], [2, 2, 2])

    def test_entry_validation(self):
        bad = [{"source": "x", "type": "local"},  # no path/url/shards
               {"source": "x", "type": "url", "shards": ["a", "b"], "sha256": "abc"},  # one hash for two shards
               {"source": "x", "type": "url", "shards": ["a"], "sha256": ["1", "2"]},
               {"source": "x", "type": "local", "path": "a.txt", "format": "csv"},
               {"source": "x", "type": "url", "url": "https://e.x/a", "auth_env": "not a name"}]
        for raw in bad:
            m = self.d / "bad.json"
            m.write_text(json.dumps({"datasets": [raw]}))
            with self.assertRaises(DatasetManifestError, msg=str(raw)):
                load_manifest(m)


class TestCurriculumAttachment(CorpusFixture):
    def test_stage1_trains_primarily_on_natural_text(self):
        self.prepare("corpus", validation=0.15, test=0.15)
        m = C2.write_stage("stage1", self.d / "s1", seed=0, scale=0.1, corpus_dir=self.d / "corpus")
        corpus = m["corpus"]
        self.assertTrue(corpus["used"])
        self.assertIn("corpus", m["mixture"])
        self.assertAlmostEqual(sum(m["mixture"].values()), 1.0, places=5)
        spec = C2.load_manifest()["stages"]["stage1"]["corpus"]
        self.assertAlmostEqual(corpus["expected_byte_share"], spec["byte_share"], places=3)  # natural text is primary
        self.assertGreater(corpus["expected_byte_share"], 0.5)
        for f in ("train_corpus.jsonl", "val_text.jsonl", "lexicon.txt"):
            self.assertTrue((self.d / "s1" / f).is_file(), f)
        self.assertEqual(corpus["natural_train_bytes"],
                         sum(len(json.loads(l)["text"].encode()) for l in (self.d / "s1" / "train_corpus.jsonl").read_text().splitlines()))
        self.assertIn("baker", (self.d / "s1" / "lexicon.txt").read_text().split())
        # later stages replay a little natural text, so language stays measurable
        m2 = C2.write_stage("stage2", self.d / "s2", seed=0, scale=0.1, corpus_dir=self.d / "corpus")
        self.assertLess(m2["corpus"]["expected_byte_share"], 0.2)
        # without --corpus nothing changes
        plain = C2.write_stage("stage1", self.d / "plain", seed=0, scale=0.1)
        self.assertNotIn("corpus", plain)
        self.assertNotIn("corpus", plain["mixture"])


if __name__ == "__main__":
    unittest.main()
