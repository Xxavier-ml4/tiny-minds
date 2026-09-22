"""Byte tokenizer vs small learned BPE vocabularies, measured on the curriculum (brief sections 2, 10).

    python benchmarks/tokenizer_compare.py --profile tiny_mobile --stage stage2 --out docs/benchmarks/tokenizer-compare.json

BPE merges are learned from the *training* text only; every length below is measured on the held-out *validation*
text, rendered through the real chat template. For each vocabulary it reports the parameter cost (exact, from the
model config: the embedding table, tied to the output head), tokens per character, the token length of a rendered
training example (mean / p95 / max) and how many training examples would exceed the 256-token context, and an
attention-cost proxy (mean of length^2 relative to the byte tokenizer — the quadratic term that dominated the
training profile).

What it does NOT measure: training speed and quality with a BPE model — see the implementation report for what
was and was not run. The native runtime has no BPE reader, so BPE is not a deployable candidate yet.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinymind.data.curriculum import build_stage  # noqa: E402
from tinymind.data.render import ChatRenderer  # noqa: E402
from tinymind.model.bpe import BPETokenizer  # noqa: E402
from tinymind.model.config import ModelConfig, count_parameters  # noqa: E402
from tinymind.model.tokenizer import ByteTokenizer  # noqa: E402


def texts_of(records):
    out = []
    for r in records:
        out += [r["text"]] if "text" in r else [m["content"] for m in r["messages"]]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="tiny_mobile")
    p.add_argument("--stage", default="stage2")
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--vocabs", type=int, nargs="+", default=[384, 512, 1024, 2048])
    p.add_argument("--out")
    args = p.parse_args()
    base = ModelConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / f"{args.profile}.yaml")
    data = build_stage(args.stage, seed=0, scale=args.scale)
    train_records = [r for rs in data["train"].values() for r in rs]
    val_records = data["val"]
    chars = sum(len(t) for t in texts_of(val_records))
    train_texts = texts_of(train_records)
    rows = []
    byte_mean_sq = None
    for label, tok in [("byte (260)", ByteTokenizer())] + [(f"bpe-{v}", BPETokenizer.train(train_texts, v)) for v in args.vocabs]:
        r = ChatRenderer(tok)
        lens = np.array([len(r.render(x)) for x in val_records])
        train_lens = np.array([len(r.render(x)) for x in train_records[:4000]])
        text_tokens = sum(len(tok.encode(t)) for t in texts_of(val_records))
        cfg = dataclasses.replace(base, vocab_size=tok.vocab_size)
        mean_sq = float((lens.astype(float) ** 2).mean())
        byte_mean_sq = byte_mean_sq or mean_sq
        rows.append({"tokenizer": label, "vocab_size": tok.vocab_size, "embedding_parameters": tok.vocab_size * base.hidden_size,
                     "model_parameters": count_parameters(cfg), "tokens_per_char_of_message_text": round(text_tokens / chars, 4),
                     "rendered_example_tokens": {"mean": round(float(lens.mean()), 1), "p95": int(np.percentile(lens, 95)), "max": int(lens.max())},
                     "train_examples_over_256_tokens": f"{int((train_lens > 256).sum())}/{len(train_lens)}",
                     "attention_cost_proxy_vs_byte(mean len^2)": round(mean_sq / byte_mean_sq, 3),
                     "sequence_shrink_vs_byte": round(float(np.mean(rows[0]["_lens"]) / lens.mean()), 2) if rows else 1.0,
                     "_lens": lens.tolist() if not rows else None})
    for r in rows:
        r.pop("_lens", None)
    result = {"profile": args.profile, "stage": args.stage, "validation_examples": len(val_records), "validation_message_chars": chars,
              "rows": rows, "note": "training speed / validation loss with a BPE model were not measured by this script"}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
