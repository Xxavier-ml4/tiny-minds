"""The tiny-model evaluation suite (brief section 30).

Input: an inference package (or any object with ``model``/``tokenizer``/
``renderer``/``generate_ids``) and held-out records — normally
``eval.jsonl`` from ``tinymind data build-curriculum``. Output: one JSON-able
dict with **separate** metrics; no composite "quality score" is computed (brief
section 32: a single number would hide the trade-offs a person has to weigh).

Reported, each on held-out data:

* ``loss``: next-token loss on the validation records and on the eval records
  (per category), teacher-forced, with the training loss mask;
* ``capabilities``: for every category with a scorer, ``n`` and ``accuracy``,
  split into ``paraphrase`` (phrasings never trained on) and ``novel`` where
  the record says which it is;
* ``tool_behavior`` (section 58): correct / wrong / missing / malformed tool
  calls, argument accuracy, false-positive calls when none was needed,
  clarification on missing arguments, follow-up completion, tool-result
  incorporation — each its own number;
* ``generation``: mean response length, share of responses that never emitted
  EOS within the budget, and character-n-gram repetition;
* ``timing``: wall-clock generation throughput of the *Python reference
  runtime* on this machine (a native/mobile figure is measured separately).

Generation is greedy, so a result is reproducible for a given model.
"""
from __future__ import annotations

import collections
import math
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from tinymind.evaluation.scoring import looks_like_call, parse_tool_call, repetition, score
from tinymind.model.tensor import no_grad

SUITE_VERSION = 1
_TOOL_SCORERS = ("tool_call", "tool_name")


def _mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def teacher_forced_loss(pkg: Any, records: Iterable[dict[str, Any]], batch_size: int = 8) -> dict[str, Any]:
    """Token-mean loss over the training loss mask (``val_loss``) plus per-category means."""
    from tinymind.training.data import TokenizedDataset, sequential_batches  # evaluation may use training helpers; inference may not

    records = list(records)
    ds = TokenizedDataset.from_records(records, pkg.renderer, pkg.model.config.max_seq_len, overflow="drop", name="eval-loss")
    total, tokens = 0.0, 0
    per_cat: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0])
    with no_grad():
        for cat in sorted({e.category or "(none)" for e in ds.examples}):
            group = [e for e in ds.examples if (e.category or "(none)") == cat]
            sub = type(ds)(group, pkg.renderer, name=cat)
            s, t = 0.0, 0
            for b in sequential_batches(sub, batch_size, pkg.model.config.max_seq_len, pkg.tokenizer.pad_token_id):
                out = pkg.model(b.input_ids, labels=b.labels, loss_normalizer=1.0)
                s += float(out.loss.item())
                t += b.num_loss_tokens
            per_cat[cat] = [s, t]
            total += s
            tokens += t
    loss = total / tokens if tokens else None
    return {"loss": loss, "perplexity": math.exp(min(loss, 30.0)) if loss is not None else None, "loss_tokens": tokens,
            "by_category": {c: (s / t if t else None) for c, (s, t) in per_cat.items()}, "dropped_too_long": ds.dropped.get("too_long", 0)}


def run_suite(pkg: Any, eval_records: Sequence[dict[str, Any]], val_records: Sequence[dict[str, Any]] | None = None, *,
              max_new_tokens: int = 96, limit_per_category: int | None = None,
              progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    t_start = time.perf_counter()
    result: dict[str, Any] = {"suite_version": SUITE_VERSION, "model": {
        "parameters": pkg.model.count_parameters(), "config_hash": pkg.model.config.stable_hash(),
        "provenance": pkg.manifest.get("provenance") if getattr(pkg, "manifest", None) else None}}
    result["loss"] = {}
    if val_records:
        result["loss"]["validation"] = teacher_forced_loss(pkg, val_records)
    result["loss"]["eval"] = teacher_forced_loss(pkg, eval_records)

    graded = [r for r in eval_records if "score" in r.get("meta", {}) and r.get("messages")]
    if limit_per_category:
        seen: dict[str, int] = collections.Counter()
        picked = []
        for r in graded:
            if seen[r["category"]] < limit_per_category:
                seen[r["category"]] += 1
                picked.append(r)
        graded = picked

    rows: list[dict[str, Any]] = []
    gen_seconds, gen_tokens = 0.0, 0
    skipped = 0
    max_ctx = pkg.model.config.max_seq_len
    for i, rec in enumerate(graded):
        prompt = rec["messages"][:-1]
        tools = rec.get("tools", [])
        if len(pkg.prompt_ids(prompt, tools)) >= max_ctx - 2:
            skipped += 1
            continue
        t0 = time.perf_counter()
        new_ids, reason = pkg.generate_ids(prompt, tools=tools, max_new_tokens=max_new_tokens)
        gen_seconds += time.perf_counter() - t0
        gen_tokens += len(new_ids)
        text = pkg.renderer.decode_completion(new_ids)
        res = score(text, rec["meta"])
        rows.append({"id": rec["id"], "category": rec["category"], "sub": rec["meta"].get("sub"), "scorer": rec["meta"]["score"],
                     "paraphrase": rec["meta"].get("paraphrase"), "ok": bool(res["ok"]), "detail": res,
                     "text": text, "tokens": len(new_ids), "finish": reason})
        if progress and (i + 1) % 100 == 0:
            progress(f"[eval] {i + 1}/{len(graded)} generated")

    caps: dict[str, Any] = {}
    for cat in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == cat]
        entry: dict[str, Any] = {"n": len(group), "accuracy": _rate(sum(r["ok"] for r in group), len(group))}
        para = [r for r in group if r["paraphrase"] is True]
        nov = [r for r in group if r["paraphrase"] is False]
        if para and nov:
            entry["paraphrase_accuracy"] = _rate(sum(r["ok"] for r in para), len(para))
            entry["seen_phrasing_accuracy"] = _rate(sum(r["ok"] for r in nov), len(nov))
        caps[cat] = entry
    result["capabilities"] = caps

    # ---- tool behaviour, each measure separate --------------------------------------------------------------
    need_tool = [r for r in rows if r["scorer"] in _TOOL_SCORERS]
    n = len(need_tool)
    no_tool = [r for r in rows if r["scorer"] not in _TOOL_SCORERS]
    by_cat = lambda c: [r for r in rows if r["category"] == c]  # noqa: E731
    result["tool_behavior"] = {
        "requests_needing_a_tool": n,
        "correct_tool_rate": _rate(sum(r["detail"].get("name_ok", False) for r in need_tool), n),
        "argument_accuracy": _rate(sum(r["ok"] for r in need_tool if r["scorer"] == "tool_call"), sum(1 for r in need_tool if r["scorer"] == "tool_call")),
        "wrong_tool_rate": _rate(sum(1 for r in need_tool if r["detail"].get("status") == "ok" and not r["detail"].get("name_ok")), n),
        "no_call_rate": _rate(sum(1 for r in need_tool if r["detail"].get("status") == "none"), n),
        "malformed_call_rate": _rate(sum(1 for r in need_tool if r["detail"].get("status") == "malformed"), n),
        "requests_needing_no_tool": len(no_tool),
        "false_positive_call_rate": _rate(sum(looks_like_call(r["text"]) for r in no_tool), len(no_tool)),
        "clarification_on_missing_argument_accuracy": _acc(by_cat("clarification")),
        "clarification_followup_call_accuracy": _acc(by_cat("clarification_followup")),
        "tool_result_incorporation_accuracy": _acc(by_cat("tool_result")),
        "refusal_accuracy": _acc(by_cat("refusal")),
    }

    lens = [r["tokens"] for r in rows]
    result["generation"] = {
        "graded_examples": len(rows), "skipped_prompt_too_long": skipped, "max_new_tokens": max_new_tokens,
        "mean_response_tokens": _mean(lens), "mean_response_chars": _mean([len(r["text"]) for r in rows]),
        "unterminated_rate": _rate(sum(r["finish"] == "length" for r in rows), len(rows)),
        "mean_char6gram_repetition": _mean([repetition(r["text"]) for r in rows]),
        "looping_rate_repetition_over_0.5": _rate(sum(repetition(r["text"]) > 0.5 for r in rows), len(rows)),
    }
    result["timing"] = {"generated_tokens": gen_tokens, "generation_seconds": round(gen_seconds, 3),
                        "tokens_per_second_python_runtime": round(gen_tokens / gen_seconds, 2) if gen_seconds else None,
                        "suite_seconds": round(time.perf_counter() - t_start, 2)}
    result["failures_sample"] = [{k: r[k] for k in ("id", "category", "text")} for r in rows if not r["ok"]][:25]
    result["_rows"] = rows  # dropped by ``strip_rows`` before writing unless per-example output was requested
    return result


def _acc(group: list[dict[str, Any]]) -> float | None:
    return _rate(sum(r["ok"] for r in group), len(group))


def strip_rows(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "_rows"}


def format_report(result: dict[str, Any]) -> str:
    """Human-readable table of the machine-readable result (separate metrics, no aggregate)."""
    def f(x: Any) -> str:
        return "  n/a " if x is None else (f"{x:6.3f}" if isinstance(x, float) else f"{x:>6}")
    lines = [f"model: {result['model']['parameters']:,} parameters"]
    for name, blk in result["loss"].items():
        lines.append(f"loss[{name}]: {f(blk['loss'])}  ppl {f(blk['perplexity'])}  ({blk['loss_tokens']} loss tokens)")
    lines.append("capability accuracy (held-out):")
    for cat, e in result["capabilities"].items():
        extra = f"   paraphrase {f(e.get('paraphrase_accuracy'))}  seen-phrasing {f(e.get('seen_phrasing_accuracy'))}" if "paraphrase_accuracy" in e else ""
        lines.append(f"  {cat:24s} n={e['n']:4d}  acc {f(e['accuracy'])}{extra}")
    lines.append("tool behaviour:")
    for k, v in result["tool_behavior"].items():
        lines.append(f"  {k:46s} {f(v)}")
    g = result["generation"]
    lines.append(f"generation: {g['mean_response_tokens']:.1f} tokens/response, unterminated {f(g['unterminated_rate'])}, "
                 f"repetition {f(g['mean_char6gram_repetition'])}, looping {f(g['looping_rate_repetition_over_0.5'])}")
    t = result["timing"]
    lines.append(f"python-runtime generation: {t['tokens_per_second_python_runtime']} tokens/s over {t['generated_tokens']} tokens")
    return "\n".join(lines)


def compare(a: dict[str, Any], b: dict[str, Any], labels: tuple[str, str] = ("A", "B")) -> list[dict[str, Any]]:
    """Side-by-side of every scalar metric two results share (FP32 vs INT8, stage N vs N+1). Returns rows
    ``{"metric","A","B","delta"}``; nothing is aggregated."""
    def flat(d: Any, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if k in ("model", "failures_sample", "_rows", "provenance", "suite_version"):
                    continue
                out.update(flat(v, f"{prefix}{k}."))
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            out[prefix[:-1]] = float(d)
        return out
    fa, fb = flat(a), flat(b)
    return [{"metric": k, labels[0]: fa[k], labels[1]: fb[k], "delta": fb[k] - fa[k]} for k in fa if k in fb]
