"""Where does one training step's time actually go?

Works against any tree that has ``tinymind.model.tensor.Tensor`` and
``tinymind.model.module.Module`` (both the Phase 3A tree and the Phase 3B
tree), so the same script produces the "before" and "after" numbers:

    python benchmarks/audit/profile_training_step.py --tree <path> [--hidden 128 ...]

It does not change what the model computes: the backward pass is re-driven
here only to put a stopwatch around each node's local gradient rule.

Categories (see ``_categorize``): the module that was executing when a
tensor was created decides which bucket its backward closure is charged to.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import resource
import statistics
import sys
import time
from pathlib import Path


def _install_tree(tree: str) -> None:
    sys.path.insert(0, str(Path(tree).resolve()))


def _topo(root):
    """Iterative topological order (same order the engine's recursive DFS gives)."""
    order, seen, stack = [], set(), [(root, iter(root._prev))]
    seen.add(id(root))
    while stack:
        node, it = stack[-1]
        for child in it:
            if id(child) not in seen:
                seen.add(id(child))
                stack.append((child, iter(child._prev)))
                break
        else:
            order.append(node)
            stack.pop()
    return order


def _categorize(stack: list[str], op: str) -> str:
    inner = stack[-1] if stack else ""
    if "CausalSelfAttention" in stack:
        return "attention.linear" if inner == "Linear" else "attention.core(rope/scores/softmax/ctx)"
    if "SwiGLUMLP" in stack:
        return "mlp.linear" if inner == "Linear" else "mlp.elementwise(silu/gate)"
    if inner == "RMSNorm" or "RMSNorm" in stack:
        return "norm"
    if op in ("cross_entropy",):
        return "loss"
    if op in ("getitem",) and not stack:
        return "embedding"
    return "other(embed/lm_head/residual)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--intermediate", type=int, default=384)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--vocab", type=int, default=260)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--json", action="store_true", help="print JSON only")
    args = p.parse_args()
    _install_tree(args.tree)

    import numpy as np
    from tinymind.model import ModelConfig, TinyMindTransformer
    from tinymind.model import tensor as tensor_mod
    from tinymind.model.module import Module
    from tinymind.model.optim import AdamW
    from tinymind.model.tensor import Tensor

    cfg = ModelConfig(hidden_size=args.hidden, num_layers=args.layers, num_heads=args.heads,
                      num_kv_heads=args.kv_heads, intermediate_size=args.intermediate,
                      max_seq_len=args.seq, vocab_size=args.vocab)
    model = TinyMindTransformer(cfg, seed=0)
    opt = AdamW(model.parameters(), learning_rate=1e-3)
    rng = np.random.default_rng(0)
    ids = rng.integers(4, args.vocab, size=(args.batch, args.seq))

    # ---- instrumentation -------------------------------------------------
    stack: list[str] = []
    category_of: dict[int, str] = {}
    fwd_incl = collections.defaultdict(float)

    orig_call = Module.__call__
    orig_init = Tensor.__init__

    def timed_call(self, *a, **k):
        name = type(self).__name__
        stack.append(name)
        t0 = time.perf_counter()
        try:
            return orig_call(self, *a, **k)
        finally:
            fwd_incl[name] += time.perf_counter() - t0
            stack.pop()

    def tagged_init(self, data, requires_grad=False, _children=(), _op=""):
        orig_init(self, data, requires_grad, _children, _op)
        if _op:
            category_of[id(self)] = _categorize(stack, _op)

    Module.__call__ = timed_call
    Tensor.__init__ = tagged_init

    step_rows = []
    bw_by_op = collections.defaultdict(float)
    bw_by_cat = collections.defaultdict(float)
    nodes_by_op = collections.Counter()
    graph_bytes = 0
    for step in range(args.steps + 1):  # step 0 is an untimed warm-up
        fwd_incl.clear() if step == 0 else None
        category_of.clear()
        opt.zero_grad()
        t0 = time.perf_counter()
        out = model(ids, labels=ids)
        t1 = time.perf_counter()
        order = _topo(out.loss)
        t_topo = time.perf_counter()
        out.loss._accumulate(np.ones_like(out.loss.data))
        per_op = collections.defaultdict(float)
        per_cat = collections.defaultdict(float)
        for node in reversed(order):
            if node.requires_grad:
                s = time.perf_counter()
                node._backward()
                d = time.perf_counter() - s
                per_op[node._op] += d
                per_cat[category_of.get(id(node), "other(embed/lm_head/residual)")] += d
        t2 = time.perf_counter()
        opt.step(grad_clip_norm=1.0)
        t3 = time.perf_counter()
        if step == 0:
            continue
        step_rows.append({"forward": t1 - t0, "topo_sort": t_topo - t1, "backward": t2 - t_topo,
                          "optimizer": t3 - t2})
        for k, v in per_op.items():
            bw_by_op[k] += v / args.steps
        for k, v in per_cat.items():
            bw_by_cat[k] += v / args.steps
        if step == args.steps:
            for node in order:
                nodes_by_op[node._op or "leaf"] += 1
            seen = set()
            for node in order:
                for arr in (node.data, node.grad):
                    if arr is not None and id(arr) not in seen:
                        seen.add(id(arr))
                        graph_bytes += arr.nbytes
        # Drop this step's graph before the next one is built (the engine's closures form
        # reference cycles, so refcounting alone does not free them; see the audit doc).
        del out, order
        gc.collect()
    Module.__call__ = orig_call
    Tensor.__init__ = orig_init

    med = {k: statistics.median(r[k] for r in step_rows) for k in step_rows[0]}
    total = sum(med.values())
    tokens = args.batch * args.seq

    # ---- pure-Python per-node overhead (1-element tensors) ----------------
    a = Tensor(np.ones(1, dtype=np.float32), requires_grad=True)
    n = 20000
    t0 = time.perf_counter()
    y = a
    for _ in range(n):
        y = y + a
    fwd_node = (time.perf_counter() - t0) / n
    chain = _topo(y)  # iterative: the engine's own recursive backward() would overflow on a 20k-node chain
    y._accumulate(np.ones_like(y.data))
    t0 = time.perf_counter()
    for node in reversed(chain):
        if node.requires_grad:
            node._backward()
    bwd_node = (time.perf_counter() - t0) / n
    n_nodes = sum(nodes_by_op.values())

    result = {
        "config": {"hidden": args.hidden, "layers": args.layers, "heads": args.heads, "kv_heads": args.kv_heads,
                   "intermediate": args.intermediate, "seq": args.seq, "batch": args.batch, "vocab": args.vocab,
                   "parameters": model.count_parameters()},
        "step_ms_median": {k: round(v * 1e3, 1) for k, v in med.items()} | {"total": round(total * 1e3, 1)},
        "tokens_per_sec": round(tokens / total),
        "forward_inclusive_ms_by_module": {k: round(v / (args.steps + 1) * 1e3, 1) for k, v in sorted(fwd_incl.items(), key=lambda kv: -kv[1])},
        "backward_ms_by_category": {k: round(v * 1e3, 1) for k, v in sorted(bw_by_cat.items(), key=lambda kv: -kv[1])},
        "backward_ms_by_op_top10": {k: round(v * 1e3, 1) for k, v in sorted(bw_by_op.items(), key=lambda kv: -kv[1])[:10]},
        "graph": {"nodes": n_nodes, "nodes_by_op_top8": dict(nodes_by_op.most_common(8)),
                  "retained_MB_data_plus_grad": round(graph_bytes / 2**20, 1)},
        "python_overhead_estimate": {
            "us_per_node_forward_1elem": round(fwd_node * 1e6, 2),
            "us_per_node_backward_1elem": round(bwd_node * 1e6, 2),
            "estimated_ms_per_step": round(n_nodes * (fwd_node + bwd_node) * 1e3, 1),
            "share_of_step": round(n_nodes * (fwd_node + bwd_node) / total, 4)},
        "peak_rss_MB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    print(json.dumps(result, indent=None if args.json else 2))


if __name__ == "__main__":
    main()
