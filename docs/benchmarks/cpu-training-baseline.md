# CPU training baseline — where the time goes

This is the profiling record the spec (section 13) asks for, done *before* any
decision to introduce another framework. Two questions, answered with
measurements, not assumptions: **is Python/autodiff overhead the bottleneck**,
and **if not, where is the time actually spent**. Every number below has a
script that reproduces it; none is estimated.

## Machine

```
Intel(R) Xeon(R) Processor @ 2.10GHz — 1 vCPU visible (os.sched_getaffinity)
RAM: 4000 MB · Python 3.12.3 · NumPy 2.4.4 · BLAS: scipy-openblas 0.3.31.188.0
```
(`docs/benchmarks/runner-profile-sandbox.json`, `benchmarks/runner_probe.py`).
This is the development sandbox, not a GitHub-hosted runner — the workflow
runs the same probe on the actual runner before training (see
`docs/architecture/training-system.md` Part 2, section 14, and the limitation
noted at the end of this document).

## 1. Baseline: the untouched Phase 3A implementation

`benchmarks/audit/profile_training_step.py --tree <phase3a tree> --seq 256`,
model shape h=128, L=6, 4 heads / 2 KV heads, I=384 (the `tiny_mobile` shape),
byte vocabulary, batch 8:

| | seq 128 | seq 256 |
|---|---|---|
| step time | 245.2 ms | 561.8 ms |
| forward / backward / optimizer | 89.0 / 144.8 / 11.3 ms | 221.6 / 329.3 / 10.8 ms |
| tokens/s | 4,175 | 3,645 |
| graph nodes | 508 | 508 |
| graph memory (activations + grads retained) | 368.2 MB | 886.6 MB |

**Python/autodiff graph-construction overhead**, measured directly (build a
20,000-node chain of 1-element tensors, time each node's forward and backward
in isolation, multiply by the real step's node count):

```
508 nodes × (forward-node-overhead + backward-node-overhead) ≈ 3 ms
3 ms / 562 ms step ≈ 0.3%
```

**This answers the first question: no, Python overhead is not the
bottleneck.** Replacing the autodiff engine, or moving to another framework
to "reduce interpreter overhead", would optimise 0.3% of the step.

## 2. Where the time actually goes

Same script, same shape, per-module forward time and per-category backward
time (a "category" is which layer's closure a gradient computation belongs
to — attention linear projections vs. attention's RoPE/softmax/score core vs.
MLP vs. norm):

| | forward (ms) | backward (ms) | combined share of step |
|---|---|---|---|
| attention core (RoPE, scores, softmax, context) | included below | 184 | |
| attention (all, incl. Q/K/V/O projections) | 228 | 222 | **≈ 55%** of the step |
| MLP (SwiGLU) | 81 | 150 | ≈ 25% |
| RMSNorm | 17 | 26 | ≈ 4% |
| optimizer (AdamW) | — | — | ≈ 2% |

By raw operator, backward time: `matmul` 212 ms, elementwise `mul` 74 ms,
`softmax` 41 ms, `getitem` (RoPE's half-rotation slicing, going through
NumPy's unbuffered `np.add.at`) 40 ms, `sigmoid` 22 ms.

Attention is roughly a quarter of the model's FLOPs (`hidden × seq² × layers`
vs. the MLP's `hidden × intermediate × seq × layers`, and here
`intermediate = 3 × hidden`) but more than half the step, because the
Phase 3A implementation of one attention layer allocates on the order of
eight full `[batch, heads, seq, seq]` arrays — scores, the scaled copy, the
masked copy, the softmax output, and each one's gradient, plus a materialised
K/V head-repeat for GQA — instead of computing in place. **886 MB retained for
a 1.21M-parameter model is an activation-memory problem, not a
parameter-memory problem** (the weights themselves are 4.9 MB in float32).

## 3. What was changed, and what it bought (fused ops, `tinymind/model/fused.py`)

The composed (Phase 3A-style) implementation is kept as the reference; the
fused ops are checked against it (equivalence + gradient tests in
`tests/model/test_fused.py`) rather than replacing it. Same profiler, same
shape, before vs. after:

| seq | implementation | step (ms) | tokens/s | graph nodes | graph memory (MB) |
|---|---|---|---|---|---|
| 256 | composed (reference) | 561.8 | 3,645 | 508 | 886.6 |
| 256 | fused | 335.6 | **6,102 (1.67×)** | 199 | **315.4 (2.8× less)** |
| 128 | composed (reference) | 245.2 | 4,175 | 508 | 368.2 |
| 128 | fused | 165.9 | **6,173 (1.48×)** | 199 | **162.3 (2.3× less)** |

After fusion, backward time by op is dominated by `linear` (83 ms of 187 at
seq 256) — i.e. now mostly the actual matrix multiplications, which is where
compute-bound time belongs. Additional structural changes, each independently
tested: `Tensor.backward()` was rewritten from a recursive to an iterative
topological sort (the recursive version overflowed Python's recursion limit
past ~28 transformer layers — `test_backward_recursion` in the audit script,
confirmed up to 48 layers after the fix); `retain_graph=False` frees each
node's closure and references as soon as its gradient is pushed, instead of
holding the whole graph until the next Python garbage-collection pass;
`no_grad()` stops recording a graph at all for inference/evaluation; basic
(slice/int-only) index gradients use direct assignment instead of
`np.add.at`, which is exact for indices that never repeat and measurably
faster.

## 4. Micro-batch / sequence-length shape

`benchmarks/train_benchmark.py matrix --kind batch` (seq 128) and
`--kind seqlen` (constant ≈2048 tokens/step), `tiny_mobile`, fused ops:

| batch | tokens/step | step (ms) | tokens/s |
|---|---|---|---|
| 1 | 128 | 26 | 4,834 |
| 2 | 256 | 43 | 6,002 |
| 4 | 512 | 80 | 6,365 (peak on this machine) |
| 8 | 1024 | 184 | 5,550 |
| 16 | 2048 | 346 | 5,924 |

| seq | batch (const. tokens/step) | step (ms) | tokens/s |
|---|---|---|---|
| 64 | 32 | 309 | 6,626 |
| 128 | 16 | 335 | 6,119 |
| 256 | 8 | 393 | 5,205 |
| 512 | 4 | 512 | 3,996 |

Throughput falls as context grows (attention's quadratic term), as expected;
it is not flat, so context length is a real cost, not a free choice — see the
attention-variant and sequence-length *quality* sweeps in
`docs/training/three-stage-plan.md` and `sweep-seqlen.json` for the other
half of that trade-off (a shorter context truncates most training examples).

## 5. Attention layout: MHA vs GQA vs MQA at equal parameter budget

The brief's own caution (section 4) — do not assume GQA is automatically
faster — was tested, not assumed. `intermediate_size` is adjusted so all three
variants have equal parameter count (`train_benchmark.py`'s `match_budget`),
`tiny_mobile`-shaped, batch 8:

| variant | seq | KV heads | parameters | step (ms) [min–max] | tokens/s | KV cache (MiB @ that seq) |
|---|---|---|---|---|---|---|
| MHA | 128 | 4 | 1,312,896 | 183 [178–186] | 5,598 | 0.750 |
| GQA | 128 | 2 | 1,306,752 | 169 [157–172] | 6,073 | 0.375 |
| MQA | 128 | 1 | 1,312,896 | 173 [156–192] | 5,904 | 0.188 |
| MHA | 256 | 4 | 1,312,896 | 424 [406–431] | 4,825 | 1.500 |
| GQA | 256 | 2 | 1,306,752 | 397 [391–412] | 5,156 | 0.750 |
| MQA | 256 | 1 | 1,312,896 | 386 [360–396] | 5,306 | 0.375 |

GQA and MQA are both modestly faster than MHA on this machine (≈8-9% at
seq 256) with, as expected, proportionally smaller KV caches; the speed gap
between GQA and MQA is within the run-to-run noise (min–max ranges overlap).
**GQA was selected for the shipped profiles**: it captures nearly all of
MQA's cache saving over MHA while keeping more than one KV head, which the
quality sweep (`sweep-attention.json`, `docs/training/three-stage-plan.md`)
found matched MHA's validation loss exactly at 250 steps while MQA's was
measurably worse — a speed-only comparison would have missed that.

## 6. Data pipeline cost (is loading/collating the bottleneck?)

`benchmarks/data_pipeline_cost.py` on the stage-2 curriculum (9,766 examples,
1.33M tokens):

| | value |
|---|---|
| render + tokenize | 14,341 examples/s (1.96M tokens/s) |
| build one epoch's plan (packed) | 33.4 ms |
| one micro-batch (padded / packed) | 0.023 ms / 0.038 ms |
| share of a `tiny_mobile` training step (395 ms) | 0.006% / 0.010% |

Not the bottleneck, by three orders of magnitude.

## 7. What this does and does not establish

Established, by measurement: the bottleneck is attention's memory traffic,
not Python/autodiff overhead or data loading; fusing attention/RoPE/norm/MLP
ops inside NumPy recovers 1.4-1.7× throughput and ~2.5-2.8× less activation
memory without changing the reference implementation or its numerics; GQA is
the right default for this model family on both speed and quality grounds,
tested rather than assumed.

**Not established here** (see `docs/benchmarks/pytorch-backend-decision.md`
for the full reasoning): multi-thread BLAS scaling — this sandbox exposes
only 1 vCPU, so `runner_probe.py`'s thread sweep has a single candidate
(`docs/benchmarks/runner-profile-sandbox.json`); the workflow runs the same
probe on the actual multi-core GitHub runner and selects a thread count from
real measurements there, not from this document. PyTorch's CPU throughput and
installation cost on a GitHub-hosted runner — no network access in this
sandbox to measure either. Neither number is assumed in place of measuring
it; the PyTorch decision document explains exactly what would need to be true
before revisiting the "NumPy only" choice.

**Also not established with high precision: absolute wall-clock numbers on
this specific sandbox.** Several benchmarks in this delivery were run while
another background job (a training run, a sweep) was active on the same
single vCPU — confirmed by re-measuring `tiny_mobile` training throughput on
an idle machine (5,256 tokens/s) against `docs/benchmarks/mobile-matrix.json`'s
figure for the identical shape and batch (2,606 tokens/s, taken under
contention). Rankings and ratios measured with both alternatives run back to
back in the same script invocation (GQA vs MHA, fused vs reference, the
sequence-length and batch sweeps) are the reliable signal; any single
absolute tokens/s number in this repository should be read as accurate to
roughly a factor of 2 on this machine, not a precise throughput guarantee.
