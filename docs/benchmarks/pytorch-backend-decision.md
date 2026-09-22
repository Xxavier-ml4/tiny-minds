# Optional accelerated backend (PyTorch): investigation and decision

**Decision: no PyTorch backend was added.** The NumPy autodiff engine stays the reference *and* the training
implementation, so there is one architecture, one export path and no numerical-equivalence contract to maintain.

## What was investigated, and what could not be

| Question | Result |
|---|---|
| Is Python/autodiff overhead the bottleneck (the usual reason to move to a framework)? | **No.** Measured: 508 graph nodes per step cost ≈ 1.8 ms (0.33 %) of a 562 ms step (`docs/benchmarks/phase3a-profile-T256.json`). |
| Where did the time go? | NumPy kernels, chiefly attention (67 % of instrumented forward time and 54 % of backward time, from materialising ~8 `[B,H,T,T]` arrays per layer for ≈ 25 % of the FLOPs). |
| Can that be fixed inside NumPy? | Yes, and was: fused attention / RoPE / RMSNorm / SwiGLU / linear ops (`tinymind/model/fused.py`, equivalence and gradient tests in `tests/model/test_fused.py`). Measured on this machine: 1.37–1.67× faster at T = 256 (1.09–1.48× at T = 128), 2.8× less activation memory (887 → 315 MB) (`docs/benchmarks/bench-fused-vs-reference.json`, `phase3b-fused-profile-T256.json`). |
| Is PyTorch installable on a GitHub-hosted runner at a reasonable cost? | **Not measured.** This sandbox has no network access, so neither the wheel download/installation time nor PyTorch CPU throughput on a runner could be observed, and no number is claimed. |
| Does PyTorch's threaded CPU kernels beat NumPy+OpenBLAS at these sizes (hidden 128–192)? | **Unknown**, for the same reason; the matrices are small enough that thread scaling may be modest either way. `benchmarks/runner_probe.py` measures NumPy's thread scaling on the actual runner. |

## When to revisit (all must hold)

1. `runner_profile.json` from a real run shows NumPy throughput too low for the stage token budget in
   `docs/training/three-stage-plan.md` within 5.5 h × the number of jobs one is willing to chain (stages already
   resume across jobs, so *slow* is a cost, not a blocker).
2. A cheaper lever has been ruled out first: more chained jobs; a shorter context; a learned tokenizer (fewer tokens for
   the same text — see the tokenizer note in the implementation report).
3. A PyTorch install of under ~3 minutes on the runner, measured.

## Contract a backend would have to meet (so it cannot silently become a second architecture)

* Same `ModelConfig`; parameter names, shapes and order = `parameter_shapes(config)`; weights exchanged through the
  checkpoint's `model.npz` and exported through `export_package` (never a second `.tm` writer).
* Same math: pre-norm RMSNorm (`eps` from the config), RoPE half-rotation with `theta` from the config, GQA where head
  `i` reads KV head `i // (heads/kv_heads)`, SwiGLU `silu(gate) * up`, tied or untied head, causal / block-diagonal mask,
  token-normalised loss with `loss_normalizer`.
* A numerical-equivalence test on small models (same config, same weights, same ids): logits within 1e-4 and gradients
  within 1e-3 (relative), plus the resume-determinism test run against it.
* If equivalence fails, the backend is not selectable — never a silent fallback in either direction.
