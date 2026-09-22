### Profiles: training throughput (batch 8, each profile's own context)

| profile | parameters | h × L | heads / kv | seq | step (ms) | tokens/s | s per 100 steps | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|
| tiny_debug | 115,264 | 64 × 2 | 4 / 2 | 128 | 28 | 36,964 | 2.8 | 71.4 |
| tiny_mobile | 1,214,592 | 128 × 6 | 4 / 2 | 256 | 396 | 5,178 | 39.6 | 306.0 |
| tiny_mobile_plus | 3,493,824 | 192 × 8 | 6 / 2 | 256 | 832 | 2,462 | 83.2 | 563.9 |

### Profiler runs, tiny_mobile shape, batch 8 (`benchmarks/audit/profile_training_step.py`)

| seq | implementation | step (ms) | tokens/s | graph nodes | graph memory (MB) |
|---|---|---|---|---|---|
| 256 | Phase 3A (composed ops) | 561.8 | 3,645 | 508 | 886.6 |
| 256 | Phase 3B (fused ops) | 335.6 | 6,102 | 199 | 315.4 |
| 128 | Phase 3A (composed ops) | 245.2 | 4,175 | 508 | 368.2 |
| 128 | Phase 3B (fused ops) | 165.9 | 6,173 | 199 | 162.3 |

### Fused vs reference ops, tiny_mobile, batch 8 (`train_benchmark.py matrix --kind fused`)

| seq | ops | step (ms) | tokens/s | forward / backward (ms) | peak RSS (MB) |
|---|---|---|---|---|---|
| 128 | reference | 202 | 5,063 | 85 / 104 | 233.1 |
| 128 | fused | 186 | 5,517 | 87 / 86 | 171.1 |
| 256 | reference | 536 | 3,817 | 264 / 254 | 500.4 |
| 256 | fused | 392 | 5,230 | 180 / 200 | 306.1 |

### MHA vs GQA vs MQA at equal parameter budget (speed)

| variant | seq | kv heads | intermediate | parameters | step (ms) [min–max] | tokens/s | KV cache fp32 at seq (MiB) |
|---|---|---|---|---|---|---|---|
| MHA | 128 | 4 | 384 | 1,312,896 | 183 [178–186] | 5,598 | 0.75 |
| GQA | 128 | 2 | 424 | 1,306,752 | 169 [157–172] | 6,073 | 0.375 |
| MQA | 128 | 1 | 448 | 1,312,896 | 173 [156–192] | 5,904 | 0.188 |
| MHA | 256 | 4 | 384 | 1,312,896 | 424 [406–431] | 4,825 | 1.5 |
| GQA | 256 | 2 | 424 | 1,306,752 | 397 [391–412] | 5,156 | 0.75 |
| MQA | 256 | 1 | 448 | 1,312,896 | 386 [360–396] | 5,306 | 0.375 |

### Sequence length at a constant 2048 tokens per step (tiny_mobile)

| seq | batch | step (ms) | tokens/s | examples/s | peak RSS (MB) |
|---|---|---|---|---|---|
| 64 | 32 | 309 | 6,626 | 103.54 | 270.0 |
| 128 | 16 | 335 | 6,119 | 47.8 | 282.0 |
| 256 | 8 | 393 | 5,205 | 20.33 | 306.0 |
| 512 | 4 | 512 | 3,996 | 7.8 | 355.1 |

### Micro-batch size at seq 128 (tiny_mobile)

| batch | tokens/step | step (ms) | tokens/s | peak RSS (MB) |
|---|---|---|---|---|
| 1 | 128 | 26 | 4,834 | 71.1 |
| 2 | 256 | 43 | 6,002 | 85.8 |
| 4 | 512 | 80 | 6,365 | 114.8 |
| 8 | 1024 | 184 | 5,550 | 171.2 |
| 16 | 2048 | 346 | 5,924 | 282.0 |

### Learning-rate sweep (tiny_mobile, 250 steps, batch 8, seq 256, stage2 curriculum scale 0.08; seed 1)

| peak LR | initial val loss | final val loss | train loss (last 10) | grad norm mean / max | steps clipped | loss spikes | NaN/Inf | tokens/s |
|---|---|---|---|---|---|---|---|---|
| 0.0001 | 6.257 | 1.9712 | 2.0360 | 3.82 / 10.61 | 250/250 | 0 | no | 4,698 |
| 0.0003 | 6.257 | 1.1331 | 1.1767 | 3.46 / 10.61 | 250/250 | 22 | no | 4,578 |
| 0.001 | 6.257 | 0.5736 | 0.5582 | 2.42 / 8.83 | 250/250 | 66 | no | 4,517 |
| 0.003 | 6.257 | 0.4639 | 0.4340 | 1.55 / 8.83 | 221/250 | 66 | no | 4,550 |
| 0.01 | 6.257 | 1.0977 | 1.1949 | 1.49 / 8.83 | 217/250 | 6 | no | 3,363 |

Repeat with seed 2 of the two best:

| peak LR | final val loss (seed 2) |
|---|---|
| 0.003 | 0.4568 |
| 0.001 | 0.5784 |

### Attention layout, quality at equal budget (250 steps, lr 0.003)

| variant | kv heads | intermediate | parameters | final val loss | tokens/s |
|---|---|---|---|---|---|
| MHA | 4 | 384 | 1,312,896 | 0.4627 | 3,333 |
| GQA | 2 | 424 | 1,306,752 | 0.4627 | 3,140 |
| MQA | 1 | 448 | 1,312,896 | 0.4777 | 4,270 |

### Training context length (250 steps, lr 0.003, ~2048 tokens per step)

| train seq | batch | training examples kept | dropped as too long | val loss (common short subset) | tokens/s |
|---|---|---|---|---|---|
| 64 | 32 | 223/2400 | 90.7% | 2.5880 | 5,648 |
| 128 | 16 | 670/2400 | 72.1% | 1.5265 | 4,090 |
| 256 | 8 | 2400/2400 | 0.0% | 1.6231 | 3,816 |

### Padding utilisation: padded vs packed batches (batch 8, one epoch, context 256)

| curriculum | examples | mean tokens/example | padded utilisation | packed utilisation |
|---|---|---|---|---|
| stage2 | 9,766 | 136.42 | 68.4% | 77.2% |
| stage0 | 1,200 | 41.92 | 85.3% | 93.3% |

### Data pipeline cost (stage-2 curriculum, scale 0.35)

| quantity | value |
|---|---|
| render + tokenize | 14,341 examples/s (1,956,332 tokens/s) |
| epoch plan (packed) | 0.0334 s per epoch |
| micro-batch, padded / packed | 0.023 ms / 0.038 ms |
| share of a tiny_mobile step (395 ms) | 0.006% / 0.010% |

### Benchmark matrix (random-initialised weights; latency and memory do not depend on weight values)

| profile | parameters | train tok/s | s/100 steps | Python prefill 64 (ms) | Python decode (ms/tok) | native prefill 64 (ms) | native decode (ms/tok) | native RSS above bare process (MiB) | fp32 .tm (MiB) | int8 .tm, embed fp32 (MiB) | KV cache fp32 @128 / @256 (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny_debug | 115,264 | 18,145 | 5.6 | 1.6 | 0.744 | 7.88 | 0.209 | 1.0 | 0.442 | 0.167 | 0.062 / 0.125 |
| tiny_mobile | 1,214,592 | 2,606 | 78.6 | 10.69 | 2.614 | 87.89 | 1.731 | 6.4 | 4.637 | 1.298 | 0.375 / 0.75 |
| tiny_mobile_plus | 3,493,824 | 1,180 | 173.6 | 26.65 | 4.181 | 261.2 | 4.818 | 15.8 | 13.333 | 3.554 | 0.5 / 1.0 |

### FP32 vs INT8 on the stage-3 model (same held-out data; `benchmarks/quantization_eval.py`)

| variant | .tm bytes | validation loss | held-out eval loss | correct tool | argument acc. | false-positive calls | Python tok/s | weights in RAM (MiB) |
|---|---|---|---|---|---|---|---|---|
| float32 | 4,863,136 | 0.1770 | 0.4832 | 0.837 | 0.025 | 0.316 | 743.77 | 4.63 |
| int8, embedding float32 | 1,360,960 | 0.1771 | 0.4829 | 0.833 | 0.025 | 0.319 | 738.24 | 4.63 |
| int8, embedding int8 | 1,262,272 | 0.1771 | 0.4830 | 0.833 | 0.025 | 0.312 | 737.38 | 4.63 |

### Machine these numbers come from

Intel(R) Xeon(R) Processor @ 2.10GHz; visible cores 1; RAM 4000 MB; Python 3.12.3; NumPy 2.4.4; BLAS scipy-openblas 0.3.31.188.0; thread selection 1 (measured {'1': 5998} tokens/s).

