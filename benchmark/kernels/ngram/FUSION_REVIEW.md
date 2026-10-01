# PLE fusion / TPU hash follow-up

2026-09-23. Base: `192a361c`, branch `p3-ngram`. Tests: `v6e-1`, TPU v6 lite x1,
JAX/jaxlib 0.11.1, libtpu 0.0.46.1. Changes are local; no commit or push.

## Implemented, default off

`NGramEmbedding.forward_decode` and `forward_extend` accept a static
`use_pallas=False` keyword. `True` selects the separate Pallas path; the old
path is unchanged. Both paths return **PLE delta**, not the outer residual.

```python
from functools import partial
import jax

# Fused PLE decode, including K/V projections; pool is positional argument 2.
decode = jax.jit(partial(layer.forward_decode, use_pallas=True), donate_argnums=(2,))
# Fused PLE packed prefill, including K/V projections.
extend = jax.jit(partial(layer.forward_extend, use_pallas=True), donate_argnums=(2,))
```

Alternatively call `ngram_decode_pallas` / `ngram_extend_pallas` from
`sgl_jax.srt.kernels.ngram_fused` with already projected K/V. Donate argument 7
in that interface. Each mode has one Pallas call combining key/query norm,
gate, conv norm, dilated depthwise conv, SiLU, delta addition and state writeback.
The two dense projections, host lookup and transfers are outside that call.

The public pool remains `[slots, channels, time]`; the kernel sees a transposed
`[slots, time, channels]` view so channels occupy the vector lanes. Each live
slot has one owner per channel group. Slot zero is read-only, other unselected
slots are preserved, and a fresh request ignores stale state. Prefill carries
state across token tiles and writes its final state once.

Current limits: one data shard, hidden size divisible by 128, whole RMSNorm
groups per tensor shard, equal activation/state dtypes, kernel size >= 2.
Nonzero slot indices must be unique. Only **TP=1** was tested; multi-chip,
multi-host donation and full-model accuracy/serving were not tested.

## Numerical checks

Do not infer compiled precision boundaries from the Python casts alone.
The first prototype actually rounded normalized K/Q to BF16 before their dot.
For one near-zero dot this produced -0.00212 versus the JIT reference's
+0.00281; the trained `sign(dot)*sqrt(abs(dot))` magnified the difference.
A separate FP64 calculation with unrounded normalized inputs gave +0.002814.

The final fused branch keeps the key projection accumulator and normalized K/Q
in FP32. It still uses the original BF16 projection inputs/weights. This avoids
extra rounding at the new custom-call boundary. U, conv state and delta retain
the activation dtype. **This is tolerance parity, not bitwise parity.**
The full T=8192 comparison passed `atol=rtol=0.025`, with delta max absolute
error 0.046875 and RMS 0.001589. The benchmark checks both delta and the entire
state pool before timing; it does not skip a failed check to report speed.

Regression: **106 passed, 1 skipped, 14 subtests passed**, including 17 new
kernel/hash tests. Coverage includes FP32 and BF16, mixed FP32-key/BF16-query,
ragged/tail/empty requests, fresh/continuing slots, duplicate padding slot zero,
untouched slots, repeated donation, prefill-to-decode, zero gate dot and several
kernel/dilation sizes. The skipped existing test requires multiple devices.

## Measured PLE performance

Released dimensions: HS=2560, HC=4, embedding=2560, kernel=4, dilation=3.
Full PLE includes both projections and completed state writeback, not host
lookup or any transformer layer. Warmup 5; median of 20 or 30 synchronized calls.
Both variants donate the same BF16 pool, re-created and made ready outside each
timed call. The compiled fused path contains one PLE Pallas custom call.

| Workload | Slots | Baseline ms | Fused ms | Speedup |
|---|---:|---:|---:|---:|
| Decode B=256 | 1024 | 1.943 | 1.408 | 1.38x |
| Prefill T=2048, B=4 | 1024 | 4.365 | 2.803 | 1.56x |
| Prefill T=8192, B=4 | 1024 | 20.668 | 8.129 | 2.54x |
| Decode B=256 | 4096 | 4.131 | 3.618 | 1.14x |

The first native-layout prototype was slower (decode 1.93 -> 6.48 ms). Keeping
the nine time entries on the innermost 128-lane axis caused severe padding and
layout overhead; it is not the retained version. HLO inspection confirms **two
pool-sized copies remain** around the fused call (`copy.17` and `copy.13` in
the B=256/1024-slot compilation). The baseline also has two such copies.
Selected-slot writeback is fused, but pool layout conversion is not eliminated.
This explains why larger pools dilute the speedup. Allocation/layout integration
remains separate work; `--hlo` prints the relevant operations. A repeat of the
1024-slot decode gave 1.944 -> 1.413 ms (1.38x).

## Why native int64 fails, and the working alternative

On the pinned stack,
`jax/_src/pallas/mosaic/lowering.py::_convert_element_type_lowering_rule`
explicitly raises `NotImplementedError("64-bit types are not supported")`
when the destination dtype has `itemsize == 8` (installed source lines
3106-3107). Thus int32 -> int64 already fails, before multiplication/XOR/modulo.
Passing int64 across `pallas_call` also fails XLA's 64-bit type rewrite.
Turning x64 off truncates to int32 and produces wrong ids; it is not a fix.

`bench_ngram_hash_limb.py` is a standalone, **not scheduler-integrated**
alternative. It keeps each 64-bit product/XOR as two uint32 words. Remainder
uses bounded 7-bit steps (and a 4-bit word prefix), with an FP32 quotient
estimate corrected by exact integer arithmetic. It supports the released
primes below 2^25 and total table rows below 2^31. Context/EOS preparation is
JAX; multiply/XOR/remainder are one Pallas call. No native int64 is used on TPU.

The old README's `k <= 4` argument was incorrect: 6-bit steps fit signed int32,
and 7-bit steps fit uint32. The old 16-nibble timing was one implementation,
not a lower bound. Host consumption of ids is a transfer cost, not a blocker.

| Hash workload | CPU NumPy ms | TPU, inputs ready ms | TPU + ids D2H ms | Inputs H2D + TPU + ids D2H ms |
|---|---:|---:|---:|---:|
| Decode B=256 | 0.0508 | 0.1266 | 0.2103 | 0.3817 |
| Prefill T=8192, B=4 | 0.9146 | 0.4177 | 0.5749 | 0.7430 |

These are synchronized call latencies, not pure device-kernel times. Each case
used 48 independently generated samples (8 warmups, 40 measured); **every id**
matched CPU for all 16 heads, including EOS barriers. Additional TPU tests
cover product limbs, remainder just below/at/above multiples, ragged T=B and
non-tile-aligned tails. CPU remains preferable for this small decode hash;
large prefill is worth integrating and measuring end-to-end. Neither row
includes host embedding gather or embedding H2D.

## Overlap and layer 0/1

No need to wait for the whole model to start implementing the prefetch API and
two-stage dispatch. But this PR still has no model execution path that constructs
`NGramEmbedding`, so **actual layer 0/1 times and real overlap are not measured**.
Isolated PLE timing must not be called layer 1 timing.

Today `_merge_ngram_ple` finishes CPU hash/gather during batch construction,
before current-batch model dispatch. This synchronous path does not overlap
its lookup with the current batch's layer 0/1. `ple_layer_ids=[2]` is 1-based:
PLE is consumed at zero-based layer 1, before that layer's attention.

The intended dependency is:

```text
TPU: hash -> token embedding / layer 0 ----------+-> layer 1 PLE -> attention
CPU:        ids D2H -> lookup -> embeddings H2D -+
```

Submit both branches without waiting on the host lookup, then join at the PLE
consumer. The useful window is primarily layer 0, not all of layer 1. Keeping
lookup inside batch construction, waiting on a future before layer 0 submission,
or hiding both stages in one execution boundary can lose that window. Real
integration needs a trace with CPU lookup, transfers and TPU layer boundaries.

## Reproduce

Run from the repository root on `v6e-1` with `/home/junyi/venv-sgl/bin/python`.
The isolated test copy is `/tmp/ngram-pr1679-review.kTrAQm`; the user's existing
server checkouts were not modified.

Local raw logs: `/tmp/ngram-pr1679-review.cjW3B4/fusion_final.log`,
`fusion_large.log`, and `fusion_hlo_and_hash.log` in the same directory.

```bash
# PR default path + fused PLE + exact limb hash regression, single TPU.
PYTHONPATH=python:. /home/junyi/venv-sgl/bin/python -m pytest -q python/sgl_jax/test/layers/test_ngram_embedding.py python/sgl_jax/test/layers/test_ngram_table.py python/sgl_jax/test/mem_cache/test_recurrent_short_conv.py python/sgl_jax/test/test_short_conv.py python/sgl_jax/test/layers/test_ngram_fused.py benchmark/kernels/ngram/test_hash_limb.py

# Qwen4Exp PLE decode B=256/1024 slots: baseline vs fused, projections included.
PYTHONPATH=python /home/junyi/venv-sgl/bin/python benchmark/kernels/ngram/bench_ngram_fused.py --mode decode --tokens 256 --batch 256 --slots 1024 --reps 30 --hlo

# Qwen4Exp PLE decode B=256/4096 slots: baseline vs fused, projections included.
PYTHONPATH=python /home/junyi/venv-sgl/bin/python benchmark/kernels/ngram/bench_ngram_fused.py --mode decode --tokens 256 --batch 256 --slots 4096 --reps 20

# Qwen4Exp PLE prefill T=2048/B=4: baseline vs fused, projections included.
PYTHONPATH=python /home/junyi/venv-sgl/bin/python benchmark/kernels/ngram/bench_ngram_fused.py --mode extend --tokens 2048 --batch 4 --slots 1024 --reps 30

# Qwen4Exp PLE prefill T=8192/B=4: baseline vs fused, projections included.
PYTHONPATH=python /home/junyi/venv-sgl/bin/python benchmark/kernels/ngram/bench_ngram_fused.py --mode extend --tokens 8192 --batch 4 --slots 1024 --reps 20

# Qwen4Exp decode B=256 hash: CPU vs limb Pallas, including completed ids D2H.
PYTHONPATH=python /home/junyi/venv-sgl/bin/python benchmark/kernels/ngram/bench_ngram_hash_limb.py --tokens 256 --batch 256

# Qwen4Exp prefill T=8192/B=4 hash: CPU vs limb Pallas, including completed ids D2H.
PYTHONPATH=python /home/junyi/venv-sgl/bin/python benchmark/kernels/ngram/bench_ngram_hash_limb.py --tokens 8192 --batch 4

# Native int64 failure reproduction, int32 kernel I/O and int64 intermediates.
/home/junyi/venv-sgl/bin/python benchmark/kernels/ngram/probe_pallas_i64.py --x64
```
