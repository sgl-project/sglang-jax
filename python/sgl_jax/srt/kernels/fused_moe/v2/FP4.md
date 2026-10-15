# MXFP4 group32 routed experts

`fused_ep_moe_v2(..., mxfp4=True)` accepts E2M1 packed weights with E8M0
scales along K in groups of 32. This is an opt-in routed-expert kernel;
checkpoint loading, router computation and shared experts remain external.
G512/G256 and post-dot scaling are not part of this implementation.

## Requirements and input contract

The native path uses the Pallas TPU ACC API validated with **JAX/jaxlib
0.11.2 on TPU v7x**. The repository's current 0.11.1 pin does not export
these primitives: the new route raises an explicit error on that version.
This change does not upgrade the repository-wide dependency or affect the
existing BF16/FP8 route. CPU interpretation is only a numerical test path.

Use TP=1 and DP=EP, with mesh axes `data=P, tensor=1`. The wrapper shards
input tokens/routes over `data` and weights over experts. It uses the
existing assignment dispatch/combine protocol, without destination-rank
deduplication.

| Input | Global shape | Type |
|---|---|---|
| tokens | `[T,H]` | BF16 |
| Gate / Up packed weights | `[E,H/2,I]` | uint8 |
| Down packed weights | `[E,I/2,H]` | uint8 |
| Gate / Up scales | `[E,H/32,I]` | uint8 E8M0 |
| Down scales | `[E,I/32,H]` | uint8 E8M0 |
| routing weights | `[T,top_k]` | FP32 |
| expert IDs | `[T,top_k]` | int32 |

The low nibble represents even K, the high nibble odd K. H must be a
multiple of 512, I a multiple of 256, and T/E divisible by EP. `bf` must
be a positive multiple of 256; `bts` and `btc` must be multiples of 16,
`btc` divides `bts`, and `bts <= 1024` for the full-H ACC allocation.
Resource availability still constrains valid configurations.

Prepare physical panel padding once, **outside the repeated invocation**.
Cache separately for each `bf`; retain the returned logical I so padding
never becomes real work. This helper does not requantize weights.

```python
from sgl_jax.srt.kernels.fused_moe.v2.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe_v2,
    prepare_fp4_weights,
)

weights, quant_args = prepare_fp4_weights(
    gate, down, up, gate_scale, down_scale, up_scale, bf=512
)
config = FusedMoEBlockConfig(bt=512, bts=80, btc=80, bf=512, bse=512)
y = fused_ep_moe_v2(
    mesh, tokens, *weights, routing_weights, expert_ids, 6,
    block_config=config, swiglu_limit=1.0, **quant_args,
)
```

The example configuration is for global T8192/DP8. The activation limit
must be chosen from the caller's model contract; 1.0 is the synthetic test
setting, not a checkpoint accuracy claim.

## Code organization

All runtime implementation lives in `kernel.py`: `prepare_fp4_weights` is a
public load-time preparation helper, while `_fp4_*` helpers implement only
FP4-specific decoding, rounding, buffer handling and ACC computation. There
is no separate FP4 backend module or group-selector dispatch layer. Reference
unpacking stays in the test file. Existing BF16/FP8 activation is not reused
for the FP4 path because its clamp/route-weight rounding order differs.

## Computation and buffer lifetime

The original expert/bts/bf loops, weight panel slots, deferred Down wait,
`bf+2` prefetch, and expert-level communication remain in use. Packed
weights and scales have separate DMA transfers; both are drained before
consumption. X and route-weight loads are started before either is waited
on. A separate FP32 route mailbox preserves route precision.

Each K256 tile decodes eight G32 scale groups into BF16, using two
`[256,256]` decoded slots. Gate/Up uses a separate `[btc,256]` ACC per row
tile, keeps the sum across all H, and only pops at the end. RHS is reused
across row tiles. Down pops once per bf panel and accumulates in the
existing output buffer. Native ACC references are never sliced.

Gate/Up sums round to BF16, then activation runs in FP32. When a limit is
provided, Gate is upper-clamped and Up is clamped to `[-limit,limit]`
before SiLU. The route weight multiplies the activated intermediate in
FP32 before BF16 rounding and Down. Combine must therefore not multiply
the route weight a second time. Each returned assignment rounds to BF16;
owner accumulation is FP32 and the final output BF16. This is a distinct
rounding contract from applying routing weights after Down.

## Validation and measured scope

```sh
PYTHONPATH=python python -m pytest -q test/srt/kernels/fused_moe/test_fp4_g32.py
```

Tests cover packed codes, E8M0 encoding, interpreted FFN panels and tails,
independent unsliced ACC references, actual ACC lowering rules/MLIR
verification, DMA start/wait order, conditional tail masks, preparation,
and abstract tracing of the distributed FP4 and existing BF16 paths.
Native ACC tests are skipped on 0.11.1; wrapper tests verify the explicit
unsupported-version error there. Neither CPU interpretation nor abstract
tracing proves distributed TPU execution.

The source from which this port was extracted was tested on v7x with
H5120/I2304/E384/top6, BF16 activation, TP1/DP=EP, synthetic weights:

| Devices | Global tokens | bt / bts / btc / bf | Median (ms) | p90 (ms) |
|---:|---:|---|---:|---:|
| 8 | 8192 | 512 / 80 / 80 / 512 | 16.670 | 16.746 |
| 8 | 16384 | 512 / 80 / 80 / 512 | 57.232 | 57.324 |
| 8 | 2304 | 288 / 64 / 64 / 1024 | 7.030 | 7.044 |
| 16 | 4608 | 288 / 128 / 128 / 1024 | 5.261 | 5.556 |

Each sample is one distributed routed MoE call, taking the maximum device
duration per round; 10 valid rounds, tracing enabled, bounded tuning.
Decode checks covered all rows; prefill checks sampled 64 rows. These are
prior source-version results, not a new TPU run of this integration commit,
not an FP8 speedup comparison, and not real-checkpoint quality validation.
Final LLO of that source confirms next-tile decoding between MXU issue and
result reads; this does not quantify hardware busy overlap. Extreme-scale
TPU subnormal/overflow behavior is not covered by the synthetic benchmark.
