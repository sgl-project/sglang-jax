# Fused MoE v4 — Tensor-Parallel kernel (bf16-only)

## TL;DR

v1/v2 are Expert-Parallel: each chip holds `1/tp` of experts at full intermediate
width and exchanges tokens via all-to-all scatter/gather + barrier. AInfer's v4
profile shows ~40% of decode wall-clock spent on this ICI coordination.

v4 flips to Tensor-Parallel: every chip holds **all** experts but only a `1/tp`
slice of the intermediate dimension. Each chip runs the full grouped FFN over
active experts with its weight slice, then **one** `psum` combines partials —
no scatter/gather/barrier.

Weight HBM traffic stays roughly conserved; the win is purely from eliminating
ICI coordination.

## What's in this port (minimal scope)

Ported from `AInfer/ainfer/kernels/tpu/fused_moe/v4/kernel.py` (commit reviewed
2026-06-23). Differences from the upstream:

- TP axis default `"tp"` → `"tensor"` (sglang-jax mesh convention).
- **Removed** `AINFER_MOE_V4_BF16_PSUM` env knob (default f32-reduce only).
- **Removed** w13-fuse (REPLACE) mode + entry parameter `w13_local`.
- **Removed** Pallas DMA decode helper (`decode_kernel.py`) — purely a w13-fuse
  perf companion.
- **Removed** shared-expert-as-top_k+1-slot fusion in `tp_moe_decode`. sglang-jax
  models handle shared experts externally; matches the existing FusedEPMoE convention.

## API

- `tp_moe(...)` — prefill path: sort + ragged_dot grouped FFN. Returns partial.
- `tp_moe_decode(...)` — decode path: gather + einsum. Returns partial.
- `tp_moe_per_device(...)` — selects path by `T <= 64`, then `lax.psum`. Must run
  inside a `shard_map` with `tp_axis_name` as a manual/collective axis.
- `fused_tp_moe_v4(...)` — standalone `shard_map` entry for tests / scripts.

## Known performance caveats on ling_v3_flash

This kernel was originally tuned for AInfer workloads with decode-sized token
batches. When used through sglang-jax engine with `precompile_token_paddings`
≥128, **the decode path is never reached** — all calls go through `tp_moe`
(prefill ragged_dot). In that regime:

- `I_local = moe_intermediate_size / tp = 768 / 4 = 192` is not a multiple of
  the MXU tile (128). Expect ~25% padding waste in matmul.
- `max_active = min(top_k * T, num_experts) = num_experts` whenever the batch
  visits every expert (very likely at T=256, top_k=8, E=512). The whole
  expert table participates in the grouped FFN — effectively dense TP-MoE.

Net: in this configuration v4 is expected to be **slower** than v1; the port
validates correctness, not perf parity. Real perf comparison should drop
`precompile_token_paddings` below 64 to activate the decode path, or compare
on a model with fewer experts / smaller top_k / I_local divisible by 128.

## Hard requirements (must not violate)

- bf16 weights & activations. No quantization. The wrapper raises if
  `quantization_config` is non-trivial.
- Shared expert handled by the caller (not fused into the kernel).
- EPLB **must be disabled** when using FUSED_V4 — TP replicates every expert
  on every chip, so the physical→logical map is meaningless and would silently
  mis-route. The wrapper raises if EPLB metadata is registered.

## File layout

```
v4/
├── __init__.py          # re-exports tp_moe / tp_moe_per_device / fused_tp_moe_v4
├── kernel.py            # the kernel itself, pure JAX + lax.ragged_dot
└── README.md            # this file
```

No Pallas; no auto-tuning block configs. The entire kernel is ~300 lines.
