---
title: "TPU Topology Reference"
---

# TPU Topology Reference

Single-source table for TPU generation, per-chip HBM, JAX device-per-chip ratio, and topologies that appear in SGL-JAX recipes. **All numbers below derive from `python/sgl_jax/srt/utils/jax_utils.py` (`get_device_name` for the normalised name, `get_device_hbm_limit` for HBM) and from kernel comments under `python/sgl_jax/srt/kernels/`** — when this page conflicts with a recipe, the recipe is wrong.

## Generations

| TPU | JAX `device_kind` (raw) | Normalised name | HBM / JAX device | JAX devices / chip |
|---|---|---|---|---|
| v4 | `TPU v4` | `TPU v4` | 32 GiB | 1 |
| v5e | `TPU v5 lite` | `TPU v5e` | 16 GiB | 1 |
| v5p | `TPU v5p` (or `TPU v5`) | `TPU v5p` | 95 GiB | 1 |
| v6e (Trillium) | `TPU v6 lite` | `TPU v6e` | 32 GiB | 1 |
| v7x | `TPU7x` | `TPU v7` | **96 GiB** (half a 192 GiB chip) | **2** |

> Raw `device_kind` strings come from `jax.devices()[0].device_kind` and depend on the JAX runtime version. `get_device_name()` normalises them (strips trailing `" lite"`, etc.) — always go through that function rather than matching raw strings.

### Why v7x is special

Each v7x chip carries 192 GiB HBM but JAX exposes it as **two logical devices** of 96 GiB each. So:

- A `v7x-8` host has **4 chips × 2 = 8 JAX devices**. `--tp-size=8` saturates a single host.
- A `v7x-16` slice (`2x2x4`, 4 nodes) has **16 chips × 2 = 32 JAX devices**. `--tp-size=32` saturates the slice.
- `get_device_hbm_limit()` returns 96 GiB because that's what each `jax.devices()[i]` actually sees, not the full per-chip HBM.

v6e (and every other generation) is 1:1 chip→device, so `--tp-size` matches chip count directly.

## Topologies referenced by cookbook recipes

| Slice name | Nodes | Chips/node | JAX devices | Used by |
|---|---|---|---|---|
| `v6e-4` | 1 | 4 | 4 | [Qwen-7B-Chat](../autoregressive/Qwen/Qwen.md), [Qwen3-8B / 32B](../autoregressive/Qwen/Qwen3.md), [Qwen2.5-VL](../autoregressive/Qwen/Qwen2.5-VL.md), [Wan 2.1](../diffusion/Wan/Wan2.1.md), [Wan 2.2](../diffusion/Wan/Wan2.2.md) |
| `v6e-16` (`4x4`) | 4 | 4 | 16 | [MiMo-V2-Flash](../autoregressive/Xiaomi/MiMo-V2-Flash.md) (multi-node) |
| `v6e-32` | 8 | 4 | 32 | [Grok-2](../autoregressive/Grok/Grok2.md) |
| `v6e-64` (`4x4x4`) | 16 | 4 | 64 | [MiMo-V2.5-Pro](../autoregressive/Xiaomi/MiMo-V2.5-Pro.md) |
| `v7x-8` | 1 | 4 | 8 | [MiMo-V2-Flash](../autoregressive/Xiaomi/MiMo-V2-Flash.md) (single-node) |
| `v7x-16` (`2x2x4`) | 4 | 4 | 32 | [MiMo-V2.5-Pro](../autoregressive/Xiaomi/MiMo-V2.5-Pro.md) |

## Choosing `--tp-size`

Rule: **`--tp-size` = total JAX devices across all nodes**, where JAX-devices-per-chip is per the table above (1 for everything except v7x where it's 2).

```
v6e-N    →  --tp-size = N
v7x-N    →  --tp-size = N * 2      (N is chip count)
```

For MoE models also set `--ep-size = --tp-size` to put one expert shard per device, and split attention with `--dp-size` (attention-TP becomes `tp_size / dp_size`).

## Per-chip HBM accounting

For TP planning:

```
weight_GB_per_device ≈ params_billions * dtype_bytes / tp_size
```

with `dtype_bytes = 2` for `bfloat16`, `1` for FP8. The MiMo-V2-Flash recipe records `~20 GB/chip in FP8` for its 256-expert configuration — use that as a sanity reference for similar architectures.

`--mem-fraction-static` defaults to 0.88 on TPU (`server_args.py` `__post_init__`); raising to 0.95 is common for dedicated serving but leaves no room for short-lived buffers from other processes on the same host.

## Adapting to other topologies

Each cookbook recipe ships **the slice we had access to when measuring** — that's the reproducibility contract, not a designed "tier" claim. If you have a different (larger / smaller / cross-generation) slice, the launch flags follow a small set of mechanical scaling rules:

### 1. Tested config vs minimum requirement

A recipe's `## 2.1 Hardware Matrix` will surface two facts:

- **Tested configuration** — the slice we ran the §4 numbers on. Launch command in §2.3 is verbatim copyable.
- **Minimum requirement** — the smallest slice the model fits, derived from BF16/FP8 weight footprint + per-token KV + activation peak.

Anything between (and many things above) the minimum is also valid; the recipe just doesn't carry §4 numbers for those slices yet.

### 2. Scale up to a larger slice (same generation)

`--tp-size` and `--ep-size` scale linearly with chip count. Worked example using DeepSeek-V3 on v6e:

| Slice | Chips | `--tp-size` | `--dp-size` | `--ep-size` | Notes |
|---|---|---|---|---|---|
| v6e-32 | 32 | 32 | 4 | 32 | hits FP8 shared-expert collapse at `dp=4` — see DeepSeek-V3 §2.4 |
| v6e-64 (tested) | 64 | 64 | **8** | 64 | tested configuration; `dp=8` mandatory for FP8 mesh |
| v6e-128 | 128 | 128 | 16 | 128 | extrapolation; `dp=16` still satisfies `144 % tensor == 0` and shared-expert `2048/(tp/dp)=128` |

The model-specific mesh constraints (`dp` divisibility, MoE expert divisibility, GLA `num_groups` ≤ tensor axis) usually pin `--dp-size` and force `--tp-size` to be a multiple of something — check §2.4 of the target recipe before scaling.

### 3. Swap v6e ↔ v7x

v7x exposes **2 JAX devices per chip**, so `--tp-size = chip_count × 2`. A v7x-N slice maps to the same `--tp-size` as v6e-(2N). Examples:

| v6e slice | Equivalent `--tp-size` | v7x slice | Notes |
|---|---|---|---|
| v6e-16 | 16 | v7x-8 | same `--tp-size`, v7x has more HBM per device |
| v6e-32 | 32 | v7x-16 | same launch shape; v7x interconnect lower latency |
| v6e-64 | 64 | v7x-32 | both viable for trillion-class MoE |

Memory math also changes — v7x's 96 GiB / JAX device vs v6e's 32 GiB means a tighter v6e slice can be replaced by a smaller v7x chip count. Re-derive the minimum requirement from HBM math, not just chip count.

### 4. Scale down toward the minimum

Lower bound is set by **weight footprint + activation peak + KV pool**, not by `--tp-size` math. Common failure modes when too small:

- `RESOURCE_EXHAUSTED: ... Used 31.X G of 31.25 G hbm` during EXTEND precompile — drop `--chunked-prefill-size` and `--max-running-requests` before lowering `--tp-size`.
- Block-wise FP8 accuracy collapse if `tensor_axis` divides `moe_intermediate_size` to ≤ `block_size_out` (DSV-V3 footnote).
- MoE expert-axis ≥ `ep_size`; can't reduce `--ep-size` below `num_local_experts`.

The recipe's §5 troubleshooting table usually flags these.

### 5. When to add a new tested row to the cookbook

If you run a recipe successfully on a different slice and want to upstream the result: file a PR that adds a row to that recipe's §2.1 hardware matrix and a `#### Multi-host — TPU <slice>` subsection in §2.3 with the measured numbers. Don't list a slice you haven't run end-to-end.

## GKE / SkyPilot identifiers

| Identifier | Value | Used in |
|---|---|---|
| GKE accelerator label (v7x) | `cloud.google.com/gke-tpu-accelerator: tpu7x` | [MiMo-V2.5-Pro GKE manifest](../autoregressive/Xiaomi/MiMo-V2.5-Pro.md) |
| GKE topology label | `cloud.google.com/gke-tpu-topology: 2x2x4` | same |
| Docker image | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` | same |
| SkyPilot resource (v6e) | `tpu-v6e-N` (N ∈ {1, 4, 8, 16, 32, 64}) | [SkyPilot launcher](../deployment/skypilot.md), `scripts/launch_tpu.sh` |
| SkyPilot runtime (v6e) | `runtime_version: v2-alpha-tpuv6e` | `scripts/tpu_resource.sky.yaml`, [`developer_guide/tpu_resources_guide.md`](../../developer_guide/tpu_resources_guide.md) |

## What this page intentionally does NOT cover

- Pricing / region availability — see Google Cloud TPU docs.
- Provisioning / quota workflows — see [`developer_guide/tpu_resources_guide.md`](../../developer_guide/tpu_resources_guide.md) (SkyPilot) and per-recipe GKE manifests.
- Per-recipe TP/DP/EP numbers and the §4 benchmark data points — those live in each recipe's `## 2.1 Hardware Matrix`. This page covers the **scaling math** to adapt them, not the specific tested numbers.
