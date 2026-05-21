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
| `v6e-4` | 1 | 4 | 4 | [Qwen-7B-Chat](../autoregressive/qwen.md), [Qwen3-8B / 32B](../autoregressive/qwen3.md) |
| `v6e-16` (`4x4`) | 4 | 4 | 16 | [MiMo-V2-Flash](../autoregressive/mimo-v2-flash.md) (multi-node) |
| `v6e-32` | 8 | 4 | 32 | [Grok-2](../autoregressive/grok2.md) |
| `v6e-64` (`4x4x4`) | 16 | 4 | 64 | [MiMo-V2.5-Pro](../autoregressive/mimo-v2.5-pro.md) |
| `v7x-8` | 1 | 4 | 8 | [MiMo-V2-Flash](../autoregressive/mimo-v2-flash.md) (single-node) |
| `v7x-16` (`2x2x4`) | 4 | 4 | 32 | [MiMo-V2.5-Pro](../autoregressive/mimo-v2.5-pro.md) |

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

## GKE / SkyPilot identifiers

| Identifier | Value | Used in |
|---|---|---|
| GKE accelerator label (v7x) | `cloud.google.com/gke-tpu-accelerator: tpu7x` | [MiMo-V2.5-Pro GKE manifest](../autoregressive/mimo-v2.5-pro.md) |
| GKE topology label | `cloud.google.com/gke-tpu-topology: 2x2x4` | same |
| Docker image | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` | same |
| SkyPilot resource (v6e) | `tpu-v6e-N` (N ∈ {1, 4, 8, 16, 32, 64}) | [SkyPilot deployment](../deployment/skypilot.md), `scripts/launch_tpu.sh` |
| SkyPilot runtime (v6e) | `runtime_version: v2-alpha-tpuv6e` | `scripts/tpu_resource.sky.yaml`, [`developer_guide/tpu_resources_guide.md`](../../developer_guide/tpu_resources_guide.md) |

## What this page intentionally does NOT cover

- Pricing / region availability — see Google Cloud TPU docs.
- Provisioning / quota workflows — see [`developer_guide/tpu_resources_guide.md`](../../developer_guide/tpu_resources_guide.md) (SkyPilot) and per-recipe GKE manifests.
- Per-recipe TP/DP/EP picks — those live in each recipe's *Hardware Requirements* table.
