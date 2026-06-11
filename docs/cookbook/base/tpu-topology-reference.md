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

Each cookbook recipe ships **one** TPU configuration — the slice we ran the §4 numbers on. The recipe doesn't claim that's the only valid slice; it's the only slice we measured. This section is the math for sizing your own slice when you have different hardware.

### Step 1 — Compute per-chip HBM footprint

Three pieces add up per JAX device:

```
weight_GiB_per_device   = params_billions × dtype_bytes / tp_size
kv_GiB_per_device       = (kv_bytes_per_token × max_running_requests × max_context_length) / tp_size
activation_peak_GiB     = chunked_prefill_size × hidden_dim × dtype_bytes × layer_count_factor / tp_size
```

| Symbol | Where to find it |
|---|---|
| `params_billions` | HF `config.json` (`num_parameters`) or the model card. For MoE, use the **total** params figure, not the activated count — every expert lives in HBM. |
| `dtype_bytes` | `bfloat16` → 2; FP8 → 1; channel-wise FP8 with BF16 fall-back layers → mostly 1, a few 2 |
| `kv_bytes_per_token` | `2 × num_kv_heads × head_dim × num_hidden_layers × dtype_bytes`. For MLA models the head dim is the latent dim (DSV-V3: 512 × 1 × 61 × 2 / kv-group = much smaller than dense GQA). |
| `max_running_requests` | Recipe §2.3 launch flag; start at the recipe's value. |
| `max_context_length` | Recipe §2.3 `--context-length` (or model native max if not set). |
| `chunked_prefill_size` | Recipe §2.3 `--chunked-prefill-size`; the EXTEND precompile peak is roughly `chunked_prefill_size × hidden_dim × dtype_bytes`. |
| `layer_count_factor` | 1–3 in practice; depends on attention layout. Use 2 as a starter when you can't measure. |

Add the three terms. Multiply by `1 / mem_fraction_static` to get the required HBM/device, then compare against the table at the top of this page:

```
v6e:   32 GiB / device
v7x:   96 GiB / device     (each chip exposes 2 devices, see "Why v7x is special")
v5p:   95 GiB / device
v5e:   16 GiB / device
v4:    32 GiB / device
```

If the sum exceeds available HBM, raise `tp_size` (more chips → less weight per chip) or lower `max_running_requests` / `max_context_length` / `chunked_prefill_size`. The recipe's §2.4 Configuration Tips usually documents which knob is safe to push first.

**Worked example**: serve Qwen3-8B on v6e-4 with the recipe defaults (`--max-running-requests 256`, `--context-length 32768`, `--chunked-prefill-size 2048`, `--tp-size 4`):

- weights: `8 × 2 / 4 = 4 GiB/device`
- KV: `2 × 8 × 128 × 36 × 2 = 147 KB/token × 256 × 32768 / 4 = 308 GiB/device` (impossible at full context × max concurrency — recipe assumes most requests are far shorter)
- Practical: at `avg_context × 32` concurrency the KV pool needs maybe ~6 GiB; activation peak ~5 GiB
- Total ~15 GiB/device vs 32 GiB/device on v6e — fits comfortably; doubling `--max-running-requests` is safe.

The KV term is the one that surprises — most "won't fit" failures are KV pool exhaustion, not weight footprint.

### Step 2 — Pick the parallelism shape

Default rule:

```
tp_size = total_jax_devices = chip_count × devices_per_chip   # v7x = 2, everything else = 1
ep_size = tp_size                                              # one expert shard per device, for MoE
dp_size = 1                                                    # unless the recipe §2.4 says otherwise
```

`tp_size = ep_size` is the textbook starting point for MoE. `dp_size` is **almost always pinned by model-specific constraints**, not by free choice:

- **GQA `num_kv_heads`**: tensor axis (`tp_size / dp_size`) must divide `num_kv_heads`. Check HF `config.json` for the value. Llama 3.1 8B has 8 → tensor axis ∈ {1, 2, 4, 8}.
- **GLA `num_groups`** (linear-attention models): tensor axis ≤ `num_groups`. Ling-2.6 / Kimi-Linear have 8 → on v6e-64 you need `--dp-size 8` so tensor axis = 64/8 = 8.
- **FP8 block-quant constraint** (DSV-V3 / R1): per-rank shared-expert `out_dim = moe_intermediate_size / tensor_axis` must be > `block_size_out` (default 128). DSV-V3 has `moe_intermediate_size=2048`, so tensor axis ≤ 16; combined with the dense-MLP scale grid constraint (`144 % tensor == 0`), only tensor axis = 8 works → `--dp-size 8` on v6e-64.
- **Pre-sharded checkpoint** (Grok-2): `--tp-size` must be a multiple of the pre-shard count. Grok-2 ships TP-{000..007} files → `--tp-size` ∈ {8, 16, 24, 32, ...}.

Always read the target recipe's §2.4 before setting `--dp-size` / `--tp-size` — the constraint is model-specific and usually forces exactly one choice.

### Step 3 — Constraint checklist before launch

Before you hit "go", verify against `python/sgl_jax/srt/configs/model_config.py` + the HF `config.json`:

| # | Check | Failure mode | Source of truth |
|---|---|---|---|
| 1 | `tp_size % num_kv_heads == 0` (GQA models) | KV head replication fails, shape error at first prefill | HF `config.json` `num_key_value_heads` |
| 2 | `num_local_experts % ep_size == 0` (MoE) | Expert dim mis-shards at load time | HF `config.json` `num_local_experts` (or equivalent) |
| 3 | `moe_intermediate_size % 512 == 0` if using `--moe-backend fused` | Fused kernel `tile_n` assert at startup | HF `config.json` `moe_intermediate_size` |
| 4 | FP8 block-quant: `per_rank_shared_expert_out_dim > block_size_out` | Block-wise quant accuracy collapse (silent for `fused`, asserted for `epmoe`) | HF `config.json` `quantization_config.weight_block_size` + `moe_intermediate_size` |
| 5 | Hybrid recurrent models (Ling-2.6, Kimi-Linear) require `--disable-radix-cache` | Server asserts on startup | Recipe §2.4 |
| 6 | MLA models (DeepSeek family) require `--page-size >= 2` | MLA backend asserts on startup | Recipe §2.4 |

The recipe's §5 Troubleshooting table will have a symptom row for each — start there if launch fails.

### Step 4 — If launch hits OOM

Tune in this order (each step is safer than the next):

1. **Lower `--max-running-requests`** — shrinks KV pool, the most common OOM culprit. Halve it and retry.
2. **Lower `--chunked-prefill-size`** — bounds activation peak during prefill. Drop from 2048 to 1024 to 512 progressively.
3. **Lower `--context-length`** — model native max is usually overkill for production traffic; `--context-length 32768` (vs 256K) frees substantial KV budget.
4. **Lower `--mem-fraction-static`** — last resort. Default 0.88 leaves ~12% headroom; dropping to 0.85 helps tiny overruns but creates fragmentation noise.
5. **Raise `--tp-size`** — needs more chips. Re-check the constraints in Step 3 because changing `tp_size` cascades into `dp_size` / `ep_size`.

### Step 5 — v6e ↔ v7x conversion

One rule: v7x has **2 JAX devices per chip**. So:

```
v7x_chip_count   = v6e_chip_count / 2
v7x_tp_size      = v6e_tp_size                # same tp_size, half the chips
v7x_HBM_per_chip = 192 GiB total, 96 GiB / JAX device
```

v7x's higher per-device HBM means a model that needs a v6e-32 slice (32 chips × 32 GiB = 1 TiB chip HBM) fits on a v7x-8 (8 chips × 192 GiB = 1.5 TiB chip HBM), with `--tp-size 16` (8 chips × 2 devices). Re-derive Step 1 from the v7x numbers — don't just port `--tp-size` over without recomputing the per-device math.

### Step 6 — PR back when you've measured a new slice

If you size a slice this section's math suggests, run the recipe end-to-end, and get §4-equivalent measurements, file a PR adding your slice as a new row in the target recipe's §2.1 Hardware Matrix and a `#### Multi-host — TPU <slice>` subsection in §2.3. The cookbook's "tested config only" rule means we don't list a slice until someone has run it — your measurement makes it shippable.

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
