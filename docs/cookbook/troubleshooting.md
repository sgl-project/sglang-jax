---
title: "Troubleshooting"
description: "Cross-recipe generic failure modes — startup OOM, multi-node hang, JIT cache miss, SWA pool exhaustion."
---

# Troubleshooting

Common failure modes across all SGL-JAX cookbook recipes, with the fix and what to grep in logs to confirm. Model-specific gotchas live in each recipe's §7; this page covers everything else.

## Startup

### First request takes ~4 minutes

**Symptom**: Server logs `Uvicorn running on...`, but the first HTTP request hangs for minutes before any token streams back.

**Cause**: JAX/Pallas compilation cache is empty. Every kernel is being precompiled.

**Fix**: Make sure `JAX_COMPILATION_CACHE_DIR` is set **and** points to a path that persists across restarts (host volume mount in Docker, `emptyDir` won't survive pod restart in GKE — use a PVC or hostPath if you need cache persistence between job runs).

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server ...
```

### OOM during weight load

**Symptom**: Process exits with `RESOURCE_EXHAUSTED` or `XlaRuntimeError: Out of memory while trying to allocate ...` before the server reaches the warmup phase.

**Cause**: `--tp-size` mismatch (weights too big per device), or `--mem-fraction-static` too aggressive.

**Fix**:
1. Check `--tp-size` matches device count: `python -c "import jax; print(len(jax.devices()))"` inside the container.
2. For v7x, remember each chip exposes 2 JAX devices — see [`base/tpu-topology-reference.md`](base/tpu-topology-reference.md).
3. Lower `--mem-fraction-static` (default 0.88 on TPU). 0.85 is conservative; only raise to 0.95 when nothing else runs on the host.

### "Architecture not supported"

**Symptom**: `ValueError: Model architectures [...] are not supported for now. Supported architectures: dict_keys([...])`.

**Cause**: The HF config's `architectures` field doesn't match any class registered by `python/sgl_jax/srt/models/registry.py`.

**Fix**:
- Confirm your model is in the supported list — see the cookbook [autoregressive index](autoregressive/index.md) or [`python/sgl_jax/srt/models/`](https://github.com/sgl-project/sglang-jax/tree/main/python/sgl_jax/srt/models).
- Some Qwen-derived models (MiMo-7B → Qwen2, MiMo-V2.5-Pro → MiMo-V2) reuse base classes via inheritance — the HF config's `architectures` field still has to match the actual class name SGL-JAX registers, not the inherited one. If your model's config lists a class SGL-JAX doesn't recognise, override `--json-model-override-args '{"architectures": ["<RegisteredClass>"]}'`.

## Multi-node

### Hang at `jax.distributed.initialize`

**Symptom**: Pods come up but the launch command sits silent for minutes, no logs after the import banner.

**Cause**: Rank-0 unreachable from other nodes (or `--dist-init-addr` resolves to the wrong host).

**Fix**:
1. Verify the rank-0 IP/port is reachable from every other node:
   ```bash
   # on rank-N for N != 0
   nc -zv <rank0-host> <dist-init-port>
   ```
2. On GKE, confirm headless Service is correctly resolving:
   ```bash
   kubectl exec <pod-1> -- nslookup <job>-0.<job>-headless-svc
   ```
3. Bump `--dist-timeout` if you're on a slow scheduler.

### "TPU_PROCESS_ADDRESSES" mismatch

**Symptom**: First step crashes with `Mismatched TPU process count` or similar JAX runtime error.

**Cause**: `TPU_PROCESS_ADDRESSES` length ≠ `--nnodes`, or the rank-0 entry doesn't match `MASTER_ADDR`.

**Fix**: `TPU_PROCESS_ADDRESSES` must enumerate every node's `<host>:8471` exactly `--nnodes` times. Check by:

```bash
echo $TPU_PROCESS_ADDRESSES | tr ',' '\n' | wc -l   # must equal $NNODES
```

## At runtime

### SWA pool exhaustion (MiMo, hybrid-attention models)

**Symptom**: Long-running requests start failing; server logs show `swa token usage` consistently near 100%.

**Cause**: Sliding-window-attention pool is too small for the concurrent decode load. The `--swa-full-tokens-ratio` flag controls `swa_tokens_per_layer / full_tokens_per_layer` — see [`base/launch-flags-reference.md`](base/launch-flags-reference.md) for the per-layer semantics (it is **not** a pool fraction).

**Fix**: Either raise `--swa-full-tokens-ratio` (gives each SWA layer more KV slots relative to full layers) or lower `--max-running-requests`. Conversely, if the full-attention side is the bottleneck (`full token usage` near 100%), lower the ratio.

### High latency at TTFT only

**Symptom**: TTFT large but ITL normal once decoding starts.

**Cause**: Prefill chunking too aggressive, or `--max-prefill-tokens` too low for the prompt size.

**Fix**: Raise `--chunked-prefill-size` (default 4096) or `--max-prefill-tokens` (default 16384). For very long prompts, both flags interact.

### Throughput collapses at high concurrency

**Symptom**: Output tokens/s plateaus or drops as concurrency rises past some threshold.

**Cause**: `--max-running-requests` not set, scheduler over-admits and KV cache thrashes.

**Fix**: Set `--max-running-requests` to a value matched to your KV pool size. As a starting point, per-recipe authoritative values: MiMo-V2-Flash uses 128 on v6e-16, MiMo-V2.5-Pro uses 512 on v6e-64 / v7x-16 (see those recipes' Key Flags tables for the up-to-date numbers). For dense models start at 256 and tune.

## MoE-specific

### "fused" MoE backend slower than expected on small EP

**Symptom**: At EP ≤ 8 on a single host, `--moe-backend fused` gives lower output tok/s than `epmoe`.

**Cause**: At small EP the per-device expert count is high enough that the GMM path (`epmoe`) wins on memory bandwidth. The fused Pallas kernel's overhead pays off when EP ≥ 16.

**Fix**: Use `--moe-backend epmoe` on single-host v7x-8 and small slices; switch to `--moe-backend fused` for multi-host setups. MiMo-V2-Flash recipe has measured numbers — see [its §6.2](autoregressive/Xiaomi/MiMo-V2-Flash.md#62-throughput--bench_serving).

## Tokenizer / model path

### Tokenizer missing

**Symptom**: `OSError: Can't load tokenizer for <model>` or `tokenizer.json not found`.

**Cause**: Some model checkpoints (Grok-2) ship without a tokenizer file.

**Fix**: Pass `--tokenizer-path <community-tokenizer-repo>` explicitly. Example: Grok-2 uses `alvarobartt/grok-2-tokenizer` — see [`autoregressive/grok2.md`](autoregressive/Grok/Grok2.md).

### `--trust-remote-code` required

**Symptom**: Model load fails with HuggingFace warning about untrusted code.

**Fix**: Every model with a custom `modeling_*.py` (Qwen, MiMo, Bailing, Kimi-Linear, ...) needs `--trust-remote-code`. When in doubt, set it.

## Compilation cache hygiene

The JIT cache at `JAX_COMPILATION_CACHE_DIR` keys on the full kernel shape — changing `--page-size`, `--chunked-prefill-size`, `--tp-size`, or model shape invalidates cached entries (you'll see ~4 min recompile for the changed kernels). Don't share a cache dir across recipes with different shapes; give each recipe its own subdirectory.

## When to file an issue

If symptoms don't match anything above and the recipe's command line is taken verbatim from the cookbook, [file an issue](https://github.com/sgl-project/sglang-jax/issues) including:

1. Cookbook recipe page + commit SHA you're at.
2. Full launch command + env vars.
3. First ~200 lines of server log.
4. Output of `python -c "import jax; print(jax.devices(), jax.__version__)"` from inside the container.
