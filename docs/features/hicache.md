# HiCache

HiCache extends prefix KV caching from device memory (L1) to host memory (L2).
Device eviction can retain a host copy; a later prefix hit restores that KV
instead of recomputing prefill. Host eviction releases the copy, so a later miss
in both tiers falls back to prefill computation.

## Configuration

`--hicache-storage none` enables local device/host caching and selects the unified
radix cache automatically. HiCache is disabled by default. External storage
(`--hicache-storage file`) is not supported yet.

| Option | Behavior |
| --- | --- |
| `--hicache-transfer-backend jax` | Default JAX device/host copy path; no Raiden wheel required |
| `--hicache-transfer-backend raiden` | Optional native TPU transfer path |
| `--hicache-ratio 2` | Host KV capacity relative to device KV capacity |
| `--hicache-write-policy write_through` | Back up reused prefixes |
| `--hicache-write-policy write_back` | Back up pages at device eviction |
| `--hicache-write-through-threshold` | Reuse threshold for write-through backup |

Budget host memory for the KV cache in addition to the model and runtime.

## JAX backend

Start with the default backend in an installed sglang-jax environment:

```bash
python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --device tpu --tp-size 8 --dtype bfloat16 --page-size 128 \
  --hicache-storage none --hicache-transfer-backend jax \
  --hicache-write-policy write_through --hicache-ratio 2
```

## Raiden backend (experimental)

Raiden replaces the transfer backend while retaining HiCache prefix lookup,
write policies, and eviction/restore behavior.

### In scope

- Single-host TPU, FULL MHA/GQA attention KV, tensor/data parallel layouts.
- Dedicated host pool and transfer manager per DP rank.
- Write-through and write-back with native completion checks.
- Synchronous transfer: KV ready → D2H/H2D → completion → scheduling continues.

### Not supported yet

- PD, multi-host, speculative decoding, SWA, recurrent, and MLA KV pools.
- Background transfer/forward overlap, including when scheduler overlap is enabled.
- External Store integration and repeated native-manager recreation.

### Build and install Raiden

Use a Linux x86-64 environment with Python 3.12, Git, and network access for
Bazel dependencies. Run these commands from the sglang-jax checkout with the
serving environment activated. TPU access is required for the native tests and
serving steps, but not for compiling the wheel.

The existing build script pins TPU Sync and applies the JAX 0.11.1 compatibility
patch from [PR #1602](https://github.com/sgl-project/sglang-jax/pull/1602).
Installing newer Python packages alone does not update the native JAX/XLA ABI.

```bash
# Install sglang-jax and the runtime versions used by the build script.
python3.12 -m pip install -e ./python
python3.12 -m pip install 'jax==0.11.1' 'jaxlib==0.11.1' 'libtpu==0.0.46.1'

# Use a dedicated source checkout: the build script applies a compatibility patch.
export RAIDEN_SRC="$HOME/src/tpu-sync-hicache"
export RAIDEN_CACHE_ROOT="$HOME/.cache/raiden-hicache"
git clone https://github.com/google/tpu-sync.git "$RAIDEN_SRC"
git -C "$RAIDEN_SRC" checkout 6d43141191210b73359d22743be619532ad587dc

# The upstream build bootstraps Bazel. Keep its output on a disk with ample space.
export BAZEL_OUTPUT_BASE="$HOME/.cache/tpu-sync-bazel-output"
export PYTHON_BIN=python3.12
bash scripts/disaggregation/build_raiden_wheel.sh

# Install the wheel selected by the build script's READY marker.
wheel_dir="$RAIDEN_CACHE_ROOT/$(git -C "$RAIDEN_SRC" rev-parse HEAD)-jax0.11.1-jaxlib0.11.1-libtpu0.0.46.1"
wheel=$(cat "$wheel_dir/READY")
python3.12 -m pip install --no-deps "$wheel_dir/$wheel"
```

Re-running the build script reuses its validated wheel cache. To use an existing
wheel instead, install it in the same runtime environment with `--no-deps`.
The wheel is optional and is not installed by the base package.

Check extension loading and the required host-cache APIs in a fresh process:

```bash
python3.12 - <<'PYCODE'
from sgl_jax.raiden import preload_raiden, get_raiden_kv_cache_manager

preload_raiden()  # Must precede JAX imports.
manager = get_raiden_kv_cache_manager()
for name in ("d2h_auto_allocate", "h2d", "unlock_blocks"):
    assert hasattr(manager, name), name
print("Raiden loader and host-cache API checks passed")
PYCODE
```

This import check does not exercise DMA. On an otherwise idle v7x-8 host, run
both native tests sequentially before serving:

```bash
python3.12 test/manual/raiden_hicache_native.py --dp-size 1 --rounds 200
python3.12 test/manual/raiden_hicache_native.py --dp-size 2 --rounds 200
```

The runtime must also support `host_blocks_to_allocate` and completion
`Await`/`IsReady`; the native regression exercises allocation and completion.
Older PD-only wheels may lack these host-cache APIs.

### Launch with Raiden

```bash
python3.12 -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --device tpu --tp-size 8 --dtype bfloat16 --page-size 128 \
  --hicache-storage none --hicache-transfer-backend raiden \
  --hicache-write-policy write_through --hicache-ratio 2
```

The launcher preloads the native extension before JAX. Embedded entrypoints must
call `sgl_jax.raiden.preload_raiden()` before importing JAX. Both `tpu_sync` and
legacy `tpu_raiden` namespaces are recognized; the Python API and native extension
must come from the same namespace.

### Correctness and failure behavior

Device eviction retains a completed host copy. A host hit reserves the complete
restore chain and pins its source pages before allocating device pages; allocation
can itself trigger write-back and host eviction. The tree publishes restored KV
only after native completion. Host eviction releases native chunks, and a later
request for an evicted host prefix recomputes it as a cache miss.

A native error does not prove that every transfer has stopped accessing memory.
The backend quarantines affected resources and raises an error; restart the
serving process rather than reclaiming those pages and continuing. The tested
runtime has no explicit cancel/close contract. Normal cache flush/reset drains
transfers and releases host chunks, but does not recreate the native manager.

## Regression tests

Run CPU control-plane tests without a native wheel:

```bash
python -m pytest python/sgl_jax/test/mem_cache/test_raiden_hicache.py -q
```

In a fresh TPU process with the compatible wheel installed, run the manual native
regression once for each layout. Do not run these processes concurrently on the
same TPU allocation:

```bash
python test/manual/raiden_hicache_native.py --dp-size 1 --rounds 200
python test/manual/raiden_hicache_native.py --dp-size 2 --rounds 200
```

The native test writes known values, backs them up, evicts device pages, clears
device KV, restores and checks exact values. It also fills host capacity and
checks that pressure on one DP rank preserves other ranks' host allocations.
CPU fake-DMA tests cover publication, pins, failure isolation and allocation
rollback; they do not replace the native test.

To reproduce the serving comparison, launch JAX and Raiden servers sequentially
with identical model, seed and scheduler settings. For the bounded pressure
sequence, set `--max-total-tokens 2048 --context-length 1024 --max-seq-len 1024`,
`--page-size 128 --hicache-ratio 4 --disable-overlap-schedule --random-seed 0`.
Record the JAX run, restart with `--hicache-transfer-backend raiden`, and record
that run. Repeat for `--hicache-write-policy write_back` as needed.

```bash
python test/manual/raiden_hicache_serving.py --out jax.json
# Restart the server with the Raiden backend and otherwise identical settings.
python test/manual/raiden_hicache_serving.py --out raiden.json
python test/manual/raiden_hicache_serving.py --out raiden.json --reference jax.json
```

This sends 13 requests alternating five prefixes, checks cache reuse and flush,
and compares generated text and output logprobs. Cached-token metadata alone does
not establish that native restoration occurred; the native regression separately
checks eviction, overwritten device buffers and exact restoration. Use matching
scheduler modes for numerical comparisons.
