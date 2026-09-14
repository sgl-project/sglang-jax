# Raiden HiCache (experimental)

Raiden HiCache keeps evicted FULL attention KV pages in host memory and restores
matching prefixes into the TPU KV pool. It uses the optional tpu-sync / tpu-raiden
runtime for transfers. The default HiCache backend remains `jax`.

## Supported configurations

- One host, TPU, FULL attention KV pool. Tensor and data parallel layouts are
  supported; each DP rank has a dedicated host pool and transfer manager.
- `--hicache-storage none` enables local device/host caching. External storage is
  not part of this backend.
- `--hicache-write-policy write_through` backs up reused prefixes;
  `write_back` backs up pages when they are evicted from device memory.
- `--hicache-ratio` determines host capacity relative to device KV capacity;
  capacity is divided across DP ranks. Budget host memory in addition to model,
  runtime and request memory.

This release rejects PD disaggregation, multiple hosts, speculative decoding,
SWA and recurrent KV pools. Scheduler overlap may be enabled, but transfers still
wait for the relevant donation to be published and for the KV pool to become
ready. Transfer/forward overlap is not implemented in this release.

## Runtime installation and launch

Install a compatible native wheel using the same distribution channel as Raiden
PD. The wheel is optional and is not installed by the base sgl-jax package. Its
JAX/libtpu compatibility must match the serving environment; the runtime is not
an interchangeable pure-Python dependency.

The validated environment used Python 3.12, JAX/JAXLIB 0.11.1, libtpu 0.0.46.1,
Flax 0.12.9 and `tpu_raiden_jax` 0.0.1.dev20260907100311. The tested wheel's SHA256
is `f2650941afb6a41ac86bfde1ceb2af8216586b63f98a4d4a71369000d88882cc`.
A compatible wheel must expose `KVCacheManager` with `host_blocks_to_allocate`,
`d2h_auto_allocate`, `h2d`, `unlock_blocks`, and completion `Await`/`IsReady`.
Do not assume an older PD-only wheel provides the host allocator API.

```bash
python -m pip install /path/to/compatible-tpu-raiden-jax.whl
python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --device tpu --tp-size 8 --dtype bfloat16 --page-size 128 \
  --hicache-storage none --hicache-transfer-backend raiden \
  --hicache-write-policy write_through --hicache-ratio 2
```

The launcher loads the native extension before JAX. For an embedded Python
entrypoint, call `sgl_jax.raiden.preload_raiden()` before importing JAX or sgl-jax
modules that import JAX. Both the `tpu_sync` and legacy `tpu_raiden` namespaces are
recognized; the Python manager must come from the namespace of the loaded native
extension. Missing internal dependencies are reported without falling back to a
second native runtime.

## Correctness and failure behavior

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
