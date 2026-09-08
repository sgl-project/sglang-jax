# Experimental Raiden PD scheduler overlap

`--disaggregation-enable-overlap-schedule` enables scheduler overlap independently
on a Raiden prefill or decode server. The default remains disabled. This is an
experimental implementation; CPU tests do not establish native TPU buffer safety
or performance, and device validation is required before deployment.

## Configuration

Add these flags to an otherwise working Raiden PD launch:

```shell
--disaggregation-use-raiden \
--disable-radix-cache \
--disaggregation-enable-overlap-schedule
```

Keep `--disaggregation-mode prefill` or `decode` on the corresponding role.
Prefill requests must carry a `bootstrap_room`, as provided by the PD router.
Chunk transfer remains independently controlled by
`--disaggregation-enable-chunk-prefill-transfer`; use matching chunk settings on
both roles. The bootstrap protocol is unchanged.

The initial supported scope is one JAX process per serving instance with a
full-attention text generation model and ChunkCache. Pathways PD, multiple hosts
per instance, hybrid attention, multimodal models, speculative decoding, LoRA,
and HiCache are rejected. The flag conflicts with `--disable-overlap-schedule`.
Removing it restores the ordinary PD loop.

The early runtime loader prefers `tpu_sync`, falling back to `tpu_raiden` only
when the top-level package is absent. Internal import and ABI failures propagate.
The native extension must load before JAX; `sgl_jax.launch_server` handles this
when Raiden is requested. Install a wheel built for the exact Python, platform,
JAX, JAXLIB and libtpu versions in the serving image.

## Ordering and ownership

- Decode uses the ordinary future-token/result pipeline, including the first
  EXTEND after KV reception. New native receive admission fences previous KV
  compute, because Raiden writes are outside JAX's dependency chain. Polling
  existing receives and steady decode do not require that admission fence.
- Prefill snapshots each scheduled request's transfer identity, page IDs, token
  range and final-chunk status before another round mutates the request. The
  forward thread waits for KV readiness before publishing its result and before
  donating the pool to another forward. CPU scheduling can proceed during this
  wait; no old donated JAX array is retained in a handoff descriptor.
- Prefill pages are released only after both queued compute ownership and sender
  completion have retired. Cancellation cannot publish another chunk from an
  already queued forward. Pause drains pending results before retraction.
- When a chunk sender has pending registrations, prefill pauses new scheduling
  while continuing result resolution and transport polling. This is conservative
  global backpressure, including across local DP ranks. Native read windows remain
  independently bounded by the transport.
- Optional KV debug extraction waits for the newest queued forward's donation
  barrier. Debug extraction adds synchronization and must be disabled for timing
  comparisons.

These fences deliberately favor correctness. This implementation does not claim
unrestricted asynchronous TPU compute/transfer concurrency. The registered raw
buffer addresses, disjoint-page accesses during donation, and release ordering
must also pass native-device probes.

With PD timing logging enabled, `decode_ready` means received KV entered the
decode waiting queue. `first_token` is marked after the first real output token
is resolved and appended, rather than at enqueue. `decode_start` measures the
gap between these events; this is server-side availability, not client TTFT.

## Validation sequence

1. Build a fixed Raiden source revision and patch with the serving runtime. Verify
   extension preload, runtime versions, TPU discovery, manager construction, and
   bitwise float32/bfloat16 transfer. Publish wheel, SHA256 and provenance; publish
   the cache READY marker only after smoke tests and download verification pass.
2. On two single-chip v6e hosts, reproduce ordinary PD with the same wheel and
   model before enabling overlap. Run native page-isolation, donation-address,
   late-completion and cancellation probes.
3. Compare four role combinations (both off, D only, P only, both on), with chunk
   transfer independently off/on. Start with DP=1. Check deterministic token IDs,
   short requests, multi-chunk prompts, concurrency, cancellation, timeout,
   retraction and a healthy sentinel request after failures. Record pool usage
   after requests drain. Larger local DP configurations need larger TPU shapes.
4. After correctness passes, alternate warmed A/B runs with identical requests
   and at least three repetitions. Record client TTFT, TPOT, p95/p99, goodput and
   raw request samples. Capture separate profiler runs to establish actual CPU
   scheduling, TPU compute and Raiden transfer overlap. Do not infer device
   overlap from CPU tests or attribute D-only improvements to transport.

CPU regression command (install the project's test dependencies first):

```shell
PYTHONPATH=python python -m pytest -q \
  test/srt/disaggregation \
  python/sgl_jax/test/test_tp_worker_overlap_thread.py \
  python/sgl_jax/test/test_scheduler_chunked_ownership.py \
  python/sgl_jax/test/test_scheduler_retraction.py \
  python/sgl_jax/test/test_scheduler_idle_check.py
```
