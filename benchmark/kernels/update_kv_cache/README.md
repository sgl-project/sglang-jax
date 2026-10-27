# Token-wise KV cache slice selection

`update_fused_kv_cache_vectorized` copies one token per DMA slice, including
when the physical cache has 128- or 256-token pages. Its scratch therefore needs
one token per slice. Blocks are bounded at 128 slices to limit the unrolled DMA
start/wait loops while retaining overlap. The existing VMEM headroom still
applies. Callers that copy whole pages retain page-sized scratch selection.

`bench_slice_selection.py` compares the production token-wise update operations
with the former policy: reserve one physical page per slice, then choose as many
slices as fit in VMEM. It uses fused 5D inputs, unique destination slots, padding,
and nonzero initial cache contents. Every variant must match a full-cache NumPy
reference exactly, including untouched entries.

The benchmark donates and reuses the cache, as model execution does. It records
20 device execution durations from the TPU XLA module track after compilation
and warmup; it fails if those timings are unavailable. The measured update
includes slot-mapping construction. It does not measure end-to-end serving.

```bash
python benchmark/kernels/update_kv_cache/bench_slice_selection.py \
  --heads 1 8 32 --tokens 16 128 1024 --page-sizes 128 \
  --output /tmp/kv-slices.jsonl

# Sweep the token-wise block size independently of physical cache pages.
python benchmark/kernels/update_kv_cache/bench_slice_selection.py \
  --heads 8 --tokens 128 1024 --page-sizes 128 \
  --slices 16 32 64 128 256 --output /tmp/kv-slice-sweep.jsonl

# Check page-size-one, larger head dimensions, and contiguous writes too.
python benchmark/kernels/update_kv_cache/bench_slice_selection.py \
  --heads 8 --tokens 16 128 1024 --page-sizes 1 \
  --output /tmp/kv-slices-page1.jsonl
python benchmark/kernels/update_kv_cache/bench_slice_selection.py \
  --heads 8 --tokens 129 1024 --page-sizes 256 --head-dims 256 \
  --dtype float32 --output /tmp/kv-slices-fp32.jsonl
python benchmark/kernels/update_kv_cache/bench_slice_selection.py \
  --heads 8 --tokens 129 1024 --page-sizes 128 --pattern contiguous \
  --output /tmp/kv-slices-contiguous.jsonl
```

## TPU v6e measurements (2026-10-27)

One TPU v6 lite device; JAX/jaxlib 0.11.1, libtpu 0.0.46.3. These are BF16,
head-dimension-128 updates with randomly placed slots and an 8192-token cache.
Heads count the original K heads (the fused buffer contains both K and V).

| Heads | New tokens | Cache page size | Former median, us | Selected median, us | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 128 | 73.306 | 37.349 | 1.96x |
| 8 | 128 | 128 | 10.399 | 5.739 | 1.81x |
| 8 | 1024 | 128 | 41.706 | 37.579 | 1.11x |
| 32 | 128 | 128 | 10.656 | 5.943 | 1.79x |
| 32 | 1024 | 128 | 65.566 | 38.582 | 1.70x |
| 8 | 1024 | 1 | 37.535 | 37.601 | 1.00x |

For 8 heads, 128 tokens, and 128-token pages, explicit scratch falls from
66,584,576 bytes to 524,288 bytes. For 1 head and 1024 tokens, block selection
falls from 1022 to 128 slices; compilation took 3.12 s versus 0.39 s in this run.
Compilation can increase when the former VMEM limit selected a much smaller
block: the 32-head, 128-token case took 0.13 s versus 0.37 s.

All 32 variants across 16 workloads passed exact full-cache comparisons. The
16-token and page-size-one cases changed by less than 1% in most comparisons.
The FP32, dimension-256, page-size-256 cases changed from 385.82 to 388.49 us at
129 tokens and from 479.02 to 457.67 us at 1024 tokens. These measurements cover
one TPU generation and do not establish a serving-throughput improvement.
