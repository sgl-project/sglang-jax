# Radix top-k tuning

## Standalone DSA indexer + top-k benchmark

Run the production GLM-5.2 extend shape without starting a model server. The
default is two sequences, each with a 128K prefix and 1K extend, using the
135168-token padded score bucket and exact SparseCore radix top-k:

```bash
PYTHONPATH=python:. python3 benchmark/kernels/radix_topk/bench_dsa_indexer_topk.py \
  --output /tmp/dsa-indexer-topk.json
```

Optionally capture an XProf after compilation and warmup:

```bash
PYTHONPATH=python:. python3 benchmark/kernels/radix_topk/bench_dsa_indexer_topk.py \
  --trace-dir /tmp/tpu_logs/dsa-indexer-topk \
  --profile-iters 5 \
  --output /tmp/dsa-indexer-topk.json
```

The benchmark dispatches only to `--device-index` (default `0`). It includes
indexer score construction, causal/padding masking, and top-k selection, but
excludes projection/RoPE/Hadamard, KV-cache writes, sparse MLA, and all server
overhead. For a small CPU smoke test, select `--topk-impl exact_lax` and reduce
the dimensions. Score experiments can select `--q-dtype float32|bfloat16`.
Use `--score-query-block-size` to tune how many query rows share one score
tile. At long context, the tuned 32-row tile exposes 1024 query-head rows to
MXU while keeping score/top-k pipeline buffers bounded. It stays on the dense
path so XLA can fuse ReLU and the 32-head weighted reduction instead of
repeatedly updating a loop-carried score matrix.

## Decode paged-score A/B benchmark

Compare the original gathered-JAX serial decode loop against the production
paged-cache Pallas scorer followed by one batched top-k. Each sequence
contributes one complete `[1, score_size]` score row; the benchmark checks that
both variants select the same exact top-k set and alternates their timing order:

```bash
PYTHONPATH=python:. python3 benchmark/kernels/radix_topk/bench_dsa_indexer_decode.py \
  --num-seqs 2 \
  --kv-len 131072 \
  --score-size 135168 \
  --block-k 22528 \
  --score-scheduler persistent_two_seq \
  --topk-impl radix \
  --trace-dir /tmp/dsa-indexer-decode-xprof \
  --profile-variant batched \
  --profile-iters 100 \
  --output /tmp/dsa-indexer-decode-metrics.jsonl \
  --summary-output /tmp/dsa-indexer-decode-summary.json
```

This is a TPU benchmark when `--topk-impl=radix`. For a small CPU correctness
smoke test, reduce the dimensions and use `--topk-impl=exact_lax`. The trace
directory is a complete XProf logdir containing `plugins/profile`; omit
`--trace-dir` for timing-only runs.

`--first-dot-dtype`, `--score-scheduler`, and `--page-dma` expose the BF16,
independent-scheduler, and page-DMA ablations. The default page mode verifies
that every active sequence has a contiguous page table before issuing block
DMA and falls back to exact per-page DMA otherwise. The remaining defaults use
the exact FP32, persistent-two-sequence path with batched sequence dots, tuned
for the v7x decode shape above. To profile the two score matrix products
independently, run:

```bash
PYTHONPATH=python:. python3 benchmark/kernels/radix_topk/bench_dsa_score_stages.py \
  --num-seqs 2 \
  --num-heads 32 \
  --head-dim 128 \
  --block-k 2048 \
  --trace-dir /tmp/dsa-score-stages-xprof \
  --output /tmp/dsa-score-stages.jsonl
```

## Tune radix top-k

The runtime lookup is keyed only by `(score_size, topk)` under each TPU device.
It does not depend on a token block or batch-row dimension.

Run the GLM-5.2 128K shape on TPU:

```bash
PYTHONPATH=python:. python3 benchmark/kernels/radix_topk/tune_radix_topk.py \
  --score-sizes 135168 \
  --topks 2048 \
  --output /tmp/radix_topk_tune.jsonl
```

The tuner checks exact membership against `jax.lax.top_k`, profiles every valid
configuration (`8x4` with both TC-tiling modes and `4x8` without TC tiling), and
prints entries ready to paste into `TUNED_RADIX_TOPK_CONFIGS`. TPU v7x firmware
cannot safely execute the `4x8` + TC-tiling combination, so it is rejected before
lowering. Exact selection currently uses one sequential window: multi-window
configs do not preserve global top-k membership at this shape.
