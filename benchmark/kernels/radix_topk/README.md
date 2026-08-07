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
tile. At long context, the tuned tile keeps the all-head temporary below the
dense-path threshold so XLA can fuse ReLU and the 32-head weighted reduction
instead of repeatedly updating a loop-carried score matrix.

## Decode sequence-tile A/B benchmark

Compare the original serial decode loop against the sequence-level ping-pong
pipeline. Each sequence contributes one complete `[1, score_size]` tile; the
benchmark checks that both variants select the same exact top-k set and then
alternates their timing order:

```bash
PYTHONPATH=python:. python3 benchmark/kernels/radix_topk/bench_dsa_indexer_decode.py \
  --num-seqs 2 \
  --kv-len 131072 \
  --score-size 135168 \
  --topk-impl radix \
  --trace-dir /tmp/dsa-indexer-decode-xprof \
  --profile-variant pipeline \
  --profile-iters 100 \
  --output /tmp/dsa-indexer-decode-metrics.jsonl \
  --summary-output /tmp/dsa-indexer-decode-summary.json
```

This is a TPU benchmark when `--topk-impl=radix`. For a small CPU correctness
smoke test, reduce the dimensions and use `--topk-impl=exact_lax`. The trace
directory is a complete XProf logdir containing `plugins/profile`; omit
`--trace-dir` for timing-only runs.

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
