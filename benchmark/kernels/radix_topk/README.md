# Radix top-k tuning

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
