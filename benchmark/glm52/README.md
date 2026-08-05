# GLM-5.2 exact DSA DP16 benchmark

This directory contains the correctness-first GLM-5.2 exact DSA serving checks
for the global `TP16 / DP16 / EP16` topology. Sequence parallelism is disabled:
`--tp-size 16` is the global/sum TP size, so the tensor size inside each DP rank
is one.

## Validated baseline

Falcon experiment `exp-5lpq3yzomq` succeeded on `tpu-training-antgroup` with:

- 2 replicas, 8 v7x devices per replica, topology `2x2x2`;
- 32 requests globally, 2 requests per DP rank;
- a shared 131,072-token prefix, 1,024-token extend, and 1,024-token decode;
- exact Pallas DSA attention and the correctness-first `jax.lax.top_k` path;
- BF16 KV cache, page size 64, and sequence parallelism disabled.

The run produced a 131,072-token cache hit for all 32 requests. Its measured
TTFT mean was 32.13 seconds, TPOT p50 was 249.31 ms, and output throughput was
114.10 token/s. The no-SP result was effectively identical to the historical
SP-enabled control.

With `--max-total-tokens 135168`, the measured C32 extend was admitted as two
C16 waves. This was not an extend chunking issue: each request processed its
full 1,024-token extension in one pass. The shared-prefix working set exactly
filled the per-DP pool:

```text
131072 + 2 * (1024 extend + 1024 decode) = 135168
```

The scheduler uses a strict exact-fit admission check. The next run therefore
uses `135296`, two 64-token pages above the exact working set.

## Next-run server command

The Falcon manifest supplies `WORLD`, `RANK`, and `MASTER_ADDR`. The standalone
server command is:

```bash
serve_args=(
  --model-path /models/GLM-5.2-FP8
  --trust-remote-code
  --device tpu
  --dtype bfloat16
  --kv-cache-dtype bf16
  --attention-backend dsa_sparse
  --dsa-sparse-impl exact
  --dsa-topk-impl exact_lax
  --dsa-use-pallas
  --page-size 64
  --chunked-prefill-size 2048
  --max-prefill-tokens 34816
  --context-length 135168
  --tp-size 16
  --dp-size 16
  --dp-schedule-policy round_robin
  --ep-size 16
  --moe-backend epmoe
  --mem-fraction-static 0.83
  --max-running-requests 32
  --max-total-tokens 135296
  --precompile-bs-paddings 32
  --precompile-token-paddings 32768
  --skip-server-warmup
  --random-seed 3
  --stream-output
  --stream-interval 1
  --nnodes "$WORLD"
  --node-rank "$RANK"
  --dist-init-addr "$MASTER_ADDR:25000"
  --host 0.0.0.0
  --port 30000
)
python3 -m sgl_jax.launch_server "${serve_args[@]}"
```

Do not add `--enable-sequence-parallel`. Do not set
`SGLANG_JAX_SKIP_GCSFUSE_WARMUP=1` or `SGLANG_MOE_DISABLE_BULK_READ=1`; the
validated startup path uses GCSFuse warmup and MoE bulk reads. Although
`max_prefill_tokens` has two pages of global admission headroom, the compiled
extend shape remains 32,768 tokens because `chunked_prefill_size=2048` is a
per-DP-rank limit and the worker shape is `2048 * DP16`.

## Run and validate

Submit the checked-in manifest:

```bash
falcon workflow exp submit \
  -f benchmark/glm52/falcon_e2e_128k_shared_prefix.yaml \
  --output json | tee /tmp/glm52-dsa-submit.json

export EXP_ID="$(jq -r '.ids.exp_id' /tmp/glm52-dsa-submit.json)"
falcon workflow exp wait "$EXP_ID" --timeout 2h --output json \
  | tee /tmp/glm52-dsa-wait.json
```

The manifest first runs `eval_basic_generation.py`, then
`bench_dsa_cache_hit.py`. A successful next run must satisfy all of the
following:

- Falcon selects `tpu-training-antgroup` and reaches `SUCCEEDED`;
- server info reports `enable_sequence_parallel=false`, `tp_size=16`,
  `dp_size=16`, and `ep_size=16`;
- all 32 requests report `cached_tokens=131072` and produce 1,024 tokens;
- the measured extend is one global batch with `#new-seq: 32` and
  `#new-token: 32768`, rather than two C16 waves;
- decode remains `[2, 2, ..., 2]` across the 16 DP ranks;
- there is no retraction, eviction/recompute, traceback, or OOM.

The basic generation script checks transport and non-empty decoding only. It
is not a model-quality evaluation.
