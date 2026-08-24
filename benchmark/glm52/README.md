# GLM-5.2 exact DSA DP16 benchmark

> For the supported 8/16-chip block-wise and channel-wise delivery entry
> points, start with [`delivery/README.md`](delivery/README.md). The dated files
> in this directory are retained as historical experiment records.

## Natural 128K + 1K workload data

`prepare_longbench_v2.py` builds the natural-text/code input used by the 64-way
long-context serving comparison. It pins LongBench v2 revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`, keeps only the `Code repo QA` and
`Financial` sub-domains, and uses the serving model's tokenizer rather than a
character or word-count estimate.

The default output contains 32 requests from each sub-domain. Every request has
an exact 131,072-token prefix and 1,024-token extension; the question, all four
choices, and the answer marker fit in that final extension. The output length is
recorded as 1,024 tokens for the serving client. Long source records are split
into deterministic non-overlapping windows, with first windows preferred before
later windows to maximize source diversity.

The Falcon CPU preparation job persists the pinned raw source, exact input IDs,
checksums, tokenizer identity, and selection audit under:

```text
gs://inference-model-storage-poc-tpu-hns/benchmark-datasets/LongBench-v2/glm52-code-financial-128k-prefix-1k-extend-c64-v1/
```

The corresponding serving mount is:

```text
/models/benchmark-datasets/LongBench-v2/glm52-code-financial-128k-prefix-1k-extend-c64-v1/
```

`requests.jsonl.gz` is directly consumable: each row contains exact
`input_ids`, source/window provenance, SHA-256 hashes, and the runtime output
length. `_SUCCESS.json` is written last so a partial GCSFuse copy cannot be
mistaken for a completed dataset.

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

## MLA tuning for the dense prefix

GLM-5.2 uses dense MLA for layers 0-2 and exact DSA for the remaining layers.
The serving topology maps the MLA shapes as follows:

```text
2 Falcon replicas * 8 devices = 16 DP ranks
attention_tp = TP16 / DP16 = 1
decode: global BS32 / DP16 = 2 tokens per rank
extend: global 32768 / DP16 = 2048 tokens per rank
extend sequences: C32 / DP16 = 2 sequences per rank, 1024 tokens each
```

MLA has no attention collective when `attention_tp=1`, so tuning does not need
the full serving topology. Use Falcon's minimum v7x allocation: one replica,
eight devices, four chips, topology `2x2x1`. The tuner explicitly selects one
local device because every device executes the same independent DP-rank
kernel shape.

Run the reviewed GLM-5.2 scenario with:

```bash
python benchmark/kernels/mla/get_block_spec_config_mla.py \
  --scenario glm52-dp16-128k \
  --device-index 0 \
  --tries 5 \
  --output-jsonl /tmp/mla-tune/metrics.jsonl
```

The preset tunes only two table keys:

- decode: 64 query heads, page size 64, local token bucket 2, KV length
  133,120;
- mixed/extend: 64 query heads, page size 64, local token bucket 2,048,
  two 1,024-token sequences, KV length 132,096.

The decode tuner measures the active `MLA-d` tail for the historical
`decode_batch_size=4` fallback when the local bucket is only two tokens. It
measures `MLA-bd` for divisible candidate batch sizes, avoiding the empty-grid
timing bug that would otherwise make the fallback comparison invalid.
The JSONL output records latency in milliseconds together with the best and
fallback configurations, attempted/failed candidate counts, and whether the
measured gain clears the table-entry threshold.

The production preset was tuned on Falcon `exp-segouwkl9w` (TPU v7,
`v7x-8`, topology `2x2x1`):

- decode selected `(32, 1, 2)`: 1.1511 ms -> 0.4033 ms, a 65.0% kernel-time
  reduction;
- mixed/extend selected `(8, 64)`: 435.2249 ms -> 71.3998 ms, an 83.6%
  kernel-time reduction.

Compile-time VMEM rejections are expected during a sweep. They are counted in
the JSONL audit row and logged as `SKIP_VMEM`; the tuner continues with the
remaining candidates.

After checking in the selected entries, verify the production lookup path
against the explicit fallback with:

```bash
python benchmark/kernels/mla/bench_mla.py \
  --scenario glm52-dp16-128k \
  --output-jsonl /tmp/mla-lookup/metrics.jsonl
```

That lookup A/B was verified on Falcon `exp-aovgukczde`; the full MLA wrapper
(including cache update and dispatch overhead) measured:

- decode: 2.1335 ms -> 1.3069 ms, a 38.7% reduction;
- mixed/extend: 437.3139 ms -> 73.4938 ms, an 83.2% reduction.

Falcon `operator-analysis` record `an-ez63cziwkk` accepted both structured
metric rows with status `OK` and no warnings.
