# Weight loading benchmark results

These are **intermediate-candidate measurements**, not final-head performance
acceptance. Baseline: `47684fa729b73d8cac0ecbfa5e9fcd169552c3b1`.
Runtime refactor commit: `a63c591a7e12b5240a905b74da2e611fd0a8048b`.
The measured candidates predate that commit's final mmap-lifetime changes.

## Completed load-only pairs

One v7x host, 8 JAX devices; CLI TP=8, DP=2, EP=8, with a data=2/tensor=4
mesh. Both variants used Local/GCSFuse and ran serially in separate Python
processes, baseline first. No serving, model forward, generation warmup, or
inference KV cache allocation was included.

| Model | Load time, s (base / candidate) | Peak RSS, GiB (base / candidate) | Peak HBM per device, GiB (base / candidate) | Matching shards |
|---|---:|---:|---:|---:|
| MiMo-V2-Flash | 385.75 / 253.76 | 21.60 / 18.01 | 39.78 / 39.16 | 3,395 / 3,395 |
| Qwen3-4B | 22.04 / 6.48 | 14.66 / 14.45 | 1.89 / 1.89 | 1,157 / 1,157 |
| Qwen3.5-27B + vision | 79.98 / 34.78 | 50.24 / 58.30 | 13.13 / 13.13 | 2,838 / 2,838 |
| Gemma4-31B-it + vision | 379.33 / 42.59 | 64.93 / 65.32 | 15.38 / 15.38 | 2,421 / 2,421 |

GiB means 2^30 bytes. Load time ends after every model array is ready. RSS is
sampled during configuration/loading; HBM is the maximum device peak reported
before fingerprinting. Shard counts are unique parameter/index positions;
shape, dtype, PartitionSpec, global index, and raw-byte SHA-256 matched exactly
at every compared position. Fingerprint D2H/hash time is outside the load timer.

## What these measurements do and do not show

- These are one-pair observations with uncontrolled cache state. They do not
  establish a reproducible speedup or no-regression result.
- GCSFuse prefetch alone took 15.0/0.7 s for Qwen3, 39.2/1.4 s for Qwen3.5,
  and 332.5/6.9 s for Gemma4 (baseline/candidate). Cache order explains much of
  the apparent wall-time reduction; the raw times must not be attributed
  entirely to the refactor.
- Qwen3.5 peak RSS increased from 50.24 to 58.30 GiB. The final runtime commit
  orders independent groups by file and releases completed mmap handles after
  transfers finish. That change still needs large-model memory remeasurement.
- Gemma4's complete text and vision weights matched. The same matrix then
  stopped in the **Kimi baseline constructor**, before candidate loading:
  `FusedEPMoE.__init__() got an unexpected keyword argument 'activation_fn'`.
  Kimi needs a rerun with its existing `epmoe` backend; DeepSeek was not reached.

## Reproduction and evidence

| Run | Candidate snapshot | Falcon experiment / artifact |
|---|---|---|
| MiMo-V2-Flash | Source bundle `0edffe0fdd854361b1bbacc2ed9b5ea0f43c4558d2d71dfb063a21976933a788` | `exp-sh3gxi825y` / `art-k8k6bjqa79` |
| Qwen3, Qwen3.5, Gemma4 | SRT source SHA-256 `c2c92e3105e1a2a17d00d193d6f0bc74ef4940d7ffddd51497323561c2e2b8ea` | `exp-y0z9tbe4tm` / `art-qjuhp8y0vw` |

Dependencies were pinned to JAX/jaxlib 0.11.1, libtpu 0.0.46.1, Flax 0.12.9,
NumPy 2.2.6, Transformers 5.12.1, and safetensors 0.6.2. GCSFuse used the
training profile with a 1 GiB file cache and parallel downloads. Conversion
compilation is included in loading time. See `scripts/benchmark_weight_loading.py`
for the timing and manifest contract.

## Remaining acceptance work

Final runtime commit measurements, at least three alternating A/B pairs for
representative large models, MiMo-V2.5-Pro, GLM-5.2, native RunAI/GCS, and the
remaining model matrix are pending. A four-host MiMo-V2.5-Pro native RunAI run
has been submitted as `exp-k7hhayynvx` / `art-wwjcxxygbt`, source bundle
`827861b0`; no result from that run is included above.
