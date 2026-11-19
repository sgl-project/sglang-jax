# DeepSeek-V4.1-Flash on TPU v6e-32 — our JAX engine

Own JAX implementation of V4.1 Flash (not a fork): exact FP4 expert weights (stored FP8 + per-row scale, zero inexact conversions), activation
fake-quant as in DeepSeek's reference, Engram tables in host RAM, 40-layer SPMD over 8 hosts x 4 chips (mesh dp=4 x tp=8, experts over all 32).
CPU-verified against DeepSeek's reference model (rel ~6e-6, no-quant); see tests.

## Files
| file | role |
|---|---|
| model_ref.py / engine.py | model port; `chunk_forward` = unified prefill/decode step (per-row positions, `active`, padded chunks via `nvalid`, tiled exact indexer top-k, capacity view `cap`) |
| generate.py | `Generator` (single request) and `BatchScheduler` (continuous batching, padded fixed-size chunk prefill shared by all waiting prompts, decode buckets, packed per-dp slots, PrefixCache, sampler incl. top-p/top-k, per-round stats) |
| server.py / http_api.py | multi-host lockstep server; OpenAI-style HTTP API incl. SSE streaming, /admin/config, /admin/profile |
| loader.py / config_real.py / sharding.py | checkpoint reader (FP4->FP8 rowscale, Engram LUT dequant), param table, mesh/sharding, `to_kernel_layout` for the SGL-JAX fused MoE kernel |
| engram_jax.py | Engram hashing/gating (host hasher + device gate) |
| run_server.sh / restart_all.sh | start on one host / two-phase kill -9 restart on all hosts (`ENVS="..." bash restart_all.sh`) |
| profrun.py, memprobe.py, op_report.py, analyze_trace.py, bench_moe_kernel.py | profiling / micro-benchmarks on the slice |
| ../scripts/loadtest_v41.py, bench_v41.py, needle_v41.py, eval_v41.sh, resume_bench.sh | steady-state load test with cost, quick bench, needle-in-haystack, lm-eval quality, one-shot deploy+bench |

## Run (every host; see restart_all.sh)
Throughput config: `MAX_SEQ=16384 MAX_RUNNING_REQUESTS=256 BS_PADDINGS=16,64,256 CTX_LADDER=16384 V41_MOE=kernel V41_PREFILL_M=1,4 CHUNKS=512,256,128,64,32,8`
(256 slots only fit at <=16K context; 64-128 slots at 32K.) 1M-context config: `MAX_SEQ=1048576 MAX_RUNNING_REQUESTS=4 CHUNKS=512,128,32,8,2 CTX_LADDER=1048576`
without `V41_MOE=kernel` (the fused kernel does not compile with 1M caches).
Other knobs: `V41_PREFIX_CACHE=1` (+`V41_PREFIX_SLOTS`, `V41_PREFIX_MAX`, `V41_PREFIX_GRANULE`), `V41_STATS=0`, `V41_KERNEL_MIN_TOKENS`, `V41_KERNEL_ANCHOR_MAX_TOKENS`,
`V41_MOE=loop|grouped|dense`, `JAX_LOG_COMPILES=1` (on in run_server.sh). Runtime: `POST /admin/config {"max_batch": N}`, `POST /admin/profile {"rounds": N}` (trace -> /tmp/prof_live per host).
Restarts: ~15 min when only scheduler code changed (compile cache), 20-40 min when model code or shapes changed. gcloud login expires periodically.

## Tests (CPU, `source scratchpad/v41env/bin/activate`; args: ref_cpu dir [+ synth ckpt])
test_vs_ref, test_engine, test_perrow, test_padded (padded chunks vs reference), test_longctx (tiled indexer), test_batch (scheduler == single-request tokens; try V41_PREFILL_M=1,2 ROWS=2/4/8),
test_prefix (prefix cache), test_sampler, test_http_api + test_stream_engine (SSE streaming), test_moe_grouped, test_int8_experts, test_engram, test_loader*, test_generate.

## Measured (v6e-32; 510-in/256-out unless noted)
single user 17.4 ms/token (50 tok/s); decode-only 64 users 600 tok/s; mixed 64 users 428 tok/s; steady 1000-in/300-out, 64 users: 230 tok/s out, 953 tok/s in (before padded prefill),
~$104 per 1M output tokens at an assumed $2.70/chip-hour. Quality: GSM8K-500 (8-shot) 96.2/96.4 vs V4 Flash 97.6/97.4; needle passes to 165K tokens. Provider price for comparison: $0.30 in / $1.20 out per 1M.
## Open items
Run MMLU-Pro/IFEval/full GSM8K on the final config; enable+validate prefix cache on TPU; SGL-JAX decode/indexer kernels (csa_decode_attention, topk_threshold); INT8 experts
(99.97% of rows exact); MTP speculative decoding; paged/tp-sharded caches (more slots and 1M with the fused kernel); client-disconnect cancel; compile-cache for fast restarts.
