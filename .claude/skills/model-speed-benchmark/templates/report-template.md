# Benchmark

## Status

- State: {success | partial | failed}
- Target resource: {target resource}
- Endpoint: {endpoint}
- Model path: {model path}
- Server: {healthy | unhealthy | unknown}
- Server id: {PID or exp_id, or `unknown`}
- Radix cache: {enabled | disabled}
- Hardware:
  - TPU type: {e.g. tpu-v6e}
  - Topology: {e.g. 4x4}
  - Chips: {e.g. 16}
- Versions:
  - engine: {engine name + git commit / branch, determined by --backend, e.g. sglang-jax @ 6f6d2844 (skills/model-eval-benchmark)}
  - `jax`: {version}
  - `libtpu`: {version}
- Server launch args:

```bash
{the sgl_jax.launch_server command (with flags) used to start the server}
```

## Speed Benchmark

```bash
{actual command — expand the raw bench_serving loop with the concrete bs values}
```

- Status: {completed | failed}
- Output dir: {output directory}
- Summary CSV: {summary.csv path}
- Backend: {--backend value, e.g. sgl-jax / sglang / vllm}
- Dataset: {e.g. random / sharegpt}
- Workload shape: {dataset-specific shape flags used, e.g. `input_len=16384, output_len=1024` for random, or `sharegpt_output_len=1024` for sharegpt}
- Cache hit rate: {from result.jsonl; ~1.0 with radix cache on means prefill was served from cache}
- Metrics:

| Max Concurrency                                    | Input tok/s | Output tok/s | Peak output tok/s | Total tok/s | Mean E2E ms | Mean TTFT ms | Mean TPOT ms | Mean ITL ms | P99 ITL ms |
| -------------------------------------------------- | ----------: | -----------: | ----------------: | ----------: | ----------: | -----------: | -----------: | ----------: | ---------: |
| {one row per batch point, values from summary.csv} |             |              |                   |             |             |              |              |             |            |

- Artifacts:
  - `{bs_<N>/result.jsonl path on the target machine}`

## Notes

- {any deviation from the full workflow, or `None`}
