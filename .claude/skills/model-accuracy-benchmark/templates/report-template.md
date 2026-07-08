# Benchmark

## Status

- State: {success | partial | failed}
- Target resource: {target resource}
- Endpoint: {endpoint}
- Model path: {model path}
- Server: {healthy | unhealthy | unknown}
- Server id: {PID or exp_id, or `unknown`}
- Hardware:
  - TPU type: {e.g. tpu-v6e}
  - Topology: {e.g. 4x4}
  - Chips: {e.g. 16}
- Versions:
  - engine (sglang-jax): {git commit / branch, e.g. 6f6d2844 (skills/model-eval-benchmark)}
  - `evalscope`: {version}
  - `jax`: {version}
  - `libtpu`: {version}
- Server launch args:

```bash
{the sgl_jax.launch_server command (with flags) used to start the server}
```

## Accuracy Benchmark

<!-- Repeat one `### {dataset}` block per dataset that was actually run. -->

### {dataset}

```bash
{actual command that was executed}
```

| Dataset   | Metric                  | Sample count | Correct                            | Score   |
| --------- | ----------------------- | ------------ | ---------------------------------- | ------- |
| {dataset} | {metric, e.g. mean_acc} | {n}          | {correct}/{n} (= round(n × score)) | {score} |

<!-- Optional: include the perf row only when the runner reports it; it is informational and does not change the accuracy result. -->

| Average latency (s) | Average throughput (tok/s) | Average input (tok) | Average output (tok) |
| ------------------- | -------------------------- | ------------------- | -------------------- |
| {lat}               | {thpt}                     | {in}                | {out}                |

- Artifacts:
  - `{primary artifact path on the target machine}`

## Notes

- {any deviation from the full workflow, or `None`}
