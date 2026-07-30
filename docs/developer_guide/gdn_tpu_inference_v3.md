# TPU-Inference v3 GDN Prefill

SGL-JAX includes an opt-in fused Conv1D+Gated DeltaNet (GDN) prefill path for
Qwen3.5 models. The implementation is vendored from TPU-Inference and adapted
to the SGL-JAX recurrent-state pool. The existing SGL-JAX implementation
remains the default.

## Selection and Scope

Set the implementation before starting the server:

```bash
export SGLANG_JAX_GDN_PREFILL_IMPL=tpu_inference_v3
```

Supported values:

| Value | Prefill | Decode |
|---|---|---|
| `reference` | Existing SGL-JAX Conv1D and GDN recurrence | Existing SGL-JAX reference decode |
| `tpu_inference_v3` | Vendored fused Conv1D+GDN v3 | Existing SGL-JAX reference decode |

`reference` is the default. The selector is read once when
`GDNAttnBackend` is initialized. Startup logs record
`requested`, `effective`, and `fallback_reason`.

The optimized path changes prefill only. It does not change decode, scheduler
behavior, the recurrent-state pool schema, data/tensor/expert parallelism
semantics, or other attention backends.

## Capability Requirements

`tpu_inference_v3` requires:

- a JAX mesh containing TPU devices;
- BF16 activations;
- positive GDN head counts and dimensions;
- key and value head dimensions divisible by the TPU lane width, 128;
- `num_v_heads` divisible by `num_k_heads`;
- a Conv1D kernel size of at least 2.

The request fails during backend initialization when these requirements are
not met. There is no silent fallback to `reference`. CPU Pallas interpret is
not a supported execution path for this fused DMA/state pipeline.

## State Contract

The SGL-JAX adapter preserves the existing GDN contract:

- output, final Conv state, and final recurrent state;
- fresh, continuing, and mixed recurrent state;
- packed/ragged and partial-tile requests;
- zero-length, dummy-slot, unused-slot, and track/checkpoint isolation;
- prefill-to-decode continuity;
- input immutability and finite outputs.

Decode always uses the existing `decode_gated_delta_rule_ref` callable.

## Provenance

The vendored source is frozen at:

```text
repository: https://github.com/vllm-project/tpu-inference
commit: a9072c881843622226efc101de1a62c731ab572f
source: tpu_inference/kernels/gdn/v3
license: Apache-2.0
```

The copied modules retain the upstream copyright and Apache-2.0 license
headers. `PROVENANCE.md` in the vendored package is the machine-checked source
record.

## Verification

Run the local selector, provenance, and state-contract regression batch:

```bash
uv run --project python --extra cpu --with pytest --frozen \
  python -m pytest \
  python/sgl_jax/test/kernels/test_gdn_tpu_inference_v3_vendor.py \
  python/sgl_jax/test/test_gdn_tpu_inference_prefill_dispatch.py \
  python/sgl_jax/test/test_gdn_tpu_inference_state_contract.py \
  python/sgl_jax/test/test_gdn_attention.py \
  python/sgl_jax/test/models/test_qwen3_5.py \
  -q --tb=short

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run --project python --extra cpu --with pytest --frozen \
  python -m pytest \
  python/sgl_jax/test/test_gdn_attention_dp.py \
  -q --tb=short
```

The complete fused DMA/state numerical suite is TPU-only. It compares the real
vendored path against the independent reference for lengths
`1, 63, 64, 65, 127, 128, 129`, including state and slot-isolation cases, with
TPU tolerances `rtol=2e-2, atol=5e-2`.

Run the following command on every host in the initialized multi-host TPU
runtime:

```bash
SGLANG_JAX_GDN_PREFILL_IMPL=tpu_inference_v3 \
PYTHONPATH=python \
python -m pytest \
  python/sgl_jax/test/test_gdn_tpu_inference_prefill.py \
  python/sgl_jax/test/test_gdn_tpu_inference_prefill_dp.py \
  -q --tb=short
```

Production serving performance is measured with the repository-owned wrapper:

```bash
.claude/skills/model-speed-benchmark/scripts/speed_benchmark.sh \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random-ids \
  --python-bin <SERVER_PYTHON> \
  --batch-sizes "16" \
  --num-prompts-multiplier 5 \
  --request-rate inf \
  --out <OUTPUT_DIR> \
  -- \
  --random-input-len 4096 \
  --random-output-len 1 \
  --random-range-ratio 1 \
  --seed 1 \
  --output-details \
  --flush-cache \
  --warmup-requests 16 \
  --tokenize-prompt
```

Use an identical serve configuration, workload, and fresh compilation cache
for the `reference` and `tpu_inference_v3` variants. Repeat with
`--random-input-len 1024 --random-output-len 100` for the decode cell. The
recorded A/B ran each cell five times.

## Qwen3.5-397B-A17B Serving A/B

The production A/B used:

- model `Qwen/Qwen3.5-397B-A17B`, revision
  `8472618112abcbd45acbcdc58436aff4233c23f7`;
- BF16 on TPU v7x64 (`4x4x4`, 16 hosts);
- JAX/jaxlib `0.10.2` and libtpu `0.0.43`;
- TP/DP/EP `128/8/128`;
- five measured rounds per cell after 16 warmup requests;
- 80 measured requests per round;
- exact 64-token identity between variants.

Median results:

| Workload | Metric | `reference` | `tpu_inference_v3` | Change |
|---|---|---:|---:|---:|
| 4096 to 1, concurrency 16 | Input throughput | 9,485.999241 tok/s | 40,821.238522 tok/s | +330.331455% |
| 4096 to 1, concurrency 16 | Mean TTFT | 6,515.492437 ms | 1,513.621697 ms | -76.768883% |
| 1024 to 100, concurrency 16 | Output throughput | 275.949103 tok/s | 398.225380 tok/s | +44.311170% |
| 1024 to 100, concurrency 16 | Mean TPOT | 36.643958 ms | 36.005768 ms | -1.741596% |

All ten measured rounds completed without request errors or non-finite
metrics. The prefill improvement gate and the decode TPOT regression limit
(`<= 5%`) both passed.

The complete per-round results, manifests, and checksums are retained in the
internal experiment archive. Internal infrastructure identifiers are
intentionally omitted from this public document.

These results are specific to the recorded model, runtime, topology, serve
configuration, and workload. Re-run the benchmark when any of those inputs
change.
