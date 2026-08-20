# Fused reduce-scatter MoE kernel (experimental)

This package ports the production-oriented kernel from
`vllm-project/tpu-inference#3040`, including the large-M kernel changes from
the follow-up `vllm-project/tpu-inference#3388`. It implements the routed
expert path as:

```text
all-gather (JAX) -> gather -> GMM1 -> SiLU -> GMM2 -> reduce-scatter (Pallas)
```

The upstream all-gather remains outside the Pallas call, matching the source
PR. The output collective is implemented by direct remote writes inside the
kernel. BF16, FP8 block-quantized, and FP8 per-output-channel expert
weights are supported.

sglang-jax normally computes grouped top-k routing before entering its MoE
backend. Callers should therefore pass `topk_weights` and `topk_indices`:

```python
from sgl_jax.srt.kernels.fused_moe.fused_rs import fused_moe_func_rs

out = fused_moe_func_rs(
    hidden_states=hidden_states,
    w1=gate_weight,
    w3=up_weight,
    w2=down_weight,
    w1_scale=gate_scale,
    w3_scale=up_scale,
    w2_scale=down_scale,
    w1_bias=None,
    w2_bias=None,
    gating_output=None,
    topk=8,
    renormalize=False,
    mesh=mesh,
    activation="silu",
    scoring_fn="softmax",
    topk_weights=topk_weights,
    topk_indices=topk_indices,
)
```

The separate gate/up form is the production interface. It lets the hybrid
model backend reuse the same parameters for fused-v2 decode and fused-RS
prefill; the Pallas kernel joins only the active weight tile in VMEM. The
original pre-merged form remains supported for source compatibility.

Use `--moe-backend fused_rs` with `GlmMoeDsaForCausalLM` to enable the hybrid
policy: `EXTEND`, `MIXED`, and `DRAFT_EXTEND` use RS; `DECODE`,
`TARGET_VERIFY`, and `IDLE` retain fused-v2. A mixed batch is one compiled MoE
call, so its prefill work determines the RS choice for the whole call. Routed
and shared weights are not duplicated.

Current scope: GLM-5.x on ICI expert parallelism. DCN expert parallelism,
expert remapping/EPLB, and the PR's instructional AG+GMM1-only example are not
part of this initial kernel port.

For large prefill shapes whose packed routing metadata no longer fits scalar
prefetch SMEM (GLM-5.2 64K tokens has 524,288 routed rows), the kernel stages
one routing-index tile at a time from HBM through double-buffered VMEM. The
per-channel path folds the whole-K weight scale outside the activation-block
loop to avoid repeated VPU scale multiplies.

Reproduce the routed EP32 tuning sweep with `benchmark/moe/bench_fused_rs_moe.py`
and `benchmark/moe/falcon_glm52_fused_rs_ep32_64k_tuning.yaml`. The tuning mode
checks the upstream weight-cache contract (`buffers >= weight steps`), uses
expert/channel-distinct FP8 weights and scales, compares candidates with a
canonical full-N RS result, and verifies same-backend active-prefix/padding
fidelity before timing. V2 output is not a numerical gate. The benchmark reports
strict 32-device Pallas critical-path samples, backend wall samples, effective
config, compile status, and correctness metrics. Shared-expert execution and the
final caller-layout reshard belong to the model-layer boundary and must be
measured separately after selecting the routed-kernel config.
