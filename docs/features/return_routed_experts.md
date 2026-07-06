# Return Routed Experts

Return routed experts exposes Mixture-of-Experts routing decisions in generation responses. When enabled, SGL-JAX can return the expert ids selected for each output token and MoE layer.

This feature is useful for RL workflows, routing analysis, expert utilization debugging, and model behavior research. It is only meaningful for MoE models.

## Enable the Feature

The feature has two levels of opt-in:

| Level | Control | Meaning |
|---|---|---|
| Server | `--enable-return-routed-experts` or `Engine(..., enable_return_routed_experts=True)` | Initializes the routed expert capture infrastructure. |
| Request | `return_routed_experts=True` | Captures and returns routed experts for that request. |

Both levels are required. A request-level flag without the server-level flag cannot read from the host capture buffer.

### Server

```bash
uv run python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --enable-return-routed-experts
```

### Engine

```python
from sgl_jax import Engine
from sgl_jax.srt.layers.routed_experts_capturer import (
    extract_routed_experts_from_meta_info,
)

engine = Engine(
    model_path="Qwen/Qwen3-30B-A3B",
    enable_return_routed_experts=True,
)

response = engine.generate(
    prompt="Explain how mixture-of-experts routing works.",
    sampling_params={"max_new_tokens": 32},
    return_routed_experts=True,
)

routed_experts = extract_routed_experts_from_meta_info(response)
```

For batched requests, `return_routed_experts` can be a single boolean or a list of booleans, matching the normalization behavior of other per-request generation fields.

## Response Format

The routed expert payload is attached to `meta_info["routed_experts"]` as a base64-encoded byte string. Use `extract_routed_experts_from_meta_info()` to decode it into an `int32` NumPy array.

The flattened array is ordered by generated token, layer, and top-k expert slot. Reshape it with the target model's MoE dimensions when doing analysis:

```python
expert_ids = routed_experts.reshape(seq_len, num_layers, num_experts_per_tok)
```

Some MoE-family models have non-MoE layers. SGL-JAX uses `-1` as the placeholder expert id for layers or slots where no routed expert exists.

## Runtime Notes

- The scheduler records routed experts only for requests with `return_routed_experts=True`.
- `ScheduleBatch.return_routed_experts` is enabled when any request in the batch asks for routed experts.
- Captured arrays are transferred to host memory and encoded for the response, so expect overhead relative to plain generation.
- Streaming and non-streaming clients should verify the exact response shape they consume before depending on this field in automation.

## Implementation Entry Points

- `python/sgl_jax/srt/server_args.py`: `enable_return_routed_experts` server flag.
- `python/sgl_jax/srt/entrypoints/engine.py`: `generate()` and `async_generate()` request argument.
- `python/sgl_jax/srt/entrypoints/openai/protocol.py`: OpenAI-compatible `return_routed_experts` extra field.
- `python/sgl_jax/srt/layers/routed_experts_capturer.py`: capture buffer and decode helper.
- `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py`: request-level routed expert collection.
