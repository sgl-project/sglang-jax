# Dtype Config

`DtypeConfig` lets selected model submodules use a different dtype from the global `--dtype`. It is useful when a model mostly runs in `bfloat16`, but a few projections, layer norms, or attention softmax operations need higher precision.

Status: implemented for model code paths that explicitly consume `DtypeConfig`. Llama is wired through the nested config path; other models only take effect after their constructors pass the relevant child configs down.

## Configuration

Use `--dtype-config` with either an inline JSON object or a path to a JSON file:

```bash
python3 -u -m sgl_jax.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --trust-remote-code \
  --dtype=bfloat16 \
  --dtype-config '{"default":"bfloat16","model":{"layers":{"self_attn":{"softmax":"float32","o_proj":"float32"}}}}'
```

The same value can be passed through the Python API as a dictionary:

```python
from sgl_jax.srt.entrypoints.engine import Engine

engine = Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    dtype_config={
        "default": "bfloat16",
        "model": {
            "layers": {
                "self_attn": {
                    "softmax": "float32",
                    "o_proj": "float32",
                }
            }
        },
    },
)
```

## Structure

The nested dictionary mirrors the instantiated model hierarchy. `DtypeConfig.get_config("...")` descends into a child object, while `get_dtype("...")` returns a concrete dtype with fallback to the nearest `"default"` value.

Example for Llama:

```json
{
  "default": "bfloat16",
  "lm_head": "bfloat16",
  "model": {
    "embed_tokens": "float32",
    "layers": {
      "self_attn": {
        "q_proj": "bfloat16",
        "k_proj": "bfloat16",
        "v_proj": "bfloat16",
        "o_proj": "float32",
        "softmax": "float32"
      },
      "mlp": {
        "gate_proj": "float32",
        "up_proj": "float32",
        "down_proj": "bfloat16"
      },
      "input_layernorm": "float32",
      "post_attention_layernorm": "float32"
    },
    "norm": "float32"
  }
}
```

## Implementation Notes

- The wrapper lives in `python/sgl_jax/srt/configs/dtype_config.py`.
- `ServerArgs` parses `--dtype-config` as inline JSON when the string starts with `{`; otherwise it loads the path as JSON.
- `ModelConfig` wraps the parsed dictionary in `DtypeConfig` and validates that the global dtype matches the config default when both are provided.
- Llama passes child configs through `LlamaForCausalLM` → `LlamaModel` → `LlamaDecoderLayer` → attention/MLP modules.
- MLA softmax dtype is not configurable through `DtypeConfig`; the runtime logs a warning if an MLA model is launched with a softmax override.
