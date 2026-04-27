Enable Configurable Submodule dtype

## 1\. Overview

By introducing a `DtypeConfig` class, users will be able to set different precision types (e.g., `float32`, `bfloat16`) for individual model submodules such as attention projections, MLP components, and critical operations like the softmax inside attention kernels.

## 2\. Motivation

Currently, each model layer (e.g., `LlamaDecoderLayer` or `Qwen2DecoderLayer`) applies a single globally defined `dtype` across all of its submodules during initialization.

However, as highlighted in Issue #661, applying higher precision (e.g., `float32`) uniformly across the entire model limits performance and memory efficiency. To strike a better balance between numerical stability and inference throughput, we need fine-grained control to selectively elevate precision only for high-leverage operations. Typical use cases include:

- Executing `softmax` inside attention with `float32`.
- Using higher precision for critical projection layers like `o_proj` in attention or `down_proj` in MLPs.

## 3\. Design

### 3.1 `DtypeConfig` Design and Structure Rules

To seamlessly support various model architectures—ranging from standard MLPs (like Llama) to Mixture of Experts (like Grok or Qwen2-MoE)—we will use a **generic, nested dictionary design**.

The core principle is: **The structure of `DtypeConfig` perfectly mirrors the variable names and Python structural hierarchy of the instantiated models.**

**Structure Rules:**

1. **Top-level Keys:** Must match the exact attribute/submodule names defined inside the decoder layer (e.g., `self_attn`, `mlp`, `input_layernorm`).
2. **Nested Keys:** Must match the names of the inner projections or computations (e.g., `q_proj`, `c_fc`, `expert_w1`).
3. **Operational Keys:** Operational parameters that do not reflect strict weight modules (like `softmax`) should be placed structurally where the computation happens (e.g., inside `self_attn`).
4. **Fallback mechanism:** If a module is omitted from the config, it safely falls back to the default `dtype` passed to the layer. Any unused keys relevant to other architectures are ignored via `.get()`.

**Example `DtypeConfig` implementation:**

```python
class DtypeConfig:
    def __init__(
        self, config_dict: dict[str, Any] | None = None, default_dtype: jnp.dtype | None = None
    ):
        # Validate at least one of config_dict and default_dtype is provided
        if config_dict is None and default_dtype is None:
            raise ValueError("At least one of config_dict and default_dtype must be provided.")

        self.config_dict = self._parse_dict(config_dict or {})

        # Resolve the default dtype for this level
        self.default_dtype = self.config_dict.get("default", default_dtype)

    def _parse_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Recursively parses a dictionary, converting string dtypes to jnp.dtype."""
        parsed = {}
        for k, v in d.items():
            if isinstance(v, dict):
                parsed[k] = self._parse_dict(v)
            elif isinstance(v, str) and v.lower() in STR_DTYPE_TO_JAX_DTYPE:
                parsed[k] = STR_DTYPE_TO_JAX_DTYPE[v.lower()]
            else:
                raise ValueError(f"Unknown dtype: {v}")
        return parsed

    def get_config(self, key: str) -> "DtypeConfig":
        """Returns a child config covering the sub-dictionary, preserving the default."""
        return DtypeConfig(
            config_dict=self.config_dict.get(key, {}), default_dtype=self.default_dtype
        )

    def get_dtype(self, key: str) -> jnp.dtype | None:
        """Returns the specific dtype, or falls back to the default."""
        val = self.config_dict.get(key, self.default_dtype)
        if isinstance(val, dict):
            return self.default_dtype
        return val
```

**Example `dtype_config` Payload (Llama as the example):**

```python
# Note: every `{}` in the dtype_config is a DtypeConfig object!
dtype_config = {
    "default": jnp.bfloat16,
    # Aligns with layer.lm_head in LlamaForCausalLM
    "lm_head": jnp.bfloat16,
    # Aligns with layer.model (LlamaModel)
    "model": {
        "embed_tokens": jnp.float32,

        # Aligns with self.layers (list of LlamaDecoderLayers)
        "layers": {
            # Inside LlamaDecoderLayer components
            "self_attn": {
                "q_proj": jnp.bfloat16,
                "k_proj": jnp.bfloat16,
                "v_proj": jnp.bfloat16,
                "o_proj": jnp.float32,
                "softmax": jnp.float32,  # Configurable precision inside attention kernels
            },

            "mlp": {
                "gate_proj": jnp.float32,
                "up_proj": jnp.float32,
                "down_proj": jnp.bfloat16,
            },

            "input_layernorm": jnp.float32,
            "post_attention_layernorm": jnp.float32,
        },

        "norm": jnp.float32, # Final RMSNorm of the model
    }
}
```

**Example `dtype_config` usage:**

```python
# Initialize the dtype config
dtype_config = DtypeConfig(dtype_config)

# Get the dtype for a specific module
# Returns jnp.bfloat16
dtype = dtype_config.get_config("model").get_config("layers").get_config("self_attn").get_dtype("q_proj")

# Get the dtype for a specific module
# Returns jnp.bfloat16, inherited from the default dtype of the parent config
dtype = dtype_config.get_config("model").get_config("layers").get_dtype("self_attn")

# Get the child config for a specific module
# Returns DtypeConfig object wrapping the "self_attn" sub-dictionary
child_config = dtype_config.get_config("model").get_config("layers").get_config("self_attn")
```

### 3.2 Code Changes

The implementation will touch model definitions and the attention backends.

#### 3.2.1 Define and Expose `DtypeConfig` to Users (CLI and API)

`DtypeConfig` will be surfaced in two ways:

1. **CLI Flag (`--dtype-config`)**: We will introduce this argument to `server_args.py`. If the string begins with `{`, it will be parsed directly as a JSON dictionary. Otherwise, it will be treated as a path to a `.json` file and loaded natively.
2. **Python API (`Engine` / `Runtime`)**: Because the `Engine` class (inside `engine.py`) takes `**kwargs` and expands them seamlessly into `ServerArgs`, users can directly pass Python dictionaries to the API: `engine = Engine(model_path="...", dtype_config={"model": ...})`.

The dictionary produced by `ServerArgs` will then be explicitly passed to `ModelConfig` inside `engine.py`, which transports it to `model_loader.py`, injecting it perfectly into the base initialization of target models (e.g., `LlamaForCausalLM`).

To elegantly handle missing keys without passing a monolithic dictionary around, we will define a lightweight `DtypeConfig` wrapper class (e.g., in `sgl_jax/srt/utils/dtype_config.py`). This class acts as the single source of truth and automatically propagates the `"default"` fallback down the layer hierarchy. The sample implementation is provided in the `DtypeConfig` class above.

#### 3.2.2 Update Model Submodules (e.g., `llama.py`, `qwen.py`)

By passing a `DtypeConfig` object down, each child seamlessly extracts its own slice:

```python
# Inside LlamaDecoderLayer.__init__
# We extract the relevant subset simply by calling get_config
self.self_attn = LlamaAttention(
    ...
    dtype_config=dtype_config.get_config("self_attn"),
)
```

Each intermediate block (`LlamaAttention`, `LlamaMLP`, `LlamaDecoderLayer`, etc.) will have its `__init__` signature updated to accept `dtype_config: DtypeConfig`.

For models and intermediate blocks that support `DtypeConfig`, we keep both traditional `dtype` and `dtype_config: DtypeConfig`. At least one of them must be provided. If both of them are provided, the value of `dtype` must equal to `dtype_config.default_dtype`. Once this condition is validated, the implementation will only refer to `dtype_config`. The reasons we want to keep `dtype`:

1. Backwards Compatibility and industry standard: Thousands of existing Hugging Face models provide a top-level dtype and it is an industry standard to name the precision control parameter as `dtype.`
2. Different purpose: It is true that their purposes overlap heavily, both `dtype` and `DtypeConfig` can co-exist and serve different purposes. dtype is the Global/Default Precision while `DtypeConfig` provides granular control

#### 3.2.3 Update Attention Kernels (`radix_attention.py`, backends)

To configure the precision of the softmax computation:

1. `LlamaAttention` (and others) will extract the `softmax` config:


| softmax\_dtype \= dtype\_config.get("softmax", None) if dtype\_config else Noneself.attn \= RadixAttention(..., softmax\_dtype=softmax\_dtype) |
| :---- |



2. `RadixAttention` will pass `softmax_dtype` into the `forward_batch.attn_backend()`.
3. Kernels like Paged Attention (`ragged_paged_attention.py` or Native Backend) will use `softmax_dtype` to explicitly cast the accumulated attention matrix before performing the softmax scaling operation:

```python
if softmax_dtype is not None:
    # Cast BEFORE softmax for numerical stability
    attn = attn.astype(softmax_dtype)


attn = jax.nn.softmax(attn, axis=-1)
if attn.dtype != v.dtype:
    # Cast back to v.dtype before the final MatMul (attn @ V)
    # to prevent implicit upcasting and preserve memory/speed
    attn = attn.astype(v.dtype)
```

## 4\. Test Plan

### 4.1 Unit Tests (Topology Validation)

Create a new test module `test/srt/models/test_dtype_config.py`.

1. **Model Selection:** Test one typical dense model (e.g., Llama) and one typical MoE model (e.g., Qwen2-MoE or Grok). This simplifies testing while ensuring robust architectural coverage.
2. **Assertion Check:** Validate that all influenced modules (e.g., specific projections in attention, MLP, MoE experts, layernorms, and the extracted softmax\_dtype) map correctly to the expected dtype override.
3. **MoE/Structural Ignorance Validation:** Inject incorrect keys to a generic architecture (e.g., inject router\_proj to Llama) to ensure the DtypeConfig wrapper ignores them securely without crashing.

### 4.2 Integration and Precision Evaluation Tests

To ensure that mixed-precision settings actually influence mathematical outcomes and do not break standard execution, we will perform comprehensive precision evaluations.

1. **Logit Consistency**: Use `np.testing.assert_allclose` to compare the logits of a mixed-precision forward pass against pure `float32` and pure `bfloat16` forward passes. Mixed-precision output errors (relative to pure `float32`) should be bounded and demonstrably smaller than pure `bfloat16` errors.
   Specifically:
   diff\_fp32\_bf16=max\_diff(logits\_with\_fp32-logits\_with\_bf16)
   diff\_bf16\_mixed=max\_diff(logits\_with\_bf16-logits\_with\_mixed)
   diff\_fp32\_mixed=max\_diff(logits\_with\_fp32-logits\_with\_mixed)
   assert diff\_fp32\_bf16 \> diff\_bf16\_mixed
   assert diff\_fp32\_bf16 \> diff\_fp32\_mixed

2. **End-to-End Evaluation**: Select an evaluation dataset (e.g., **GSM8K**) and a popular dense model (e.g., **Llama** or **Qwen3 dense**). We will run evaluations with `temperature=0.0`.
   - **Run 1:** Baseline pure `bfloat16` (compute average grade across 3 runs).
   - **Run 2:** Baseline pure `float32` (compute average grade across 3 runs).
   - **Run 3 (Mixed-Precision):** Set default `dtype=bfloat16` and configure critical submodules to `float32` (e.g., `softmax`, `o_proj`, `down_proj`). Refer to the EvalScope from the contribution document for details about the testing setup.
   - **Expectation**: The average grade of Run 3 is expected to safely locate between Run 1 and Run 2, confirming that applying `float32` to targeted sub-modules measurably elevates the reasoning precision over pure `bfloat16`.
