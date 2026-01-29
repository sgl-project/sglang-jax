# Quantization

sglang-jax supports post-training quantization (PTQ) for reducing memory footprint and increasing inference throughput.

## Overview

The quantization system consists of two components:

1. **Linear Quantization** - For dense layers (attention, MLP, embeddings, etc.)
2. **MoE Quantization** - For Mixture-of-Experts layers (expert weights and activations)

## Supported Features

### Quantization Targets
- **Weight quantization** - Quantize model weights (pre-quantized at load time)
- **Activation quantization** - Quantize activations on-the-fly during inference

### Numeric Formats
- `int8` - 8-bit integer quantization
- `float8_e4m3fn` - 8-bit floating point (4 exponent bits, 3 mantissa bits, finite values + NaN)

### Quantization Strategy
- **Weights**: Per-channel quantization (computed once at load time)
- **Activations**: Dynamic per-token quantization (computed at runtime)

## Configuration

Quantization behavior is defined through YAML configuration files with two sections:

```yaml
quantization:
  # Quantization rules for dense layers
  dense:
    rules:
      - module_path: '<regex pattern>'
        weight_dtype: '<dtype>'        # Required: weight quantization type
        activation_dtype: '<dtype>'    # Optional: activation quantization type (null for weight-only)

  # MoE-specific settings (for MoE models only)
  moe:
    weight_dtype: '<dtype>'            # Expert weight quantization
    activation_dtype: '<dtype>'        # Expert activation quantization (null to disable)
```

### Available Configuration File Examples

| Config File | Weights | Activations | Description |
|-------------|---------|-------------|-------------|
| `fp8.yaml` | FP8 | None | FP8 weight-only quantization |
| `fp8_w8a8.yaml` | FP8 | FP8 | Full FP8 quantization |
| `int8.yaml` | INT8 | None | INT8 weight-only quantization |
| `int8_w8a8.yaml` | INT8 | INT8 | Full INT8 quantization |

## Usage Example

Pass the configuration file path using `--quantization-config-path`:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
uv run python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen-7B-Chat \
  --trust-remote-code \
  --dist-init-addr=0.0.0.0:10011 \
  --nnodes=1 \
  --tp-size=4 \
  --device=tpu \
  --random-seed=3 \
  --node-rank=0 \
  --mem-fraction-static=0.8 \
  --max-prefill-tokens=8192 \
  --download-dir=/tmp \
  --dtype=bfloat16 \
  --skip-server-warmup \
  --quantization-config-path=int8.yaml
```
