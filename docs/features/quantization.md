# Quantization

Quantization in sglang-jax primarily uses the qwix quantization library. For implementation details and supported features, see the qwix repository: https://github.com/google/qwix

## Motivation
- Reduce memory footprint
- Increase inference throughput

# Supported features
## Quantization targets
- Weight quantization.
- Activation quantization

## Quantization modes:
- Post training quantization (PTQ)

## numeric formats
- int8, fp8

# Configuration and Customization
Quantization behavior is defined through qwix configuration rules.

Example configuration files can be found under:

```python
sgl_jax/srt/utils/quantization/configs/
```

qwix supports multiple quantization granularities:

- Per-channel

- Per-matrix

- Per-tile

You can define custom rules in YAML to control how weights and activations are quantized.

To apply a quantization rule, pass the configuration file path using the
"--quantization-config-path" flag.

example command:
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

# Supported models (To be updated)
- Dense models

### TODOs:
- add dynamic range quantization support
