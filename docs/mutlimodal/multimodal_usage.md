# Multimodal Usage Guide

This guide covers how to use SGL-JAX for multimodal inference with models like Wan 2.1.

## Overview

SGL-JAX provides a unified, high-performance inference framework for multimodal models. The framework supports heterogeneous compute patterns—integrating Auto-Regressive (AR) decoding with Diffusion denoising—within a single pipeline.

For architecture details, see the [RFC: Multimodal Architecture Design](design/[RFC]multimodal_architechure.md).

## Supported Models

| Model | Description |
|-------|-------------|
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers | Video generation model supporting text-to-video generation |
| Wan-AI/Wan2.1-T2V-14B-Diffusers | Video generation model supporting text-to-video generation |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | Video generation model supporting text-to-video generation |

## Quick Start

### Offline Inference

> Still Under Development

### Online Inference (OpenAI-Compatible API)

SGL-JAX provides an OpenAI-compatible API for online inference.

#### Launch Server

```bash
uv run python3 -u -m sgl_jax.launch_server \
    --multimodal \
    --model-path=Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --log-requests
```

#### Image Generation

```bash
curl http://localhost:30000/api/v1/images/generation \
    -H "Content-Type: application/json" \
    -d '{"prompt": "A curious raccoon", "size": "480*832"}'
```

#### Video Generation

```bash
curl http://localhost:30000/api/v1/videos/generation \
    -H "Content-Type: application/json" \
    -d '{"prompt": "A curious raccoon", "size": "480*832", "num_frames": 41}'
```

## Configuration

### Stage Configuration

Multimodal models are composed of multiple stages (e.g., ViT, Diffusion, AR). Each stage can be configured independently.

> If not provided, the default config from `python/sgl_jax/srt/multimodal/models/static_configs` will be used.

```yaml
stage_args:
  - stage_id: 0
    run_time:
      num_tpus: 2
      sharding_spec: ["tensor"]
    launch_args:
      attention_backend: fa
      tp_size: 2
    input_type: image
    output_type: tensor
```

## Performance Tips

1. **Independent Scheduler**: Each Stage has its own scheduler to maximize TPU utilization
2. **Stage Overlap**: The framework automatically overlaps computation across different stages
3. **Memory Management**: Each stage maintains its own memory pool for efficient cache management

## Related Documentation

- [RFC: Multimodal Architecture Design](design/[RFC]multimodal_architechure.md)
- [SGL-JAX Architecture](../architecture/project-core-structure.md)
