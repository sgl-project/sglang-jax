# Basic Usage

This section keeps the model-neutral serving path small. Use it to understand the common launch and request flow before choosing a model recipe.

For model-specific hardware matrices, launch flags, and benchmark summaries, use the [cookbook overview](../cookbook_overview.md). The cookbook navigation itself lives in `docs/cookbook/docs.json`.

## Start a Server

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-8B \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 30000
```

Pick the model, topology, parallelism, and memory flags from the matching cookbook recipe. For reusable launcher templates, see [Deployment](../deployment/index.md). For flag categories and tuning notes, see [Server Arguments](../features/server_arguments.md).

## Send a Request

SGL-JAX exposes OpenAI-compatible endpoints once the server is ready:

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

The native generation endpoint is available at `/generate`, and the server exposes OpenAPI docs at `/docs`, `/redoc`, and `/openapi.json`.

## Next Steps

- [Deployment](../deployment/index.md): Docker, GKE Indexed Job, SkyPilot, and troubleshooting.
- [Server Arguments](../features/server_arguments.md): model, parallelism, memory, scheduling, JIT, and API flags.
- [Radix Cache](../features/radix_cache.md): prefix KV cache behavior.
- [Architecture](../architecture/index.md): runtime internals for contributors.
