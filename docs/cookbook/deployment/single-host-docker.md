---
title: "Single-host Docker"
---

# Single-Host Docker Launcher

Reusable Docker recipe for serving a model on a single TPU host (v6e-4, v6e-8, v7x-8). Cookbook recipes that fit on one host reference this page.

## Prerequisites

- A provisioned TPU VM (single host, any generation) with Docker installed.
- Outbound network for the model weights (HuggingFace) — or pre-staged weights on a mounted volume.

If you're provisioning the host via SkyPilot or GKE, use those launchers instead — they handle multi-node coordination too. This page is for bare-VM single-host serving.

## Quickstart (prebuilt image)

The fastest path: use the prebuilt `lmsysorg/sglang-jax` image, which already has sglang-jax installed.

> Prebuilt images are published at <https://hub.docker.com/r/lmsysorg/sglang-jax>. Pick a `<TAG>` that matches the recipe's `Tested build` (cookbook recipes pin to `sglang-jax 0.1.0` and the matching image tag).

### Pull image

```bash
docker pull lmsysorg/sglang-jax:<TAG>
```

### Run service

Replace the placeholders with values from the cookbook recipe you're following:

```bash
docker run --rm -it \
  --name sglang-jax \
  --privileged \
  --network=host \
  -e JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
  -e HF_HOME=/tmp/models \
  -v /tmp/jit_cache:/tmp/jit_cache \
  lmsysorg/sglang-jax:<TAG> \
  python3 -u -m sgl_jax.launch_server \
  --model-path <HF_REPO_OR_LOCAL_PATH> \
  --trust-remote-code \
  --tp-size <DEVICE_COUNT> \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static <0.88-0.95> \
  --download-dir /tmp/models \
  --host 0.0.0.0 --port 30000
```

| Placeholder | How to fill |
|---|---|
| `<TAG>` | Pick the image tag matching the recipe's `Tested build`. |
| `<HF_REPO_OR_LOCAL_PATH>` | HuggingFace repo id (e.g. `Qwen/Qwen-7B-Chat`) or absolute path inside the container. |
| `<DEVICE_COUNT>` | Total JAX devices on this host. v6e: number of chips. v7x: number of chips × 2 (see [TPU topology reference](../base/tpu-topology-reference.md)). |
| `<0.88-0.95>` | `--mem-fraction-static`. 0.88 is the TPU default; raise to 0.9–0.95 for dedicated serving with no other processes on the host. |

If you have pre-downloaded weights on the host, add `-v /path/to/models:/models` and point `--model-path` at the in-container mount.

### Verify the service

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<HF_REPO_OR_LOCAL_PATH>",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

The server is ready once you see `Uvicorn running on http://0.0.0.0:30000` in the logs.

## Build from base image (alternative)

If you need a custom build (e.g., to land a local patch before the next image tag), boot the JAX TPU base image and install sglang-jax inside.

### Boot the container

```bash
docker run -it --privileged \
  --shm-size=32g \
  --ipc=host \
  --network=host \
  -v /dev:/dev \
  -v /tmp/jit_cache:/tmp/jit_cache \
  us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1 bash
```

**Flag walkthrough**
- `--privileged` + `-v /dev:/dev` — required for the container to see TPU devices.
- `--network=host` — avoids Docker NAT around port 30000 (and any custom port you use).
- `--ipc=host` + `--shm-size=32g` — required for JAX inter-process collectives on a single host.
- `-v /tmp/jit_cache:/tmp/jit_cache` — persists the JAX compilation cache across container restarts (saves ~4 min cold-start).

If you have a model PVC / pre-downloaded weights, add `-v /path/to/models:/models`.

### Install SGL-JAX (inside the container)

```bash
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax
pip install -e "python[tpu]"
```

Or, with `uv` for faster installs:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e "python[tpu]"
```

### Launch the server

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
  python -m sgl_jax.launch_server \
  --model-path <HF_REPO_OR_LOCAL_PATH> \
  --trust-remote-code \
  --tp-size <DEVICE_COUNT> \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static <0.88-0.95> \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Then use the same `curl` verification shown in the Quickstart section above.

For multi-host launches (TPU slices spanning more than one node), use [GKE Indexed Job launcher](gke-indexed-job.md) or [SkyPilot launcher](skypilot.md).

## Related

- [GKE Indexed Job launcher](gke-indexed-job.md) — multi-host on Kubernetes.
- [SkyPilot launcher](skypilot.md) — multi-host via SkyPilot.
- [Launch flags reference](../base/launch-flags-reference.md) — full launch flag inventory.
