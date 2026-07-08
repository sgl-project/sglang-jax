---
title: "Single-host Docker"
---

# Single-host Docker

Use this deployment pattern for one TPU host, such as v6e-4, v6e-8, or v7x-8. Model recipes provide the exact `--model-path`, topology, memory, and scheduler flags.

Common conventions:

- Image: `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1`.
- Install: `pip install -e "python[tpu]"` after cloning `https://github.com/sgl-project/sglang-jax.git`.
- Entrypoint: `python -m sgl_jax.launch_server`.
- Compilation cache: set `JAX_COMPILATION_CACHE_DIR` to a writable path such as `/tmp/jit_cache`.

Follow the current recipe's §2.3 launch command after preparing the container. For runtime flags, see [Launch flags reference](/base/launch-flags-reference). For failures, see [Troubleshooting](/deployment/troubleshooting).

The full canonical template lives in the main Sphinx docs at `docs/deployment/single-host-docker.md`.
