---
title: "Install SGL-JAX"
---

# Install SGL-JAX

Cookbook recipes assume SGL-JAX is installed from source in the serving environment.

```bash
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax
pip install -e "python[tpu]"
```

Use the extra that matches your hardware:

- `python[tpu]` for TPU serving.
- `python[gpu]` for GPU experiments.
- `python[cpu]` for CPU-only debugging.

After installation, return to the model recipe and use its §2.3 launch command. For reusable launcher patterns, see [Deployment references](/deployment).

The full canonical installation guide lives in the main Sphinx docs at `docs/get_started/install.md`.
