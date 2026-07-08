---
title: "Troubleshooting"
---

# Troubleshooting

Use this page for cross-recipe failures. Model-specific constraints remain in each recipe's §2.4 Configuration Tips.

## Startup

- First request hangs for minutes: set `JAX_COMPILATION_CACHE_DIR` so XLA/Pallas kernels can be cached.
- Weight load OOM: verify `--tp-size` matches available JAX devices, then lower `--mem-fraction-static`.
- Unsupported architecture: confirm the model architecture is registered in `python/sgl_jax/srt/models/`, or use the recipe's `--json-model-override-args` guidance when applicable.

## Multi-node

- Hang at distributed initialization: verify rank 0 is reachable from every other node and that `--dist-init-addr` resolves to the intended host.
- TPU process count mismatch: verify `TPU_PROCESS_ADDRESSES` has exactly one entry per node and matches `--nnodes`.

## Runtime

### SWA pool exhaustion (MiMo, hybrid-attention models)

Tune `--swa-full-tokens-ratio` and `--max-running-requests` using the recipe's validated values as the starting point.

- High TTFT with normal ITL: raise `--chunked-prefill-size` or `--max-prefill-tokens` if HBM allows.
- Throughput collapse at high concurrency: set `--max-running-requests` to a value matched to the KV pool.

## Tokenizer and Cache

- Tokenizer missing: pass `--tokenizer-path` when a checkpoint ships without tokenizer files.
- Remote model code required: add `--trust-remote-code` for checkpoints with custom modeling files.
- Cache churn: do not share one `JAX_COMPILATION_CACHE_DIR` across recipes with different shapes.

The full canonical troubleshooting page lives in the main Sphinx docs at `docs/deployment/troubleshooting.md`.
