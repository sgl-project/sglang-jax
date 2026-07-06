# Cookbook

The cookbook is the model-oriented documentation surface. It keeps a Mintlify-style structure under `docs/cookbook/` so contributors can add or validate one model recipe without changing the main Sphinx navigation.

Start here:

| Page | Purpose |
|---|---|
| [Cookbook index](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/index.md) | Model and hardware coverage overview. |
| [Add a cookbook recipe](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/CONTRIBUTING.md) | Checklist for adding or validating a model recipe. |
| [Autoregressive models](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/autoregressive/index.md) | Text and vision-language decoder recipes. |
| [Diffusion models](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/diffusion/index.md) | Text-to-image/video recipes. |
| [Benchmark references](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/base/benchmarks/index.md) | Detailed benchmark and evaluation reports. |
| [Troubleshooting](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/troubleshooting.md) | Cross-recipe failure modes. |

The Sphinx build excludes the cookbook source tree intentionally. Keep cookbook navigation in `docs/cookbook/docs.json`.
