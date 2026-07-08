# Cookbook

The cookbook is the model-oriented documentation surface. It keeps a Mintlify-style structure under `docs/cookbook/` so contributors can add or validate one model recipe without changing the main Sphinx navigation.

The Sphinx build excludes `docs/cookbook/**` intentionally. Keep cookbook navigation inside `docs/cookbook/docs.json`; that file is the source of truth for the cookbook surface.

## Scope

- Model recipes live under `docs/cookbook/autoregressive/` and `docs/cookbook/diffusion/`.
- Shared recipe references live under `docs/cookbook/base/`.
- Cookbook contribution rules live in `docs/cookbook/add-recipe.md`.
- Cross-cutting launcher templates and troubleshooting live in [Deployment](deployment/index.md), not in the cookbook.

Use the cookbook for model-specific launch commands, validation status, benchmark summaries, and hardware matrices. Use the main Sphinx docs for installation, deployment templates, runtime features, architecture, and developer workflow.
