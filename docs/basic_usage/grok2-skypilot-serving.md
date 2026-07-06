# Grok-2 Usage

Grok-2 model launch and benchmark notes now live in the cookbook. The old SkyPilot-only page was intentionally collapsed into pointers so the validated hardware path, tokenizer note, base-model caveats, and benchmark numbers stay in one place.

| Use case | Current page |
|---|---|
| Validated Grok-2 recipe | [Grok-2 cookbook recipe](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/autoregressive/Grok/Grok2.md) |
| SkyPilot launcher mechanics | [SkyPilot deployment template](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/deployment/skypilot.md) |
| GKE Indexed Job pattern | [GKE Indexed Job launcher](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/deployment/gke-indexed-job.md) |
| Cross-recipe troubleshooting | [Cookbook troubleshooting](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/troubleshooting.md) |

Grok-2 is a base model without a native chat template. Use the raw completions path described in the cookbook recipe rather than chat-format eval examples.
