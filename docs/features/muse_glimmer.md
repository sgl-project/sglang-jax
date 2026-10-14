# Muse Glimmer on TPU

## Motivation

Muse Glimmer is a text-generation architecture that emits Harmony-style reasoning
channels and ATEM tool calls. Its assistant checkpoint can also be used by
DFlash speculative decoding. SGLang-JAX needs model loading, output parsing, and
DFlash runtime support to serve the architecture correctly on TPU.

## Goals

- Load Hugging Face-compatible Muse Glimmer target and assistant checkpoints.
- Run the target model with tensor parallelism on TPU.
- Parse reasoning separately from user-visible content in streaming and
  non-streaming responses.
- Parse native ATEM tool calls, including `tool_choice="required"` requests.
- Support DFlash verification with greedy and target-only stochastic sampling.
- Keep long-context and long-lived-server KV-cache behavior correct.

## Non-goals

- Grammar-constrained DFlash decoding.
- DFlash support for repetition, frequency, or presence penalties.
- DFlash support for `min_p` or request logprobs.
- Publishing or downloading model weights as part of SGLang-JAX.

## Design

### Model and configuration

Two local configuration classes normalize target and assistant checkpoints into
SGLang-JAX's existing model interfaces. The target implementation provides:

- gated attention with per-head Q/K normalization;
- alternating sliding-window and global-attention layers;
- Gemma-style RMS normalization and gated MLP blocks;
- tensor-parallel weight mappings and auxiliary hidden-state capture for the
  draft model.

The assistant reuses the existing DFlash model and accepts the checkpoint's
encoder and output-normalization weight aliases.

### Reasoning and tool calls

Muse Glimmer output uses Harmony channel headers. A reasoning detector routes
`to=self` content to `reasoning_content` and `to=user` content to the visible
answer while preserving partial markers across streaming chunks.

ATEM tool calls are parsed from the model's native XML-like format. Because the
model emits this format itself, required tool calls bypass JSON-schema output
constraints. Both streaming and non-streaming paths normalize tool names and
JSON-encode arguments for the OpenAI-compatible response.

### DFlash verification

The DFlash path treats the checkpoint block width as a maximum and verifies the
runtime-requested number of draft positions. Verification uses target-only
rejection sampling: deterministic draft proposals are accepted according to the
target probability; rejected and bonus tokens are sampled from the filtered
target distribution. `temperature`, `top_k`, and `top_p` are supported.

Long-context position arithmetic remains `int32` to avoid overflow above 32,767
tokens. Draft KV capacity follows the target's post-hybrid capacity, and
request-turnover page mappings are cleared when draft layers use full KV backing.
These rules keep persistent servers correct as requests enter and leave the
batch.

## User interface

A representative launch uses the existing server entry point:

```bash
python -m sgl_jax.launch_server \
  --model-path /path/to/muse-glimmer-target \
  --device tpu \
  --tp-size 2 \
  --attention-backend fa \
  --page-size 16 \
  --reasoning-parser muse_glimmer \
  --tool-call-parser muse_glimmer \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /path/to/muse-glimmer-assistant \
  --speculative-num-draft-tokens 4
```

The exact context length, prefill limits, and precompile buckets should be tuned
for the TPU topology and workload.

## Validation

The implementation includes focused unit tests for:

- target and assistant configuration normalization;
- model weight mappings;
- Harmony reasoning and ATEM tool-call parsing;
- runtime-width DFlash verification and overlap metadata;
- stochastic acceptance and target-token sampling;
- long-context attention position arithmetic;
- hybrid KV capacity and request-turnover cache mappings.

An end-to-end TPU v7e validation completed 979 persistent-server requests at
concurrency four with no failures. The run exercised reasoning, required tool
calls, prefix-cache reuse, long-context requests, stochastic sampling, and
DFlash overlap scheduling.
