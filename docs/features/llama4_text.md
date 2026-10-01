# Llama 4 text-only implementation (development draft)

This change adds native JAX/NNX `Llama4ForCausalLM` and the **text subtree only**
of `Llama4ForConditionalGeneration`. It is intended for Maverick architecture
onboarding, not a validated deployment recipe for the full 400B checkpoint.
Image, video and audio requests are rejected explicitly, including OpenAI chat
content before template conversion. No vision encoder or projector is loaded.

## Architecture and loading

- The configured MoE layer schedule selects dense SwiGLU or routed plus shared
  experts. Routing takes top-k **raw logits**, then FP32 sigmoid without
  renormalization. Scores scale each selected expert's **input**, not its output.
  The shared expert is always added without a gate.
- Selected token/expert pairs use existing expert-parallel `EPMoE` grouped
  matrix multiplication. There is no dense all-experts production fallback.
  This initial implementation requires `--moe-backend epmoe`.
- RoPE uses interleaved pairs with FP32 rotation and no Q/K weight permutation.
  `no_rope_layers` controls rotation; `layer_types` controls chunked versus full
  attention. Q/K normalization is weightless and only enabled on RoPE layers
  when configured. NoPE temperature tuning uses absolute request positions.
  Scout's Q/K norm and scaled-RoPE settings are not imposed on Maverick.
- Fixed chunk-local masking is distinct from a rolling window. The existing
  flash attention RPA v3 kernel additionally compares the integer chunk IDs of
  absolute Q/K positions. All KV entries and page-table positions remain intact,
  so global layers and prefix reuse do not lose earlier chunks. This intentionally
  does not introduce chunk-based cache eviction or claim optimized long-context
  memory/throughput. Chunk-local layers require the flash attention backend;
  custom speculative tree masks are unsupported.
- Strict `WeightLoader` mappings accept text `model.*` / `lm_head.weight`, or
  official conditional-generation `language_model.model.*` /
  `language_model.lm_head.weight`. Fused expert `[E,H,2I]` weights are split on
  the final axis; down weights retain `[E,I,H]`. Missing required text weights
  fail; unused vision tensors are not substituted for required text parameters.

Quantization (including official FP8 variants), vision, LoRA, sequence
parallelism, speculative decoding, and expert-load balancing are outside this
initial scope. Multi-device TP/EP is unvalidated. No maximum-context or
throughput claim is made. Do not infer the gated official Maverick configuration
from the default Transformers configuration.

## Tests and open gates

From the repository root with the normal development dependencies:

```sh
JAX_PLATFORMS=cpu python -m unittest sgl_jax.test.test_llama4 -v
python -m pytest python/sgl_jax/test/test_rpa_v3_kv_writeback.py -k chunk_local
```

The first file is registered in `unit-test-cpu`. It exercises real JAX/NNX,
strict safetensors loading, EPMoE, rotary/norm/temperature, and a test-only
backend calling the repository's CPU paged-attention reference. Tiny random
HF logits comparison additionally requires PyTorch and Transformers 5.12.1.
These tests use no downloaded model weights.

The second command requires TPU DMA support and is registered through the
existing `unit-test-tpu-v6e-1` file. Its cases cover cold/warm prefill, chunked
prefill, decoding around 8192 and 32768, ragged sequences, KV writeback and
subsequent global consumption of a retained prefix. CPU reference tests are
not evidence that this Pallas kernel compiled or executed on TPU.

Local validation: **7 focused CPU tests passed**, including tiny HF FP32 logits
parity (`atol=rtol=3e-5`), native registry resolution, both checkpoint namespaces,
missing/malformed expert rejection, input-scaled EPMoE, rotary/norm/temperature,
prefill/decode/prefix reuse and multimedia rejection. Black, isort, Ruff,
in-memory Python compilation and `git diff --check` passed.

The validation host is Windows with x64 Python 3.13, JAX 0.11.1, Flax 0.12.9,
Transformers 5.12.1 and PyTorch CPU. A session-only import adapter supplies
fail-fast Unix `resource` operations and the standard-library asyncio policy
in place of unavailable `uvloop`. It is not committed and does not replace
model or kernel operations. These results are not Linux serving validation.
GMM ran in CPU interpret mode and attention used the paged reference, not TPU.
The four TPU chunk-local cases collected successfully and skipped on CPU.
Before upstream acceptance, require:

1. Reproduce the CPU suite in the standard Linux development environment.
2. Actual TPU RPA/MoE regression execution and BF16 accuracy, including chunk boundaries.
3. Official authorized checkpoint/tokenizer accuracy and deterministic generation.
4. Multi-device TP/EP, service-level prefix caching and long-context validation.
5. Hardware-specific memory, throughput and latency measurements.

No official checkpoint was executed; official metadata/config access is gated.
The Llama 4 Community License remains the checkpoint user's responsibility.

## Sources

- [Transformers 5.12.1 Llama 4 implementation](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/llama4/modeling_llama4.py)
  and [configuration](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/llama4/configuration_llama4.py).
- [Meta model card](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md).
- Language-first upstream precedent: [SGLang #5092](https://github.com/sgl-project/sglang/pull/5092);
  later vision: [#5254](https://github.com/sgl-project/sglang/pull/5254);
  long-context cache correction: [#6162](https://github.com/sgl-project/sglang/pull/6162).
- JAX contribution precedents: [narrow onboarding #1566](https://github.com/sgl-project/sglang-jax/pull/1566),
  [strict loading #1743](https://github.com/sgl-project/sglang-jax/pull/1743),
  [validation #1656](https://github.com/sgl-project/sglang-jax/pull/1656).
