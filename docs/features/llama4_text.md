# Llama 4 text-only (development draft)

Native `Llama4ForCausalLM` and the text subtree of
`Llama4ForConditionalGeneration` reuse the Llama model/decoder flow, `RMSNorm`,
`TopK`, `EPMoE` and strict `WeightLoader` recipes. Model-specific code handles
input-scaled routing, interleaved RoPE/NoPE and fixed chunk-local attention.
The full KV cache is retained; chunk-local masking is not a sliding window.

Requires `--moe-backend epmoe` and flash attention. Image, video and audio
requests are rejected. Vision, quantization, LoRA, sequence parallelism,
speculative decoding and expert-load balancing are outside this draft's scope.
This is not a validated deployment recipe for the official 400B checkpoint.

## Validation

From the repository root with development dependencies (HF parity needs PyTorch):

```sh
JAX_PLATFORMS=cpu python -m unittest sgl_jax.test.test_llama4 -v
python -m pytest python/sgl_jax/test/test_rpa_v3_kv_writeback.py -k chunk_local
```

CPU tests cover HF FP32 parity (`atol=rtol=3e-5`), loading, routing,
rotary/norm/temperature, cache reuse and request rejection using synthetic weights.
GMM uses CPU interpret mode and attention uses the paged reference. Four
chunk/cache cases require TPU and skip on CPU. Tests are registered in the
existing CPU and TPU suites.

Local Windows validation uses an uncommitted OS-import adapter (`resource` and
standard asyncio instead of `uvloop`), not model/kernel stubs. Linux serving,
TPU/BF16, authorized official-checkpoint, multi-device, long-context and
performance validation remain open. No context-length or throughput claim.

Architecture reference: [Transformers 5.12.1 Llama 4](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/llama4/modeling_llama4.py).
Official checkpoint access and the Llama 4 Community License remain the user's responsibility.
