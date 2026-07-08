# Attention Backend

SGL-JAX exposes a small user-facing attention backend switch, while the runtime may wrap that backend for MLA or hybrid linear-recurrent models.

## User-facing choices

`--attention-backend` accepts three values:

| Value | Runtime behavior |
|---|---|
| `fa` | Default. Uses FlashAttention for MHA/GQA models and the absorbed MLA Pallas backend for MLA models. |
| `fa_mha` | Forces MLA models through the decompressed MHA FlashAttention path. This is useful for kernel A/B checks, but uses much more KV cache than absorbed MLA. |
| `native` | Pure JAX/native attention path, mainly for CPU/debugging. If `fa` or `fa_mha` is requested on CPU, the runtime falls back to `native`. |

Example:

```bash
python3 -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen-7B-Chat \
  --trust-remote-code \
  --device=tpu \
  --attention-backend=fa
```

## Runtime backend matrix

| Backend class | Selected by | Main use |
|---|---|---|
| `FlashAttention` | `--attention-backend=fa` for MHA/GQA, or `fa_mha` for MLA fallback | TPU production attention with paged KV cache, SWA metadata, and Pallas kernels. |
| `MLAAttentionBackend` | `--attention-backend=fa` when `model_config.attention_arch == MLA` | Absorbed MLA path for DeepSeek-family models. |
| `NativeAttention` | `--attention-backend=native`, or CPU fallback | Debugging and CPU execution. |
| `HybridLinearAttnBackend` | Automatic wrapper for hybrid recurrent models | Routes full-attention layers to `FlashAttention`/`MLAAttentionBackend` and linear recurrent layers to KDA/GDN/Lightning backends. |
| `KDAAttnBackend` | Automatic under `HybridLinearAttnBackend` for Kimi Linear | Kimi Delta Attention recurrent branch. |
| `GDNAttnBackend` | Automatic under `HybridLinearAttnBackend` for Qwen3.5 hybrid configs | Gated DeltaNet recurrent branch. |
| `LightningAttnBackend` | Automatic under `HybridLinearAttnBackend` for Bailing MoE V2.5 / Ling-2.6-flash | Lightning / Simple GLA recurrent branch. |

## Notes for contributors

All attention backends inherit from `AttentionBackend` in `python/sgl_jax/srt/layers/attention/base_attn_backend.py`. A backend provides:

- `get_forward_metadata(batch)` for host-side metadata construction.
- `__call__(q, k, v, layer, forward_batch, **kwargs)` for the JIT-side attention computation.
- PyTree flatten/unflatten support when backend state crosses the JIT boundary.

Backend selection is centralized in `ModelRunner._get_attention_backend()`. Hybrid recurrent wrapping happens after the full-attention backend is created. For implementation details, see [Layers and Attention](../architecture/06-layers-and-attention.md).
