# Attention Backend

SGL-JAX exposes a small user-facing attention backend switch, while the runtime may wrap that backend for MLA or hybrid linear-recurrent models.

## User-facing choices

`--attention-backend` accepts four values:

| Value | Runtime behavior |
|---|---|
| `fa` | Default. Uses FlashAttention for MHA/GQA models and the absorbed MLA Pallas backend for MLA models. |
| `fa_mha` | Forces MLA models through the decompressed MHA FlashAttention path. This is useful for kernel A/B checks, but uses much more KV cache than absorbed MLA. |
| `native` | Pure JAX/native attention path, mainly for CPU/debugging. If `fa` or `fa_mha` is requested on CPU, the runtime falls back to `native`. |
| `dsa_sparse` | DeepSeek Sparse Attention (DSA): a lightning-indexer selects the top-scoring pages of past tokens per query and attention runs only over that page set (page-block sparse guided by DSA scores), with IndexShare cross-layer reuse of the selection. For MLA models whose config carries the `index_*` fields only. See [DeepSeek Sparse Attention](#deepseek-sparse-attention-dsa_sparse). |

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
| `DSASparseAttentionBackend` | `--attention-backend=dsa_sparse` for MLA models with `index_*` config | DeepSeek Sparse Attention: lightning-indexer top-k + sparse MLA over the selected pages (page-block), with IndexShare. |
| `NativeAttention` | `--attention-backend=native`, or CPU fallback | Debugging and CPU execution. |
| `HybridLinearAttnBackend` | Automatic wrapper for hybrid recurrent models | Routes full-attention layers to `FlashAttention`/`MLAAttentionBackend` and linear recurrent layers to KDA/GDN/Lightning backends. |
| `KDAAttnBackend` | Automatic under `HybridLinearAttnBackend` for Kimi Linear | Kimi Delta Attention recurrent branch. |
| `GDNAttnBackend` | Automatic under `HybridLinearAttnBackend` for Qwen3.5 hybrid configs | Gated DeltaNet recurrent branch. |
| `LightningAttnBackend` | Automatic under `HybridLinearAttnBackend` for Bailing MoE V2.5 / Ling-2.6-flash | Lightning / Simple GLA recurrent branch. |

## DeepSeek Sparse Attention (`dsa_sparse`)

DSA cuts the cost of long-context attention by attending to a **selected subset** of past
tokens instead of the full causal history. A small "lightning indexer" scores past tokens
for each query and the highest-scoring ones (budget `index_topk`) are chosen. The selection
is computed on the indexer layers and **reused across the following layers (IndexShare)**,
so the indexer cost is amortized.

Selection here is at **page granularity** (`page_size` tokens per unit), not exact per-token:
the indexer's per-token scores are max-pooled to a score per page and the top
`ceil(index_topk / page_size)` pages are attended in full. This is therefore **page-block
sparse attention guided by the DSA indexer scores** — a superset of the exact token top-`k`
set (attending whole pages shifts the softmax denominator), not a bit-exact token-level DSA.
In practice it tracks dense closely on long-context retrieval (see the PR's NIAH/GSM8K
results); treat page size and `index_topk` as the accuracy/speed knobs.

Only MLA models whose config carries the DSA `index_*` fields (indexer head dim / heads /
top-k) are eligible; other models ignore the flag.

### Enabling it

```bash
python3 -u -m sgl_jax.launch_server \
  --model-path <deepseek-sparse-MLA-checkpoint> \
  --device=tpu \
  --attention-backend=dsa_sparse \
  --dsa-use-pallas          # use the Pallas kernels (default: the jnp reference)
```

Two environment variables tune the sparse path:

| Env var | Meaning |
|---|---|
| `DSA_INDEX_TOPK` | Override the config's `index_topk` (selection budget, in tokens) at backend construction. Larger = closer to dense, slower. The kernel is compiled for this value, so **changing it requires a relaunch**. |
| `DSA_PREFILL_SPARSE=1` | Opt in to running **prefill** through the fused sparse-MLA prefill kernel (below). Default off ⇒ prefill runs dense and only decode is sparse. |

### Sparse prefill and its current scope

With `DSA_PREFILL_SPARSE=1`, the prefill (EXTEND) step also attends only to the indexer's
selected pages via a fused self-write + sparse-MLA Pallas kernel, which is where the
long-context prefill speedup comes from.

This initial version supports a **single-sequence, single-shot** prefill only. It does
**not** yet support radix/prefix caching, request batching, or chunked prefill (a cache
hit / multi-seq / split prompt turns prefill into a partial extend the sparse page table
cannot represent). The runtime **fails fast with an actionable message** when
`DSA_PREFILL_SPARSE=1` is combined with an incompatible config, so it must be launched
with:

```bash
DSA_PREFILL_SPARSE=1 DSA_INDEX_TOPK=4096 python3 -u -m sgl_jax.launch_server \
  --model-path <checkpoint> --device=tpu \
  --attention-backend=dsa_sparse --dsa-use-pallas \
  --disable-radix-cache \           # radix/prefix cache OFF
  --max-running-requests 1 \        # single sequence
  --context-length 49152 \
  --chunked-prefill-size 49152      # >= context => single-shot (no chunking)
```

Radix-cache, batching, and chunked-prefill support are follow-ups. The guard is gated on
`DSA_PREFILL_SPARSE`, so the default (dense-prefill) `dsa_sparse` path, other backends, and
CI are unaffected.

### Validation

Kernel correctness is covered in CI by
`test/srt/kernels/dsa/test_sparse_mla_prefill_parity.py`, which checks the sparse-MLA
prefill kernel against a masked-softmax reference on CPU (`interpret=True`, no TPU needed),
in both flat and paged-cache modes.

## Notes for contributors

All attention backends inherit from `AttentionBackend` in `python/sgl_jax/srt/layers/attention/base_attn_backend.py`. A backend provides:

- `get_forward_metadata(batch)` for host-side metadata construction.
- `__call__(q, k, v, layer, forward_batch, **kwargs)` for the JIT-side attention computation.
- PyTree flatten/unflatten support when backend state crosses the JIT boundary.

Backend selection is centralized in `ModelRunner._get_attention_backend()`. Hybrid recurrent wrapping happens after the full-attention backend is created. For implementation details, see [Layers and Attention](../architecture/06-layers-and-attention.md).
