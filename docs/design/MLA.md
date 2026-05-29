# RFC: MLA Integration (`sglang-jax`) — v5

## 1. Background

`sglang-jax` needs MLA absorbed-path support to run DeepSeek-V2/V3 and other MLA-based models. The current runtime only has an MHA attention backend ([flashattention_backend.py](../../python/sgl_jax/srt/layers/attention/flashattention_backend.py)) and `MHATokenToKVPool` ([memory_pool.py](../../python/sgl_jax/srt/mem_cache/memory_pool.py));

## 2. Goals & non-goals

### Goals

- Integrate the MLA v2 Pallas kernel as a dedicated attention backend.
- Add a native MLA KV cache that stores only `[c_kv ‖ k_pe]` per token, shared across heads.
- Run the absorb path as factored two-step matmuls through the low-rank bottleneck (`W_UQ_nope → W_UK` on Q, `W_UV → W_O` on output) rather than pre-fusing.
- Support prefill and mixed modes, and decode via the kernel's ragged-paged interface.

### Non-goals

- **FP8 / quantized KV cache.** Latents are stored in bf16. The extract-and-requantize path that sglang-GPU and tpu-inference use to put `W_UK` / `W_UV` in FP8 alongside an FP8 cache is deliberately out of scope for this RFC; revisit in a follow-up once the bf16 path is stable.
- **Speculative decoding on the MLA backend.** The MLA v2 kernel has no `custom_mask`, so EAGLE draft-extend and target-verify forward modes cannot share this backend (see §3.5).
- **DP attention.** The cache is replicated across TP ranks because the MLA latent is single-head (§3.1, §3.8). Sharding by requests with a TP all-reduce is a follow-up, not part of v1.


## 3. Design

### 3.1 Decision summary

| Concern | Decision | Core reason |
|---|---|---|
| Attention backend | **New** `MLAAttentionBackend` class | Kernel arity, cache rank, sharding specs, and alignment constraints all differ from RPA v3 (§3.4, §3.5, §3.6). |
| Memory pool | **New** `MLATokenToKVPool` class | The MHA pool's 5D interleaved `[K,V]` layout and packing semantics do not match MLA v2's 4D concatenated layout (§3.4). |
| Absorb-path weights | **Kept factored** at runtime: `W_UQ_nope → W_UK` on Q, `W_UV → W_O` on the output | Pre-fusing erases the rank-`Dk` / rank-`R` bottleneck and inflates FLOPs by ~3× (§3.3). |
| Cache sharding | Replicated across TP (no `kv_partition_axis` split) | The MLA latent is single-head; splitting along a non-existent head axis has no benefit and TP attention requires DP attention to recombine — follow-up. |

### 3.2 Alternatives considered: extend-in-place vs. new classes

The central architectural question is whether MLA should be a branch inside `FlashAttentionBackend` + `MHATokenToKVPool`, or a pair of new classes. Below is the same workload written both ways.

#### 3.2.A Extend-in-place sketch

```python
# memory_pool.py
class MHATokenToKVPool(KVCache):
    def __init__(self, ..., is_mla=False,
                 kv_lora_rank=None, qk_rope_head_dim=None):
        self.is_mla = is_mla
        if is_mla:
            # 4D concat layout, packing axis is TOKENS
            self.buffer_shape = (num_pages,
                                 page_size_per_kv_packing,
                                 kv_packing,
                                 align_to(kv_lora_rank + qk_rope_head_dim, 128))
            self.kv_sharding = NamedSharding(mesh, P(None, None, None, None))
        else:
            # 5D interleaved layout, packing axis is K/V
            self.buffer_shape = (num_pages, page_size,
                                 head_num * 2 // packing, packing, head_dim)
            self.kv_sharding = NamedSharding(
                mesh, P(None, None, kv_partition_axis, None, None))
        self._create_buffers()

    def set_kv_buffer(self, layer_id, loc, k, v, is_decode):
        if self.is_mla:
            # k actually carries [c_kv ‖ k_pe]; v is unused — API misnomer
            payload = k
            self.kv_buffer[layer_id] = update_concat_cache(
                payload, loc, self.kv_buffer[layer_id], ...)
        else:
            fused = merge_kv(k, v)  # 3D -> 5D packed
            self.kv_buffer[layer_id] = update_fused_kv_cache(
                fused, loc, self.kv_buffer[layer_id], ...)

    def get_fused_kv_buffer(self, layer_id):
        # Returns 4D or 5D depending on self.is_mla — every caller must branch.
        return self.kv_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id):
        if self.is_mla:
            kv = self.kv_buffer[layer_id]
            return kv[..., :self.kv_lora_rank], kv[..., self.kv_lora_rank:]
        return self.kv_buffer[layer_id]

    def tree_flatten(self):
        # aux_data must conditionally carry is_mla/kv_lora_rank/qk_rope_head_dim
        ...


# flashattention_backend.py
class FlashAttention(AttentionBackend):
    def __call__(self, q, k, v, layer, forward_batch, pool, ...):
        if layer.is_mla:
            ql_nope, q_pe, new_kv_c, new_k_pe = self._split_mla_inputs(q, k, v)
            # r_dim = 64 violates r_dim % 128 == 0; pad here
            q_pe     = pad_last(q_pe,     target=128)
            new_k_pe = pad_last(new_k_pe, target=128)

            cache = pool.get_fused_kv_buffer(layer.layer_id)    # 4D
            in_specs_mla  = (P(None, None), P(None, None),      # ql_nope, q_pe
                             P(None), P(None),                  # new_kv_c, new_k_pe
                             P(None, None, None, None),         # 4D cache
                             P(), P(), P(), P())                # metadata (no cu_kv_lens)
            out_specs_mla = (P(None, None),                     # o_latent
                             P(None, None, None, None))         # updated cache
            return jax.shard_map(mla_ragged_paged_attention,
                                 in_specs_mla, out_specs_mla, ...)(...)
        else:
            cache = pool.get_fused_kv_buffer(layer.layer_id)    # 5D
            in_specs_mha  = (P(None, self.kv_partition_axis), ...,
                             P(None, None, self.kv_partition_axis, None, None),
                             P(), P(), P(), P(), P(), P(), P())
            out_specs_mha = (P(None, self.kv_partition_axis),
                             P(None, None, self.kv_partition_axis, None, None))
            return jax.shard_map(ragged_paged_attention_v3,
                                 in_specs_mha, out_specs_mha, ...)(...)

    def get_forward_metadata(self, batch):
        md = FlashAttentionMetadata()
        ...  # builds cu_kv_lens, custom_mask, etc.
        if layer_is_mla(batch):
            md.cu_kv_lens = None      # MLA doesn't use it
            md.custom_mask = None     # MLA doesn't support it
            assert_alignment(md.page_indices, align=128)  # MLA-only check
        return md
```

What this costs:
- Every method forks on `self.is_mla`; no non-trivial code is shared between the two branches. `set_kv_buffer`, `get_*_buffer`, `_create_buffers`, and `tree_flatten/unflatten` each have two implementations glued together.
- `FlashAttentionMetadata` carries `cu_kv_lens`, `custom_mask`, and eagle/spec fields that MLA never uses; the MLA branch must null them out and keep them alive through the pytree.
- `in_specs`/`out_specs` have different ranks (5D vs 4D), so `shard_map` cannot be reused — two parallel call sites.
- The `k`/`v` names in `set_kv_buffer` are a lie in the MLA branch (the payload is `[c_kv ‖ k_pe]`), which will mislead future readers.
- `tree_flatten` aux_data gains optional MLA fields; mismatched flatten/unflatten paths are a common JIT-retrace bug.

#### 3.2.B New-class sketch

```python
# memory_pool.py — new class alongside MHATokenToKVPool, no is_mla flag
class MLATokenToKVPool(KVCache):
    def __init__(self, size, page_size, dtype,
                 kv_lora_rank, qk_rope_head_dim, layer_num, mesh, ...):
        super().__init__(...)
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self._create_buffers()       # 4D concat layout only

    def set_kv_buffer(self, layer_id, loc, payload, is_decode):
        # payload: [T, lkv_dim + r_dim]
        self.kv_buffer[layer_id] = update_concat_cache(
            payload, loc, self.kv_buffer[layer_id], ...)

    def get_fused_kv_buffer(self, layer_id):
        return self.kv_buffer[layer_id - self.start_layer]  # 4D, always


# mla_backend.py — new file, new class
class MLAAttentionBackend(AttentionBackend):
    def __call__(self, payload, layer, forward_batch, pool):
        ql_nope, q_pe, new_kv_c, new_k_pe = payload
        q_pe     = pad_last(q_pe,     target=128)
        new_k_pe = pad_last(new_k_pe, target=128)
        cache    = pool.get_fused_kv_buffer(layer.layer_id)
        return jax.shard_map(mla_ragged_paged_attention,
                             MLA_IN_SPECS, MLA_OUT_SPECS, ...)(
                                 ql_nope, q_pe, new_kv_c, new_k_pe, cache,
                                 self.forward_metadata.cu_q_lens,
                                 self.forward_metadata.page_indices,
                                 self.forward_metadata.seq_lens,
                                 self.forward_metadata.distribution)
```

What this costs: two small new files. Every method has one code path. Metadata carries exactly the fields the MLA kernel needs. Sharding specs are fixed at the class level. The MHA path is untouched, so MHA regressions from this change are impossible by construction.

#### 3.2.C Verdict

Proceed with the new-class design (3.2.B). The extend-in-place design (3.2.A) turns the shared classes into multiplexers with near-zero shared code, pollutes every caller with rank- and layout-dependent branches, and keeps inapplicable metadata fields alive. The cost of the split is one file (`mla_backend.py`) and one class (`MLATokenToKVPool`); both are naturally-bounded scopes. Dispatch happens once in `model_runner.py` (§3.10).

### 3.3 Why the absorb path stays factored

**FLOPs, per head per token:**

- Pre-fused (one matmul): `T · q_lora · R = T · 1536 · 512 = T · 786K`
- Factored (two matmuls): `T · q_lora · Dk + T · Dk · R = T · 196K + T · 65K = T · 262K`
- Pre-fusion costs ~3× the FLOPs because it skips the `Dk = 128` bottleneck.

Implication: `W_UQ_nope`, `W_UQ_rope`, `W_UK`, `W_UV`, `W_O` stay as separate per-head parameters. They are not fused at load time. `kv_b_proj` from the original DeepSeek weights is reinterpreted as the per-head `W_UK` / `W_UV` factors and otherwise disappears from the forward path.

### 3.4 Layout divergence (MHA pool vs. MLA v2 kernel cache)

MHA pool (5D):

```text
[num_pages, page_size, head_num*2 // packing, packing, head_dim]
                       ▲
                       └── packing axis holds K/V of the same token
```

MLA v2 kernel cache (4D):

```text
[num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
                                      ▲
                                      └── packing axis holds consecutive TOKENS of the same latent
```

The rank differs, the packing axis semantics differ (K/V-per-token vs. tokens-per-latent), and the last dim concatenates `[c_kv ‖ k_pe]` with 128-alignment padding instead of being a per-head `head_dim`. Reinterpreting bytes across layouts is not possible without shuffling.

The "reuse MHA with `head_num=1`" trick lands the singleton at axis 3, directly before the feature dimension, which flips the TPU tile size between `(1, 128)` and `(8, 128)` and is costly.

### 3.5 Kernel interface divergence (RPA v3 vs. MLA v2)

| Field | RPA v3 | MLA v2 |
|---|---|---|
| Q inputs | `queries [T, n_h, head_dim]` — single | `ql_nope [T, n_h, lkv_dim] + q_pe [T, n_h, r_dim]` — split, `ql_nope` is absorbed |
| Current K | `keys [T, n_kv, head_dim]` — per-head | `new_kv_c [T, lkv_dim]` — single-head latent |
| Current V | `values [T, n_kv, head_dim]` — per-head | `new_k_pe [T, r_dim]` — rotated key, no V |
| Cache | `[pages, page_size, n_kv*2, head_dim]` — interleaved | `[pages, page_size_per_kv_packing, kv_packing, align_to(lkv+r_dim, 128)]` — concat |
| Shared metadata | `kv_lens`, `page_indices`, `cu_q_lens`, `cu_kv_lens`, `distribution` | Same — our MLA metadata builder must still produce `cu_kv_lens` |
| `custom_mask` | ✓ (used for TARGET_VERIFY / spec-decode forward modes) | ✗ unsupported — MLA backend will not support speculative decoding (EAGLE draft-extend, target-verify, and related params) |
| `attention_sink` | ✓ | ✗ unsupported — this is for a sink/window-attention variant; not used by the DeepSeek-family MLA we target |
| `xai_temperature_len` | ✓ | ✗ unsupported — Grok-specific temperature knob, not applicable to MLA models |
| `causal` | Configurable (flipped to `0` when `custom_mask` is present) | we only run causal attention |
| Output 1 | `[T, n_h, head_dim]` — ready for `W_O` | `o_latent [T, n_h, lkv_dim]` — needs `W_UV` then `W_O` (factored, §3.3) |
| Output 2 | Updated interleaved cache | Updated concat cache |

The layer-level implication: MLA applies `W_UQ_nope → W_UK` on Q before the kernel and `W_UV → W_O` on the output after the kernel. `kv_b_proj` disappears from the forward path.

Backend-level implication: the MLA dispatch path in `model_runner.py` (§3.10) rejects speculative-decoding forward modes — no `custom_mask` means no target-verify/draft-extend variants can share the MLA backend without kernel-side changes.

### 3.7 `MLATokenToKVPool`

Location: new class in [memory_pool.py](../../python/sgl_jax/srt/mem_cache/memory_pool.py), alongside `MHATokenToKVPool`, sharing the `KVCache` base.

Buffer shape per layer:

```text
(num_pages, page_size_per_kv_packing, kv_packing, align_to(kv_lora_rank + qk_rope_head_dim, 128))
```

Last-dim layout: `[c_kv (lkv_dim) | k_pe_rotated (r_dim) | pad to 128-align]`.

Interface:

- `set_kv_buffer(layer_id, loc, payload, is_decode)` — single payload `[T, lkv_dim + r_dim]`, no K/V split.
- `get_fused_kv_buffer(layer_id)` — returns the 4D buffer directly.

No `merge_kv` (no interleaving). RoPE is applied on write: positions are known at write time and MLA v2 assumes pre-rotated `k_pe`.

Eviction accounting: per-token size is ~1.1 KB vs. MHA's ~98 KB, so upstream memory-budget logic must read from `pool.get_kv_size_bytes()` rather than assume an MHA-sized cache.

### 3.8 `MLAAttentionBackend`

Location: new file `sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py`. Inherits `AttentionBackend`.

`MLAAttentionMetadata`: `cu_q_lens`, `cu_kv_lens`, `page_indices`, `seq_lens`, `distribution`. `num_seqs` is dropped — it exists on `FlashAttentionMetadata` only to help shard-bookkeeping around `distribution`, and the MLA kernel's public signature does not consume it, so keeping it would just invite someone to pass it into the kernel call and trigger a retrace.

`__call__(payload, layer, forward_batch, pool)`:

1. Unpack `(ql_nope, q_pe, new_kv_c, new_k_pe)` from the layer.
2. Pad `new_k_pe` and `q_pe` from 64 to 128 to satisfy `r_dim % 128 == 0`.
3. Write `[new_kv_c ‖ new_k_pe]` into the MLA pool; the kernel handles this via its `updated_cache_kv` return.
4. Call `mla_ragged_paged_attention(ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, …)`.
5. Return `(o_latent, updated_cache_kv)`.

Sharding uses `shard_map` specs that match MLA v2's 4D tensor ranks, unlike FlashAttention's 5D specs.

### 3.9 `MLAAttention.__call__` absorbed path

```python
c_q      = q_a_layernorm(q_a_proj(hidden))                     # [T, q_lora_rank]
q_nope   = einsum('tc,hcd->thd', c_q,    W_UQ_nope)            # [T, n_h, qk_nope_head_dim=Dk]
ql_nope  = einsum('thd,hdr->thr', q_nope, W_UK)                # [T, n_h, kv_lora_rank=R]
q_pe     = RoPE(positions, q_b_rope_proj(c_q))                 # [T, n_h, qk_rope_head_dim]

kv_a     = kv_a_proj(hidden)
c_kv     = kv_a_layernorm(kv_a[:, :kv_lora_rank])
k_pe     = RoPE(positions, kv_a[:, kv_lora_rank:])

o_latent = mla_backend(
   payload=(ql_nope, q_pe, c_kv, k_pe),
   layer=self, forward_batch=forward_batch, pool=pool,
)                                                               # [T, n_h, kv_lora_rank=R]

o_v      = einsum('thr,hrd->thd', o_latent, W_UV)              # [T, n_h, v_head_dim=Dv]
return     einsum('thd,hde->te',  o_v,      W_O)               # [T, hidden]
```

Removed from the forward path: the original `kv_b_proj` (its weights are reinterpreted as the per-head `W_UK` / `W_UV` factors above) and V padding/unpadding. `o_proj` is kept as the second factor `W_O` of the output chain rather than fused into a wider `W_OV` (§3.3).

Each attention layer wraps core MLA with two factored matmul chains:

- **Q-side** (through the `Dk = 128` bottleneck): `c_q · W_UQ_nope → q_nope`, then `q_nope · W_UK → ql_nope`.
- **O-side** (through the `R = 512` latent): `attn_output · W_UV → o_v`, then `o_v · W_O → out`.

Each matmul sees either `Dk` or `R` as a contraction dim, never the wider fused product — the FLOP win from §3.3.

### 3.10 Dispatch

In `model_runner.py`:

- Instantiate `MLATokenToKVPool` when the model config indicates MLA (`kv_lora_rank` is set).
- Instantiate `MLAAttentionBackend` under the same condition.

Both live behind the same `KVCache` / `AttentionBackend` interfaces, so no call site beyond the factory changes.

### 3.11 Prior art

| Aspect | sglang (GPU, PyTorch) | tpu-inference (JAX/Pallas) | This RFC (`sglang-jax`) |
|---|---|---|---|
| Attention backend | Dedicated: `FlashMLABackend`, `FlashInferMLAAttnBackend`, `CutlassMLABackend`, `TRTLLMMLABackend` (e.g. `sglang/python/sglang/srt/layers/attention/flashmla_backend.py`) | Dedicated: `PallasMLAttentionBackendImpl` (`tpu_inference/layers/vllm/backends/flash_attn_mla.py`) | Dedicated: new `MLAAttentionBackend` |
| KV cache pool | Dedicated `MLATokenToKVPool` (`sglang/srt/mem_cache/memory_pool.py` L1437–1668); also FP4 subclass | **Reuses** vLLM's unified paged cache; no MLA-specific pool | Dedicated `MLATokenToKVPool` |
| Cache layout | 3D: `(size + page_size, 1, kv_lora_rank + qk_rope_head_dim)` — simple fused | 4D packed: `(pages, page_size/packing, packing, align_to(kv_dim, 128))` | 4D packed (same family as tpu-inference) |
| Absorb path | **Factored at runtime** — `w_kc`, `w_vc` applied via `bmm` in forward | **Factored at runtime** — `W_UK_T` applied on Q before kernel, `W_UV` applied on output after kernel (`flash_attn_mla.py` L152-156, L199-203); weight-time work is only extract-and-requantize of `W_UK_T` / `W_UV` from `kv_b_proj`, then `kv_b_proj` is deleted | **Factored at runtime** (same family) — see §3.3 FLOPs argument |
| Kernel | FlashMLA (custom CUDA w/ tile scheduler), FlashInfer `BatchMLAPagedAttentionWrapper`, cutlass | `mla_ragged_paged_attention` Pallas v2 | Same Pallas v2 kernel as tpu-inference |
| RoPE on `k_pe` | In forward pass before write | Applied pre-write or in kernel | Applied on write (pre-rotated before cache update) |

Where we diverge from each reference and why:

- **vs. tpu-inference on absorb**: we match tpu-inference on the runtime factoring (Q-side `q_nope @ W_UK_T` before the kernel, O-side `o_latent @ W_UV` after it) — neither side pre-fuses the full `W_UQ_nope · W_UK` or `W_UV · W_O` products that would skip the `Dk` / `R` bottleneck. The remaining differences are weight-loading plumbing: tpu-inference dequantizes the original `kv_b_proj`, splits it into per-head `W_UK_T` / `W_UV`, re-quantizes each factor to the KV-cache dtype, and deletes `kv_b_proj`. We don't re-quantize — we load `W_UK` and `W_UV` as plain bf16 per-head parameters and never keep a `kv_b_proj` shell around.
- **vs. tpu-inference on the pool**: tpu-inference piggybacks on vLLM's generic paged cache because vLLM already abstracts over MLA-sized tokens. `sglang-jax` does not have that abstraction — our paged cache is parametrised by `(head_num, head_dim)` and the 5D packed layout. Shoehorning MLA in (§3.2.A) keeps every caller branching on `is_mla`. A dedicated pool is cleaner and matches sglang-GPU's choice.
- **vs. sglang-GPU on layout**: sglang-GPU uses a trivial `(size, 1, kv_dim)` tensor because CUDA attention kernels don't need TPU packing. We need the 4D packed/aligned layout because Pallas vector-tile alignment requires it (same reasoning as tpu-inference).

## 4. Work breakdown

- [ ] `MLATokenToKVPool` in [memory_pool.py](../../python/sgl_jax/srt/mem_cache/memory_pool.py).
- [ ] `MLAAttentionBackend` + `MLAAttentionMetadata` in `sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py`.
- [ ] Expose `W_UQ_nope`, `W_UQ_rope`, `W_UK`, `W_UV`, `W_O` as separate per-head params on `MLAAttention` (no fusion at load time).
- [ ] Rewrite `MLAAttention.__call__` to the two-step factored absorb path (§3.9): `W_UQ_nope → W_UK` on Q, `W_UV → W_O` on output.
- [ ] Wire dispatch in `model_runner.py` (§3.10).
- [ ] Pad `r_dim=64` → 128 inside `MLAAttentionBackend`.
- [ ] Unit tests for pool (shape, set/get, pytree round-trip, `get_kv_size_bytes`) and backend (kernel call, shard_map specs, decode + extend parity vs. a naive reference).
- [ ] Unit test for the MLA backend (mirror of [test_flashattention.py](../../python/sgl_jax/test/test_flashattention.py)) verifying `ForwardBatch` compatibility and numerical accuracy.

## 5. Additional notes

- **`r_dim = 64` alignment.** Violates `r_dim % 128 == 0`. The backend pads `q_pe` and `new_k_pe` to 128 on the fly; the pool stores the padded width so writes and reads stay aligned.
- **TP sharding of the cache.** MLA's latent is single-head, so there is no head axis to shard along. The pool replicates across the `tensor` axis rather than splitting it. Doing proper DP attention (shard by requests, all-reduce across TP) is a follow-up — without it, the MLA path runs replicated on all TP ranks, which still works but wastes HBM on larger TP degrees.

## Appendix A: Weight glossary & absorbed-path derivation

Dimension names used throughout: `hidden = 7168`, `n_h = num_heads = 128`, `q_lora = 1536`, `kv_lora = 512`, `D_k = qk_nope_head_dim = 128`, `r_dim = qk_rope_head_dim = 64`, `D_v = v_head_dim = 128`.

### A.1 Weight glossary

Two categories of weights appear in this RFC: **base weights** (loaded from the DeepSeek-V3 checkpoint as-is) and **absorb factors** (per-head reshapes of those base weights, used in the absorbed forward path). No new learned parameters are introduced.

**Base weights — checkpoint-native:**

| Name | Shape | Meaning |
|---|---|---|
| `q_a_proj`           | `[hidden, q_lora]`                      | Down-projects hidden state to the Q latent (single-head LoRA-style). |
| `q_a_layernorm`      | `[q_lora]` (RMSNorm scale)              | RMSNorm applied to the Q latent. |
| `q_b_proj`           | `[q_lora, n_h · (D_k + r_dim)]`         | Up-projects the Q latent to per-head Q. Each head block concatenates the `D_k`-wide nope portion and the `r_dim`-wide rope-input portion. In the absorbed path it is split into `W_UQ_nope` and `W_UQ_rope` (see below) — the monolithic matmul is never invoked. |
| `kv_a_proj_with_mqa` | `[hidden, kv_lora + r_dim]`             | Single projection producing the shared KV latent (`kv_lora`-dim) and the rope input for K (`r_dim`-wide, shared across all heads). Also referred to as `kv_a_proj` in §3.9's pseudocode. |
| `kv_a_layernorm`     | `[kv_lora]` (RMSNorm scale)             | RMSNorm applied to the KV latent `c_kv`. |
| `kv_b_proj`          | `[kv_lora, n_h · (D_k + D_v)]`          | Up-projects the KV latent to per-head `(K_nope ‖ V)`. **Deleted after load** in the absorbed path — its two slices become `W_UK` and `W_UV`. |
| `o_proj`             | `[n_h · D_v, hidden]`                   | Output projection. **Reshaped** into `W_O` and kept as the second O-side factor. |

**Absorb factors — derived at load time from the base weights:**

| Name | Shape | Derived from | Role in forward pass |
|---|---|---|---|
| `W_UQ_nope` | `[n_h, q_lora, D_k]`   | `q_b_proj` — per-head nope slice (`reshape(q_lora, n_h, D_k + r_dim)[:, :, :D_k].transpose(1, 0, 2)`) | First Q-side factor. `c_q · W_UQ_nope → q_nope` — lifts Q latent into per-head `D_k` space (the bottleneck dim). |
| `W_UQ_rope` | `[n_h, q_lora, r_dim]` | `q_b_proj` — per-head rope slice (same reshape, slice `[:, :, D_k:]`) | Produces the rope-input half of Q (`q_pe_raw`) which then goes through RoPE. Called `q_b_rope_proj` in §3.9's pseudocode — same thing. |
| `W_UK`      | `[n_h, D_k, kv_lora]`  | `kv_b_proj` — per-head K_nope slice with the last two axes transposed (`reshape(kv_lora, n_h, D_k + D_v)[:, :, :D_k].transpose(1, 2, 0)`) | Second Q-side factor. `q_nope · W_UK → ql_nope` — projects Q from `D_k` space into `kv_lora` latent space so the kernel's nope score can dot directly against cached `c_kv`. The `transpose(1, 2, 0)` step puts the `D_k` axis in the contraction position for the forward einsum `'thd,hdr->thr'`. |
| `W_UV`      | `[n_h, kv_lora, D_v]`  | `kv_b_proj` — per-head V slice (`reshape(kv_lora, n_h, D_k + D_v)[:, :, D_k:].transpose(1, 0, 2)`) | First O-side factor. `o_latent · W_UV → o_v` — decompresses the kernel's latent output back into per-head `D_v` space. |
| `W_O`       | `[n_h, D_v, hidden]`   | `o_proj.reshape(n_h, D_v, hidden)`    | Second O-side factor. `o_v · W_O → out`. The contraction over `n_h · D_v` is where the TP all-reduce happens (§3.8). |

**Hypothetical weight that we deliberately do NOT build.** `W_OV` is mentioned in §3.9 as the full pre-fused product `W_UV · W_O` of shape `[n_h, kv_lora, hidden]`. Building it would skip the `D_v` bottleneck and cost ~3× FLOPs (§3.3), so we keep the factors separate.

**Reference naming in other repos.**
- sglang-GPU: `w_kc` = `W_UK`, `w_vc` = `W_UV`.
- tpu-inference: `W_UK_T` = `W_UK` with the last two axes transposed (same tensor content, just the layout its einsum expects).

### A.2 Standard (non-absorbed) forward

For context, the naive MLA forward (what DeepSeek-V3 does literally if you run `q_b_proj` and `kv_b_proj` as monolithic matmuls — this is NOT what we implement):

```text
c_q                 = RMSNorm(x · q_a_proj)                          [T, q_lora]
q_nope, q_pe_raw    = split_last(c_q · q_b_proj, [D_k, r_dim])       [T, n_h, D_k], [T, n_h, r_dim]
q_pe                = RoPE(q_pe_raw)                                 [T, n_h, r_dim]

c_kv, k_pe_raw      = split_last(x · kv_a_proj_with_mqa, [kv_lora, r_dim])
c_kv                = RMSNorm(c_kv)                                  [T, kv_lora]
k_pe                = RoPE(k_pe_raw)                                 [T, r_dim]   (shared across heads)
k_nope, v           = split_last(c_kv · kv_b_proj, [D_k, D_v])       [T, n_h, D_k], [T, n_h, D_v]

K  = concat_last(k_nope, broadcast(k_pe))                            [T, n_h, D_k + r_dim]
Q  = concat_last(q_nope, q_pe)                                       [T, n_h, D_k + r_dim]
o  = softmax(Q Kᵀ / √(D_k + r_dim)) · V                              [T, n_h, D_v]
out = o.reshape(T, n_h · D_v) · o_proj                               [T, hidden]
```

Per-token cached K/V here is `n_h · (D_k + D_v)` elements (~33 KB at bf16).

### A.3 Absorbing the "up" projections

The per-head nope score factors through the KV latent. Writing the K_nope production directly in terms of `W_UK` (shape `[n_h, D_k, kv_lora]`, contraction over `D_k` lands in the `kv_lora` latent):

```text
k_nope[s, h]        = c_kv[s] · W_UKᵀ[h]                             (W_UKᵀ has shape [kv_lora, D_k])
score_nope[t, s, h] = q_nope[t, h] · k_nope[s, h]
                    = (c_q[t] · W_UQ_nope[h]) · (c_kv[s] · W_UKᵀ[h])
                    = (c_q[t] · W_UQ_nope[h] · W_UKᵀ[h]) · c_kv[s]ᵀ
                    =: ql_nope[t, h] · c_kv[s]ᵀ                      where ql_nope ∈ [kv_lora]
```

The V-branch output factors the same way:

```text
o[t, h] = Σ_s softmax_s · v[s, h]
       = Σ_s softmax_s · (c_kv[s] · W_UV[h])
       = (Σ_s softmax_s · c_kv[s]) · W_UV[h]
       =: o_latent[t, h] · W_UV[h]
```

The attention kernel therefore only needs `ql_nope`, `q_pe`, `c_kv`, and `k_pe` — `k_nope` and `v` are never materialized. Per-token cache drops to `kv_lora + r_dim` = 576 elements (~1.1 KB at bf16), the ~57× reduction cited in §3.1.

The concrete derivation of each absorb factor from its parent base weight (slices, transposes, and shapes) is in the "Absorb factors" table in §A.1.
