# [RFC] Lightning Linear Attention Module for Ling-2.5-1T

## Background

Ling-2.5-1T is a ~1T parameter sparse MoE model with a hybrid attention architecture:

> - **10 MLA (Multi-Latent Attention) layers** — DeepSeek-V2 style compressed KV
> - **70 Lightning Linear Attention (Simple GLA) layers** — constant-size recurrent state
> - `layer_group_size=8`, 1:7 MLA:Linear ratio

Linear Attention replaces the standard KV Cache with a fixed-size recurrent state, avoiding memory growth with sequence length. The core computation uses two kernels from the `tops` library: `simple_gla_fwd` (dispatching to `chunk_simple_gla_fwd_varlen`) for prefill and `fused_recurrent_simple_gla` for decode.

---

## Goal

Implement `BailingMoeV2_5LinearAttention` as a Flax NNX module, supporting prefill and decode modes, using a fixed-size recurrent state in place of KV Cache.

---

## Implementation Spec

| Parameter | Value | Source |
|-----------|-------|--------|
| hidden_size | 8192 | model config |
| num_attention_heads (H) | 64 | model config |
| head_dim (K=V) | 128 | model config |
| rope_dim (= head_dim × partial_rotary_factor) | 64 | derived from partial_rotary_factor=0.5 |
| use_qk_norm | true | model config |
| group_norm_size | 8 | model config |
| chunk_size | 64 | kernel default |
| rms_norm_eps | 1e-6 | model config |
| use_qkv_bias | false | model config |
| use_bias (dense) | false | model config |
| rope_theta | 6,000,000 | model config |
| max_position_embeddings | 131072 | model config |

---

## Design

### Forward Pre-processing

```
[At model load time]
Ling model __init__
    Creates LinearAttentionBackend; all LinearAttention layers share a single instance
model_runner.load_model()
    Reads via getattr and stores as model_runner.linear_attn_backend
    # Non-LinearAttention models: None, zero impact

[Before each forward, outside JIT]
tp_worker.py forward_batch_generation()
    ↓ metadata = model_runner.linear_attn_backend.get_forward_metadata(batch)
        Returns LinearAttentionMetadata pytree:
          .cu_seqlens_dev   # [N_padded+1], chunk-aligned boundaries per request
          .scatter_idx  # [T], tight-packed → chunk-aligned position mapping
        backend.T_packed_bucket  # Σchunk-aligned lengths aligned to token_paddings, static
    ↓ forward_batch.linear_attn_metadata = metadata
    ↓ model_runner.forward(forward_batch)                     ← JIT boundary
        model_def contains T_packed_bucket (static, triggers recompile on change)
        forward_batch contains linear_attn_metadata (pytree, dynamic, traced through ForwardBatch.tree_flatten)
        ↓ BailingMoeV2_5LinearAttention.__call__(...)         ← see Forward Flow
```

### Forward Flow

```
hidden_states [T, 8192]
    # T padded to {64, 128, ..., 8192} by the scheduler (static JAX JIT shape)
    ↓ QKV projection (fused)
[T, 3*H*head_dim] = [T, 24576]
    ↓ split + reshape
q, k, v each [T, H, head_dim] = [T, 64, 128]
    ↓ RMSNorm on Q and K (per-head, 128-dim); V unchanged
    ↓ Partial RoPE (NeoX style): apply position encoding to first rope_dim(64) dims; last 64 dims unchanged
      pass is_neox_style=True to RotaryEmbedding
    # q, k, v shape: [T, H, K], tight-packed (tokens from all requests concatenated;
    #   T = outer padding bucket, no per-seq chunk-aligned padding yet)
    ↓ Cast recurrent_state to float32 for numerical stability
    ↓ select kernel based on forward_batch.forward_mode:
      if forward_batch.forward_mode.is_decode():
        # decode: each request processes 1 token (token_paddings == bs_paddings)
        # B=T, each slot is an independent request; state isolation is natural
        # Reshard recurrent_state along H so scan carry matches q/k/v sharding
        recurrent_state = jax.sharding.reshard(
            recurrent_state, NamedSharding(mesh, P(None, "tensor", None, None)))
        q, k, v → reshape → [T, 1, H, K]
        output, new_state = fused_recurrent_simple_gla(
            q, k, v,
            g_gamma=slopes,                   # [H] fixed decay, different per layer
            initial_state=recurrent_state,    # [T, H, K, V]
            output_final_state=True,
            scale=None,  # tops defaults to K^-0.5 when scale=None
        )
      elif forward_batch.forward_mode == ForwardMode.EXTEND:
        # MIXED is converted to EXTEND by the scheduler before reaching this point
        # q, k, v are tight-packed [T, H, K]; scatter to chunk-aligned layout for the chunk kernel
        # LinearAttentionBackend has pre-computed boundaries outside JIT
        T_pb        = self.backend.T_packed_bucket        # chunk-aligned packed buffer length (static shape)
        cu_seqlens  = forward_batch.linear_attn_metadata.cu_seqlens_dev   # [N_padded+1], chunk-aligned boundaries per request
        scatter_idx = forward_batch.linear_attn_metadata.scatter_idx  # [T], tight-packed → chunk-aligned position mapping
        # Pallas/Mosaic kernel cannot be partitioned by GSPMD (custom_call is opaque to the XLA partitioner);
        # use shard_map for explicit partitioning: each device runs scatter + simple_gla_fwd on its local H shard.
        # cu_seqlens is passed as P() (replicated) so each device has complete boundary information.
        # Reshard slope and recurrent_state onto the mesh before shard_map.
        slope_sm = jax.sharding.reshard(slopes, NamedSharding(mesh, P("tensor")))
        h0_sm    = jax.sharding.reshard(recurrent_state, NamedSharding(mesh, P(None, "tensor", None, None)))
        def _prefill_fn(q_local, k_local, v_local, gamma, h0, scatter_idx_p, cu_seqlens_p):
            q_p = scatter_to_packed(q_local, scatter_idx_p, T_pb)   # [1, T_pb, H_local, K]
            k_p = scatter_to_packed(k_local, scatter_idx_p, T_pb)
            v_p = scatter_to_packed(v_local, scatter_idx_p, T_pb)
            return simple_gla_fwd(
                q_p, k_p, v_p,
                g_gamma=gamma,          # [H_local]
                h0=h0,                  # [N_padded, H_local, K, V]
                cu_seqlens_dev=cu_seqlens_p,
                scale=None, use_ht=True, chunk_size=64,
            )
        output_packed, new_state = shard_map(
            _prefill_fn, mesh=self.mesh,
            in_specs=(
                P(None, "tensor", None),        # q
                P(None, "tensor", None),        # k
                P(None, "tensor", None),        # v
                P("tensor"),                    # slopes [H_local]
                P(None, "tensor", None, None),  # h0
                P(),                            # scatter_idx (replicated)
                P(),                            # cu_seqlens (replicated)
            ),
            out_specs=(
                P(None, None, "tensor", None),  # output_packed [1, T_pb, H_local, V]
                P(None, "tensor", None, None),  # new_state     [N, H_local, K, V]
            ),
            check_vma=False,
        )(q, k, v, slope_sm, h0_sm, scatter_idx, cu_seqlens)
        # output_packed [1, T_pb, H, V]; gather back to [T, H, V]
        output = gather_from_packed(output_packed, scatter_idx)  # [T, H, V]
        # new_state [N_padded, H, K, V] (trailing padding slots have zero state)
      else:
        raise NotImplementedError(forward_batch.forward_mode)
    ↓ reshape → [T, H*head_dim] = [T, 8192]
    ↓ GroupRMSNorm(output) * sigmoid(g_proj(hidden_states))
    ↓ dense projection: Linear(H*head_dim → hidden_size)
returns (output [T, 8192], new_state [N_padded, H, K, V] for prefill / [T, H, K, V] for decode)
```

### Key Design Notes

**ALiBi slopes as decay**
`g_gamma` (shape `[H]`) = `-build_slope_tensor(H) * (1 - (layer_idx-1)/(num_hidden_layers-1) + 1e-5)`, layer_idx 0-indexed. Use the HF reference implementation as ground truth:

```python
slope = -BailingMoeV2_5LinearAttention.build_slope_tensor(self.num_heads) * (
    1 - (self.layer_idx - 1) / (self.config.num_hidden_layers - 1) + 1e-5
)
```

**Tensor Parallelism**

The two execution paths use different sharding strategies because Pallas/Mosaic kernels are opaque to GSPMD (compiled as `custom_call` nodes that the XLA partitioner cannot analyze — sharded inputs trigger implicit all-gather rather than kernel partitioning).

- **Decode** (`fused_recurrent_simple_gla`, pure JAX `lax.scan`): GSPMD automatically propagates H-dimension sharding; `g_gamma=self.slope` (shape `[H]`) is sharded along with q's sharding. `recurrent_state` is explicitly resharded to `P(None, "tensor", None, None)` before the kernel call to ensure the scan carry matches q/k/v H-dim sharding. TP=1 and TP>1 share the same code path.
- **Prefill** (`simple_gla_fwd` → Pallas kernel): Uses `shard_map` for explicit partitioning. `self.slope` is resharded to `P("tensor")` and `recurrent_state` to `P(None, "tensor", None, None)` before being passed into `shard_map`; scatter and kernel call run independently per device on the local H shard — no all-gather. `cu_seqlens` is passed as `P()` (replicated) so each device has complete boundary information.

This pattern (GSPMD for pure-JAX kernels, `shard_map` for Pallas kernels) is consistent with how other Pallas kernels are handled in the project (see `flashattention_backend.py`). TP consistency verified on CPU and TPU v6e-4 (TP=2/4, H=64, prefill+decode): output `max abs diff < 6e-1` (bf16 row-parallel dense all-reduce addition order differs from TP=1, producing up to ~0.5 max diff at bf16 precision); state `max abs diff < 5e-2` (state is not affected by dense all-reduce, each head is independent across TP shards).

**Recurrent State**
- Prefill returns `[N_padded, H, K, V]` (N_padded = padded batch size); Decode returns `[T, H, K, V]` (T = padded batch size); shape does not grow with sequence length
- Passed as an explicit parameter; storage and lifecycle managed externally by the model runner
- Prefill uses `simple_gla_fwd` (dispatching to `chunk_simple_gla_fwd_varlen`); decode uses `fused_recurrent_simple_gla` (no chunk alignment constraint, avoids excess decay accumulation)
- **TP sharding**: recurrent_state is expected to be H-dim sharded, consistent with q's sharding; this module does not enforce sharding — the caller is responsible for ensuring correct sharding on gather

**LinearAttentionBackend**
- A standalone `LinearAttentionBackend(nnx.Module)` handles prefill metadata pre-computation, parallel to `FlashAttentionBackend`; all LinearAttention layers share a single backend instance
- `get_forward_metadata(batch)` is called before the JIT boundary in `tp_worker.py forward_batch_generation()`; dispatches by forward_mode:
  - **DECODE**: returns `LinearAttentionMetadata()` with `cu_seqlens_dev=None, scatter_idx=None` (scatter/gather metadata is only needed for prefill)
  - **EXTEND**: uses numpy `batch.extend_seq_lens` to compute chunk-aligned lengths; returns `LinearAttentionMetadata(cu_seqlens_dev=..., scatter_idx=...)`
- The returned `LinearAttentionMetadata` is a pytree (registered via `@register_pytree_node_class`) and is stored on `forward_batch.linear_attn_metadata`, flowing through `ForwardBatch.tree_flatten` into JIT as traced values — matching the `FlashAttentionMetadata` pattern
- `T_packed_bucket` (Python int) stored as a plain attribute on the backend, enters NNX graphdef (static); changes trigger recompilation
- `cu_seqlens_dev` shape uses padded batch size (`len(batch.seq_lens)`, aligned to `bs_paddings`); trailing padding slots correspond to zero-length sequences and are skipped by the kernel; the kernel guarantees zero state output for these trailing slots — no masking required on write-back

**scatter_to_packed / gather_from_packed**
- `get_forward_metadata` (outside JIT, numpy) pre-computes `scatter_idx: [T]` (T = outer padding bucket, aligned with q's first dimension): iterates over N_real requests, mapping each real token to its target position in the chunk-aligned buffer; trailing outer-padding slots are mapped to position `T_pb` (a dummy slot — one extra slot allocated beyond T_pb, so writes from padding tokens never corrupt real data). `scatter_idx` is returned as part of `LinearAttentionMetadata` pytree, updated alongside `cu_seqlens_dev`
- scatter (inside JIT): `jnp.zeros([1, T_pb+1, H, K]).at[0, scatter_idx].set(q)[:, :T_pb]`
- gather (inside JIT): `jnp.pad(output_packed, ((0,0),(0,1),(0,0),(0,0)))[0, scatter_idx]`, returning `[T, H, V]` directly (padding slots read from the dummy zero column and do not affect subsequent computation); traced indices are compiled by XLA to scatter/gather ops — no Python loops

**Prefill: Multi-Request Handling**
All requests' q/k/v are scattered into a chunk-aligned `[1, T_packed_bucket, H, K]` buffer and processed in a single `simple_gla_fwd` call. `cu_seqlens_dev` is passed as sequence boundaries (dispatching to `chunk_simple_gla_fwd_varlen`), which resets state at boundaries — natural per-request state isolation. This avoids per-request Python loops and repeated kernel launches.

**Intra-chunk padding and state accuracy**
After scatter, positions in the chunk-aligned buffer that correspond to intra-chunk padding are filled with zeros (from `jnp.zeros` initialization). The kernel processes these as `h_t = decay * h_{t-1}`, causing `(chunk_size - seq_len % chunk_size) % chunk_size` extra decay steps. Worst case: `seq_len=1` padded to 64, state multiplied by `decay^63`.

Empirical measurements (g_gamma values: -0.05, -0.10, -0.15, -0.20; h0 scale 0.1; float32):

| Scenario | Extra positions | Head (g_gamma) | decay^N | State max diff |
|----------|---------------:|----------------|--------:|---------------:|
| T=100→128 | 28 | Head 2 (-0.05) | 0.2466 | 1.10e+01 |
| T=100→128 | 28 | Head 0 (-0.10) | 0.0608 | 1.03e+01 |
| T=100→128 | 28 | Head 3 (-0.15) | 0.0150 | 9.33e+00 |
| T=100→128 | 28 | Head 1 (-0.20) | 0.0037 | 8.10e+00 |
| T=1→64   | 63 | Head 2 (-0.05) | 0.0429 | 5.48e+00 |
| T=1→64   | 63 | Head 0 (-0.10) | 0.0018 | 1.02e+01 |
| T=1→64   | 63 | Head 3 (-0.15) | 0.0001 | 7.44e+00 |
| T=1→64   | 63 | Head 1 (-0.20) | 0.0000 | 5.40e+00 |

Accepted rationale: intra-chunk padding only affects the prefill path; `seq_len=1` prefill is rare in practice; the decode path fully avoids this via `fused_recurrent_simple_gla`.

**Decode: Multi-Request Handling**
q/k/v reshaped to `[T, 1, H, K]`; each B slot is an independent request, state isolation is natural; all requests processed in a single kernel call.

**Padding levels**
- Total T is padded by the scheduler to one of `{64, 128, ..., 8192}` (static JAX JIT shape); used directly by this module
- Prefill: `LinearAttentionBackend.get_forward_metadata` computes per-request chunk-aligned lengths and `cu_seqlens_dev` outside JIT; scatter/gather executed inside JIT as traced ops. Decode: no additional padding needed

### Implementation Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Decode kernel | `fused_recurrent_simple_gla` | Decode has 1 token per step; chunk+padding would pad to 64, multiplying state by `decay^64` instead of `decay^1` |
| Prefill multi-request strategy | scatter → single kernel call (cu_seqlens_dev) | All requests in one `simple_gla_fwd` call; cu_seqlens_dev resets state at boundaries in-kernel; avoids Python loops and repeated kernel launch overhead |
| cu_seqlens_dev construction | numpy pre-compute outside JIT, returned in `LinearAttentionMetadata` pytree | Consistent with `FlashAttentionMetadata` pattern; metadata flows through `ForwardBatch.tree_flatten` into JIT as traced values; cu_seqlens_dev shape fixed to padded batch size to prevent recompilation |
| scatter_idx construction | numpy pre-compute outside JIT, returned in `LinearAttentionMetadata` pytree | Shape `[T]` is static (no recompilation); `at[].set()` / advanced indexing inside JIT compiled to XLA scatter/gather, no Python loops |
| Slopes storage | Computed in `__init__`, stored as attribute | JAX JIT treats Python attributes as constants, equivalent to PyTorch `register_buffer` |
| Prefill TP partitioning | `shard_map` explicit partitioning | Pallas kernel compiles to `custom_call`; GSPMD cannot analyze its internals and inserts all-gather for sharded inputs. `shard_map` lets each device run scatter + kernel on its local H shard — no communication overhead; consistent with FlashAttention and other Pallas kernels in the project |
| GroupRMSNorm integration | Direct integration (`layers/attention/fla/group_rmsnorm.py`) | GroupRMSNorm is already available; no stub needed |

---

## Interface

```python
class LinearAttentionBackend(nnx.Module):
    def __init__(self, mesh=None): ...

    def get_forward_metadata(self, batch: ModelWorkerBatch) -> LinearAttentionMetadata:
        # Called before the JIT boundary in tp_worker.py forward_batch_generation()
        # Returns LinearAttentionMetadata pytree with cu_seqlens_dev and scatter_idx
        # Also updates self.T_packed_bucket (int, static) as a side effect
        ...

class BailingMoeV2_5LinearAttention(nnx.Module):
    def __init__(self, config, layer_idx, mesh, backend: LinearAttentionBackend,
                 dtype: jnp.dtype = jnp.bfloat16): ...

    def __call__(
        self,
        positions: jax.Array,              # [T]
        hidden_states: jax.Array,          # [T, hidden_size]
        forward_batch: ForwardBatch,       # contains forward_mode for prefill/decode dispatch
        recurrent_state: jax.Array,            # prefill: [N_padded, H, K, V], N_padded = padded batch size
                                            # decode:  [T, H, K, V], T = padded token count = padded batch size (1 token per request)
                                            # zeros array on first call, provided by caller; never None
    ) -> tuple[jax.Array, jax.Array]:
        # returns: (output [T, hidden_size], new_state [N_padded, H, K, V] for prefill / [T, H, K, V] for decode)
        ...
```

---

## Testing

### Black-box Tests

| Test | Method |
|------|--------|
| Output shape | `output.shape == [tokens, 8192]` |
| new_state shape (prefill) | `new_state.shape == [N_padded, H, K, V]`, i.e. `[N_padded, 64, 128, 128]` |
| new_state shape (decode) | `new_state.shape == [T, H, K, V]`, i.e. `[T, 64, 128, 128]` |
| State updates on decode | new_state differs between two consecutive calls |
| State propagation | Run one decode step to get new_state; run a second step with the same q/k/v using recurrent_state=zeros vs recurrent_state=new_state; outputs differ |
| First prefill (zeros state) | recurrent_state is all-zeros [N_padded, H, K, V]; no error, correct output shape |
| Prefill runs correctly | seq_len=64/128/512 (T must be a multiple of chunk_size=64, guaranteed by scheduler), no error, correct output shape |
| Decode runs correctly | Single-step decode (N=1), no error, correct output shape |
| Non-chunk-aligned prefill followed by N decode steps (integration) | seq_len not a multiple of chunk_size; after prefill, run N decode steps; state transfers correctly, each decode output differs, new_state updates each step |

### White-box Tests

| Test | Method |
|------|--------|
| QKV projection shape | `q.shape == [tokens, 64, 128]` |
| V not normalized | V values unchanged before and after norm |
| RoPE applied to first 64 dims only | Last 64 dims unchanged before and after RoPE |
| Gating values | gate values in [0, 1] |
| Output shape unchanged by gating | `[tokens, 8192]` preserved |
| Dense projection applied | Output differs before and after dense |
| ALiBi slopes correctness | All slopes negative; magnitude decreases with layer_idx; values match formula in Design section |
| g_gamma path correctness | Feeding g_gamma=[H] and equivalent expanded g produce identical output and new_state; prefill: g=[1, T_pb, H]; decode: g=[T, 1, H] (T=decode batch, 1=single step) |
| GLA wrapper correctness (prefill) | Scatter q, k, v with same scatter_idx to obtain packed tensors; directly call `simple_gla_fwd` with same cu_seqlens_dev; output and new_state match module's internal call |
| GLA wrapper correctness (decode) | Directly calling `fused_recurrent_simple_gla` with same q, k, v, g_gamma, h0 matches module's internal output and new_state |
| Decode state isolation | Two requests decoded individually produce same output and new_state as batched (reshape to [T,1,H,K]) decode |
| Prefill state isolation | Two requests prefilled individually produce same output and new_state as combined (scatter → single `simple_gla_fwd`, cu_seqlens_dev boundary) prefill |

### Multi-Card Tests (TP=2)

| Test | Method |
|------|--------|
| Output shape under TP=2 | Construct 2-device mesh; `output.shape == [tokens, 8192]` |
| TP=2 matches single-card | Same input: output `max abs diff < 6e-1` (bf16 row-parallel dense all-reduce reordering); state `max abs diff < 5e-2` (state does not pass through dense all-reduce, heads are independent across TP shards) |

### Cross-Framework Consistency

Compare JAX implementation against HuggingFace PyTorch `BailingMoeV2_5LinearAttention` with a fixed random seed:

| Check | Requirement |
|-------|-------------|
| Output shape matches | `output.shape == torch_output.shape` |
| float32 numerical alignment | Both sides run with `dtype=float32`; `max abs diff < 1e-5` |
| bfloat16 numerical alignment | Both sides run with `dtype=bfloat16`; `max abs diff < 0.05` (bfloat16 precision ~1e-2, includes XLA/PyTorch operation ordering differences) |
| new_state numerical alignment | Same, verified per dtype |

> Tolerance tiers by dtype: float32 uses `1e-5`, bfloat16 uses `0.05`. Differences arise from floating-point operation ordering between XLA and PyTorch, plus dtype precision limits.

---

## Work Breakdown

- [ ] Implement `LinearAttentionBackend` (`linear_attention_backend.py`): `get_forward_metadata` computes `T_packed_bucket` and returns `LinearAttentionMetadata` with `cu_seqlens_dev` and `scatter_idx`
- [ ] `model_runner.py`: add `self.linear_attn_backend = getattr(self.model, "linear_attn_backend", None)` at the end of `load_model()`
- [ ] `tp_worker.py`: in `forward_batch_generation`, call `linear_attn_backend.get_forward_metadata(batch)` and store result in `forward_batch.linear_attn_metadata`
- [ ] Implement `__init__`:
  - QKV proj (`scope_name="query_key_value"`), g_proj (`scope_name="g_proj"`): column-parallel, `kernel_axes=(None, "tensor")`
  - dense (`scope_name="dense"`): row-parallel, `kernel_axes=("tensor", None)`
  - Q/K RMSNorm (`scope_name="query_layernorm"`/`"key_layernorm"`, note `param_dtype=dtype`)
  - RotaryEmbedding, ALiBi slopes (stored as `self.slope`)
  - g_norm: `GroupRMSNorm(hidden_size=H*head_dim, num_groups=group_norm_size, epsilon=rms_norm_eps, scope_name="g_norm")` from `layers/attention/fla/group_rmsnorm.py`
- [ ] Implement forward: QKV projection → split + reshape → Q/K norm → Partial RoPE → kernel dispatch (decode/prefill branches, prefill includes scatter/gather) → gating → dense → return state
- [ ] Integrate `GroupRMSNorm` (`layers/attention/fla/group_rmsnorm.py`)
- [ ] Write unit tests and integration tests

---

## Dependencies

| Dependency | Status |
|------------|--------|
| `simple_gla_fwd` / `chunk_simple_gla_fwd_varlen` cu_seqlens_dev support (tops library, pallas-kernel feat/varlen branch) | **In progress** (parameter signature exists; Pallas TPU kernel internal support expected soon) |
| `fused_recurrent_simple_gla` (tops library, for decode correctness) | **Ready** ([pallas-kernel PR #92](https://github.com/primatrix/pallas-kernel/pull/92), merged 2026-03-30) |
| `GroupRMSNorm` layer | **Ready** (`python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py`) |
| DecoderLayer-level dispatch | Downstream task |
| Model runner recurrent state management | **Blocking dependency** — current model runner only supports KV Cache; needs extension to support per-request recurrent state storage and retrieval |

---

## References

| Resource | Link |
|----------|------|
| HuggingFace model | https://huggingface.co/inclusionAI/Ling-2.5-1T |
| PyTorch reference implementation | `BailingMoeV2_5LinearAttention` in `modeling_bailing_moe_v2_5.py` (HF model repo) |
| Existing JAX reference | `python/sgl_jax/srt/models/bailing_moe.py` (MHA layer; LinearBase / RMSNorm / RotaryEmbedding usage) |
| Kernel implementation | `tops/ops/simple_gla/__init__.py` (`simple_gla_fwd`, dispatch entry), `tops/ops/simple_gla/chunk.py` (`chunk_simple_gla_fwd_varlen`, prefill varlen), `tops/ops/simple_gla/fused_recurrent.py` (`fused_recurrent_simple_gla`, decode) ([pallas-kernel](https://github.com/primatrix/pallas-kernel)) |
