# Issue #948 Decomposition: KDA Inference Components

| Field | Value |
|-------|-------|
| **Parent issue** | [#948 — Support KDA (Kimi Delta Attention) for Kimi-Linear](https://github.com/sgl-project/sglang-jax/issues/948) |
| **Design doc** | [kda_inference_components.md](https://github.com/primatrix/wiki/blob/main/docs/projects/sglang-jax/design_docs/kda_inference_components.md) |
| **Date** | 2026-04-25 |

---

## Overview

Issue #948 delivers KDA inference components: kernel, backend, dispatch entry, layer, and numerical alignment tests. This document defines how to decompose it into parallelizable sub-issues.

### Team

- **@MokusMokun** — owner of all sub-issues (dump, kernel interface, backend, layer, tests)
- **`@pathfinder-pf`** — fills kernel implementation bodies (Sub-1 kernel code via PR #964); @MokusMokun merges their code

### Key constraints

- **RFC-0015 not landed yet**: `RecurrentStatePool` / `MemoryPools` / `HybridReqToTokenPool` are unavailable. Backend must use interface stubs for state pool consumption.
- **No naive prefill**: `chunk_kda_jax_naive` is not in scope. v1 ships pallas prefill + naive decode.

### Deviation from design doc

- **Base class file location**: Design doc §3.2/§3.3 places `LinearRecurrentAttnBackend` + `LinearRecurrentAttnBackendMetadata` in `kda_backend.py`. This decomposition keeps them in `hybrid_linear_attn_backend.py` instead, because: (1) they are shared base classes for KDA / GDN / Mamba2, not KDA-specific; (2) PR #961 already placed stubs there; (3) `kda_backend.py` already imports from `hybrid_linear_attn_backend.py`. The design doc should be updated to reflect this.
- **Kernel dispatcher fallback**: Design doc §4.2 shows symmetric fallback for both prefill and decode (`chunk_kda_pallas if _HAS_PALLAS else chunk_kda_jax_naive`, same for decode). This decomposition deviates: **prefill = pallas only** (no naive prefill impl exists — `chunk_kda_jax_naive` is out of scope), **decode = naive only** (no pallas decode impl exists). The dispatcher in `__init__.py` directly re-exports `chunk_kda_fwd` from `kda.py` and `naive_recurrent_kda` from `naive.py` without try/except fallback.

### Development workflow

```
epic/support_kimi_linear  (upstream, PR #961/#962 merged)
  │
  └─ feat/kda-inference-components  (主分支, branched from epic)
       │
       ├─ worktree: sub0-dump-upgrade        ──→ merge --no-ff → feat/kda-inference-components
       ├─ worktree: sub1-kernel-dispatcher   ──→ merge --no-ff → feat/kda-inference-components
       ├─ worktree: sub2-backend-dispatch    ──→ merge --no-ff → feat/kda-inference-components
       └─ worktree: sub3-layer-tests         ──→ merge --no-ff → feat/kda-inference-components
```

- **主分支**: `feat/kda-inference-components`，从 `epic/support_kimi_linear` 拉出
- **并行开发**: 每个 sub-issue 一个 worktree + 分支，按工作内容命名
- **合入**: worktree 完成后 `merge --no-ff` 进主分支（保留 merge commit，方便回溯）
- **Upstream 同步**: 只有 `feat/kda-inference-components` 跟 `epic/support_kimi_linear` rebase

---

## Sub-issues

### Sub-0: GPU Dump Upgrade (HF real config + weights)

**Owner**: @MokusMokun
**Location**: `kda_gpu/` (existing package at `_Delivery/202604B_sgl_jax_KDA_ling3/2026-04-23/kda_gpu/`)

**Current state**: 12 cases exist with small synthetic config (`hidden=128, heads=4, head_dim=32`, random-init weights). Scripts (`run_kda_gpu.py`, `fixed_kda_module.py`) and dump schema are solid.

**What needs to change**:

| Gap | Current | Required (design doc §6.1) |
|-----|---------|---------------------------|
| Config | `hidden=128, heads=4, head_dim=32` | `hidden=4096, heads=32, head_dim=128` (HF checkpoint actual config) |
| Weights | Random init | Extracted from HF safetensors (`model.layers.{idx}.self_attn.*`) |

**What stays**: Existing small-config dumps are preserved as-is for fast local debugging. New real-config dumps are added alongside (e.g. `dumps_real/` or separate `case_real_*` prefix).

**Deliverables**:
1. Updated `run_kda_gpu.py` supporting both configs (small synthetic for debug, real HF for alignment)
2. `dumps/weights.npz` with real HF weights for one KDA layer
3. 12 `case_*.npz` files regenerated with real config + weights
4. Re-run on H100, verify sanity (chunk vs fused_recurrent rtol)

**Dependencies**: None. Can start immediately.
**Blocks**: Sub-3 (tests consume these dumps as ground truth).

---

### Sub-1: KDA Kernel + Dispatcher

**Owner**: @MokusMokun (kernel impl by `@pathfinder-pf` via PR #964)
**Status**: Kernel implementation complete (PR #964). Dispatcher `__init__.py` still empty — needs re-export wiring.

**PR #964 delivers**:
- `kda.py` (1286 lines): `chunk_kda_fwd` — Pallas TPU chunked prefill kernel
- `naive.py` (109 lines): `naive_recurrent_kda` — pure JAX naive decode kernel
- `__init__.py`: empty (dispatcher not yet wired)

**Remaining work** (to be done when merging PR #964):

| File | Deliverable |
|------|-------------|
| `python/sgl_jax/srt/kernels/kda/__init__.py` | Wire dispatcher: re-export `chunk_kda_fwd` as `chunk_kda` and `naive_recurrent_kda` as `fused_recurrent_kda` (unified names consumed by `KDAAttnBackend`). **Prefill**: pallas only (no naive fallback). **Decode**: naive only (no pallas decode). |

**Actual kernel signatures** (from PR #964):

```python
# kda.py — prefill (pallas)
def chunk_kda_fwd(
    q: jax.Array,           # [1, T_packed, H, K]  (B=1 packed layout)
    k: jax.Array,           # [1, T_packed, H, K]
    v: jax.Array,           # [1, T_packed, H, V]
    g: jax.Array,           # [1, T_packed, H, K]
    beta: jax.Array,        # [1, T_packed, H]
    scale: float,
    initial_state: jax.Array,       # [N, H, K, V]
    output_final_state: bool,
    cu_seqlens: jax.Array,          # [N+1] int32
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: jax.Array | None = None,
    chunk_size: int = 64,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: None = None,
    transpose_state_layout: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    ...

# naive.py — decode (naive JAX)
def naive_recurrent_kda(
    q: jax.Array,           # [B, T, H, K]  (T=1 for decode)
    k: jax.Array,           # [B, T, H, K]
    v: jax.Array,           # [B, T, H, V]
    g: jax.Array,           # [B, T, H, K]
    beta: jax.Array,        # [B, T, H]
    scale: float | None = None,
    initial_state: jax.Array | None = None,  # [B, H, K, V]
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    ...
```

**Dependencies**: None (PR #964 already open).
**Blocks**: Sub-2 (dispatcher needed by backend), Sub-3.

---

### Sub-2: Backend + Dispatch Infrastructure

**Owner**: @MokusMokun
**Files**:

| File | Deliverable |
|------|-------------|
| `python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py` | **Flesh out existing stubs** for `LinearRecurrentAttnBackend` + `LinearRecurrentAttnBackendMetadata` (base class, `@register_pytree_node_class`). These stay here — shared by KDA / future GDN / Mamba2. `HybridLinearAttnBackend` stub remains untouched (PR #961 scope, not this issue). |
| `python/sgl_jax/srt/layers/attention/linear/__init__.py` | Package init (currently missing). |
| `python/sgl_jax/srt/layers/attention/linear/kda_backend.py` | **KDA-specific only**: `KDAAttnBackend` + `KDAAttnBackendMetadata` (subclasses). Imports base from `hybrid_linear_attn_backend.py` (fixing the broken `from python.sgl_jax...` path to `from sgl_jax...`). |
| `python/sgl_jax/srt/layers/radix_linear_attention.py` | `RadixLinearAttention` nnx.Module — layer-to-backend dispatch entry. |

**File ownership rationale**: `LinearRecurrentAttnBackend` is the shared base for KDA, GDN, Mamba2. Naming it after any one subclass (`kda_backend.py`) would be misleading. It lives in `hybrid_linear_attn_backend.py` alongside `HybridLinearAttnBackend` which orchestrates them — this matches the existing PR #961 structure and import graph.

**Key implementation details**:

1. **Metadata pytree** (`LinearRecurrentAttnBackendMetadata` in `hybrid_linear_attn_backend.py`):
   - Fields: `cu_q_lens: jax.Array` (`[N+1]` int32), `recurrent_indices: jax.Array` (`[B]` int32, pool slot indices — passed in by external caller, not computed by backend)
   - `@register_pytree_node_class`; children = `(cu_q_lens, recurrent_indices)`, aux_data = `{}`

2. **Backend base** (`LinearRecurrentAttnBackend` in `hybrid_linear_attn_backend.py`):
   - Extends `AttentionBackend`
   - `get_forward_metadata(model_worker_batch, recurrent_indices)` — computes `cu_q_lens` from batch; `recurrent_indices` is passed in by external caller (`HybridLinearAttnBackend`, PR #961 scope). Returns `LinearRecurrentAttnBackendMetadata`.
   - Single `__call__` (same pattern as `FlashAttention`):
     ```python
     def __call__(
         self,
         mixed_qkv: jax.Array,
         a: jax.Array,           # forget gate
         b: jax.Array,           # beta
         layer: RadixLinearAttention,
         forward_batch: ForwardBatch,
         recurrent_state_pool,   # passed by RadixLinearAttention
         **kwargs,
     ):
     ```
     Internally reads `forward_batch.forward_mode` and dispatches:
     - `EXTEND` → `_dispatch_chunk(q, k, v, g, beta, initial_state, cu_seqlens)`
     - `DECODE` → `_dispatch_recurrent(q, k, v, g, beta, initial_state)`
     - `IDLE/DUMMY_FIRST` → zeros
     - `DRAFT_EXTEND/TARGET_VERIFY` → `NotImplementedError`

     State access: `recurrent_state_pool.get_linear_recurrent_layer_cache(layer.layer_id)` → single-layer view (conv + recurrent states). Slot indexing via `self.forward_metadata.recurrent_indices`.

     Write-back: after kernel dispatch, calls `recurrent_state_pool.set_linear_recurrent_layer_cache(layer.layer_id, recurrent_indices, new_recurrent, new_conv)` for in-place update. No functional return of new buffers — matches KV cache's Pallas in-place update pattern.
   - `@register_pytree_node_class`; children = `(forward_metadata,)`
   - **No `get_conv_state`/`set_conv_state` pass-through**: conv state is accessed directly via `recurrent_state_pool.get/set_linear_recurrent_layer_cache` inside `__call__`, not through separate backend methods.

3. **KDA subclass** (`KDAAttnBackend` in `kda_backend.py`):
   - Imports `LinearRecurrentAttnBackend` from `hybrid_linear_attn_backend.py`
   - `_dispatch_chunk` → `chunk_kda` (from `kernels/kda`)
   - `_dispatch_recurrent` → `fused_recurrent_kda` (from `kernels/kda`)

4. **RadixLinearAttention**:
   - `__init__`: `layer_id, num_q_heads, num_k_heads, num_v_heads, head_q_dim, head_k_dim, head_v_dim, conv_weights, bias, activation, A_log, dt_bias`
   - `__call__(self, forward_batch, mixed_qkv, a, b, recurrent_state_pool)`: calls `forward_batch.attn_backend(mixed_qkv, a, b, self, forward_batch, recurrent_state_pool=recurrent_state_pool)` — uses `__call__`, consistent with `RadixAttention` calling `forward_batch.attn_backend(q, k, v, self, forward_batch, ...)`
   - Receives `recurrent_state_pool` directly (not `memory_pool`). Upper model code (`KimiDecoderLayer`, follow-up issue) is responsible for extracting `recurrent_state_pool` from `memory_pool` before calling `KimiDeltaAttention`.

5. **ForwardBatch integration path**:

   ```
   Production (hybrid model, PR #961):
     forward_batch.attn_backend = HybridLinearAttnBackend

     jitted_run_model(forward_batch, memory_pool, ...)
       → model.forward(..., memory_pool)
         → KimiDecoderLayer: recurrent_state_pool = memory_pool.recurrent_state_pool
           → KimiDeltaAttention.__call__(..., recurrent_state_pool)
             → RadixLinearAttention.__call__(..., recurrent_state_pool)
               → forward_batch.attn_backend(mixed_qkv, a, b, self, forward_batch, recurrent_state_pool=...)
                 → HybridLinearAttnBackend routes by layer to KDAAttnBackend
                   → KDAAttnBackend.__call__(..., recurrent_state_pool)

   Sub-3 tests (KDA only, no hybrid):
     forward_batch.attn_backend = KDAAttnBackend (directly)
     Works because KDAAttnBackend.__call__ has the same signature
   ```

   This issue (Sub-2) delivers `KDAAttnBackend` only. `HybridLinearAttnBackend` (the per-layer-id dispatcher that sits on `forward_batch.attn_backend` in production) is PR #961 scope.

6. **Metadata lifecycle** (design doc §4.3.1):

   ```
   TpModelWorker.forward_batch_generation()                              ← JIT 外
     ├─ forward_metadata = attn_backend.get_forward_metadata(batch)
     │    → HybridLinearAttnBackend.get_forward_metadata():              (PR #961)
     │      ├─ full_attn_backend.get_forward_metadata(...)
     │      │     → FlashAttentionMetadata
     │      └─ linear_attn_backend.get_forward_metadata(batch, recurrent_indices)
     │             recurrent_indices computed by HybridLinearAttnBackend
     │             → LinearRecurrentAttnBackendMetadata(cu_q_lens, recurrent_indices)
     │             → linear_attn_backend.forward_metadata = result       ← sub-backend stores its own
     ├─ attn_backend.forward_metadata = forward_metadata                 ← pytree child, crosses JIT
     └─ jitted_run_model(forward_batch, memory_pool, ...)
   ```

7. **Temporary state buffer (RFC-0015 not landed)**: `LinearRecurrentAttnBackend.__call__` needs a working `recurrent_state_pool` to read/write conv + recurrent state. Since RFC-0015 is not landed, Sub-2 provides a minimal `MockRecurrentStatePool` class that implements `get_linear_recurrent_layer_cache(layer_id)` and `set_linear_recurrent_layer_cache(layer_id, indices, recurrent, conv)` using a plain dict. Sub-3 tests instantiate this mock and pass it through `RadixLinearAttention`. The mock is **not JIT-compatible** (dict is not a pytree); acceptable because Sub-3 tests don't run the full `ModelRunner._forward` JIT path. **Migration reminder**: `MockRecurrentStatePool.__init__` emits `logger.warning("Using MockRecurrentStatePool; replace with RecurrentStatePool when RFC-0015 lands")`.

**Reference patterns**:
- `FlashAttention` / `FlashAttentionMetadata` in `layers/attention/flashattention_backend.py` — pytree registration, `get_forward_metadata`, shard_map
- `LinearAttentionBackend` / `LinearAttentionMetadata` in `layers/attention/fla/linear_attention_backend.py` — linear attention metadata, scatter/gather
- `RadixAttention` in `layers/radix_attention.py` — dispatch entry pattern

**Dependencies**: Sub-1 (kernel dispatcher `__init__.py` for import).
**Blocks**: Sub-3.

---

### Sub-3: KimiDeltaAttention Layer + Numerical Alignment Tests

**Owner**: @MokusMokun
**Files**:

| File | Deliverable |
|------|-------------|
| `python/sgl_jax/srt/models/kimi_linear.py` | `KimiDeltaAttention` nnx.Module — full KDA forward. |
| `python/sgl_jax/test/layers/test_kda_backend.py` | M1 numerical alignment + prefill-to-decode invariance tests. |

**KimiDeltaAttention implementation** (design doc §4.6):

1. `__init__`: q/k/v_proj, b_proj, f_a_proj+f_b_proj, g_a_proj+g_b_proj, A_log, dt_bias, q/k/v_conv1d, o_norm (FusedRMSNormGated), o_proj, `self.attn = RadixLinearAttention(...)`
2. `__call__(hidden_states, positions, forward_batch, recurrent_state_pool)` — 4 steps:
   - Step 1: QKV + beta + forget gate + g projections (`forward_qkvbfg`)
   - Step 2: Conv1d + SiLU + L2 norm on Q/K — implemented in `KimiDeltaAttention` layer (not in backend or `RadixLinearAttention`)
   - Step 3: Dispatch via `self.attn(forward_batch, mixed_qkv, a=forget_gate, b=beta, recurrent_state_pool=recurrent_state_pool)`
   - Step 4: Output gate + FusedRMSNormGated + o_proj

   Note: `recurrent_state_pool` is passed in by upper model code (`KimiDecoderLayer`, follow-up issue), which extracts it from `memory_pool`. This layer does not see `memory_pool` directly.

**Reference**: sglang GPU `KimiDeltaAttention` (L166-408), HF `modeling_kimi.py` `KimiDeltaAttention`

**Tests** (design doc §6.2):
- Ground truth: `kda_gpu/dumps` with real HF config + weights (Sub-0 deliverable). Small-config dumps from existing package available for fast local debugging.
- 12 cases: single-seq (T=1,8,64,65,128,256,1024), varlen (balanced_4x32, unbalanced, single_T128), initial-state (2 cases)
- Tolerance: fp32 `atol=1e-5, rtol=1e-5`; bf16 `atol=1e-3, rtol=1e-3`
- Prefill-to-decode invariance: `prefill(seq[:T])` final step matches `prefill(seq[:T-k]) -> decode k steps` (k>=8)
- Weights loaded from HF safetensors by key pattern

**Dependencies**: Sub-1 (working kernel implementations via PR #964) + Sub-2 (backend + dispatch entry) + Sub-0 (real-config dumps as ground truth).

---

## Dependency Graph

```
Sub-0 (@MokusMokun: dump upgrade)  ──────────────────────┐
                                                          │
Sub-1 (kernel: PR #964 done, dispatcher remaining)        │
  │                                                       │
  └──→ Sub-2 (backend + dispatch)  ───────────────────────┤
                                                          │
                                                          └──→ Sub-3 (layer + tests)
```

| Phase | Who | What | Blocks on |
|-------|-----|------|-----------|
| Day 1 | @MokusMokun | Sub-0: dump upgrade (real config + HF weights) | Nothing |
| Day 1 | @MokusMokun | Sub-1: merge PR #964 + wire `__init__.py` dispatcher | Nothing |
| Day 1 | @MokusMokun | Sub-2: backend + dispatch | Sub-1 dispatcher |
| Day 1 | @MokusMokun | Sub-3: layer skeleton (can start with mock kernel/backend) | — |
| Merge | @MokusMokun | Sub-3: integration tests | Sub-0 + Sub-1 + Sub-2 merged |

**Maximum parallelism**: Sub-0, Sub-1 (small — just `__init__.py` wiring after PR #964 merge), and Sub-3 skeleton all start Day 1. Sub-2 starts after Sub-1's dispatcher. Sub-3 tests are the convergence point.

---

## Out of scope (deferred to follow-up issues)

- Full Kimi-Linear model assembly (`KimiLinearModel` / `KimiDecoderLayer` / 3:1 hybrid)
- `HybridLinearAttnBackend` (PR #961)
- `RecurrentStatePool` implementation (RFC-0015)
- Pallas kernel backward / training
- End-to-end MMLU-Pro evaluation
- CI registration (deferred to model integration issue)
