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

- **PR #966 not landed yet**: `RecurrentStatePool` / `MemoryPools` / `HybridReqToTokenPool` are unavailable. Backend must use interface stubs for state pool consumption.
- **No naive prefill**: `chunk_kda_jax_naive` is not in scope. v1 ships pallas prefill + naive decode.

### Deviation from design doc

- **Base class file location**: Design doc §3.2/§3.3 places `LinearRecurrentAttnBackend` + `LinearRecurrentAttnBackendMetadata` in `kda_backend.py`. This decomposition keeps them in `hybrid_linear_attn_backend.py` instead, because: (1) they are shared base classes for KDA / GDN / Mamba2, not KDA-specific; (2) PR #961 already placed stubs there; (3) `kda_backend.py` already imports from `hybrid_linear_attn_backend.py`. The design doc should be updated to reflect this.
- **Kernel dispatcher fallback**: Design doc §4.2 shows symmetric fallback for both prefill and decode (`chunk_kda_pallas if _HAS_PALLAS else chunk_kda_jax_naive`, same for decode). This decomposition deviates: **prefill = pallas only** (no naive prefill impl exists — `chunk_kda_jax_naive` is out of scope), **decode = naive only** (no pallas decode impl exists). The dispatcher in `__init__.py` directly re-exports `chunk_kda_fwd` from `kda.py` and `naive_recurrent_kda` from `naive.py` without try/except fallback.
- **L2 norm outside kernel**: GPU reference uses `use_qk_l2norm_in_kernel=True` (L2 norm done inside the kernel). JAX kernels do not support this — Pallas kernel: `assert use_qk_l2norm_in_kernel is False`; naive kernel: no such parameter. L2 norm is done in `KimiDeltaAttention` layer before dispatch.
- **Gate activation split across kernel paths**: GPU reference computes `fused_kda_gate` in the layer and passes activated gate to the kernel uniformly. In JAX, the Pallas kernel supports `use_gate_in_kernel=True` (accepts raw gate + `A_log` + `dt_bias`, does activation internally), but the naive kernel expects pre-activated gate. `KDAAttnBackend` handles this asymmetry: prefill path passes raw gate with `use_gate_in_kernel=True`; decode path computes activation inline (`-exp(A_log) * softplus(g + dt_bias)`) before calling naive kernel. `KimiDeltaAttention` always passes raw gate.

### Development workflow

```
Remotes:
  origin  → sgl-project/sglang-jax   (upstream, read-only for @MokusMokun)
  mokus   → MokusMokun/sglang-jax    (fork, push target)

Branches:
  epic/support_kimi_linear  (upstream, PR #961/#962 merged)
    │
    └─ feat/kda-inference-components  (主分支, branched from epic)
         │
         ├─ sub0/dump-upgrade        ──→ merge --no-ff → feat/kda-inference-components
         ├─ sub1/kernel-dispatcher   ──→ merge --no-ff → feat/kda-inference-components
         ├─ sub2/backend-dispatch    ──→ merge --no-ff → feat/kda-inference-components
         └─ sub3/layer-tests         ──→ merge --no-ff → feat/kda-inference-components

Worktree paths (.claude/worktrees/):
  kda-main              → feat/kda-inference-components  (主分支, merge 操作在这里)
  sub0-dump-upgrade     → sub0/dump-upgrade
  sub1-kernel-dispatcher → sub1/kernel-dispatcher
  sub2-backend-dispatch  → sub2/backend-dispatch
  sub3-layer-tests       → sub3/layer-tests
```

- **主分支**: `feat/kda-inference-components`，从 `epic/support_kimi_linear` 拉出
- **Push 目标**: 所有分支 push 到 `mokus` (MokusMokun/sglang-jax)，`origin` 无写权限
- **并行开发**: 每个 sub-issue 一个 worktree + 分支（`sub{N}/xxx`），按工作内容命名
- **合入**: worktree 完成后在 `kda-main` worktree 执行 `git merge --no-ff sub{N}/xxx`（保留 merge commit，方便回溯）
- **Upstream 同步**: 只有 `feat/kda-inference-components` 跟 `epic/support_kimi_linear` rebase
- **最终 PR**: `MokusMokun/sglang-jax:feat/kda-inference-components` → `sgl-project/sglang-jax:epic/support_kimi_linear`

---

## Sub-issues

### Sub-0: GPU Dump Upgrade (HF real config + weights)

**Owner**: @MokusMokun
**Location**: `kda_gpu/` (existing package at `_Delivery/202604B_sgl_jax_KDA_ling3/2026-04-23/kda_gpu/`)

**Current state**: 12 cases exist with small synthetic config (`hidden=128, heads=4, head_dim=32`, random-init weights). Scripts (`run_kda_gpu.py`, `fixed_kda_module.py`) and dump schema are solid.

**What needs to change**:

| Gap | Current | Required (design doc §6.1) |
|-----|---------|---------------------------|
| Config | `hidden=128, heads=4, head_dim=32` | `hidden_size=2304, heads=32, head_dim=128` (HF checkpoint actual config; `projection_size` = 32×128 = 4096) |
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
   - Fields: `cu_q_lens: jax.Array` (`[N+1]` int32), `recurrent_indices: jax.Array` (`[B]` int32, pool slot indices — computed internally via `req_to_token_pool.get_linear_recurrent_indices`)
   - `@register_pytree_node_class`; children = `(cu_q_lens, recurrent_indices)`, aux_data = `{}`

2. **Backend base** (`LinearRecurrentAttnBackend` in `hybrid_linear_attn_backend.py`):
   - Extends `AttentionBackend`
   - `get_forward_metadata(model_worker_batch)` — computes `cu_q_lens` from batch; `recurrent_indices` computed internally via `self.req_to_token_pool.get_linear_recurrent_indices(batch.req_pool_indices)`. Returns `LinearRecurrentAttnBackendMetadata`.
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

   **Sub-1 review notes for implementer**:
   - **12-tuple return**: `chunk_kda` (prefill) returns a 12-tuple `(o, final_state, g_cumsum, Aqk, Akk, None, None, None, None, None, None, initial_state)` — positions 5-10 are always `None` (intermediates released for GC). Destructure as `o, final_state, *_ = chunk_kda(...)`.
   - **State shape asymmetry**: prefill `initial_state` is `[N, H, K, V]` (N = number of sequences), decode `initial_state` is `[B, H, K, V]`. Backend must handle this shape difference when reading/writing `recurrent_state_pool`.
   - `fused_recurrent_kda` (decode) returns a clean 2-tuple `(o, final_state)` — no special handling needed.

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
     │      └─ linear_attn_backend.get_forward_metadata(batch)
     │             recurrent_indices computed internally via req_to_token_pool
     │             → LinearRecurrentAttnBackendMetadata(cu_q_lens, recurrent_indices)
     │             → linear_attn_backend.forward_metadata = result       ← sub-backend stores its own
     ├─ attn_backend.forward_metadata = forward_metadata                 ← pytree child, crosses JIT
     └─ jitted_run_model(forward_batch, memory_pool, ...)
   ```

7. **Temporary state buffer (PR #966 not landed)**: `LinearRecurrentAttnBackend.__call__` needs a working `recurrent_state_pool` to read/write conv + recurrent state. Since PR #966 is not landed, Sub-2 provides a minimal `MockRecurrentStatePool` class that implements `get_linear_recurrent_indices(req_pool_indices)`, `get_linear_recurrent_layer_cache(layer_id)` and `set_linear_recurrent_layer_cache(layer_id, indices, recurrent, conv)` using a plain dict. Sub-3 tests instantiate this mock and pass it through `RadixLinearAttention`. The mock is **not JIT-compatible** (dict is not a pytree); acceptable because Sub-3 tests don't run the full `ModelRunner._forward` JIT path. **Migration reminder**: `MockRecurrentStatePool.__init__` emits `logger.warning("Using MockRecurrentStatePool; replace with HybridReqToTokenPool + RecurrentStatePool when PR #966 lands")`.

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
| `python/sgl_jax/srt/models/kimi_linear.py` | `KimiDeltaAttention` nnx.Module — full KDA forward. **Only** this class; no KimiDecoderLayer/KimiModel/KimiLinearForCausalLM (those are PR #968 scope). |
| `python/sgl_jax/srt/layers/attention/linear/kda_backend.py` | Minor changes to `_forward_extend` and `_forward_decode` for gate activation handling. |
| `python/sgl_jax/test/layers/test_kda_backend.py` | M1 numerical alignment + prefill-to-decode invariance tests. |

**PR #968 boundary**: PR #968 (collaborator) creates the same `kimi_linear.py` with the full model skeleton + stub `KimiDeltaAttention`. Sub-3 creates the file with only `KimiDeltaAttention` implemented; the collaborator integrates it into their model skeleton afterward. `KimiDecoderLayer` pool routing (passing `recurrent_state_pool` instead of `token_to_kv_pool` to KDA layers) is part of that integration, not this sub-issue.

**Reference**: `fixed_kda_module.py` (Sub-0 GPU reference), HF `modeling_kimi.py` `KimiDeltaAttention`

Sub-3 is split into three phases:

#### Phase A: `KimiDeltaAttention` nnx.Module + `KDAAttnBackend` gate changes

**KimiDeltaAttention implementation** (design doc §4.6):

1. `__init__`: q/k/v_proj, b_proj, f_a_proj+f_b_proj, g_a_proj+g_b_proj, A_log, dt_bias, conv1d weights, o_norm weights, o_proj, `self.attn = RadixLinearAttention(..., A_log=A_log, dt_bias=dt_bias)`
2. `__call__(hidden_states, positions, forward_batch, recurrent_state_pool)` — 4 steps:
   - Step 1: QKV + beta + forget gate (raw, before activation) + output gate projections
   - Step 2: Conv1d + SiLU + **L2 norm on Q/K** — implemented in `KimiDeltaAttention` layer (not in backend or `RadixLinearAttention`)
   - Step 3: Dispatch via `self.attn(forward_batch, mixed_qkv, a=raw_gate, b=beta, recurrent_state_pool=recurrent_state_pool)` — **layer passes raw gate** (after low-rank projection, before activation)
   - Step 4: Gated RMSNorm + o_proj

   Note: `recurrent_state_pool` is passed in by upper model code (`KimiDecoderLayer`, follow-up issue), which extracts it from `memory_pool`. This layer does not see `memory_pool` directly.

**Naive inline components** (no separate utility classes):

| Component | Implementation | Why inline |
|-----------|---------------|------------|
| **Conv1d** | `jax.lax.conv_general_dilated` or slice-and-dot (kernel_size=4, causal). Conv state read/write via `recurrent_state_pool`. | No conv1d utility exists in codebase |
| **L2 norm on Q/K** | `x / max(‖x‖₂, eps)` per head. Applied after conv1d+SiLU, before dispatch. | Both JAX kernels lack in-kernel L2 norm (see Deviation §3) |
| **Gated RMSNorm** | `o_norm(o, g_out)`: split gate → RMSNorm → element-wise multiply. Reference: `fixed_kda_module.py` line 221 | No `FusedRMSNormGated` in codebase; only `RMSNorm` and `GroupRMSNorm` exist |

**`KDAAttnBackend` gate activation changes** (modifies Sub-2 delivered file):

The layer always passes **raw gate `g`** (after low-rank projection `f_b(f_a(x))`, before activation). Gate activation is handled differently per kernel path:

- **`_forward_extend`** (Pallas kernel): pass `A_log=layer.A_log, dt_bias=layer.dt_bias, use_gate_in_kernel=True` to `chunk_kda`. Kernel computes activation + cumsum internally via `kda_gate_chunk_cumsum`: `g_act = -exp(A_log) * softplus(g + dt_bias)`.
- **`_forward_decode`** (naive kernel): compute gate activation inline before calling `fused_recurrent_kda`:
  ```python
  g = -jnp.exp(layer.A_log) * jax.nn.softplus(g + layer.dt_bias)
  ```
  Naive kernel expects pre-activated gate in log-space (docstring: `"Per-element gate in log-space (e.g., -exp(A)*softplus(g))"`).

This split keeps `KimiDeltaAttention` clean (no gate activation logic) and leverages the Pallas kernel's built-in gate support. `RadixLinearAttention` already stores `A_log` and `dt_bias` as attributes (set in Sub-2), so the backend accesses them via the `layer` parameter.

#### Phase B: M1 module-level numerical alignment tests

- **Execution environment**: TPU v6e-4 (`sky-efe2-yuhao`), `conda activate sglang` (JAX 0.8.1). Pallas kernel runs on real TPU hardware.
- **Ground truth**: Sub-0 real-config dumps on GCS FUSE mount at `/models/yuhao/kimi-linear/kda_module/{L0,L6,L13,L22}/` (`gs://model-storage-sglang`). Each layer directory contains `weights.npz` + 12 `case_*.npz` files.
- **Test flow**:
  1. Load `weights.npz` → construct `KimiDeltaAttention` with numpy weights (key pattern: `weights__q_proj.weight`, `weights__A_log`, etc.)
  2. Set up `ForwardBatch` with `attn_backend = KDAAttnBackend` (direct, no `HybridLinearAttnBackend`)
  3. Set up `MockRecurrentStatePool` with initial state from dump (when `has_initial_state=True`)
  4. For each of 12 `case_*.npz`: run `KimiDeltaAttention` forward on `hidden_states` input, compare output vs `out_fp32`
  5. Intermediate comparisons (`intermediates__q_after_conv`, `intermediates__g`, `intermediates__beta`, `intermediates__o_kda_chunk`, `intermediates__o_norm`) available to pinpoint which step diverges
- **12 cases**: single-seq (T=1,8,64,65,128,256,1024), varlen (balanced_4x32, unbalanced, single_T128), initial-state (2 cases)
- **Tolerance**: fp32 `atol=1e-5, rtol=1e-5`; bf16 `atol=1e-3, rtol=1e-3`
- **Layer coverage**: L0 by default for fast validation; optionally parametrize over all 4 layers (L0/L6/L13/L22)

#### Phase C: Prefill-to-decode invariance tests

- Self-contained (no external dumps, uses random weights)
- `prefill(seq[:T])` final-step output must match `prefill(seq[:T-k]) → decode k steps` (k≥8)
- Tests that the EXTEND→DECODE transition through `KDAAttnBackend` + `MockRecurrentStatePool` state write-back is consistent
- Validates both kernel paths (Pallas prefill → naive decode) produce coherent results

**Dependencies**: Sub-1 (kernel) + Sub-2 (backend + dispatch entry) + Sub-0 (real-config dumps for Phase B).

---

## Dependency Graph

```
Sub-0 (@MokusMokun: dump upgrade)  ──────────────────────┐
                                                          │
Sub-1 (kernel: PR #964 done, dispatcher remaining)        │
  │                                                       │
  └──→ Sub-2 (backend + dispatch)  ───────────────────────┤
                                                          │
                                                          └──→ Sub-3 Phase A (layer + backend gate changes)
                                                                │
                                    Sub-0 dumps on GCS ─────────┼──→ Sub-3 Phase B (M1 alignment tests)
                                                                │
                                                                └──→ Sub-3 Phase C (prefill-to-decode invariance)
```

| Phase | Who | What | Blocks on |
|-------|-----|------|-----------|
| Day 1 | @MokusMokun | Sub-0: dump upgrade (real config + HF weights) | Nothing |
| Day 1 | @MokusMokun | Sub-1: merge PR #964 + wire `__init__.py` dispatcher | Nothing |
| Day 1 | @MokusMokun | Sub-2: backend + dispatch | Sub-1 dispatcher |
| Day 2 | @MokusMokun | Sub-3 Phase A: `KimiDeltaAttention` + `KDAAttnBackend` gate changes | Sub-1 + Sub-2 merged |
| Day 2 | @MokusMokun | Sub-3 Phase B: M1 numerical alignment tests (TPU v6e-4) | Phase A + Sub-0 dumps |
| Day 2 | @MokusMokun | Sub-3 Phase C: prefill-to-decode invariance tests | Phase A |

**Maximum parallelism**: Sub-0, Sub-1 (small — just `__init__.py` wiring after PR #964 merge) start Day 1. Sub-2 starts after Sub-1's dispatcher. Sub-3 Phase A starts after Sub-1 + Sub-2 merge. Phase B requires Phase A + Sub-0 dumps on GCS. Phase C requires only Phase A (self-contained).

---

## Out of scope (deferred to follow-up issues)

- Full Kimi-Linear model assembly (`KimiLinearModel` / `KimiDecoderLayer` / 3:1 hybrid)
- PR #968 model skeleton integration (`KimiDecoderLayer` pool routing, `KimiModel`, `KimiLinearForCausalLM`, weight mappings)
- `HybridLinearAttnBackend` (PR #961)
- `RecurrentStatePool` implementation (PR #966)
- Pallas kernel backward / training
- End-to-end MMLU-Pro evaluation
- CI registration (deferred to model integration issue)
