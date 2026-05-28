# Qwen3-VL Three-JIT Split — Design Document

## 1. Motivation

Today the entire `Qwen3VLForConditionalGeneration.__call__` — vision encoder, embed splice, and language model — lives inside a single JIT closure (`jitted_run_model` in `model_runner.py:211`). Two consequences:

1. **`pixel_values` shape participates in the JIT cache key.** Different image bucket sizes (`_MM_PATCH_BUCKETS = (256, 1024, 4096)`) each force a fresh recompilation of the *entire* 7B+ViT model graph.
2. **No precompile coverage for the vision path.** `CompilationManager` only synthesizes text-only dummy batches, so any user serving an image hits a multi-minute cold compile on the first request of each bucket.

Both the upstream sglang torch runtime and the tpu-inference JAX runtime solve this by splitting the multimodal forward into independent JIT boundaries:
- sglang torch only captures the language model in CUDA graphs; ViT runs eager and feeds `input_embeds` into the captured graph.
- tpu-inference compiles three independent functions (`embed_multimodal_fn`, `embed_input_ids_fn`, `model_fn`) and AOT-precompiles all power-of-two patch buckets at startup.

This document describes how sgl-jax adopts the three-JIT pattern for Qwen3-VL, with the bucket and precompile machinery aligned to the tpu-inference design. The work lands as a new PR independent of #1189 (which contains the ViT splash kernel and Control-plane review fixes).

## 2. Requirements and Non-Goals

### Functional Requirements

- Vision encoder runs in its own JIT boundary, keyed on `n_patches_bucket` only.
- LLM JIT cache is shared between text-only and multimodal extends (no `pixel_values` in cache key).
- Patch buckets are powers of two derived from `chunked_prefill_size * spatial_merge_unit`, replacing the hardcoded `(256, 1024, 4096)` ladder.
- Precompile coverage at startup for every power-of-two patch bucket, eliminating cold-compile cost on first image request.
- No regression on text-only paths or decode paths.
- No regression on existing tests: `test_qwen3_vl_models.py` (`test_logit_alignment_vs_hf`, `test_multi_image_prefill`).

### Non-Goals (S3 follow-ups, separate PR)

- **Per-image embedding cache** (content-hash keyed, cross-request reuse).
- **Chunked-prefill `items_offset` + `find_chunk_items_and_check_cache` + `assemble_chunk_embedding`** algorithm for cross-chunk image reuse.
- **`offload_mm_features_to_cpu`** for GPU memory release after ViT.
- **Retraction support** from CPU-side `mm_inputs` copy.
- **Applying the split to `qwen2_5_vl` / `qwen3_omni_thinker`** (Qwen3-VL only for this PR).
- **Three-JIT encoder cache integration** — depends on the per-image cache landing first.

## 3. Architecture

### Current (Monolithic JIT)

```
ForwardBatch (pixel_values, image_grid_thw, input_ids, ...)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ jitted_run_model (single JIT, ~7B + 700M params)       │
│                                                        │
│ Qwen3VLForConditionalGeneration.__call__:              │
│   if extend AND pixel_values is not None:              │
│       vision_features = self.visual(pixel_values, ...) │ ← ViT
│       embed splice → forward_batch.input_embedding     │ ← scatter
│       full_deepstack scatter                           │
│   self.model(forward_batch, ...)                       │ ← LLM
│   self.logits_processor(...)                           │
│                                                        │
│ Cache key: (input_ids_len, pixel_values.shape, ...)    │
│ → Every new pixel_values bucket recompiles entire ~8B  │
│   graph (≈30–90s per bucket on TPU v6e).               │
└────────────────────────────────────────────────────────┘
```

### Target (Three-JIT Split)

```
ForwardBatch (pixel_values, image_grid_thw, input_ids, ...)
       │
       ▼ (Python orchestration in model_runner.run_model_wrapper)
       │
       ├─ if multimodal AND extend AND pixel_values is not None:
       │
       │     ┌─────────────────────────────────────────────────┐
       │     │ jitted_visual_encode (NEW, ViT only ~700M)      │
       │     │                                                 │
       │     │   inputs:  pixel_values  [n_patches_bucket,1536]│
       │     │           image_grid_thw (static tuple)         │
       │     │           cu_seqlens, n_real_images             │
       │     │   outputs: vision_main      [N_padded_tokens,H] │
       │     │           vision_deepstack [...,H*N_ds]         │
       │     │                                                 │
       │     │   Cache key: (n_patches_bucket, image_grid_thw, │
       │     │              n_real_images_bucket)              │
       │     └─────────────────────────────────────────────────┘
       │                              │
       │                              ▼ (S3 hook point — encoder_cache lookup)
       │                              │
       │     ┌─────────────────────────────────────────────────┐
       │     │ jitted_splice_embeds (NEW)                      │
       │     │                                                 │
       │     │   inputs:  input_ids        [seq_len]           │
       │     │           vision_main      [N_padded_tokens,H]  │
       │     │           vision_deepstack [...,H*N_ds]         │
       │     │           placeholder_positions [N_padded_tokens]│
       │     │   outputs: input_embedding    [seq_len, H]      │
       │     │           deepstack_embedding [seq_len, H*N_ds] │
       │     │                                                 │
       │     │   logic:   text = embed_tokens(input_ids)       │
       │     │           input_embedding =                     │
       │     │             text.at[positions].set(vision_main) │
       │     │           deepstack = zeros.at[positions].set(  │
       │     │                       vision_deepstack)         │
       │     │                                                 │
       │     │   Cache key: (seq_len, N_padded_tokens)         │
       │     │                                                 │
       │     │   Note: embed_tokens IS here (option A, matches │
       │     │   tpu-inference). The output is fully-spliced.  │
       │     │   See §4.1.                                     │
       │     └─────────────────────────────────────────────────┘
       │                              │
       │                              ▼
       │           forward_batch = replace(forward_batch,
       │              input_embedding=input_embedding,
       │              deepstack_visual_embedding=deepstack,
       │           )
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ jitted_run_model (REUSED, LLM only ~7B)                │
│                                                        │
│ Qwen3VLForConditionalGeneration.__call__:              │
│   # ViT path removed — input_embedding is pre-filled   │
│   self.model(forward_batch, ...)                       │
│   self.logits_processor(...)                           │
│                                                        │
│ Cache key: (input_ids_len, ...) — pixel_values gone.   │
│ → Same JIT serves text-only AND multimodal extends.    │
└────────────────────────────────────────────────────────┘
```

### Text-only / Decode Path (Unchanged)

When `pixel_values` is None (text-only extend, all decode mode), the orchestrator skips both vision JITs and calls `jitted_run_model` directly. The LLM internally runs `embed_tokens(input_ids)` as today (`Qwen3VLLanguageModel.__call__:850`).

## 4. Design

### 4.1 `embed_tokens` Placement (Q1=A, matches tpu-inference)

`embed_tokens(input_ids)` runs inside `jitted_splice_embeds`, not the LLM JIT. Concretely:

- **Multimodal extend:** orchestrator calls splice JIT, which produces a fully-spliced `input_embedding` of shape `[seq_len, hidden]`. LLM JIT receives `forward_batch.input_embedding` and skips its own `embed_tokens` call.
- **Text-only extend / decode:** orchestrator does NOT call splice JIT. `forward_batch.input_embedding` stays `None`. LLM JIT runs `embed_tokens(input_ids)` internally as today.

The LLM-side `Qwen3VLLanguageModel.__call__` semantics are **unchanged from the current implementation** — it already branches on `forward_batch.input_embedding is None` (line 850). All we change is when the orchestrator populates `input_embedding`: today it happens inside the model's `__call__`; with this PR it happens via the orchestrator before `jitted_run_model` is entered.

**Cache key isolation:**

| JIT | Cache key dimensions | Variance source |
|---|---|---|
| `jitted_visual_encode` | `(n_patches_bucket, image_grid_thw, n_real_images_bucket)` | ViT inputs only |
| `jitted_splice_embeds` | `(seq_len_bucket, N_padded_tokens_bucket)` | Splice geometry |
| `jitted_run_model` (LLM) | `(seq_len_bucket, ...)` — same as text-only path | No image axes |

Critically, the LLM JIT cache is **fully shared** between text-only and multimodal extends, because `forward_batch.input_embedding` is always one of:
- `None` (text-only / decode): LLM does `embed_tokens(input_ids)`.
- `[seq_len, hidden]` (multimodal): LLM uses it directly.

In both cases, the LLM JIT input pytree has the same shape footprint. No `pixel_values` and no `N_padded_tokens` ever participate in the LLM cache key.

**Rationale for choosing A over the alternative "splice does scatter only, LLM overlays":** the overlay approach forces `placeholder_positions` (whose length depends on `N_padded_tokens`) into the LLM JIT cache key, breaking LLM cache sharing between text-only and multimodal extends. A is strictly cleaner.

### 4.2 Orchestration

Implemented in `ModelRunner._make_jitted_funcs` and `run_model_wrapper` (the existing wrapper at `model_runner.py:250`):

```python
def run_model_wrapper(forward_batch, logits_metadata):
    needs_visual = (
        self.is_multimodal_model
        and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        and forward_batch.pixel_values is not None
    )
    if needs_visual:
        vision_main, vision_deepstack = self.jitted_visual_encode(forward_batch)
        input_embedding, deepstack_embedding = self.jitted_splice_embeds(
            forward_batch.input_ids,
            vision_main,
            vision_deepstack,
            forward_batch.placeholder_positions,
        )
        forward_batch = dataclasses.replace(
            forward_batch,
            input_embedding=input_embedding,
            deepstack_visual_embedding=deepstack_embedding,
        )
    return self.jitted_run_model_impl(forward_batch, logits_metadata)
```

The `is_multimodal_model` flag is detected by checking whether the model class exposes `encode_visual` and `splice_embeds` methods. Non-VLM models and Qwen3-VL until this PR lands skip the new path entirely.

### 4.3 Model Class Changes (`qwen3_vl.py`)

Add two new pure-function methods on `Qwen3VLForConditionalGeneration`:

```python
def encode_visual(self, forward_batch):
    """Run ViT + deepstack split. Pure function of the visual fields of forward_batch.

    Inputs (read from forward_batch):
        pixel_values, image_grid_thw, cu_seqlens, n_real_images
    Returns: (vision_main [N_padded, H], vision_deepstack [N_padded, H*N_ds])
    """
    vision_features = self.visual(
        forward_batch.pixel_values,
        forward_batch.image_grid_thw,
        cu_seqlens=forward_batch.cu_seqlens,
        n_real_images=forward_batch.n_real_images,
    )
    return self.separate_deepstack_embeds(vision_features)

def splice_embeds(self, input_ids, vision_main, vision_deepstack, placeholder_positions):
    """Embed text tokens + scatter vision embeddings at placeholder positions.

    Returns: (input_embedding [seq_len, H], deepstack_embedding [seq_len, H*N_ds])
    """
    repl = NamedSharding(self.mesh, P())
    text_embeds = self.model.embed_tokens(input_ids)
    text_embeds_repl = jax.sharding.reshard(text_embeds, repl)
    input_embedding = text_embeds_repl.at[placeholder_positions].set(
        vision_main.astype(text_embeds_repl.dtype)
    )

    zero_deepstack = jnp.zeros(
        (input_embedding.shape[0], vision_deepstack.shape[-1]),
        dtype=input_embedding.dtype,
        device=repl,
    )
    deepstack_embedding = zero_deepstack.at[placeholder_positions].set(
        vision_deepstack.astype(input_embedding.dtype)
    )
    return input_embedding, deepstack_embedding
```

`Qwen3VLForConditionalGeneration.__call__` strips the inline ViT block (lines 962–1002 today): the orchestrator has already populated `forward_batch.input_embedding` and `forward_batch.deepstack_visual_embedding` when applicable. The LLM-side `Qwen3VLLanguageModel.__call__` is **unchanged** — its existing `forward_batch.input_embedding is None` branch (line 850) already handles both paths correctly.

### 4.4 Bucket Derivation

Replace the hardcoded `_MM_PATCH_BUCKETS = (256, 1024, 4096)` in `schedule_batch.py` with a derived ladder:

```python
def compute_patch_buckets(chunked_prefill_size: int, spatial_merge_size: int,
                          min_patches: int = 16) -> tuple[int, ...]:
    """Power-of-two ladder from min_patches to next_pow2(max_patches)."""
    spatial_merge_unit = spatial_merge_size ** 2
    max_patches = chunked_prefill_size * spatial_merge_unit
    min_shift = max(1, (min_patches - 1).bit_length())
    max_shift = max(min_shift, (max_patches - 1).bit_length())
    return tuple(1 << i for i in range(min_shift, max_shift + 1))
```

For Qwen3-VL with `spatial_merge_size=2` (so `spatial_merge_unit=4`) and `chunked_prefill_size=4096`:

```
max_patches = 16384
buckets = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
```

The bucket list is computed once at startup and threaded through the same way the existing `_MM_PATCH_BUCKETS` is consumed (`_collect_mm_tensors`, `CompilationManager`). The `_MM_IMAGE_BUCKETS = (1, 2, 4, 8)` for `n_padded_images` stays as-is for now.

The `_pick_bucket` overflow warning landed in PR #1189 stays in place; with the derived ladder reaching `chunked_prefill_size * spatial_merge_unit`, overflow becomes nearly impossible for in-spec requests.

### 4.5 Precompile Strategy

`CompilationManager` gains two new precompile passes. They run only when `model_runner.is_multimodal_model` is true.

**`_precompile_visual_encode`** — patches-only ladder (Q2 simplification):

```python
for n_patches in self.patch_buckets:                       # ~11 buckets
    dummy_fb = self._make_dummy_visual_batch(
        n_patches=n_patches, n_real_images=1, ...
    )
    self.jitted_visual_encode(dummy_fb)                   # triggers AOT compile
```

We fix `n_real_images=1` (most common single-image case). Multi-image cases trigger lazy compile on first occurrence; the `_pick_bucket` warning surfaces the recompile so we can grow the precompile set later if usage patterns warrant.

**`_precompile_splice`** — small Cartesian product:

```python
for seq_len in self.token_buckets:                         # ~4 buckets
    for n_padded_tokens in self.visual_token_buckets:      # ~3 buckets (small subset)
        if n_padded_tokens > seq_len: continue            # impossible
        self.jitted_splice_embeds(dummy_input_ids,
                                  dummy_vision_main,
                                  dummy_vision_deepstack,
                                  dummy_positions)
```

Splice is a single scatter — each compile is sub-second. The Cartesian product (≤12 compiles) is cheap.

`_precompile_extend` / `_precompile_decode` are unchanged. The dummy batches they generate continue to have `pixel_values=None`, so the LLM JIT graph no longer has any image-specific shape variants. **The LLM JIT is now identical for text-only and multimodal extends.**

### 4.6 ForwardBatch Contract Changes

`ForwardBatch` already exposes `input_embedding` and `deepstack_visual_embedding` (see `forward_batch_info.py:430`). The contract changes are:

- **Before:** `input_embedding` is populated inside `Qwen3VLForConditionalGeneration.__call__` via side-effecting assignment (`forward_batch.input_embedding = ...`).
- **After:** `input_embedding` is populated by the orchestrator via `dataclasses.replace(forward_batch, input_embedding=...)` before `jitted_run_model` is entered.

Verify `ForwardBatch` supports `dataclasses.replace` (it is a plain `@dataclass`, so it should). If frozen, replace with a custom `.replace()` helper that preserves pytree registration.

### 4.7 Test Plan

| Layer | What | How |
|---|---|---|
| Unit | `compute_patch_buckets` correctness | Existing test pattern in `test/srt/`; assert shape of returned tuple for representative `chunked_prefill_size` values. |
| Unit | `splice_embeds` correctness | Construct deterministic `vision_main` / `placeholder_positions`, run JIT, assert scattered output equals reference numpy implementation. |
| Unit | `encode_visual` shape contract | For each `n_patches_bucket`, dummy `pixel_values=zeros`, assert output shapes. |
| Integration | Precompile coverage | Server start with `--tp-size 4 --context-length 8192 --page-size 128` (no `--disable-precompile`); log inspection confirms 11 visual-encode buckets + 12 splice buckets compiled at startup. |
| Integration | First request latency | Send single multimodal request immediately after server ready; measure end-to-end latency. Expectation: no multi-minute cold compile (would indicate precompile missed the bucket). |
| Regression | Existing `test_qwen3_vl_models.py` | `test_logit_alignment_vs_hf` and `test_multi_image_prefill` must pass unchanged. |
| End-to-end | Full MMMU val | `lmms-eval mmmu_val --batch_size 128 --gen_kwargs max_new_tokens=4096` on ramezes-mimo-audio-v6e4 pod, expect score within noise of previous 256-sample run (54.69%). |

## 5. Implementation Order

1. **Bucket derivation** — `compute_patch_buckets` helper + replace `_MM_PATCH_BUCKETS` reads.
2. **Model class split** — add `encode_visual` / `splice_embeds`, strip ViT block from `__call__`, update `Qwen3VLLanguageModel` overlay semantics.
3. **Model runner orchestration** — `jitted_visual_encode`, `jitted_splice_embeds`, multimodal-aware `run_model_wrapper`.
4. **CompilationManager** — `_precompile_visual_encode`, `_precompile_splice`.
5. **Tests** — unit + integration + regression.
6. **End-to-end validation** — server start + MMMU eval.

Each step is independently testable and commit-able.

## 6. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `dataclasses.replace(forward_batch, ...)` breaks pytree registration | Med | Investigate `ForwardBatch` registration upfront; provide custom `.replace()` if needed. |
| LLM overlay semantics break existing test_logit_alignment | Med | Test by inserting overlay step gated behind a flag first; only switch over after parity verified. |
| First image request still cold-compiles (precompile missed a bucket) | Low | Log inspection during integration test; lazy compile path with warning catches any miss. |
| Multimodal model detection too narrow / too broad | Low | Conservative `hasattr(model, "encode_visual")` check; only Qwen3-VL has the methods in this PR. |
| Sharding mismatch between `jitted_visual_encode` output and LLM JIT input | Med | Both use the same `NamedSharding(mesh, P())` replicated layout; assert via unit test. |
| Performance regression from extra Python-level orchestration overhead | Low | Orchestration is host-only `if`/replace, no device sync; negligible vs 7B forward time. |

## 7. Follow-ups (S3 — separate PR)

Track as TODO comments in the PR description and as a follow-up issue:

- [ ] **Per-image embedding cache** — `MultiModalEmbeddingCache` keyed on `mm_input.hash`. Hook between `jitted_visual_encode` output and `jitted_splice_embeds` input: skip the ViT JIT when the image hash hits the cache.
- [ ] **Three-JIT cache integration** — wire the encoder cache into the orchestrator. Batch all cache-miss images across the entire batch into a single `jitted_visual_encode` call; per-request cache lookups feed the splice JIT.
- [ ] **`items_offset` + chunked-prefill correctness** — port sglang upstream's `items_offset = [(start, end), ...]` per image, plus `find_chunk_items_and_check_cache` (`end >= chunk_start AND start < chunk_end`) and `assemble_chunk_embedding` (`overlap_start`/`overlap_end` + `local_start`/`local_end` slicing). Fixes cross-chunk image splice and enables cross-chunk ViT reuse via cache.
- [ ] **`offload_mm_features_to_cpu`** + clear `forward_batch.mm_inputs` after ViT — release device memory of `pixel_values` once vision embedding is computed, keep CPU copy for retraction.
- [ ] **Retraction support** — when a request is retracted, restore `mm_inputs` from CPU copy so re-prefill does not re-preprocess images.
- [ ] **Apply same split to `qwen2_5_vl` and `qwen3_omni_thinker`** — once the Qwen3-VL pattern is validated, mirror the split into the other VLM model classes.

## 8. Open Questions

None blocking — Q1 (`embed_tokens` placement = **inside splice JIT**, matching tpu-inference), Q2 (precompile patches-only with `n_real_images=1`), Q3 (S3 cache integration listed in §7) all resolved before drafting this doc.

The Q1 decision was revised during spec self-review: the original draft put `embed_tokens` in the LLM JIT (with the splice JIT doing a zero-buffer scatter and the LLM overlaying), but that forces `placeholder_positions` into the LLM JIT cache key — defeating the goal of cache sharing between text-only and multimodal extends. Moving `embed_tokens` into the splice JIT keeps the LLM cache key purely a function of `seq_len`.
