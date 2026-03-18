# Plan: Remove Packed Approach for Multi-Item Scoring

## Objective
Remove the legacy "Packed Approach" (V1) for multi-item scoring. The goal is to clean up unnecessary complexity while keeping single-item scoring and the optimized "Fast Path V2" (Prefill + Extend) intact. 

## Flow Walkthrough & Why `is_multi_item_scoring` is Safe to Delete
Before detailing the changes, here is a walkthrough of the multi-item scoring flow to verify we aren't deleting anything required:

1. **Request Intake:** A request hits `POST /v1/score` with a `query` and a list of `items`.
2. **Strategy Routing (`tokenizer_manager.py`):**
   * *Legacy Packed Approach:* Appends all items to the query using a `delimiter_token_id`. It wraps this massive 1D sequence in a `GenerateReqInput` and sets the internal flag `is_multi_item_scoring=True`.
   * *Fast Path V2 (Prefill + Extend):* Sends the `query` as a pure prefill to get a `cache_handle`. It then packs the `items` into a 2D array inside a `ScoreFromCacheReqInput` and bypasses the standard queue. **It does NOT use `is_multi_item_scoring`** because the scheduler naturally treats each item in the chunk as an independent sequence. 
   * *Fallback for Fast Path V2:* If the fast path fails, it falls back to enqueuing standard independent `GenerateReqInput`s that share the prefix cache. The code explicitly notes: `We don't mark is_multi_item_scoring here because these are treated as individual requests that happen to share a prefix.`
3. **Engine Execution:** The engine (`schedule_batch.py`, `flashattention_backend.py`) currently looks for `is_multi_item_scoring=True` to apply complex custom RoPE position resets and segment-based attention masking so items in the 1D packed sequence don't attend to each other.
4. **Score Extraction:** The `scheduler_output_processor_mixin.py` uses the delimiters to extract scores from the massive 1D output.

**Conclusion:** Because we are entirely removing the Legacy Packed Approach and routing *all* multi-item requests to Fast Path V2, the engine no longer needs to deal with 1D packed sequences. Therefore, the `is_multi_item_scoring` flag, the custom RoPE resets, the segment masking, and the delimiter-based score extraction are all dead code and can be safely deleted.

---

## 0. Testing Step (Before Edit)
1. Open a terminal and activate the virtual environment: `source .venv/bin/activate`
2. Start the server in the background using the legacy command:
   ```bash
   python -m sgl_jax.launch_server \
     --model-path Qwen/Qwen3-0.6B --trust-remote-code --host 0.0.0.0 --port 30000 \
     --device tpu --tp-size 1 --nnodes 1 --log-level info --node-rank 0 \
     --dist-init-addr 0.0.0.0:10011 --dtype bfloat16 \
     --mem-fraction-static 0.7 --max-prefill-tokens 2048 --chunked-prefill-size -1 \
     --precompile-token-paddings 1024 2048 4096 8192 16384 32768 \
     --precompile-bs-paddings 1 16 32  64 \
     --max-running-requests 128 --page-size 64 --attention-backend fa \
     --skip-server-warmup --multi-item-scoring-delimiter 128001 \
     --enable-scoring-cache --max-multi-item-seq-len 2048 \
     --multi-item-mask-impl dense --multi-item-segment-fallback-threshold 0 \
     --multi-item-enable-prefill-extend --multi-item-extend-batch-size 128 \
     --disable-overlap-schedule --multi-item-enable-score-from-cache-v2 \
     --multi-item-score-from-cache-v2-items-per-step 64 \
     --multi-item-score-label-only-logprob --multi-item-score-fastpath-log-metrics &
   ```
3. Wait for the server to be ready.
4. Run the benchmark client: `python benchmark.py`
5. Record the RPS and IPS, then kill the server.

---

## Overview of Code Changes

### 1. Server Arguments & Configuration (`sgl_jax/srt/server_args.py`)
- **Remove CLI arguments & fields:**
  - `--multi-item-scoring-delimiter`
  - `--max-multi-item-seq-len`
  - `--multi-item-scoring-chunk-size`
  - `--multi-item-mask-impl`
  - `--multi-item-segment-fallback-threshold`
  - `--multi-item-enable-prefill-extend` and `--multi-item-enable-score-from-cache-v2` (they will become the default/only behavior)
- **Keep:** 
  - `--max-multi-item-count` (can still be useful to limit maximum payload sizes).
- **Remove validation logic** inside `check_server_args` related to these removed flags. Ensure Fast Path V2 is active by default.

### 2. Request Data Structures (`sgl_jax/srt/managers/io_struct.py`)
- **Remove properties from `GenerateReqInput` and related base classes:**
  - `is_multi_item_scoring`
  - `multi_item_scoring_delimiter`
  - `multi_item_algorithm`
  - `multi_item_mask_mode`
- **Clean up `_normalize_multi_item_params`** to remove normalization logic for these properties.

### 3. Request Routing & Execution (`sgl_jax/srt/managers/tokenizer_manager.py`)
- **Remove delimiter validation:** Delete `_validate_multi_item_delimiter_token()`.
- **Remove packing logic:** Delete `_build_multi_item_token_sequence()`.
- **Update `score_request()`:** 
  - Remove the legacy path that packs the query and items into a single `GenerateReqInput` block.
  - Enforce routing to `score_prefill_extend()` directly when multiple items are requested.

### 4. Scheduler Execution (`sgl_jax/srt/managers/scheduler.py` & `schedule_batch.py`)
- **Remove Scheduler Propagation:**
  - Remove `tokenizer_multi_item_packed` metrics from `ingress_score_paths`.
  - Remove assignment of `is_multi_item_scoring` and `multi_item_scoring_delimiter` in `_unpack_reqs`.
- **Remove Batching Logic (`schedule_batch.py`):**
  - Remove `is_multi_item_scoring` and `multi_item_scoring_delimiter` from `Req` initialization and `ForwardBatch`.
  - Delete `_build_multi_item_extend_positions()` and any logic that resets RoPE positions based on delimiters in `_build_extend_positions_for_req()`.

### 5. Custom Output Extraction (`sgl_jax/srt/managers/scheduler_output_processor_mixin.py`)
- **Remove the delimiter-based score extraction logic** from `process_output()`. The Fast Path V2 uses `_run_score_from_cache_v2_chunk` to extract its own probabilities, making this code dead.

### 6. Attention & Kernels (`sgl_jax/srt/layers/attention/flashattention_backend.py` & `ragged_paged_attention.py`)
- **Remove Custom Attention Modes:**
  - Delete `MULTI_ITEM_MASK_MODE_CAUSAL`, `MULTI_ITEM_MASK_MODE_DENSE`, `MULTI_ITEM_MASK_MODE_SEGMENT`.
- **Clean up `FlashAttentionMetadata`:**
  - Remove `multi_item_prefix_end`, `multi_item_row_seg_starts`, and `multi_item_mask_mode`.
- **Remove Custom Mask Builders:**
  - Delete `_calculate_row_seg_starts_jit()`, `_build_multi_item_attention_mask_jit()`, `_build_multi_item_attention_mask()`, and `_build_multi_item_segment_layout()`.
  - Simplify `get_forward_metadata()` by removing the segment mask vs. dense mask branches.
- **Update Kernel Support:**
  - Run `git checkout main -- python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py` to restore the clean kernel without the `multi_item_mask_mode` parameters.

### 7. Request Validation (`sgl_jax/srt/validation/score_validation.py`)
- Simplify `validate_multi_item_scoring_request()`:
  - Remove checks for `delimiter_collision`.
  - Remove checks for `max_total_seq_len` and the `enforce_total_seq_len` argument.

### 8. Testing & Scripts
- **Delete legacy tests:**
  - `test_multi_item_segment_mask.py`
  - `test_multi_item_positions.py`
  - `test_multi_item_scheduler_output.py`
  - `test_multi_item_chunking.py`
- **Update Benchmark / Regression Tests:**
  - Remove the packed approach fallback and delimiter arguments from `test_bench_multi_item_score.py`, `test_multi_item_regression.py`, and `benchmark_score_matrix.py`.

### 9. Documentation
- Update `docs/features/multi_item_scoring.md` to reflect that the packed approach is removed and that fast-path V2 (Prefill + Extend) is the sole backend implementation.

---

## 10. Testing Step (After Edit)
1. Start the server using the **updated** command (removed deprecated flags):
   ```bash
   python -m sgl_jax.launch_server \
     --model-path Qwen/Qwen3-0.6B --trust-remote-code --host 0.0.0.0 --port 30000 \
     --device tpu --tp-size 1 --nnodes 1 --log-level info --node-rank 0 \
     --dist-init-addr 0.0.0.0:10011 --dtype bfloat16 \
     --download-dir /home/aashishrampal_google_com/extra_storage/tmp \
     --mem-fraction-static 0.7 --max-prefill-tokens 2048 --chunked-prefill-size -1 \
     --precompile-token-paddings 1024 2048 4096 8192 16384 32768 \
     --precompile-bs-paddings 1 2 4 8 12 16 24 32 48 64 \
     --max-running-requests 128 --page-size 64 --attention-backend fa \
     --skip-server-warmup --enable-scoring-cache \
     --multi-item-extend-batch-size 128 --disable-overlap-schedule \
     --multi-item-score-from-cache-v2-items-per-step 64 \
     --multi-item-score-label-only-logprob --multi-item-score-fastpath-log-metrics &
   ```
2. Wait for the server to be ready.
3. Run the benchmark client: `python benchmark.py`
4. Verify the server processes the requests correctly and the RPS/IPS metrics are comparable or better.
5. Kill the server.

## Execution on Plan, Log & Learnings
- **[2026-03-18] Baseline Test Complete**: Successfully ran the benchmark using the legacy parameters. Results showed RPS: 7.80, IPS: 623.85. Server correctly hit the Fast Path V2 under the hood.
- **[2026-03-18] Step 1 Complete**: Modified `sgl_jax/srt/server_args.py`. 
  - *Learnings/Corrections*: While pruning arguments, it's critical to distinguish between flags exclusively meant for the 1D packed approach (e.g., `--multi-item-scoring-delimiter`, `--multi-item-mask-impl`, `--multi-item-scoring-chunk-size`) and flags used to configure the Fast Path V2 (e.g., `--multi-item-extend-batch-size`, `--multi-item-score-from-cache-v2-items-per-step`). I successfully stripped the former while preserving the latter, ensuring Fast Path V2 tuning logic remains intact. I also flipped the `--multi-item-enable-prefill-extend` and `--multi-item-enable-score-from-cache-v2` flags to default to `True`.
- **[2026-03-18] Step 2 Complete**: Modified `sgl_jax/srt/managers/io_struct.py` to strip all legacy `is_multi_item_scoring`, `multi_item_scoring_delimiter`, `multi_item_algorithm`, and `multi_item_mask_mode` properties from `GenerateReqInput` and `EmbeddingReqInput`, including their initialization parameters.
- **[2026-03-18] Step 3 Complete**: Modified `sgl_jax/srt/managers/tokenizer_manager.py` to remove legacy delimiter validation, packed sequence building, and routing. `score_request` now directly invokes `score_prefill_extend` (Fast Path V2) for all multi-item workloads without using a 1D packed sequence fallback.
- **[2026-03-18] Step 4 Complete**: Removed legacy `is_multi_item_scoring` properties, propagation flags, and metric counters from `scheduler.py` and `schedule_batch.py`. Replaced the complicated `_build_multi_item_extend_positions()` delimiter RoPE reset logic with the standard `np.arange` causal position logic.
- **[2026-03-18] Step 5 Complete**: Modified `sgl_jax/srt/managers/scheduler_output_processor_mixin.py` to completely remove the logic around `is_multi_item_scoring` and `delimiter_mask` slicing from `_process_logprobs`.
- **[2026-03-18] Step 6 Complete**: Modified `sgl_jax/srt/layers/attention/flashattention_backend.py` to remove `MULTI_ITEM_MASK_MODE_*` constants, `multi_item_prefix_end`, `multi_item_row_seg_starts`, and all custom mask builder logic. Cleaned up `get_forward_metadata` to no longer build these masks and updated `FlashAttention.forward` to stop propagating them to the `ragged_paged_attention` kernel. Successfully checked out `ragged_paged_attention.py` from the `main` branch to remove its associated kernel logic.
- **[2026-03-18] Step 7 Complete**: Modified `sgl_jax/srt/validation/score_validation.py` to completely remove `validate_multi_item_scoring_request`, as the Fast Path V2 no longer concatenates items into a single sequence that requires constraint checking (like max sequence length or delimiter collisions). Cleaned up associated imports in `tokenizer_manager.py` and `__init__.py`.
- **[Current Step]**: Proceeding to Step 8: Deleting legacy tests (`test_multi_item_segment_mask.py`, etc.) and updating benchmarks to remove delimiter arguments.