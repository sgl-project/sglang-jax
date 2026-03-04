# DP Logprobs Fix - Implementation Summary

## Problem

In DP mode, `get_top_logprobs` and `get_token_ids_logprobs` crashed with:
```
TypeError: Cannot concatenate arrays with different numbers of dimensions:
got (1, 7, 2), (1, 0), (1, 0), ...
```

**Root cause**: The functions iterated over all DP ranks (including padding slots with `pruned_len=0`), appending empty lists `[]` for padding. When `jnp.array()` tried to stack these, real entries had shape `(pruned_len, k)` (3D) while padding entries had shape `(0,)` (1D), causing dimension mismatch.

## Solution

Refactored to return **flat tensors** instead of per-request nested structures, following tpu-inference's design pattern:

### Key Changes

1. **`get_top_logprobs`** (logits_processor.py:463)
   - **Before**: Iterated per-request, built nested list, called `jnp.array()` → crash on ragged shapes
   - **After**: Returns flat tensors `(values, indices)` with shape `[total_pruned_tokens, max_k]`
   - Consumer slices using `logprob_pt` offset and truncates to per-request `k`

2. **`get_token_ids_logprobs`** (logits_processor.py:442)
   - **Before**: Nested per-request structure → `jnp.array()` crash
   - **After**: Returns flat Python lists, one element per pruned token
   - Consumer slices using `logprob_pt` offset

3. **Consumer** (scheduler_output_processor_mixin.py:468)
   - **Before**: `output.input_top_logprobs_val[i]` — indexed by request `i`
   - **After**: `output.input_top_logprobs_val[logprob_pt:logprob_pt+num_input_logprobs]` — sliced by token offset
   - Truncates each row from `max_k` to `req.top_logprobs_num`

4. **Gather logic** (scheduler.py:573)
   - Removed gather for `input_token_ids_logprobs` (already gathered per-element via `out_sharding`)

5. **tolist conversion** (tp_worker.py:662)
   - Updated to handle flat list of JAX arrays for `input_token_ids_logprobs`

## Benefits

- **Simpler logic**: No per-request iteration in logits_processor, just return raw top_k output
- **DP-safe**: Padding slots are skipped during pruning, so flat tensor only contains real data
- **Consistent with tpu-inference**: Same flat-tensor-on-device + CPU-side-slicing pattern
- **No shape mismatches**: All rows have uniform `max_k` columns

## Testing

Run the original failing test case to verify the crash is fixed:
```bash
# Via sglang-jax-skypilot-dev skill on TPU cluster
uv run --extra tpu python -m pytest test/srt/<test_file> -v
```

## Files Modified

- `python/sgl_jax/srt/layers/logits_processor.py` - Simplified return format
- `python/sgl_jax/srt/managers/tp_worker.py` - Updated tolist conversion
- `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py` - Updated consumer to slice flat tensors
- `python/sgl_jax/srt/managers/scheduler.py` - Removed unnecessary gather
