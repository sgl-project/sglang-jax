# MoE Split Weight Loading Optimization

## Problem

`_create_stacked_split_moe_lazy_tensor` loads Grok2 MoE weights slowly (2m34s for 192 groups).
Each expert's TP shards are read serially within `_load_single_expert_slice` (up to 8 GCS reads per expert).

## Approved Approach

Restructure `_load_stacked_slice` callback to parallelize all (expert, tp_shard) reads in one batch:

1. Build all read tasks: unique_experts x tp_shards
2. Submit all tasks to ThreadPoolExecutor concurrently
3. After all I/O completes, group by expert, concat, transpose, fill output array

## Key Constraints

- Keep lazy loading via `jax.make_array_from_callback` (option A)
- Minimize memory footprint (no external prefetch cache)
- Single host, GCS storage

## Expected Result

From ~512 serial GCS reads to ~512 concurrent reads per callback invocation.
