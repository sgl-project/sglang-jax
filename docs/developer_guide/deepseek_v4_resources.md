# DeepSeek V4 cache and compressor-state resources (task C)

This document specifies the resource interfaces supplied to the V4 attention
backend and runtime by [task C](https://github.com/sgl-project/sglang-jax/issues/1688).
The implementation is based on the resource code in
[`primatrix/sglang-jax` `epic/dsv4`](https://github.com/primatrix/sglang-jax/pull/345).
It covers the Flash backbone, BF16 KV/indexer storage, FP32 compressor state,
and request-owned cache without cross-request prefix reuse.

## Layout and address units

Let `P` be the original-token page size (128 or 256), `D` the KV head dimension,
`Di` the indexer dimension, `G` the usable history pages per DP rank, `S` the
usable SWA token slots per DP rank, and `R` the global request-slot capacity.
The buffers below are per layer and DP rank. Global arrays concatenate DP ranks
on the leading axis and shard that axis over mesh `data`; their feature axes
are replicated over TP/EP.

| BF16 KV family | Per-rank shape | Owner |
| --- | --- | --- |
| `swa` | `[S + P, D]` | Every backbone layer |
| `c4` | `[G + 1, P/4, D]` | Ratio-4 layers |
| `indexer` | `[G + 1, P/4, Di]` | Ratio-4 layers |
| `c128` | `[G + 1, 1, P/128, D]` | Ratio-128 layers, native HCA layout |

| FP32 state family | Per-rank shape | Owner |
| --- | --- | --- |
| `c4` | `[R + 1, 8, 4D]` | CSA KV compressor |
| `indexer` | `[R + 1, 8, 4Di]` | CSA index compressor |
| `c128` | `[R + 1, 128, 2, D]` | HCA compressor, native layout |

The ratio schedule is `config.compress_ratios[:config.num_hidden_layers]`:
zero selects SWA-only, four selects CSA, and 128 selects HCA. The extra KV
page numbered zero is padding on every DP rank. The extra state position
numbered `R` is padding; request slot zero is valid. There is no full
uncompressed-history KV tensor and no second request-state allocator.

The allocator returns **rank-local original-token locations** beginning at
`P`. One history page owns the corresponding ratio-4 KV, indexer, and
ratio-128 KV records. A completed record's flattened location is `loc // 4`
or `loc // 128`. A history page contains `P/4` CSA records or `P/128` HCA
records. `swa` has a separate physical page pool; its location is found through
`full_to_swa_index_mapping`. Zero in that mapping means unmapped or padding.
Allocating a page does not make its compressed entries numerically valid: B
must bound reads and writes by completed groups and valid-entry masks.

`DSV4_HCA_NATIVE_LAYOUT` defaults to enabled. The optional flat layout uses
`[G + 1, P/128, D]` for HCA KV and `[R + 1, 128, 2D]` for its state, with the
same byte count. Native layout avoids whole-pool relayouts at each HCA layer.

## Capacity and allocation

`DeepseekV4CacheSpec` reports the bytes per original-token history page,
per SWA token, and per request-state slot. `plan_deepseek_v4_pools` sizes
history and SWA separately from an available **per-device** budget after
model weights and execution headroom. The plan includes BF16 KV, FP32 state,
page rounding, page-zero padding, state padding, and TP/EP replication.
`build_deepseek_v4_pools` constructs `ReqToTokenPool`, `MemoryPools`, and the
allocator, and rejects a mismatch between planned and allocated array bytes.

`DeepseekV4TokenToKVPoolAllocator` holds one original-token history ledger
shared by compressed families and an independent SWA free list. Its
`estimate_extend`/`estimate_decode` methods use the same planner as
`alloc_extend`/`alloc_decode`; `can_allocate` checks both demands. Allocation
validates the full batch and both capacities before changing either ledger or
the mapping. Insufficient capacity returns `None` with no partial allocation.
`backup_state`/`restore_state` cover allocator ledgers and mappings; R must
include its request-slot and host-length changes in any wider transaction.

History release requires the complete owned extent of every page being freed.
`free_swa` accepts whole consumed SWA pages, clears their mappings, and keeps
compressed history owned. Grouped release defers page reclamation until the
group is complete. Raw locations are not generation handles: a caller must
drop the old request owner before the pages can belong to another request.

## Request-owned state and lifecycle

Each DP shard stores state for every global `ReqToTokenPool` slot because the
request free list is not DP-partitioned. CSA state uses an eight-position ring
addressed by `position % 8`. The first half of the C4/indexer state feature
axis is content and the second half is score. Native HCA state uses index zero
of its packing axis for content and index one for score. Empty content is zero
and empty scores are negative infinity.

`DeepseekV4ChunkCache` keeps a live request's original-token mapping and
compressor state across prefill chunks. It returns only that same request's
committed prefix and does not enable cross-request prefix reuse.
`cache_unfinished_req` retains continuation addresses. `release_req` frees the
request's **entire allocated extent**, including an allocated tail sharing a
page with committed tokens, clears its mapping, and returns its request slot.
Completion, cancellation, and retraction use this same release contract;
retracted requests recompute from position zero.

`reclaim_completed_swa(req)` may run only **after** the forward producing
`req.kv_committed_len` has completed. For completed length `L` and window `W`,
it releases whole SWA pages before `floor(max(0, L-W+1)/P) * P`, preserving the
page needed by the next query. R supplies the execution-completion boundary
and invokes this C interface at the safe point. Releasing SWA does not reset
compressor state or release compressed history.

Releasing a request drops ownership, not the old device-state contents. On
the next zero-prefix forward, B must initialize a new or recycled slot before
its first numerical use; C's `DeepseekV4CompressStatePool.reset` defines the
empty representation. A continuing chunk must not reset the slot. Padding and
invalid requests map to the sentinel position and cannot write a live slot.

## Backend and runtime handoff

The KV pool implements the `KVCache` resource contract and exposes
`get_swa_buffer`, `get_compressed_buffer`, `get_compressed_page_size`, and
`get_indexer_buffer`, routed by layer. The state pool exposes its per-family
buffers and request-slot indices. Both owners are PyTrees. Operators receive
arrays and metadata, not allocator objects.

B prepares read/page tables, masks, positions, and write locations from C's
address and ownership mappings. B packages semantic per-layer updates using
`swa`, `compressed`, `indexer`, and `compressor`. Each pool's
`build_buffer_updates` merges these into a complete family payload without
mutating the owner and checks shape/dtype. R commits both complete owners via
`MemoryPools.replace_all(...)` after the forward. C does not select the
runtime scheduling mode or perform that commit.

Focused resource tests cover pool layout and updates, budget consistency,
allocator atomicity and rollback, DP isolation, and request continuation,
reclamation, release, and slot reuse. R owns scheduler/runner wiring and
end-to-end validation of overlap and mixed-batch combinations.
