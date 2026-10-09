"""R's serving glue for the C pools and A/B functional attention contract.

Pool and dummy/update contracts are adapted from epic/dsv4 ce1ebb637.
Numerical execution and ownership derivation remain in A/B and C.
"""

import copy
import os

import numpy as np


def validate_runtime_config(args, *, dp_size=1, is_draft=False):
    if dp_size != 1:
        raise ValueError("V4 Flash runtime currently requires --dp-size 1")
    if is_draft or getattr(args, "speculative_algorithm", None):
        raise ValueError("V4 Flash runtime does not support speculative decoding or MTP")
    if not args.disable_radix_cache:
        raise ValueError("V4 Flash runtime requires --disable-radix-cache (no prefix reuse)")
    if args.page_size not in (128, 256):
        raise ValueError("V4 requires original-token --page-size 128 or 256")
    if args.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
        raise ValueError("V4 Flash runtime requires BF16 KV/indexer storage")
    if getattr(args, "disaggregation_mode", "null") != "null" or getattr(
        args, "pd_disaggregation", ""
    ):
        raise ValueError("V4 Flash runtime does not support PD disaggregation")
    if getattr(args, "enable_lora", False) or getattr(args, "enable_static_lora", False):
        raise ValueError("V4 Flash runtime has no LoRA adapter")
    if getattr(args, "ep_dispatch_algorithm", None) or getattr(args, "ep_num_redundant_experts", 0):
        raise ValueError("V4 Flash runtime requires identity expert placement")


def prepare_dummy_batch(batch, request_capacity):
    """Compile inactive rows without touching C's live allocation ledger."""
    batch.seq_lens = np.zeros_like(batch.seq_lens, dtype=np.int32)
    batch.req_pool_indices = np.full_like(batch.req_pool_indices, request_capacity, dtype=np.int32)
    batch.out_cache_loc = np.full_like(batch.out_cache_loc, -1, dtype=np.int32)
    batch.positions = np.zeros_like(batch.positions, dtype=np.int32)
    batch.cache_loc = np.zeros_like(batch.cache_loc, dtype=np.int32)
    if batch.extend_seq_lens is not None:
        batch.extend_seq_lens = np.zeros_like(batch.extend_seq_lens, dtype=np.int32)
        batch.extend_prefix_lens = np.zeros_like(batch.extend_prefix_lens, dtype=np.int32)


def validate_pool_updates(memory_pools, updates):
    """Validate before dispatch can donate either complete V4 owner."""
    from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4TokenToKVPool

    if not isinstance(memory_pools.token_to_kv_pool, DeepseekV4TokenToKVPool):
        return
    if not isinstance(updates, dict) or set(updates) != set(memory_pools._pools):
        raise ValueError("V4 updates must exactly match both MemoryPools owner keys")
    for name, owner in memory_pools._pools.items():
        owner.validate_buffer_updates(updates[name])


def precompile_capacity_variants(backend, mode, batch_size, history_tokens):
    """Cover B's total-history and per-request shapes, including chunk continuation.

    A context ladder alone misses multi-request EXTEND combinations: CSA sizes
    gathered history by the batch total, while tuned HCA uses its longest request.
    Keep those dimensions independent, pruning impossible count intervals. DECODE
    with page size 128 uses request-local CSA and correlated per-request buckets.
    """
    from sgl_jax.srt.layers.attention.deepseek_v4_backend import capacity_bucket

    context = min(backend.max_context_len, history_tokens)
    total = min(batch_size * context, history_tokens)
    local_decode = mode.is_decode() and backend.page_size == 128

    def buckets(count):
        return [1 << power for power in range(7, capacity_bucket(count).bit_length())]

    def minimum(capacity):
        return 0 if capacity == 128 else capacity // 2 + 1

    hca_total = not backend.use_pallas_hca or os.environ.get("DSV4_HCA_TILE_BUCKET_SUM", "0") == "1"
    if local_decode and not hca_total:
        # Boundaries are aligned at powers of two, so this union reaches every
        # pair of capacity buckets as a single request's history grows.
        points = {0, context}
        for ratio in (4, 128):
            points.update((cap // 2 + 1) * ratio for cap in buckets(context // ratio))
        yield from sorted(
            {
                (capacity_bucket(n // 4), 128, capacity_bucket(n // 128))
                for n in points
                if n <= context
            }
        )
        return

    c128_limit = total if hca_total else context
    for c4 in buckets((context if local_decode else total) // 4):
        for c128 in buckets(c128_limit // 128):
            # Every completed C128 record needs 32 completed C4 groups. Up to
            # 31 trailing C4 records per request have no C128 counterpart.
            max4 = 32 * c128 * (1 if hca_total else batch_size) + 31 * batch_size
            if local_decode:
                if c4 * batch_size < 32 * minimum(c128) or minimum(c4) > max4:
                    continue
            elif c4 < 32 * minimum(c128) or minimum(c4) > max4:
                continue
            yield c4, c128 if hca_total else 128, c128


def reclaim_batch_swa(batch, tree_cache):
    """Call C after completion, using submitted lengths even during overlap.

    The shared Req can already describe the next submitted decode. Only the completed
    batch's snapshot permits reclamation; copy its length into a temporary view and
    carry back C's reclamation cursor without rolling back the live request length.
    """
    from sgl_jax.srt.mem_cache.chunk_cache import DeepseekV4ChunkCache
    from sgl_jax.srt.mem_cache.common import reclaim_completed_v4_swa

    if not isinstance(tree_cache, DeepseekV4ChunkCache):
        return
    for info in batch.reqs_info:
        lengths = () if info.seq_lens is None else info.seq_lens
        for req, completed_len in zip(info.reqs or (), lengths, strict=True):
            if req.req_pool_idx is None or req.finished() or req.is_retracted:
                continue
            completed = copy.copy(req)
            completed.kv_committed_len = int(completed_len)
            reclaim_completed_v4_swa(completed, tree_cache)
            req.swa_evicted_seqlen = completed.swa_evicted_seqlen
