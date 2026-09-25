"""V4 capacity planning uses exactly the shapes allocated by the resource pools."""

import math
from dataclasses import dataclass

from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4TokenToKVPool
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, ReqToTokenPool


@dataclass(frozen=True)
class DeepseekV4PoolBudget:
    max_num_reqs: int
    history_tokens: int  # global usable capacity across DP ranks
    swa_tokens: int
    state_bytes_per_device: int
    kv_bytes_per_device: int
    available_bytes_per_device: int

    @property
    def allocated_bytes_per_device(self):
        return self.state_bytes_per_device + self.kv_bytes_per_device


def plan_deepseek_v4_pools(
    spec,
    available_bytes,
    max_num_reqs,
    page_size,
    dp_size=1,
    swa_full_tokens_ratio=0.8,
    max_total_tokens=None,
):
    """Fit pools within post-weight, post-temporary-reservation device bytes.

    All TP/EP replicas consume the same per-device bytes. The global request
    pool is not DP partitioned, so *each* DP shard stores max_num_reqs+1 state
    positions. Both KV families reserve one page per rank; state reserves one
    padding position per rank. The caller already subtracts mem_fraction_static
    execution headroom; it must not pass total device HBM as available_bytes.
    max_total_tokens follows the existing runner's per-DP-rank cap convention.
    """
    if page_size not in (128, 256) or dp_size <= 0:
        raise ValueError("V4 requires page_size 128/256 and positive dp_size")
    if not math.isfinite(swa_full_tokens_ratio) or swa_full_tokens_ratio <= 0:
        raise ValueError("SWA/history capacity ratio must be finite and positive")
    state_cell = spec.state_bytes_per_request
    history_page = spec.history_bytes_per_page(page_size)
    swa_page = spec.swa_bytes_per_token * page_size
    # Reserve up to a quarter of the resource budget for state by default,
    # avoiding a thousands-of-requests default that cannot fit offline C128.
    if max_num_reqs is None:
        max_num_reqs = min(2048, max(dp_size, available_bytes // max(1, 4 * state_cell)))
        max_num_reqs = max_num_reqs // dp_size * dp_size
    if max_num_reqs <= 0:
        raise ValueError("max_num_reqs must be positive")
    state_bytes = (max_num_reqs + 1) * state_cell
    kv_budget = available_bytes - state_bytes

    def size_for(pages):
        swa_pages = max(1, math.ceil(pages * swa_full_tokens_ratio))
        return swa_pages, (pages + 1) * history_page + (swa_pages + 1) * swa_page

    lo, hi = 0, max(0, kv_budget // max(1, history_page))
    # Locations are int32 in req_to_token and runtime metadata.
    hi = min(hi, (2**31 - 1) // page_size - 1)
    if max_total_tokens is not None:
        hi = min(hi, max_total_tokens // page_size)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if size_for(mid)[1] <= kv_budget:
            lo = mid
        else:
            hi = mid - 1
    if lo < 1:
        raise ValueError("V4 budget cannot fit request state, padding and one history/SWA page")
    swa_pages, kv_bytes = size_for(lo)
    if (swa_pages + 1) * page_size > 2**31 - 1:
        raise ValueError("V4 SWA capacity exceeds int32 address space")
    return DeepseekV4PoolBudget(
        max_num_reqs,
        lo * page_size * dp_size,
        swa_pages * page_size * dp_size,
        state_bytes,
        kv_bytes,
        available_bytes,
    )


def build_deepseek_v4_pools(
    spec, budget, page_size, mesh, max_context_len, dp_size=1, req_pool=None
):
    if req_pool is None:
        req_pool = ReqToTokenPool(budget.max_num_reqs, max_context_len)
    elif req_pool.size != budget.max_num_reqs:
        raise ValueError("existing request pool size differs from V4 state budget")
    kv = DeepseekV4TokenToKVPool(
        budget.history_tokens,
        budget.swa_tokens,
        page_size,
        spec,
        mesh,
        dp_size,
    )
    state = DeepseekV4CompressStatePool(budget.max_num_reqs, spec, mesh, dp_size)
    pools = MemoryPools(token_to_kv_pool=kv, compressor_state_pool=state)
    allocator = DeepseekV4TokenToKVPoolAllocator(kv)
    if (kv.nbytes + state.nbytes) // dp_size != budget.allocated_bytes_per_device:
        raise ValueError("V4 allocated array shapes differ from the capacity budget")
    return req_pool, pools, allocator
