import logging

from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)

# TODO @pc we should separate all mem cache from schedule batch to support more flexible operations


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
    dp_rank: int = 0,
):
    allocator = tree_cache.token_to_kv_pool_allocator

    evict_from_tree_cache(tree_cache, num_tokens, dp_rank=dp_rank)
    if backup_state:
        state = allocator.backup_state()
    out_cache_loc = allocator.alloc(num_tokens, dp_rank=dp_rank)
    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache=tree_cache, dp_rank=dp_rank)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)
    if backup_state:
        return out_cache_loc, state
    else:
        return out_cache_loc


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: list[int],
    seq_lens: list[int],
    last_loc: list[int],
    extend_num_tokens: int,
    backup_state: bool = False,
    dp_rank: int = 0,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens, dp_rank=dp_rank)
    if backup_state:
        state = allocator.backup_state()
    out_cache_loc = allocator.alloc_extend(
        prefix_lens, seq_lens, last_loc, extend_num_tokens, dp_rank=dp_rank
    )
    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache=tree_cache, dp_rank=dp_rank)}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    if backup_state:
        return out_cache_loc, state
    else:
        return out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int, dp_rank: int = 0):
    if tree_cache is None:
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    # Check if this is a hybrid allocator
    if hasattr(allocator, "full_available_size"):
        # Hybrid allocator
        full_available_size = allocator.full_available_size(dp_rank=dp_rank)
        swa_available_size = allocator.swa_available_size(dp_rank=dp_rank)

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(full_num_tokens, swa_num_tokens, dp_rank=dp_rank)
    else:
        # Standard allocator
        if allocator.available_size(dp_rank=dp_rank) < num_tokens:
            tree_cache.evict(num_tokens, dp_rank=dp_rank)


def available_and_evictable_str(tree_cache, dp_rank: int = 0) -> str:
    token_to_kv_pool_allocator = tree_cache.token_to_kv_pool_allocator
    # if isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
    # not support SWA yet currently , hack this branch
    if False:
        full_available_size = token_to_kv_pool_allocator.full_available_size(dp_rank=dp_rank)
        swa_available_size = token_to_kv_pool_allocator.swa_available_size(dp_rank=dp_rank)
        full_evictable_size = tree_cache.full_evictable_size(dp_rank=dp_rank)
        swa_evictable_size = tree_cache.swa_evictable_size(dp_rank=dp_rank)
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
            f"Full LRU list evictable size: {tree_cache.full_lru_list_evictable_size()}\n"
            f"SWA LRU list evictable size: {tree_cache.swa_lru_list_evictable_size()}\n"
        )
    else:
        available_size = token_to_kv_pool_allocator.available_size(dp_rank=dp_rank)
        evictable_size = tree_cache.evictable_size(dp_rank=dp_rank)
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
