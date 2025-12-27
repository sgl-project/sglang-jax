from __future__ import annotations

import os
import random
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

if TYPE_CHECKING:
    from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

import logging

logger = logging.getLogger(__name__)

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS_ESTIMATION = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


IGNORE_EOS_RESERVE_TOKENS = 1


class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""

    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""

    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"


class SchedulePolicy:
    Policy = CacheAwarePolicy | CacheAgnosticPolicy

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
    ):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
        )

    def calc_priority(self, waiting_queue: list[Req]) -> bool:
        if self.policy == CacheAgnosticPolicy.FCFS:
            # A shortcut for FCFS
            return False

        policy = self._determine_active_policy(waiting_queue)

        prefix_computed = False
        if isinstance(policy, CacheAwarePolicy):
            prefix_computed = True
            temporary_deprioritized = self._compute_prefix_matches(waiting_queue, policy)
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(waiting_queue, temporary_deprioritized)
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(waiting_queue)
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")

        return prefix_computed

    def _determine_active_policy(self, waiting_queue: list[Req]) -> Policy:
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(self, policy: str, tree_cache: BasePrefixCache) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if getattr(tree_cache, "disable", True):
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            try:
                return CacheAgnosticPolicy(policy)
            except ValueError as inner_err:
                raise ValueError(f"Unknown schedule_policy: {policy=}") from inner_err

    def _compute_prefix_matches(
        self, waiting_queue: list[Req], policy: CacheAwarePolicy
    ) -> set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        temporary_deprioritized: set[int] = set()
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.adjust_max_prefix_ids()
            extra_key = r.extra_key

            # NOTE: the prefix_indices must always be aligned with last_node
            r.prefix_indices, r.last_node, r.last_host_node, r.host_hit_length = (
                self.tree_cache.match_prefix(
                    rid=r.rid,
                    key=RadixKey(token_ids=prefix_ids, extra_key=extra_key, dp_rank=r.dp_rank),
                )
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                in_batch_matching_prefixes, _, _, _ = self.waiting_queue_radix_tree.match_prefix(
                    rid=r.rid,
                    key=RadixKey(token_ids=prefix_ids, extra_key=extra_key, dp_rank=r.dp_rank),
                )
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(
                        RadixKey(token_ids=prefix_ids, extra_key=extra_key, dp_rank=r.dp_rank),
                        np.empty(len(prefix_ids), dtype=np.bool_),
                    )
        return temporary_deprioritized

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: list[Req], temporary_deprioritized: set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -len(r.prefix_indices) if r.rid not in temporary_deprioritized else float("inf")
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(waiting_queue: list[Req], tree_cache: BasePrefixCache) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)

        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )

    @staticmethod
    def _sort_by_longest_output(waiting_queue: list[Req]) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens)."""
        waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: list[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        random.shuffle(waiting_queue)

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: dict[TreeNode, int]) -> None:
        for child in cur_node.children.values():
            SchedulePolicy._calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    @staticmethod
    def _get_dfs_priority(
        cur_node: TreeNode,
        node_to_priority: dict[TreeNode, int],
        last_node_to_reqs: dict[TreeNode, list[Req]],
        q: list,
    ) -> None:
        childs = [child for child in cur_node.children.values()]
        childs.sort(key=lambda x: -node_to_priority[x])
        for child in childs:
            SchedulePolicy._get_dfs_priority(child, node_to_priority, last_node_to_reqs, q)
        q.extend(last_node_to_reqs[cur_node])


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests


class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: int | None,
        mixed_with_decode_tokens: int = 0,
        dp_size: int = 1,
    ):
        self.page_size = page_size
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.dp_size = dp_size
        self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= mixed_with_decode_tokens
            self.rem_chunk_tokens_list = [self.rem_chunk_tokens] * dp_size
        else:
            self.rem_chunk_tokens_list = None

        # Per-DP token offsets
        self.rem_total_token_offset = [mixed_with_decode_tokens] * dp_size
        self.cur_rem_token_offset = [mixed_with_decode_tokens] * dp_size

        self.req_states = {i: None for i in range(dp_size)}  # Per-DP request states
        self.can_run_list = {i: [] for i in range(dp_size)}  # Per-DP request lists
        self.new_chunked_reqs = [None] * dp_size
        self.log_hit_tokens = 0
        self.log_input_tokens = 0

        if running_batch is not None:
            # Calculate token offset per DP rank
            for dp_rank in range(dp_size):
                info = running_batch.reqs_info[dp_rank]
                if info.reqs:
                    self.rem_total_token_offset[dp_rank] += sum(
                        [
                            min(
                                (r.sampling_params.max_new_tokens - len(r.output_ids)),
                                CLIP_MAX_NEW_TOKENS_ESTIMATION,
                            )
                            * self.new_token_ratio
                            for r in info.reqs
                        ]
                    )

        self.is_hybrid = isinstance(self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)

    def rem_total_tokens_for_dp(self, dp_rank: int) -> int:
        """Calculate remaining total tokens for a specific DP rank.

        Args:
            dp_rank: DP rank to calculate for

        Returns:
            Available tokens minus total token offset
        """
        if self.is_hybrid:
            available_and_evictable = min(
                self.token_to_kv_pool_allocator.full_available_size(dp_rank=dp_rank)
                + self.tree_cache.full_evictable_size(),
                self.token_to_kv_pool_allocator.swa_available_size(dp_rank=dp_rank)
                + self.tree_cache.swa_evictable_size(),
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size(dp_rank=dp_rank)
                + self.tree_cache.evictable_size()
            )

        return available_and_evictable - self.rem_total_token_offset[dp_rank]

    def cur_rem_tokens_for_dp(self, dp_rank: int) -> int:
        """Calculate current remaining tokens for a specific DP rank.

        Args:
            dp_rank: DP rank to calculate for

        Returns:
            Available tokens minus current token offset
        """
        if self.is_hybrid:
            available_and_evictable = min(
                self.token_to_kv_pool_allocator.full_available_size(dp_rank=dp_rank)
                + self.tree_cache.full_evictable_size(),
                self.token_to_kv_pool_allocator.swa_available_size(dp_rank=dp_rank)
                + self.tree_cache.swa_evictable_size(),
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size(dp_rank=dp_rank)
                + self.tree_cache.evictable_size()
            )

        return available_and_evictable - self.cur_rem_token_offset[dp_rank]

    @property
    def rem_total_tokens(self):
        """Global remaining total tokens (minimum across all DP ranks).

        For backward compatibility and global checks.
        """
        return min(self.rem_total_tokens_for_dp(dp_rank) for dp_rank in range(self.dp_size))

    @property
    def cur_rem_tokens(self):
        """Global current remaining tokens (minimum across all DP ranks).

        For backward compatibility and global checks.
        """
        return min(self.cur_rem_tokens_for_dp(dp_rank) for dp_rank in range(self.dp_size))

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0 or (
            self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0
        ):
            return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def add_chunked_req(self, req: Req):
        dp_rank = req.dp_rank if req.dp_rank is not None else 0
        _rem_tokens = min(
            self.rem_chunk_tokens_list[dp_rank], int(self.rem_total_tokens_for_dp(dp_rank))
        )
        truncated = req.extend_input_len > _rem_tokens
        req.extend_input_len = min(req.extend_input_len, _rem_tokens)
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list[dp_rank].append(req)
        self._update_prefill_budget(
            0,
            req.extend_input_len,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION)
                if not truncated
                else 0
            ),
            dp_rank,
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    def _update_prefill_budget(
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int, dp_rank: int
    ):
        """Update prefill budget for a specific DP rank.

        Args:
            prefix_len: Matched prefix length
            extend_input_len: Input length to extend
            max_new_tokens: Maximum new tokens to generate
            dp_rank: DP rank being updated
        """
        extend_input_len = self.ceil_paged_tokens(extend_input_len)

        self.rem_total_token_offset[dp_rank] += extend_input_len + max_new_tokens
        self.cur_rem_token_offset[dp_rank] += extend_input_len
        self.rem_input_tokens -= extend_input_len
        if self.rem_chunk_tokens_list is not None:
            self.rem_chunk_tokens_list[dp_rank] -= extend_input_len

        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        if self.is_hybrid:
            try:
                swa_uuid_for_lock = self.tree_cache.inc_lock_ref(last_node)
                yield None
            finally:
                self.tree_cache.dec_lock_ref(last_node, swa_uuid_for_lock)
        else:
            try:
                self.tree_cache.inc_lock_ref(last_node)
                yield None
            finally:
                self.tree_cache.dec_lock_ref(last_node)

    def add_one_req_ignore_eos(self, req: Req):
        dp_rank = req.dp_rank if req.dp_rank is not None else 0
        if self.ceil_paged_tokens(req.extend_input_len) > min(
            self.cur_rem_tokens_for_dp(dp_rank), self.rem_total_tokens_for_dp(dp_rank)
        ):
            return AddReqResult.NO_TOKEN

        def add_req_state(r, insert_sort=False):
            new_token_ratio = 1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(r.output_ids)
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                self.req_states[dp_rank].append((tokens_left, tokens_occupied))
            else:
                i = 0
                for i in range(len(self.req_states[dp_rank])):
                    if tokens_left <= self.req_states[dp_rank][i][0]:
                        break
                self.req_states[dp_rank].insert(i, (tokens_left, tokens_occupied))

        if self.req_states[dp_rank] is None:
            self.req_states[dp_rank] = []
            add_req_state(req)
            if self.running_batch is not None:
                info = self.running_batch.reqs_info[dp_rank]
                if info.reqs:
                    for r in info.reqs:
                        add_req_state(r)
            if dp_rank in self.can_run_list and self.can_run_list[dp_rank]:
                for r in self.can_run_list[dp_rank]:
                    add_req_state(r)
            self.req_states[dp_rank].sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid:
            cur_rem_tokens = self.cur_rem_tokens_for_dp(dp_rank) - self.ceil_paged_tokens(
                req.extend_input_len
            )
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states[dp_rank]):
                # tokens_left gives a reservative calculation as the last token is not stored
                bs = len(self.req_states[dp_rank]) - i
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
                    return AddReqResult.NO_TOKEN
                tokens_freed += tokens_occupied

        if (
            self.rem_chunk_tokens_list is None  # chunked prefill is disabled
            or req.extend_input_len <= self.rem_chunk_tokens_list[dp_rank]  # it is the last chunk
        ):
            # Non-chunked prefill
            self.can_run_list[dp_rank].append(req)
            self._update_prefill_budget(
                0,
                req.extend_input_len,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION),
                dp_rank,
            )
        else:
            if self.rem_chunk_tokens_list[dp_rank] <= 0:
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens_list[dp_rank]

            req.extend_input_len = trunc_len
            req.fill_ids = req.fill_ids[:trunc_len]
            self.can_run_list[dp_rank].append(req)
            self.new_chunked_reqs[dp_rank] = req
            self._update_prefill_budget(0, trunc_len, 0, dp_rank)

        return self.budget_state()

    def add_one_req(self, req: Req):
        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        dp_rank = req.dp_rank if req.dp_rank is not None else 0
        total_tokens = req.extend_input_len + min(
            req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION
        )

        # adjusting the input_tokens based on host_hit_length and page_size
        real_input_tokens = req.extend_input_len - req.host_hit_length
        real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
        prefix_len = len(req.prefix_indices)

        if total_tokens >= self.rem_total_tokens_for_dp(dp_rank):
            return AddReqResult.NO_TOKEN

        total_can_run = sum(len(v) for v in self.can_run_list.values())
        if real_input_tokens >= self.rem_input_tokens and total_can_run != 0:
            return AddReqResult.OTHER

        with self._lock_node(req.last_node):
            # self.rem_total_tokens may decrease after the lock acquisition
            if total_tokens >= self.rem_total_tokens_for_dp(dp_rank):
                return AddReqResult.NO_TOKEN
            req.last_matched_prefix_len = prefix_len
            input_tokens = self.ceil_paged_tokens(req.extend_input_len)

            total_can_run = sum(len(v) for v in self.can_run_list.values())
            if input_tokens >= self.rem_input_tokens and total_can_run != 0:
                return AddReqResult.OTHER

            if (
                self.rem_chunk_tokens_list is None
                or input_tokens <= self.rem_chunk_tokens_list[dp_rank]
            ):
                # Non-chunked prefill
                self.can_run_list[dp_rank].append(req)
                if self.is_hybrid:
                    swa_uuid_for_lock = self.tree_cache.inc_lock_ref(req.last_node)
                    req.swa_uuid_for_lock = swa_uuid_for_lock
                else:
                    self.tree_cache.inc_lock_ref(req.last_node)
                self._update_prefill_budget(
                    prefix_len,
                    input_tokens,
                    min(
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS_ESTIMATION,
                    ),
                    dp_rank,
                )
            else:
                # Make sure at least one page is available
                trunc_len = self.rem_chunk_tokens_list[dp_rank] // self.page_size * self.page_size
                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # Chunked prefill
                req.extend_input_len = trunc_len
                req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]

                self.can_run_list[dp_rank].append(req)
                self.new_chunked_reqs[dp_rank] = req
                if self.is_hybrid:
                    swa_uuid_for_lock = self.tree_cache.inc_lock_ref(req.last_node)
                    req.swa_uuid_for_lock = swa_uuid_for_lock
                else:
                    self.tree_cache.inc_lock_ref(req.last_node)
                self._update_prefill_budget(prefix_len, trunc_len, 0, dp_rank)

        return self.budget_state()
