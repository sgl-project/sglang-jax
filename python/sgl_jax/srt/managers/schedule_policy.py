from __future__ import annotations

import os
import random
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from jax import numpy as jnp

from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.radix_cache import RadixCache, TreeNode

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
        # Initialize only if in-batch prefix caching is enabled, otherwise set to None
        # to avoid unnecessary object creation and computation.
        if IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD != -1:
            self.waiting_queue_radix_tree: Optional[RadixCache] = RadixCache(
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
                page_size=1,
                disable=False,
            )
        else:
            self.waiting_queue_radix_tree = None

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
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
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

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
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
        and optionally handles in-batch prefix caching logic if enabled.
        Requests that qualify for in-batch deprioritization are added to the returned set.
        """
        temporary_deprioritized: set[int] = set()

        in_batch_caching_enabled = self.waiting_queue_radix_tree is not None
        if in_batch_caching_enabled:
            self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.adjust_max_prefix_ids()

            # NOTE: the prefix_indices must always be aligned with last_node
            r.prefix_indices, r.last_node, r.last_host_node, r.host_hit_length = (
                self.tree_cache.match_prefix(rid=r.rid, key=prefix_ids)
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            if (
                in_batch_caching_enabled
                and len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD
            ):
                # Retrieve in-batch matching prefixes from the waiting queue's radix tree.
                # This checks if the current request's prefix is similar to any already
                # processed requests within the same waiting queue cycle.
                (
                    in_batch_matching_prefixes,
                    _, # noqa
                    _, # noqa
                    _, # noqa
                ) = self.waiting_queue_radix_tree.match_prefix(rid=r.rid, key=prefix_ids)

                # If a sufficient number of in-batch matches are found, deprioritize this request.
                # This is to encourage better cache utilization by reducing redundant
                # prefills for very similar prefixes within the same batch.
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # If not deprioritized, add its prefix to the in-batch tree so that
                    # subsequent requests in this scheduling cycle can match against it.
                    self.waiting_queue_radix_tree.insert(r.rid, prefix_ids)

        return temporary_deprioritized