"""Cache-aware DP routing must count host-tier (HiCache) prefix hits.

``Scheduler._lookup_prefix_length`` feeds the per-rank prefix length into
``DpRouter``. Before the fix it returned only ``len(device_indices)``, so a
prefix that had been evicted from HBM but was still backed up in the pinned
host pool looked like a miss, and the router happily sent the request to
another rank where it was prefilled from scratch (the H2D restore path was
never reached). These tests pin the contract: device tokens + host tokens.
"""

import os

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

import types

import numpy as np

from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult


class _StubTree:
    """Returns a canned MatchResult and records the key it was probed with."""

    def __init__(self, device_tokens: int, host_hit_length: int):
        self._device_tokens = device_tokens
        self._host_hit_length = host_hit_length
        self.seen_params: list[MatchPrefixParams] = []

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        self.seen_params.append(params)
        return MatchResult(
            device_indices=np.arange(self._device_tokens, dtype=np.int32),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            host_hit_length=self._host_hit_length,
        )


def _lookup(tree, token_ids, extra_key=None, dp_rank=0):
    fake_self = types.SimpleNamespace(tree_cache=tree)
    return Scheduler._lookup_prefix_length(fake_self, token_ids, extra_key, dp_rank)


def test_no_tree_cache_returns_zero():
    assert _lookup(None, [1, 2, 3]) == 0


def test_device_only_hit_unchanged():
    tree = _StubTree(device_tokens=256, host_hit_length=0)
    assert _lookup(tree, list(range(512))) == 256


def test_host_hit_counts_toward_prefix_length():
    # 128 tokens still in HBM, a further 384 only in the pinned host pool.
    tree = _StubTree(device_tokens=128, host_hit_length=384)
    assert _lookup(tree, list(range(1024))) == 512


def test_host_only_hit_is_not_a_miss():
    # Whole prefix evicted to host: must still look like a 640-token hit so the
    # router keeps the request on the rank that owns the backup.
    tree = _StubTree(device_tokens=0, host_hit_length=640)
    assert _lookup(tree, list(range(1024))) == 640


def test_negative_host_hit_is_clamped():
    tree = _StubTree(device_tokens=64, host_hit_length=-5)
    assert _lookup(tree, list(range(100))) == 64


def test_probe_uses_dp_keyed_radix_key():
    tree = _StubTree(device_tokens=0, host_hit_length=0)
    _lookup(tree, [7, 8, 9], extra_key="lora-a", dp_rank=3)
    (params,) = tree.seen_params
    assert list(params.key.token_ids) == [7, 8, 9]
    assert params.key.extra_key == "lora-a"
    assert params.key.dp_rank == 3
