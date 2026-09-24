"""Explicit, destructive L1/L2 transfer probe for a freshly loaded TPU scheduler.

Invoke on the scheduler thread of a dedicated, idle test process only. The
loaded ModelRunner supplies the real layer layout and device buffers; no model
shape is reconstructed from configuration. This module does not start a server.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sgl_jax.srt.mem_cache.memory_pool import SWAKVPool, write_kv_layer
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType as CT
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def _pages(indices, page_size):
    indices = np.asarray(indices, dtype=np.int32)
    if len(indices) == 0 or len(indices) % page_size:
        raise AssertionError("probe requires nonempty, complete pages")
    pages = indices[::page_size] // page_size
    expected = (pages[:, None] * page_size + np.arange(page_size)).reshape(-1)
    np.testing.assert_array_equal(indices, expected)
    return pages


def _read_pages(subpool, indices, rank):
    pages = _pages(indices, subpool.page_size)
    stride = subpool.kv_buffer[0].shape[0] // subpool.dp_size
    out_sharding = NamedSharding(
        subpool.mesh, PartitionSpec(None, None, subpool.kv_partition_axis, None, None)
    )
    return [
        np.asarray(
            jax.device_get(buf.at[rank * stride + pages].get(out_sharding=out_sharding))
        ).copy()
        for buf in subpool.kv_buffer
    ]


def _write_marker(subpool, indices, rank, marker):
    """Use the existing in-place KV scatter, preserving Raiden registrations."""
    _pages(indices, subpool.page_size)
    # write_kv_layer splits loc/data over the DP axis and addresses each
    # shard's LOCAL page slice. Other ranks receive -1 (no write).
    per_rank_tokens = len(indices)
    loc_np = np.full(subpool.dp_size * per_rank_tokens, -1, dtype=np.int32)
    loc_np[rank * per_rank_tokens : (rank + 1) * per_rank_tokens] = indices
    loc = jax.device_put(
        jnp.asarray(loc_np),
        NamedSharding(subpool.mesh, PartitionSpec(subpool.attention_data_partition_axis)),
    )
    for layer, buf in enumerate(subpool.kv_buffer):
        values = jax.device_put(
            jnp.full((len(loc), 1) + tuple(buf.shape[2:]), marker + layer, dtype=buf.dtype),
            NamedSharding(
                subpool.mesh,
                PartitionSpec(
                    subpool.attention_data_partition_axis,
                    None,
                    subpool.kv_partition_axis,
                    None,
                    None,
                ),
            ),
        )
        subpool.kv_buffer[layer] = write_kv_layer(
            values,
            loc,
            buf,
            subpool.page_size,
            subpool.kv_partition_axis,
            subpool.attention_data_partition_axis,
            subpool.mesh,
        )
    jax.block_until_ready(subpool.kv_buffer)


def _verify_component(subpool, indices, rank, expected, layer_ids):
    actual = _read_pages(subpool, indices, rank)
    if len(actual) != len(expected) or len(actual) != len(layer_ids):
        raise AssertionError("model layer layout changed during probe")
    rows = []
    pages = _pages(indices, subpool.page_size)
    for local_layer, (got, want) in enumerate(zip(actual, expected, strict=True)):
        for ordinal, page in enumerate(pages):
            np.testing.assert_array_equal(got[ordinal], want[ordinal])
            rows.append(
                {
                    "model_layer": int(layer_ids[local_layer]),
                    "local_layer": local_layer,
                    "device_page": int(page),
                    "page_bytes": int(got[ordinal].nbytes),
                    "sha256": hashlib.sha256(got[ordinal].tobytes()).hexdigest(),
                }
            )
    return rows


def _settle(cache):
    for controller in cache.hicache_controllers.values():
        controller.drain_pending()
    cache.check_hicache_events()


def _host_capacity(host_pool, rank):
    if hasattr(host_pool, "pages_per_rank"):
        return host_pool.pages_per_rank
    return host_pool.total_size(rank)


def _pristine(cache, allocator):
    if cache.root_node.children or cache._hybrid_coordinator.pending:
        return False
    mappings = allocator.full_to_swa_index_mapping
    if not isinstance(mappings, list):
        mappings = [mappings]
    if any(np.any(mapping) for mapping in mappings):
        return False
    return all(
        allocator.full_available_size(rank) == allocator.full_attn_allocator.size_per_rank
        and allocator.swa_available_size(rank) == allocator.swa_attn_allocator.size_per_rank
        and all(
            host.available_size(rank) == _host_capacity(host, rank)
            for host in cache.host_pools.values()
        )
        for rank in range(allocator.dp_size)
    )


def _registered_addresses(cache):
    result = {}
    for ct, controller in cache.hicache_controllers.items():
        if getattr(controller, "direct_transfers", False):
            result[ct] = controller._buffer_addresses()
    return result


def _assert_addresses(cache, baseline):
    for ct, addresses in baseline.items():
        if cache.hicache_controllers[ct]._buffer_addresses() != addresses:
            raise RuntimeError(f"{ct.name} Raiden KV buffer address changed; transfer unsafe")


def _case(cache, allocator, pool, rank, mode, page_count, addresses):
    count = page_count * pool.page_size
    tokens = list(
        range(
            10000 + rank * 1000 + (mode == "swa_only") * 100,
            10000 + rank * 1000 + (mode == "swa_only") * 100 + count,
        )
    )
    key = RadixKey(token_ids=tokens, extra_key=None, dp_rank=rank)
    full = allocator.alloc(count, dp_rank=rank)
    if full is None:
        raise RuntimeError(f"rank {rank} has insufficient FULL/SWA device pages")
    swa = allocator.translate_full_to_swa(full, dp_rank=rank)
    subpools = {CT.FULL: pool.full_kv_pool, CT.SWA: pool.swa_kv_pool}
    original = {CT.FULL: full.copy(), CT.SWA: swa.copy()}
    other_ranks = {
        (ct, other): _read_pages(subpools[ct], original[ct], other)
        for ct in (CT.FULL, CT.SWA)
        for other in range(allocator.dp_size)
        if other != rank
    }
    layout = {
        ct: [
            global_id
            for _, global_id in sorted(
                (local_id, global_id)
                for global_id, (local_id, is_swa) in pool.layers_mapping.items()
                if is_swa == (ct == CT.SWA)
            )
        ]
        for ct in (CT.FULL, CT.SWA)
    }
    expected = {}
    overwritten = {}
    for ct in (CT.FULL, CT.SWA):
        _write_marker(subpools[ct], original[ct], rank, 10 + rank * 40 + int(ct) * 10)
        expected[ct] = _read_pages(subpools[ct], original[ct], rank)
    _assert_addresses(cache, addresses)
    cache.insert(InsertParams(key=key, value=full))
    _settle(cache)
    match = cache.match_prefix(MatchPrefixParams(key=key))
    node = match.last_device_node
    if mode == "dual":
        cache.evict(EvictParams(num_tokens=count, dp_rank=rank))
    else:
        cache.components[CT.SWA].evict_component(node)
        cache._update_aux_evictable_node_sets(node)
    _settle(cache)
    for ct in (CT.FULL, CT.SWA):
        if (mode == "dual" or ct == CT.SWA) and node.component_data[ct].host_value is None:
            raise AssertionError(f"{mode}: {ct.name} D2H backup absent")
        if mode == "dual" or ct == CT.SWA:
            if node.component_data[ct].value is not None:
                raise AssertionError(f"{mode}: {ct.name} device value not evicted")
            _write_marker(subpools[ct], original[ct], rank, -70 - rank * 10 - int(ct))
            observed = _read_pages(subpools[ct], original[ct], rank)
            for layer, (before, after) in enumerate(zip(expected[ct], observed, strict=True)):
                for page in range(len(before)):
                    if np.array_equal(before[page], after[page]):
                        raise AssertionError(
                            f"{mode}: {ct.name} layer {layer} page {page} not overwritten"
                        )
            overwritten[ct.name] = len(_pages(original[ct], pool.page_size))
    _assert_addresses(cache, addresses)
    match = cache.match_prefix(MatchPrefixParams(key=key))
    if match.host_hit_length != count or (
        mode == "swa_only" and match.swa_host_hit_length != count
    ):
        raise AssertionError(f"{mode}: expected complete host hit, got {match.host_hit_length}")
    restored, _, plan = cache.init_load_back(match.last_host_node, match.host_hit_length)
    cache.finish_load_back(plan)
    _assert_addresses(cache, addresses)
    # SWA healing returns the resident FULL addresses from the newly valid
    # boundary, even though it did not allocate replacement FULL pages.
    if len(restored) != count:
        raise AssertionError(f"{mode}: wrong valid-prefix length {len(restored)}")
    full_after = np.asarray(node.component_data[CT.FULL].value)
    swa_after = np.asarray(node.component_data[CT.SWA].value)
    np.testing.assert_array_equal(
        allocator.translate_full_to_swa(full_after, dp_rank=rank), swa_after
    )
    if mode == "swa_only":
        np.testing.assert_array_equal(full_after, original[CT.FULL])
        np.testing.assert_array_equal(restored, original[CT.FULL])
    rows = {
        ct.name: _verify_component(
            subpools[ct], node.component_data[ct].value, rank, expected[ct], layout[ct]
        )
        for ct in (CT.FULL, CT.SWA)
    }
    for (ct, other), before in other_ranks.items():
        after = _read_pages(subpools[ct], original[ct], other)
        for earlier, later in zip(before, after, strict=True):
            np.testing.assert_array_equal(earlier, later)
    return {
        "rank": rank,
        "mode": mode,
        "tokens": count,
        "full_indices_before": original[CT.FULL].tolist(),
        "swa_indices_before": original[CT.SWA].tolist(),
        "full_indices_after": full_after.tolist(),
        "swa_indices_after": swa_after.tolist(),
        "mapping_verified": True,
        "other_ranks_unchanged": True,
        "overwritten_original_pages": overwritten,
        "pages": rows,
    }


def run_transfer_probe(scheduler, *, destructive_opt_in=False, report_path=None, page_count=1):
    """Probe an actual loaded scheduler; call synchronously on its own thread.

    The caller must use a dedicated fresh test process, not a serving instance.
    The entire cache is flushed between cases and on exit. A failed native
    transfer may quarantine resources, in which case flush is deliberately
    skipped and the process must be discarded.
    """
    if destructive_opt_in is not True:
        raise ValueError("explicit destructive_opt_in=True is required")
    if jax.default_backend() != "tpu":
        raise RuntimeError("real TPU required; CPU CI must not run this probe")
    if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count < 1:
        raise ValueError("page_count must be a positive integer")
    if not scheduler.is_fully_idle():
        raise RuntimeError("scheduler has active or pending requests")
    cache = scheduler.tree_cache
    allocator = scheduler.token_to_kv_pool_allocator
    runner = scheduler.tp_worker.get_model_runner()
    pool = runner.token_to_kv_pool
    if (
        not isinstance(cache, UnifiedRadixCache)
        or cache.tree_components != (CT.FULL, CT.SWA)
        or not cache.hicache_enabled
        or not isinstance(pool, SWAKVPool)
        or allocator.get_kvcache() is not pool
        or cache.token_to_kv_pool_allocator is not allocator
    ):
        raise RuntimeError("loaded runner/cache is not the connected FULL+SWA HiCache stack")
    if not _pristine(cache, allocator):
        raise RuntimeError("probe requires a fresh empty tree, mappings and allocators")
    if not pool.layers_mapping or not pool.full_layer_nums or not pool.swa_layer_nums:
        raise RuntimeError("loaded model has no actual FULL+SWA layer layout")
    for r in range(allocator.dp_size):
        if page_count * pool.page_size > min(
            allocator.full_available_size(r), allocator.swa_available_size(r)
        ) or any(h.available_size(r) < page_count for h in cache.host_pools.values()):
            raise RuntimeError(f"rank {r} has insufficient pages for probe")
    backend = (
        "raiden"
        if any(getattr(c, "direct_transfers", False) for c in cache.hicache_controllers.values())
        else "jax"
    )
    if (
        len(
            {
                bool(getattr(c, "direct_transfers", False))
                for c in cache.hicache_controllers.values()
            }
        )
        != 1
    ):
        raise RuntimeError("FULL and SWA controllers use different transfer backends")
    addresses = _registered_addresses(cache)
    report = {
        "status": "running",
        "backend": backend,
        "write_policy": cache.write_policy,
        "dp_size": allocator.dp_size,
        "page_size": pool.page_size,
        "model_path": scheduler.server_args.model_path,
        "model_revision": scheduler.server_args.revision,
        "layer_mapping": {
            str(k): {"local_layer": v[0], "component": "SWA" if v[1] else "FULL"}
            for k, v in pool.layers_mapping.items()
        },
        "pool_shapes": {
            ct.name: [list(b.shape) for b in subpool.kv_buffer]
            for ct, subpool in (
                (CT.FULL, pool.full_kv_pool),
                (CT.SWA, pool.swa_kv_pool),
            )
        },
        "cases": [],
        "cleanup": None,
    }
    error = None
    try:
        for rank in range(allocator.dp_size):
            for mode in ("dual", "swa_only"):
                report["cases"].append(
                    _case(cache, allocator, pool, rank, mode, page_count, addresses)
                )
                success, message, _ = scheduler.flush_cache()
                if not success:
                    raise RuntimeError(f"flush after {mode}/rank{rank} failed: {message}")
                if not _pristine(cache, allocator):
                    raise RuntimeError(f"flush after {mode}/rank{rank} left cache state")
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - record probe failures before cleanup
        error = exc
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        healthy = not any(getattr(h, "failed", False) for h in cache.host_pools.values())
        if healthy:
            try:
                success, message, _ = scheduler.flush_cache()
                pristine = success and _pristine(cache, allocator)
                report["cleanup"] = "flushed_pristine" if pristine else f"flush_failed: {message}"
                if not pristine and error is None:
                    error = RuntimeError(message or "flush left cache state")
                    report["status"] = "failed"
            except Exception as exc:  # noqa: BLE001 - preserve cleanup failure in report
                report["cleanup"] = f"flush_failed: {type(exc).__name__}: {exc}"
                if error is None:
                    error = exc
                    report["status"] = "failed"
        else:
            report["cleanup"] = "native_transfer_unhealthy; discard process"
        if report_path is not None:
            Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if error is not None:
        raise RuntimeError(f"transfer probe failed; report={report_path}: {error}") from error
    return report
