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


def _marker_dimensions(pool, token_count):
    subpools = (pool.full_kv_pool, pool.swa_kv_pool)
    buffers = [buf for subpool in subpools for buf in subpool.kv_buffer]
    return (
        max(len(p.kv_buffer) for p in subpools),
        token_count,
        *(max(buf.shape[axis] for buf in buffers) for axis in (2, 3, 4)),
    )


def _marker_values(shape, marker, layer, digit, dimensions):
    """Encode each coordinate in base 256; every byte is exact in BF16.

    Across all digits, the signature identifies rank/component (marker), layer,
    page/token (flattened token axis), head, K/V, and head dimension. A single
    global float arange would lose these distinctions when cast to BF16.
    """
    code = np.uint64(marker * dimensions[0] + layer)
    for axis, (size, extent) in enumerate(zip(shape, dimensions[1:], strict=True)):
        coordinate_shape = [1] * len(shape)
        coordinate_shape[axis] = size
        code = code * np.uint64(extent) + np.arange(size, dtype=np.uint64).reshape(coordinate_shape)
    return ((code >> (8 * digit)) & 255).astype(np.uint8)


def _write_marker(subpool, indices, rank, marker, *, digit=0, dimensions=None):
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
        shape = (per_rank_tokens,) + tuple(buf.shape[2:])
        if marker < 0:
            active = np.full(shape, marker, dtype=buf.dtype)
        else:
            bounds = dimensions or (len(subpool.kv_buffer),) + shape
            pattern = _marker_values(shape, marker, layer, digit, bounds)
            active = pattern.astype(buf.dtype)
            np.testing.assert_array_equal(active, pattern, err_msg="KV dtype loses marker bytes")
        data = np.zeros((len(loc), 1) + tuple(buf.shape[2:]), dtype=buf.dtype)
        data[rank * per_rank_tokens : (rank + 1) * per_rank_tokens, 0] = active
        values = jax.device_put(
            data,
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
        if got.shape != want.shape:
            raise AssertionError(f"layer {local_layer}: page layout {got.shape} != {want.shape}")
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


def _case(cache, allocator, pool, rank, mode, page_count, addresses, *, digit=0, dimensions=None):
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
        _write_marker(
            subpools[ct],
            original[ct],
            rank,
            rank * 2 + int(ct),
            digit=digit,
            dimensions=dimensions or _marker_dimensions(pool, count),
        )
        expected[ct] = _read_pages(subpools[ct], original[ct], rank)
    _assert_addresses(cache, addresses)
    cache.insert(InsertParams(key=key, value=full))
    _settle(cache)
    match = cache.match_prefix(MatchPrefixParams(key=key))
    node = match.last_device_node
    evicted = {
        CT.FULL: full,
        CT.SWA: swa if mode == "dual" else node.component_data[CT.SWA].value.copy(),
    }
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
            before_overwrite = _read_pages(subpools[ct], evicted[ct], rank)
            _write_marker(subpools[ct], evicted[ct], rank, -1)
            observed = _read_pages(subpools[ct], evicted[ct], rank)
            for layer, (before, after) in enumerate(zip(before_overwrite, observed, strict=True)):
                for page in range(len(before)):
                    if np.array_equal(before[page], after[page]):
                        raise AssertionError(
                            f"{mode}: {ct.name} layer {layer} page {page} not overwritten"
                        )
            overwritten[ct.name] = len(_pages(evicted[ct], pool.page_size))
    _assert_addresses(cache, addresses)
    match = cache.match_prefix(MatchPrefixParams(key=key))
    if (
        match.host_hit_length <= 0
        or len(match.device_indices) + match.host_hit_length != count
        or (mode == "swa_only" and match.swa_host_hit_length != len(evicted[CT.SWA]))
    ):
        raise AssertionError(f"{mode}: expected complete host hit, got {match.host_hit_length}")
    missing_full, missing_swa = cache.get_load_back_sizes(match.last_host_node)
    restored, _, plan = cache.init_load_back(match.last_host_node, match.host_hit_length)
    cache.finish_load_back(plan)
    _assert_addresses(cache, addresses)
    # SWA healing returns the resident FULL addresses from the newly valid
    # boundary, even though it did not allocate replacement FULL pages.
    if len(restored) != match.host_hit_length:
        raise AssertionError(f"{mode}: wrong restored-prefix length {len(restored)}")
    healed = cache.match_prefix(MatchPrefixParams(key=key))
    full_after = np.asarray(healed.device_indices)
    if len(full_after) != count:
        raise AssertionError(f"{mode}: incomplete valid prefix after restore")
    mapped = allocator.translate_full_to_swa(full_after, dp_rank=rank, require_mapped=False)
    swa_present = mapped > 0
    swa_after = mapped[swa_present]
    resident_swa = []
    current = healed.last_device_node
    while current is not cache.root_node:
        value = current.component_data[CT.SWA].value
        if value is not None:
            resident_swa.append(value)
        current = current.parent
    np.testing.assert_array_equal(swa_after, np.concatenate(resident_swa[::-1]))
    # A short SWA window may restore only the suffix while FULL spans several
    # split nodes. Verify every resident mapped SWA page, and every FULL page.
    expected[CT.SWA] = [
        values.reshape((count,) + values.shape[2:])[swa_present].reshape(
            (-1, pool.page_size) + values.shape[2:]
        )
        for values in expected[CT.SWA]
    ]
    if mode == "swa_only":
        np.testing.assert_array_equal(full_after, original[CT.FULL])
        np.testing.assert_array_equal(restored, original[CT.FULL][-len(restored) :])
    rows = {
        ct.name: _verify_component(
            subpools[ct], full_after if ct == CT.FULL else swa_after, rank, expected[ct], layout[ct]
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
        "marker_digit": digit,
        "restored_prefix_tokens": len(restored),
        "restored_component_tokens": {"FULL": missing_full, "SWA": missing_swa},
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


def run_transfer_probe(scheduler, *, destructive_opt_in=False, report_path=None, page_count=2):
    """Probe an actual loaded scheduler; call synchronously on its own thread.

    The caller must use a dedicated fresh test process, not a serving instance.
    The entire cache is flushed between cases and on exit. A failed native
    transfer may quarantine resources, in which case flush is deliberately
    skipped and the process must be discarded. Two pages are probed by default;
    a short SWA window may only restore its tail page, recorded per component.
    Every coordinate byte gets an independent backup/overwrite/restore cycle.
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
    dimensions = _marker_dimensions(pool, page_count * pool.page_size)
    highest_code = allocator.dp_size * 2 * int(np.prod(dimensions)) - 1
    marker_digits = max(1, (highest_code.bit_length() + 7) // 8)
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
        "marker_encoding": "base256 coordinate bytes, exact in BF16",
        "marker_digits": marker_digits,
        "marker_dimensions": list(dimensions),
        "cases": [],
        "cleanup": None,
    }
    error = None
    try:
        for digit in range(marker_digits):
            for rank in range(allocator.dp_size):
                for mode in ("dual", "swa_only"):
                    report["cases"].append(
                        _case(
                            cache,
                            allocator,
                            pool,
                            rank,
                            mode,
                            page_count,
                            addresses,
                            digit=digit,
                            dimensions=dimensions,
                        )
                    )
                    # Each digit must get its own fresh backup, not reuse the
                    # previous digit's host pages or resident device prefix.
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
