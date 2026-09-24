"""Opt-in acceptance diagnostics; sampled KV reads are for correctness only.

JSONL rows describe actual method calls and snapshots, not acceptance verdicts.
Sampled KV reads synchronize donation and perturb timing. Full transfer-content
validation belongs to hybrid_hicache_transfer_probe, not these sampled hashes.
sample_pages=0 disables device reads, but Python/JSONL overhead remains: timed
runs must use matching instrumentation in both groups and disclose that cost.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType as CT
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

_rid = contextvars.ContextVar("hicache_diagnostic_rid", default=None)


def node_record(node):
    if node is None or node.key is None:
        return None
    return {
        "node_id": node.id,
        "parent_id": node.parent.id if node.parent else None,
        "dp_rank": node.key.dp_rank or 0,
        "key_length": len(node.key),
        "key_sha256": hashlib.sha256(
            np.asarray(node.key.token_ids, dtype=np.int64).tobytes()
        ).hexdigest(),
        "components": {
            ct.name: {
                "device_tokens": (
                    None
                    if (cd := node.component_data[ct]).value is None
                    else np.asarray(cd.value).tolist()
                ),
                "host_handles": (
                    None if cd.host_value is None else np.asarray(cd.host_value).tolist()
                ),
                "device_locks": cd.lock_ref,
                "host_locks": cd.host_lock_ref,
                "host_pending": cd.metadata.get("host_pending", False),
            }
            for ct in (CT.FULL, CT.SWA)
        },
    }


def snapshot(cache, *, physical=False):
    if not isinstance(cache, UnifiedRadixCache) or not cache.supports_swa():
        return {"unsupported_cache": type(cache).__name__}
    allocator = cache.token_to_kv_pool_allocator
    result = {
        "cache_class": type(cache).__name__,
        "page_size": cache.page_size,
        "window": cache.sliding_window_size,
        "ranks": [],
    }
    state = cache._hybrid_state
    result["quarantined"] = bool(state and state._quarantined)
    for rank in range(allocator.dp_size):
        row = {"dp_rank": rank, "components": {}}
        for ct, alloc in (
            (CT.FULL, allocator.full_attn_allocator),
            (CT.SWA, allocator.swa_attn_allocator),
        ):
            pool = cache.host_pools.get(ct)
            component = {
                "device_capacity_tokens": alloc.size_per_rank,
                "device_free_tokens": alloc.available_size(rank),
                "tree_evictable_tokens": cache.component_evictable_size_[ct][rank],
                "tree_protected_tokens": cache.component_protected_size_[ct][rank],
                "host_free_pages": (
                    None
                    if pool is None or getattr(pool, "failed", False)
                    else pool.available_size(rank)
                ),
                "host_capacity_pages": (
                    None
                    if pool is None
                    else getattr(pool, "pages_per_rank", None) or pool.total_size(rank)
                ),
                "host_failed": bool(pool and getattr(pool, "failed", False)),
            }
            if physical:
                for field in ("free_slots", "free_pages", "release_pages"):
                    values = getattr(alloc, field, None)
                    if values is not None:
                        component[field] = np.asarray(values[rank]).tolist()
            row["components"][ct.name] = component
        if physical:
            mapping = allocator.full_to_swa_index_mapping
            mapping = mapping[rank] if allocator.dp_size > 1 else mapping
            used = np.flatnonzero(mapping)
            row["full_to_swa_nonzero"] = np.stack((used, mapping[used]), axis=1).tolist()
        result["ranks"].append(row)
    if physical:
        pool = allocator.get_kvcache()
        result["layer_mapping"] = pool.layers_mapping
        result["pool_shapes"] = {
            ct.name: [list(value.shape) for value in subpool.kv_buffer]
            for ct, subpool in ((CT.FULL, pool.full_kv_pool), (CT.SWA, pool.swa_kv_pool))
        }
        result["pool_storage"] = {
            ct.name: [
                {
                    "dtype": str(value.dtype),
                    "nbytes": int(value.nbytes),
                    "bytes_per_page": int(value.nbytes // value.shape[0]),
                }
                for value in subpool.kv_buffer
            ]
            for ct, subpool in ((CT.FULL, pool.full_kv_pool), (CT.SWA, pool.swa_kv_pool))
        }
        stack = list(cache.root_node.children.values())
        result["nodes"] = []
        while stack:
            node = stack.pop()
            result["nodes"].append(node_record(node))
            stack.extend(node.children.values())
        result["inflight"] = {
            ct.name: {
                "write_handles": sorted(getattr(ctrl, "_inflight", ())),
                "load_handles": sorted(getattr(ctrl, "_inflight_load", ())),
            }
            for ct, ctrl in cache.hicache_controllers.items()
        }
        result["host_resources"] = {
            ct.name: (
                {
                    str(handle): dict(
                        rank=page.rank, readers=page.readers, writing=page.writing, chunk=page.chunk
                    )
                    for handle, page in host._pages.items()
                }
                if hasattr(host, "_pages")
                else {"lock_refs": list(host._lock_ref)}
            )
            for ct, host in cache.host_pools.items()
        }
    return result


def install(directory, *, sample_pages=2):
    """Install only in a dedicated acceptance launcher; returns an undo function."""
    from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
    from sgl_jax.srt.managers.schedule_policy import PrefillAdder
    from sgl_jax.srt.managers.scheduler import Scheduler
    from sgl_jax.srt.mem_cache.hybrid_hicache import HybridHiCache
    from sgl_jax.srt.mem_cache.unified_cache_components.swa_component import (
        SWAComponent,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    patches, watches = [], {}
    write_lock = threading.Lock()
    sequence = 0

    def emit(source, phase, **fields):
        nonlocal sequence
        with write_lock:
            sequence += 1
            row = dict(
                sequence=sequence,
                pid=os.getpid(),
                monotonic_ns=time.monotonic_ns(),
                source=source,
                phase=phase,
                rid=_rid.get(),
                **fields,
            )
            with (directory / f"scheduler-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        row,
                        default=lambda value: (
                            value.item() if isinstance(value, np.generic) else str(value)
                        ),
                    )
                    + "\n"
                )

    def hashes(cache, ct, indices, rank):
        from hybrid_hicache_transfer_probe import _read_pages

        if cache._donation_barrier is not None:
            cache._donation_barrier()
        pool = cache.token_to_kv_pool_allocator.get_kvcache()
        subpool = pool.full_kv_pool if ct == CT.FULL else pool.swa_kv_pool
        return [
            hashlib.sha256(values.tobytes()).hexdigest()
            for values in _read_pages(subpool, indices, rank)
        ]

    def patch(
        cls,
        name,
        *,
        get_cache,
        get_node=lambda *a, **k: None,
        get_rid=lambda *a, **k: None,
        evict=False,
    ):
        original = getattr(cls, name)
        source = f"{original.__module__}.{original.__qualname__}:{original.__code__.co_firstlineno}"

        @functools.wraps(original)
        def observed(self, *args, **kwargs):
            cache = get_cache(self)
            if (
                not isinstance(cache, UnifiedRadixCache)
                or not hasattr(cache, "root_node")
                or not cache.supports_swa()
            ):
                return original(self, *args, **kwargs)
            if name == "settle" and not self.pending:
                return original(self, *args, **kwargs)
            token = _rid.set(get_rid(self, *args, **kwargs) or _rid.get())
            node = get_node(self, *args, **kwargs)
            before = node_record(node)
            samples = {}
            settled = [pending[0] for pending in self.pending.values()] if name == "settle" else []
            try:
                if evict and node is not None and sample_pages:
                    for ct in (CT.FULL, CT.SWA):
                        value = node.component_data[ct].value
                        if value is not None and len(value):
                            indices = np.asarray(value[: sample_pages * cache.page_size]).copy()
                            samples[ct] = (
                                indices,
                                hashes(cache, ct, indices, node.key.dp_rank or 0),
                            )
                if name == "restore":
                    for path_node in self.path(node):
                        for ct in (CT.FULL, CT.SWA):
                            prior = watches.get((path_node.id, ct))
                            if prior is not None:
                                indices, earlier = prior
                                current = hashes(cache, ct, indices, path_node.key.dp_rank or 0)
                                emit(
                                    source,
                                    "old_slot_content_before_restore",
                                    node_id=path_node.id,
                                    component=ct.name,
                                    dp_rank=path_node.key.dp_rank or 0,
                                    sampled_device_tokens=indices.tolist(),
                                    before_eviction_layer_hashes=earlier,
                                    current_layer_hashes=current,
                                    changed=current != earlier,
                                    sampled=True,
                                )
                extra = {}
                if name == "restore":
                    extra = {
                        "path_nodes": [node_record(n) for n in self.path(node)],
                        "missing_component_tokens": dict(
                            zip(("FULL", "SWA"), self.sizes(node), strict=True)
                        ),
                    }
                emit(
                    source,
                    "before",
                    node=before,
                    capacity=snapshot(cache),
                    request_rids=(
                        [r.rid for info in self.reqs_info for r in (info.reqs or [])]
                        if isinstance(self, ScheduleBatch)
                        else None
                    ),
                    **extra,
                )
                result = original(self, *args, **kwargs)
                for ct, sample in samples.items():
                    if node.component_data[ct].value is None:
                        watches[(node.id, ct)] = sample
                        emit(
                            source,
                            "device_component_removed",
                            node_id=node.id,
                            component=ct.name,
                            dp_rank=node.key.dp_rank or 0,
                            sampled_device_tokens=sample[0].tolist(),
                            layer_hashes=sample[1],
                            sampled=True,
                        )
                details = {}
                if name == "add_one_req":
                    req = args[0]
                    details = dict(
                        admitted=req in self.can_run_list[req.dp_rank or 0],
                        prefix_tokens=len(req.prefix_indices),
                        extend_tokens=req.extend_input_len,
                        result=str(result),
                    )
                elif name == "retract_decode":
                    details = dict(
                        retracted_rids=[r.rid for r in result[0]],
                        aborted_rids=[r.rid for r in result[2]],
                    )
                elif name == "restore":
                    details = dict(
                        restored_prefix_tokens=len(result[0]),
                        path_nodes=[node_record(n) for n in self.path(node)],
                    )
                    if len(result[0]):
                        for path_node in self.path(node):
                            for ct in (CT.FULL, CT.SWA):
                                prior = watches.get((path_node.id, ct))
                                value = path_node.component_data[ct].value
                                if prior is not None and value is not None:
                                    indices, earlier = prior
                                    current = hashes(
                                        cache,
                                        ct,
                                        np.asarray(value[: len(indices)]),
                                        path_node.key.dp_rank or 0,
                                    )
                                    emit(
                                        source,
                                        "restored_slot_content",
                                        node_id=path_node.id,
                                        component=ct.name,
                                        dp_rank=path_node.key.dp_rank or 0,
                                        sampled_device_tokens=np.asarray(
                                            value[: len(indices)]
                                        ).tolist(),
                                        before_eviction_layer_hashes=earlier,
                                        restored_layer_hashes=current,
                                        matches=current == earlier,
                                        sampled=True,
                                    )
                elif name == "settle":
                    details = dict(settled_nodes=[node_record(n) for n in settled])
                elif name in ("backup_component", "evict_host"):
                    details = dict(returned_pages=result)
                emit(source, "after", node=node_record(node), capacity=snapshot(cache), **details)
                return result
            except BaseException as exc:
                emit(source, "error", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                _rid.reset(token)

        setattr(cls, name, observed)
        patches.append((cls, name, original))

    for name in ("backup_component", "restore", "evict_host"):
        patch(
            HybridHiCache,
            name,
            get_cache=lambda obj: obj.cache,
            get_node=(
                (lambda obj, node, *a, **k: node)
                if name != "evict_host"
                else (lambda *a, **k: None)
            ),
        )
    patch(HybridHiCache, "settle", get_cache=lambda obj: obj.cache)
    patch(
        UnifiedRadixCache,
        "_evict_device_leaf",
        get_cache=lambda obj: obj,
        get_node=lambda obj, node, *a, **k: node,
        evict=True,
    )
    patch(
        SWAComponent,
        "evict_component",
        get_cache=lambda obj: obj.cache,
        get_node=lambda obj, node, *a, **k: node,
        evict=True,
    )
    for name in ("cache_finished_req", "cache_unfinished_req"):
        patch(
            UnifiedRadixCache,
            name,
            get_cache=lambda obj: obj,
            get_rid=lambda obj, req, *a, **k: req.rid,
        )
    patch(UnifiedRadixCache, "reset", get_cache=lambda obj: obj)
    patch(
        PrefillAdder,
        "add_one_req",
        get_cache=lambda obj: obj.tree_cache,
        get_rid=lambda obj, req: req.rid,
    )
    patch(ScheduleBatch, "retract_decode", get_cache=lambda obj: obj.tree_cache)
    patch(
        Scheduler,
        "abort_request",
        get_cache=lambda obj: obj.tree_cache,
        get_rid=lambda obj, req: req.rid,
    )
    patch(Scheduler, "flush_cache", get_cache=lambda obj: obj.tree_cache)
    original_state = Scheduler.get_internal_state

    @functools.wraps(original_state)
    def internal_state(self, *args, **kwargs):
        result = original_state(self, *args, **kwargs)
        result.internal_state["hybrid_hicache_snapshot"] = snapshot(self.tree_cache, physical=True)
        requests = {}
        for batch in (self.cur_batch, self.last_batch, self.running_batch):
            if batch is not None:
                for info in batch.reqs_info:
                    for req in info.reqs or []:
                        requests[req.rid] = req
        for req in [*self.waiting_queue, *self.chunked_reqs]:
            if req is not None:
                requests[req.rid] = req
        result.internal_state["hybrid_hicache_requests"] = [
            {
                "rid": req.rid,
                "dp_rank": req.dp_rank,
                "req_pool_idx": req.req_pool_idx,
                "kv_allocated_len": req.kv_allocated_len,
                "cache_protected_len": req.cache_protected_len,
                "swa_evicted_seqlen": req.swa_evicted_seqlen,
                "full_token_indices": (
                    self.req_to_token_pool.read(req.req_pool_idx, req.kv_allocated_len).tolist()
                    if req.req_pool_idx is not None
                    else []
                ),
            }
            for req in requests.values()
        ]
        # Pending DP inputs are TokenizedGenerateReqInput, not allocated Req
        # objects. Retain their identity without inventing KV ownership fields.
        result.internal_state["hybrid_hicache_requests"].extend(
            {"rid": req.rid, "dp_rank": req.dp_rank, "stage": "pending_dp"}
            for req in self.pending_dp_reqs
        )
        emit("Scheduler.get_internal_state", "snapshot", state=result.internal_state)
        return result

    Scheduler.get_internal_state = internal_state
    patches.append((Scheduler, "get_internal_state", original_state))

    def undo():
        for cls, name, original in reversed(patches):
            setattr(cls, name, original)

    return undo
