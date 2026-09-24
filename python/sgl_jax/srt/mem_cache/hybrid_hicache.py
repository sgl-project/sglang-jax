"""Component-scoped L2 transactions for Unified FULL+SWA caches.

Tree values and mappings are published only after both transfers finish. JAX
backup keeps its synchronous gather/asynchronous host copy; restore deliberately
finishes at the donation barrier before the scheduler builds SWA request indices.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace

import numpy as np

from sgl_jax.srt.mem_cache.base_prefix_cache import EvictParams
from sgl_jax.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase as Phase,
)
from sgl_jax.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType as CT,
)


class HybridHiCache:
    def __init__(self, cache):
        self.cache = cache
        # Pending host reservations are not published in ComponentData.
        self.pending = {}

    def path(self, node):
        nodes = []
        while node is not self.cache.root_node:
            nodes.append(node)
            node = node.parent
        return nodes[::-1]

    def retained(self, node):
        return bool(node.children) or any(
            cd.host_value is not None or cd.metadata.get("host_pending", False)
            for cd in node.component_data
        )

    def prune_empty(self, node):
        while node is not self.cache.root_node:
            if self.retained(node) or any(
                cd.value is not None or cd.lock_ref or cd.host_lock_ref
                for cd in node.component_data
            ):
                return
            parent = node.parent
            self.cache.evictable_device_leaves.discard(node)
            self.cache.evictable_host_leaves.discard(node)
            for candidates in self.cache.aux_evictable_device_nodes.values():
                candidates.discard(node)
            self.cache._remove_leaf_from_parent(node)
            self.cache._update_evictable_leaf_sets(parent)
            node = parent

    def global_pages(self, ct, pages, rank):
        pool = self.cache.hicache_controllers[ct]._device_pool
        dp = self.cache.token_to_kv_pool_allocator.dp_size
        stride = pool.kv_buffer[0].shape[0] // dp
        return [int(p) + rank * stride for p in pages]

    def settle(self, wait=False):
        error = None
        for future, (node, ct, transfer) in list(self.pending.items()):
            if not wait and not future.done():
                continue
            cd = node.component_data[ct]
            try:
                future.result()
                self.cache.components[ct].commit_hicache_transfer(
                    node, Phase.BACKUP_HOST, transfers=[transfer]
                )
            except Exception as exc:
                self.cache.host_pools[ct].free(list(transfer.host_handles))
                error = error or exc
            finally:
                cd.metadata.pop("host_pending", None)
                self.pending.pop(future)
                self.cache._update_evictable_leaf_sets(node)
                self.cache._update_evictable_leaf_sets(node.parent)
                self.prune_empty(node)
        for controller in self.cache.hicache_controllers.values():
            try:
                controller.check_write_status()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def evict_host(self, count, rank=None, ct=CT.FULL):
        freed = 0
        nodes = []
        stack = list(self.cache.root_node.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            nodes.append(node)
        for node in sorted(nodes, key=lambda n: n.last_access_time):
            cd = node.component_data[ct]
            if cd.host_value is None or cd.host_lock_ref or cd.lock_ref:
                continue
            if rank is not None and (node.key.dp_rank or 0) != rank:
                continue
            handles = list(map(int, cd.host_value))
            controller = self.cache.hicache_controllers[ct]
            if controller.has_inflight(handles):
                continue
            controller.evict_callback(handles)
            cd.host_value = None
            freed += len(handles)
            self.cache._update_evictable_leaf_sets(node)
            # Keep ancestors while descendants own resources, and prune only
            # empty leaves after their final component relinquishes ownership.
            self.prune_empty(node)
            if freed >= count:
                break
        return freed

    def backup_component(self, node, ct):
        cd = node.component_data[ct]
        if cd.value is None or cd.host_value is not None or cd.metadata.get("host_pending"):
            return 0
        component = self.cache.components[ct]
        (transfer,) = component.build_hicache_transfers(node, Phase.BACKUP_HOST)
        pool = self.cache.host_pools[ct]
        controller = self.cache.hicache_controllers[ct]
        rank = transfer.device.dp_rank
        handles = pool.alloc(len(transfer.device.pages), dp_rank=rank)
        if handles is None:
            self.evict_host(len(transfer.device.pages), rank, ct)
            handles = pool.alloc(len(transfer.device.pages), dp_rank=rank)
        if handles is None:
            return 0
        handles = list(map(int, handles))
        transfer = replace(transfer, host_handles=tuple(handles))
        lock = self.cache.inc_lock_ref(node)
        try:
            if self.cache._donation_barrier is not None:
                self.cache._donation_barrier()
            if getattr(controller, "direct_transfers", False):
                controller.prepare_transfer()
                controller.submit_backup(transfer.device, handles).wait()
                component.commit_hicache_transfer(node, Phase.BACKUP_HOST, transfers=[transfer])
            else:
                future = controller.write(
                    self.global_pages(ct, transfer.device.pages, rank), handles
                )
                cd.metadata["host_pending"] = True
                self.pending[future] = (node, ct, transfer)
        except Exception:
            if not getattr(pool, "failed", False):
                pool.free(handles)
                self.cache.dec_lock_ref(node, lock.to_dec_params())
            raise
        self.cache.dec_lock_ref(node, lock.to_dec_params())
        return len(handles)

    def backup(self, node, write_back=False):
        nodes = [node] if write_back else self.path(node)
        return sum(self.backup_component(n, ct) for n in nodes for ct in (CT.FULL, CT.SWA))

    def selection(self, node):
        path = self.path(node)
        selected = {CT.FULL: [], CT.SWA: []}
        for n in path:
            cd = n.component_data[CT.FULL]
            if cd.value is None:
                if cd.host_value is None:
                    return None
                selected[CT.FULL].append(n)
        remaining = self.cache.components[CT.SWA].sliding_window_size
        for n in reversed(path):
            if remaining <= 0:
                break
            cd = n.component_data[CT.SWA]
            if cd.value is None:
                if cd.host_value is None:
                    return None
                selected[CT.SWA].append(n)
            remaining -= len(n.key)
        selected[CT.SWA].reverse()
        return selected

    def sizes(self, node):
        selected = self.selection(node)
        if selected is None:
            return 0, 0
        return tuple(sum(len(n.key) for n in selected[ct]) for ct in (CT.FULL, CT.SWA))

    def restore(self, node, host_hit_length, mem_quota=None, swa_mem_quota=None):
        empty = (np.empty(0, dtype=np.int32), node, [])
        selected = self.selection(node)
        if selected is None:
            return empty
        sizes = {ct: sum(len(n.key) for n in nodes) for ct, nodes in selected.items()}
        if not any(sizes.values()):
            return empty
        if (mem_quota is not None and sizes[CT.FULL] > mem_quota) or (
            swa_mem_quota is not None and sizes[CT.SWA] > swa_mem_quota
        ):
            return empty
        cache = self.cache
        allocator = cache.token_to_kv_pool_allocator
        rank = node.key.dp_rank or 0
        path = self.path(node)
        # Compute the prior valid device boundary, including SWA gaps, before
        # healing. Resident FULL indices after that boundary belong in return.
        validators = [cache.components[ct].create_match_validator(True) for ct in (CT.FULL, CT.SWA)]
        boundary = 0
        length = 0
        broken = False
        for n in path:
            length += len(n.key)
            broken |= n.component_data[CT.FULL].value is None
            valid = [validate(n) for validate in validators]
            if not broken and all(valid):
                boundary = length
        pins = []
        reserved = {}
        transfers = []
        operations = []
        lock = None
        committed = False
        try:
            for ct, nodes in selected.items():
                handles = [int(h) for n in nodes for h in n.component_data[ct].host_value]
                if handles:
                    cache.host_pools[ct].pin(handles)
                    for n in nodes:
                        n.component_data[ct].host_lock_ref += 1
                    pins.append((ct, nodes, handles))
            lock = cache.inc_lock_ref(node)
            if cache._direct_hicache:
                if cache._donation_barrier is not None:
                    cache._donation_barrier()
                cache.evict(
                    EvictParams(
                        num_tokens=max(0, sizes[CT.FULL] - allocator.full_available_size(rank)),
                        swa_num_tokens=max(0, sizes[CT.SWA] - allocator.swa_available_size(rank)),
                        dp_rank=rank,
                    )
                )
            # JAX restores only into free slots; it never initiates eviction
            # while a prior forward may still own donated device buffers.
            for ct, allocate in ((CT.FULL, allocator.alloc_full), (CT.SWA, allocator.alloc_swa)):
                if sizes[ct]:
                    reserved[ct] = allocate(sizes[ct], dp_rank=rank)
                    if reserved[ct] is None:
                        return empty
            for ct, nodes in selected.items():
                offset = 0
                controller = cache.hicache_controllers[ct]
                for n in nodes:
                    count = len(n.key)
                    tokens = reserved[ct][offset : offset + count]
                    offset += count
                    (transfer,) = cache.components[ct].build_hicache_transfers(
                        n, Phase.LOAD_BACK, device_indices=tokens
                    )
                    transfer = replace(
                        transfer, host_handles=tuple(map(int, n.component_data[ct].host_value))
                    )
                    transfers.append((n, ct, transfer))
                    if cache._direct_hicache:
                        controller.prepare_transfer()
                        operations.append(
                            controller.submit_restore(transfer.host_handles, transfer.device)
                        )
                    else:
                        controller.stage_load(transfer.host_handles)
            if cache._direct_hicache:
                first_error = None
                for operation in operations:
                    try:
                        operation.wait()
                    except Exception as exc:
                        first_error = first_error or exc
                if first_error is not None:
                    raise first_error
            else:
                if cache._donation_barrier is not None:
                    cache._donation_barrier()
                first_error = None
                for controller in cache.hicache_controllers.values():
                    try:
                        controller.drain_loads()
                    except Exception as exc:
                        first_error = first_error or exc
                if first_error is not None:
                    raise first_error
                for n, ct, transfer in transfers:
                    cache.hicache_controllers[ct].flush_load(
                        transfer.host_handles, self.global_pages(ct, transfer.device.pages, rank)
                    )
            full_values = {
                n: np.asarray(t.device_tokens, dtype=np.int32)
                for n, ct, t in transfers
                if ct == CT.FULL
            }
            swa_transfers = [(n, t) for n, ct, t in transfers if ct == CT.SWA]
            if swa_transfers:
                full = np.concatenate(
                    [full_values.get(n, n.component_data[CT.FULL].value) for n, _ in swa_transfers]
                )
                swa = np.concatenate(
                    [np.asarray(t.device_tokens, dtype=np.int32) for _, t in swa_transfers]
                )
                allocator.commit_swa_mapping(full, swa, dp_rank=rank)
            # Release the receipt against the original values; no callbacks or
            # eviction occur between here and publication on this thread.
            cache.dec_lock_ref(node, lock.to_dec_params())
            lock = None
            for n, ct, transfer in transfers:
                cache.components[ct].commit_hicache_transfer(
                    n, Phase.LOAD_BACK, transfers=[transfer]
                )
                counter = (
                    cache.component_protected_size_
                    if n.component_data[ct].lock_ref
                    else cache.component_evictable_size_
                )
                counter[ct][rank] += len(transfer.device_tokens)
            committed = True
            for n in path:
                cache._update_aux_evictable_node_sets(n)
                cache._update_evictable_leaf_sets(n)
            return (
                np.concatenate([n.component_data[CT.FULL].value for n in path])[boundary:],
                node,
                [],
            )
        finally:
            # Submission/validation can fail after an earlier component was
            # submitted. Await every submitted operation before deciding whether
            # its destinations can return to the allocator.
            cleanup_error = None
            if cache._direct_hicache:
                for operation in operations:
                    try:
                        operation.wait()
                    except Exception as exc:
                        cleanup_error = cleanup_error or exc
            healthy = not any(getattr(pool, "failed", False) for pool in cache.host_pools.values())
            if healthy:
                # Drain even when submission failed partway, before releasing
                # destinations that an earlier stage may still be writing.
                if not cache._direct_hicache:
                    for controller in cache.hicache_controllers.values():
                        # Preserve the original stage/flush exception.
                        with suppress(Exception):
                            controller.drain_loads()
                if not committed:
                    for ct, indices in reserved.items():
                        if indices is not None:
                            free = (
                                allocator.free_full if ct == CT.FULL else allocator.free_swa_indices
                            )
                            free(indices, dp_rank=rank)
                if lock is not None:
                    cache.dec_lock_ref(node, lock.to_dec_params())
                for ct, nodes, handles in pins:
                    cache.host_pools[ct].unpin(handles)
                    for n in nodes:
                        n.component_data[ct].host_lock_ref -= 1
            if cleanup_error is not None:
                raise cleanup_error

    def reset(self):
        self.settle(wait=True)
        for controller in self.cache.hicache_controllers.values():
            controller.drain_pending()
            controller.drain_loads()
        stack = list(self.cache.root_node.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            for ct, pool in self.cache.host_pools.items():
                cd = node.component_data[ct]
                if cd.host_value is not None:
                    pool.free(list(map(int, cd.host_value)))
                    cd.host_value = None
