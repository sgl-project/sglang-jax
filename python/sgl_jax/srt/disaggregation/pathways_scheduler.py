"""Scheduler mixin for Pathways single-controller PD.

Holds all `_pd_*` helpers and the P/D worker init so `scheduler.py` only
carries the wiring hooks. See design in issue #1427.
"""

from __future__ import annotations

import copy
import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from queue import Empty as _QEmpty
from queue import Queue as _Queue

import jax
import numpy as np
import psutil

from sgl_jax.srt.disaggregation.pathways_pd import (
    PathwaysPDKVTransfer,
    make_slice_meshes,
    migrate_reqs_p_to_d,
    slots_to_ordered_pages,
)
from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache

logger = logging.getLogger(__name__)


class PathwaysPDSchedulerMixin:
    """Mixin providing single-controller PD scheduling on top of `Scheduler`.

    The base `Scheduler` supplies: mesh/dp_size/tp_size, get_new_batch_prefill,
    process_batch_result_prefill, update_running_batch, running_batch,
    req_to_token_pool, token_to_kv_pool_allocator, tree_cache, model_config,
    enable_overlap, spec_algorithm, page_size, server_args, forward_ct,
    per_dp_max_running_requests, chunked_reqs, _extract_dp_output_ids.
    """

    def _pd_make_meshes(self, server_args) -> None:
        server_args.disable_radix_cache = True
        # PD does not support chunked prefill; raise cps to the P-pool
        # capacity so PrefillAdder never truncates a req mid-batch when
        # packing many short reqs (e.g. 32-way GSM8K -> 6x~700 > 4096).
        server_args.chunked_prefill_size = max(
            server_args.chunked_prefill_size or 0, server_args.pd_prefill_max_tokens
        )
        if not server_args.quantization_config_path:
            # Dynamic quant: cache holds BF16; after quant the fp8 model
            # plus BF16 cache exceeds device HBM (139G>103G on v7x).
            # setdefault so SGLANG_PD_WEIGHT_CACHE=0 can disable for >768G
            # checkpoints that overflow the c4-192 head-node host RAM.
            os.environ.setdefault("SGLANG_PD_WEIGHT_CACHE", "1")
        os.environ.setdefault("SGLANG_MOE_BULK_READ", "1")
        self._pd_n_prefill = max(1, server_args.pd_num_prefill)
        self.p_meshes, self.mesh = make_slice_meshes(self.dp_size, self.tp_size, self._pd_n_prefill)
        self.p_mesh = self.p_meshes[0]

    def _pd_init_workers(self, server_args, model_class, precompile_params, TpWorkerClass) -> None:
        assert self.spec_algorithm is None or self.spec_algorithm.is_none()
        assert not getattr(server_args, "enable_mixed_chunk", False)
        d_mesh = self.mesh
        p_args = copy.deepcopy(server_args)
        p_args.mem_fraction_static = server_args.pd_prefill_mem_fraction
        p_args.disable_radix_cache = True
        d_args = copy.deepcopy(server_args)
        d_args.mem_fraction_static = server_args.pd_decode_mem_fraction
        d_args.disable_radix_cache = True

        n_p = self._pd_n_prefill
        p_meshes = self.p_meshes
        os.environ["SGLANG_CI_SMALL_KV_SIZE"] = str(server_args.pd_prefill_max_tokens)
        self.tp_workers_p = []
        self.p_r2ts, self.p_allocs, self.p_trees = [], [], []
        for i, pm in enumerate(p_meshes):
            logger.info("[pathways_pd] loading P worker %d/%d on %s", i, n_p, pm.shape)
            # P worker stays sync (ModelWorker): prefill must finish writing
            # KV into the P pool before gather_to_dmesh reads it.
            w = ModelWorker(
                server_args=p_args,
                mesh=pm,
                model_class=model_class,
                precompile_params=precompile_params,
            )
            self.tp_workers_p.append(w)
            r2t, alloc = w.get_memory_pool()
            self.p_r2ts.append(r2t)
            self.p_allocs.append(alloc)
            if isinstance(alloc, SWATokenToKVPoolAllocator):
                self.p_trees.append(
                    SWAChunkCache(
                        r2t,
                        alloc,
                        page_size=server_args.page_size,
                        sliding_window_size=w.sliding_window_size,
                    )
                )
            else:
                self.p_trees.append(ChunkCache(r2t, alloc, page_size=server_args.page_size))
        self.tp_worker_p = self.tp_workers_p[0]
        self.p_r2t, self.p_alloc, self.p_tree = (
            self.p_r2ts[0],
            self.p_allocs[0],
            self.p_trees[0],
        )
        logger.info("[pathways_pd] loading D worker on %s", d_mesh.shape)
        os.environ["SGLANG_CI_SMALL_KV_SIZE"] = str(server_args.pd_decode_max_tokens)
        self.tp_worker = TpWorkerClass(
            server_args=d_args,
            mesh=d_mesh,
            model_class=model_class,
            precompile_params=precompile_params,
        )
        del os.environ["SGLANG_CI_SMALL_KV_SIZE"]
        d_kv = self.tp_worker.model_runner.token_to_kv_pool
        d_kv._donate_lock = threading.Lock()
        d_alloc = self.tp_worker.model_runner.token_to_kv_pool_allocator
        self.kv_transfers = [
            PathwaysPDKVTransfer(
                pm,
                d_mesh,
                w.model_runner.token_to_kv_pool,
                d_kv,
                p_alloc=self.p_allocs[i],
                d_alloc=d_alloc,
                page_size=server_args.page_size,
            )
            for i, (pm, w) in enumerate(zip(p_meshes, self.tp_workers_p))
        ]
        self.kv_transfer = self.kv_transfers[0]
        self._pd_prefill_qs = [_Queue(maxsize=2) for _ in range(n_p)]
        self._pd_prefill_q = self._pd_prefill_qs[0]
        self._pd_ready_q: _Queue = _Queue()
        self._pd_qempty = _QEmpty
        # reqs pushed to prefill_q but not yet drained into running_batch;
        # admission must reserve D r2t slots for them (single-item drain
        # lets ready_q backlog while running_batch undercounts).
        self._pd_inflight = 0
        self._pd_next_p = 0
        self._pd_empty_running_p = [
            ScheduleBatch.init_new(
                reqs=[[] for _ in range(self.dp_size)],
                req_to_token_pool=self.p_r2ts[i],
                token_to_kv_pool_allocator=self.p_allocs[i],
                tree_cache=self.p_trees[i],
                model_config=self.model_config,
                enable_overlap=self.enable_overlap,
                dp_size=self.dp_size,
                spec_algorithm=self.spec_algorithm,
                mesh=p_meshes[i],
            )
            for i in range(n_p)
        ]
        self._pd_prefill_threads = [
            threading.Thread(
                target=self._pd_prefill_loop,
                args=(i,),
                name=f"pd-prefill-{i}",
                daemon=True,
            )
            for i in range(n_p)
        ]
        for t in self._pd_prefill_threads:
            t.start()
        self._pd_pending_migrate: ScheduleBatch | None = None
        logger.info(
            "[pathways_pd] n_prefill=%d P pool=%d tok, D pool=%d tok",
            n_p,
            self.tp_worker_p.max_total_num_tokens,
            self.tp_worker.max_total_num_tokens,
        )

    # ---- runtime helpers ----

    def _pd_prefill_loop(self, p_idx: int = 0):
        """Pathways-PD async prefill thread: P-slice forward + cross-slice KV
        gather/device_put run here so the main loop never blocks on P mesh.
        scatter into D pool stays on the main thread (see _pd_drain_ready).
        Multi-P: one thread per P slice, each bound to its own
        worker/pool/transfer; main thread round-robins batches across queues."""
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult

        worker = self.tp_workers_p[p_idx]
        p_r2t = self.p_r2ts[p_idx]
        kv_transfer = self.kv_transfers[p_idx]
        q = self._pd_prefill_qs[p_idx]
        paddings = worker.get_precompile_paddings()
        while True:
            batch = q.get()
            if batch is None:
                return
            try:
                t0 = time.perf_counter()
                mwb = batch.get_model_worker_batch(
                    *paddings, self.page_size, self.server_args.enable_static_lora
                )
                _, ntok_dev, _ = worker.forward_batch_generation(mwb, sampling_metadata=None)
                t_disp = time.perf_counter()
                ntok = np.asarray(jax.device_get(ntok_dev))
                t_fwd = time.perf_counter()
                self._extract_dp_output_ids(ntok, mwb, batch)
                result = GenerationBatchResult(
                    logits_output=None,
                    next_token_ids=ntok.tolist(),
                    extend_input_len_per_req=None,
                    extend_logprob_start_len_per_req=None,
                    bid=mwb.bid,
                    cache_miss_count=0,
                )
                all_reqs = [r for info in batch.reqs_info if info.reqs for r in info.reqs]
                p_pages_per_req = []
                for r in all_reqs:
                    seq_len = len(r.fill_ids)
                    p_slots = p_r2t.req_to_token[r.req_pool_idx, :seq_len].copy()
                    p_pages_per_req.append(
                        (r, p_slots, slots_to_ordered_pages(p_slots, self.page_size))
                    )
                p_pages_all = np.concatenate([p for _, _, p in p_pages_per_req])
                d_stacked, bucket = kv_transfer.gather_to_dmesh(p_pages_all)
                t_gather = time.perf_counter()
                logger.info(
                    "[pd-timing] P fwd disp=%.0fms wait=%.0fms gather=%.0fms reqs=%d",
                    (t_disp - t0) * 1e3,
                    (t_fwd - t_disp) * 1e3,
                    (t_gather - t_fwd) * 1e3,
                    len(all_reqs),
                )
                self._pd_ready_q.put(
                    (p_idx, batch, result, p_pages_per_req, p_pages_all, d_stacked, bucket, t0)
                )
            except Exception:
                logger.exception("[pd_prefill %d] thread error", p_idx)
                psutil.Process().parent().send_signal(signal.SIGQUIT)
                return

    def _pd_drain_ready(self) -> None:
        """Main-thread side of async PD: pop ONE ready item per call (avoid
        burst stalling decode), rewrite req state to D side, scatter, then
        process_prefill under D context so finished reqs release D pool."""
        try:
            item = self._pd_ready_q.get_nowait()
        except self._pd_qempty:
            return
        p_idx, batch, result, p_pages_per_req, p_pages_all, d_stacked, bucket, t0 = item
        p_alloc, p_r2t = self.p_allocs[p_idx], self.p_r2ts[p_idx]
        d_pages_all = []
        _ann = jax.profiler.TraceAnnotation("pd_drain_ready")
        _ann.__enter__()
        for r, p_slots, p_pages in p_pages_per_req:
            seq_len = len(r.fill_ids)
            n_pages = len(p_pages)
            d_slots = self.token_to_kv_pool_allocator.alloc(
                n_pages * self.page_size, dp_rank=r.dp_rank or 0
            )
            if d_slots is None:
                raise RuntimeError("[pd_async] D pool OOM during insert")
            d_pages = slots_to_ordered_pages(d_slots, self.page_size)
            p_slots64 = np.asarray(p_slots, np.int64)
            offsets = p_slots64 % self.page_size
            page_pos = {int(pg): k for k, pg in enumerate(p_pages)}
            tok_page_k = np.array([page_pos[int(s)] for s in p_slots64 // self.page_size], np.int64)
            d_slot_per_tok = (
                d_pages[tok_page_k].astype(np.int64) * self.page_size + offsets
            ).astype(np.int32)
            p_alloc.free(p_slots, dp_rank=r.dp_rank or 0)
            p_r2t.free_slots.append(r.req_pool_idx)
            r.req_pool_idx = None
            self.req_to_token_pool.alloc([r])
            self.req_to_token_pool.req_to_token[r.req_pool_idx, :seq_len] = d_slot_per_tok
            r.prefix_indices = d_slot_per_tok
            r.last_node = None
            r.cache_protected_len = 0
            r.kv_committed_len = seq_len
            r.kv_allocated_len = seq_len
            d_pages_all.append(d_pages)
        _ts0 = time.perf_counter()
        with jax.profiler.TraceAnnotation("pd_scatter"):
            self.kv_transfers[p_idx].scatter_from_dmesh(
                np.concatenate(d_pages_all), d_stacked, bucket
            )
        _ts = (time.perf_counter() - _ts0) * 1e3
        _tq = (time.perf_counter() - t0) * 1e3
        if _ts > 100 or _tq > 3000:
            logger.info(
                "[pd-timing] scatter=%.0fms P->drain=%.0fms reqs=%d",
                _ts,
                _tq,
                len(p_pages_per_req),
            )
        batch.req_to_token_pool = self.req_to_token_pool
        batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        batch.tree_cache = self.tree_cache
        for info in batch.reqs_info:
            if info.reqs:
                info.req_pool_indices = np.array(
                    [r.req_pool_idx for r in info.reqs], dtype=np.int64
                )
        # process_prefill AFTER reqs are on D side: finished reqs (EOS on first
        # token) release_kv_cache against D pool here, not leak.
        with jax.profiler.TraceAnnotation("pd_process_prefill"):
            self.process_batch_result_prefill(batch, result, None)
        self._pd_inflight -= len(p_pages_per_req)
        _ann.__exit__(None, None, None)
        batch.filter_batch()
        if not batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = batch
            else:
                self.running_batch.merge_batch(batch)
        if self.forward_ct % 20 == 0:
            logger.info(
                "[pd_async] insert %d reqs e2e=%.1fms (D avail=%d)",
                len(p_pages_per_req),
                (time.perf_counter() - t0) * 1e3,
                self.token_to_kv_pool_allocator.available_size(),
            )

    def _pd_get_next_batch_async(self):
        """Pathways-PD async scheduling: main loop only ever returns decode
        batches; prefill batches are pushed to _pd_prefill_qs[i] (round-robin
        across P slices) and run on prefill threads concurrently with D decode."""
        self._pd_drain_ready()
        n_p = self._pd_n_prefill
        for _ in range(n_p):
            i = self._pd_next_p
            self._pd_next_p = (i + 1) % n_p
            if self._pd_prefill_qs[i].full():
                continue
            saved = self.per_dp_max_running_requests
            d_running = sum(len(x.reqs) for x in self.running_batch.reqs_info if x.reqs)
            # swap_p_pool replaces running_batch with an empty one so PrefillAdder
            # stops mixing D-side future-token reservation into the P pool budget;
            # the D r2t slot constraint moves here explicitly (was implicit via
            # get_new_batch_prefill's len(running_batch.reqs) check).
            self.per_dp_max_running_requests = max(0, saved - self._pd_inflight - d_running)
            try:
                with self._pd_swap_p_pool(i):
                    new_batch = self.get_new_batch_prefill()
            finally:
                self.per_dp_max_running_requests = saved
            if new_batch is None:
                break
            assert all(
                r is None for r in self.chunked_reqs
            ), "[pd_async] chunked prefill not supported (single-req IL exceeds pd_prefill_max_tokens)"
            assert not any(
                r.return_logprob for info in new_batch.reqs_info if info.reqs for r in info.reqs
            ), "[pd_async] return_logprob / hidden_states not yet supported (async prefill drops logits_output)"
            self._pd_inflight += sum(len(info.reqs) for info in new_batch.reqs_info if info.reqs)
            self._pd_prefill_qs[i].put_nowait(new_batch)
            break
        if not self.running_batch.is_empty() and not self.running_batch.is_prefill_only:
            self.running_batch = self.update_running_batch(self.running_batch)
            return self.running_batch if not self.running_batch.is_empty() else None
        return None

    def _pd_gather_output(self, arr, is_prefill: bool) -> np.ndarray:
        """Sub-mesh forward output (addressable on one side only) -> full numpy."""
        return np.asarray(jax.device_get(arr))

    @contextmanager
    def _pd_swap_p_pool(self, p_idx: int = 0):
        """Temporarily point self.{tree_cache, req_to_token_pool,
        token_to_kv_pool_allocator, mesh} at the P[p_idx] side while
        get_new_batch_prefill / process_batch_result_prefill run.
        """
        saved = (
            self.tree_cache,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.mesh,
            self.running_batch,
            self.max_total_num_tokens,
        )
        self.tree_cache = self.p_trees[p_idx]
        self.req_to_token_pool = self.p_r2ts[p_idx]
        self.token_to_kv_pool_allocator = self.p_allocs[p_idx]
        self.max_total_num_tokens = self.tp_workers_p[p_idx].max_total_num_tokens
        self.mesh = self.p_meshes[p_idx]
        # PrefillAdder reads running_batch for rem_total_tokens AND writes
        # batch_is_full back into it on NO_TOKEN; without this swap the
        # D-side decode reqs' future-token reservation starves the P pool
        # admission and the sticky batch_is_full stalls push until a D req
        # finishes (multi-P 1.16x root cause).
        empty = self._pd_empty_running_p[p_idx]
        for info in empty.reqs_info:
            info.batch_is_full = False
        self.running_batch = empty
        try:
            yield
        finally:
            (
                self.tree_cache,
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.mesh,
                self.running_batch,
                self.max_total_num_tokens,
            ) = saved

    def _pd_migrate(self, batch: ScheduleBatch) -> bool:
        """Migrate reqs (prefill done, chunked/finished already filtered) from
        the P pool into the D pool (KV transfer + r2t/alloc/req rewrite) so
        they can merge into the D running_batch. Returns False if D pool lacks
        space (caller should park and retry after D frees)."""
        all_reqs = []
        for info in batch.reqs_info:
            if info.reqs:
                all_reqs.extend(info.reqs)
        if not all_reqs:
            return True
        need = sum(
            ((len(r.fill_ids) + self.page_size - 1) // self.page_size) * self.page_size
            for r in all_reqs
        )
        if self.token_to_kv_pool_allocator.available_size() < need:
            return False
        t0 = time.perf_counter()
        migrate_reqs_p_to_d(
            all_reqs,
            self.page_size,
            self.p_r2t,
            self.p_alloc,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.kv_transfer,
        )
        # Rebind batch metadata to D pool (merge_batch doesn't check, but
        # prepare_for_decode reads these).
        batch.req_to_token_pool = self.req_to_token_pool
        batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        batch.tree_cache = self.tree_cache
        for info in batch.reqs_info:
            if info.reqs:
                info.req_pool_indices = np.array(
                    [r.req_pool_idx for r in info.reqs], dtype=np.int64
                )
        if self.forward_ct % 50 == 0:
            logger.info(
                "[pathways_pd] migrate %d reqs in %.1fms (D avail=%d)",
                len(all_reqs),
                (time.perf_counter() - t0) * 1e3,
                self.token_to_kv_pool_allocator.available_size(),
            )
        return True
