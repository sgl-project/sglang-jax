"""Scheduler mixin for Pathways single-controller PD.

Holds all `_pd_*` helpers and the P/D worker init so `scheduler.py` only
carries the wiring hooks. See design in issue #1427.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from queue import Empty as _QEmpty
from queue import Queue as _Queue

import jax
import numpy as np

from sgl_jax.srt.disaggregation.pathways_pd import (
    PathwaysPDKVTransfer,
    group_pages_by_dp_rank,
    make_slice_meshes,
    migrate_reqs_p_to_d,
    slots_to_ordered_pages,
)
from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache

logger = logging.getLogger(__name__)
_PD_DBG = bool(os.environ.get("SGLANG_PD_DBG"))
_PD_DBG_TIMING = bool(os.environ.get("SGLANG_PD_DBG_TIMING")) or _PD_DBG


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
        if not server_args.quantization_config_path:
            # Dynamic quant: cache holds BF16; after quant the fp8 model
            # plus BF16 cache can exceed device HBM. setdefault so
            # SGLANG_PD_WEIGHT_CACHE=0 can disable for large checkpoints
            # that overflow head-node host RAM.
            os.environ.setdefault("SGLANG_PD_WEIGHT_CACHE", "1")
        os.environ.setdefault("SGLANG_MOE_BULK_READ", "1")
        self._pd_n_prefill = max(1, server_args.pd_num_prefill)
        self._pd_n_decode = max(1, server_args.pd_num_decode)
        # Hetero-TP: --tp-size is D (base Scheduler + main event loop run on
        # d_mesh); --pd-prefill-tp-size overrides P only. 0 = homogeneous.
        self._pd_tp_p = server_args.pd_prefill_tp_size or self.tp_size
        if self._pd_tp_p != self.tp_size:
            # weight_cache._pd_remap_sharding assumes P/D meshes are same size
            # (1:1 dev id map); hetero-tp shardings differ so cache can't be
            # reused across sides -- each loads its own (slower but correct).
            os.environ["SGLANG_PD_WEIGHT_CACHE"] = "0"
        self.p_meshes, self.d_meshes = make_slice_meshes(
            self.dp_size,
            self.tp_size,
            self._pd_n_prefill,
            self._pd_n_decode,
            tp_prefill=self._pd_tp_p,
        )
        self.p_mesh = self.p_meshes[0]
        self.mesh = self.d_meshes[0]

    def _pd_init_workers(self, server_args, model_class, precompile_params, TpWorkerClass) -> None:
        assert self.spec_algorithm is None or self.spec_algorithm.is_none()
        assert not getattr(server_args, "enable_mixed_chunk", False)
        p_args = copy.deepcopy(server_args)
        p_args.mem_fraction_static = server_args.pd_prefill_mem_fraction
        p_args.disable_radix_cache = True
        if self._pd_tp_p != self.tp_size:
            # ModelWorker/ModelRunner read tp/ep from server_args (tp_worker.py
            # :75/:130, moe.py:79 world_size from mesh); mesh alone isn't
            # enough -- attention_tp = tp_size//dp_size drives kv_heads
            # replication + validate_tensor_parallel_config. ep must divide
            # both p_mesh world_size and num_experts; default = keep D's
            # ep/tp ratio (pure-EP stays pure-EP on the bigger P slice).
            p_args.tp_size = self._pd_tp_p
            p_args.ep_size = server_args.pd_prefill_ep_size or max(
                1, server_args.ep_size * self._pd_tp_p // self.tp_size
            )
            assert (
                p_args.tp_size % p_args.ep_size == 0
            ), f"derived P ep_size={p_args.ep_size} must divide P tp_size={p_args.tp_size}; set --pd-prefill-ep-size explicitly"
            logger.info(
                "[pathways_pd] hetero-tp P: tp=%d ep=%d (D: tp=%d ep=%d)",
                p_args.tp_size,
                p_args.ep_size,
                self.tp_size,
                server_args.ep_size,
            )
        d_args = copy.deepcopy(server_args)
        d_args.mem_fraction_static = server_args.pd_decode_mem_fraction
        d_args.disable_radix_cache = True

        n_p = self._pd_n_prefill
        p_meshes = self.p_meshes
        p_args.max_total_tokens = server_args.pd_prefill_max_tokens
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
        n_d = self._pd_n_decode
        d_args.max_total_tokens = server_args.pd_decode_max_tokens
        self.tp_workers_d = []
        self.d_kvs, self.d_allocs = [], []
        for j, dm in enumerate(self.d_meshes):
            logger.info("[pathways_pd] loading D worker %d/%d on %s", j, n_d, dm.shape)
            wd = TpWorkerClass(
                server_args=d_args,
                mesh=dm,
                model_class=model_class,
                precompile_params=precompile_params,
            )
            self.tp_workers_d.append(wd)
            d_kv = wd.model_runner.token_to_kv_pool
            d_kv._donate_lock = threading.Lock()
            self.d_kvs.append(d_kv)
            self.d_allocs.append(wd.model_runner.token_to_kv_pool_allocator)
        # Base Scheduler.__init__ reads self.tp_worker for D0 pool/tree/running_batch;
        # D1.. get their own copies in _pd_init_decode_extras (called post-init).
        self.tp_worker = self.tp_workers_d[0]
        sw = getattr(self.tp_worker.worker.model_runner, "sliding_window_size", 0) or 0
        swa_tail = -(-(sw + server_args.page_size) // server_args.page_size) if sw else 0
        # kv_transfers[p_idx][d_idx]: each (P,D) pair needs its own gather/scatter
        # jits (gather runs on P mesh, d_stack_shard + scatter on D mesh). A done
        # batch goes to ONE D (batch-level RR via _pd_next_d), not broadcast --
        # 1P-ND is fan-out routing, not replication.
        self.kv_transfers = [
            [
                PathwaysPDKVTransfer(
                    pm,
                    dm,
                    w.model_runner.token_to_kv_pool,
                    self.d_kvs[j],
                    p_alloc=self.p_allocs[i],
                    d_alloc=self.d_allocs[j],
                    page_size=server_args.page_size,
                    swa_tail_pages=swa_tail,
                )
                for j, dm in enumerate(self.d_meshes)
            ]
            for i, (pm, w) in enumerate(zip(p_meshes, self.tp_workers_p))
        ]
        self.kv_transfer = self.kv_transfers[0][0]
        # Base Scheduler skips run_precompile under PD (`and not self.pd` in
        # scheduler.py), so P/D workers would first-compile on the first real
        # request (~minutes on 78L models). Precompile all of them here, plus
        # the KV-transfer gather/[stack/reshard/unstack]/scatter per bucket.
        if not server_args.disable_precompile:
            for i, w in enumerate(self.tp_workers_p):
                logger.info("[pathways_pd] precompiling P worker %d forward (extend only)", i)
                w.run_precompile(only="extend")
            for j, w in enumerate(self.tp_workers_d):
                logger.info("[pathways_pd] precompiling D worker %d forward (decode only)", j)
                w.run_precompile(only="decode")
            for row in self.kv_transfers:
                for kt in row:
                    kt.precompile()
        # Per-dp RR counters: chunked prefill emits done batches of ~1 req in
        # strict dp0,dp1,dp0,.. order (select_dp_for_request round_robin), so a
        # single global RR counter phase-locks with dp -> dp0 all land on D0,
        # dp1 all on D1 -> each D runs half its dp ranks empty. Per-dp counters
        # give each dp its own D0->D1->D0 rotation, breaking the lock.
        self._pd_next_d_per_dp = [0] * max(8, self.server_args.dp_size)
        self._pd_next_d_lock = threading.Lock()
        self._pd_prefill_qs = [_Queue(maxsize=2) for _ in range(n_p)]
        self._pd_prefill_q = self._pd_prefill_qs[0]
        # cap ready_q so at most `max_inflight_transfers` gathered d_stacked
        # buffers sit on D HBM waiting for scatter; prefill thread blocks on
        # put() until the main loop drains one.
        _max_inflight = max(1, server_args.disaggregation_max_inflight_transfers)
        self._pd_ready_q: _Queue = _Queue(maxsize=_max_inflight)
        self._pd_qempty = _QEmpty
        # a chunked req's chunk-N is in the prefill pipeline; main thread must
        # drain it (updates prefix_indices) before building chunk-N+1. Per-P:
        # each P slice has its own chunked_reqs (swapped in _pd_swap_p_pool) so
        # req_A's chunk-2..N stay on the same P that holds chunk-1's KV.
        self._pd_chunk_pending = [False] * n_p
        self.p_chunked_reqs = [[None] * self.dp_size for _ in range(n_p)]
        self._pd_t_pending_cleared = 0.0
        self._pd_t_last_get_batch = 0.0
        # ready_q items whose D-pool alloc was deferred (avail < need+reserved)
        self._pd_defer: list = []
        # reqs pushed to prefill_q but not yet drained into running_batch;
        # tracked by rid so a chunked req counts once, not once-per-chunk.
        self._pd_inflight_rids: set[str] = set()
        self._pd_reserved_per = server_args.disaggregation_num_reserved_decode_tokens
        self._pd_sliding_window = self.tp_workers_p[0].sliding_window_size or 0
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
            "[pathways_pd] n_prefill=%d n_decode=%d P pool=%d tok, D pool=%d tok",
            n_p,
            n_d,
            self.tp_worker_p.max_total_num_tokens,
            self.tp_worker.max_total_num_tokens,
        )

    def _pd_init_decode_extras(self) -> None:
        """Post base-__init__ setup for D1..N-1. Base __init__ already built
        self.{req_to_token_pool, token_to_kv_pool_allocator, tree_cache,
        running_batch} from tp_workers_d[0]; mirror those for the remaining
        D workers so _pd_swap_d_pool can rotate the whole set."""
        from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache

        self.d_r2ts = [self.req_to_token_pool]
        self.d_trees = [self.tree_cache]
        self.d_running_batches = [self.running_batch]
        for j in range(1, self._pd_n_decode):
            wd = self.tp_workers_d[j]
            r2t, alloc = wd.get_memory_pool()
            assert alloc is self.d_allocs[j]
            self.d_r2ts.append(r2t)
            tree = build_kv_cache(
                server_args=self.server_args,
                model_config=self.model_config,
                req_to_token_pool=r2t,
                token_to_kv_pool_allocator=alloc,
                page_size=self.page_size,
                is_hybrid=self.is_hybrid,
                sliding_window_size=self.sliding_window_size,
                tp_size=self.tp_size,
                spec_algorithm=self.spec_algorithm,
            )
            self.d_trees.append(tree)
            self.d_running_batches.append(
                ScheduleBatch.init_new(
                    reqs=[[] for _ in range(self.dp_size)],
                    req_to_token_pool=r2t,
                    token_to_kv_pool_allocator=alloc,
                    tree_cache=tree,
                    model_config=self.model_config,
                    enable_overlap=self.enable_overlap,
                    dp_size=self.dp_size,
                    spec_algorithm=self.spec_algorithm,
                    mesh=self.d_meshes[j],
                )
            )
        # slot 0 aliases the base scheduler attrs; keep in sync at swap boundaries.
        self.d_allocs[0] = self.token_to_kv_pool_allocator
        self._pd_cur_d = 0

    def _pd_set_d(self, d_idx: int) -> None:
        """Point self.{tp_worker, tree_cache, req_to_token_pool,
        token_to_kv_pool_allocator, mesh, running_batch} at D[d_idx].

        Stateful (not a context manager): saves the CURRENT self.running_batch
        back into d_running_batches[_pd_cur_d] first so any reassignment
        (e.g. `self.running_batch = batch` in _pd_drain_ready) is captured.
        _pd_swap_p_pool still layers on top via save/restore of whatever D
        is loaded. n_decode==1: always d_idx==0 == cur_d, no-op."""
        cur = self._pd_cur_d
        if cur == d_idx:
            return
        self.d_running_batches[cur] = self.running_batch
        self._pd_cur_d = d_idx
        self.tp_worker = self.tp_workers_d[d_idx]
        self.tree_cache = self.d_trees[d_idx]
        self.req_to_token_pool = self.d_r2ts[d_idx]
        self.token_to_kv_pool_allocator = self.d_allocs[d_idx]
        self.mesh = self.d_meshes[d_idx]
        self.running_batch = self.d_running_batches[d_idx]

    def _pd_total_d_running(self) -> int:
        if self._pd_n_decode == 1:
            return sum(len(x.reqs) for x in self.running_batch.reqs_info if x.reqs)
        # d_running_batches[cur_d] may be stale; self.running_batch is authoritative
        tot = 0
        for j, rb in enumerate(self.d_running_batches):
            if j == self._pd_cur_d:
                rb = self.running_batch
            tot += sum(len(x.reqs) for x in rb.reqs_info if x.reqs)
        return tot

    # ---- runtime helpers ----

    def _pd_prefill_loop(self, p_idx: int = 0):
        """Pathways-PD async prefill thread: P-slice forward + cross-slice KV
        gather/device_put run here so the main loop never blocks on P mesh.
        scatter into D pool stays on the main thread (see _pd_drain_ready).

        chunked prefill: batch.reqs_info[dp].chunked_req marks the mid-chunk
        req (still has more chunks queued on the main thread). Mid-chunk reqs
        are NOT gathered — they stay in the P pool; only done reqs (final
        chunk) are gathered + P-freed here. This thread updates the mid req's
        prefix_indices (== ChunkCache.cache_unfinished_req, which neither
        ChunkCache nor SWAChunkCache-derived state needs beyond this numpy
        read) and clears _pd_chunk_pending itself, right after forward, so
        the main loop can build chunk N+1 without waiting a full drain cycle
        (was ~104ms/chunk of host round-trip; see [pd-chunk-cycle] log)."""
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult

        worker = self.tp_workers_p[p_idx]
        p_r2t = self.p_r2ts[p_idx]
        kv_transfers_d = self.kv_transfers[p_idx]  # [d_idx] -> transfer
        n_d = self._pd_n_decode
        q = self._pd_prefill_qs[p_idx]
        paddings = worker.get_precompile_paddings()

        def _disp(b):
            _t0 = time.perf_counter()
            _mwb = b.get_model_worker_batch(
                *paddings, self.page_size, self.server_args.enable_static_lora
            )
            _, _ntok, _ = worker.forward_batch_generation(_mwb, sampling_metadata=None)
            return _mwb, _ntok, _t0, time.perf_counter()

        def _is_mid_only(b):
            return all(
                info.chunked_req is not None and len(info.reqs or ()) <= 1
                for info in b.reqs_info
                if info.reqs
            )

        _stash = None  # pre-dispatched (batch, mwb, ntok_dev, t0, t_disp)
        while True:
            if _stash is not None:
                batch, mwb, ntok_dev, t0, t_disp = _stash
                _stash = None
            else:
                batch = q.get()
                if batch is None:
                    return
                mwb, ntok_dev, t0, t_disp = _disp(batch)
            t_enq_p = getattr(batch, "_pd_t_enqueue_prefill", None)
            try:
                # depth-1 pipeline: mid-only chunk clears pending right after
                # dispatch (prefix_indices readable now — prepare_for_extend
                # wrote P r2t pre-dispatch), then eagerly pull+dispatch N+1 so
                # its mwb+disp hides under N's device wait. Mirrors the
                # multi-process PD prefill overlap loop. done-chunk keeps late clear
                # to guard gather-abort slot double-use. Stash chunks ALSO
                # eager-dispatch (unbounded depth is safe: _stash is scalar so
                # depth stays 1; _pd_eager_cleared prevents double-build).
                if _is_mid_only(batch):
                    for info in batch.reqs_info:
                        r = info.chunked_req
                        if r is not None:
                            r.prefix_indices = p_r2t.req_to_token[
                                r.req_pool_idx, : len(r.fill_ids)
                            ].copy()
                    self._pd_chunk_pending[p_idx] = False
                    self._pd_t_pending_cleared = t_disp
                    batch._pd_eager_cleared = True
                    try:
                        nxt = q.get(timeout=0.04)
                    except _QEmpty:
                        nxt = None
                    if nxt is None:
                        pass  # timeout or sentinel; sentinel re-read next iter via q.put below
                    else:
                        try:
                            _stash = (nxt, *_disp(nxt))
                        except Exception as _e:
                            logger.exception("[pd_prefill %d] eager-stash dispatch failed", p_idx)
                            for info in nxt.reqs_info:
                                for r in info.reqs or ():
                                    r.set_finish_with_abort(f"PD eager-stash error: {_e}")
                            _tn = time.perf_counter()
                            self._pd_ready_q.put((p_idx, 0, nxt, None, [], None, None, 0, _tn, _tn))
                ntok = np.asarray(jax.device_get(ntok_dev))
                if os.environ.get("SGLANG_PD_DBG_NTOK"):
                    try:
                        _sh = [
                            (
                                s.device.slice_index,
                                str(s.index),
                                np.asarray(s.data).flatten()[:4].tolist(),
                            )
                            for s in ntok_dev.addressable_shards
                        ]
                    except Exception as _e:
                        _sh = f"err:{_e}"
                    logger.info(
                        "[ntok-shard p%d] shape=%s spec=%s full=%s shards=%s",
                        p_idx,
                        ntok.shape,
                        getattr(ntok_dev.sharding, "spec", "?"),
                        ntok.flatten().tolist()[:16],
                        _sh,
                    )
                t_fwd = time.perf_counter()
                _rids = [r.rid[:8] for info in batch.reqs_info if info.reqs for r in info.reqs]
                _plens = [
                    len(r.prefix_indices)
                    for info in batch.reqs_info
                    if info.reqs
                    for r in info.reqs
                ]
                if _PD_DBG:
                    logger.info(
                        "[pd-dbg p%d] rids=%s prefix_lens=%s ntok=%s pool_idx=%s",
                        p_idx,
                        _rids,
                        _plens,
                        ntok.flatten()[:4].tolist(),
                        [
                            r.req_pool_idx
                            for info in batch.reqs_info
                            if info.reqs
                            for r in info.reqs
                        ],
                    )
                self._extract_dp_output_ids(ntok, mwb, batch)
                result = GenerationBatchResult(
                    logits_output=None,
                    next_token_ids=ntok.tolist(),
                    extend_input_len_per_req=None,
                    extend_logprob_start_len_per_req=None,
                    bid=mwb.bid,
                    cache_miss_count=0,
                )
                mid_reqs = [
                    info.chunked_req for info in batch.reqs_info if info.chunked_req is not None
                ]
                mid_set = {id(r) for r in mid_reqs}
                all_reqs = [r for info in batch.reqs_info if info.reqs for r in info.reqs]
                done_per_req = []
                for r in all_reqs:
                    if id(r) in mid_set:
                        continue
                    seq_len = len(r.fill_ids)
                    p_slots = p_r2t.req_to_token[r.req_pool_idx, :seq_len].copy()
                    if _PD_DBG:
                        logger.info(
                            "[pd-done-dbg p%d] rid=%s seq_len=%d n_pages=%d pool_idx=%d prefix=%d",
                            p_idx,
                            r.rid[:8],
                            seq_len,
                            (seq_len + self.page_size - 1) // self.page_size,
                            r.req_pool_idx,
                            len(r.prefix_indices),
                        )
                    done_per_req.append(
                        (r, p_slots, slots_to_ordered_pages(p_slots, self.page_size))
                    )
                # prefix_indices is a pure read of the P r2t table (already
                # written by this batch's prepare_for_extend before dispatch)
                # -- safe to update regardless of what happens below.
                for r in mid_reqs:
                    r.prefix_indices = p_r2t.req_to_token[r.req_pool_idx, : len(r.fill_ids)].copy()
                if done_per_req:
                    # Batch-level RR to a single D target: gather once on P mesh
                    # then device_put to d_meshes[d_idx]. done batches are small
                    # (chunked prefill -> done=0~2) so batch-RR ~= req-RR for
                    # load balance without splitting one gather across D targets.
                    # RR counter is per-dp so dp-alternating done order doesn't
                    # phase-lock dp<->D (see _pd_next_d_per_dp init comment).
                    lead_dp = done_per_req[0][0].dp_rank or 0
                    with self._pd_next_d_lock:
                        d_idx = self._pd_next_d_per_dp[lead_dp] % n_d
                        self._pd_next_d_per_dp[lead_dp] += 1
                    kv_transfer = kv_transfers_d[d_idx]
                    _dpl = [(r.dp_rank or 0, p) for r, _, p in done_per_req]
                    p_pages_all = self._pd_group(_dpl, kv_transfer.p_dp_size)
                    _tail = kv_transfer._swa_tail_pages
                    p_swa_all = (
                        self._pd_group([(dr, p[-_tail:]) for dr, p in _dpl], kv_transfer.p_dp_size)
                        if _tail
                        else None
                    )
                    d_stacked, bucket = kv_transfer.gather_to_dmesh(p_pages_all, p_swa_all)
                else:
                    d_idx, p_pages_all, d_stacked, bucket = 0, None, None, 0
                t_gather = time.perf_counter()
                # Only now -- after gather_to_dmesh succeeded for any done_per_req
                # in this SAME batch -- is it safe to let the main loop build
                # chunk N+1 reusing this req's req_pool_idx. Clearing pending
                # any earlier (e.g. right after the prefix_indices update above)
                # would let chunk N+1 be built+queued before we know whether a
                # gather failure below is about to abort this whole batch
                # (including mid_reqs) and free that same req_pool_idx out from
                # under the newly-queued chunk N+1 -- a slot double-use.
                if mid_reqs and not getattr(batch, "_pd_eager_cleared", False):
                    self._pd_chunk_pending[p_idx] = False
                    self._pd_t_pending_cleared = time.perf_counter()
                if _PD_DBG_TIMING:
                    logger.info(
                        "[pd-timing] P fwd pq_wait=%.0fms disp=%.0fms wait=%.0fms gather=%.0fms "
                        "done=%d mid=%d",
                        (t0 - t_enq_p) * 1e3 if t_enq_p else -1.0,
                        (t_disp - t0) * 1e3,
                        (t_fwd - t_disp) * 1e3,
                        (t_gather - t_fwd) * 1e3,
                        len(done_per_req),
                        len(mid_set),
                    )
                _t_put0 = time.perf_counter()
                self._pd_ready_q.put(
                    (
                        p_idx,
                        d_idx,
                        batch,
                        result,
                        done_per_req,
                        p_pages_all,
                        d_stacked,
                        bucket,
                        t0,
                        t_gather,
                    )
                )
                _put_ms = (time.perf_counter() - _t_put0) * 1e3
                if _PD_DBG_TIMING and _put_ms > 10:
                    logger.info(
                        "[pd-put-block] ready_q.put blocked=%.0fms qsize=%d",
                        _put_ms,
                        self._pd_ready_q.qsize(),
                    )
            except Exception as e:
                logger.exception("[pd_prefill %d] batch failed, aborting reqs", p_idx)
                for info in batch.reqs_info:
                    for r in info.reqs or ():
                        r.set_finish_with_abort(f"PD prefill error: {e}")
                _tnow = time.perf_counter()
                self._pd_ready_q.put((p_idx, 0, batch, None, [], None, None, 0, _tnow, _tnow))

    @staticmethod
    def _pd_group(rank_pages: list[tuple[int, np.ndarray]], dp_size: int) -> np.ndarray:
        """(dp_rank, pages)[] -> flat concat (dp==1) or [dp_size, n] (dp>1).

        dp>1 empty-rank rows are left as page-index 0 (see
        group_pages_by_dp_rank). This is SAFE: allocator.py never hands out
        index 0 (`arange(1, size+1)`), so page 0 on both P and D pools is a
        reserved sentinel no req's slots ever point at -- gather reads
        uninitialised-zero and scatter overwrites the same never-read slot.
        """
        if dp_size == 1:
            return np.concatenate([p for _, p in rank_pages])
        pages, valid = group_pages_by_dp_rank(rank_pages, dp_size)
        _prr = valid.sum(axis=1).tolist()
        if _PD_DBG and (len(set(_prr)) > 1 or 0 in _prr):
            logger.info(
                "[pd-group-dbg] UNBALANCED per_rank_real=%s max_n=%d in=%s",
                _prr,
                pages.shape[1],
                [(dr, len(p)) for dr, p in rank_pages],
            )
        return pages

    def _pd_drain_ready(self) -> None:
        """Main-thread side of async PD: pop ONE ready item, migrate done reqs
        to D pool (capacity-gated: defer when avail < need+reserved instead of
        raising), scatter, process_prefill, then merge into running_batch.
        Mid-chunk reqs stay on P — cache_unfinished_req updates their
        prefix_indices so the next get_new_batch_prefill builds chunk N+1."""
        if self._pd_defer:
            item = self._pd_defer.pop(0)
        else:
            try:
                item = self._pd_ready_q.get_nowait()
            except self._pd_qempty:
                return
        t_dequeue_ready = time.perf_counter()
        (
            p_idx,
            d_idx,
            batch,
            result,
            done_per_req,
            p_pages_all,
            d_stacked,
            bucket,
            t0,
            t_enq_ready,
        ) = item
        p_alloc, p_r2t, p_tree = self.p_allocs[p_idx], self.p_r2ts[p_idx], self.p_trees[p_idx]
        chunked_excl = {
            dp: info.chunked_req
            for dp, info in enumerate(batch.reqs_info)
            if info.chunked_req is not None
        }
        if result is None:
            # prefill thread reported failure; free P, clear inflight/chunked.
            # TODO: proper stream-abort to client (currently client times out).
            all_reqs = []
            for info in batch.reqs_info:
                for r in info.reqs or ():
                    if r.req_pool_idx is not None:
                        p_tree.cache_finished_req(r)
                        p_r2t.free_slots.append(r.req_pool_idx)
                    self._pd_inflight_rids.discard(r.rid)
                    all_reqs.append(r)
            self.stream_output(all_reqs, False, False)
            for dp in chunked_excl:
                self.p_chunked_reqs[p_idx][dp] = None
            self._pd_chunk_pending[p_idx] = False
            return

        # Load D[d_idx] state so allocator/r2t/tree_cache/running_batch below all
        # point at the target D. n_decode==1: no-op (d_idx==_pd_cur_d==0).
        self._pd_set_d(d_idx)
        allocator = self.token_to_kv_pool_allocator
        if done_per_req:
            n_running = sum(len(x.reqs) for x in self.running_batch.reqs_info if x.reqs)
            reserved = self._pd_reserved_per * (n_running + len(self._pd_inflight_rids))
            reserved_per_rank = reserved // max(1, self.dp_size)
            need_by_rank = defaultdict(int)
            for r, _, p in done_per_req:
                need_by_rank[r.dp_rank or 0] += len(p) * self.page_size
            if any(
                nd + reserved_per_rank > allocator.available_size(dp_rank=rk)
                for rk, nd in need_by_rank.items()
            ):
                self._pd_defer.append(item)
                return
        d_pages_all = []
        for r, p_slots, p_pages in done_per_req:
            seq_len = len(r.fill_ids)
            n_pages = len(p_pages)
            d_slots = allocator.alloc(n_pages * self.page_size, dp_rank=r.dp_rank or 0)
            if d_slots is None:
                raise RuntimeError("[pd_async] D pool OOM after capacity gate")
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
            d_pages_all.append((r.dp_rank or 0, d_pages))
        if d_pages_all:
            _ts0 = time.perf_counter()
            kt = self.kv_transfers[p_idx][d_idx]
            _dpg = self._pd_group(d_pages_all, kt.d_dp_size)
            _tail = kt._swa_tail_pages
            _dsw = (
                self._pd_group([(dr, p[-_tail:]) for dr, p in d_pages_all], kt.d_dp_size)
                if _tail
                else None
            )
            if _PD_DBG:
                logger.info(
                    "[pd-scatter-dbg p%d] dpg=%s rids=%s",
                    p_idx,
                    np.asarray(_dpg).reshape(kt.d_dp_size, -1)[:, :3].tolist(),
                    [(r.rid[:8], r.dp_rank) for r, _, _ in done_per_req],
                )
            with jax.profiler.TraceAnnotation("pd_scatter"):
                kt.scatter_from_dmesh(_dpg, d_stacked, bucket, _dsw)
            # PD migrate hands D the full seq_len of swa slots (P kept them all
            # for chunked prefill), but decode only needs the last sliding
            # window. Without this immediate evict the D swa pool fills at
            # ~seq_len/req (MiMo 16K/4K: 12.5K/req vs window=128) and caps
            # running at ~56 (swa usage 0.91 never dropped -- the standard
            # maybe_evict_swa path didn't fire on migrated reqs). Mirrors
            # multi-process PD which only pulls the window-tail swa pages. Must
            # run AFTER scatter (freed slots could otherwise be re-alloc'd by
            # a later req in this same done_per_req loop and then overwritten
            # by scatter).
            if self._pd_sliding_window and hasattr(allocator, "free_swa"):
                for r, _, _ in done_per_req:
                    seq_len = len(r.fill_ids)
                    tail = (
                        (seq_len - self._pd_sliding_window - self.page_size) // self.page_size
                    ) * self.page_size
                    if tail > r.swa_evicted_seqlen:
                        allocator.free_swa(
                            self.req_to_token_pool.req_to_token[r.req_pool_idx, :tail],
                            dp_rank=r.dp_rank or 0,
                        )
                        r.swa_evicted_seqlen = tail
            if _PD_DBG_TIMING:
                logger.info(
                    "[pd-timing] scatter_disp=%.0fms P->drain=%.0fms reqs=%d",
                    (time.perf_counter() - _ts0) * 1e3,
                    (time.perf_counter() - t0) * 1e3,
                    len(done_per_req),
                )
        batch.req_to_token_pool = self.req_to_token_pool
        batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        batch.tree_cache = self.tree_cache
        for info in batch.reqs_info:
            if info.reqs:
                info.req_pool_indices = np.array(
                    [r.req_pool_idx for r in info.reqs], dtype=np.int64
                )
        with jax.profiler.TraceAnnotation("pd_process_prefill"):
            self.process_batch_result_prefill(batch, result, None)
        for r in chunked_excl.values():
            # Redundant with the prefill thread's own update (same numpy read,
            # done right after forward — see _pd_prefill_loop) but harmless
            # and kept as a safety net. _pd_chunk_pending is NOT touched here:
            # the prefill thread is now the sole authority for clearing it
            # (success path), right after forward completes, so the main loop
            # doesn't have to wait for this drain cycle to build chunk N+1.
            # Only the error path (result is None, above) still clears it —
            # that's a distinct failure recovery, not this normal-path drain.
            p_tree.cache_unfinished_req(r)
        if _PD_DBG_TIMING and chunked_excl:
            logger.info(
                "[pd-chunk-cycle] rq_wait=%.0fms drain=%.0fms (fwd_done->dequeue->drain_done)",
                (t_dequeue_ready - t_enq_ready) * 1e3,
                (time.perf_counter() - t_dequeue_ready) * 1e3,
            )
        for r, _, _ in done_per_req:
            self._pd_inflight_rids.discard(r.rid)
        if _PD_DBG and done_per_req and hasattr(p_alloc, "swa_available_size"):
            logger.info(
                "[pd-diag] after free %d done: P full_avail=%d swa_avail=%d",
                len(done_per_req),
                p_alloc.full_available_size(),
                p_alloc.swa_available_size(),
            )
        batch.filter_batch(chunked_req_to_exclude=chunked_excl)
        if not batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = batch
            else:
                self.running_batch.merge_batch(batch)
        if self.forward_ct % 20 == 0:
            logger.info(
                "[pd_async] drain done=%d mid=%d e2e=%.1fms (D avail=%d)",
                len(done_per_req),
                len(chunked_excl),
                (time.perf_counter() - t0) * 1e3,
                allocator.available_size(),
            )

    def _pd_drain_ready_multi(self) -> int:
        """Drain up to `ready_q.maxsize + 1` items per call instead of one.

        At concurrency (many reqs each producing a ready_q item near-
        simultaneously at their own chunk boundaries), a 1-pop-per-tick drain
        falls behind the arrival rate and ready_q backlogs for seconds
        (measured rq_wait 2.4-4.8s at C32, feeding directly into decode ITL
        spikes since draining also merges finished reqs into running_batch).
        Draining up to the full ready_q capacity per tick keeps the backlog
        from compounding; each item is cheap (~2ms observed, including
        done-item scatter, at this workload) so this doesn't itself stall
        decode.

        Stops early if a call makes no progress (ready_q size and defer-list
        length both unchanged): a single D-pool-full item at the head of
        _pd_defer gets re-popped and re-appended every call, so without this
        guard the loop would burn its full budget retrying the same stuck
        item instead of exiting once nothing more can be drained this tick.

        Returns the number of `_pd_drain_ready()` calls made (test hook)."""
        n_calls = 0
        for _ in range(self._pd_ready_q.maxsize + 1):
            if not self._pd_defer and self._pd_ready_q.empty():
                break
            n_defer_before, n_ready_before = len(self._pd_defer), self._pd_ready_q.qsize()
            self._pd_drain_ready()
            n_calls += 1
            if len(self._pd_defer) == n_defer_before and self._pd_ready_q.qsize() == n_ready_before:
                break
        return n_calls

    def _pd_maybe_push_prefill(self) -> None:
        """Build at most one P batch (round-robin across P slices) and push to
        its prefill_q. Extracted so both the n_decode==1 path
        (_pd_get_next_batch_async via base event_loop_overlap) and the
        n_decode>1 path (event_loop_overlap_pd_nd) share the same P admission.

        Chunked prefill: while a chunked req's chunk-N is in-flight
        (_pd_chunk_pending), skip building chunk-N+1. The prefill thread
        clears pending itself right after chunk-N's forward (prefix_indices
        updated there too), so this only blocks for actual forward latency,
        not a full drain-cycle round-trip."""
        _tnow = time.perf_counter()
        if (
            _PD_DBG_TIMING
            and self._pd_t_last_get_batch > 0
            and _tnow - self._pd_t_last_get_batch > 0.3
        ):
            logger.info(
                "[pd-tick-gap] %.0fms since last get_batch (D running=%d)",
                (_tnow - self._pd_t_last_get_batch) * 1e3,
                sum(len(x.reqs) for x in self.running_batch.reqs_info if x.reqs),
            )
        self._pd_t_last_get_batch = _tnow
        n_p = self._pd_n_prefill
        # One P build per tick (round-robin). Building for BOTH P in the same
        # tick has get_new_batch_prefill(P0) walk waiting_queue and mutate
        # reqs (init_next_round_input/add_one_req truncate) that P1 will pick
        # up in the same tick -> observed 2P1D-only correctness bug on the
        # first attempt. D tick ~29ms << P cycle ~330ms so 1 build/tick is
        # plenty to keep both P threads fed.
        for _off in range(n_p):
            i = (self._pd_next_p + _off) % n_p
            if self._pd_chunk_pending[i]:
                continue
            if self._pd_prefill_qs[i].full():
                continue
            self._pd_next_p = (i + 1) % n_p
            saved = self.per_dp_max_running_requests
            d_running = self._pd_total_d_running()
            # swap_p_pool replaces running_batch with an empty one so PrefillAdder
            # stops mixing D-side future-token reservation into the P pool budget;
            # the D r2t slot constraint moves here explicitly. `saved` is per-dp
            # but inflight/d_running are TOTAL across all dp ranks (single-process
            # dp>1) — divide by dp_size so the subtraction is unit-consistent.
            # (Previously subtracted totals directly: dp4 saved=64 hit 0 once
            # d_running reached 64, freezing P admission for ~2min bursts and
            # capping C128 D running at 64.)
            # n_decode>1: total D r2t slots = n_d * saved * dp_size, and
            # d_running/_pd_inflight_rids are already summed across all D.
            self.per_dp_max_running_requests = max(
                0,
                saved * self._pd_n_decode
                - (len(self._pd_inflight_rids) + d_running + self.dp_size - 1) // self.dp_size,
            )
            try:
                with self._pd_swap_p_pool(i):
                    new_batch = self.get_new_batch_prefill()
            finally:
                self.per_dp_max_running_requests = saved
            if new_batch is None:
                if any(r is not None for r in self.p_chunked_reqs[i]):
                    self._pd_chunk_stall = getattr(self, "_pd_chunk_stall", 0) + 1
                    if self._pd_chunk_stall % 500 == 1:
                        pa = self.p_allocs[i]
                        logger.warning(
                            "[pd-diag] chunked stall x%d: P full=%d swa=%d",
                            self._pd_chunk_stall,
                            getattr(pa, "full_available_size", pa.available_size)(),
                            pa.swa_available_size() if hasattr(pa, "swa_available_size") else -1,
                        )
                break
            self._pd_chunk_stall = 0
            assert not any(
                r.return_logprob or r.return_hidden_states
                for info in new_batch.reqs_info
                if info.reqs
                for r in info.reqs
            ), "[pd_async] return_logprob / hidden_states not yet supported (async prefill drops logits_output)"
            for info in new_batch.reqs_info:
                for r in info.reqs or ():
                    self._pd_inflight_rids.add(r.rid)
            # Set pending only when this batch has a mid-chunk req (whose next
            # chunk depends on this one draining). Final-chunk / non-chunked
            # batches leave pending untouched so the main loop can immediately
            # build the next req's chunk-1 while this one is still forwarding
            # (depth=2 overlap). Drain only clears pending on a mid-chunk item.
            if any(info.chunked_req is not None for info in new_batch.reqs_info):
                self._pd_chunk_pending[i] = True
            new_batch._pd_t_enqueue_prefill = time.perf_counter()
            self._pd_prefill_qs[i].put_nowait(new_batch)
            break

    def _pd_get_next_batch_async(self):
        """n_decode==1 entry point (called from base event_loop_overlap via
        get_next_batch_to_run): push P, drain, return the D0 decode batch."""
        self._pd_maybe_push_prefill()
        # drain AFTER pushing next P batch so P thread's pq_wait doesn't
        # include the ~130ms migrate of done reqs (drain overlaps P forward).
        # chunked req's prefix_indices are already updated by P thread (:269),
        # and new-req chunk-1 only needs waiting_queue (not drain), so build-
        # before-drain is safe. done reqs merge into running_batch one tick
        # later, harmless.
        self._pd_drain_ready_multi()
        if not self.running_batch.is_empty() and not self.running_batch.is_prefill_only:
            self.running_batch = self.update_running_batch(self.running_batch)
            return self.running_batch if not self.running_batch.is_empty() else None
        return None

    def event_loop_overlap_pd_nd(self):
        """n_decode>1 event loop: one main-thread tick dispatches ALL D slices.

        Each D has its own overlap ModelWorkerClient (own forward thread), so
        dispatch is just N x input_queue.put (~7ms python each). Device-side
        forwards run physically parallel on disjoint slices; the Pathways
        dispatch queue serializes DISPATCH but each Execute is ~7ms so
        N=2 dispatch (~14ms) < D forward (~24ms). resolve_last_batch_result
        per D waits on that D's own output_queue -- by the time we resolve,
        the previous tick's forward is already done, so it's D2H (~2ms) +
        process_batch_result_decode (~3ms) per D. N=2 total main-thread
        ~24ms ~= device 24.7ms (feasibility gate). N>=3 likely main-thread
        bound; would need per-D sub-loop threads (not done here)."""
        import gc as _gc
        from collections import deque

        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        assert self.enable_overlap, "pd_num_decode>1 requires --enable-overlap-schedule"
        _gc.collect()
        _gc.freeze()
        _gc.set_threshold(700, 10, 10000)
        n_d = self._pd_n_decode
        self.result_queue = deque()
        d_result_queues = [deque() for _ in range(n_d)]
        d_last_batch: list = [None] * n_d
        d_cur_batch: list = [None] * n_d
        logger.info("[pathways_pd] entering event_loop_overlap_pd_nd n_decode=%d", n_d)

        while True:
            _it0 = time.perf_counter()
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            recv_reqs = self.select_dp_for_request(recv_reqs)
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            self._pd_maybe_push_prefill()
            self._pd_drain_ready_multi()
            _it1 = time.perf_counter()

            any_batch = False
            for j in range(n_d):
                self._pd_set_d(j)
                rb = self.running_batch
                if not rb.is_empty() and not rb.is_prefill_only:
                    self.running_batch = self.update_running_batch(rb)
                    batch = self.running_batch if not self.running_batch.is_empty() else None
                else:
                    batch = None
                d_cur_batch[j] = batch
                self.cur_batch = batch
                if batch:
                    any_batch = True
                    batch.launch_done = threading.Event()
                    result = self.run_batch(batch)
                    d_result_queues[j].append((batch.copy(), result))
                    if d_last_batch[j] is None:
                        tmp = ScheduleBatch.init_new(
                            reqs=[[] for _ in range(self.dp_size)],
                            req_to_token_pool=self.req_to_token_pool,
                            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                            tree_cache=self.tree_cache,
                            model_config=self.model_config,
                            enable_overlap=self.enable_overlap,
                            dp_size=self.dp_size,
                            spec_algorithm=self.spec_algorithm,
                            mesh=self.mesh,
                        )
                        tmp.forward_mode = ForwardMode.DUMMY_FIRST
                        tmp.next_batch_sampling_info = self.tp_worker.cur_sampling_info
                        self.process_batch_result(tmp, None, batch.launch_done)
            _it2 = time.perf_counter()

            for j in range(n_d):
                self._pd_set_d(j)
                if d_last_batch[j]:
                    tmp_b, tmp_r = d_result_queues[j].popleft()
                    tmp_b.next_batch_sampling_info = (
                        self.tp_worker.cur_sampling_info if d_cur_batch[j] else None
                    )
                    self.process_batch_result(
                        tmp_b, tmp_r, d_cur_batch[j].launch_done if d_cur_batch[j] else None
                    )
                d_last_batch[j] = d_cur_batch[j]
                self.last_batch = d_cur_batch[j]

            if not any_batch and all(b is None for b in d_last_batch):
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            _it3 = time.perf_counter()
            self.d_running_batches[self._pd_cur_d] = self.running_batch
            if _PD_DBG_TIMING and (self.forward_ct % 40 == 0 or _it3 - _it0 > 0.3):
                _runs = [
                    sum(len(x.reqs) for x in rb.reqs_info if x.reqs)
                    for rb in self.d_running_batches
                ]
                logger.info(
                    "[pd-iter-nd] total=%.0fms push+drain=%.0f dispatch=%.0f resolve=%.0f "
                    "running=%s inflight=%d",
                    (_it3 - _it0) * 1e3,
                    (_it1 - _it0) * 1e3,
                    (_it2 - _it1) * 1e3,
                    (_it3 - _it2) * 1e3,
                    _runs,
                    len(self._pd_inflight_rids),
                )

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
            self.chunked_reqs,
        )
        self.chunked_reqs = self.p_chunked_reqs[p_idx]
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
                self.chunked_reqs,
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
