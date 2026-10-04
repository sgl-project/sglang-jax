from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sgl_jax.srt.model_executor.batch_inputs import BatchInputBuffer
from sgl_jax.srt.model_executor.batch_layout import (
    BatchLayoutPlan,
    SequenceLayout,
    serving_shape_buckets,
)
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.multimodal.in_model.host_orchestration import build_multimodal_batch
from sgl_jax.srt.multimodal.in_model.lane_packing import encoder_num_lanes
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.sampling.sampling_params import DEFAULT_SAMPLING_SEED
from sgl_jax.srt.speculative.overlap_utils import use_legacy_eagle3_non_overlap
from sgl_jax.srt.utils.common_utils import pad_to_bucket

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch


class BatchInputBuilder:
    """Translate one scheduler snapshot into rank-local input buffers."""

    def __init__(self, batch, plan=None):
        self.batch = batch
        self.plan = plan

    def _sequences(self, page_size):
        batch = self.batch
        seqs = [info.seq_lens if info.seq_lens is not None else [] for info in batch.reqs_info]
        prefixes = (
            [info.prefix_lens if info.prefix_lens is not None else [] for info in batch.reqs_info]
            if batch.forward_mode.is_extend()
            else None
        )
        return SequenceLayout.create(seqs, prefixes, None, page_size)

    def _build_inputs(self, plan):
        batch = self.batch
        extend = batch.forward_mode.is_extend()
        b, t = plan.request_capacity, plan.token_capacity
        fields = {
            "input_ids": ((t,), np.int32, 0),
            "seq_lens": ((b,), np.int32, 0),
            "out_cache_loc": ((t,), np.int32, -1),
            "positions": ((t,), np.int32, 0),
            "req_pool_indices": ((b,), np.int32, -1),
        }
        if extend:
            fields.update(
                {name: ((b,), np.int32, 0) for name in ("extend_prefix_lens", "extend_seq_lens")}
            )
        recurrent = [
            name
            for name in ("recurrent_indices", "recurrent_cow_src_indices")
            if any(getattr(info, name) is not None for info in batch.reqs_info)
        ]
        if any(info.recurrent_track_mask is not None for info in batch.reqs_info):
            recurrent += ["recurrent_track_indices", "recurrent_track_mask"]
        fields.update({name: ((b,), np.int32, 0) for name in recurrent})
        buffer = BatchInputBuffer(plan.dp_size, fields)
        sequences = plan.sequences
        for rank, info in enumerate(batch.reqs_info):
            n, nt = plan.request_counts[rank], plan.token_counts[rank]
            real = sequences.requests(rank)
            if n:
                buffer.view("seq_lens", rank)[:n] = sequences.kv_lengths[real]
                buffer.view("req_pool_indices", rank)[:n] = info.req_pool_indices
                if extend:
                    buffer.view("extend_prefix_lens", rank)[:n] = sequences.prefix_lengths[real]
                    buffer.view("extend_seq_lens", rank)[:n] = sequences.query_lengths[real]
                for name in recurrent:
                    value = getattr(info, name)
                    if value is not None:
                        buffer.view(name, rank)[:n] = value
            if nt:
                buffer.view("input_ids", rank)[:nt] = info.input_ids
                positions = buffer.view("positions", rank)
                if extend:
                    # Request-local positions in one vectorized pass per rank.
                    # The final output is the staging view, not a merged array.
                    np.add(
                        np.repeat(
                            sequences.prefix_lengths[real] - sequences.query_starts[real],
                            sequences.query_lengths[real],
                        ),
                        np.arange(nt, dtype=np.int32),
                        out=positions[:nt],
                    )
                else:
                    positions[:n] = sequences.prefix_lengths[real]
                if info.out_cache_loc is not None:
                    size = min(len(info.out_cache_loc), nt)
                    buffer.view("out_cache_loc", rank)[:size] = info.out_cache_loc[:size]
        # Consume one-step recurrent actions only after the snapshot is filled.
        for info in batch.reqs_info:
            if "recurrent_cow_src_indices" in recurrent:
                info.recurrent_cow_src_indices = None
                for req in info.reqs or []:
                    req.recurrent_cow_src_index = None
            if "recurrent_track_mask" in recurrent:
                info.recurrent_track_indices = info.recurrent_track_mask = None
        return buffer.finish()

    def _extend_metadata(self, plan):
        starts = logits = None
        initial = np.ones(plan.request_capacity, dtype=np.bool_)
        if self.batch.forward_mode.is_extend():
            starts = np.zeros(plan.request_capacity, dtype=np.int32)
            logits = np.zeros(plan.request_capacity, dtype=np.int32)
            seq = plan.sequences
            for rank, info in enumerate(self.batch.reqs_info):
                dst, src = plan.request_slice(rank), seq.requests(rank)
                logits[dst] = seq.query_starts[src] + seq.query_lengths[src] - 1
                initial[dst] = seq.prefix_lengths[src] > 0
                if info.extend_logprob_start_lens is not None:
                    starts[dst] = info.extend_logprob_start_lens
        return starts, logits, initial

    def _merge_multimodal(self) -> dict:
        """Assemble all per-token multimodal tensors in one DP-interleaved pass.

        Single traversal of ``reqs_info[*].reqs`` that produces, on the same
        rank-offset layout from ``BatchLayoutPlan`` (per-rank token capacity; within a rank the per-req EXTEND window
        ``[prefix_len, seq_len)``), all three multimodal tensors at once:

        - ``input_embedding`` ``[total_token_size, hidden]`` -- per-req merged
          embedding sliced to its extend window.
        - ``mrope_positions`` ``[3, total_token_size]`` -- 3-D mRoPE positions;
          extend slices ``mm_positions[:, prefix:prefix+ext]`` (delta / arange
          fallback), decode advances ``seq_len-1 (+delta)``.
        - ``deepstack_visual_embedding`` ``[num_layers, total_token_size,
          hidden]`` -- sparse visual rows densified into the batched layout with
          non-visual rows zero, plus the derived ``apply_for_deepstack``.

        Data stays on ``Req`` (no new ScheduleReqsInfo fields); this only reads it. Collapses the
        three previously separate rank-offset loops so the layout logic lives in
        exactly one place. Each field is ``None`` / ``False`` when no request
        carries it, keeping pure-text / non-multimodal paths unchanged (0-diff).
        """
        from sgl_jax.srt.managers.schedule_batch import (
            _as_int_scalar,
            _extract_mm_value,
        )

        batch = self.batch
        plan = self.plan
        total_token_size = plan.token_capacity
        is_extend = batch.forward_mode.is_extend()
        is_decode = batch.forward_mode.is_decode()

        has_mrope = any(
            _extract_mm_value(req.mm_inputs, "mrope_positions") is not None
            or _extract_mm_value(req.mm_inputs, "mrope_position_delta") is not None
            for info in batch.reqs_info
            if info.reqs
            for req in info.reqs
        )

        # input_embedding / deepstack are extend-only; mrope also refreshes on
        # decode. Nothing to assemble otherwise -> all None/False (0-diff).
        emb = None
        mrope = np.zeros((3, total_token_size), dtype=np.int32) if has_mrope else None
        dense = None
        if not is_extend and mrope is None:
            return {
                "input_embedding": None,
                "mrope_positions": None,
                "apply_for_deepstack": False,
                "deepstack_visual_embedding": None,
            }

        for dp_rank in range(batch.dp_size):
            info = batch.reqs_info[dp_rank]
            offset = plan.token_slice(dp_rank).start
            if not info.reqs or info.seq_lens is None or len(info.seq_lens) == 0:
                continue
            local = 0

            if is_decode:
                # Decode: one token per request; only mrope advances (embedding
                # and deepstack are extend-only and stay None/False).
                if mrope is not None:
                    for req, seq_len in zip(info.reqs, info.seq_lens):
                        base_pos = int(seq_len) - 1
                        delta = _extract_mm_value(req.mm_inputs, "mrope_position_delta")
                        if delta is not None:
                            base_pos += _as_int_scalar(delta)
                        mrope[:, offset + local] = base_pos
                        local += 1
                continue

            # Extend: write each req's [prefix_len, seq_len) window.
            real = plan.sequences.requests(dp_rank)
            for req, prefix_len, ext_len, local in zip(
                info.reqs,
                plan.sequences.prefix_lengths[real],
                plan.sequences.query_lengths[real],
                plan.sequences.query_starts[real],
            ):
                ext_len = int(ext_len)
                if ext_len <= 0:
                    continue
                start = int(prefix_len or 0)
                end = start + ext_len

                # input_embedding: per-req merged embedding, extend window.
                mm_emb = getattr(req, "multimodal_embedding", None)
                if mm_emb is not None:
                    mm_full = np.asarray(mm_emb)
                    chunk = mm_full[start:end]
                    if emb is None:
                        emb = np.zeros((total_token_size, mm_full.shape[1]), dtype=mm_full.dtype)
                    emb[offset + local : offset + local + chunk.shape[0]] = chunk

                # mrope_positions: 3-D positions, slice with fallback.
                if mrope is not None:
                    mm_positions = _extract_mm_value(req.mm_inputs, "mrope_positions")
                    if mm_positions is None:
                        # Text-only req in a mixed mrope batch: 1-D positions
                        # broadcast to 3 rows (T==H==W), matching the model's
                        # non-mrope fallback for these tokens.
                        base = np.arange(start, start + ext_len, dtype=np.int32)
                        mchunk = np.broadcast_to(base.reshape(1, -1), (3, ext_len))
                    else:
                        mm_positions = np.asarray(mm_positions)
                        positions_len = mm_positions.shape[1]
                        known_end = min(end, positions_len)
                        known_len = max(known_end - start, 0)
                        mchunk = np.empty((3, ext_len), dtype=np.int32)
                        if known_len:
                            mchunk[:, :known_len] = mm_positions[:, start:known_end]

                        # mRoPE positions only cover the original multimodal
                        # prompt.  A retracted decode request is re-prefilled
                        # with ``origin_input_ids + output_ids``, so its extend
                        # window can straddle the end of that array.  Continue
                        # generated-token positions exactly like decode mode
                        # instead of assigning a short slice into ``ext_len``.
                        if known_len < ext_len:
                            delta = _extract_mm_value(req.mm_inputs, "mrope_position_delta")
                            tail_start = start + known_len
                            base = np.arange(tail_start, end, dtype=np.int32)
                            if delta is not None:
                                base = base + _as_int_scalar(delta)
                            mchunk[:, known_len:] = base
                    mrope[:, offset + local : offset + local + ext_len] = mchunk

                # deepstack: densify sparse visual rows into batched layout,
                # non-visual rows stay zero (so the model can add to all tokens).
                ds_emb = getattr(req, "deepstack_visual_embedding", None)
                ds_mask = getattr(req, "deepstack_visual_pos_mask", None)
                if (
                    getattr(req, "apply_for_deepstack", False)
                    and ds_emb is not None
                    and ds_mask is not None
                ):
                    full_mask = np.asarray(ds_mask).astype(bool)
                    emb_arr = np.asarray(ds_emb)  # (num_layers, num_visual, hidden)
                    # Only valid when the per-req mask spans the full prompt
                    # (skips the audio-only dummy [1]-length fallback).
                    if full_mask.shape[0] >= end and emb_arr.ndim == 3:
                        window_mask = full_mask[start:end]
                        nvis = int(window_mask.sum())
                        if nvis > 0:
                            vstart = int(full_mask[:start].sum())
                            window_emb = emb_arr[:, vstart : vstart + nvis, :]
                            if dense is None:
                                dense = np.zeros(
                                    (emb_arr.shape[0], total_token_size, emb_arr.shape[2]),
                                    dtype=emb_arr.dtype,
                                )
                            vis_pos = offset + local + np.nonzero(window_mask)[0]
                            dense[:, vis_pos, :] = window_emb

        return {
            "input_embedding": emb,
            "mrope_positions": mrope,
            "apply_for_deepstack": dense is not None,
            "deepstack_visual_embedding": dense,
        }

    def _merge_cache_loc(
        self,
        bs_paddings: list,
        cache_loc_paddings: list,
        page_size: int,
    ) -> np.ndarray:
        """Merge cache_loc from all DP ranks with page alignment.

        Returns:
            cache_loc array
        """
        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        batch = self.batch
        # Calculate total cache_loc size needed
        total_cache_loc_size = 0
        if batch.forward_mode.is_extend():
            total_cache_loc_size = cache_loc_paddings[-1]  # Use largest padding
        else:
            # For decode mode, use the cache_loc_padding that corresponds to the bs bucket.
            total_bs = self.plan.request_capacity
            _, bs_index = pad_to_bucket(total_bs, bs_paddings)
            total_cache_loc_size = cache_loc_paddings[bs_index]

        per_dp_cache_loc_size = total_cache_loc_size // batch.dp_size
        # View into the persistent buffer; intentionally NOT re-zeroed per step.
        # Safe because:
        #  - padding slots are never read on-device: attention kernels (RPA v3 /
        #    MLA v2 / native) bound page reads by cu_kv_lens / seq_lens, and every
        #    real-request page slot lands on a written position.
        #  - every buffer value is a valid in-bounds KV slot index (init is
        #    np.zeros + only valid slots are ever written), so even SWA's
        #    host-side mapping[cache_loc] lookup (flashattention_backend) can't go
        #    OOB. This REQUIRES the init buffer to be np.zeros, not np.empty.
        cache_loc_host_buf = batch.req_to_token_pool.cache_loc_host_buf
        assert (
            cache_loc_host_buf is not None and cache_loc_host_buf.shape[0] >= total_cache_loc_size
        ), (
            "cache_loc_host_buf is not initialized or too small: "
            f"capacity={0 if cache_loc_host_buf is None else cache_loc_host_buf.shape[0]}, "
            f"required={total_cache_loc_size}"
        )
        cache_loc_cpu = cache_loc_host_buf[:total_cache_loc_size]

        req_to_token = batch.req_to_token_pool.req_to_token
        max_context_len = req_to_token.shape[1]
        req_to_token_flat = req_to_token.reshape(-1)
        page_ramp = np.arange(page_size, dtype=req_to_token.dtype) if page_size > 1 else None

        for dp_rank in range(batch.dp_size):
            info = batch.reqs_info[dp_rank]
            offset_bs = dp_rank * per_dp_cache_loc_size

            if info.seq_lens is None or len(info.seq_lens) == 0:
                continue

            seq_lens = np.asarray(info.seq_lens)
            req_pool_indices = np.asarray(info.req_pool_indices)

            n_reqs = len(seq_lens)
            sequences = self.plan.sequences
            real = sequences.requests(dp_rank)
            n_pages = sequences.page_counts[real]
            page_cum = sequences.page_starts[real]
            offsets = page_cum * page_size

            if page_size > 1:
                # PagedTokenToKVPoolAllocator writes page-contiguous slot indices
                # (req_to_token[i, p*ps+j] == req_to_token[i, p*ps] + j), so the
                # per-req loop can be replaced by one gather of page-start values
                # plus a broadcast-add. Padding tail [seq_len:aligned] lands in
                # the same allocated page so remains safe.
                total_pages = int(n_pages.sum())
                if total_pages > 0:
                    total_aligned = total_pages * page_size
                    # flat_src[g] = idx[r]*W + p*ps = (idx[r]*W - page_cum[r]*ps) + g*ps
                    row_base = req_pool_indices.astype(np.int64) * max_context_len
                    flat_src = np.repeat(row_base - page_cum * page_size, n_pages)
                    flat_src += np.arange(total_pages, dtype=np.int64) * page_size
                    page_starts = req_to_token_flat[flat_src]
                    dest = cache_loc_cpu[offset_bs : offset_bs + total_aligned]
                    np.add(
                        page_starts.reshape(total_pages, 1),
                        page_ramp.reshape(1, page_size),
                        out=dest.reshape(total_pages, page_size),
                    )
            else:
                # Non-paged allocator has no page-contiguity guarantee.
                for r in range(n_reqs):
                    sl = int(seq_lens[r])
                    dest_start = int(offsets[r]) + offset_bs
                    cache_loc_cpu[dest_start : dest_start + sl] = req_to_token[
                        int(req_pool_indices[r]), :sl
                    ]

        # cache_loc_cpu is a view into the reusable host_buf; PD eager-stash
        # can overwrite it via _disp(nxt) before this batch's H2D consumes the
        # view. Single-threaded (native/colocated) callers don't need the copy.
        if global_server_args_dict.get("pd_disaggregation") == "pathways":
            return cache_loc_cpu.copy()
        return cache_loc_cpu

    def _merge_sampling_info(self):
        """Merge sampling info from all DP ranks.

        Returns:
            Merged SamplingBatchInfo
        """
        from sgl_jax.srt.managers.schedule_batch import ModelWorkerSamplingInfo

        batch = self.batch
        plan = self.plan
        total_bs = plan.request_capacity
        # Initialize merged arrays (with padding)
        grammars = [None] * total_bs if batch.has_grammar else None
        temperatures = np.ones((total_bs, 1), dtype=np.float32)
        top_ps = np.ones(total_bs, dtype=np.float32)
        top_ks = np.ones(total_bs, dtype=np.int32)
        min_ps = np.zeros(total_bs, dtype=np.float32)
        sampling_seeds = None
        linear_penalty = None  # lazily allocated only if any DP rank has penalties

        has_sampling_seeds = False
        vocab_size = 0
        is_all_greedy = True

        for dp_rank in range(batch.dp_size):
            info = batch.reqs_info[dp_rank]
            slots = plan.request_slice(dp_rank)
            offset_bs = slots.start

            if grammars is not None:
                for i, req in enumerate(info.reqs or []):
                    grammars[offset_bs + i] = req.grammar

            if info.sampling_info is None or info.seq_lens is None or len(info.seq_lens) == 0:
                continue

            dp_bs = len(info.seq_lens)
            dp_sampling = info.sampling_info
            if vocab_size == 0:
                vocab_size = dp_sampling.vocab_size

            if not info.sampling_info.is_all_greedy:
                is_all_greedy = False

            # Copy sampling parameters
            temperatures[slots] = dp_sampling.temperatures[:dp_bs]
            top_ps[slots] = dp_sampling.top_ps[:dp_bs]
            top_ks[slots] = dp_sampling.top_ks[:dp_bs]
            min_ps[slots] = dp_sampling.min_ps[:dp_bs]

            if dp_sampling.sampling_seeds is not None:
                if sampling_seeds is None:
                    sampling_seeds = np.full(total_bs, DEFAULT_SAMPLING_SEED, dtype=np.int64)
                    has_sampling_seeds = True
                sampling_seeds[slots] = dp_sampling.sampling_seeds[:dp_bs]

            # Write directly into a fresh merged buffer. Never reuse storage
            # across steps: the previous batch may still be transferring to TPU.
            orchestrator = dp_sampling.penalizer_orchestrator
            if orchestrator is not None:
                penalty_out = None
                if orchestrator.is_required:
                    if linear_penalty is None:
                        linear_penalty = np.zeros(
                            (total_bs, dp_sampling.vocab_size), dtype=np.float32
                        )
                    penalty_out = linear_penalty[slots]
                dp_sampling.update_penalties(out=penalty_out)
            elif dp_sampling.linear_penalty is not None and dp_sampling.linear_penalty.size:
                # Worker-side sampling info may already contain computed penalties.
                if linear_penalty is None:
                    linear_penalty = np.zeros(
                        (total_bs, dp_sampling.linear_penalty.shape[1]),
                        dtype=dp_sampling.linear_penalty.dtype,
                    )
                linear_penalty[slots] = dp_sampling.linear_penalty[:dp_bs]

        return ModelWorkerSamplingInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            vocab_size=vocab_size,
            is_all_greedy=is_all_greedy,
            sampling_seeds=sampling_seeds if has_sampling_seeds else None,
            linear_penalty=linear_penalty,
            grammars=grammars,
        )

    def _merge_lora_ids(self, enable_static_lora):
        lora_ids = ["0"] * self.plan.request_capacity
        if not enable_static_lora:
            for rank, info in enumerate(self.batch.reqs_info):
                lora_ids[self.plan.request_slice(rank)] = [req.lora_id for req in info.reqs or []]
        return lora_ids

    def build_spec_decode(
        self, bs_paddings: list, enable_static_lora: bool, draft_token_num: int = 1
    ) -> ModelWorkerBatch:
        """DP-aware spec-decode ModelWorkerBatch (#1053 P1-5b).

        Uses the same DP layout and staging buffer as ordinary serving. ``spec_info`` is global (DP-padded
        order, see ``EagleDraftInput`` docstring) and lives only on
        ``reqs_info[0]``. ``input_ids``/``positions``/``cache_loc`` are
        placeholders — ``EagleDraftWorker.padding_for_decode`` rebuilds them.
        """
        from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, acc_global_bid

        batch = self.batch
        request_counts = tuple(
            len(i.seq_lens) if i.seq_lens is not None else 0 for i in batch.reqs_info
        )
        total_bs, _ = pad_to_bucket(
            max(max(request_counts), 1) * batch.dp_size, bs_paddings or [batch.dp_size]
        )
        plan = BatchLayoutPlan(request_counts, (0,) * batch.dp_size, total_bs, 0)
        self.plan = plan
        per_dp_bs = plan.requests_per_rank
        batch.per_dp_bs_size = per_dp_bs
        real_bs, real_bs_per_dp = plan.real_requests, list(plan.request_counts)
        logits_indices_selector = plan.request_selector()
        sampling_info = self._merge_sampling_info()
        for dp_rank, info in enumerate(batch.reqs_info):
            spec_info_dp = info.spec_info
            future_indices = getattr(spec_info_dp, "future_indices", None)
            if future_indices is None:
                continue
            num_reqs = len(info.reqs) if info.reqs is not None else 0
            assert len(future_indices) == num_reqs, (
                f"future_indices length mismatch on dp_rank={dp_rank}: "
                f"{len(future_indices)=}, {num_reqs=}"
            )
        # Concat per-rank spec_info into a cross-rank-flat EagleDraftInput,
        # then scatter into DP-padded (total_bs, ...) slots so spec_info[i]
        # aligns with seq_lens[i]. Returns a new object — does not mutate
        # the per-rank cross-round state on reqs_info[r].spec_info.
        flat_spec = batch._concat_spec_info_per_rank([info.spec_info for info in batch.reqs_info])
        legacy_eagle3_non_overlap = use_legacy_eagle3_non_overlap(
            batch.enable_overlap, batch.spec_algorithm
        )
        spec_info = batch._scatter_spec_info_to_dp_slots(
            flat_spec,
            logits_indices_selector,
            total_bs,
            mesh=batch.mesh,
            legacy_host_scatter=legacy_eagle3_non_overlap,
        )
        # Per-rank out_cache_loc chunks (set in spec prepare_for_decode) have
        # variable length (∝ accept_len). DP-segment: pad each to max_len with
        # -1 so the P("data") shard in ForwardBatch.init_new gives rank r its
        # own slots (fa_backend doesn't use it, but native_backend would).
        ocl_chunks = [
            (
                np.asarray(i.out_cache_loc, dtype=np.int32)
                if i.out_cache_loc is not None and len(i.out_cache_loc) > 0
                else np.empty(0, dtype=np.int32)
            )
            for i in batch.reqs_info
        ]
        # Spec prepare_for_decode conservatively reserves up to
        # 2 * ALLOC_LEN_PER_DECODE tokens per request. Keep the JIT-visible
        # out_cache_loc shape on that same conservative bucket instead of the
        # smaller draft-token count, otherwise high-running batches can escape
        # precompile with shapes like 656/928/1024.
        max_chunk_len = max((len(c) for c in ocl_chunks), default=0)
        if use_legacy_eagle3_non_overlap(batch.enable_overlap, batch.spec_algorithm):
            target_per_rank_ocl = max(per_dp_bs * draft_token_num, max_chunk_len)
        else:
            target_per_rank_ocl = per_dp_bs * draft_token_num * 2
            assert max_chunk_len <= target_per_rank_ocl, (
                "spec decode out_cache_loc escaped precompile bucket: "
                f"max_chunk_len={max_chunk_len}, bucket_per_rank={target_per_rank_ocl}, "
                f"per_dp_bs={per_dp_bs}, draft_token_num={draft_token_num}"
            )
        buffer = BatchInputBuffer(
            plan.dp_size,
            {
                "input_ids": ((0,), np.int32, 0),
                "seq_lens": ((total_bs,), np.int32, 0),
                "out_cache_loc": ((target_per_rank_ocl * plan.dp_size,), np.int32, -1),
                "positions": ((0,), np.int32, 0),
                "req_pool_indices": ((total_bs,), np.int32, -1),
            },
        )
        for rank, info in enumerate(batch.reqs_info):
            n = plan.request_counts[rank]
            if n:
                buffer.view("seq_lens", rank)[:n] = info.seq_lens
                buffer.view("req_pool_indices", rank)[:n] = info.req_pool_indices
            buffer.view("out_cache_loc", rank)[: len(ocl_chunks[rank])] = ocl_chunks[rank]
        inputs = buffer.finish()
        model_worker_batch = ModelWorkerBatch(
            inputs=inputs,
            layout=plan,
            bid=acc_global_bid(),
            forward_mode=batch.forward_mode,
            real_input_ids_len=0,
            return_logprob=batch.return_logprob,
            return_output_logprob_only=batch.return_output_logprob_only,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            sampling_info=sampling_info,
            cache_loc=np.empty(0, dtype=np.int32),
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            logits_indices=None,
            lora_ids=self._merge_lora_ids(enable_static_lora),
            real_bs=real_bs,
            real_bs_per_dp=real_bs_per_dp,
            dp_size=batch.dp_size,
            per_dp_bs_size=per_dp_bs,
            logits_indices_selector=logits_indices_selector,
            capture_hidden_mode=getattr(spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL),
            launch_done=batch.launch_done,
            spec_info_padded=spec_info,
            spec_algorithm=batch.spec_algorithm,
            tree_cache=batch.tree_cache,
            mrope_positions=None,
        )
        return model_worker_batch

    def build(
        self,
        token_paddings: list,
        bs_paddings: list,
        cache_loc_paddings: list,
        page_size: int,
        enable_static_lora: bool = False,
    ) -> ModelWorkerBatch:
        from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, acc_global_bid

        batch = self.batch
        token_paddings, bs_paddings, cache_loc_paddings = serving_shape_buckets(
            batch.forward_mode, token_paddings, bs_paddings, cache_loc_paddings
        )

        bid = acc_global_bid()

        request_counts = tuple(
            len(i.seq_lens) if i.seq_lens is not None else 0 for i in batch.reqs_info
        )
        token_counts = tuple(
            len(i.input_ids) if i.input_ids is not None else 0 for i in batch.reqs_info
        )
        plan = BatchLayoutPlan.from_buckets(
            request_counts,
            token_counts,
            bs_paddings,
            token_paddings,
            sequences=self._sequences(page_size),
        )
        self.plan = plan
        per_dp_bs_padding, total_bs = plan.requests_per_rank, plan.request_capacity
        per_dp_token_padding, total_token_size = plan.tokens_per_rank, plan.token_capacity
        batch.per_dp_bs_size = per_dp_bs_padding
        inputs = self._build_inputs(plan)
        real_input_ids_len, real_bs = plan.real_tokens, plan.real_requests
        real_bs_per_dp = list(plan.request_counts)
        logits_indices_selector = plan.request_selector()
        extend_logprob_start_lens, logits_indices, has_initial_state_cpu = self._extend_metadata(
            plan
        )
        cache_loc_cpu = self._merge_cache_loc(bs_paddings, cache_loc_paddings, page_size)
        sampling_info = self._merge_sampling_info()

        # Generate trace info if needed
        if precision_tracer.get_trace_active():
            batch._generate_trace_info(real_bs, bid)

        # Align adapters with the DP-padded request metadata.
        lora_ids = self._merge_lora_ids(enable_static_lora)

        # Assemble all per-token multimodal tensors (input_embedding,
        # mrope_positions, deepstack) in a single DP-interleaved pass over
        # reqs_info[*].reqs; see BatchInputBuilder._merge_multimodal. Each is
        # None/False for pure-text batches, so non-multimodal paths stay
        # unchanged.
        _mm = self._merge_multimodal()
        input_embedding = _mm["input_embedding"]
        mrope_positions = _mm["mrope_positions"]
        apply_for_deepstack = _mm["apply_for_deepstack"]
        deepstack_visual_embedding = _mm["deepstack_visual_embedding"]
        # Keep items whose placeholder rows intersect the current prefill window.
        if batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            multimodal_batch = build_multimodal_batch(
                batch.reqs_info,
                batch.dp_size,
                batch.model_config,
                per_dp_token_padding,
                embedding_pool=batch.embedding_pool,
                num_encoder_lanes=encoder_num_lanes(
                    batch.mesh,
                    tensor_parallel=batch.model_config.hf_config.vision_encoder_parallel == "tp",
                ),
            )
        else:
            multimodal_batch = None

        top_logprobs_nums = token_ids_logprobs = None
        if batch.return_logprob:
            top_logprobs_nums = [0] * total_bs
            token_ids_logprobs = [None] * total_bs
            for rank, info in enumerate(batch.reqs_info):
                slots = plan.request_slice(rank)
                if info.top_logprobs_nums is not None:
                    top_logprobs_nums[slots] = info.top_logprobs_nums
                if info.token_ids_logprobs is not None:
                    token_ids_logprobs[slots] = info.token_ids_logprobs

        # Hidden-state capture retains the original tensor independently of
        # logprob selection, so it also uses the DP-padded logprob path.
        input_logprob_indices = None
        merged_extend_input_logprob_token_ids = None
        if batch.return_logprob and batch.forward_mode.is_extend():
            input_logprob_indices = np.zeros(total_token_size, dtype=np.int32)
            merged_extend_input_logprob_token_ids = np.zeros(total_token_size, dtype=np.int32)
            for rank, info in enumerate(batch.reqs_info):
                out_pt = plan.token_slice(rank).start
                token_id_pt = 0
                starts = info.extend_logprob_start_lens or []
                token_ids = info.extend_input_logprob_token_ids
                real = plan.sequences.requests(rank)
                for extend_len, start_len, local_pt in zip(
                    info.extend_lens or [], starts, plan.sequences.query_starts[real]
                ):
                    num_logprobs = max(extend_len - start_len, 0)
                    if num_logprobs > 0:
                        end_pt = out_pt + num_logprobs
                        input_logprob_indices[out_pt:end_pt] = np.arange(
                            local_pt + start_len,
                            local_pt + extend_len,
                            dtype=np.int32,
                        )
                        if token_ids is not None:
                            merged_extend_input_logprob_token_ids[out_pt:end_pt] = token_ids[
                                token_id_pt : token_id_pt + num_logprobs
                            ]
                        out_pt = end_pt
                        token_id_pt += num_logprobs
        return ModelWorkerBatch(
            inputs=inputs,
            layout=plan,
            bid=bid,
            forward_mode=batch.forward_mode,
            real_input_ids_len=real_input_ids_len,
            return_logprob=batch.return_logprob,
            return_output_logprob_only=batch.return_output_logprob_only,
            top_logprobs_nums=top_logprobs_nums,
            token_ids_logprobs=token_ids_logprobs,
            sampling_info=sampling_info,
            mrope_positions=mrope_positions,
            cache_loc=cache_loc_cpu,
            extend_logprob_start_lens=extend_logprob_start_lens,
            extend_input_logprob_token_ids=merged_extend_input_logprob_token_ids,
            input_logprob_indices=input_logprob_indices,
            logits_indices=logits_indices,
            lora_ids=lora_ids,
            real_bs=real_bs,
            real_bs_per_dp=real_bs_per_dp,
            logits_indices_selector=logits_indices_selector,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if batch.return_hidden_states
                or (batch.spec_algorithm is not None and (not batch.spec_algorithm.is_none()))
                else CaptureHiddenMode.NULL
            ),
            dp_size=batch.dp_size,
            per_dp_bs_size=per_dp_bs_padding,
            launch_done=batch.launch_done,
            input_embedding=input_embedding,
            multimodal_batch=multimodal_batch,
            apply_for_deepstack=apply_for_deepstack,
            deepstack_visual_embedding=deepstack_visual_embedding,
            has_initial_state=has_initial_state_cpu,
            spec_algorithm=batch.spec_algorithm,
        )
