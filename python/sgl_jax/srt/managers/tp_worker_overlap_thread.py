"""A tensor parallel worker."""

import copy
import dataclasses
import logging
import signal
import threading
from queue import Queue
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)

PHASE_B_DEVICE_RELAY_FIELDS = (
    "topk_index",
    "topk_p",
    "verified_id",
    "hidden_states",
    "previous_token_list",
)


class ModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        model_class=None,
        precompile_params: dict | None = None,
    ):
        # Load the model
        self.worker = ModelWorker(server_args, mesh=mesh)
        # overlap mode set worker need_prepare_lora_batch to False
        self.worker.need_prepare_lora_batch = False

        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = jnp.zeros((self.max_running_requests * 5,), dtype=jnp.int32)
        self.mesh = mesh
        sharding = NamedSharding(mesh, PartitionSpec(None))
        self.future_token_ids_map = jax.device_put(self.future_token_ids_map, sharding)
        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.spec_worker = None
        self.pending_spec_draft_extend_result = None
        # JAX handles device execution automatically, no need for explicit streams
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            daemon=bool(server_args.enable_single_process),
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        self.async_gather_fn = jax.jit(lambda x: x, out_shardings=replicated_sharding)

    def get_model_runner(self):
        return self.worker.get_model_runner()

    @property
    def model_config(self):
        return self.worker.model_config

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool_allocator,
        )

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def get_max_padded_size(self):
        return self.worker.get_max_padded_size()

    def get_precompile_paddings(self):
        return self.worker.get_precompile_paddings()

    def forward_thread_func(self):
        try:
            self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("ModelWorkerClient hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def forward_thread_func_(self):
        while True:
            (
                work_kind,
                model_worker_batch,
                future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            ) = self.input_queue.get()
            if not model_worker_batch:
                break
            if work_kind == "spec_full":
                assert self.spec_worker is not None
                self._apply_pending_spec_draft_extend_to_batch(model_worker_batch)
                with jax.profiler.TraceAnnotation(
                    f"forward_batch_speculative_generation {model_worker_batch.bid}"
                ):
                    result = self.spec_worker.forward_batch_speculative_generation(
                        model_worker_batch
                    )
                self.output_queue.put(("spec_full", result))
                continue

            if work_kind == "spec_split":
                assert self.spec_worker is not None
                chained_candidate = self._pop_matching_same_batch_spec_chain_candidate(
                    model_worker_batch
                )
                if chained_candidate is None:
                    self._apply_pending_spec_draft_extend_to_batch(model_worker_batch)
                    with jax.profiler.TraceAnnotation(
                        f"forward_batch_speculative_verify_phase {model_worker_batch.bid}"
                    ):
                        verify_async_result = (
                            self.spec_worker.forward_batch_speculative_verify_phase_enqueue(
                                model_worker_batch,
                            )
                        )
                    self._start_same_batch_spec_chain_prepare_prewarm(model_worker_batch)
                else:
                    model_worker_batch = chained_candidate.model_worker_batch
                    verify_async_result = chained_candidate.verify_async_result

                phase_a_holder = {}
                pending_dispatch_done = threading.Event()
                pending_dispatch_holder = {}

                def _resolve_spec_verify_phase(
                    model_worker_batch=model_worker_batch,
                    verify_async_result=verify_async_result,
                    phase_a_holder=phase_a_holder,
                    pending_dispatch_done=pending_dispatch_done,
                    pending_dispatch_holder=pending_dispatch_holder,
                ):
                    try:
                        with jax.profiler.TraceAnnotation(
                            f"resolve_spec_verify_phase_a {model_worker_batch.bid}"
                        ):
                            verify_result = self.spec_worker.materialize_speculative_verify_phase(
                                verify_async_result
                            )
                        phase_a_holder["result"] = verify_result
                        phase_a_holder["phase_a_ready"].set()
                        pending_dispatch_done.wait()
                        pending_result = pending_dispatch_holder.get("result")
                        prebuilt_chain_candidate = None
                        if pending_result is not None and not pending_dispatch_holder.get(
                            "stashed_chain_candidate",
                            False,
                        ):
                            prebuilt_chain_candidate = (
                                self._prebuild_same_batch_spec_chain_candidate_after_phase_a(
                                    model_worker_batch,
                                    verify_result,
                                )
                            )
                            phase_a_holder["prebuilt_chain_candidate"] = prebuilt_chain_candidate
                            padded_new_seq_lens_host = getattr(
                                verify_result,
                                "padded_new_seq_lens_host",
                                None,
                            )
                            if padded_new_seq_lens_host is not None:
                                pending_result.padded_new_seq_lens_host = padded_new_seq_lens_host
                            if prebuilt_chain_candidate is not None:
                                self._stash_prebuilt_same_batch_spec_chain_candidate(
                                    prebuilt_chain_candidate,
                                    pending_result,
                                )
                            else:
                                self._stash_same_batch_spec_chain_candidate(
                                    model_worker_batch,
                                    pending_result,
                                )
                        self.output_queue.put(("spec_verify", verify_result))
                    except Exception:
                        traceback = get_exception_traceback()
                        phase_a_holder["exception"] = traceback
                        phase_a_holder["phase_a_ready"].set()
                        self.output_queue.put(("spec_verify_exception", traceback))

                phase_a_holder["phase_a_ready"] = threading.Event()
                phase_a_thread = threading.Thread(
                    target=_resolve_spec_verify_phase,
                    name=f"spec-phase-a-resolver-{model_worker_batch.bid}",
                )
                phase_a_thread.start()
                from sgl_jax.srt.speculative.draft_extend_fused import (
                    spec_decode_dispatch_draft_extend_for_pending,
                )

                with jax.profiler.TraceAnnotation(
                    f"dispatch_spec_draft_extend_after_verify_enqueue {model_worker_batch.bid}"
                ):
                    try:
                        self.pending_spec_draft_extend_result = (
                            spec_decode_dispatch_draft_extend_for_pending(
                                self.spec_worker,
                                model_worker_batch,
                                verify_async_result,
                            )
                        )
                        pending_dispatch_holder["result"] = self.pending_spec_draft_extend_result
                        if self.pending_spec_draft_extend_result is not None:
                            candidate = self._prebuild_same_batch_spec_chain_candidate_after_phase_b_dispatch(
                                model_worker_batch,
                                self.pending_spec_draft_extend_result,
                            )
                            if candidate is not None:
                                self._stash_prebuilt_same_batch_spec_chain_candidate(
                                    candidate,
                                    self.pending_spec_draft_extend_result,
                                )
                                pending_dispatch_holder["stashed_chain_candidate"] = True
                    finally:
                        pending_dispatch_done.set()
                phase_a_thread.join()
                if "exception" in phase_a_holder:
                    raise RuntimeError(phase_a_holder["exception"])
                if self.pending_spec_draft_extend_result is None:
                    verify_kind, verify_payload = self.output_queue.get()
                    if verify_kind == "spec_verify_exception":
                        raise RuntimeError(verify_payload)
                    assert verify_kind == "spec_verify"
                    with jax.profiler.TraceAnnotation(
                        f"forward_batch_speculative_draft_extend_phase {model_worker_batch.bid}"
                    ):
                        self.pending_spec_draft_extend_result = (
                            self.spec_worker.forward_batch_speculative_draft_extend_phase(
                                model_worker_batch,
                                verify_payload,
                            )
                        )
                    self.output_queue.put((verify_kind, verify_payload))
                continue

            # Resolve future tokens in the input
            input_ids = model_worker_batch.forward_batch.input_ids
            model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                input_ids, self.future_token_ids_map, self.mesh
            )

            # Run forward
            with jax.profiler.TraceAnnotation(f"forward_batch_generation {model_worker_batch.bid}"):
                logits_output, next_token_ids, cache_miss_count = (
                    self.worker.forward_batch_generation(
                        model_worker_batch,
                        model_worker_batch.launch_done,
                        sampling_metadata=sampling_metadata,
                        forward_metadata=forward_metadata,
                    )
                )
            next_token_ids = self.async_gather_fn(next_token_ids)
            # Update the future token ids map
            self.future_token_ids_map = set_future_token_ids(
                self.future_token_ids_map,
                future_token_ids_ct,
                next_token_ids,
                self.mesh,
            )
            self.output_queue.put(("generation", logits_output, next_token_ids, cache_miss_count))

    def resolve_last_batch_result(self, launch_done: threading.Event | None = None):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.

        Uses jax.copy_to_host_async to start all device-to-host copies in
        parallel, then materializes them. This lets the four arrays we need
        overlap on PCIe rather than serializing the per-array sync that
        jax.device_get does.
        """
        kind, logits_output, next_token_ids, cache_miss_count = self.output_queue.get()
        assert kind == "generation", f"expected generation result, got {kind!r}"
        # Step 1: kick off async D2H copies for everything we need
        async_next_logprobs = (
            jax.copy_to_host_async(logits_output.next_token_logprobs)
            if logits_output.next_token_logprobs is not None
            else None
        )
        async_input_logprobs = (
            jax.copy_to_host_async(logits_output.input_token_logprobs)
            if logits_output.input_token_logprobs is not None
            else None
        )
        async_hidden_states = (
            jax.copy_to_host_async(logits_output.hidden_states)
            if logits_output.hidden_states is not None
            else None
        )
        async_next_tokens = jax.copy_to_host_async(next_token_ids)

        # Step 2: materialize. The first np.asarray waits for that array's
        # copy; the others have been making progress in parallel.
        if async_next_logprobs is not None:
            logits_output.next_token_logprobs = np.asarray(async_next_logprobs).tolist()
        if async_input_logprobs is not None:
            logits_output.input_token_logprobs = np.asarray(async_input_logprobs).tolist()
        if async_hidden_states is not None:
            logits_output.hidden_states = np.asarray(async_hidden_states)
        next_token_ids = np.asarray(async_next_tokens).tolist()

        if launch_done is not None:
            launch_done.wait()

        return logits_output, next_token_ids, cache_miss_count

    def resolve_last_spec_full_result(self):
        kind, result = self.output_queue.get()
        assert kind == "spec_full", f"expected spec_full result, got {kind!r}"
        return result

    def resolve_last_spec_verify_result(self):
        kind, result = self.output_queue.get()
        if kind == "spec_verify_exception":
            raise RuntimeError(result)
        assert kind == "spec_verify", f"expected spec_verify result, got {kind!r}"
        return result

    def resolve_last_spec_draft_extend_result(self):
        kind, result = self.output_queue.get()
        assert kind == "spec_draft_extend", f"expected spec_draft_extend result, got {kind!r}"
        return result

    def _apply_pending_spec_draft_extend_to_batch(self, model_worker_batch: ModelWorkerBatch):
        """Apply the previous split Phase-B state to the next decode batch.

        This runs on the single ordered worker thread. It uses only host arrays
        that Phase B has already materialized, so it does not introduce an
        extra scheduler-side JAX launch or cross-host program-order hazard.
        """
        pending = getattr(self, "pending_spec_draft_extend_result", None)
        if pending is None:
            return
        if not model_worker_batch.forward_mode.is_decode():
            return

        next_draft_input = pending.next_draft_input
        phase_req_pool_indices = pending.req_pool_indices
        if next_draft_input is None or phase_req_pool_indices is None:
            self.pending_spec_draft_extend_result = None
            return

        spec_info = model_worker_batch.spec_info_padded
        if spec_info is None:
            self.pending_spec_draft_extend_result = None
            return

        phase_req_pool_indices = np.asarray(phase_req_pool_indices, dtype=np.int32)
        phase_pos = {int(req_pool_idx): i for i, req_pool_idx in enumerate(phase_req_pool_indices)}
        current_req_pool_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)
        padded_next_draft_input = getattr(pending, "padded_next_draft_input", None)
        padded_req_pool_indices = getattr(pending, "padded_req_pool_indices", None)
        padded_phase_pos = {}
        if padded_next_draft_input is not None and padded_req_pool_indices is not None:
            real_req_pool = {int(req_pool_idx) for req_pool_idx in phase_req_pool_indices}
            for i, req_pool_idx in enumerate(np.asarray(padded_req_pool_indices, dtype=np.int32)):
                req_pool_idx = int(req_pool_idx)
                if req_pool_idx in real_req_pool and req_pool_idx not in padded_phase_pos:
                    padded_phase_pos[req_pool_idx] = i

        direct_padded_fields = set()
        if (
            padded_next_draft_input is not None
            and padded_req_pool_indices is not None
            and current_req_pool_indices.shape
            == np.asarray(padded_req_pool_indices, dtype=np.int32).shape
            and np.array_equal(
                current_req_pool_indices,
                np.asarray(padded_req_pool_indices, dtype=np.int32),
            )
        ):
            # Exact slot layout match: keep the next-verify topk device handle
            # in the ordered worker thread. Scheduler-side layout changes still
            # fall back to row materialization below.
            with jax.profiler.TraceAnnotation("apply_padded_phase_b_topk_fastpath"):
                for field in (
                    "topk_index",
                    "verified_id",
                    "new_seq_lens",
                    "allocate_lens",
                    "verify_write_lens",
                ):
                    value = getattr(padded_next_draft_input, field, None)
                    if value is not None:
                        setattr(spec_info, field, value)
                        direct_padded_fields.add(field)
                previous_token_list = getattr(
                    padded_next_draft_input,
                    "previous_token_list",
                    None,
                )
                if previous_token_list is not None:
                    spec_info.previous_token_list = previous_token_list

        fields = (
            "topk_p",
            "topk_index",
            "hidden_states",
            "verified_id",
            "allocate_lens",
            "verify_write_lens",
            "new_seq_lens",
            "accept_length",
            "accept_length_cpu",
        )
        for slot, req_pool_idx in enumerate(current_req_pool_indices):
            req_pool_idx = int(req_pool_idx)
            phase_i = phase_pos.get(req_pool_idx)
            padded_phase_i = padded_phase_pos.get(req_pool_idx)
            if phase_i is None and padded_phase_i is None:
                continue
            for field in fields:
                if field in direct_padded_fields:
                    continue
                src = getattr(next_draft_input, field, None)
                src_i = phase_i
                if src is not None and src_i is not None:
                    try:
                        if src_i >= src.shape[0]:
                            src = None
                            src_i = None
                    except AttributeError:
                        pass
                if (
                    (src is None or src_i is None)
                    and padded_next_draft_input is not None
                    and padded_phase_i is not None
                ):
                    padded_src = getattr(padded_next_draft_input, field, None)
                    if padded_src is not None and padded_phase_i < padded_src.shape[0]:
                        src = padded_src
                        src_i = padded_phase_i
                dst = getattr(spec_info, field, None)
                if src is None or dst is None or src_i is None:
                    continue
                dst[slot] = np.asarray(src)[src_i]

        self.pending_spec_draft_extend_result = None

    def _pop_matching_same_batch_spec_chain_candidate(
        self,
        model_worker_batch: ModelWorkerBatch,
    ):
        candidate = getattr(self, "pending_same_batch_spec_chain_candidate", None)
        if candidate is None:
            return None
        self.pending_same_batch_spec_chain_candidate = None
        current_req_pool = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)
        candidate_req_pool = np.asarray(candidate.req_pool_indices, dtype=np.int32)
        if current_req_pool.shape != candidate_req_pool.shape:
            return None
        if not np.array_equal(current_req_pool, candidate_req_pool):
            return None
        self.pending_spec_draft_extend_result = None
        return candidate

    def _stash_same_batch_spec_chain_candidate(
        self,
        model_worker_batch: ModelWorkerBatch,
        pending,
    ) -> None:
        candidate_batch = self._build_same_batch_spec_chain_candidate_batch(
            model_worker_batch,
            pending,
        )
        if candidate_batch is None:
            self.pending_same_batch_spec_chain_candidate = None
            return
        with jax.profiler.TraceAnnotation(
            f"forward_batch_speculative_chained_verify_phase {candidate_batch.bid}"
        ):
            verify_async_result = self.spec_worker.forward_batch_speculative_verify_phase_enqueue(
                candidate_batch,
            )
        self.pending_same_batch_spec_chain_candidate = SimpleNamespace(
            req_pool_indices=np.asarray(candidate_batch.req_pool_indices, dtype=np.int32).copy(),
            verify_async_result=verify_async_result,
            model_worker_batch=candidate_batch,
        )

    def _prebuild_same_batch_spec_chain_candidate_after_phase_a(
        self,
        model_worker_batch: ModelWorkerBatch,
        verify_result,
    ):
        with jax.profiler.TraceAnnotation("prebuild_same_batch_spec_chain_candidate_after_phase_a"):
            from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

            padded_new_seq_lens_host = getattr(verify_result, "padded_new_seq_lens_host", None)
            if padded_new_seq_lens_host is None:
                return None
            prebuild_draft_input = EagleDraftInput(
                new_seq_lens=np.asarray(padded_new_seq_lens_host, dtype=np.int32),
            )
            prebuild_pending = SimpleNamespace(
                padded_next_draft_input=prebuild_draft_input,
                padded_req_pool_indices=np.asarray(
                    model_worker_batch.req_pool_indices,
                    dtype=np.int32,
                ).copy(),
                padded_new_seq_lens_host=np.asarray(
                    padded_new_seq_lens_host,
                    dtype=np.int32,
                ).copy(),
            )
            candidate = self._build_same_batch_spec_chain_candidate_batch(
                model_worker_batch,
                prebuild_pending,
            )
            if candidate is not None:
                candidate.prepared_fused_greedy_verify_launch = (
                    self._prepare_chained_verify_launch_after_phase_a(candidate)
                )
            return candidate

    def _prewarm_same_batch_spec_chain_prepare_cache(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> None:
        if not getattr(model_worker_batch, "allow_same_batch_spec_chain", False):
            return
        current_seq_lens = getattr(model_worker_batch, "seq_lens", None)
        if current_seq_lens is None:
            return
        from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

        prewarm_pending = SimpleNamespace(
            padded_next_draft_input=EagleDraftInput(
                new_seq_lens=np.asarray(current_seq_lens, dtype=np.int32),
            ),
            padded_req_pool_indices=np.asarray(
                model_worker_batch.req_pool_indices,
                dtype=np.int32,
            ).copy(),
            padded_new_seq_lens_host=np.asarray(current_seq_lens, dtype=np.int32).copy(),
        )
        candidate = self._build_same_batch_spec_chain_candidate_batch(
            model_worker_batch,
            prewarm_pending,
        )
        if candidate is None:
            return
        with jax.profiler.TraceAnnotation("prewarm_chained_verify_launch_cache"):
            self._prepare_chained_verify_launch_after_phase_a(candidate)

    def _prebuild_same_batch_spec_chain_candidate_after_phase_b_dispatch(
        self,
        model_worker_batch: ModelWorkerBatch,
        pending,
    ):
        with jax.profiler.TraceAnnotation(
            "prebuild_same_batch_spec_chain_candidate_after_phase_b_dispatch"
        ):
            candidate = self._build_same_batch_spec_chain_candidate_batch(
                model_worker_batch,
                pending,
            )
            if candidate is not None:
                candidate.prepared_fused_greedy_verify_launch = (
                    self._prepare_chained_verify_launch_after_phase_a(candidate)
                )
            return candidate

    def _start_same_batch_spec_chain_prepare_prewarm(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> None:
        if not getattr(model_worker_batch, "allow_same_batch_spec_chain", False):
            return

        def _run_prewarm():
            try:
                self._prewarm_same_batch_spec_chain_prepare_cache(model_worker_batch)
            except Exception:
                logger.debug(
                    "same_batch_chain prepare cache prewarm failed",
                    exc_info=True,
                )

        threading.Thread(
            target=_run_prewarm,
            name=f"spec-chain-prepare-prewarm-{model_worker_batch.bid}",
            daemon=True,
        ).start()

    def _prepare_chained_verify_launch_after_phase_a(self, candidate_batch: ModelWorkerBatch):
        try:
            from sgl_jax.srt.speculative.draft_extend_fused import (
                prepare_fused_greedy_verify_launch,
            )

            padded_allocate_lens = np.asarray(candidate_batch.spec_info_padded.allocate_lens)
            selector = np.asarray(candidate_batch.logits_indices_selector)
            if selector.size > 0 and int(np.max(selector)) < len(padded_allocate_lens):
                compact_allocate_lens = padded_allocate_lens[selector]
            else:
                compact_allocate_lens = padded_allocate_lens
            with jax.profiler.TraceAnnotation("prepare_chained_verify_launch_after_phase_a"):
                return prepare_fused_greedy_verify_launch(
                    self.spec_worker,
                    candidate_batch,
                    padded_allocate_lens,
                    compact_allocate_lens,
                    require_previous=False,
                )
        except Exception:
            logger.debug(
                "same_batch_chain prepared launch unavailable",
                exc_info=True,
            )
            return None

    def _stash_prebuilt_same_batch_spec_chain_candidate(
        self,
        candidate_batch: ModelWorkerBatch,
        pending,
    ) -> None:
        padded_next_draft_input = getattr(pending, "padded_next_draft_input", None)
        padded_req_pool_indices = getattr(pending, "padded_req_pool_indices", None)
        if padded_next_draft_input is None or padded_req_pool_indices is None:
            self.pending_same_batch_spec_chain_candidate = None
            return
        if not np.array_equal(
            np.asarray(candidate_batch.req_pool_indices, dtype=np.int32),
            np.asarray(padded_req_pool_indices, dtype=np.int32),
        ):
            self.pending_same_batch_spec_chain_candidate = None
            return

        candidate_spec_info = candidate_batch.spec_info_padded
        prepared_launch = getattr(candidate_batch, "prepared_fused_greedy_verify_launch", None)
        for field in PHASE_B_DEVICE_RELAY_FIELDS:
            value = getattr(padded_next_draft_input, field, None)
            if value is not None:
                setattr(candidate_spec_info, field, value)

        if prepared_launch is not None:
            previous_verified_id = getattr(padded_next_draft_input, "verified_id", None)
            previous_token_list = getattr(padded_next_draft_input, "previous_token_list", None)
            if previous_token_list is None:
                topk_index = getattr(padded_next_draft_input, "topk_index", None)
                if topk_index is not None:
                    previous_token_list = topk_index[:, :, 0]
            if previous_verified_id is None or previous_token_list is None:
                self.pending_same_batch_spec_chain_candidate = None
                return
            prepared_launch = prepared_launch._replace(
                previous_verified_id=previous_verified_id,
                previous_token_list=previous_token_list,
            )
            candidate_batch.prepared_fused_greedy_verify_launch = prepared_launch
        elif (
            getattr(candidate_spec_info, "topk_index", None) is None
            and getattr(candidate_spec_info, "previous_token_list", None) is None
        ) or getattr(candidate_spec_info, "verified_id", None) is None:
            self.pending_same_batch_spec_chain_candidate = None
            return

        with jax.profiler.TraceAnnotation(
            f"forward_batch_speculative_chained_verify_phase {candidate_batch.bid}"
        ):
            verify_async_result = self.spec_worker.forward_batch_speculative_verify_phase_enqueue(
                candidate_batch,
                prepared_launch=prepared_launch,
            )
        self.pending_same_batch_spec_chain_candidate = SimpleNamespace(
            req_pool_indices=np.asarray(candidate_batch.req_pool_indices, dtype=np.int32).copy(),
            verify_async_result=verify_async_result,
            model_worker_batch=candidate_batch,
        )

    def _build_same_batch_spec_chain_candidate_batch(
        self,
        model_worker_batch: ModelWorkerBatch,
        pending,
    ):
        """Build a discardable same-layout chained verify candidate batch.

        This is data construction only. The caller owns any later launch and
        commit/discard decision.
        """
        if not getattr(model_worker_batch, "allow_same_batch_spec_chain", False):
            with jax.profiler.TraceAnnotation("same_batch_chain_build_skip:not_allowed"):
                pass
            return None
        padded_next_draft_input = getattr(pending, "padded_next_draft_input", None)
        padded_req_pool_indices = getattr(pending, "padded_req_pool_indices", None)
        preview_req_pool_indices = getattr(
            model_worker_batch,
            "same_batch_chain_req_pool_indices",
            None,
        )
        if padded_next_draft_input is None or padded_req_pool_indices is None:
            with jax.profiler.TraceAnnotation("same_batch_chain_build_skip:missing_padded"):
                pass
            return None
        padded_req_pool_indices = np.asarray(padded_req_pool_indices, dtype=np.int32)
        if preview_req_pool_indices is not None:
            preview_req_pool_indices = np.asarray(preview_req_pool_indices, dtype=np.int32)
            if padded_req_pool_indices.shape != preview_req_pool_indices.shape:
                with jax.profiler.TraceAnnotation("same_batch_chain_build_skip:preview_shape"):
                    pass
                return None
            if not np.array_equal(padded_req_pool_indices, preview_req_pool_indices):
                with jax.profiler.TraceAnnotation("same_batch_chain_build_skip:preview_req_pool"):
                    pass
                return None
        current_req_pool_indices = np.asarray(
            getattr(model_worker_batch, "req_pool_indices", []),
            dtype=np.int32,
        )
        if padded_req_pool_indices.shape != current_req_pool_indices.shape:
            with jax.profiler.TraceAnnotation("same_batch_chain_build_skip:current_shape"):
                pass
            return None
        if not np.array_equal(padded_req_pool_indices, current_req_pool_indices):
            with jax.profiler.TraceAnnotation("same_batch_chain_build_skip:current_req_pool"):
                pass
            return None
        out_cache_loc_chunks = getattr(
            model_worker_batch,
            "same_batch_chain_out_cache_loc_chunks",
            None,
        )
        verify_write_lens = getattr(
            model_worker_batch,
            "same_batch_chain_verify_write_lens",
            None,
        )
        allocate_lens = getattr(
            model_worker_batch,
            "same_batch_chain_allocate_lens",
            None,
        )
        if out_cache_loc_chunks is None or verify_write_lens is None or allocate_lens is None:
            preview = self._peek_same_batch_spec_chain_preview_from_pending(
                model_worker_batch,
                pending,
            )
            if preview is None:
                with jax.profiler.TraceAnnotation("same_batch_chain_build_skip:preview_none"):
                    pass
                return None
            out_cache_loc_chunks, verify_write_lens, allocate_lens = preview

        candidate = copy.copy(model_worker_batch)
        candidate.out_cache_loc = self._pad_same_batch_spec_chain_out_cache_loc(
            model_worker_batch,
            out_cache_loc_chunks,
        )
        candidate.spec_info_padded = copy.copy(padded_next_draft_input)
        candidate.spec_info_padded.allocate_lens = np.asarray(allocate_lens, dtype=np.int32)
        candidate.spec_info_padded.verify_write_lens = np.asarray(
            verify_write_lens,
            dtype=np.int32,
        )
        candidate_seq_lens = getattr(pending, "padded_new_seq_lens_host", None)
        if candidate_seq_lens is None:
            candidate_seq_lens = getattr(padded_next_draft_input, "new_seq_lens", None)
        if candidate_seq_lens is not None and not isinstance(candidate_seq_lens, jax.Array):
            candidate.seq_lens = np.asarray(candidate_seq_lens, dtype=np.int32).copy()
            candidate.seq_lens_sum = int(candidate.seq_lens.sum())
        candidate.allow_same_batch_spec_chain = True
        candidate.skip_fused_verify_padding_for_decode = True
        candidate.same_batch_chain_req_pool_indices = None
        candidate.same_batch_chain_out_cache_loc_chunks = None
        candidate.same_batch_chain_verify_write_lens = np.asarray(
            verify_write_lens,
            dtype=np.int32,
        ).copy()
        candidate.same_batch_chain_allocate_lens = np.asarray(
            allocate_lens,
            dtype=np.int32,
        ).copy()
        return candidate

    def _peek_same_batch_spec_chain_preview_from_pending(
        self,
        model_worker_batch: ModelWorkerBatch,
        pending,
    ):
        padded_next_draft_input = getattr(pending, "padded_next_draft_input", None)
        padded_req_pool_indices = getattr(pending, "padded_req_pool_indices", None)
        spec_info = getattr(model_worker_batch, "spec_info_padded", None)
        if padded_next_draft_input is None or padded_req_pool_indices is None or spec_info is None:
            with jax.profiler.TraceAnnotation("same_batch_chain_peek_skip:missing_input"):
                pass
            logger.info(
                "same_batch_chain_peek_skip reason=missing_input padded=%s req_pool=%s spec=%s",
                padded_next_draft_input is not None,
                padded_req_pool_indices is not None,
                spec_info is not None,
            )
            return None

        new_seq_lens = getattr(pending, "padded_new_seq_lens_host", None)
        if new_seq_lens is None:
            new_seq_lens = getattr(padded_next_draft_input, "new_seq_lens", None)
        old_verify_write_lens = getattr(
            model_worker_batch,
            "same_batch_chain_verify_write_lens",
            None,
        )
        if old_verify_write_lens is None:
            old_verify_write_lens = getattr(spec_info, "verify_write_lens", None)
        allocate_lens = getattr(
            model_worker_batch,
            "same_batch_chain_allocate_lens",
            None,
        )
        if allocate_lens is None:
            allocate_lens = getattr(spec_info, "allocate_lens", None)
        if new_seq_lens is None or old_verify_write_lens is None or allocate_lens is None:
            with jax.profiler.TraceAnnotation("same_batch_chain_peek_skip:missing_frontier"):
                pass
            logger.info(
                "same_batch_chain_peek_skip reason=missing_frontier new_seq=%s old_write=%s alloc=%s",
                new_seq_lens is not None,
                old_verify_write_lens is not None,
                allocate_lens is not None,
            )
            return None
        req_pool_indices = np.asarray(padded_req_pool_indices, dtype=np.int32)
        old_verify_write_lens = np.asarray(old_verify_write_lens, dtype=np.int32)
        allocate_lens = np.asarray(allocate_lens, dtype=np.int32)

        if isinstance(new_seq_lens, jax.Array):
            return self._peek_same_batch_spec_chain_device_lens_reserved_suffix(
                model_worker_batch,
                padded_next_draft_input,
                req_pool_indices,
                old_verify_write_lens,
                allocate_lens,
            )

        new_seq_lens = np.asarray(new_seq_lens, dtype=np.int32)
        if (
            req_pool_indices.shape != new_seq_lens.shape
            or req_pool_indices.shape != old_verify_write_lens.shape
            or req_pool_indices.shape != allocate_lens.shape
        ):
            with jax.profiler.TraceAnnotation("same_batch_chain_peek_skip:shape"):
                pass
            logger.info(
                "same_batch_chain_peek_skip reason=shape req_pool=%s new_seq=%s old_write=%s alloc=%s",
                req_pool_indices.shape,
                new_seq_lens.shape,
                old_verify_write_lens.shape,
                allocate_lens.shape,
            )
            return None

        alloc_len_per_decode = padded_next_draft_input.get_spec_adjust_token_coefficient()
        valid_slots = (req_pool_indices >= 0) & (new_seq_lens > 0)
        required_write_lens = np.where(
            valid_slots,
            new_seq_lens + alloc_len_per_decode - 1,
            old_verify_write_lens,
        )
        if np.any(required_write_lens[valid_slots] > allocate_lens[valid_slots]):
            with jax.profiler.TraceAnnotation("same_batch_chain_peek_skip:reserve"):
                pass
            logger.info(
                "same_batch_chain_peek_skip reason=reserve max_required=%d max_alloc=%d min_slack=%d",
                int(np.max(required_write_lens[valid_slots])),
                int(np.max(allocate_lens[valid_slots])),
                int(np.min(allocate_lens[valid_slots] - required_write_lens[valid_slots])),
            )
            return None

        req_to_token_pool = getattr(
            getattr(self.worker, "model_runner", None), "req_to_token_pool", None
        )
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        if req_to_token is None:
            with jax.profiler.TraceAnnotation("same_batch_chain_peek_skip:req_to_token"):
                pass
            logger.info("same_batch_chain_peek_skip reason=req_to_token")
            return None

        dp_size = int(getattr(model_worker_batch, "dp_size", 1) or 1)
        per_dp_bs = int(getattr(model_worker_batch, "per_dp_bs_size", 0) or 0)
        out_cache_loc_chunks = []
        if dp_size > 1 and per_dp_bs > 0 and req_pool_indices.shape[0] == dp_size * per_dp_bs:
            for dp_rank in range(dp_size):
                start = dp_rank * per_dp_bs
                end = start + per_dp_bs
                loc_chunks = [
                    req_to_token[req_pool_idx, int(old_write) : int(required_write)]
                    for req_pool_idx, old_write, required_write, valid in zip(
                        req_pool_indices[start:end],
                        old_verify_write_lens[start:end],
                        required_write_lens[start:end],
                        valid_slots[start:end],
                        strict=True,
                    )
                    if bool(valid) and int(required_write) > int(old_write)
                ]
                out_cache_loc_chunks.append(
                    np.concatenate(loc_chunks).astype(np.int32, copy=False)
                    if loc_chunks
                    else np.empty(0, dtype=np.int32)
                )
        else:
            loc_chunks = [
                req_to_token[req_pool_idx, int(old_write) : int(required_write)]
                for req_pool_idx, old_write, required_write, valid in zip(
                    req_pool_indices,
                    old_verify_write_lens,
                    required_write_lens,
                    valid_slots,
                    strict=True,
                )
                if bool(valid) and int(required_write) > int(old_write)
            ]
            out_cache_loc_chunks = [
                (
                    np.concatenate(loc_chunks).astype(np.int32, copy=False)
                    if loc_chunks
                    else np.empty(0, dtype=np.int32)
                )
            ]
        return (
            out_cache_loc_chunks,
            required_write_lens.astype(np.int32, copy=False),
            allocate_lens.astype(np.int32, copy=False),
        )

    def _peek_same_batch_spec_chain_device_lens_reserved_suffix(
        self,
        model_worker_batch: ModelWorkerBatch,
        padded_next_draft_input,
        req_pool_indices: np.ndarray,
        old_verify_write_lens: np.ndarray,
        allocate_lens: np.ndarray,
    ):
        if (
            req_pool_indices.shape != old_verify_write_lens.shape
            or req_pool_indices.shape != allocate_lens.shape
        ):
            with jax.profiler.TraceAnnotation("same_batch_chain_peek_skip:device_shape"):
                pass
            logger.info(
                "same_batch_chain_peek_skip reason=device_shape req_pool=%s old_write=%s alloc=%s",
                req_pool_indices.shape,
                old_verify_write_lens.shape,
                allocate_lens.shape,
            )
            return None

        alloc_len_per_decode = padded_next_draft_input.get_spec_adjust_token_coefficient()
        valid_slots = req_pool_indices >= 0
        required_write_lens = np.where(
            valid_slots,
            old_verify_write_lens + alloc_len_per_decode,
            old_verify_write_lens,
        )
        if np.any(required_write_lens[valid_slots] > allocate_lens[valid_slots]):
            with jax.profiler.TraceAnnotation("same_batch_chain_peek_skip:device_reserve"):
                pass
            logger.info(
                "same_batch_chain_peek_skip reason=device_reserve max_required=%d max_alloc=%d min_slack=%d",
                int(np.max(required_write_lens[valid_slots])),
                int(np.max(allocate_lens[valid_slots])),
                int(np.min(allocate_lens[valid_slots] - required_write_lens[valid_slots])),
            )
            return None

        req_to_token_pool = getattr(
            getattr(self.worker, "model_runner", None), "req_to_token_pool", None
        )
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        if req_to_token is None:
            with jax.profiler.TraceAnnotation("same_batch_chain_peek_skip:device_req_to_token"):
                pass
            logger.info("same_batch_chain_peek_skip reason=device_req_to_token")
            return None

        dp_size = int(getattr(model_worker_batch, "dp_size", 1) or 1)
        per_dp_bs = int(getattr(model_worker_batch, "per_dp_bs_size", 0) or 0)
        out_cache_loc_chunks = []
        if dp_size > 1 and per_dp_bs > 0 and req_pool_indices.shape[0] == dp_size * per_dp_bs:
            for dp_rank in range(dp_size):
                start = dp_rank * per_dp_bs
                end = start + per_dp_bs
                loc_chunks = [
                    req_to_token[req_pool_idx, int(old_write) : int(required_write)]
                    for req_pool_idx, old_write, required_write, valid in zip(
                        req_pool_indices[start:end],
                        old_verify_write_lens[start:end],
                        required_write_lens[start:end],
                        valid_slots[start:end],
                        strict=True,
                    )
                    if bool(valid) and int(required_write) > int(old_write)
                ]
                out_cache_loc_chunks.append(
                    np.concatenate(loc_chunks).astype(np.int32, copy=False)
                    if loc_chunks
                    else np.empty(0, dtype=np.int32)
                )
        else:
            loc_chunks = [
                req_to_token[req_pool_idx, int(old_write) : int(required_write)]
                for req_pool_idx, old_write, required_write, valid in zip(
                    req_pool_indices,
                    old_verify_write_lens,
                    required_write_lens,
                    valid_slots,
                    strict=True,
                )
                if bool(valid) and int(required_write) > int(old_write)
            ]
            out_cache_loc_chunks = [
                (
                    np.concatenate(loc_chunks).astype(np.int32, copy=False)
                    if loc_chunks
                    else np.empty(0, dtype=np.int32)
                )
            ]

        with jax.profiler.TraceAnnotation("same_batch_chain_peek_device_reserved_suffix"):
            pass
        return (
            out_cache_loc_chunks,
            required_write_lens.astype(np.int32, copy=False),
            allocate_lens.astype(np.int32, copy=False),
        )

    @staticmethod
    def _pad_same_batch_spec_chain_out_cache_loc(
        model_worker_batch: ModelWorkerBatch,
        out_cache_loc_chunks,
    ) -> np.ndarray:
        chunks = [np.asarray(chunk, dtype=np.int32) for chunk in out_cache_loc_chunks]
        dp_size = int(getattr(model_worker_batch, "dp_size", 1) or 1)
        per_dp_bs = int(getattr(model_worker_batch, "per_dp_bs_size", 0) or 0)
        draft_token_num = int(getattr(model_worker_batch, "speculative_num_draft_tokens", 0) or 0)
        if dp_size <= 1 or per_dp_bs <= 0 or draft_token_num <= 0:
            return (
                np.concatenate(chunks).astype(np.int32, copy=False)
                if chunks
                else np.empty(0, dtype=np.int32)
            )

        target_per_rank = per_dp_bs * draft_token_num
        if len(chunks) < dp_size:
            chunks = chunks + [np.empty(0, dtype=np.int32)] * (dp_size - len(chunks))
        return np.concatenate(
            [
                np.pad(
                    chunk[:target_per_rank],
                    (0, max(0, target_per_rank - len(chunk))),
                    constant_values=-1,
                )
                for chunk in chunks[:dp_size]
            ]
        ).astype(np.int32, copy=False)

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        sampling_metadata: SamplingMetadata = None,
    ) -> tuple[None, jax.Array, int]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        if sampling_metadata is None:
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                0,
                self.mesh,
                self.worker.model_config.vocab_size,
            )

        forward_metadata = self.worker.model_runner.attn_backend.get_forward_metadata(
            model_worker_batch
        )

        # Prepare LoRA batch if LoRA is enabled
        if self.worker.server_args.enable_lora:
            self.worker.prepare_lora_batch(model_worker_batch)

        model_worker_batch.forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.worker.get_model_runner()
        )

        # Push a new batch to the queue (JAX handles synchronization automatically)
        self.input_queue.put(
            (
                "generation",
                model_worker_batch,
                self.future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            )
        )

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)

        future_next_token_ids = np.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=np.int32,
        )
        self.future_token_ids_ct = (self.future_token_ids_ct + bs) % self.future_token_ids_limit
        return None, future_next_token_ids, 0

    def forward_batch_speculative_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        *,
        use_split_verify_phase: bool,
    ):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult

        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        self.input_queue.put(
            (
                "spec_split" if use_split_verify_phase else "spec_full",
                model_worker_batch,
                None,
                None,
                None,
            )
        )
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=None,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            bid=model_worker_batch.bid,
            cache_miss_count=0,
        )
        result.spec_overlap_split = use_split_verify_phase
        return result

    def run_precompile(self):
        self.worker.run_precompile(self.future_token_ids_map)

    @property
    def page_size(self) -> int:
        return self.worker.page_size

    @property
    def sliding_window_size(self) -> int | None:
        return self.worker.sliding_window_size

    @property
    def is_hybrid(self) -> bool:
        return self.worker.is_hybrid

    def get_tokens_per_layer_info(self):
        return self.worker.get_tokens_per_layer_info()

    def __delete__(self):
        self.input_queue.put((None, None, None, None, None))
