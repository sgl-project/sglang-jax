import logging
import signal
import threading
from queue import Queue

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


def publish_spec_decode_new_seq_lens(batch_output):
    new_seq_lens = batch_output.next_draft_input.new_seq_lens
    if new_seq_lens is not None and hasattr(new_seq_lens, "copy_to_host_async"):
        new_seq_lens.copy_to_host_async()
    return new_seq_lens


def can_use_spec_decode_overlap(enable_overlap, spec_algorithm, batch) -> bool:
    if not enable_overlap:
        return False
    if spec_algorithm is None or spec_algorithm.is_none():
        return False
    if not batch.forward_mode.is_decode():
        return False
    return not (batch.return_logprob or batch.return_output_logprob_only)


def resolve_spec_decode_token_ids(result, batch, draft_token_num: int):
    """Resolve per-request accepted token ids from a speculative verify result."""
    if hasattr(result.next_token_ids, "copy_to_host_async"):
        result.next_token_ids.copy_to_host_async()
    if hasattr(result.accept_lens, "copy_to_host_async"):
        result.accept_lens.copy_to_host_async()
    next_token_ids = np.asarray(result.next_token_ids)
    accept_lens = np.asarray(result.accept_lens).tolist()
    per_dp_bs = batch.per_dp_bs_size
    total_bs = per_dp_bs * batch.dp_size
    predict_tokens: list[list[int]] = [[] for _ in range(total_bs)]
    for dp_rank, info in enumerate(batch.reqs_info):
        base = dp_rank * per_dp_bs
        for j, _req in enumerate(info.reqs or []):
            i = base + j
            a = accept_lens[i]
            predict_tokens[i] = next_token_ids[
                i * draft_token_num : i * draft_token_num + a
            ].tolist()
    return predict_tokens, accept_lens


class SpecWorkerClient:
    """Queue-backed speculative worker client for overlap scheduling."""

    def __init__(self, worker):
        self.worker = worker
        self.mesh = worker.mesh
        self.max_running_requests = worker.target_worker.max_running_requests
        self.prefill_token_relay_ct = 0
        self.prefill_token_relay_limit = self.max_running_requests * 3
        self.prefill_token_relay_map = jnp.zeros((self.max_running_requests * 5,), dtype=jnp.int32)
        self.prefill_token_relay_map = jax.device_put(
            self.prefill_token_relay_map,
            NamedSharding(self.mesh, PartitionSpec(None)),
        )
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            daemon=bool(worker.server_args.enable_single_process),
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def run_spec_decode_precompile(self):
        self._precompile_prefill_token_relay()
        self.worker.run_spec_decode_precompile()

    def _precompile_prefill_token_relay(self):
        sharding = NamedSharding(self.mesh, PartitionSpec("data"))
        for num_tokens in self.worker.precompile_token_paddings:
            input_ids = jax.device_put(
                -jnp.ones((num_tokens,), dtype=jnp.int32),
                sharding,
            )
            jax.block_until_ready(
                resolve_future_token_ids(input_ids, self.prefill_token_relay_map, self.mesh)
            )

        for bs in self.worker.precompile_bs_paddings:
            next_token_ids = jax.device_put(
                jnp.ones((bs,), dtype=jnp.int32),
                sharding,
            )
            self.prefill_token_relay_map = set_future_token_ids(
                self.prefill_token_relay_map,
                0,
                next_token_ids,
                self.mesh,
            )
            jax.block_until_ready(self.prefill_token_relay_map)

        self.prefill_token_relay_map = jax.device_put(
            jnp.zeros((self.max_running_requests * 5,), dtype=jnp.int32),
            NamedSharding(self.mesh, PartitionSpec(None)),
        )

    def forward_thread_func(self):
        try:
            self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("SpecWorkerClient hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def forward_thread_func_(self):
        while True:
            model_worker_batch, prefill_token_relay_ct = self.input_queue.get()
            if model_worker_batch is None:
                break

            assert model_worker_batch.forward_mode.is_extend()
            if model_worker_batch.forward_batch is not None:
                model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                    model_worker_batch.forward_batch.input_ids,
                    self.prefill_token_relay_map,
                    self.mesh,
                )
            batch_output = self.worker.forward_batch_speculative_generation(
                model_worker_batch,
                launch_done=model_worker_batch.launch_done,
            )
            self.prefill_token_relay_map = set_future_token_ids(
                self.prefill_token_relay_map,
                prefill_token_relay_ct,
                batch_output.next_token_ids,
                self.mesh,
            )
            self.output_queue.put(batch_output)

    def forward_batch_speculative_generation(self, model_worker_batch):
        if not model_worker_batch.forward_mode.is_extend():
            return self.worker.forward_batch_speculative_generation(model_worker_batch)

        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
        from sgl_jax.srt.speculative.draft_extend_fused import (
            prepare_spec_prefill_forward_batch,
        )
        from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

        self.worker._prepare_overlap_sampling_info(model_worker_batch)
        if (
            self.worker._can_use_fused_spec_prefill(model_worker_batch)
            and getattr(model_worker_batch, "forward_batch", None) is None
        ):
            prepare_spec_prefill_forward_batch(self.worker, model_worker_batch)
        pending_prefill = PendingPrefillResult(
            self,
            return_logprob=model_worker_batch.return_logprob,
            return_output_logprob_only=model_worker_batch.return_output_logprob_only,
        )
        self.input_queue.put((model_worker_batch, self.prefill_token_relay_ct))

        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = np.arange(
            -(self.prefill_token_relay_ct + 1),
            -(self.prefill_token_relay_ct + 1 + bs),
            -1,
            dtype=np.int32,
        )
        self.prefill_token_relay_ct = (
            self.prefill_token_relay_ct + bs
        ) % self.prefill_token_relay_limit
        sel = model_worker_batch.logits_indices_selector
        next_draft_input = EagleDraftInput(
            allocate_lens=np.asarray(model_worker_batch.seq_lens)[sel],
            pending_draft_extend_result=pending_prefill,
        )
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=future_next_token_ids,
            next_draft_input=next_draft_input,
            bid=model_worker_batch.bid,
            cache_miss_count=0,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    def take_last_prefill_result(self):
        return self.output_queue.get()

    def materialize_prefill_result(
        self,
        batch_output,
        launch_done: threading.Event | None = None,
        *,
        return_hidden_states: bool = False,
        return_logprob: bool = False,
        return_output_logprob_only: bool = False,
    ):
        logits_output = batch_output.logits_output
        need_next_logprobs = return_logprob or return_output_logprob_only
        async_next_logprobs = (
            jax.copy_to_host_async(logits_output.next_token_logprobs)
            if need_next_logprobs
            and logits_output is not None
            and logits_output.next_token_logprobs is not None
            else None
        )
        async_input_logprobs = (
            jax.copy_to_host_async(logits_output.input_token_logprobs)
            if return_logprob
            and logits_output is not None
            and logits_output.input_token_logprobs is not None
            else None
        )
        async_hidden_states = (
            jax.copy_to_host_async(logits_output.hidden_states)
            if return_hidden_states
            and logits_output is not None
            and logits_output.hidden_states is not None
            else None
        )
        next_token_ids = batch_output.next_token_ids
        if hasattr(next_token_ids, "sharding"):
            from jax.experimental.multihost_utils import process_allgather

            next_token_ids = process_allgather(next_token_ids, tiled=True)
        async_next_tokens = jax.copy_to_host_async(next_token_ids)

        if batch_output.logits_output is not None:
            if async_next_logprobs is not None:
                logits_output.next_token_logprobs = np.asarray(async_next_logprobs).tolist()
            if async_input_logprobs is not None:
                logits_output.input_token_logprobs = np.asarray(async_input_logprobs).tolist()
            if async_hidden_states is not None:
                logits_output.hidden_states = np.asarray(async_hidden_states)

        batch_output.next_token_ids = np.asarray(async_next_tokens).tolist()
        if launch_done is not None:
            launch_done.wait()
        return batch_output

    def resolve_last_prefill_result(self, launch_done: threading.Event | None = None):
        return self.materialize_prefill_result(
            self.take_last_prefill_result(),
            launch_done,
        )


class PendingPrefillResult:
    def __init__(
        self,
        client,
        *,
        return_hidden_states: bool = False,
        return_logprob: bool = False,
        return_output_logprob_only: bool = False,
    ):
        self.client = client
        self.batch_output = None
        self.return_hidden_states = return_hidden_states
        self.return_logprob = return_logprob
        self.return_output_logprob_only = return_output_logprob_only

    def resolve(self):
        if self.batch_output is None:
            self.batch_output = self.client.take_last_prefill_result()
        return self.batch_output

    def materialize(self, launch_done: threading.Event | None = None):
        self.batch_output = self.client.materialize_prefill_result(
            self.resolve(),
            launch_done,
            return_hidden_states=self.return_hidden_states,
            return_logprob=self.return_logprob,
            return_output_logprob_only=self.return_output_logprob_only,
        )
        return self.batch_output
