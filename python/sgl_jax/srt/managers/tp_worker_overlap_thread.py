"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
import time
from queue import Queue
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import psutil

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import (
    JAX_PRECOMPILE_DEFAULT_DECODE_BS_PADDINGS,
    JAX_PRECOMPILE_DEFAULT_PREFILL_TOKEN_PADDINGS,
)
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@jax.jit
def resolve_future_token_ids(input_ids, future_token_ids_map):
    return jnp.where(
        input_ids < 0,
        future_token_ids_map[jnp.clip(-input_ids, a_min=0)],
        input_ids,
    )


class ModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
    ):
        # Load the model
        self.worker = ModelWorker(server_args, mesh=mesh)
        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = jnp.zeros(
            (self.max_running_requests * 5,), dtype=jnp.int32
        )

        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        # JAX handles device execution automatically, no need for explicit streams
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()

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

    def get_prefill_padded_size(self):
        return self.worker.get_prefill_padded_size()

    def get_precompile_paddings(self):
        return self.worker.get_precompile_paddings()

    def forward_thread_func(self):
        try:
            self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"ModelWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGQUIT)

    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * 2

        while True:
            model_worker_batch, future_token_ids_ct = self.input_queue.get()
            if not model_worker_batch:
                break

            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            batch_lists[batch_pt % 2] = model_worker_batch
            batch_pt += 1

            # Resolve future tokens in the input
            input_ids = model_worker_batch.input_ids
            input_ids = resolve_future_token_ids(input_ids, self.future_token_ids_map)
            model_worker_batch.input_ids = input_ids

            # Run forward
            logits_output, next_token_ids, cache_miss_count = (
                self.worker.forward_batch_generation(
                    model_worker_batch, model_worker_batch.launch_done
                )
            )

            # Update the future token ids map
            # Only count non-padded sequences (seq_lens > 0)
            effective_bs = model_worker_batch.real_bs
            self.future_token_ids_map = self.future_token_ids_map.at[
                future_token_ids_ct + 1 : future_token_ids_ct + effective_bs + 1
            ].set(next_token_ids[:effective_bs])

            # Copy results to the CPU
            if model_worker_batch.return_logprob:
                logits_output.next_token_logprobs = jax.device_get(
                    logits_output.next_token_logprobs
                )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = jax.device_get(
                        logits_output.input_token_logprobs
                    )
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = jax.device_get(
                    logits_output.hidden_states
                )
            next_token_ids = jax.device_get(next_token_ids).tolist()

            self.output_queue.put(
                (None, logits_output, next_token_ids, cache_miss_count)
            )

    def resolve_last_batch_result(self, launch_done: Optional[threading.Event] = None):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.
        """
        _, logits_output, next_token_ids, cache_miss_count = self.output_queue.get()

        if launch_done is not None:
            launch_done.wait()
        # JAX handles synchronization automatically, no explicit sync needed

        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = (
                logits_output.next_token_logprobs.tolist()
            )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
        # next_token_ids = next_token_ids.tolist()
        return logits_output, next_token_ids, cache_miss_count

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> Tuple[None, jax.Array, int]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
        )

        # Push a new batch to the queue (JAX handles synchronization automatically)
        self.input_queue.put((model_worker_batch, self.future_token_ids_ct))

        # Allocate output future objects
        # Only count non-padded sequences (seq_lens > 0)
        effective_bs = model_worker_batch.real_bs
        future_next_token_ids = np.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + effective_bs),
            -1,
            dtype=np.int32,
        )
        self.future_token_ids_ct = (
            self.future_token_ids_ct + effective_bs
        ) % self.future_token_ids_limit
        return None, future_next_token_ids, 0

    def run_precompile(self):
        start_time = time.perf_counter()
        logger.info(
            f"[ModelWorkerClient] begins to run resolve_future_token_ids precompile."
        )
        (
            _,
            precompile_prefill_token_paddings,
            precompile_decode_bs_paddings,
            _,
            _,
        ) = self.get_precompile_paddings()
        token_paddings = (
            precompile_prefill_token_paddings + precompile_decode_bs_paddings
        )
        for token_padding in token_paddings:
            input_ids = jnp.arange(0, token_padding, dtype=jnp.int32)
            resolve_future_token_ids(input_ids, self.future_token_ids_map)
        end_time = time.perf_counter()
        logger.info(
            f"[ModelWorkerClient] completes resolve_future_token_ids precompile. Time cost: {end_time - start_time} seconds"
        )

        self.worker.run_precompile()

    def __delete__(self):
        self.input_queue.put((None, None))
