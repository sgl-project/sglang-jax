"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
from queue import Queue
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


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
        self.mesh = mesh
        sharding = NamedSharding(mesh, PartitionSpec(None))
        self.future_token_ids_map = jax.device_put(self.future_token_ids_map, sharding)
        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        # JAX handles device execution automatically, no need for explicit streams
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()

    def get_model_runner(self):
        return self.worker.get_model_runner()

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
            logger.error(f"ModelWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGQUIT)

    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * 2

        while True:
            (
                model_worker_batch,
                future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            ) = self.input_queue.get()
            if not model_worker_batch:
                break

            # Keep a reference of model_worker_batch by storing it into a list.
            batch_lists[batch_pt % 2] = model_worker_batch
            batch_pt += 1

            # Resolve future tokens in the input
            input_ids = model_worker_batch.forward_batch.input_ids
            model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                input_ids, self.future_token_ids_map
            )

            # Run forward
            logits_output, next_token_ids, cache_miss_count = (
                self.worker.forward_batch_generation(
                    model_worker_batch,
                    model_worker_batch.launch_done,
                    sampling_metadata=sampling_metadata,
                    forward_metadata=forward_metadata,
                )
            )

            # Update the future token ids map
            self.future_token_ids_map = set_future_token_ids(
                self.future_token_ids_map, future_token_ids_ct, next_token_ids
            )

            self.output_queue.put(
                (None, logits_output, next_token_ids, cache_miss_count)
            )

    def resolve_last_batch_result(self, launch_done: Optional[threading.Event] = None):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.
        """
        _, logits_output, next_token_ids, cache_miss_count = self.output_queue.get()
        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = jax.device_get(
                logits_output.next_token_logprobs
            ).tolist()
        if logits_output.input_token_logprobs is not None:
            logits_output.input_token_logprobs = jax.device_get(
                logits_output.input_token_logprobs
            ).tolist()
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = jax.device_get(logits_output.hidden_states)
        next_token_ids = jax.device_get(next_token_ids).tolist()

        if launch_done is not None:
            launch_done.wait()

        return logits_output, next_token_ids, cache_miss_count

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        sampling_metadata: SamplingMetadata,
    ) -> Tuple[None, jax.Array, int]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
        )

        forward_metadata = self.worker.model_runner.attn_backend.get_forward_metadata(
            model_worker_batch
        )

        # Push a new batch to the queue (JAX handles synchronization automatically)
        self.input_queue.put(
            (
                model_worker_batch,
                self.future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            )
        )

        # Allocate output future objects
        # Only count non-padded sequences (seq_lens > 0)
        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = np.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=np.int32,
        )
        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit
        return None, future_next_token_ids, 0

    def run_precompile(self):
        self.worker.run_precompile(self.future_token_ids_map)

    def __delete__(self):
        self.input_queue.put((None, None, None, None))
