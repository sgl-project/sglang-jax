import logging

import jax
import jax.sharding

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq, ProfileReq
from sgl_jax.srt.managers.scheduler_profiler_mixing import SchedulerProfilerMixin
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.encoder.encoder_model_worker import (
    EncoderModelWorker,
)

logger = logging.getLogger(__name__)


class EncoderScheduler(SchedulerProfilerMixin):
    def __init__(
        self,
        server_args: MultimodalServerArgs,
        mesh: jax.sharding.Mesh,
        communication_backend: CommunicationBackend,
        model_class: str | list[str] = None,
        stage_sub_dir: str | None = None,
        **kwargs,
    ):
        """Initialize the EncoderScheduler.

        Args:
            server_args: Multimodal server config used to construct worker.
            mesh: JAX device mesh for sharding and placement.
            communication_backend: backend implementing
                `recv_requests()` and `send_pyobj()`.
            model_class: encoder model class passed to the worker.
        """

        self.communication_backend = communication_backend
        self.mesh = mesh
        self.encoder_worker = EncoderModelWorker(
            server_args,
            mesh=mesh,
            model_class=model_class,
            stage_sub_dir=stage_sub_dir,
            tokenizer=kwargs.get("tokenizers", "tokenizer"),
        )
        self.forward_ct = 0
        self.init_profier()

        if not server_args.disable_precompile:
            logger.info("[Encoder Scheduler] Begins to run encoder worker precompile.")
            self.encoder_worker.run_precompile()
            logger.info("[Encoder Scheduler] Completes encoder worker precompile.")
        # Track aborted request IDs to skip processing
        self.aborted_rids: set[str] = set()
        # Current request being processed (for abort checking during steps)
        self._current_rid: str | None = None

    def event_loop_normal(self):
        """Blocking event loop for processing incoming encoder requests.

        Continuously polls `communication_backend.recv_requests()` and for
        each received `Req` invokes `run_diffusion_step` to handle the
        inference. AbortReq messages are processed to track aborted request
        IDs, and any Req whose rid matches an aborted ID is skipped.
        """
        while True:
            reqs = self.communication_backend.recv_requests()
            if len(reqs) > 0:
                for req in reqs:
                    if isinstance(req, AbortReq):
                        # Record the aborted rid so we can skip it later
                        logger.info("EncoderScheduler received abort for rid=%s", req.rid)
                        self.aborted_rids.add(req.rid)
                    elif isinstance(req, ProfileReq):
                        result = self.profile(req)
                        self.communication_backend.send_pyobj(result)
                    elif isinstance(req, Req):
                        # Check if this request was aborted
                        if req.rid in self.aborted_rids:
                            logger.info("EncoderScheduler skipping aborted request rid=%s", req.rid)
                            self.aborted_rids.discard(req.rid)
                            continue
                        req = self.encoder_worker.forward(req)

                        self.forward_ct += 1
                        self._profile_batch_predicate(None)
                        self.communication_backend.send_pyobj(req)
                    else:
                        logger.warning(
                            "EncoderScheduler received unknown request type: %s", type(req)
                        )
            else:
                self.communication_backend.wait_for_new_requests(0.001)
