import logging

import jax.sharding

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq, ProfileReq
from sgl_jax.srt.managers.scheduler_profiler_mixing import SchedulerProfilerMixin
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_worker import (
    DiffusionModelWorker,
)

logger = logging.getLogger(__name__)


class DiffusionScheduler(SchedulerProfilerMixin):
    """Scheduler responsible for diffusion-model inference steps.

    The DiffusionScheduler receives requests via a `CommunicationBackend`,
    prepares them for the JAX-based `DiffusionModelWorker`, and forwards
    results back through the communication backend. It expects a `mesh` to
    be provided for sharding/placement and delegates the actual model
    execution to `DiffusionModelWorker`.

    Responsibilities:
    - Poll `communication_backend` for incoming `Req` objects.
    - Prepare and batch requests suitable for JIT-compiled model execution.
    - Invoke the diffusion worker and send outputs back to the caller.
    """

    def __init__(
        self,
        server_args: MultimodalServerArgs,
        mesh: jax.sharding.Mesh,
        communication_backend: CommunicationBackend,
        model_class,
        stage_sub_dir: str | None = None,
        precompile_params: dict | None = None,
    ):
        """Initialize the DiffusionScheduler.

        Args:
            server_args: Multimodal server config used to construct worker.
            mesh: JAX device mesh for sharding and placement.
            communication_backend: backend implementing
                `recv_requests()` and `send_pyobj()`.
            model_class: diffusion model class passed to the worker.
        """

        self.communication_backend = communication_backend
        self.mesh = mesh
        self.diffusion_worker = DiffusionModelWorker(
            server_args, mesh=mesh, model_class=model_class, stage_sub_dir=stage_sub_dir
        )
        self.forward_ct = 0
        self.init_profier()
        # Track aborted request IDs to skip processing
        self.aborted_rids: set[str] = set()
        # Current request being processed (for abort checking during steps)
        self._current_rid: str | None = None

    def event_loop_normal(self):
        """Blocking event loop for processing incoming diffusion requests.

        Continuously polls `communication_backend.recv_requests()` and for
        each received `Req` invokes `run_diffusion_step` to handle the
        inference. AbortReq messages are processed to track aborted request
        IDs, and any Req whose rid matches an aborted ID is skipped.
        """
        while True:
            reqs = self.communication_backend.recv_requests()
            if reqs is not None and len(reqs) > 0:
                for req in reqs:
                    if isinstance(req, AbortReq):
                        # Record the aborted rid so we can skip it later
                        logger.info("DiffusionScheduler received abort for rid=%s", req.rid)
                        self.aborted_rids.add(req.rid)
                    elif isinstance(req, ProfileReq):
                        result = self.profile(req)
                        self.communication_backend.send_pyobj(result)
                    elif isinstance(req, Req):
                        # Check if this request was aborted
                        if req.rid in self.aborted_rids:
                            logger.info(
                                "DiffusionScheduler skipping aborted request rid=%s", req.rid
                            )
                            self.aborted_rids.discard(req.rid)
                            continue
                        self.run_diffusion_step(req)
                    else:
                        logger.warning(
                            "DiffusionScheduler received unknown request type: %s", type(req)
                        )
            else:
                self.communication_backend.wait_for_new_requests(0.001)

    def check_abort(self) -> bool:
        """Check if current request should be aborted.

        This method is called between diffusion steps to check for abort
        requests. It drains the input queue for any AbortReq messages and
        returns True if the current request should be aborted.

        Returns:
            True if the current request should be aborted, False otherwise.
        """
        # Drain any pending abort/profile requests from the queue
        while True:
            try:
                msg = self.communication_backend._in_queue.get_nowait()
                if isinstance(msg, AbortReq):
                    logger.info("DiffusionScheduler received abort during step for rid=%s", msg.rid)
                    self.aborted_rids.add(msg.rid)
                elif isinstance(msg, ProfileReq):
                    result = self.profile(msg)
                    self.communication_backend.send_pyobj(result)
                else:
                    self.communication_backend._in_queue.put_nowait(msg)
                    break
            except Exception:
                break

        # Check if current request is aborted
        if self._current_rid and self._current_rid in self.aborted_rids:
            logger.info("DiffusionScheduler aborting current request rid=%s", self._current_rid)
            return True
        return False

    def _on_step(self):
        """Called after each denoising step to increment the step counter."""
        self.forward_ct += 1
        self._profile_batch_predicate(None)

    def run_diffusion_step(self, req: Req):
        """Execute a single diffusion inference step for `req`.

        Typical flow:
        1. Prepare/pad the request into a worker-friendly batch via
            `prepare_diffusion_batch`.
        2. Call `self.diffusion_worker.forward(batch, self.mesh)` to run the
            model.
        3. Send the resulting batch (or a wrapped `Req`) back through the
            communication backend.

        The forward call is passed an abort_checker callback that checks for
        abort requests between diffusion steps. If an abort is detected, the
        forward returns early and the request is not sent to the next stage.
        """
        self._current_rid = req.rid

        # padding request data for JIT
        # schedule_batch -> worker_batch -> forward_batch
        batch = self.prepare_diffusion_batch(req)
        aborted = self.diffusion_worker.forward(
            batch, self.mesh, abort_checker=self.check_abort, step_callback=self._on_step
        )

        self._current_rid = None

        if aborted:
            # Request was aborted during diffusion, clean up
            self.aborted_rids.discard(req.rid)
            logger.info("DiffusionScheduler discarding aborted request rid=%s", req.rid)
            return

        self.communication_backend.send_pyobj(batch)

    def prepare_diffusion_batch(self, req: Req) -> Req:
        """Prepare and (optionally) pad a `Req` for JIT-friendly execution.

        This hook may perform batching, padding, dtype conversion, or other
        transformations required by the JIT-compiled diffusion worker. By
        default it returns the request unchanged; override or modify as
        needed to support model input requirements.
        """

        return req
