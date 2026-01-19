import jax.sharding

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_worker import (
    DiffusionModelWorker,
)


class DiffusionScheduler:
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
            server_args, mesh=mesh, model_class=model_class
        )

    def event_loop_normal(self):
        """Blocking event loop for processing incoming diffusion requests.

        Continuously polls `communication_backend.recv_requests()` and for
        each received `Req` invokes `run_diffusion_step` to handle the
        inference. This loop is intended for simple, synchronous usage
        contexts.
        """

        while True:
            req = self.communication_backend.recv_requests()
            if req is not None and len(req) > 0:
                req = req[0]
                assert isinstance(req, Req)
                self.run_diffusion_step(req)

    def run_diffusion_step(self, req: Req):
        """Execute a single diffusion inference step for `req`.

        Typical flow:
        1. Prepare/pad the request into a worker-friendly batch via
            `prepare_diffusion_batch`.
        2. Call `self.diffusion_worker.forward(batch, self.mesh)` to run the
            model.
        3. Send the resulting batch (or a wrapped `Req`) back through the
            communication backend.
        """

        # padding request data for JIT
        # schedule_batch -> worker_batch -> forward_batch
        batch = self.prepare_diffusion_batch(req)
        self.diffusion_worker.forward(batch, self.mesh)
        # mock_req = Req(rid=batch.rid)
        self.communication_backend.send_pyobj(batch)

    def prepare_diffusion_batch(self, req: Req) -> Req:
        """Prepare and (optionally) pad a `Req` for JIT-friendly execution.

        This hook may perform batching, padding, dtype conversion, or other
        transformations required by the JIT-compiled diffusion worker. By
        default it returns the request unchanged; override or modify as
        needed to support model input requirements.
        """

        return req
