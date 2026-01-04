import jax.sharding

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_worker import (
    DiffusionModelWorker,
)


class DiffusionScheduler:
    def __init__(
        self,
        server_args: MultimodalServerArgs,
        mesh: jax.sharding.Mesh,
        communication_backend: CommunicationBackend = None,
    ):
        self.communication_backend = communication_backend
        self.mesh = mesh
        self.diffusion_worker = DiffusionModelWorker(server_args, mesh=mesh)

    def event_loop(self):

        while True:
            req = self.communication_backend.recv_request()
            if req is not None:
                self.run_diffusion_step(req)

    def run_diffusion_step(self, req):
        # padding request data for JIT
        # schedule_batch -> worker_batch -> forward_batch
        batch = self.prepare_diffusion_batch(req)
        self.diffusion_worker.forward(batch, self.mesh)

    def prepare_diffusion_batch(self, req):
        # prepare batch for diffusion worker
        return req
