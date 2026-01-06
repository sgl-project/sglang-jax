import jax.sharding

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
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
            req = self.communication_backend.recv_requests()
            if req is not None and len(req) > 0:
                req = req[0]
                assert isinstance(req, Req)
                self.run_diffusion_step(req)

    def run_diffusion_step(self, req: Req):
        # padding request data for JIT
        # schedule_batch -> worker_batch -> forward_batch
        batch = self.prepare_diffusion_batch(req)
        self.diffusion_worker.forward(batch, self.mesh)
        mock_req = Req(rid=batch.rid)
        self.communication_backend.send_pyobj(mock_req)

    def prepare_diffusion_batch(self, req: Req) -> Req:

        return req
