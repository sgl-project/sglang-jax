from queue import Queue

import jax.sharding

from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_worker import (
    DiffusionModelWorker,
)


class DiffusionScheduler:
    def __init__(self, server_args, in_queue: Queue, out_queue: Queue, mesh: jax.sharding.Mesh):
        self._in_queue = in_queue
        self._out_queue = out_queue
        self.mesh = mesh
        self.diffusion_worker = DiffusionModelWorker(server_args.model_config, mesh=mesh)

    def event_loop(self):

        while True:
            req = self._in_queue.get()
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
