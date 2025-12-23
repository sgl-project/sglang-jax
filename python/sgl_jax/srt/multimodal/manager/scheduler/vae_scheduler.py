from multiprocessing import Queue

import jax.sharding

from sgl_jax.srt.multimodal.model_executor.vae.vae_model_worker import VaeModelWorker


class VaeScheduler:
    def __init__(self, server_args, in_queue: Queue, out_queue: Queue, mesh: jax.sharding.Mesh):
        self._in_queue = in_queue
        self._out_queue = out_queue
        self.mesh = mesh
        self.vae_worker = VaeModelWorker(server_args.model_config, mesh=mesh)

    def event_loop(self):

        while True:
            req = self._in_queue.get()
            if req is not None:
                self.run_vae_batch(req)

    def run_vae_batch(self, req):

        self.vae_worker.forward(req, self.mesh)
