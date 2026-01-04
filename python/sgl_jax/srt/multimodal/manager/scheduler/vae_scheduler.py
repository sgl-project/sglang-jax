import queue

import jax.numpy as jnp
import jax.sharding
import numpy as np

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vae.vae_model_worker import VaeModelWorker


class VaeScheduler:
    def __init__(
        self,
        server_args=None,
        communication_backend: CommunicationBackend = None,
        mesh: jax.sharding.Mesh = None,
        **kwargs,
    ):
        self._comm_backend = communication_backend
        self.mesh = mesh
        self.vae_worker = VaeModelWorker(None, mesh=mesh)

    def event_loop(self):
        while True:
            reqs = self._comm_backend.recv_requests()
            if len(reqs) > 0:
                for req in reqs:
                    if req.latents is None:
                        req.latents = jnp.array(np.arange(1 * 5 * 3 * 4 * 16), dtype=jnp.float32).reshape(
                            1, 5, 3, 4, 16
                        )
                    assert req.latents is not None
                self.run_vae_batch(reqs)

    def run_vae_batch(self, batch:list[Req]):
        for req in batch:
            output = self.vae_worker.forward(req)
            req.output = output
            self._comm_backend.send_pyobj(req)


if __name__ == "__main__":
    in_queue = queue.Queue()
    out_queue = queue.Queue()
    scheduler = VaeScheduler(server_args=None, in_queue=in_queue, out_queue=out_queue, mesh=None)
    x = jnp.array(np.arange(1 * 5 * 3 * 4 * 16), dtype=jnp.float32).reshape(1, 5, 3, 4, 16)
    req = Req(rid="111", latents=x)
    scheduler.run_vae_batch([req])
    y = out_queue.get()
    print(y)
