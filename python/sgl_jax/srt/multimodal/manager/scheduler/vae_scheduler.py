import queue
from multiprocessing import Queue

import jax.numpy as jnp
import jax.sharding
import numpy as np

from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vae.vae_model_worker import VaeModelWorker


class VaeScheduler:
    def __init__(
        self,
        server_args=None,
        in_queue: Queue = None,
        out_queue: Queue = None,
        mesh: jax.sharding.Mesh = None,
        **kwargs,
    ):
        self._in_queue = in_queue
        self._out_queue = out_queue
        self.mesh = mesh
        self.vae_worker = VaeModelWorker(None, mesh=mesh)

    def event_loop(self):
        while True:
            req = self._in_queue.get()
            if req.latents is None:
                req.latents = jnp.array(np.arange(1 * 5 * 3 * 4 * 16), dtype=jnp.float32).reshape(
                    1, 5, 3, 4, 16
                )
            assert req.latents is not None
            if req is not None:
                self.run_vae_batch(req)

    def run_vae_batch(self, req):
        output = self.vae_worker.forward(req)
        req.output = output
        self._out_queue.put(req)


if __name__ == "__main__":
    in_queue = queue.Queue()
    out_queue = queue.Queue()
    scheduler = VaeScheduler(server_args=None, in_queue=in_queue, out_queue=out_queue, mesh=None)
    x = jnp.array(np.arange(1 * 5 * 3 * 4 * 16), dtype=jnp.float32).reshape(1, 5, 3, 4, 16)
    req = Req(rid="111", latents=x)
    scheduler.run_vae_batch(req)
    y = out_queue.get()
    print(y)
