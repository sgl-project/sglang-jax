from queue import Queue

import jax.sharding

from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_runner import (
    DiffusionModelRunner,
)


class DiffusionScheduler:
    def __init__(self, server_args, in_queue: Queue, out_queue: Queue, mesh: jax.sharding.Mesh):
        self._in_queue = in_queue
        self._out_queue = out_queue
        self.mesh = mesh
        self.worker = DiffusionModelRunner()

    def event_loop(self):
        import time

        while True:
            obj = self._in_queue.get()
            print("diffusion scheduler recv", obj)
            time.sleep(10)
