import queue

import jax.numpy as jnp
import jax.sharding
import numpy as np
from jax import NamedSharding
from jax.sharding import PartitionSpec

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vae.vae_model_worker import VaeModelWorker
from sgl_jax.srt.utils.jax_utils import device_array


class VaeScheduler:
    def __init__(
        self,
        server_args: MultimodalServerArgs,
        communication_backend: CommunicationBackend = None,
        mesh: jax.sharding.Mesh = None,
        model_class=None,
        **kwargs,
    ):
        self._comm_backend = communication_backend
        self.mesh = mesh
        self.vae_worker = VaeModelWorker(
            model_class=model_class, mesh=mesh, server_args=server_args
        )
        self.server_args = server_args
        self.model_config = model_class.get_config_class()()

    def event_loop_normal(self):
        while True:
            reqs = self._comm_backend.recv_requests()
            if len(reqs) > 0:
                for req in reqs:
                    assert req.latents is not None
                    self.preprocess(req)
                    req.latents = device_array(
                        req.latents, sharding=NamedSharding(self.mesh, PartitionSpec())
                    )
                self.run_vae_batch(reqs)

    def preprocess(self, req):
        if hasattr(self.model_config, "scaling_factor"):
            req.latents = req.latents / self.model_config.scaling_factor
        if hasattr(self.model_config, "shift_factor"):
            req.latents += self.model_config.shift_factor

    def run_vae_batch(self, batch: list[Req]):
        for req in batch:
            output, _ = self.vae_worker.forward(req)
            req.output = jax.device_get(output)
            req.latents = None
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
