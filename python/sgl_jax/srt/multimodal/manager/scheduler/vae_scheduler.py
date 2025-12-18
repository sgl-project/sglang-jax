import jax.sharding
from jax import NamedSharding
from jax.sharding import PartitionSpec

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vae.vae_model_worker import VaeModelWorker
from sgl_jax.srt.utils.jax_utils import device_array


class VaeScheduler:
    """Scheduler for VAE model inference within the multimodal pipeline.

    Responsibilities:
    - Receive batched requests via a communication backend and prepare inputs.
    - Preprocess latents according to model config (scaling/shift).
    - Move input arrays onto JAX devices using the provided `mesh` and a
      `NamedSharding`/`PartitionSpec` before forwarding to the VAE worker.
    - Run the VAE forward pass and return/send outputs via the communication
      backend.

    The scheduler assumes a `VaeModelWorker` handles model execution and that
    `communication_backend` provides `recv_requests()` and `send_pyobj()`.
    """

    def __init__(
        self,
        server_args: MultimodalServerArgs,
        communication_backend: CommunicationBackend,
        mesh: jax.sharding.Mesh,
        model_class,
        **kwargs,
    ):
        """Initialize the VaeScheduler.

        Args:
            server_args: Multimodal server configuration.
            communication_backend: Backend used to receive/send requests.
            mesh: JAX device mesh used for sharding inputs/outputs.
            model_class: The VAE model class; used to build `VaeModelWorker` and
                to obtain model-specific configuration values.
        """

        self._comm_backend = communication_backend
        self.mesh = mesh
        self.vae_worker = VaeModelWorker(
            model_class=model_class, mesh=mesh, server_args=server_args
        )
        self.server_args = server_args
        self.model_config = model_class.get_config_class()()

    def event_loop_normal(self):
        """Main blocking loop used in non-async environments.

        Repeatedly polls the `communication_backend` for requests, applies
        `preprocess`, shards `req.latents` onto `self.mesh` with a
        `NamedSharding(PartitionSpec())`, and then processes the batch via
        `run_vae_batch`.
        """

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
        """Apply model-specific preprocessing to a single `Req`.

        Common operations: divide by `scaling_factor` and add `shift_factor`
        if those attributes exist on the model config. This prepares latents
        to match the expected value range for the VAE decoder.
        """

        if hasattr(self.model_config, "scaling_factor"):
            req.latents = req.latents / self.model_config.scaling_factor
        if hasattr(self.model_config, "shift_factor"):
            req.latents += self.model_config.shift_factor

    def run_vae_batch(self, batch: list[Req]):
        """Run the VAE forward pass for a batch of requests.

        For each `Req` in `batch`, invokes the `VaeModelWorker.forward`, moves
        the result back to host memory with `jax.device_get`, clears the
        latent to free memory, and sends the completed request through the
        communication backend.
        """

        for req in batch:
            output, _ = self.vae_worker.forward(req)
            req.output = jax.device_get(output)
            req.latents = None
            self._comm_backend.send_pyobj(req)
