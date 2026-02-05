import logging

import jax
import jax.sharding
from jax import NamedSharding
from jax.sharding import PartitionSpec

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vae.vae_model_worker import VaeModelWorker
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


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
        stage_sub_dir: str | None = None,
        precompile_params: dict | None = None,
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
            model_class=model_class,
            mesh=self.mesh,
            server_args=server_args,
            stage_sub_dir=stage_sub_dir,
        )
        self.server_args = server_args
        self.model_config = model_class.get_config_class()()
        # Track aborted request IDs to skip processing
        self.aborted_rids: set[str] = set()

    def event_loop_normal(self):
        """Main blocking loop used in non-async environments.

        Repeatedly polls the `communication_backend` for requests, applies
        `preprocess`, shards `req.latents` onto `self.mesh` with a
        `NamedSharding(PartitionSpec())`, and then processes the batch via
        `run_vae_batch`. AbortReq messages are processed to track aborted
        request IDs, and any Req whose rid matches an aborted ID is skipped.
        """

        while True:
            reqs = self._comm_backend.recv_requests()
            if len(reqs) > 0:
                # Process abort requests first
                valid_reqs = []
                for req in reqs:
                    if isinstance(req, AbortReq):
                        logger.info("VaeScheduler received abort for rid=%s", req.rid)
                        self.aborted_rids.add(req.rid)
                    elif isinstance(req, Req):
                        # Check if this request was aborted
                        if req.rid in self.aborted_rids:
                            logger.info("VaeScheduler skipping aborted request rid=%s", req.rid)
                            self.aborted_rids.discard(req.rid)
                            continue
                        assert req.latents is not None
                        self.preprocess(req)
                        req.latents = device_array(
                            req.latents, sharding=NamedSharding(self.mesh, PartitionSpec())
                        )
                        valid_reqs.append(req)
                    else:
                        logger.warning("VaeScheduler received unknown request type: %s", type(req))

                if valid_reqs:
                    self.run_vae_batch(valid_reqs)
            else:
                self._comm_backend.wait_for_new_requests(0.001)

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
        req.latents = jax.device_get(req.latents)

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
