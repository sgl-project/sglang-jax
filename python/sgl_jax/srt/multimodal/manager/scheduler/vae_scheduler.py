import logging

import jax
import jax.sharding
import numpy as np

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq, ProfileReq
from sgl_jax.srt.managers.scheduler_profiler_mixing import SchedulerProfilerMixin
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vae.vae_model_worker import VaeModelWorker

logger = logging.getLogger(__name__)


class VaeScheduler(SchedulerProfilerMixin):
    """Scheduler for VAE model inference within the multimodal pipeline.

    Responsibilities:
    - Receive batched requests via a communication backend and prepare inputs.
    - Preprocess latents according to model config (scaling/shift).
    - Group shape-compatible requests into a larger decode batch so VAE decode
      can shard the batch dimension with SPMD.
    - Run the VAE forward pass and return/send outputs via the communication
      backend in the original request order.

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
        self.forward_ct = 0
        self.init_profier()
        self.model_config = model_class.get_config_class()()
        # Track aborted request IDs to skip processing
        self.aborted_rids: set[str] = set()
        if not server_args.disable_precompile:
            logger.info("[VAE Scheduler] Begins to run vae worker precompile.")
            self.vae_worker.run_precompile()
            logger.info("[VAE Scheduler] Completes vae worker precompile.")

    def event_loop_normal(self):
        """Main blocking loop used in non-async environments.

        Repeatedly polls the `communication_backend` for requests, applies
        `preprocess`, and then processes the batch via `run_vae_batch`.
        AbortReq messages are processed to track aborted request IDs, and any
        Req whose rid matches an aborted ID is skipped.
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
                    elif isinstance(req, ProfileReq):
                        result = self.profile(req)
                        self._comm_backend.send_pyobj(result)
                    elif isinstance(req, Req):
                        # Check if this request was aborted
                        if req.rid in self.aborted_rids:
                            logger.info("VaeScheduler skipping aborted request rid=%s", req.rid)
                            self.aborted_rids.discard(req.rid)
                            continue
                        assert req.latents is not None
                        self.preprocess(req)
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
        latents_t_padding = 0
        if self.server_args.vae_decode_precompile_frame_paddings is not None and hasattr(
            self.model_config, "scale_factor_temporal"
        ):
            for n_frame in self.server_args.vae_decode_precompile_frame_paddings:
                latents_t = (n_frame - 1) // self.model_config.scale_factor_temporal + 1
                if latents_t >= req.latents.shape[1]:
                    latents_t_padding = latents_t - req.latents.shape[1]
                    break
        req.latents = np.pad(
            req.latents,
            pad_width=((0, 0), (0, latents_t_padding), (0, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    def _get_req_group_key(self, req: Req) -> tuple:
        return (tuple(req.latents.shape[1:]), req.latents.dtype)

    def _get_output_num_frames(self, req: Req) -> int:
        if isinstance(req.num_frames, (list, tuple, np.ndarray)):
            return int(max(req.num_frames))
        return int(req.num_frames)

    def run_vae_batch(self, batch: list[Req]):
        """Run the VAE forward pass for a batch of requests.

        Requests with the same latent shape are concatenated on batch dim so
        the worker can use `pjit` batch sharding. Outputs are split back into
        their original requests and emitted in input order.
        """

        grouped_reqs: dict[tuple, list[tuple[int, Req]]] = {}
        completed_reqs: list[Req | None] = [None] * len(batch)

        for idx, req in enumerate(batch):
            grouped_reqs.setdefault(self._get_req_group_key(req), []).append((idx, req))

        for req_group in grouped_reqs.values():
            group_latents = np.concatenate([req.latents for _, req in req_group], axis=0)
            output, cache_miss = self.vae_worker.forward_latents(group_latents)
            logger.info(
                "VAE forward pass cache miss: %s (group_size=%d, batch=%d)",
                cache_miss,
                len(req_group),
                group_latents.shape[0],
            )
            output = np.asarray(jax.device_get(output))

            batch_start = 0
            for idx, req in req_group:
                req_batch_size = req.latents.shape[0]
                req_output = output[batch_start : batch_start + req_batch_size]
                batch_start += req_batch_size
                req.output = req_output[:, : self._get_output_num_frames(req), :, :, :]
                req.latents = None
                self.forward_ct += 1
                self._profile_batch_predicate(None)
                completed_reqs[idx] = req

        for req in completed_reqs:
            if req is None:
                continue
            req.latents = None
            self._comm_backend.send_pyobj(req)
