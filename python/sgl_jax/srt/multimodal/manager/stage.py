import logging
import signal
from queue import Queue
from typing import Any

import jax
import psutil

from sgl_jax.srt.managers.communication import QueueBackend
from sgl_jax.srt.managers.scheduler import Scheduler as AutoRegressiveScheduler
from sgl_jax.srt.models.qwen2 import Qwen2ForCausalLM
from sgl_jax.srt.models.umt5 import UMT5EncoderModel
from sgl_jax.srt.multimodal.manager.device_manager import DeviceManager
from sgl_jax.srt.multimodal.manager.scheduler.diffusion_scheduler import (
    DiffusionScheduler,
)
from sgl_jax.srt.multimodal.manager.scheduler.embed_scheduler import EmbedScheduler
from sgl_jax.srt.multimodal.manager.scheduler.vae_scheduler import VaeScheduler
from sgl_jax.srt.multimodal.manager.scheduler.vit_scheduler import VitScheduler
from sgl_jax.srt.multimodal.models.qwen2_5VL.qwen2_5_vit import Qwen2_5_VL_VisionModel
from sgl_jax.srt.multimodal.models.qwen2_5VL.qwen2_5_vl_generation import (
    Qwen2_5_VL_Generation,
)
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.qwen3_embedding import (
    Qwen3OmniEmbedding,
)
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.qwen3_omni_thinker import (
    Qwen3OmniMoeTinkerTextForConditionalGeneration,
)
from sgl_jax.srt.multimodal.models.wan.diffusion.wan_dit import (
    WanDualTransformer3DModel,
    WanTransformer3DModel,
)
from sgl_jax.srt.multimodal.models.wan.vaes.wanvae import AutoencoderKLWan
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class Stage:
    """Represents a single pipeline stage running a scheduler.

    A `Stage` owns a device `mesh` (created from `stage_config` and
    allocated devices via `DeviceManager`), an inbound queue and an outbound
    queue used to communicate with the global scheduler, and a scheduler
    instance that performs the stage-specific work (e.g. diffusion, VAE,
    autoregressive).

    Lifecycle:
    - Constructed in the parent process; device indices are allocated.
    - `set_in_queue` / `set_out_queue` attach communication queues.
    - `run_stage` is executed in a dedicated thread: it initializes the
      stage scheduler, signals readiness on the out queue, and runs the
      scheduler's event loop.
    """

    def __init__(
        self, stage_config: Any, *, device_manager: DeviceManager, server_args: ServerArgs
    ):
        """Initialize stage resources and create the device mesh.

        Args:
            stage_config: Configuration for this stage (includes `runtime` and
                scheduler information).
            device_manager: `DeviceManager` used to reserve device indices.
            server_args: Global server arguments passed to schedulers.
        """
        self._in_queue = None
        self._out_queue = None
        runtime = stage_config.runtime
        device_kind = getattr(runtime, "device_kind", "tpu")
        num_devices = runtime.num_tpus
        if device_kind == "cpu":
            cpu_devices = jax.devices("cpu")
            if num_devices > len(cpu_devices):
                raise RuntimeError(
                    f"Requested {num_devices} CPU devices, but only {len(cpu_devices)} available."
                )
            self.mesh = create_device_mesh(
                ici_parallelism=[-1, num_devices],
                dcn_parallelism=[1, 1],
                devices=cpu_devices[:num_devices],
            )
        else:
            # this parallelism setting is accord to stage config
            self.mesh = create_device_mesh(
                ici_parallelism=[-1, num_devices],
                dcn_parallelism=[1, 1],
                device_indexes=device_manager.allocate(num_devices),
            )
        self.stage_config = stage_config
        self.server_args = server_args
        self.stage_id = stage_config.stage_id
        # mesh

    def set_in_queue(self, in_queue: Queue):
        """Attach the inbound queue used to receive requests for this stage."""

        self._in_queue = in_queue

    def set_out_queue(self, out_queue: Queue):
        """Attach the outbound queue used to publish status/results."""

        self._out_queue = out_queue

    def run_stage(self):
        """Thread entrypoint: initialize scheduler and run its event loop.

        The method creates the scheduler instance specified in
        `stage_config`, wraps the stage's in/out queues with a `QueueBackend`,
        signals readiness by putting `{"status": "ready"}` on the out
        queue, and then runs the scheduler's event loop. On unexpected
        exceptions it logs the traceback and signals the parent process to
        terminate.
        """

        parent_process = psutil.Process().parent()
        try:
            logger.info(
                "Stage-%d is initializing, Scheduler:%s, Params:%s",
                self.stage_id,
                self.stage_config.scheduler,
                self.stage_config.scheduler_params,
            )
            # todo according to config to decide which scheduler to use
            scheduler_class = get_scheduler_class(self.stage_config.scheduler)
            comm_backend = QueueBackend(in_queue=self._in_queue, out_queue=self._out_queue)
            model_class = get_model_class(self.stage_config.model_class)
            stage_sub_dir = getattr(self.stage_config, "stage_sub_dir", None)
            precompile_params = getattr(self.stage_config, "precompile_params", None)
            self._stage_scheduler = scheduler_class(
                communication_backend=comm_backend,
                mesh=self.mesh,
                server_args=self.server_args,
                model_class=model_class,
                stage_sub_dir=stage_sub_dir,
                precompile_params=precompile_params,
                **self.stage_config.scheduler_params,
            )
            self._out_queue.put_nowait({"status": "ready"})
            logger.info(
                "Stage-%d initialized successfully, Scheduler:%s",
                self.stage_id,
                self.stage_config.scheduler,
            )
            if getattr(self._stage_scheduler, "enable_overlap", False):
                raise AssertionError(
                    "currently we not support overlap for autoregressive scheduler"
                )
                # self._stage_scheduler.event_loop_overlap()
            else:
                self._stage_scheduler.event_loop_normal()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("Stage-%d hit exception: %s", self.stage_id, traceback)
            parent_process.send_signal(signal.SIGQUIT)

    def try_collect(self):
        """Attempt to read one item from the stage's out queue without blocking.

        Returns the queued object if present, otherwise `None`.
        """

        assert self._out_queue is not None
        try:
            return self._out_queue.get_nowait()
        except Exception:
            return None


def get_scheduler_class(name: str):
    if name == "diffusion":
        return DiffusionScheduler
    elif name == "auto_regressive":
        # TODO add eventloop for auto regressive scheduler
        return AutoRegressiveScheduler
    elif name == "vae":
        # TODO add eventloop for VAE scheduler
        return VaeScheduler
    elif name == "vit":
        return VitScheduler
    elif name == "embedding":
        return EmbedScheduler
    else:
        raise ValueError(f"Unknown scheduler name: {name}")


def get_model_class(name: str):
    if name == "AutoencoderKLWan":
        return AutoencoderKLWan
    elif name == "UMT5EncoderModel":
        return UMT5EncoderModel
    elif name == "WanTransformer3DModel":
        return WanTransformer3DModel
    elif name == "WanDualTransformer3DModel":
        return WanDualTransformer3DModel
    elif name == "Qwen2_5_VL_Generation":
        return Qwen2_5_VL_Generation
    elif name == "Qwen2_5_VL_VisionModel":
        return Qwen2_5_VL_VisionModel
    elif name == "Qwen2ForCausalLM":
        return Qwen2ForCausalLM
    elif name == "Qwen3OmniEmbedding":
        return Qwen3OmniEmbedding
    elif name == "Qwen3OmniMoeTinkerTextForConditionalGeneration":
        return Qwen3OmniMoeTinkerTextForConditionalGeneration

    else:
        raise ValueError(f"Unknown model name: {name}")
