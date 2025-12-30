from queue import Queue
from typing import Any

from sgl_jax.srt.managers.scheduler import Scheduler as AutoRegressiveScheduler
from sgl_jax.srt.multimodal.manager.device_manager import device_manager
from sgl_jax.srt.multimodal.manager.scheduler.diffusion_scheduler import (
    DiffusionScheduler,
)
from sgl_jax.srt.multimodal.manager.scheduler.vae_scheduler import VaeScheduler
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


class Stage:
    def __init__(self, stage_config: Any):
        self._in_queue = None
        self._out_queue = None
        # this parallelism setting is accord to stage config
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, stage_config.runtime.num_tpus],
            dcn_parallelism=[1, 1],
            device_indexes=device_manager.allocate(stage_config.runtime.num_tpus),
        )
        self.stage_config = stage_config
        # mesh

    def set_in_queue(self, in_queue: Queue):
        self._in_queue = in_queue

    def set_out_queue(self, out_queue: Queue):
        self._out_queue = out_queue

    def set_stage_index(self, stage_index):
        self.stage_index = stage_index

    def run_stage(self):
        print(f"stage start {self.stage_index}")
        # todo according to config to decide which scheduler to use
        scheduler_class = get_scheduler_class(self.stage_config.scheduler)
        self._stage_scheduler = scheduler_class(
            in_queue=self._in_queue, out_queue=self._out_queue, **self.stage_config.scheduler_params
        )
        self._stage_scheduler.event_loop()

    def try_collect(self):
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
    else:
        raise ValueError(f"Unknown scheduler name: {name}")
