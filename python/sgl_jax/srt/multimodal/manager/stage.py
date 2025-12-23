from queue import Queue
from typing import Any

from sgl_jax.srt.multimodal.manager.device_manager import device_manager
from sgl_jax.srt.multimodal.manager.sched.diffusion_scheduler import DiffusionScheduler
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


class Stage:
    def __init__(self, stage_config: Any):
        self._in_queue = None
        self._out_queue = None
        # self._stage_scheduler = []
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, stage_config.runtime.num_tpus],
            dcn_parallelism=[1, 1],
            device_indexes=device_manager.allocate(stage_config.runtime.num_tpus),
        )
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
        self._stage_scheduler = DiffusionScheduler(
            None, in_queue=self._in_queue, out_queue=self._out_queue, mesh=self.mesh
        )
        self._stage_scheduler.event_loop()
