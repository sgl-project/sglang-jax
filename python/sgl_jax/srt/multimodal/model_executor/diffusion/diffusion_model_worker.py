import logging
import time
from collections.abc import Callable

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.configs.config_registry import get_diffusion_config
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_runner import (
    DiffusionModelRunner,
)
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


class DiffusionModelWorker:
    def __init__(
        self,
        server_args: MultimodalServerArgs,
        mesh: jax.sharding.Mesh = None,
        model_class=None,
        stage_sub_dir: str | None = None,
        scheduler: str | None = None,
    ):
        self.mesh = mesh
        self.scheduler = scheduler
        self.model_runner = DiffusionModelRunner(
            server_args,
            self.mesh,
            model_class=model_class,
            stage_sub_dir=stage_sub_dir,
            scheduler=self.scheduler,
        )
        self.initialize()
        self.precompile_width_heights = server_args.precompile_width_heights
        self.precompile_frame_paddings = server_args.precompile_frame_paddings
        self.model_config = get_diffusion_config(server_args.model_path)

    def initialize(self):
        pass
        # self.model_loader.load_model()
        # init cache here if needed
        # init different attention backend if needed

    def forward(
        self,
        batch: Req,
        mesh: jax.sharding.Mesh,
        abort_checker: Callable[[], bool] | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> bool:
        """Generate video from text embeddings using the diffusion model.

        Args:
            batch: Request batch containing text embeddings and parameters.
            mesh: JAX device mesh for sharding.
            abort_checker: Optional callback that returns True if the request
                should be aborted. Called between diffusion steps.

        Returns:
            True if the request was aborted, False otherwise.
        """
        return self.model_runner.forward(
            batch, mesh, abort_checker=abort_checker, step_callback=step_callback
        )

    def run_precompile(self):
        self.precompile()

    def precompile(self):
        start_time = time.perf_counter()
        logger.info(
            "[DIFFUSION] Begin to precompile width*height=%s",
            self.precompile_width_heights,
        )

        with tqdm(
            self.precompile_width_heights, desc="[DIFFUSION] PRECOMPILE", leave=False
        ) as pbar:
            for wh in pbar:
                whs = wh.split("*")
                width, height = int(whs[0]), int(whs[1])
                assert width % self.model_config.scale_factor_spatial == 0
                assert height % self.model_config.scale_factor_spatial == 0
                for t in self.precompile_frame_paddings:
                    pbar.set_postfix(wh=wh, t=t)
                    embeds = np.random.random((2, 512, self.model_config.text_dim))
                    embeds = device_array(embeds, sharding=NamedSharding(self.mesh, P()))
                    # FLUX needs pooled_embeds for CLIP pooled projections
                    pooled = np.random.random((1, 768)).astype(np.float32)
                    pooled = device_array(pooled, sharding=NamedSharding(self.mesh, P()))
                    req = Req(
                        prompt_embeds=embeds[0],
                        negative_prompt_embeds=embeds[1],
                        pooled_embeds=[pooled],
                        do_classifier_free_guidance=True,
                        width=width,
                        height=height,
                        num_frames=t,
                        num_inference_steps=1,
                    )
                    self.model_runner.forward(req, self.mesh)
        end_time = time.perf_counter()
        logger.info("[DIFFUSION] Precompile finished in %.0f secs", end_time - start_time)
