import logging
import time

import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from sgl_jax.srt.multimodal.configs.config_registry import get_vae_config
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vae.vae_model_runner import VaeModelRunner
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


class VaeModelWorker:
    def __init__(
        self,
        server_args: ServerArgs = None,
        mesh=None,
        model_class=None,
        stage_sub_dir: str | None = None,
    ):
        self.mesh = mesh
        self.model_runner = VaeModelRunner(
            server_args, mesh, model_class=model_class, stage_sub_dir=stage_sub_dir
        )
        self.server_args = server_args
        self.vae_decode_precompile_width_height = server_args.vae_decode_precompile_width_height
        self.vae_decode_precompile_frame_paddings = server_args.vae_decode_precompile_frame_paddings
        self.model_config = get_vae_config(self.server_args.model_path)
        # Initialize model here based on model_config

    def forward(self, batch: Req):
        # Implement the vae model inference logic here
        # return batch
        return self.model_runner.forward(batch.latents, "decode")

    def run_precompile(self):
        self.decode_precompile()
        self.encode_precompile()

    def decode_precompile(self):
        start_time = time.perf_counter()
        logger.info(
            "[VAE DECODE] Begin to precompile width*height=%s",
            self.vae_decode_precompile_width_height,
        )

        with tqdm(
            self.vae_decode_precompile_width_height, desc="[VAE DECODE] PRECOMPILE", leave=False
        ) as pbar:
            for wh in pbar:
                whs = wh.split("*")
                width, height = int(whs[0]), int(whs[1])
                assert width % self.model_config.scale_factor_spatial == 0
                assert height % self.model_config.scale_factor_spatial == 0
                for t in self.vae_decode_precompile_frame_paddings:
                    pbar.set_postfix(wh=wh, t=t)
                    latents_cpu = np.random.random(
                        (
                            1,
                            t // self.model_config.scale_factor_temporal + 1,
                            height // self.model_config.scale_factor_spatial,
                            width // self.model_config.scale_factor_spatial,
                            self.model_config.z_dim,
                        )
                    )
                    latents = device_array(latents_cpu, sharding=(NamedSharding(self.mesh, P())))
                    self.model_runner.forward(latents, "decode")
        end_time = time.perf_counter()
        logger.info("[VAE DECODE] Precompile finished in %.0f secs", end_time - start_time)

    def encode_precompile(self):
        pass
