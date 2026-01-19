# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs


@dataclass
class VAEConfig(MultiModalModelConfigs):

    load_encoder: bool = True
    load_decoder: bool = True

    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_num_frames: int = 16
    tile_sample_stride_height: int = 192
    tile_sample_stride_width: int = 192
    tile_sample_stride_num_frames: int = 12
    blend_num_frames: int = 0

    use_tiling: bool = True
    use_temporal_tiling: bool = True
    use_parallel_tiling: bool = True
    use_temporal_scaling_frames: bool = True

    def __post_init__(self):
        self.blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames

    def post_init(self):
        pass

    def get_vae_scale_factor(self):
        return 2 ** (len(self.arch_config.block_out_channels) - 1)

    def encode_sample_mode(self):
        return "argmax"
