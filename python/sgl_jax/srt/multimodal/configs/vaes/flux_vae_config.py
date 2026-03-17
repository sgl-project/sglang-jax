# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sgl_jax.srt.multimodal.configs.vaes.vae_base_config import VAEConfig


@dataclass
class FluxVAEConfig(VAEConfig):
    model_class: type | None = None
    arch_config: Any = None

    in_channels: int = 3
    out_channels: int = 3
    down_block_types: tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    latent_channels: int = 16
    norm_num_groups: int = 32
    sample_size: int = 1024
    scaling_factor: float = 0.3611
    shift_factor: float | None = 0.1159
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
    force_upcast: bool = True
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False
    mid_block_add_attention: bool = True

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    use_temporal_scaling_frames: bool = False

    z_dim: int = 16
    scale_factor_temporal: int = 1
    scale_factor_spatial: int = 8

    def __post_init__(self):
        super().__post_init__()
        self.arch_config = self
        self.z_dim = self.latent_channels
        self.scale_factor_spatial = 2 ** (len(self.block_out_channels) - 1)
        self.scale_factor_temporal = 1
        self.tile_sample_min_num_frames = 1
        self.tile_sample_stride_num_frames = 1
        self.blend_num_frames = 0

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "FluxVAEConfig":
        field_names = set(cls.__dataclass_fields__.keys())
        init_dict = {}
        for key, value in config_dict.items():
            if key not in field_names:
                continue
            if key in {"down_block_types", "up_block_types", "block_out_channels"} and isinstance(
                value, list
            ):
                value = tuple(value)
            if key in {"latents_mean", "latents_std"} and isinstance(value, list):
                value = tuple(value)
            init_dict[key] = value
        return cls(**init_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> "FluxVAEConfig":
        config_path = Path(pretrained_model_name_or_path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        with config_path.open(encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
