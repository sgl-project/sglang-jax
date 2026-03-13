"""
Configuration for LTX-2 VideoDecoder.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LTX2VAEConfig:
    """Configuration for LTX-2 Video VAE Decoder."""

    # Framework fields
    revision: str | None = None
    dtype: str = "bfloat16"
    scale_factor_spatial: int = 32
    scale_factor_temporal: int = 8
    z_dim: int = 128

    # Model architecture
    in_channels: int = 128
    out_channels: int = 3
    decoder_blocks: List[Tuple[str, dict]] = field(default_factory=lambda: [
        # Matches checkpoint: 7 up_blocks, channels 1024->512->256->128
        ("res_x", {"num_layers": 5}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5}),
    ])
    patch_size: int = 4
    norm_layer: str = "pixel_norm"
    causal: bool = False
    timestep_conditioning: bool = False
    decoder_spatial_padding_mode: str = "reflect"

    # Weight loading
    load_decoder: bool = True
    load_encoder: bool = False

    # HuggingFace model identifier
    model_id: str = "Lightricks/LTX-Video"
    subfolder: str = "vae"
