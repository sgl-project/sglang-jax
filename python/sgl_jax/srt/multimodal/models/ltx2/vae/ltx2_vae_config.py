"""
Configuration for LTX-2 VideoDecoder.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LTX2VAEConfig:
    revision: str | None = None
    """Configuration for LTX-2 Video VAE Decoder.

    Attributes:
        in_channels: Number of input latent channels (default 128)
        out_channels: Number of output video channels (default 3 for RGB)
        decoder_blocks: List of decoder block specifications
        patch_size: Final spatial expansion factor (default 4)
        norm_layer: Normalization layer type ('pixel_norm' or 'group_norm')
        causal: Whether to use causal convolutions (default False)
        timestep_conditioning: Whether to use timestep conditioning (default False)
        decoder_spatial_padding_mode: Padding mode for spatial dimensions
        load_decoder: Whether to load the decoder (default True)
        load_encoder: Whether to load the encoder (default False, decoder only)
    """

    # Model architecture
    in_channels: int = 128
    out_channels: int = 3
    decoder_blocks: List[Tuple[str, dict]] = field(default_factory=lambda: [
        ("compress_all", {"residual": True, "multiplier": 1}),
        ("res_x", {"num_layers": 3}),
        ("compress_all", {"residual": True, "multiplier": 1}),
        ("res_x", {"num_layers": 3}),
        ("compress_time", {}),
        ("res_x", {"num_layers": 3}),
        ("compress_space", {}),
        ("res_x", {"num_layers": 3}),
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
