import dataclasses

import jax.numpy as jnp
from jax.lax import Precision

from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs


@dataclasses.dataclass
class LTX2ModelConfig(MultiModalModelConfigs):
    """
    Configuration for LTX-2 audio-video transformer model.

    LTX-2 is the first DiT-based audio-video foundation model that supports:
    - Synchronized audio and video generation
    - Text-to-video, image-to-video, and video-to-video generation
    - High fidelity outputs with multiple performance modes
    """

    # Model architecture parameters (from LTX-2 config)
    num_layers: int = 48
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    hidden_size: int = 4096  # num_attention_heads * attention_head_dim

    # Video parameters
    in_channels: int = 128
    out_channels: int = 128
    cross_attention_dim: int = 4096  # Video cross attention dimension

    # Audio parameters
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_cross_attention_dim: int = 2048  # Audio cross attention dimension

    # Common parameters
    caption_channels: int = 3840
    epsilon: float = 1e-6
    qk_norm: str = "rms_norm"

    # Positional embedding parameters
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: tuple[int, int, int] = (20, 2048, 2048)  # (frames, height, width)
    audio_positional_embedding_max_pos: tuple[int] = (20,)
    use_middle_indices_grid: bool = True
    rope_type: str = "split"  # "interleaved" or other rope types
    double_precision_rope: bool = True

    # Timestep embedding parameters
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: float = 1000.0  # Audio-video cross attention timestep scale

    # Patch embedding parameters
    patch_size: tuple[int, int, int] = (1, 2, 2)  # (temporal, height, width)

    # Model identity
    model_type: str = "ltx2"

    # Model type flags
    is_video_enabled: bool = True
    is_audio_enabled: bool = True
    apply_gated_attention: bool = False

    # Runtime/inference params
    weights_dtype: jnp.dtype = jnp.bfloat16
    dtype: jnp.dtype = jnp.bfloat16
    precision: Precision = Precision.HIGHEST

    # Generation parameters
    max_text_len: int = 512
    num_frames: int = 5
    latent_size: tuple[int, int] = (64, 64)
    num_inference_steps: int = 50
    guidance_scale: float = 5.0

    # Latent space parameters
    latent_input_dim: int = 128
    scale_factor_temporal: int = 8   # VAE temporal compression ratio
    scale_factor_spatial: int = 32   # VAE spatial compression ratio

    # Scheduler and execution parameters
    scheduler_type: str = "EulerScheduler"
    use_dynamic_shifting: bool = True
    base_shift: float = 0.95
    max_shift: float = 2.05
    stg_mode: bool = True
    fps: float = 24.0
    max_sequence_length: int = 1024
    flow_shift: float | None = None

    model_class: None = None  # To be set to the model class
    quantization_config: QuantizationConfig | None = None

    # Default example prompt following LTX-2 prompting best practices:
    # single continuous paragraph, present tense, specific camera choreography,
    # physical details, temporal connectors, atmospheric audio description.
    default_prompt: str = (
        "An animated cinematic shot of a matte-gray humanoid robot walking slowly "
        "down the center of a rain-slicked city sidewalk at dusk. The camera begins "
        "in a medium wide shot, dollying steadily backward to keep the robot framed "
        "at center as it takes heavy, deliberate steps, its metal feet clanking "
        "softly against the wet pavement. Warm amber streetlights reflect off its "
        "polished shoulder plates and the shallow puddles beneath it. As the robot "
        "continues forward, its pace gradually increases into a slow, heavy run, "
        "mechanical joints grinding with each stride, arms pumping stiffly at its "
        "sides. Then it stops abruptly, one foot planted forward, steam venting from "
        "its knee actuators. The camera keeps pulling back smoothly, widening the "
        "frame, until a second robot with a blue metallic shell appears from behind "
        "the camera in an over-the-shoulder composition, its silhouette filling the "
        "right edge of frame. The blue robot stands motionless, facing the gray one "
        "from a distance. Cold blue neon signage and warm orange storefront lights "
        "create contrasting pools of color on the glistening asphalt. Ambient city "
        "sounds fill the scene: distant traffic hum, the patter of light rain on "
        "metal, and the low mechanical whir of cooling fans inside the robots."
    )

    def get_total_num_kv_heads(self) -> int:
        return self.num_attention_heads


@dataclasses.dataclass
class LTX2VideoOnlyModelConfig(LTX2ModelConfig):
    """Configuration for LTX-2 video-only model (no audio)."""

    is_video_enabled: bool = True
    is_audio_enabled: bool = False


@dataclasses.dataclass
class LTX2AudioOnlyModelConfig(LTX2ModelConfig):
    """Configuration for LTX-2 audio-only model (no video)."""

    is_video_enabled: bool = False
    is_audio_enabled: bool = True
