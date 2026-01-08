import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
from jax.lax import Precision


@dataclasses.dataclass(frozen=True)
class WanModelConfig:
    """Configuration for Wan2.1 Diffusion Transformer.

    Field names match the HuggingFace diffusers config_json format.
    """

    # Core architecture params (from HF config.json)
    num_layers: int = 30
    num_attention_heads: int = 12
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    ffn_dim: int = 8960
    freq_dim: int = 256
    text_dim: int = 4096
    image_dim: int | None = None  # None for T2V, set for I2V
    patch_size: tuple[int, int, int] = (1, 2, 2)
    cross_attn_norm: bool = True
    qk_norm: str | None = "rms_norm_across_heads"
    eps: float = 1e-6
    added_kv_proj_dim: int | None = None  # None for T2V, set for I2V
    rope_max_seq_len: int = 1024

    # Runtime/inference params (not in HF config)
    weights_dtype: jnp.dtype = jnp.bfloat16
    dtype: jnp.dtype = jnp.bfloat16
    precision: Precision = Precision.HIGHEST
    max_text_len: int = 512
    num_frames: int = 11
    latent_size: tuple[int, int] = (60, 90)
    num_inference_steps: int = 30
    guidance_scale: float = 5.0

    @property
    def hidden_dim(self) -> int:
        """Computed hidden dimension from num_attention_heads * attention_head_dim."""
        return self.num_attention_heads * self.attention_head_dim

    @property
    def num_heads(self) -> int:
        """Alias for num_attention_heads."""
        return self.num_attention_heads

    @property
    def head_dim(self) -> int:
        """Alias for attention_head_dim."""
        return self.attention_head_dim

    @property
    def epsilon(self) -> float:
        """Alias for eps."""
        return self.eps

    @classmethod
    def from_json(cls, path: str | Path) -> "WanModelConfig":
        """Load config from HuggingFace config.json file.

        Args:
            path: Path to config.json file or directory containing it.

        Returns:
            WanModelConfig instance.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"

        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "WanModelConfig":
        """Load config from dictionary.

        Args:
            data: Dictionary with config values (HF format).

        Returns:
            WanModelConfig instance.
        """
        # Map HF field names to our field names
        field_mapping = {
            "_class_name": None,  # Skip
            "_diffusers_version": None,  # Skip
        }

        # Get valid field names from dataclass
        valid_fields = {f.name for f in dataclasses.fields(cls)}

        kwargs = {}
        for key, value in data.items():
            # Skip fields we don't need
            if key in field_mapping and field_mapping[key] is None:
                continue

            # Map field name if needed
            mapped_key = field_mapping.get(key, key)

            # Only include valid fields
            if mapped_key in valid_fields:
                # Handle special conversions
                if mapped_key == "patch_size" and isinstance(value, list):
                    value = tuple(value)
                kwargs[mapped_key] = value

        return cls(**kwargs)
