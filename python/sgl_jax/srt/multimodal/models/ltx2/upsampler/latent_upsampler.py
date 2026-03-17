"""LTX-2 Latent Upsampler: 2x spatial upsampling in latent space.

Matches PyTorch ltx_core LatentUpsampler architecture:
  Conv3d(128→1024) → GroupNorm → SiLU → 4×ResBlock3d →
  [per-frame Conv2d(1024→4096) → PixelShuffle2D] →
  4×ResBlock3d → Conv3d(1024→128)
"""

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class GroupNorm(nnx.Module):
    """GroupNorm matching PyTorch nn.GroupNorm."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(num_channels))
        self.bias = nnx.Param(jnp.zeros(num_channels))

    def __call__(self, x):
        # x: [..., C, *spatial] — channels-first
        orig_shape = x.shape
        c_axis = -len(orig_shape) + 1 if x.ndim >= 3 else -1
        # For [B, C, ...] format
        B = x.shape[0]
        C = self.num_channels
        G = self.num_groups
        spatial = x.shape[2:]
        x = x.reshape(B, G, C // G, *spatial)
        mean = x.mean(axis=tuple(range(2, x.ndim)), keepdims=True)
        var = x.var(axis=tuple(range(2, x.ndim)), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(B, C, *spatial)
        # Apply affine: weight/bias are [C], broadcast over spatial
        shape = [1, C] + [1] * len(spatial)
        x = x * self.weight.value.reshape(shape) + self.bias.value.reshape(shape)
        return x


class Conv3d(nnx.Module):
    """Conv3d matching PyTorch nn.Conv3d (channels-first)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        # PyTorch: weight [out, in, D, H, W]
        # JAX lax.conv: kernel [D, H, W, in, out]
        self.weight = nnx.Param(jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size)))
        self.bias = nnx.Param(jnp.zeros(out_channels))

    def __call__(self, x):
        # x: [B, C_in, D, H, W]
        # Transpose weight from [out, in, D, H, W] to [D, H, W, in, out]
        kernel = jnp.transpose(self.weight.value, (2, 3, 4, 1, 0))
        out = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding=[(self.padding, self.padding)] * 3,
            dimension_numbers=('NCDHW', 'DHWIO', 'NCDHW'),
        )
        return out + self.bias.value.reshape(1, -1, 1, 1, 1)


class Conv2d(nnx.Module):
    """Conv2d matching PyTorch nn.Conv2d (channels-first)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = nnx.Param(jnp.zeros((out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = nnx.Param(jnp.zeros(out_channels))

    def __call__(self, x):
        # x: [B, C_in, H, W]
        kernel = jnp.transpose(self.weight.value, (2, 3, 1, 0))
        out = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1),
            padding=[(self.padding, self.padding)] * 2,
            dimension_numbers=('NCHW', 'HWIO', 'NCHW'),
        )
        return out + self.bias.value.reshape(1, -1, 1, 1)


class ResBlock3d(nnx.Module):
    """ResBlock with Conv3d + GroupNorm + SiLU, residual before final activation."""

    def __init__(self, channels: int):
        self.conv1 = Conv3d(channels, channels, 3, 1)
        self.norm1 = GroupNorm(32, channels)
        self.conv2 = Conv3d(channels, channels, 3, 1)
        self.norm2 = GroupNorm(32, channels)

    def __call__(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = jax.nn.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = jax.nn.silu(x + residual)
        return x


def pixel_shuffle_2d(x, upscale_h=2, upscale_w=2):
    """PixelShuffle for 2D: [B, C*r1*r2, H, W] → [B, C, H*r1, W*r2]."""
    B, C_in, H, W = x.shape
    C_out = C_in // (upscale_h * upscale_w)
    x = x.reshape(B, C_out, upscale_h, upscale_w, H, W)
    x = x.transpose(0, 1, 4, 2, 5, 3)  # [B, C, H, r1, W, r2]
    x = x.reshape(B, C_out, H * upscale_h, W * upscale_w)
    return x


class LatentUpsampler(nnx.Module):
    """LTX-2 latent upsampler: 2x spatial upsampling of VAE latents.

    Architecture: Conv3d → GN → SiLU → 4×ResBlock → [Conv2d+PixelShuffle] → 4×ResBlock → Conv3d
    Input:  [B, 128, F, H, W]
    Output: [B, 128, F, 2H, 2W]
    """

    def __init__(self, in_channels=128, mid_channels=1024, num_blocks_per_stage=4):
        self.initial_conv = Conv3d(in_channels, mid_channels, 3, 1)
        self.initial_norm = GroupNorm(32, mid_channels)

        self.res_blocks = nnx.List([ResBlock3d(mid_channels) for _ in range(num_blocks_per_stage)])
        self.upsampler_conv = Conv2d(mid_channels, mid_channels * 4, 3, 1)
        self.post_res_blocks = nnx.List([ResBlock3d(mid_channels) for _ in range(num_blocks_per_stage)])

        self.final_conv = Conv3d(mid_channels, in_channels, 3, 1)

    def __call__(self, x):
        """x: [B, C=128, F, H, W] → [B, C=128, F, 2H, 2W]"""
        B, C, F, H, W = x.shape

        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = jax.nn.silu(x)

        for block in self.res_blocks:
            x = block(x)

        # Spatial upsample: rearrange to per-frame 2D, conv+pixelshuffle, rearrange back
        x = x.reshape(B * F, -1, H, W)  # [B*F, 1024, H, W]
        x = self.upsampler_conv(x)       # [B*F, 4096, H, W]
        x = pixel_shuffle_2d(x, 2, 2)   # [B*F, 1024, 2H, 2W]
        x = x.reshape(B, -1, F, H * 2, W * 2)  # [B, 1024, F, 2H, 2W]

        for block in self.post_res_blocks:
            x = block(x)

        x = self.final_conv(x)
        return x

    def load_weights(self, checkpoint_path: str):
        """Load weights from safetensors checkpoint."""
        from safetensors import safe_open

        logger.info(f"Loading LatentUpsampler weights from {checkpoint_path}")

        mappings = {
            "initial_conv.weight": "initial_conv.weight",
            "initial_conv.bias": "initial_conv.bias",
            "initial_norm.weight": "initial_norm.weight",
            "initial_norm.bias": "initial_norm.bias",
            "final_conv.weight": "final_conv.weight",
            "final_conv.bias": "final_conv.bias",
        }

        # Upsampler conv (SpatialRationalResampler.conv)
        mappings["upsampler.conv.weight"] = "upsampler_conv.weight"
        mappings["upsampler.conv.bias"] = "upsampler_conv.bias"
        # upsampler.blur_down.kernel is a fixed buffer (stride=1 → no-op), skip it

        # Pre-upsample res blocks
        for i in range(4):
            for j, name in enumerate(["conv1", "conv2"]):
                mappings[f"res_blocks.{i}.{name}.weight"] = f"res_blocks.{i}.{name}.weight"
                mappings[f"res_blocks.{i}.{name}.bias"] = f"res_blocks.{i}.{name}.bias"
            for j, name in enumerate(["norm1", "norm2"]):
                mappings[f"res_blocks.{i}.{name}.weight"] = f"res_blocks.{i}.{name}.weight"
                mappings[f"res_blocks.{i}.{name}.bias"] = f"res_blocks.{i}.{name}.bias"

        # Post-upsample res blocks
        for i in range(4):
            for name in ["conv1", "conv2"]:
                mappings[f"post_upsample_res_blocks.{i}.{name}.weight"] = f"post_res_blocks.{i}.{name}.weight"
                mappings[f"post_upsample_res_blocks.{i}.{name}.bias"] = f"post_res_blocks.{i}.{name}.bias"
            for name in ["norm1", "norm2"]:
                mappings[f"post_upsample_res_blocks.{i}.{name}.weight"] = f"post_res_blocks.{i}.{name}.weight"
                mappings[f"post_upsample_res_blocks.{i}.{name}.bias"] = f"post_res_blocks.{i}.{name}.bias"

        loaded = 0
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            all_keys = set(f.keys())
            for ckpt_key, target_path in mappings.items():
                if ckpt_key in all_keys:
                    tensor = f.get_tensor(ckpt_key).float().numpy()
                    # Navigate to target attribute
                    parts = target_path.split(".")
                    obj = self
                    for p in parts[:-1]:
                        if p.isdigit():
                            obj = obj[int(p)]
                        else:
                            obj = getattr(obj, p)
                    attr_name = parts[-1]
                    existing = getattr(obj, attr_name)
                    existing_val = existing.value if isinstance(existing, nnx.Param) else existing
                    jax_tensor = jnp.array(tensor, dtype=existing_val.dtype)
                    if jax_tensor.shape != existing_val.shape:
                        logger.warning(f"Shape mismatch for {ckpt_key}: {jax_tensor.shape} != {existing_val.shape}")
                    else:
                        setattr(obj, attr_name, nnx.Param(jax_tensor))
                        loaded += 1
                else:
                    logger.warning(f"Key {ckpt_key} not found in checkpoint")

        logger.info(f"Loaded {loaded} upsampler weights")
