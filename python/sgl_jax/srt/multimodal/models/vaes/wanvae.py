## adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.lax import Precision

CACHE_T = 2


class CausalConv3d(nnx.Module):
    """Causal 3D convolution that doesn't look into the future."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        *,
        rngs: nnx.Rngs,
        padding: tuple[int, int, int] = (0, 0, 0),
        strides: tuple[int, int, int] | None = None,
    ):
        self.kernel_size = kernel_size
        self.temporal_padding = padding[0]  # Save for cache size calculation
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding="VALID",  # We'll handle padding manually
            rngs=rngs,
            precision=Precision.HIGHEST,  # todo make this parameters
        )
        self.padding = (
            (0, 0),
            (2 * padding[0], 0),
            (padding[1], padding[1]),
            (padding[2], padding[2]),
            (0, 0),
        )

    def __call__(self, x: jax.Array, cache: jax.Array = None) -> tuple[jax.Array, jax.Array]:
        """Forward pass with optional caching.
        Args:
            x: [B, T, H, W, C] input (JAX channel-last format)
            cache: [B, CACHE_T, H, W, C] cached frames from previous call, or None
        Returns:
            out: [B, T_out, H_out, W_out, C_out] output
            new_cache: [B, CACHE_T, H, W, C] cache for next call, or None
        """
        # Cache size is 2*padding because we pad left by (2*padding, 0) for causality
        cache_t = 2 * self.temporal_padding
        if cache is not None and cache_t > 0:
            x = jnp.concatenate([cache, x], axis=1)  # [B, T+CACHE_T, H, W, C]
            padding = list(self.padding)
            padding[1] = (max(0, self.padding[1][0] - cache.shape[1]), 0)  # Reduce left padding
            padding = tuple(padding)
        else:
            padding = self.padding

        x_padded = jnp.pad(x, padding, mode="constant")
        out = self.conv(x_padded)

        # Extract cache for next iteration: last cache_t frames of INPUT (before conv)
        # Always create cache if we have temporal padding (even on first frame)
        if cache_t > 0:
            new_cache = x[:, -cache_t:, :, :, :]  # [B, <=CACHE_T, H, W, C]
            # Pad on the left if we do not yet have cache_t frames (e.g., first call with T=1).
            # if new_cache.shape[1] < cache_t:
            #     pad_t = cache_t - new_cache.shape[1]
            #     new_cache = jnp.pad(new_cache, ((0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)), mode="constant")
        else:
            new_cache = None

        return out, new_cache


class RMSNorm(nnx.Module):
    """RMS Normalization with L2 normalize and learned scale.
    Based on F.normalize approach: normalize to unit norm, then scale.
    For videos (images=False), uses 3D spatial+temporal normalization.
    """

    def __init__(self, dim: int, images: bool = True, *, rngs: nnx.Rngs):
        self.scale_factor = dim**0.5
        # gamma shape: (dim,) will broadcast to [B, T, H, W, C] or [B, H, W, C]
        shape = (1, 1, dim) if images else (1, 1, 1, dim)
        self.scale = nnx.Param(jnp.ones(shape), dtype=jnp.float32)
        self.eps = 1e-12

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [B, T, H, W, C] for 3D or [B, H, W, C] for 2D
        # Normalize to unit RMS along the channel dimension manually since jax.nn.normalize is unavailable.
        rms = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        x_normalized = x / rms
        return x_normalized * self.scale_factor * self.scale.get_value()


class ResidualBlock(nnx.Module):
    """Residual block with RMSNorm and SiLU activation."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0, *, rngs: nnx.Rngs
    ):
        self.norm1 = RMSNorm(in_channels, images=False, rngs=rngs)
        self.conv1 = CausalConv3d(
            in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), rngs=rngs
        )
        self.norm2 = RMSNorm(out_channels, images=False, rngs=rngs)
        self.conv2 = CausalConv3d(
            out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout)
        if in_channels != out_channels:
            self.skip_conv = CausalConv3d(
                in_channels, out_channels, kernel_size=(1, 1, 1), rngs=rngs
            )
        else:
            self.skip_conv = None

    def __call__(
        self, x: jax.Array, cache_list: tuple[Any, ...] = None, cache_idx: list[int] = None
    ) -> tuple[jax.Array, tuple[Any, ...] | None]:
        residual = x
        if self.skip_conv is not None:
            residual, _ = self.skip_conv(x)
        x = self.norm1(x)
        x = nnx.silu(x)

        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv1(x, cache_list[idx])
            if new_cache.shape[2] < 2 and cache_list[idx] is not None:
                new_cache = jnp.concatenate(
                    [jnp.expand_dims(cache_list[idx][:, -1, :, :, :], 1), new_cache], axis=1
                )
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
            cache_idx[0] += 1
        else:
            x, _ = self.conv1(x)

        x = self.norm2(x)
        x = nnx.silu(x)
        x = self.dropout(x)

        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv2(x, cache_list[idx])
            if new_cache.shape[2] < 2 and cache_list[idx] is not None:
                new_cache = jnp.concatenate(
                    [jnp.expand_dims(cache_list[idx][:, -1, :, :, :], 1), new_cache], axis=1
                )
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
            cache_idx[0] += 1
        else:
            x, _ = self.conv2(x)
        return x + residual, cache_list


class Upsample2d(nnx.Module):
    """Spatial 2x upsample that also halves channels, mirroring torch Resample."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )

    def __call__(
        self, x: jax.Array, cache_list: tuple[jax.Array, ...] = None, cache_idx: list[int] = None
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        # x: [B, T, H, W, Cin]
        b, t, h, w, _ = x.shape
        orig_dtype = x.dtype
        x = x.reshape(b * t, h, w, self.in_channels)
        x = jax.image.resize(
            x.astype(jnp.float32), (b * t, h * 2, w * 2, self.in_channels), method="nearest"
        ).astype(orig_dtype)
        x = self.spatial_conv(x)
        return x.reshape(b, t, h * 2, w * 2, self.out_channels), cache_list


class Upsample3d(nnx.Module):
    """Temporal+spatial 2x upsample with channel reduction (like torch Resample)."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_conv = CausalConv3d(
            in_channels, in_channels * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0), rngs=rngs
        )
        self.spatial_conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )

    def __call__(
        self, x: jax.Array, cache_list: tuple[jax.Array, ...] = None, cache_idx: list[int] = None
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        b, t, h, w, _ = x.shape
        orig_dtype = x.dtype
        t_out = t
        if cache_list is not None:
            idx = cache_idx[0]
            # First frame: skip time_conv, only do spatial upsampling
            if cache_list[idx] is None:
                # Use zero array as sentinel with SAME shape as real cache
                # This ensures consistent pytree structure for JIT
                # We use zeros with shape [B, 2, H, W, C] where 2 = cache size for 3x1x1 conv
                sentinel = jnp.zeros((b, 2, h, w, self.in_channels), dtype=orig_dtype)
                cache_list = (*cache_list[:idx], sentinel, *cache_list[idx + 1 :])
                cache_idx[0] += 1
                t_out = t
            else:
                # Always pass the cached features (including the zero sentinel) so the
                # time_conv sees a length-2 cache and returns a length-2 cache, matching
                # the torch behavior where the sentinel seeds the cache.
                x, new_cache = self.time_conv(x, cache_list[idx])
                if new_cache.shape[2] < 2 and cache_list[idx] is not None:
                    new_cache = jnp.concatenate(
                        [jnp.expand_dims(cache_list[idx][:, -1, :, :, :], 1), new_cache], axis=1
                    )
                cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
                cache_idx[0] += 1

                x = x.reshape(b, t, h, w, 2, self.in_channels)
                x = jnp.moveaxis(x, 4, 2)  #  [B, T, H, W, 2, Cin] -> [B, T, 2, H, W, Cin]
                t_out = t * 2
                x = x.reshape(b, t_out, h, w, self.in_channels)

        # Spatial upsampling (always applied)
        bt = b * t_out
        x = x.reshape(bt, h, w, self.in_channels)
        x = jax.image.resize(
            x.astype(jnp.float32), (bt, h * 2, w * 2, self.in_channels), method="nearest"
        ).astype(orig_dtype)
        x = self.spatial_conv(x)
        return x.reshape(b, t_out, h * 2, w * 2, self.out_channels), cache_list

class Downsample2d(nnx.Module):
    """Spatial 2x upsample that also halves channels, mirroring torch Resample."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
        self.padding = (
            (0, 0),
            (0, 0),
            (0, 1),
            (0, 1),
            (0, 0),
        )

    def __call__(
        self, x: jax.Array, cache_list: tuple[jax.Array, ...] = None, cache_idx: list[int] = None
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        # x: [B, T, H, W, Cin]
        b, t, h, w, _ = x.shape
        x = jnp.pad(x, self.padding, mode="constant")
        x = self.spatial_conv(x)
        return x, cache_list


class Downsample3d(nnx.Module):
    """Temporal+spatial 2x upsample with channel reduction (like torch Resample)."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_conv = CausalConv3d(
            in_channels, in_channels, kernel_size=(3, 1, 1), strides = (2, 1, 1), rngs=rngs
        )
        self.spatial_conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides = (2, 2),
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
        self.padding = (
            (0, 0),
            (0, 0),
            (0, 1),
            (0, 1),
            (0, 0),
        )

    def __call__(
        self, x: jax.Array, cache_list: tuple[jax.Array, ...] = None, cache_idx: list[int] = None
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        b, t, h, w, _ = x.shape
        x = jnp.pad(x, self.padding, mode = "constant")
        x = self.spatial_conv(x)
        if cache_list is not None:
            idx = cache_idx[0]
            if cache_list[idx] is None:
                cache_list = (*cache_list[:idx], x.copy(), *cache_list[idx + 1 :])
                cache_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x, _ = self.time_conv(jnp.concatenate([cache_list[idx][:, -1:, :, :, :], x], 1))
                cache_list = (*cache_list[:idx], cache_x, *cache_list[idx + 1 :])
                cache_idx[0] += 1
        return x, cache_list


class AttentionBlock(nnx.Module):
    """Spatial attention block with batched frame processing."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.norm = RMSNorm(channels, rngs=rngs)
        self.qkv = nnx.Conv(
            in_features=channels,
            out_features=channels * 3,
            kernel_size=(1, 1),
            use_bias=True,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
        self.proj = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=(1, 1),
            use_bias=True,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [B, T, H, W, C]
        b, t, h, w, c = x.shape
        residual = x

        x = x.reshape(b * t, h, w, c)
        x = self.norm(x)
        # QKV projection: [B*T, H, W, C] -> [B*T, H, W, 3*C]
        qkv = self.qkv(x)

        # Reshape for attention: [B*T, H, W, 3*C] -> [B*T, H*W, 3*C] -> split to Q, K, V
        qkv = qkv.reshape(b * t, h * w, 3 * c)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # Each: [B*T, H*W, C]
        # todo use kernel attention to do this
        # Scaled dot-product attention
        scale = c**-0.5
        attn = jax.nn.softmax(jnp.einsum("bic,bjc->bij", q, k) * scale, axis=-1)  # [B*T, H*W, H*W]
        out = jnp.einsum("bij,bjc->bic", attn, v)  # [B*T, H*W, C]

        # Reshape back to spatial: [B*T, H*W, C] -> [B*T, H, W, C]
        out = out.reshape(b * t, h, w, c)

        # Output projection
        out = self.proj(out)

        # Reshape back to video: [B*T, H, W, C] -> [B, T, H, W, C]
        out = out.reshape(b, t, h, w, c)

        return out + residual


class MidBlock(nnx.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
    """

    def __init__(
        self, dim: int, dropout: float = 0.0, num_layers: int = 1, *, rngs: nnx.Rngs = None
    ):
        self.dim = dim

        # Create the components
        resnets = [ResidualBlock(dim, dim, dropout, rngs=rngs)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(AttentionBlock(dim, rngs=rngs))
            resnets.append(ResidualBlock(dim, dim, dropout, rngs=rngs))
        self.attentions = nnx.List(attentions)
        self.resnets = nnx.List(resnets)

        self.gradient_checkpointing = False

    def __call__(self, x, cache_list: tuple[jax.Array, ...] = None, cache_idx=[0]):
        # First residual block
        x, cache_list = self.resnets[0](x, cache_list=cache_list, cache_idx=cache_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x, cache_list = resnet(x, cache_list=cache_list, cache_idx=cache_idx)

        return x, cache_list


class UpBlock(nnx.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        *,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(ResidualBlock(current_dim, out_dim, dropout, rngs=rngs))
            current_dim = out_dim

        self.resnets = nnx.List(resnets)

        # Add upsampling layer if needed
        self.upsamplers = nnx.data(None)
        upsamplers = []
        if upsample_mode is not None:
            if upsample_mode == "upsample2d":
                upsamplers.append(Upsample2d(out_dim, out_dim // 2, rngs=rngs))
            elif upsample_mode == "upsample3d":
                upsamplers.append(Upsample3d(out_dim, out_dim // 2, rngs=rngs))
        self.upsamplers = nnx.List(upsamplers)

        self.gradient_checkpointing = False

    def __call__(self, x: jax.Array, cache_list: jax.Array = None, cache_idx=[0], first_chunk=None):
        """
        Forward pass through the upsampling block.

        Args:
            x (jax.Array): Input tensor
            cache_list (list, optional): Feature cache for causal convolutions
            cache_idx (list, optional): Feature index for cache management

        Returns:
            jax.Array: Output tensor
        """
        for resnet in self.resnets:
            if cache_list is not None:
                x, cache_list = resnet(x, cache_list=cache_list, cache_idx=cache_idx)
            else:
                x, _ = resnet(x)

        if self.upsamplers is not None and len(self.upsamplers) > 0:
            if cache_list is not None:
                x, cache_list = self.upsamplers[0](x, cache_list=cache_list, cache_idx=cache_idx)
            else:
                x = self.upsamplers[0](x)
        return x, cache_list


class Decoder3d(nnx.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_upsample=(True, True, False),
        dropout=0.0,
        out_channels: int = 3,
        is_residual: bool = False,
        *,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        self.z_dim = z_dim
        dim_mult = list(dim_mult)
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_upsample = list(temperal_upsample)
        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = CausalConv3d(z_dim, dims[0], (3, 3, 3), rngs=rngs, padding=(1, 1, 1))

        self.mid_block = MidBlock(dims[0], dropout, num_layers=1, rngs=rngs)
        self.up_blocks = nnx.List([])
        if is_residual:
            raise RuntimeError("not implemented")
        else:
            # in : [384, 384, 384, 192]
            # out: [384, 384, 192, 96]
            for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
                # 4
                # residual (+attention) blocks
                if i > 0 and not is_residual:
                    # wan vae 2.1
                    in_dim = in_dim // 2

                # determine if we need upsampling
                up_flag = i != len(dim_mult) - 1
                # determine upsampling mode, if not upsampling, set to None
                upsample_mode = None
                if up_flag and temperal_upsample[i]:
                    upsample_mode = "upsample3d"
                elif up_flag:
                    upsample_mode = "upsample2d"
                # 0 (384, 384)
                # 1 (192, 384)
                # 2 (192, 192)
                # 3 (192, 96)
                up_block = UpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    rngs=rngs,
                )
                self.up_blocks.append(up_block)
        # output blocks
        self.norm_out = RMSNorm(out_dim, images=False, rngs=rngs)
        self.conv_out = CausalConv3d(out_dim, out_channels, (3, 3, 3), padding=(1, 1, 1), rngs=rngs)

    def __call__(
        self, z: jax.Array, cache_list: tuple[Any, ...] = None, cache_idx: list[int] = None
    ) -> tuple[Array, tuple[Any, ...] | None]:
        # Initial convolution
        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv_in(z, cache_list[idx])
            if new_cache.shape[2] < 2 and cache_list[idx] is not None:
                new_cache = jnp.concatenate(
                    [jnp.expand_dims(cache_list[idx][:, -1, :, :, :], 1), new_cache], axis=1
                )
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
            cache_idx[0] += 1
        else:
            x, _ = self.conv_in(z, None)
        x, cache_list = self.mid_block(x, cache_list, cache_idx)
        for block in self.up_blocks:
            x, cache_list = block(x, cache_list, cache_idx)
        x = self.norm_out(x)
        x = nnx.silu(x)
        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv_out(x, cache_list[idx])
            if new_cache.shape[2] < 2 and cache_list[idx] is not None:
                new_cache = jnp.concatenate(
                    [jnp.expand_dims(cache_list[idx][:, -1, :, :, :], 1), new_cache], axis=1
                )
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
            cache_idx[0] += 1
        else:
            x, _ = self.conv_out(x, None)
        return x, cache_list


class Encoder3d(nnx.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
        *,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = CausalConv3d(in_channels, dims[0], (3,3,3), padding=(1,1,1), rngs = rngs)

        # downsample blocks
        self.down_blocks = nnx.List([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if is_residual:
                raise NotImplementedError
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(ResidualBlock(in_dim, out_dim, dropout, rngs=rngs))
                    if scale in attn_scales:
                        self.down_blocks.append(AttentionBlock(out_dim, rngs=rngs))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    if temperal_downsample[i]:
                        self.down_blocks.append(Downsample3d(out_dim, out_dim,  rngs = rngs))
                    else:
                        self.down_blocks.append(Downsample2d(out_dim, out_dim, rngs = rngs))
                    scale /= 2.0

        # middle blocks
        self.mid_block = MidBlock(out_dim, dropout, num_layers=1, rngs=rngs)

        # output blocks
        self.norm_out = RMSNorm(out_dim, images=False, rngs = rngs)
        self.conv_out = CausalConv3d(out_dim, z_dim, (3,3,3), padding=(1,1,1), rngs = rngs)

        self.gradient_checkpointing = False

    def __call__(self, x, cache_list = None, cache_idx=[0]):
        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv_in(x, cache_list[idx])
            if new_cache.shape[2] < 2 and cache_list[idx] is not None:
                new_cache = jnp.concatenate(
                    [jnp.expand_dims(cache_list[idx][:, -1, :, :, :], 1), new_cache], axis = 1
                )
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1:])
            cache_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            if cache_list is not None:
                x, cache_list = layer(x, cache_list=cache_list, cache_idx=cache_idx)
            else:
                x = layer(x)

        ## middle
        x, cache_list = self.mid_block(x, cache_list=cache_list, cache_idx=cache_idx)

        ## head
        x = self.norm_out(x)
        x = nnx.relu(x)
        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv_out(x, cache_list[idx])
            if new_cache.shape[2] < 2 and cache_list[idx] is not None:
                new_cache = jnp.concatenate(
                    [jnp.expand_dims(cache_list[idx][:, -1, :, :, :], 1), new_cache], axis = 1
                )
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1:])
            cache_idx[0] += 1
        else:
            x, _ = self.conv_out(x)

        return x, cache_list


if __name__ == "__main__":
    # model = Decoder3D(dim=96, z_dim=16, rngs=nnx.Rngs(0))
    # model_def, model_state = nnx.split(model)
    # for key, value in model_state.flat_state():
    #     print(key, value.shape)
    # key = jax.random.PRNGKey(42)
    # key, subkey = jax.random.split(key)
    # x = jax.random.uniform(subkey, shape=(1, 2, 3, 4, 16))
    # y, cache_list = model(x, cache_list=tuple([None] * 32), cache_idx=[0])
    # print(y.shape)
    # for i, cache in enumerate(cache_list):
    #     print(i, cache.shape)

    model = Encoder3d(dim=96, z_dim=16, rngs=nnx.Rngs(0))
    model_def, model_state = nnx.split(model)
    for key, value in model_state.flat_state():
        print(key, value.shape)
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey, shape=(1, 2, 3, 4, 16))
    y, cache_list = model(x, cache_list=tuple([None] * 32), cache_idx=[0])
    print(y.shape)
    for i, cache in enumerate(cache_list):
        print(i, cache.shape)

# class AutoencoderKLWan(nnx.Module):
#     def __init__(
#             self,
#             config: WanVAEConfig,
#         ) -> None:
#         self.temperal_downsample = list(config.temperal_downsample)
#         self.temperal_upsample = list(config.temperal_downsample)[::-1]
#         if config.decoder_base_dim is None:
#             decoder_base_dim = config.base_dim
#         else:
#             decoder_base_dim = config.decoder_base_dim
#
#         self.latents_mean = list(config.latents_mean)
#         self.latents_std = list(config.latents_std)
#         self.shift_factor = config.shift_factor
#
#         self.quant_conv = CausalConv3d(self.z_dim * 2, self.z_dim * 2, (1, 1, 1))
#         self.post_quant_conv = CausalConv3d(self.z_dim, self.z_dim, (1, 1, 1))
#
#         if config.load_decoder:
#             self.decoder = Decoder3D(
#                 dim = decoder_base_dim,
#                 z_dim = self.z_dim,
#                 dim_mult = config.dim_mult,
#                 num_res_blocks = config.num_res_blocks,
#                 attn_scales = config.attn_scales,
#                 temperal_upsample = self.temperal_upsample,
#                 dropout = config.dropout,
#                 out_channels = config.out_channels,
#                 is_residual = config.is_residual,
#             )
#
#         self.use_feature_cache = config.use_feature_cache
