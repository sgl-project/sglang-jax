"""
LTX-2 VideoDecoder ported from PyTorch to JAX/Flax.

This module implements the decoder portion of the LTX-2 Video VAE.
It decodes latent representations into video frames.

Reference: /Users/chandrasekhardevarakonda/Downloads/ltx/LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py
"""

import logging
import math
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.lax import Precision

logger = logging.getLogger(__name__)


class PixelNorm(nnx.Module):
    """
    Per-pixel (per-location) RMS normalization layer.

    For each element along the channel dimension, normalizes by the RMS.
    Corresponds to PyTorch implementation in ltx_core.model.common.normalization.PixelNorm
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C] (channel-last JAX format)
        # Compute RMS along channel dimension
        mean_sq = jnp.mean(x**2, axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_sq + self.eps)
        return x / rms


class CausalConv3d(nnx.Module):
    """
    Causal 3D convolution that supports causal temporal padding.

    When causal=True, pads the temporal dimension to prevent future frame leakage.
    When causal=False, uses symmetric padding (allows future frame dependencies).

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size (int or tuple)
        stride: Stride (int or tuple)
        padding: Spatial padding (applied to H,W dimensions)
        causal: Whether to use causal convolution (True) or support this mode
        spatial_padding_mode: 'zeros' or 'reflect' for spatial padding
        rngs: Flax RNG state
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int, int] = 3,
        *,
        stride: int | Tuple[int, int, int] = 1,
        padding: int = 0,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
        rngs: nnx.Rngs,
    ):
        # Convert scalar to tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.time_kernel_size = kernel_size[0]
        self.stride = stride
        self.spatial_padding = padding
        self.spatial_padding_mode = spatial_padding_mode
        self.support_causal = causal

        # JAX Conv expects kernel_size in format (T, H, W) for 3D
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",  # We handle padding manually
            rngs=rngs,
            precision=Precision.HIGHEST,
        )

    def __call__(self, x: Array, causal: bool = True) -> Array:
        """
        Forward pass with causal or symmetric temporal padding.

        Args:
            x: Input [B, T, H, W, C] (channel-last JAX format)
            causal: If True, use causal padding. If False, use symmetric padding.

        Returns:
            Output [B, T', H', W', C']
        """
        # Temporal padding
        if causal:
            # Causal: pad left by (kernel_size - 1), right by 0
            # Duplicate first frame for padding
            first_frame_pad = jnp.repeat(x[:, :1, :, :, :], self.time_kernel_size - 1, axis=1)
            x = jnp.concatenate([first_frame_pad, x], axis=1)
        else:
            # Symmetric: pad both sides
            left_pad = (self.time_kernel_size - 1) // 2
            right_pad = (self.time_kernel_size - 1) // 2
            first_frame_pad = jnp.repeat(x[:, :1, :, :, :], left_pad, axis=1)
            last_frame_pad = jnp.repeat(x[:, -1:, :, :, :], right_pad, axis=1)
            x = jnp.concatenate([first_frame_pad, x, last_frame_pad], axis=1)

        # Spatial padding (H, W dimensions)
        if self.spatial_padding > 0:
            pad_width = (
                (0, 0),  # batch
                (0, 0),  # time
                (self.spatial_padding, self.spatial_padding),  # height
                (self.spatial_padding, self.spatial_padding),  # width
                (0, 0),  # channels
            )
            mode = "constant" if self.spatial_padding_mode == "zeros" else self.spatial_padding_mode
            x = jnp.pad(x, pad_width, mode=mode)

        # Apply convolution
        return self.conv(x)


def make_conv_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    causal: bool = False,
    spatial_padding_mode: str = "zeros",
    rngs: nnx.Rngs,
) -> nnx.Module:
    """
    Create a convolution layer (2D or 3D).

    Args:
        dims: 2 for Conv2D, 3 for Conv3D
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
        causal: Whether to use causal convolution (only for 3D)
        spatial_padding_mode: Padding mode for spatial dimensions
        rngs: Flax RNG state

    Returns:
        Convolution module
    """
    if dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                causal=True,
                spatial_padding_mode=spatial_padding_mode,
                rngs=rngs,
            )
        else:
            # Non-causal 3D conv
            return nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                strides=(stride, stride, stride),
                padding=padding,
                rngs=rngs,
                precision=Precision.HIGHEST,
            )
    elif dims == 2:
        return nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding=padding,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


def make_linear_nd(dims: int, in_channels: int, out_channels: int, *, rngs: nnx.Rngs) -> nnx.Module:
    """Create a 1x1 convolution (linear projection)."""
    if dims == 3:
        return nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(1, 1, 1),
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
    elif dims == 2:
        return nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(1, 1),
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


class ResnetBlock3D(nnx.Module):
    """
    3D Residual block for LTX-2 VideoDecoder.

    Follows the implementation in ltx_core.model.video_vae.resnet.ResnetBlock3D
    Uses two 3D convolutions with normalization and SiLU activation.

    Args:
        dims: Convolution dimensions (3 for 3D)
        in_channels: Input channels
        out_channels: Output channels (if None, same as in_channels)
        eps: Epsilon for normalization
        groups: Number of groups for GroupNorm (if using GroupNorm)
        norm_layer: 'pixel_norm' or 'group_norm'
        inject_noise: Whether to inject noise (not implemented yet)
        timestep_conditioning: Whether to use timestep conditioning (not implemented yet)
        spatial_padding_mode: Padding mode for spatial dimensions
        rngs: Flax RNG state
    """

    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        eps: float = 1e-6,
        groups: int = 32,
        norm_layer: str = "pixel_norm",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise
        self.timestep_conditioning = timestep_conditioning

        # Normalization layers
        if norm_layer == "group_norm":
            # GroupNorm: normalize over groups of channels
            # JAX doesn't have built-in GroupNorm, we'll implement it
            self.norm1 = GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, rngs=rngs)
            self.norm2 = GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, rngs=rngs)
        elif norm_layer == "pixel_norm":
            self.norm1 = PixelNorm(eps=eps)
            self.norm2 = PixelNorm(eps=eps)
        else:
            raise ValueError(f"Unknown norm_layer: {norm_layer}")

        # Convolution layers
        self.conv1 = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
            rngs=rngs,
        )

        self.conv2 = make_conv_nd(
            dims=dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
            rngs=rngs,
        )

        # Skip connection
        if in_channels != out_channels:
            self.conv_shortcut = make_linear_nd(dims=dims, in_channels=in_channels, out_channels=out_channels, rngs=rngs)
            # GroupNorm with 1 group is equivalent to LayerNorm
            self.norm3 = GroupNorm(num_groups=1, num_channels=in_channels, eps=eps, rngs=rngs)
        else:
            self.conv_shortcut = None
            self.norm3 = None

    def __call__(self, x: Array, causal: bool = True, generator: Optional[Any] = None) -> Array:
        """
        Forward pass.

        Args:
            x: Input [B, T, H, W, C]
            causal: Whether to use causal convolution
            generator: Random generator (for noise injection, not implemented)

        Returns:
            Output [B, T, H, W, C']
        """
        residual = x

        # Main path
        h = self.norm1(x)
        h = nnx.silu(h)
        h = self.conv1(h, causal=causal) if isinstance(self.conv1, CausalConv3d) else self.conv1(h)

        h = self.norm2(h)
        h = nnx.silu(h)
        h = self.conv2(h, causal=causal) if isinstance(self.conv2, CausalConv3d) else self.conv2(h)

        # Skip connection
        if self.conv_shortcut is not None:
            residual = self.norm3(residual)
            residual = self.conv_shortcut(residual)

        return residual + h


class GroupNorm(nnx.Module):
    """
    Group Normalization for JAX.

    Normalizes over groups of channels. Each group is normalized independently.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        # Learnable affine parameters
        self.weight = nnx.Param(jnp.ones((num_channels,)))
        self.bias = nnx.Param(jnp.zeros((num_channels,)))

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C] or [B, H, W, C]
        orig_shape = x.shape
        batch_size = orig_shape[0]
        num_channels = orig_shape[-1]

        if num_channels != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {num_channels}")

        # Reshape to [B, T*H*W, C] or [B, H*W, C]
        spatial_dims = orig_shape[1:-1]
        spatial_size = math.prod(spatial_dims)
        x = x.reshape(batch_size, spatial_size, num_channels)

        # Reshape to [B, num_groups, group_size, spatial]
        group_size = num_channels // self.num_groups
        x = x.reshape(batch_size, spatial_size, self.num_groups, group_size)
        x = x.transpose(0, 2, 3, 1)  # [B, num_groups, group_size, spatial]

        # Normalize over group_size and spatial dimensions
        mean = jnp.mean(x, axis=(2, 3), keepdims=True)
        var = jnp.var(x, axis=(2, 3), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)

        # Reshape back
        x = x.transpose(0, 3, 1, 2)  # [B, spatial, num_groups, group_size]
        x = x.reshape(batch_size, spatial_size, num_channels)
        x = x.reshape(batch_size, *spatial_dims, num_channels)

        # Apply affine transformation
        return x * self.weight.value + self.bias.value


class UNetMidBlock3D(nnx.Module):
    """
    UNet middle block with multiple residual blocks.

    Corresponds to ltx_core.model.video_vae.resnet.UNetMidBlock3D

    Args:
        dims: Convolution dimensions (3 for 3D)
        in_channels: Input channels
        num_layers: Number of residual blocks
        resnet_eps: Epsilon for residual blocks
        resnet_groups: Number of groups for GroupNorm
        norm_layer: 'pixel_norm' or 'group_norm'
        inject_noise: Whether to inject noise
        timestep_conditioning: Whether to use timestep conditioning
        spatial_padding_mode: Padding mode for spatial dimensions
        rngs: Flax RNG state
    """

    def __init__(
        self,
        dims: int,
        in_channels: int,
        *,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: str = "pixel_norm",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
        rngs: nnx.Rngs,
    ):
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # Create residual blocks
        self.resnets = []
        for _ in range(num_layers):
            resnet = ResnetBlock3D(
                dims=dims,
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                norm_layer=norm_layer,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
                rngs=rngs,
            )
            self.resnets.append(resnet)

    def __call__(self, x: Array, causal: bool = True, timestep: Optional[Array] = None, generator: Optional[Any] = None) -> Array:
        """
        Forward pass.

        Args:
            x: Input [B, T, H, W, C]
            causal: Whether to use causal convolution
            timestep: Timestep for conditioning (not implemented)
            generator: Random generator (not implemented)

        Returns:
            Output [B, T, H, W, C]
        """
        for resnet in self.resnets:
            x = resnet(x, causal=causal, generator=generator)
        return x


class DepthToSpaceUpsample(nnx.Module):
    """
    Depth-to-space upsampling for LTX-2 VideoDecoder.

    Corresponds to ltx_core.model.video_vae.sampling.DepthToSpaceUpsample

    This operation rearranges channels into spatial/temporal dimensions.
    For example, with stride=(2, 2, 2) and in_channels=512:
    - Input: [B, T, H, W, 512]
    - After conv: [B, T, H, W, 512*8=4096]
    - After rearrange: [B, T*2, H*2, W*2, 512]

    Args:
        dims: Convolution dimensions (3 for 3D)
        in_channels: Input channels
        stride: Upsampling stride (T_stride, H_stride, W_stride)
        residual: Whether to add residual connection
        out_channels_reduction_factor: Factor to reduce output channels
        spatial_padding_mode: Padding mode for spatial dimensions
        rngs: Flax RNG state
    """

    def __init__(
        self,
        dims: int,
        in_channels: int,
        *,
        stride: Tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: str = "zeros",
        rngs: nnx.Rngs,
    ):
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
            rngs=rngs,
        )

    def __call__(self, x: Array, causal: bool = True) -> Array:
        """
        Forward pass.

        Args:
            x: Input [B, T, H, W, C]
            causal: Whether to use causal convolution

        Returns:
            Output [B, T*stride[0], H*stride[1], W*stride[2], C']
        """
        # Residual path (if enabled)
        if self.residual:
            # Rearrange input for residual
            b, t, h, w, c = x.shape
            # Reshape: [B, T, H, W, C] -> [B, T, H, W, C/(p1*p2*p3), p1, p2, p3]
            x_in = x.reshape(b, t, h, w, c // math.prod(self.stride), *self.stride)
            # Permute to get upsampling pattern
            x_in = x_in.transpose(0, 1, 5, 2, 6, 3, 7, 4)
            x_in = x_in.reshape(b, t * self.stride[0], h * self.stride[1], w * self.stride[2], c // math.prod(self.stride))

            # Repeat to match output channels
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            x_in = jnp.repeat(x_in, num_repeat, axis=-1)

            # Remove first frame if temporal upsampling
            if self.stride[0] == 2:
                x_in = x_in[:, 1:, :, :, :]

        # Main path
        x = self.conv(x, causal=causal) if isinstance(self.conv, CausalConv3d) else self.conv(x)

        # Depth-to-space rearrangement
        b, t, h, w, c = x.shape
        # Reshape: [B, T, H, W, C_out] where C_out = C' * p1 * p2 * p3
        # Goal: [B, T*p1, H*p2, W*p3, C']
        output_channels = c // math.prod(self.stride)
        x = x.reshape(b, t, h, w, output_channels, *self.stride)
        # Permute: [B, T, H, W, C', p1, p2, p3] -> [B, T, p1, H, p2, W, p3, C']
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(b, t * self.stride[0], h * self.stride[1], w * self.stride[2], output_channels)

        # Remove first frame if temporal upsampling
        if self.stride[0] == 2:
            x = x[:, 1:, :, :, :]

        # Add residual if enabled
        if self.residual:
            x = x + x_in

        return x


class PerChannelStatistics(nnx.Module):
    """
    Per-channel statistics for normalizing and denormalizing latents.

    Stores mean and std for each channel, computed over the training dataset.
    """

    def __init__(self, latent_channels: int = 128, *, rngs: nnx.Rngs):
        # Initialize buffers (will be loaded from checkpoint)
        self.std_of_means = nnx.Variable(jnp.ones((latent_channels,)))
        self.mean_of_means = nnx.Variable(jnp.zeros((latent_channels,)))

    def un_normalize(self, x: Array) -> Array:
        """Denormalize latents: x_denorm = x * std + mean"""
        # x: [B, T, H, W, C]
        std = self.std_of_means.value.reshape(1, 1, 1, 1, -1)
        mean = self.mean_of_means.value.reshape(1, 1, 1, 1, -1)
        return x * std + mean

    def normalize(self, x: Array) -> Array:
        """Normalize latents: x_norm = (x - mean) / std"""
        # x: [B, T, H, W, C]
        std = self.std_of_means.value.reshape(1, 1, 1, 1, -1)
        mean = self.mean_of_means.value.reshape(1, 1, 1, 1, -1)
        return (x - mean) / std


def unpatchify(x: Array, patch_size_hw: int, patch_size_t: int = 1) -> Array:
    """
    Rearrange channels back into spatial dimensions (inverse of patchify).

    Moves pixels from channels back into patch_size x patch_size blocks (depth-to-space).

    Args:
        x: Input [B, T, H, W, C*(patch_size_hw^2)*(patch_size_t)]
        patch_size_hw: Spatial patch size
        patch_size_t: Temporal patch size

    Returns:
        Output [B, T*patch_size_t, H*patch_size_hw, W*patch_size_hw, C]
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    b, t, h, w, c = x.shape

    # Calculate output channels
    out_channels = c // (patch_size_hw**2 * patch_size_t)

    # Reshape: [B, T, H, W, C] -> [B, T, H, W, out_C, patch_t, patch_h, patch_w]
    x = x.reshape(b, t, h, w, out_channels, patch_size_t, patch_size_hw, patch_size_hw)

    # Permute: [B, T, H, W, out_C, patch_t, patch_h, patch_w] -> [B, T, patch_t, H, patch_h, W, patch_w, out_C]
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)

    # Reshape: [B, T*patch_t, H*patch_h, W*patch_w, out_C]
    x = x.reshape(b, t * patch_size_t, h * patch_size_hw, w * patch_size_hw, out_channels)

    return x


class VideoDecoder(nnx.Module):
    """
    LTX-2 VideoDecoder ported to JAX/Flax.

    Decodes latent representations into video frames. The decoder upsamples latents
    through a series of upsampling operations (inverse of encoder).

    Standard LTX-2 configuration:
    - Input: [B, T', H', W', 128] latent (e.g., [1, 5, 16, 16, 128])
    - Output: [B, T, H, W, 3] video (e.g., [1, 33, 512, 512, 3])
    - Upsampling factors: T'->T (8x), H'->H (32x), W'->W (32x)

    Args:
        convolution_dimensions: Number of dimensions (3 for 3D)
        in_channels: Input latent channels (default 128)
        out_channels: Output video channels (default 3 for RGB)
        decoder_blocks: List of (block_name, block_config) tuples
        patch_size: Final spatial expansion factor (default 4)
        norm_layer: 'pixel_norm' or 'group_norm'
        causal: Whether to use causal convolutions (default False for LTX-2)
        timestep_conditioning: Whether to use timestep conditioning
        decoder_spatial_padding_mode: Padding mode for spatial dimensions
        rngs: Flax RNG state
    """

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        *,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: list[Tuple[str, int | dict]] = None,
        patch_size: int = 4,
        norm_layer: str = "pixel_norm",
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: str = "reflect",
        rngs: nnx.Rngs,
    ):
        if decoder_blocks is None:
            # Default LTX-2 decoder blocks (in reverse order from encoder)
            decoder_blocks = [
                ("compress_all", {"residual": True, "multiplier": 1}),
                ("res_x", {"num_layers": 3}),
                ("compress_all", {"residual": True, "multiplier": 1}),
                ("res_x", {"num_layers": 3}),
                ("compress_time", {}),
                ("res_x", {"num_layers": 3}),
                ("compress_space", {}),
                ("res_x", {"num_layers": 3}),
            ]

        self.patch_size = patch_size
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels, rngs=rngs)

        # Compute initial feature channels by going through blocks in reverse
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            elif block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)

        # Input convolution
        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
            rngs=rngs,
        )

        # Upsampling blocks
        self.up_blocks = []
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = self._make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
                rngs=rngs,
            )
            self.up_blocks.append(block)

        # Output normalization
        if norm_layer == "group_norm":
            self.conv_norm_out = GroupNorm(num_groups=self._norm_num_groups, num_channels=feature_channels, rngs=rngs)
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        else:
            raise ValueError(f"Unknown norm_layer: {norm_layer}")

        # Output convolution
        out_channels_with_patch = out_channels * patch_size**2
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels_with_patch,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
            rngs=rngs,
        )

    def _make_decoder_block(
        self,
        block_name: str,
        block_config: dict,
        in_channels: int,
        convolution_dimensions: int,
        norm_layer: str,
        timestep_conditioning: bool,
        norm_num_groups: int,
        spatial_padding_mode: str,
        rngs: nnx.Rngs,
    ) -> Tuple[nnx.Module, int]:
        """Create a decoder block based on the block name."""
        out_channels = in_channels

        if block_name == "res_x":
            block = UNetMidBlock3D(
                dims=convolution_dimensions,
                in_channels=in_channels,
                num_layers=block_config["num_layers"],
                resnet_eps=1e-6,
                resnet_groups=norm_num_groups,
                norm_layer=norm_layer,
                inject_noise=block_config.get("inject_noise", False),
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
                rngs=rngs,
            )
        elif block_name == "res_x_y":
            out_channels = in_channels // block_config.get("multiplier", 2)
            block = ResnetBlock3D(
                dims=convolution_dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                eps=1e-6,
                groups=norm_num_groups,
                norm_layer=norm_layer,
                inject_noise=block_config.get("inject_noise", False),
                timestep_conditioning=False,
                spatial_padding_mode=spatial_padding_mode,
                rngs=rngs,
            )
        elif block_name == "compress_time":
            block = DepthToSpaceUpsample(
                dims=convolution_dimensions,
                in_channels=in_channels,
                stride=(2, 1, 1),
                spatial_padding_mode=spatial_padding_mode,
                rngs=rngs,
            )
        elif block_name == "compress_space":
            block = DepthToSpaceUpsample(
                dims=convolution_dimensions,
                in_channels=in_channels,
                stride=(1, 2, 2),
                spatial_padding_mode=spatial_padding_mode,
                rngs=rngs,
            )
        elif block_name == "compress_all":
            out_channels = in_channels // block_config.get("multiplier", 1)
            block = DepthToSpaceUpsample(
                dims=convolution_dimensions,
                in_channels=in_channels,
                stride=(2, 2, 2),
                residual=block_config.get("residual", False),
                out_channels_reduction_factor=block_config.get("multiplier", 1),
                spatial_padding_mode=spatial_padding_mode,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unknown block: {block_name}")

        return block, out_channels

    def __call__(
        self,
        sample: Array,
        timestep: Optional[Array] = None,
        generator: Optional[Any] = None,
    ) -> Array:
        """
        Decode latent representation into video frames.

        Args:
            sample: Latent tensor [B, T', H', W', C] (e.g., [1, 5, 16, 16, 128])
            timestep: Timestep for conditioning (if timestep_conditioning=True)
            generator: Random generator (for noise injection, not implemented)

        Returns:
            Decoded video [B, T, H, W, 3] (e.g., [1, 33, 512, 512, 3])
            Note: First frame is removed after temporal upsampling.
        """
        # Denormalize latents
        sample = self.per_channel_statistics.un_normalize(sample)

        # Input convolution
        sample = self.conv_in(sample, causal=self.causal) if isinstance(self.conv_in, CausalConv3d) else self.conv_in(sample)

        # Upsampling blocks
        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                sample = up_block(sample, causal=self.causal, timestep=timestep, generator=generator)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            elif isinstance(up_block, DepthToSpaceUpsample):
                sample = up_block(sample, causal=self.causal)
            else:
                # Fallback for other block types
                sample = up_block(sample)

        # Output normalization and activation
        sample = self.conv_norm_out(sample)
        sample = nnx.silu(sample)

        # Output convolution
        sample = self.conv_out(sample, causal=self.causal) if isinstance(self.conv_out, CausalConv3d) else self.conv_out(sample)

        # Unpatchify: move channels back to spatial dimensions
        # [B, T, H, W, C*patch_size^2] -> [B, T, H*patch_size, W*patch_size, C]
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample
