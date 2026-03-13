"""
LTX-2 Audio VAE: Encoder and Decoder for audio spectrograms.

Ported from PyTorch reference at:
  ltx_core/model/audio_vae/audio_vae.py

The Audio VAE uses 2D causal convolutions on spectrograms [B, C, H, W] where:
  - H = time axis (causal dimension with causality_axis=HEIGHT)
  - W = frequency axis (mel bins)
  - C = channels (2 for stereo)

JAX format is channel-last: [B, H, W, C]

Architecture:
  - Encoder: conv_in → downsample blocks → mid block → conv_out → normalize
  - Decoder: denormalize → conv_in → mid block → upsample blocks → conv_out
  - PixelNorm instead of GroupNorm (for causal compat)
  - ResnetBlock with causal 2D convolutions
  - AttnBlock with 1x1 conv projections (no attention in default config)
  - Downsample: strided conv with causal padding
  - Upsample: nearest interpolation + causal conv + drop first element

Config (from checkpoint):
  ch=128, ch_mult=(1,2,4), num_res_blocks=2, z_channels=8, double_z=True,
  in_channels=2 (stereo), out_ch=2, resolution=256, norm_type=pixel,
  causality_axis=height, attn_resolutions={} (no attention at 256→128→64→32)
"""

import logging
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.lax import Precision

logger = logging.getLogger(__name__)

LATENT_DOWNSAMPLE_FACTOR = 4


class AudioLatentShape(NamedTuple):
    batch: int
    channels: int
    frames: int
    mel_bins: int


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class PixelNorm2D(nnx.Module):
    """Per-pixel RMS normalization for 2D feature maps (channel-last)."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        # x: [B, H, W, C]
        return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)


class PerChannelStatistics2D(nnx.Module):
    """Per-channel mean/std for normalizing and denormalizing 2D latents."""

    def __init__(self, latent_channels: int = 128, *, rngs: nnx.Rngs):
        self.std_of_means = nnx.Variable(jnp.ones((latent_channels,)))
        self.mean_of_means = nnx.Variable(jnp.zeros((latent_channels,)))

    def normalize(self, x: Array) -> Array:
        # x: [B, T, C] (patchified)
        std = self.std_of_means.value
        mean = self.mean_of_means.value
        return (x - mean) / std

    def un_normalize(self, x: Array) -> Array:
        std = self.std_of_means.value
        mean = self.mean_of_means.value
        return x * std + mean


# ---------------------------------------------------------------------------
# Causal Conv2D
# ---------------------------------------------------------------------------


class CausalConv2d(nnx.Module):
    """2D convolution with causal padding along the height (time) axis.

    Causality axis = HEIGHT means:
      - H padding: all before (top), none after (bottom)
      - W padding: symmetric

    PyTorch format [B, C, H, W] → JAX format [B, H, W, C]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.dilation = dilation

        # Causal height padding: all before, symmetric width
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]
        # For JAX channel-last [B, H, W, C]:
        # pad_width = ((batch), (H_top, H_bottom), (W_left, W_right), (C))
        self.pad_width = (
            (0, 0),                          # batch
            (pad_h, 0),                      # height: causal (all before)
            (pad_w // 2, pad_w - pad_w // 2),  # width: symmetric
            (0, 0),                          # channels
        )

        if isinstance(stride, int):
            stride = (stride, stride)

        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",
            kernel_dilation=dilation,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )

    def __call__(self, x: Array) -> Array:
        x = jnp.pad(x, self.pad_width, mode="constant", constant_values=0)
        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    *,
    rngs: nnx.Rngs,
) -> CausalConv2d:
    """Create a causal 2D conv (causality_axis=HEIGHT always for audio VAE)."""
    return CausalConv2d(
        in_channels, out_channels, kernel_size, stride, rngs=rngs,
    )


def make_conv2d_1x1(
    in_channels: int,
    out_channels: int,
    *,
    rngs: nnx.Rngs,
) -> nnx.Conv:
    """Create a 1x1 convolution (linear projection, no padding needed)."""
    return nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=(1, 1),
        rngs=rngs,
        precision=Precision.HIGHEST,
    )


# ---------------------------------------------------------------------------
# ResnetBlock (2D, causal)
# ---------------------------------------------------------------------------


class ResnetBlock(nnx.Module):
    """Residual block with causal 2D convolutions and PixelNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = PixelNorm2D()
        self.conv1 = make_conv2d(in_channels, out_channels, 3, rngs=rngs)
        self.norm2 = PixelNorm2D()
        self.conv2 = make_conv2d(out_channels, out_channels, 3, rngs=rngs)

        if in_channels != out_channels:
            self.nin_shortcut = make_conv2d_1x1(in_channels, out_channels, rngs=rngs)
        else:
            self.nin_shortcut = None

    def __call__(self, x: Array) -> Array:
        h = self.norm1(x)
        h = nnx.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nnx.silu(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


# ---------------------------------------------------------------------------
# AttnBlock (no attention in default config, but included for completeness)
# ---------------------------------------------------------------------------


class AttnBlock(nnx.Module):
    """Self-attention block for 2D feature maps with 1x1 conv projections."""

    def __init__(self, in_channels: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.norm = PixelNorm2D()
        self.q = make_conv2d_1x1(in_channels, in_channels, rngs=rngs)
        self.k = make_conv2d_1x1(in_channels, in_channels, rngs=rngs)
        self.v = make_conv2d_1x1(in_channels, in_channels, rngs=rngs)
        self.proj_out = make_conv2d_1x1(in_channels, in_channels, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        h = self.norm(x)
        q = self.q(h)  # [B, H, W, C]
        k = self.k(h)
        v = self.v(h)

        b, h_dim, w_dim, c = q.shape
        hw = h_dim * w_dim

        q = q.reshape(b, hw, c)  # [B, HW, C]
        k = k.reshape(b, hw, c)  # [B, HW, C]
        v = v.reshape(b, hw, c)  # [B, HW, C]

        # Attention: [B, HW, HW]
        scale = c ** (-0.5)
        attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) * scale
        attn_weights = nnx.softmax(attn_weights, axis=-1)

        # Attend to values
        h_out = jnp.matmul(attn_weights, v)  # [B, HW, C]
        h_out = h_out.reshape(b, h_dim, w_dim, c)
        h_out = self.proj_out(h_out)

        return x + h_out


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------


class Downsample2D(nnx.Module):
    """Strided conv2d downsampling with causal height padding."""

    def __init__(self, in_channels: int, *, rngs: nnx.Rngs):
        # Causal height: pad (2, 0) top/bottom, (0, 1) left/right
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
        # Causal height padding for stride=2: pad_top=2, pad_bottom=0, pad_left=0, pad_right=1
        self.pad_width = (
            (0, 0),  # batch
            (2, 0),  # height (causal)
            (0, 1),  # width
            (0, 0),  # channels
        )

    def __call__(self, x: Array) -> Array:
        x = jnp.pad(x, self.pad_width, mode="constant", constant_values=0)
        return self.conv(x)


class Upsample2D(nnx.Module):
    """Nearest-neighbor 2x upsample + causal conv, drops first height element."""

    def __init__(self, in_channels: int, *, rngs: nnx.Rngs):
        self.conv = make_conv2d(in_channels, in_channels, 3, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        b, h, w, c = x.shape
        # Nearest-neighbor 2x upsampling
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        x = self.conv(x)
        # Drop first element along causal axis (height) to undo encoder padding
        x = x[:, 1:, :, :]
        return x


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------


class AudioEncoder(nnx.Module):
    """Encodes stereo mel spectrograms into latent representations.

    Input:  [B, H, W, 2]  (time, freq, stereo channels) - JAX channel-last
    Output: [B, H', W', z_channels]  latent

    Default config: ch=128, ch_mult=(1,2,4), num_res_blocks=2,
                    z_channels=8, double_z=True, in_channels=2
    """

    def __init__(
        self,
        config=None,
        *,
        dtype=None,
        mesh=None,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 8,
        double_z: bool = True,
        in_channels: int = 2,
        resolution: int = 256,
        attn_resolutions: set[int] | None = None,
        rngs: nnx.Rngs = None,
    ):
        self.mesh = mesh
        self.dtype = dtype

        if config is not None:
            ch = getattr(config, "ch", ch)
            ch_mult = tuple(getattr(config, "ch_mult", ch_mult))
            num_res_blocks = getattr(config, "num_res_blocks", num_res_blocks)
            z_channels = getattr(config, "z_channels", z_channels)
            double_z = getattr(config, "double_z", double_z)
            in_channels = getattr(config, "in_channels", in_channels)
            resolution = getattr(config, "resolution", resolution)
            attn_resolutions = getattr(config, "attn_resolutions", attn_resolutions)

        if rngs is None:
            rngs = nnx.Rngs(0)
        if attn_resolutions is None:
            attn_resolutions = set()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.double_z = double_z

        # Per-channel statistics for normalizing latents
        self.per_channel_statistics = PerChannelStatistics2D(latent_channels=ch, rngs=rngs)

        # Input conv
        self.conv_in = make_conv2d(in_channels, ch, 3, rngs=rngs)

        # Downsampling path
        self.down = []
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch
        curr_res = resolution

        for i_level in range(self.num_resolutions):
            stage_blocks = []
            stage_attns = []
            block_in_level = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks):
                stage_blocks.append(ResnetBlock(block_in_level, block_out, rngs=rngs))
                block_in_level = block_out
                if curr_res in attn_resolutions:
                    stage_attns.append(AttnBlock(block_in_level, rngs=rngs))

            downsample = None
            if i_level != self.num_resolutions - 1:
                downsample = Downsample2D(block_in_level, rngs=rngs)
                curr_res = curr_res // 2

            self.down.append((tuple(stage_blocks), tuple(stage_attns), downsample))
            block_in = block_in_level

        self.down = tuple(self.down)

        # Mid block
        self.mid_block_1 = ResnetBlock(block_in, block_in, rngs=rngs)
        self.mid_block_2 = ResnetBlock(block_in, block_in, rngs=rngs)

        # Output
        self.norm_out = PixelNorm2D()
        out_channels = 2 * z_channels if double_z else z_channels
        self.conv_out = make_conv2d(block_in, out_channels, 3, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Encode spectrogram [B, H, W, C] → latent [B, H', W', z_channels]."""
        h = self.conv_in(x)

        # Downsampling
        for i_level in range(self.num_resolutions):
            blocks, attns, downsample = self.down[i_level]
            for j, block in enumerate(blocks):
                h = block(h)
                if j < len(attns):
                    h = attns[j](h)
            if downsample is not None:
                h = downsample(h)

        # Mid block
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)

        # Output
        h = self.norm_out(h)
        h = nnx.silu(h)
        h = self.conv_out(h)

        # Normalize: take first half of channels (mean), patchify, normalize, unpatchify
        return self._normalize_latents(h)

    def _normalize_latents(self, h: Array) -> Array:
        if self.double_z:
            # Split along channel dim, take first half (mean)
            half = h.shape[-1] // 2
            means = h[..., :half]
        else:
            means = h
        # Patchify: [B, T, F, C] → [B, T, C*F] with C-slow, F-fast ordering
        # PyTorch patchifies as rearrange("b c t f -> b t (c f)") giving C-slow, F-fast.
        # Our input is [B, T, F, C] (channel-last), so swap to [B, T, C, F] first.
        b, t, f, c = means.shape
        means_cf = jnp.swapaxes(means, -1, -2)  # [B, T, C, F]
        patched = means_cf.reshape(b, t, c * f)  # [B, T, C*F] C-slow, F-fast ✓
        # Normalize
        patched = self.per_channel_statistics.normalize(patched)
        # Unpatchify: [B, T, C*F] → [B, T, C, F] → [B, T, F, C]
        result = patched.reshape(b, t, c, f)           # [B, T, C, F]
        return jnp.swapaxes(result, -1, -2)             # [B, T, F, C]

    def load_weights(self, model_config, *args, **kwargs):
        from jax import ShapeDtypeStruct
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.utils.weight_utils import WeightLoader
        from sgl_jax.srt.multimodal.models.ltx2.utils import get_ltx2_checkpoint_dir, cleanup_ltx2_checkpoint_dir
        from .weight_mappings import create_audio_vae_encoder_weight_mappings

        _ckpt_dir = get_ltx2_checkpoint_dir()
        if _ckpt_dir:
            model_config.model_path = _ckpt_dir

        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh,
            dtype=self.dtype or jnp.bfloat16,
        )
        loader.load_weights_from_safetensors(create_audio_vae_encoder_weight_mappings())
        cleanup_ltx2_checkpoint_dir(_ckpt_dir)

        def _replace_abstract(x):
            if isinstance(x, ShapeDtypeStruct):
                pspec = PartitionSpec()
                if self.mesh:
                    return jax.device_put(jnp.zeros(x.shape, x.dtype), NamedSharding(self.mesh, pspec))
                return jnp.zeros(x.shape, x.dtype)
            return x

        state = nnx.state(self)
        nnx.update(self, jax.tree_util.tree_map(_replace_abstract, state))
        logger.info("AudioEncoder weights loaded")

    @staticmethod
    def get_config_class():
        from .ltx2_audio_vae_config import LTX2AudioVAEEncoderConfig
        return LTX2AudioVAEEncoderConfig


# ---------------------------------------------------------------------------
# AudioDecoder
# ---------------------------------------------------------------------------


class AudioDecoder(nnx.Module):
    """Decodes latent representations back to stereo mel spectrograms.

    Input:  [B, H', W', z_channels]  latent
    Output: [B, H, W, 2]  stereo spectrogram (time, freq, channels)

    Default config: ch=128, out_ch=2, ch_mult=(1,2,4), num_res_blocks=2,
                    z_channels=8, resolution=256, causality_axis=height
    """

    def __init__(
        self,
        config=None,
        *,
        dtype=None,
        mesh=None,
        ch: int = 128,
        out_ch: int = 2,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 8,
        resolution: int = 256,
        attn_resolutions: set[int] | None = None,
        mel_bins: int | None = None,
        rngs: nnx.Rngs = None,
    ):
        self.mesh = mesh
        self.dtype = dtype

        if config is not None:
            ch = getattr(config, "ch", ch)
            out_ch = getattr(config, "out_ch", out_ch)
            ch_mult = tuple(getattr(config, "ch_mult", ch_mult))
            num_res_blocks = getattr(config, "num_res_blocks", num_res_blocks)
            z_channels = getattr(config, "z_channels", z_channels)
            resolution = getattr(config, "resolution", resolution)
            attn_resolutions = getattr(config, "attn_resolutions", attn_resolutions)
            mel_bins = getattr(config, "mel_bins", mel_bins)

        if rngs is None:
            rngs = nnx.Rngs(0)
        if attn_resolutions is None:
            attn_resolutions = set()

        self.ch = ch
        self.out_ch = out_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.mel_bins = mel_bins

        # Per-channel statistics
        self.per_channel_statistics = PerChannelStatistics2D(latent_channels=ch, rngs=rngs)

        # Base channels at deepest level
        base_block_channels = ch * ch_mult[-1]  # 128*4 = 512

        # Input conv: z_channels → base_block_channels
        self.conv_in = make_conv2d(z_channels, base_block_channels, 3, rngs=rngs)

        # Mid block
        self.mid_block_1 = ResnetBlock(base_block_channels, base_block_channels, rngs=rngs)
        self.mid_block_2 = ResnetBlock(base_block_channels, base_block_channels, rngs=rngs)

        # Upsampling path (reversed order)
        self.up = []
        block_in = base_block_channels
        curr_res = resolution // (2 ** (self.num_resolutions - 1))

        for level in reversed(range(self.num_resolutions)):
            stage_blocks = []
            stage_attns = []
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                stage_blocks.append(ResnetBlock(block_in, block_out, rngs=rngs))
                block_in = block_out
                if curr_res in attn_resolutions:
                    stage_attns.append(AttnBlock(block_in, rngs=rngs))

            upsample = None
            if level != 0:
                upsample = Upsample2D(block_in, rngs=rngs)
                curr_res *= 2

            # Insert at beginning to maintain level indexing
            self.up.insert(0, (tuple(stage_blocks), tuple(stage_attns), upsample))

        self.up = tuple(self.up)

        # Output
        self.norm_out = PixelNorm2D()
        self.conv_out = make_conv2d(block_in, out_ch, 3, rngs=rngs)

    def __call__(self, sample: Array) -> Array:
        """Decode latent [B, H', W', z_channels] → spectrogram [B, H, W, out_ch]."""
        sample, target_shape = self._denormalize_latents(sample)

        h = self.conv_in(sample)

        # Mid block
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)

        # Upsampling (iterate from deepest to shallowest)
        for level in reversed(range(self.num_resolutions)):
            blocks, attns, upsample = self.up[level]
            for j, block in enumerate(blocks):
                h = block(h)
                if j < len(attns):
                    h = attns[j](h)
            if upsample is not None:
                h = upsample(h)

        # Output
        h = self.norm_out(h)
        h = nnx.silu(h)
        h = self.conv_out(h)

        return self._adjust_output_shape(h, target_shape)

    def _denormalize_latents(self, sample: Array) -> tuple[Array, AudioLatentShape]:
        b, t, f, c = sample.shape
        latent_shape = AudioLatentShape(batch=b, channels=c, frames=t, mel_bins=f)

        # Patchify → denormalize → unpatchify
        # PyTorch patchifies as rearrange("b c t f -> b t (c f)") giving C-slow, F-fast.
        # Our input is [B, T, F, C] (channel-last), so swap to [B, T, C, F] first
        # to get the same C-slow, F-fast ordering when flattened.
        sample_cf = jnp.swapaxes(sample, -1, -2)  # [B, T, C, F]
        patched = sample_cf.reshape(b, t, c * f)   # [B, T, C*F] C-slow, F-fast ✓
        patched = self.per_channel_statistics.un_normalize(patched)
        # Unpatchify back: [B, T, C*F] → [B, T, C, F] → [B, T, F, C]
        sample = patched.reshape(b, t, c, f)            # [B, T, C, F]
        sample = jnp.swapaxes(sample, -1, -2)           # [B, T, F, C]

        # Target output shape
        target_frames = t * LATENT_DOWNSAMPLE_FACTOR
        # Causal: subtract padding frames
        target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=b,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=self.mel_bins if self.mel_bins is not None else f,
        )
        return sample, target_shape

    def _adjust_output_shape(self, h: Array, target_shape: AudioLatentShape) -> Array:
        """Crop/pad output to match target dimensions."""
        # h: [B, H, W, C] in channel-last
        _, cur_h, cur_w, _ = h.shape
        target_h = target_shape.frames
        target_w = target_shape.mel_bins
        target_c = target_shape.channels

        # Crop
        h = h[:, :min(cur_h, target_h), :min(cur_w, target_w), :target_c]

        # Pad if needed
        pad_h = target_h - h.shape[1]
        pad_w = target_w - h.shape[2]
        if pad_h > 0 or pad_w > 0:
            h = jnp.pad(h, ((0, 0), (0, max(pad_h, 0)), (0, max(pad_w, 0)), (0, 0)))

        # Final crop for safety
        return h[:, :target_h, :target_w, :target_c]

    def decode(self, x: Array) -> Array:
        return self(x)

    def load_weights(self, model_config, *args, **kwargs):
        from jax import ShapeDtypeStruct
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.utils.weight_utils import WeightLoader
        from sgl_jax.srt.multimodal.models.ltx2.utils import get_ltx2_checkpoint_dir, cleanup_ltx2_checkpoint_dir
        from .weight_mappings import create_audio_vae_decoder_weight_mappings

        _ckpt_dir = get_ltx2_checkpoint_dir()
        if _ckpt_dir:
            model_config.model_path = _ckpt_dir

        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh,
            dtype=self.dtype or jnp.float32,
        )
        loader.load_weights_from_safetensors(create_audio_vae_decoder_weight_mappings())
        cleanup_ltx2_checkpoint_dir(_ckpt_dir)

        def _replace_abstract(x):
            if isinstance(x, ShapeDtypeStruct):
                pspec = PartitionSpec()
                if self.mesh:
                    return jax.device_put(jnp.zeros(x.shape, x.dtype), NamedSharding(self.mesh, pspec))
                return jnp.zeros(x.shape, x.dtype)
            return x

        state = nnx.state(self)
        nnx.update(self, jax.tree_util.tree_map(_replace_abstract, state))
        logger.info("AudioDecoder weights loaded")

    @staticmethod
    def get_config_class():
        from .ltx2_audio_vae_config import LTX2AudioVAEDecoderConfig
        return LTX2AudioVAEDecoderConfig
