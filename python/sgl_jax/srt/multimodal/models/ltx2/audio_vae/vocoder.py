"""
LTX-2 Vocoder: HiFi-GAN-based model that converts mel spectrograms to waveforms.

Ported from PyTorch reference at:
  ltx_core/model/audio_vae/vocoder.py

Architecture:
  - conv_pre: Conv1d(128, 1024, 7) - stereo input (2 channels × 64 mel_bins = 128)
  - 5 upsample stages with ConvTranspose1d
  - 3 ResBlock1 per stage (kernel_sizes=[3,7,11], dilations=[1,3,5])
  - conv_post: Conv1d(32, 2, 7) - stereo output
  - Total upsample factor: 6×5×2×2×2 = 240

All 1D convolutions use channel-last format in JAX: [B, T, C]
PyTorch Conv1d format: [B, C, T]
"""

import logging
import math
import os

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

logger = logging.getLogger(__name__)

LRELU_SLOPE = 0.1


class ResBlock1(nnx.Module):
    """HiFi-GAN ResBlock type 1: 3 dilated conv pairs with residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        *,
        rngs: nnx.Rngs,
    ):
        self.convs1 = []
        self.convs2 = []
        for d in dilation:
            self.convs1.append(
                nnx.Conv(
                    in_features=channels,
                    out_features=channels,
                    kernel_size=(kernel_size,),
                    kernel_dilation=(d,),
                    padding="SAME",
                    rngs=rngs,
                )
            )
            self.convs2.append(
                nnx.Conv(
                    in_features=channels,
                    out_features=channels,
                    kernel_size=(kernel_size,),
                    kernel_dilation=(1,),
                    padding="SAME",
                    rngs=rngs,
                )
            )

        self.convs1 = tuple(self.convs1)
        self.convs2 = tuple(self.convs2)

    def __call__(self, x: Array) -> Array:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = jax.nn.leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = jax.nn.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = xt + x
        return x


class ConvTranspose1d(nnx.Module):
    """1D transposed convolution (channel-last JAX format)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.stride = stride
        # Flax ConvTranspose padding != PyTorch ConvTranspose1d padding.
        # PyTorch clips output by 2*padding.
        # Flax clips output by 2*(kernel-1-padding), where padding is the
        # "forward conv" padding. To match PyTorch: flax_pad = kernel - 1 - pytorch_pad.
        flax_padding = kernel_size - 1 - padding
        self.conv_transpose = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            strides=(stride,),
            padding=((flax_padding, flax_padding),),
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.conv_transpose(x)


class Vocoder(nnx.Module):
    """HiFi-GAN vocoder that converts mel spectrograms to audio waveforms.

    Default config:
      resblock_kernel_sizes = [3, 7, 11]
      upsample_rates = [6, 5, 2, 2, 2]
      upsample_kernel_sizes = [16, 15, 8, 4, 4]
      resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
      upsample_initial_channel = 1024
      stereo = True
      output_sample_rate = 24000
    """

    def __init__(
        self,
        config=None,
        *,
        dtype=None,
        mesh=None,
        resblock_kernel_sizes: list[int] | None = None,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        upsample_initial_channel: int = 1024,
        stereo: bool = True,
        output_sample_rate: int = 24000,
        rngs: nnx.Rngs = None,
    ):
        self.mesh = mesh
        self.dtype = dtype

        if config is not None:
            resblock_kernel_sizes = getattr(config, "resblock_kernel_sizes", resblock_kernel_sizes)
            upsample_rates = getattr(config, "upsample_rates", upsample_rates)
            upsample_kernel_sizes = getattr(config, "upsample_kernel_sizes", upsample_kernel_sizes)
            resblock_dilation_sizes = getattr(config, "resblock_dilation_sizes", resblock_dilation_sizes)
            upsample_initial_channel = getattr(config, "upsample_initial_channel", upsample_initial_channel)
            stereo = getattr(config, "stereo", stereo)
            output_sample_rate = getattr(config, "output_sample_rate", output_sample_rate)

        if rngs is None:
            rngs = nnx.Rngs(0)
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [6, 5, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sample_rate = output_sample_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        in_channels = 128 if stereo else 64

        # Pre-conv: in_channels → upsample_initial_channel
        self.conv_pre = nnx.Conv(
            in_features=in_channels,
            out_features=upsample_initial_channel,
            kernel_size=(7,),
            padding=((3, 3),),
            rngs=rngs,
        )

        # Upsample layers
        self.ups = []
        for i, (stride, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                    rngs=rngs,
                )
            )
        self.ups = tuple(self.ups)

        # Residual blocks
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(
                    ResBlock1(ch, kernel_size, tuple(dilations), rngs=rngs)
                )
        self.resblocks = tuple(self.resblocks)

        # Post-conv
        out_channels = 2 if stereo else 1
        final_channels = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = nnx.Conv(
            in_features=final_channels,
            out_features=out_channels,
            kernel_size=(7,),
            padding=((3, 3),),
            rngs=rngs,
        )

        self.upsample_factor = math.prod(upsample_rates)

    def __call__(self, x: Array) -> Array:
        """Convert mel spectrogram to waveform.

        Args:
            x: [B, H, W, C] spectrogram in JAX channel-last format
               where H=time, W=mel_bins, C=2 (stereo)

        Returns:
            [B, audio_length, 2] stereo waveform
        """
        # x: [B, time, mel_bins, channels] = [B, T, 64, 2]
        # PyTorch does: (B,2,T,64) → transpose → (B,2,64,T) → rearrange("b s c t -> b (s c) t")
        # giving stereo-slow, mel-fast: [s0_f0..s0_f63, s1_f0..s1_f63]
        # For JAX channel-last, swap to [B, T, channels, mel_bins] then flatten:
        x = jnp.swapaxes(x, -1, -2)  # [B, T, channels, mel_bins] = [B, T, 2, 64]
        b, t, c, mel = x.shape
        x = x.reshape(b, t, c * mel)  # [B, T, 128] stereo-slow, mel-fast ✓

        # conv_pre
        x = self.conv_pre(x)

        # Upsample + resblocks
        for i in range(self.num_upsamples):
            x = jax.nn.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels

            # Average resblock outputs
            block_outputs = [self.resblocks[idx](x) for idx in range(start, end)]
            x = sum(block_outputs) / len(block_outputs)

        # Reference uses default F.leaky_relu(x) = slope 0.01, NOT LRELU_SLOPE
        x = jax.nn.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        x = jnp.tanh(x)

        return x  # [B, audio_length, out_channels]

    def load_weights(self, model_config, *args, **kwargs):
        from jax import ShapeDtypeStruct
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.utils.weight_utils import WeightLoader
        from sgl_jax.srt.multimodal.models.ltx2.utils import get_ltx2_checkpoint_dir, cleanup_ltx2_checkpoint_dir
        from .weight_mappings import create_vocoder_weight_mappings

        _ckpt_dir = get_ltx2_checkpoint_dir()
        if _ckpt_dir:
            model_config.model_path = _ckpt_dir

        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh,
            dtype=self.dtype or jnp.float32,
        )
        loader.load_weights_from_safetensors(create_vocoder_weight_mappings())
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

        # FIX: PyTorch ConvTranspose1d kernel has reverse spatial dimension compared to Flax.
        # We must flip the kernel weights along the spatial axis (axis 0) after loading.
        for up in self.ups:
            up.conv_transpose.kernel.value = jnp.flip(up.conv_transpose.kernel.value, axis=0)

        logger.info("Vocoder weights loaded")

    @staticmethod
    def get_config_class():
        from .ltx2_audio_vae_config import LTX2VocoderConfig
        return LTX2VocoderConfig
