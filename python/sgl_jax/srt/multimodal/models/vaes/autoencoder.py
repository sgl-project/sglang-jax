from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.multimodal.configs.vaes.flux_vae_config import FluxVAEConfig
from sgl_jax.srt.multimodal.models.vaes.common import (
    Decoder,
    DiagonalGaussianDistribution,
    Encoder,
)
from sgl_jax.srt.multimodal.models.vaes.flux_vae_weight_mappings import to_mappings
from sgl_jax.srt.multimodal.utils import BaseOutput
from sgl_jax.srt.utils.weight_utils import WeightLoader


@dataclass
class DecoderOutput(BaseOutput):
    sample: jax.Array


@dataclass
class AutoencoderKLOutput(BaseOutput):
    latent_dist: DiagonalGaussianDistribution


class AutoencoderKL(nnx.Module):
    def __init__(
        self,
        config: FluxVAEConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        mesh=None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.mesh = mesh if mesh is not None else self._default_mesh()
        self.rngs = rngs or nnx.Rngs(0)

        self.encoder = Encoder(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            down_block_types=tuple(config.down_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            double_z=True,
            dtype=dtype,
            rngs=self.rngs,
        )
        self.decoder = Decoder(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=tuple(config.up_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            dtype=dtype,
            rngs=self.rngs,
        )

        if config.use_quant_conv:
            self.quant_conv = nnx.Conv(
                in_features=2 * config.latent_channels,
                out_features=2 * config.latent_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                param_dtype=dtype,
                rngs=self.rngs,
            )

        if config.use_post_quant_conv:
            self.post_quant_conv = nnx.Conv(
                in_features=config.latent_channels,
                out_features=config.latent_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                param_dtype=dtype,
                rngs=self.rngs,
            )

        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_size = config.sample_size
        sample_size = (
            config.sample_size[0]
            if isinstance(config.sample_size, (list, tuple))
            else config.sample_size
        )
        self.tile_latent_min_size = int(
            sample_size / (2 ** (len(config.block_out_channels) - 1))
        )
        self.tile_overlap_factor = 0.25

    @classmethod
    def from_config(
        cls,
        config: FluxVAEConfig | dict[str, Any],
        *,
        dtype: jnp.dtype = jnp.float32,
        mesh=None,
    ) -> "AutoencoderKL":
        if not isinstance(config, FluxVAEConfig):
            config = FluxVAEConfig.from_config(config)
        return cls(config=config, dtype=dtype, mesh=mesh)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        dtype: jnp.dtype = jnp.float32,
        mesh=None,
    ) -> "AutoencoderKL":
        config = FluxVAEConfig.from_pretrained(pretrained_model_name_or_path)
        model_path = Path(pretrained_model_name_or_path)
        config.model_path = str(model_path if model_path.name == "vae" else model_path / "vae")
        return cls(config=config, dtype=dtype, mesh=mesh)

    @staticmethod
    def get_config_class() -> type[FluxVAEConfig]:
        return FluxVAEConfig

    def load_weights(self, model_config: Any | None = None) -> None:
        normalized_model_config = model_config or self.config
        model_path = getattr(normalized_model_config, "model_path", None) or getattr(
            self.config, "model_path", None
        )
        if model_path is None:
            raise ValueError("`model_path` must be set before loading FLUX VAE weights.")

        vae_dir = Path(model_path)
        if vae_dir.name != "vae":
            vae_dir = vae_dir / "vae"
        normalized_model_config.model_path = str(vae_dir)

        loader = WeightLoader(
            model=self,
            model_config=normalized_model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(to_mappings(self.config))

    def encode(
        self,
        x: jax.Array,
        return_dict: bool = True,
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        moments = self._encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _encode(self, x: jax.Array) -> jax.Array:
        if self.use_tiling and (
            x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
        ):
            raise NotImplementedError("Tiled VAE encode is not implemented yet.")

        x_nhwc = self._to_nhwc(x)
        moments_nhwc = self.encoder(x_nhwc)
        if hasattr(self, "quant_conv"):
            moments_nhwc = self.quant_conv(moments_nhwc)
        return self._to_nchw(moments_nhwc)

    def decode(
        self,
        z: jax.Array,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[jax.Array]:
        return self._decode(z, return_dict=return_dict)

    def _decode(
        self,
        z: jax.Array,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[jax.Array]:
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
        ):
            raise NotImplementedError("Tiled VAE decode is not implemented yet.")

        z_nhwc = self._to_nhwc(z)
        if hasattr(self, "post_quant_conv"):
            z_nhwc = self.post_quant_conv(z_nhwc)
        sample_nhwc = self.decoder(z_nhwc)
        sample = self._to_nchw(sample_nhwc)

        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)

    def __call__(
        self,
        sample: jax.Array,
        sample_posterior: bool = False,
        key: jax.Array | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[jax.Array]:
        posterior = self.encode(sample, return_dict=True).latent_dist
        latents = posterior.sample(key) if sample_posterior else posterior.mode()
        return self.decode(latents, return_dict=return_dict)

    @staticmethod
    def _default_mesh():
        devices = np.asarray(jax.devices(), dtype=object)
        return jax.sharding.Mesh(devices, axis_names=("tensor",))

    @staticmethod
    def _to_nhwc(x: jax.Array) -> jax.Array:
        return jnp.transpose(x, (0, 2, 3, 1))

    @staticmethod
    def _to_nchw(x: jax.Array) -> jax.Array:
        return jnp.transpose(x, (0, 3, 1, 2))


EntryClass = AutoencoderKL
