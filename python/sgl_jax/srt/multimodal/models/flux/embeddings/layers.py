import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import modeling_flax_utils

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.layers.linear import LinearBase


def _resolve_rngs(rngs: nnx.Rngs | None) -> nnx.Rngs:
    return rngs or nnx.Rngs(0)


def _apply_flux_rotary_emb(
    x: jax.Array,
    image_rotary_emb: tuple[jax.Array, jax.Array] | None,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 1,
) -> jax.Array:
    if image_rotary_emb is None:
        return x
    if not use_real:
        raise NotImplementedError(
            "Complex rotary embeddings are not implemented for Flux in sglang-jax."
        )

    cos, sin = image_rotary_emb
    if sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)

    if use_real_unbind_dim == -1:
        x_real, x_imag = jnp.split(x.reshape(*x.shape[:-1], -1, 2), 2, axis=-1)
        x_rotated = jnp.stack(
            [-jnp.squeeze(x_imag, axis=-1), jnp.squeeze(x_real, axis=-1)], axis=-1
        )
        x_rotated = x_rotated.reshape(x.shape)
    elif use_real_unbind_dim == -2:
        x_real, x_imag = jnp.split(x.reshape(*x.shape[:-1], 2, -1), 2, axis=-2)
        x_rotated = jnp.concatenate(
            [(-x_imag).squeeze(axis=-2), x_real.squeeze(axis=-2)],
            axis=-1,
        )
    else:
        raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

    return (x.astype(jnp.float32) * cos + x_rotated.astype(jnp.float32) * sin).astype(x.dtype)


def _get_1d_rotary_pos_embed(
    dim: int,
    pos: jax.Array,
    theta: float,
) -> tuple[jax.Array, jax.Array]:
    if dim % 2 != 0:
        raise ValueError(f"Rotary dimension must be even, got dim={dim}.")
    freqs_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=freqs_dtype) / dim))
    freqs = pos.astype(freqs_dtype)[:, None] * inv_freq[None, :]
    cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1).astype(jnp.float32)
    sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1).astype(jnp.float32)
    return cos, sin


class Timesteps(nnx.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        scale: float = 1.0,
        max_period: int = 10000,
    ):
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def __call__(self, timesteps: jax.Array) -> jax.Array:
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * jnp.arange(half_dim, dtype=jnp.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = jnp.exp(exponent)
        emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
        emb = self.scale * emb
        if self.flip_sin_to_cos:
            emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)
        else:
            emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        if self.num_channels % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb


class FluxTimestepEmbedding(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        mesh: Mesh,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.linear_1 = LinearBase(
            input_size=in_channels,
            output_size=time_embed_dim,
            use_bias=True,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.linear_2 = LinearBase(
            input_size=time_embed_dim,
            output_size=time_embed_dim,
            use_bias=True,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=("tensor", None),
        )
        self.act = modeling_flax_utils.ACT2FN["silu"]

    def __call__(self, sample: jax.Array) -> jax.Array:
        sample, _ = self.linear_1(sample)
        sample = self.act(sample)
        sample, _ = self.linear_2(sample)
        return sample


class FluxTextProjection(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        mesh: Mesh,
        act_fn: str = "silu",
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.linear_1 = LinearBase(
            input_size=in_features,
            output_size=hidden_size,
            use_bias=True,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.linear_2 = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=True,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=("tensor", None),
        )
        self.act_fn = act_fn
        self.act = modeling_flax_utils.ACT2FN[act_fn]

    def __call__(self, caption: jax.Array) -> jax.Array:
        hidden_states, _ = self.linear_1(caption)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class LabelEmbedding(nnx.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float,
        mesh: Mesh,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = Embed(
            num_embeddings=num_classes + use_cfg_embedding,
            features=hidden_size,
            dtype=params_dtype,
            param_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
            mesh=mesh,
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self,
        labels: jax.Array,
        force_drop_ids: jax.Array | None = None,
    ) -> jax.Array:
        if force_drop_ids is None:
            return labels
        drop_ids = force_drop_ids.astype(bool)
        return jnp.where(drop_ids, self.num_classes, labels)

    def __call__(
        self,
        labels: jax.Array,
        force_drop_ids: jax.Array | None = None,
    ) -> jax.Array:
        labels = labels.astype(jnp.int32)
        if force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class CombinedTimestepLabelEmbeddings(nnx.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        mesh: Mesh,
        class_dropout_prob: float = 0.1,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = _resolve_rngs(rngs)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = FluxTimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            mesh=mesh,
            params_dtype=params_dtype,
            rngs=_rngs,
        )
        self.class_embedder = LabelEmbedding(
            num_classes=num_classes,
            hidden_size=embedding_dim,
            dropout_prob=class_dropout_prob,
            mesh=mesh,
            params_dtype=params_dtype,
            rngs=_rngs,
        )

    def __call__(
        self,
        timestep: jax.Array,
        class_labels: jax.Array,
        hidden_dtype: jnp.dtype | None = None,
    ) -> jax.Array:
        timesteps_proj = self.time_proj(timestep)
        if hidden_dtype is not None:
            timesteps_proj = timesteps_proj.astype(hidden_dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        class_emb = self.class_embedder(class_labels)
        return timesteps_emb + class_emb


class CombinedTimestepTextProjEmbeddings(nnx.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        mesh: Mesh,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = _resolve_rngs(rngs)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = FluxTimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            mesh=mesh,
            params_dtype=params_dtype,
            rngs=_rngs,
        )
        self.text_embedder = FluxTextProjection(
            in_features=pooled_projection_dim,
            hidden_size=embedding_dim,
            mesh=mesh,
            act_fn="silu",
            params_dtype=params_dtype,
            rngs=_rngs,
        )

    def __call__(self, timestep: jax.Array, pooled_projection: jax.Array) -> jax.Array:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        pooled_emb = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_emb


class CombinedTimestepGuidanceTextProjEmbeddings(nnx.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        mesh: Mesh,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = _resolve_rngs(rngs)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = FluxTimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            mesh=mesh,
            params_dtype=params_dtype,
            rngs=_rngs,
        )
        self.guidance_embedder = FluxTimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            mesh=mesh,
            params_dtype=params_dtype,
            rngs=_rngs,
        )
        self.text_embedder = FluxTextProjection(
            in_features=pooled_projection_dim,
            hidden_size=embedding_dim,
            mesh=mesh,
            act_fn="silu",
            params_dtype=params_dtype,
            rngs=_rngs,
        )

    def __call__(
        self,
        timestep: jax.Array,
        guidance: jax.Array,
        pooled_projection: jax.Array,
    ) -> jax.Array:
        timesteps_proj = self.time_proj(timestep)
        guidance_proj = self.time_proj(guidance)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        guidance_emb = self.guidance_embedder(guidance_proj)
        pooled_emb = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_emb
