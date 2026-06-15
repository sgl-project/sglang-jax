import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import modeling_flax_utils

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.layers.mlp import MLP, get_act_fn


class PatchEmbed(nnx.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d (or Conv3d for video)

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer
    """

    def __init__(
        self,
        patch_size: int | tuple[int, ...] | list[int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[[int], nnx.Module] | None = None,
        flatten: bool = True,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        prefix: str = "",
        *,
        rngs: nnx.Rngs = None,
    ):
        # Convert patch_size to tuple
        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) == 1:
                self.patch_size = (patch_size[0], patch_size[0])
            else:
                self.patch_size = tuple(patch_size)
        else:
            self.patch_size = (patch_size, patch_size)

        self.flatten = flatten

        self.proj = nnx.Conv(
            in_features=in_chans,
            out_features=embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=bias,
            dtype=dtype,
            padding="VALID",
            rngs=rngs,
        )

        self.norm = norm_layer(embed_dim) if norm_layer else lambda x: x

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape (B, D0, D1, ..., C) -> Channel Last
        """
        x = self.proj(x)
        if self.flatten:
            # Flatten spatial dims to (B, N, C)
            b = x.shape[0]
            c = x.shape[-1]
            x = x.reshape(b, -1, c)
        x = self.norm(x)
        return x


class TimestepEmbedder(nnx.Module):
    def __init__(
        self,
        hidden_size,
        act_layer="silu",
        frequency_embedding_size=256,
        max_period=10000,
        dtype=None,
        freq_dtype=jnp.float32,
        prefix: str = "",
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        self.mlp = MLP(
            input_dim=frequency_embedding_size,
            mlp_hidden_dim=hidden_size,
            output_dim=hidden_size,
            act_type=act_layer,
            dtype=dtype,
            mesh=mesh,
        )
        self.freq_dtype = freq_dtype

    def __call__(self, t: jax.Array, timestep_seq_len: int | None = None) -> jax.Array:
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period, dtype=self.freq_dtype
        ).astype(self.mlp.fc_in.weight.dtype)
        if timestep_seq_len is not None:
            assert (
                t_freq.shape[0] % timestep_seq_len == 0
            ), "timestep length is not divisible by timestep_seq_len"
            batch_size = t_freq.shape[0] // timestep_seq_len
            t_freq = t_freq.reshape((batch_size, timestep_seq_len, -1))
        t_emb = self.mlp(t_freq)
        return t_emb


def timestep_embedding(
    t: jax.Array,
    dim: int,
    max_period: int = 10000,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """
    Create sinusoidal timestep embeddings (JAX version).

    Args:
        t: Array of shape [B] with timesteps
        dim: Embedding dimension
        max_period: Controls the minimum frequency of the embeddings
        dtype: Output data type

    Returns:
        Array of shape [B, dim] with embeddings
    """
    half = dim // 2

    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=dtype) / half)

    args = t[:, None].astype(dtype) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)

    return embedding


class ModulateProjection(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        factor: int = 2,
        act_layer: str = "silu",
        dtype: jnp.dtype | None = None,
        mesh: jax.sharding.Mesh | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.factor = factor
        self.hidden_size = hidden_size
        self.linear = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size * factor,
            mesh=mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
        )
        self.act = get_act_fn(act_layer)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply the modulated projection to the input tensor.
        """
        x = self.act(x)
        x, _ = self.linear(x)
        return x


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
        act_fn: str | None = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        del rngs
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_1 = LinearBase(
            input_size=in_channels,
            output_size=time_embed_dim,
            use_bias=sample_proj_bias,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.cond_proj = (
            LinearBase(
                input_size=cond_proj_dim,
                output_size=in_channels,
                use_bias=False,
                mesh=mesh,
                params_dtype=params_dtype,
                kernel_axes=(None, "tensor"),
            )
            if cond_proj_dim is not None
            else None
        )
        self.linear_2 = LinearBase(
            input_size=time_embed_dim,
            output_size=time_embed_dim_out,
            use_bias=sample_proj_bias,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=("tensor", None),
        )
        self.act = modeling_flax_utils.ACT2FN[act_fn] if act_fn is not None else None
        self.post_act = modeling_flax_utils.ACT2FN[post_act_fn] if post_act_fn is not None else None

    def __call__(
        self,
        sample: jax.Array,
        condition: jax.Array | None = None,
    ) -> jax.Array:
        if condition is not None:
            if self.cond_proj is None:
                raise ValueError(
                    "`condition` was provided, but `cond_proj_dim` is not set for FluxTimestepEmbedding."
                )
            cond, _ = self.cond_proj(condition)
            sample = sample + cond
        sample, _ = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample, _ = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class FP32SiLU(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.silu(x.astype(jnp.float32)).astype(x.dtype)


class PixArtAlphaTextProjection(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        mesh: Mesh,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        del rngs
        if out_features is None:
            out_features = hidden_size
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
            output_size=out_features,
            use_bias=True,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=("tensor", None),
        )
        self.act_fn = act_fn
        if act_fn == "silu":
            self.act_1 = modeling_flax_utils.ACT2FN["silu"]
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")

    def __call__(self, caption: jax.Array) -> jax.Array:
        hidden_states, _ = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
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
        del rngs
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
        self.text_embedder = PixArtAlphaTextProjection(
            in_features=pooled_projection_dim,
            hidden_size=embedding_dim,
            out_features=embedding_dim,
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
        self.text_embedder = PixArtAlphaTextProjection(
            in_features=pooled_projection_dim,
            hidden_size=embedding_dim,
            out_features=embedding_dim,
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
