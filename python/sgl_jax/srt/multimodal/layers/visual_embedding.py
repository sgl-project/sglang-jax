import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

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
