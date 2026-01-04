from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx


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
