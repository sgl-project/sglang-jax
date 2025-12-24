import jax
import jax.numpy as jnp
from flax import nnx
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
    ):
        self.kernel_size = kernel_size
        self.temporal_padding = padding[0]  # Save for cache size calculation
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
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

    def __call__(
        self, x: jax.Array, cache: jax.Array | None | None = None
    ) -> tuple[jax.Array, jax.Array | None | None]:
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
            ## todo new_cache
            # Pad on the left if we do not yet have cache_t frames (e.g., first call with T=1).
            if new_cache.shape[1] < cache_t:
                pad_t = cache_t - new_cache.shape[1]
                new_cache = jnp.pad(
                    new_cache, ((0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)), mode="constant"
                )
        else:
            new_cache = None

        return out, new_cache


rngs = nnx.Rngs(0)
model = CausalConv3d(3, 4, rngs=rngs, padding=(1, 1, 1))

key = jax.random.PRNGKey(42)

key, subkey = jax.random.split(key)
x = jax.random.uniform(subkey, shape=(1, 2, 3, 4, 3))

y = model(x)
print(y[0].shape, y[1] == x)
