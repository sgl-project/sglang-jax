import jax
import jax.numpy as jnp
from flax import nnx


class OneDRotaryEmbedding(nnx.Module):
    """1D rotary positional embedding."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        theta_rescale_factor: float = 1.0,
        interpolation_factor: float = 1.0,
        dtype: jnp.dtype = jnp.float32,
        use_real: bool = False,
        repeat_interleave_real: bool = False,
    ):
        self.dim = dim
        self.theta = theta
        self.theta_rescale_factor = theta_rescale_factor
        self.interpolation_factor = interpolation_factor
        self.dtype = dtype
        self.use_real = use_real
        self.repeat_interleave_real = repeat_interleave_real

    def __call__(self, pos: jax.Array) -> tuple[jax.Array, jax.Array]:
        theta = self.theta
        if self.theta_rescale_factor != 1.0:
            theta *= self.theta_rescale_factor ** (self.dim / (self.dim - 2))

        # freqs calculation
        idx = jnp.arange(0, self.dim, 2, dtype=self.dtype)[: (self.dim // 2)]
        freqs = 1.0 / (theta ** (idx / self.dim))

        freqs = jnp.outer(pos * self.interpolation_factor, freqs)
        freqs_cos = jnp.cos(freqs)
        freqs_sin = jnp.sin(freqs)

        if self.use_real and self.repeat_interleave_real:
            freqs_cos = jnp.repeat(freqs_cos, 2, axis=1)
            freqs_sin = jnp.repeat(freqs_sin, 2, axis=1)

        return freqs_cos, freqs_sin

    def forward_from_grid(self, size: int, base_offset: int) -> tuple[jax.Array, jax.Array]:
        pos = jnp.arange(size, dtype=self.dtype) + base_offset
        return self.__call__(pos)


class NDRotaryEmbedding(nnx.Module):
    """N-dimensional rotary positional embedding."""

    def __init__(
        self,
        rope_dim_list: list[int],
        rope_theta: float,
        theta_rescale_factor: float | list[float] = 1.0,
        interpolation_factor: float | list[float] = 1.0,
        use_real: bool = False,
        repeat_interleave_real: bool = False,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.rope_dim_list = rope_dim_list
        self.ndim = len(rope_dim_list)
        self.rope_theta = rope_theta
        self.dtype = dtype
        self.use_real = use_real
        self.repeat_interleave_real = repeat_interleave_real

        # Handle factors list/float normalization
        if isinstance(theta_rescale_factor, (int, float)):
            self.theta_rescale_factor = [theta_rescale_factor] * self.ndim
        else:
            self.theta_rescale_factor = theta_rescale_factor

        if isinstance(interpolation_factor, (int, float)):
            self.interpolation_factor = [interpolation_factor] * self.ndim
        else:
            self.interpolation_factor = interpolation_factor

        self.rope_generators = []
        for i in range(self.ndim):
            self.rope_generators.append(
                OneDRotaryEmbedding(
                    dim=self.rope_dim_list[i],
                    theta=self.rope_theta,
                    theta_rescale_factor=self.theta_rescale_factor[i],
                    interpolation_factor=self.interpolation_factor[i],
                    dtype=self.dtype,
                    use_real=use_real,
                    repeat_interleave_real=repeat_interleave_real,
                )
            )

    def forward_from_grid(
        self,
        grid_size: tuple[int, ...],
        shard_dim: int = 0,
        start_frame: int = 0,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Computes embeddings for a structured grid.
        Ignores manual SP sharding, returning global embeddings for compiler-based sharding.
        """
        sizes = grid_size
        starts = (0,) * self.ndim

        # Pre-allocate output lists to concatenate later
        cos_list = []
        sin_list = []

        for i in range(self.ndim):
            size_i = sizes[i]
            base_offset = starts[i]
            if i == 0 and start_frame > 0:
                base_offset += start_frame

            # Get 1D embedding
            cos_1d, sin_1d = self.rope_generators[i].forward_from_grid(size_i, base_offset)

            # Expand logic
            # repeats_per_entry (inner repeat) -> jnp.repeat
            # tile_count (outer tile) -> jnp.tile

            repeats_per_entry = 1
            for j in range(i + 1, self.ndim):
                repeats_per_entry *= sizes[j]

            tile_count = 1
            for j in range(0, i):
                tile_count *= sizes[j]

            # Expand
            if repeats_per_entry > 1:
                cos_expanded = jnp.repeat(cos_1d, repeats_per_entry, axis=0)
                sin_expanded = jnp.repeat(sin_1d, repeats_per_entry, axis=0)
            else:
                cos_expanded = cos_1d
                sin_expanded = sin_1d

            if tile_count > 1:
                cos_expanded = jnp.tile(cos_expanded, (tile_count, 1))
                sin_expanded = jnp.tile(sin_expanded, (tile_count, 1))

            cos_list.append(cos_expanded)
            sin_list.append(sin_expanded)

        cos = jnp.concatenate(cos_list, axis=1)
        sin = jnp.concatenate(sin_list, axis=1)

        return cos.astype(jnp.float32), sin.astype(jnp.float32)
