"""Physical layout contract for absorbed-MLA cache buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp

from sgl_jax.srt.kernels.mla.v2.kernel import align_to, get_kv_cache_shape


@dataclass(frozen=True)
class MLACacheLayout:
    """Describe latent and indexer buffers in the kernel-native page layout.

    Capacity profiling, allocation, and resident-byte reporting use this same
    module so changes to packing or alignment cannot drift between those paths.
    The two shape methods make the cache role explicit; feature width never
    determines whether a buffer is latent KV or an indexer key cache.
    """

    page_size: int
    dtype: jnp.dtype
    kv_lora_rank: int
    qk_rope_head_dim: int
    indexer_key_dim: int = 0

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
        for name in ("kv_lora_rank", "qk_rope_head_dim", "indexer_key_dim"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    @property
    def nope_dim(self) -> int:
        return align_to(self.kv_lora_rank, 128)

    @property
    def rope_dim(self) -> int:
        return align_to(self.qk_rope_head_dim, 128)

    @property
    def latent_dim(self) -> int:
        return self.nope_dim + self.rope_dim

    @property
    def indexer_dim(self) -> int:
        return align_to(self.indexer_key_dim, 128) if self.indexer_key_dim else 0

    def latent_shape(self, total_num_pages: int) -> tuple[int, ...]:
        return self._shape(total_num_pages=total_num_pages, feature_dim=self.latent_dim)

    def indexer_shape(self, total_num_pages: int) -> tuple[int, ...]:
        return self._shape(total_num_pages=total_num_pages, feature_dim=self.indexer_dim)

    def bytes_per_token(self, *, num_latent_layers: int, num_indexer_layers: int) -> int:
        """Return a conservative integer coefficient for capacity profiling."""
        if num_latent_layers < 0 or num_indexer_layers < 0:
            raise ValueError("cache layer counts must be non-negative")
        if num_indexer_layers and not self.indexer_dim:
            raise ValueError("indexer layers require a positive indexer_key_dim")

        bytes_per_page = self.latent_bytes(total_num_pages=1) * num_latent_layers
        bytes_per_page += self.indexer_bytes(total_num_pages=1) * num_indexer_layers
        return (bytes_per_page + self.page_size - 1) // self.page_size

    def latent_bytes(self, total_num_pages: int) -> int:
        return self._shape_bytes(self.latent_shape(total_num_pages))

    def indexer_bytes(self, total_num_pages: int) -> int:
        return self._shape_bytes(self.indexer_shape(total_num_pages))

    def _shape(self, *, total_num_pages: int, feature_dim: int) -> tuple[int, ...]:
        if total_num_pages < 0:
            raise ValueError(f"total_num_pages must be non-negative, got {total_num_pages}")
        return get_kv_cache_shape(
            total_num_pages=total_num_pages,
            page_size=self.page_size,
            kv_dim=feature_dim,
            kv_dtype=self.dtype,
        )

    def _shape_bytes(self, shape: tuple[int, ...]) -> int:
        return math.prod(shape) * jnp.dtype(self.dtype).itemsize
