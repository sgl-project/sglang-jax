"""Test-only recurrent state pool helper for Lightning attention tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class MockRecurrentStatePool:
    """Small test double matching the RecurrentStatePool methods used by backends."""

    def __init__(self, layer_caches: dict[int, tuple[jax.Array, jax.Array | None]] | None = None):
        self.layer_caches = {} if layer_caches is None else dict(layer_caches)

    def get_linear_recurrent_indices(self, req_pool_indices: np.ndarray) -> np.ndarray:
        return np.asarray(req_pool_indices, dtype=np.int32)

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        return self.layer_caches[layer_id]

    def set_linear_recurrent_layer_cache(
        self,
        layer_id: int,
        indices: jax.Array,
        recurrent: jax.Array,
        conv: jax.Array | None,
    ) -> None:
        if layer_id not in self.layer_caches:
            self.layer_caches[layer_id] = (recurrent, conv)
            return

        recurrent_cache, conv_cache = self.layer_caches[layer_id]
        recurrent_cache = recurrent_cache.at[indices].set(recurrent)
        if conv is not None:
            if conv_cache is None:
                conv_cache = jnp.zeros(
                    (recurrent_cache.shape[0],) + conv.shape[1:],
                    dtype=conv.dtype,
                )
            conv_cache = conv_cache.at[indices].set(conv)
        self.layer_caches[layer_id] = (recurrent_cache, conv_cache)
