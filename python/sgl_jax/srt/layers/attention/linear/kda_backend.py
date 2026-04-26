from dataclasses import dataclass

import jax
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.kda import chunk_kda, fused_recurrent_kda
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
    LinearRecurrentAttnBackendMetadata,
)


@register_pytree_node_class
@dataclass
class KDAAttnBackendMetadata(LinearRecurrentAttnBackendMetadata):
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(cu_q_lens=children[0], recurrent_indices=children[1])


class KDAAttnBackend(LinearRecurrentAttnBackend):
    def __init__(self, mesh: jax.sharding.Mesh | None = None):
        super().__init__(mesh=mesh)
        self.forward_metadata = KDAAttnBackendMetadata()

    def _dispatch_chunk(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        cu_seqlens: jax.Array,
        layer,
    ) -> tuple[jax.Array, jax.Array]:
        result = chunk_kda(
            q[None, ...],
            k[None, ...],
            v[None, ...],
            g[None, ...],
            beta[None, ...],
            scale=self._get_scale(layer),
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        return result[0], result[1]

    def _dispatch_recurrent(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        layer,
    ) -> tuple[jax.Array, jax.Array]:
        return fused_recurrent_kda(
            q[:, None, ...],
            k[:, None, ...],
            v[:, None, ...],
            g[:, None, ...],
            beta[:, None, ...],
            scale=self._get_scale(layer),
            initial_state=initial_state,
            output_final_state=True,
        )


__all__ = ["KDAAttnBackend", "KDAAttnBackendMetadata"]
