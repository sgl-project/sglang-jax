"""HybridLinearAttnBackend — dispatches per-layer to a full-attention sub-backend
(MLA / FlashAttention) or a linear-attention sub-backend (KDA).

The class itself owns no memory pool and allocates no device buffers; it only
holds two sub-backends + a `full_attn_layers` whitelist and routes calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
from flax import nnx
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    AttentionBackendMetadata,
)

if TYPE_CHECKING:
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.model_executor.model_runner import ModelRunner


# --- Placeholder linear (state-based) attention base classes ----------------
# Real implementation lands in a separate PR; these stubs exist so that
# `linear/kda_backend.py` (already on the epic branch) keeps importing.


@register_pytree_node_class
@dataclass
class LinearRecurrentAttnBackendMetadata:
    """Stub metadata for linear-recurrent backends; KDA PR will flesh out fields."""

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


class LinearRecurrentAttnBackend(AttentionBackend):
    """Stub base class for linear-recurrent backends (KDA, etc.).

    Inherits AttentionBackend (nnx.Module), so it is auto-registered as a pytree
    via flax-nnx's metaclass. tree_flatten / tree_unflatten are provided
    explicitly for symmetry with the metadata stub.
    """

    def get_forward_metadata(self, batch):
        # Concrete subclasses (KDAAttnBackend, ...) override this. The stub
        # returns an empty metadata so it can be instantiated for tests.
        return LinearRecurrentAttnBackendMetadata()

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


# --- HybridLinearAttnBackend -----------------------------------------------


@register_pytree_node_class
@dataclass
class HybridLinearAttentionBackendMetadata:
    """Aggregate metadata returned by HybridLinearAttnBackend.get_forward_metadata.

    The setter on HybridLinearAttnBackend.forward_metadata unpacks these two
    fields and assigns them to the corresponding sub-backend's forward_metadata.
    """

    full_attn_metadata: AttentionBackendMetadata = field(default=None)
    linear_attn_metadata: LinearRecurrentAttnBackendMetadata = field(default=None)

    def tree_flatten(self):
        return ((self.full_attn_metadata, self.linear_attn_metadata), {})

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(full_attn_metadata=children[0], linear_attn_metadata=children[1])


class HybridLinearAttnBackend(AttentionBackend):
    """Routes by layer.layer_id to a full or linear sub-backend.

    Owns no memory pool / device buffers — sub-backends do.
    """

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: nnx.Module,
        full_attn_layers,
    ):
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        # Stored as a frozenset so membership checks are O(1) and the value is
        # hashable (safe to use as pytree aux_data).
        self.full_attn_layers = frozenset(full_attn_layers)
        self._forward_metadata = nnx.data(HybridLinearAttentionBackendMetadata())

    def get_forward_metadata(self, batch):
        return HybridLinearAttentionBackendMetadata(
            full_attn_metadata=self.full_attn_backend.get_forward_metadata(batch),
            linear_attn_metadata=self.linear_attn_backend.get_forward_metadata(batch),
        )

    @property
    def forward_metadata(self):
        return self._forward_metadata

    @forward_metadata.setter
    def forward_metadata(self, value: HybridLinearAttentionBackendMetadata):
        self._forward_metadata = value
        self.full_attn_backend.forward_metadata = value.full_attn_metadata
        self.linear_attn_backend.forward_metadata = value.linear_attn_metadata

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer,  # RadixAttention or RadixLinearAttention
        forward_batch: ForwardBatch,
        pool,  # state_pool or token_to_kv_pool
        mixed_qkv: jax.Array | None = None,  # For linear attention
        a: jax.Array | None = None,  # For linear attention
        b: jax.Array | None = None,  # For linear attention
        **kwargs,
    ):
        """Dispatch by layer.layer_id.

        full-attention sub-backend gets (q, k, v, layer, forward_batch, pool, **kwargs).
        linear-attention sub-backend additionally gets mixed_qkv / a / b before pool.
        """
        if layer.layer_id in self.full_attn_layers:
            return self.full_attn_backend(
                q,
                k,
                v,
                layer=layer,
                forward_batch=forward_batch,
                pool=pool,
                **kwargs,
            )
        return self.linear_attn_backend(
            q,
            k,
            v,
            layer=layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            pool=pool,
            **kwargs,
        )

    def get_max_running_reqests(self, max_context_len, page_size):
        return min(
            self.full_attn_backend.get_max_running_reqests(max_context_len, page_size),
            self.linear_attn_backend.get_max_running_reqests(max_context_len, page_size),
        )

    # Pytree registration: full_attn_layers is hashable (frozenset) and goes to
    # aux_data so structure changes trigger retracing; everything else is data.
    def tree_flatten(self):
        children = (
            self.full_attn_backend,
            self.linear_attn_backend,
            self._forward_metadata,
        )
        aux_data = {"full_attn_layers": self.full_attn_layers}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.full_attn_backend = children[0]
        obj.linear_attn_backend = children[1]
        obj._forward_metadata = children[2]
        obj.full_attn_layers = aux_data["full_attn_layers"]
        return obj


def attn_backend_wrapper(
    runner: ModelRunner,
    full_attn_backend: AttentionBackend,
):
    """Wrap full_attn_backend in HybridLinearAttnBackend for hybrid models.

    Mirrors upstream `sglang/srt/layers/attention/attention_registry.py:attn_backend_wrapper`.
    When no hybrid (linear-recurrent) config is set, returns `full_attn_backend`
    unchanged so the caller can invoke this unconditionally.
    """
    if runner.linear_recurrent_config is not None:
        # KDAAttnBackend lives in a separate PR — lazy import keeps this PR
        # self-contained.
        try:
            from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
        except ImportError as e:
            raise ImportError(
                "HybridLinearAttnBackend needs KDAAttnBackend " "(delivered by a separate PR)."
            ) from e

        linear_attn_backend = KDAAttnBackend(runner)
        full_attn_layers = runner.linear_recurrent_config.full_attention_layer_ids
        return HybridLinearAttnBackend(full_attn_backend, linear_attn_backend, full_attn_layers)

    return full_attn_backend
