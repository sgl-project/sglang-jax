from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    AttentionBackendMetadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.model_executor.model_runner import ModelRunner


# --- Linear (state-based) attention base classes ----------------------------


@register_pytree_node_class
@dataclass
class LinearRecurrentAttnBackendMetadata:
    cu_q_lens: jax.Array = None
    recurrent_indices: jax.Array = None
    has_initial_state: jax.Array = None

    def tree_flatten(self):
        children = (self.cu_q_lens, self.recurrent_indices, self.has_initial_state)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            cu_q_lens=children[0],
            recurrent_indices=children[1],
            has_initial_state=children[2],
        )


@dataclass
class LinearRecurrentAttnBackend(AttentionBackend):
    """Base class for linear recurrent attention backends (KDA, GDN, Mamba2).

    Provides metadata computation and pytree infrastructure.
    Subclasses implement ``__call__`` with model-specific forward logic.
    """

    def __init__(
        self,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.mesh = mesh
        self.forward_metadata = nnx.data(LinearRecurrentAttnBackendMetadata())

    def get_forward_metadata(
        self,
        batch: ModelWorkerBatch,
    ) -> LinearRecurrentAttnBackendMetadata:
        """Return the metadata for a forward pass."""
        metadata = LinearRecurrentAttnBackendMetadata()

        # Unified 2D reshape logic for all dp_size (including dp_size=1)
        if batch.forward_mode == ForwardMode.EXTEND:
            ext_2d = batch.extend_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
            cu_q_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
            cu_q_2d[:, 1:] = np.cumsum(ext_2d, axis=1)
            cu_q_lens = cu_q_2d.ravel()
        elif batch.forward_mode == ForwardMode.DECODE:
            single_cu = np.arange(batch.per_dp_bs_size + 1, dtype=np.int32)
            cu_q_lens = np.tile(single_cu, batch.dp_size)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        # put array to devices
        (
            metadata.cu_q_lens,
            metadata.recurrent_indices,
            metadata.has_initial_state,
        ) = device_array(
            (cu_q_lens, batch.recurrent_indices, batch.has_initial_state),
            sharding=(NamedSharding(self.mesh, P("data"))),
        )

        # [KDA-E2E DEBUG] log per-rank metadata sharding once per call (remove after validation)
        import os as _os

        if (
            not getattr(self, "_kda_dbg_logged", False)
            and _os.environ.get("KDA_E2E_DEBUG", "0") == "1"
        ):
            print(
                f"[KDA-E2E DEBUG][get_forward_metadata] mode={batch.forward_mode} "
                f"dp_size={batch.dp_size} per_dp_bs={batch.per_dp_bs_size} "
                f"cu_q_lens.shape={metadata.cu_q_lens.shape} "
                f"recurrent_indices.shape={metadata.recurrent_indices.shape} "
                f"sharding={metadata.cu_q_lens.sharding}",
                flush=True,
            )
            self._kda_dbg_logged = True

        return metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls()
        obj.forward_metadata = children[0]
        return obj

    @staticmethod
    def get_layer_cache(recurrent_state_pool, layer_id: int):
        """Returns (recurrent_cache, conv_cache) for the given layer."""
        return recurrent_state_pool.get_linear_recurrent_layer_cache(layer_id)

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        # hack setting to 1024, because linear attention backend has no limitation
        return 1024


# --- HybridLinearAttnBackend -----------------------------------------------


@register_pytree_node_class
@dataclass
class HybridLinearAttnBackendMetadata:
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


@dataclass
class HybridLinearAttnBackend(AttentionBackend):
    """Routes by layer.layer_id to a full or linear sub-backend.

    Owns no memory pool / device buffers — sub-backends do.
    Inherits AttentionBackend (nnx.Module) → auto-registered as a pytree;
    `_forward_metadata` is wrapped in `nnx.data(...)` so its leaves participate
    in flatten/unflatten without us writing custom tree methods.
    """

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: LinearRecurrentAttnBackend,
        full_attn_layers,
    ):
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        # Stored as a frozenset so membership checks are O(1) and the value is
        # hashable.
        self.full_attn_layers = frozenset(full_attn_layers)
        self._forward_metadata = nnx.data(HybridLinearAttnBackendMetadata())

    def get_forward_metadata(self, batch):
        return HybridLinearAttnBackendMetadata(
            full_attn_metadata=self.full_attn_backend.get_forward_metadata(batch),
            linear_attn_metadata=self.linear_attn_backend.get_forward_metadata(batch),
        )

    @property
    def forward_metadata(self):
        return self._forward_metadata

    @forward_metadata.setter
    def forward_metadata(self, value: HybridLinearAttnBackendMetadata):
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
        pool,  # token_to_kv_pool (full attn) or recurrent_state_pool (linear attn)
        mixed_qkv: jax.Array | None = None,  # For linear attention
        a: jax.Array | None = None,  # For linear attention
        b: jax.Array | None = None,  # For linear attention
        **kwargs,
    ):
        """Dispatch by layer.layer_id.

        full-attn sub-backend gets pool as `token_to_kv_pool=`; linear-attn
        sub-backend gets the same value as `recurrent_state_pool=` plus the
        linear-only mixed_qkv / a / b kwargs.
        """
        if layer.layer_id in self.full_attn_layers:
            return self.full_attn_backend(
                q,
                k,
                v,
                layer=layer,
                forward_batch=forward_batch,
                token_to_kv_pool=pool,
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
            recurrent_state_pool=pool,
            **kwargs,
        )

    def get_max_running_reqests(self, max_context_len, page_size):
        return self.full_attn_backend.get_max_running_reqests(max_context_len, page_size)


def attn_backend_wrapper(
    runner: ModelRunner,
    full_attn_backend: AttentionBackend,
):
    """Wrap full_attn_backend in HybridLinearAttnBackend for hybrid models."""
    return full_attn_backend
