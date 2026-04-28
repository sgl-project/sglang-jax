"""HybridLinearAttnBackend — dispatches per-layer to a full-attention sub-backend
(MLA / FlashAttention) or a linear-attention sub-backend (KDA).

The class itself owns no memory pool and allocates no device buffers; it only
holds two sub-backends + a `full_attn_layers` whitelist and routes calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
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

logger = logging.getLogger(__name__)


# --- Linear (state-based) attention base classes ----------------------------


@register_pytree_node_class
@dataclass
class LinearRecurrentAttnBackendMetadata:
    cu_q_lens: jax.Array = None
    recurrent_indices: jax.Array = None

    def tree_flatten(self):
        children = (self.cu_q_lens, self.recurrent_indices)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(cu_q_lens=children[0], recurrent_indices=children[1])


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

        # cu_q_lens
        if batch.forward_mode == ForwardMode.EXTEND:
            cu_q_lens = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(batch.extend_seq_lens, dtype=np.int32),
                ]
            )
        elif batch.forward_mode == ForwardMode.DECODE:
            cu_q_lens = np.arange(len(batch.seq_lens) + 1, dtype=np.int32)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        # put array to devices
        (
            metadata.cu_q_lens,
            metadata.recurrent_indices,
        ) = device_array(
            (cu_q_lens, batch.recurrent_indices),
            sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
        )

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
        """Returns (recurrent_cache, conv_cache) for the given layer.

        Matches RecurrentStatePool.get_linear_recurrent_layer_cache (PR #966)
        which returns a (recurrent, conv) tuple.
        """
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
    """Wrap full_attn_backend in HybridLinearAttnBackend for hybrid models.

    Mirrors upstream `sglang/srt/layers/attention/attention_registry.py:attn_backend_wrapper`.
    `runner.linear_recurrent_config` is the cheap "is this hybrid?" detector;
    dispatch to a concrete sub-backend uses the specific config properties
    (e.g. `runner.kimi_linear_config`).
    """
    cfg = runner.linear_recurrent_config
    if cfg is None:
        return full_attn_backend
    if runner.kimi_linear_config is not None:
        # KDAAttnBackend lives in a separate PR — lazy import keeps this PR
        # self-contained.
        try:
            from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
        except ImportError as e:
            raise ImportError(
                "HybridLinearAttnBackend needs KDAAttnBackend " "(delivered by a separate PR)."
            ) from e

        linear_attn_backend = KDAAttnBackend(runner.mesh)
    else:
        raise NotImplementedError(f"No linear backend wired for hybrid config {type(cfg).__name__}")
    return HybridLinearAttnBackend(
        full_attn_backend, linear_attn_backend, cfg.full_attention_layer_ids
    )


# --- Helpers ----------------------------------------------------------------


class MockRecurrentStatePool:
    def __init__(self, layer_caches: dict[int, tuple[jax.Array, jax.Array | None]] | None = None):
        logger.warning(
            "Using MockRecurrentStatePool; replace with HybridReqToTokenPool + "
            "RecurrentStatePool when PR #966 lands"
        )
        self.layer_caches = {} if layer_caches is None else dict(layer_caches)

    def get_linear_recurrent_indices(self, req_pool_indices: np.ndarray) -> np.ndarray:
        """Identity mapping — slot i maps to recurrent index i."""
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
