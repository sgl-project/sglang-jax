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
    from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
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
    metadata_cls = LinearRecurrentAttnBackendMetadata

    def __init__(self, mesh: jax.sharding.Mesh | None = None):
        self.mesh = mesh
        self.forward_metadata = nnx.data(self.metadata_cls())

    def get_forward_metadata(
        self,
        batch: ModelWorkerBatch,
        recurrent_indices: np.ndarray | jax.Array,
    ) -> LinearRecurrentAttnBackendMetadata:
        if batch.forward_mode == ForwardMode.EXTEND:
            q_lens = np.asarray(batch.extend_seq_lens, dtype=np.int32)
            cu_q_lens = np.concatenate(
                [np.array([0], dtype=np.int32), np.cumsum(q_lens, dtype=np.int32)]
            )
        elif batch.forward_mode == ForwardMode.DECODE:
            cu_q_lens = np.arange(len(batch.seq_lens) + 1, dtype=np.int32)
        elif batch.forward_mode in (ForwardMode.IDLE, ForwardMode.DUMMY_FIRST):
            cu_q_lens = np.array([0], dtype=np.int32)
        else:
            raise NotImplementedError(
                f"Linear recurrent attention does not support {batch.forward_mode}"
            )

        recurrent_indices = np.asarray(recurrent_indices, dtype=np.int32)
        sharding = (
            NamedSharding(self.mesh, P())
            if self.mesh is not None and jax.process_count() == 1
            else None
        )
        cu_q_lens_dev, recurrent_indices_dev = device_array(
            (cu_q_lens, recurrent_indices),
            sharding=sharding,
        )
        metadata = self.metadata_cls(
            cu_q_lens=cu_q_lens_dev,
            recurrent_indices=recurrent_indices_dev,
        )
        self.forward_metadata = nnx.data(metadata)
        return metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {"mesh": self.mesh}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(mesh=aux_data["mesh"])
        obj.forward_metadata = children[0]
        return obj

    def __call__(
        self,
        mixed_qkv: jax.Array,
        a: jax.Array,
        b: jax.Array,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
        **kwargs,
    ) -> jax.Array:
        q, k, v = self._split_mixed_qkv(mixed_qkv, layer)
        g = self._reshape_gate(a, q)
        beta = self._reshape_beta(b, q)

        if forward_batch.forward_mode in (ForwardMode.IDLE, ForwardMode.DUMMY_FIRST):
            return jnp.zeros((q.shape[0], q.shape[1] * v.shape[-1]), dtype=v.dtype)
        if forward_batch.forward_mode in (
            ForwardMode.DRAFT_EXTEND,
            ForwardMode.TARGET_VERIFY,
        ):
            raise NotImplementedError(
                f"Linear recurrent attention does not support {forward_batch.forward_mode}"
            )

        recurrent_indices = self.forward_metadata.recurrent_indices
        recurrent_cache, _ = self._get_layer_cache(
            recurrent_state_pool,
            layer.layer_id,
        )
        initial_state = recurrent_cache[recurrent_indices].astype(jnp.float32)

        if forward_batch.forward_mode == ForwardMode.EXTEND:
            output, new_recurrent = self._dispatch_chunk(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                self.forward_metadata.cu_q_lens,
                layer,
            )
            output = output[0]
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            output, new_recurrent = self._dispatch_recurrent(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                layer,
            )
            output = output[:, 0]
        else:
            raise NotImplementedError(
                f"Linear recurrent attention does not support {forward_batch.forward_mode}"
            )

        recurrent_state_pool.set_linear_recurrent_layer_cache(
            layer.layer_id,
            recurrent_indices,
            new_recurrent,
            None,
        )
        return output.reshape(output.shape[0], -1)

    def _dispatch_chunk(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        cu_seqlens: jax.Array,
        layer: RadixLinearAttention,
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError()

    def _dispatch_recurrent(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        layer: RadixLinearAttention,
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError()

    @staticmethod
    def _split_mixed_qkv(
        mixed_qkv: jax.Array | tuple[jax.Array, jax.Array, jax.Array],
        layer: RadixLinearAttention,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(mixed_qkv, tuple):
            if len(mixed_qkv) != 3:
                raise ValueError("mixed_qkv tuple must contain q, k, v")
            return mixed_qkv

        if mixed_qkv.ndim == 4 and mixed_qkv.shape[1] == 3:
            return mixed_qkv[:, 0], mixed_qkv[:, 1], mixed_qkv[:, 2]

        if mixed_qkv.ndim == 3:
            q_end = layer.head_q_dim
            k_end = q_end + layer.head_k_dim
            v_end = k_end + layer.head_v_dim
            if mixed_qkv.shape[-1] != v_end:
                raise ValueError(
                    f"mixed_qkv last dim must be {v_end}, got {mixed_qkv.shape[-1]}"
                )
            return (
                mixed_qkv[:, :, :q_end],
                mixed_qkv[:, :, q_end:k_end],
                mixed_qkv[:, :, k_end:v_end],
            )

        if mixed_qkv.ndim == 2:
            q_size = layer.num_q_heads * layer.head_q_dim
            k_size = layer.num_k_heads * layer.head_k_dim
            v_size = layer.num_v_heads * layer.head_v_dim
            if mixed_qkv.shape[-1] != q_size + k_size + v_size:
                raise ValueError(
                    "mixed_qkv flat dim does not match q/k/v head configuration"
                )
            q_raw, k_raw, v_raw = jnp.split(
                mixed_qkv,
                [q_size, q_size + k_size],
                axis=-1,
            )
            return (
                q_raw.reshape(mixed_qkv.shape[0], layer.num_q_heads, layer.head_q_dim),
                k_raw.reshape(mixed_qkv.shape[0], layer.num_k_heads, layer.head_k_dim),
                v_raw.reshape(mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim),
            )

        raise ValueError(f"Unsupported mixed_qkv shape: {mixed_qkv.shape}")

    @staticmethod
    def _reshape_gate(gate: jax.Array, q: jax.Array) -> jax.Array:
        if gate.shape == q.shape:
            return gate
        if gate.ndim == 2 and gate.shape[-1] == q.shape[1] * q.shape[2]:
            return gate.reshape(q.shape)
        raise ValueError(f"Gate shape {gate.shape} is incompatible with q shape {q.shape}")

    @staticmethod
    def _reshape_beta(beta: jax.Array, q: jax.Array) -> jax.Array:
        if beta.ndim == 3 and beta.shape[-1] == 1:
            beta = beta[..., 0]
        if beta.shape == q.shape[:2]:
            return beta
        if beta.ndim == 1 and beta.shape[0] == q.shape[0] * q.shape[1]:
            return beta.reshape(q.shape[:2])
        raise ValueError(f"Beta shape {beta.shape} is incompatible with q shape {q.shape}")

    @staticmethod
    def _get_scale(layer: RadixLinearAttention) -> float:
        return layer.scaling if layer.scaling is not None else layer.head_q_dim**-0.5

    @staticmethod
    def _get_layer_cache(recurrent_state_pool, layer_id: int):
        layer_cache = recurrent_state_pool.get_linear_recurrent_layer_cache(layer_id)
        if isinstance(layer_cache, dict):
            recurrent_cache = layer_cache.get("recurrent")
            if recurrent_cache is None:
                recurrent_cache = layer_cache.get("recurrent_state")
            return recurrent_cache, layer_cache.get("conv")
        if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
            return layer_cache
        raise TypeError(
            "linear recurrent layer cache must be a dict or a (recurrent, conv) tuple"
        )

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        del page_size
        return max_context_len


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
        linear_attn_backend: nnx.Module,
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

        linear_attn_backend = KDAAttnBackend(runner)
    else:
        raise NotImplementedError(f"No linear backend wired for hybrid config {type(cfg).__name__}")
    return HybridLinearAttnBackend(
        full_attn_backend, linear_attn_backend, cfg.full_attention_layer_ids
    )


# --- Helpers ----------------------------------------------------------------


class MockRecurrentStatePool:
    def __init__(self, layer_caches: dict[int, tuple[jax.Array, jax.Array | None]] | None = None):
        logger.warning(
            "Using MockRecurrentStatePool; replace with RecurrentStatePool when RFC-0015 lands"
        )
        self.layer_caches = {} if layer_caches is None else dict(layer_caches)

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
        if conv_cache is not None and conv is not None:
            conv_cache = conv_cache.at[indices].set(conv)
        self.layer_caches[layer_id] = (recurrent_cache, conv_cache)
