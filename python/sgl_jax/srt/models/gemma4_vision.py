# Adapted from vllm-project/tpu-inference's Gemma 4 JAX vision encoder.
# Copyright 2026 Google LLC and the SGLang-JAX authors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    precompile_mrope_vision_model,
)
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    VisionAttentionMetadata,
    make_vision_attention_backend,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import VisionShardSpecs

POSITIONS_PAD_VALUE = -1


@dataclasses.dataclass
class Gemma4VisionMetadata:
    """Bucket-shaped host metadata consumed by the Gemma 4 vision tower."""

    position_ids: Any  # int32[B, patch_capacity, 2]
    pool_indices: Any  # int32[B, patch_capacity], -1 for padding
    attention: VisionAttentionMetadata


jax.tree_util.register_dataclass(
    Gemma4VisionMetadata,
    data_fields=["position_ids", "pool_indices", "attention"],
    meta_fields=[],
)


def apply_multidimensional_rope(
    inputs: jax.Array,
    positions: jax.Array,
    base_frequency: float,
    rotary_fraction: float | None = None,
) -> jax.Array:
    """Apply Gemma 4's interleaved two-dimensional RoPE to ``[B,T,H,D]``."""

    batch, sequence_length, num_heads, head_dim = inputs.shape
    num_dimensions = positions.shape[-1]
    rotated_channels = head_dim
    if rotary_fraction is not None:
        rotated_channels = round(rotated_channels * rotary_fraction)
    channels_per_dimension = 2 * (rotated_channels // (2 * num_dimensions))
    half_channels = channels_per_dimension // 2
    rotary_dim = num_dimensions * channels_per_dimension

    rotated = inputs[..., :rotary_dim].reshape(
        batch,
        sequence_length,
        num_heads,
        num_dimensions,
        2,
        half_channels,
    )
    first, second = rotated[..., 0, :], rotated[..., 1, :]
    inv_freq = 1.0 / (
        base_frequency
        ** (jnp.arange(0, channels_per_dimension, 2, dtype=jnp.float32) / channels_per_dimension)
    )
    # Padded positions do not participate in attention; map their sentinel to
    # zero so the padded Q/K values remain finite.
    positions = jnp.maximum(positions, 0).astype(jnp.float32)
    freqs = positions[..., None] * inv_freq
    cos = jnp.cos(freqs)[:, :, None].astype(inputs.dtype)
    sin = jnp.sin(freqs)[:, :, None].astype(inputs.dtype)
    rotated = jnp.stack(
        (first * cos - second * sin, second * cos + first * sin),
        axis=-2,
    ).reshape(batch, sequence_length, num_heads, rotary_dim)
    if rotary_dim == head_dim:
        return rotated
    return jnp.concatenate((rotated, inputs[..., rotary_dim:]), axis=-1)


class Gemma4VisionPatchEmbedder(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, specs):
        self.patch_size = int(config.patch_size)
        self.patch_dim = 3 * self.patch_size**2
        self.hidden_size = int(config.hidden_size)
        self.specs = specs
        self.input_proj = LinearBase(
            self.patch_dim,
            self.hidden_size,
            mesh,
            use_bias=False,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        rngs = rngs or nnx.Rngs(0)
        table_shape = (2, int(config.position_embedding_size), self.hidden_size)
        self.position_embedding_table = nnx.Param(
            jax.random.normal(
                rngs.params(),
                table_shape,
                dtype=dtype,
                out_sharding=specs.sharding(None, None, specs.tensor_axis),
            )
        )

    def __call__(self, patches: jax.Array, position_ids: jax.Array) -> jax.Array:
        if patches.ndim != 3 or patches.shape[-1] != self.patch_dim:
            raise ValueError(
                f"Gemma 4 patches must have shape [B,T,{self.patch_dim}], got {patches.shape}"
            )
        if position_ids.shape != (*patches.shape[:2], 2):
            raise ValueError(
                "Gemma 4 pixel_position_ids must have shape [B,T,2], "
                f"got {position_ids.shape} for patches {patches.shape}"
            )
        col = self.specs.sharding(
            self.specs.batch_axis,
            None,
            self.specs.tensor_axis,
        )
        hidden, _ = self.input_proj(2.0 * (patches - 0.5), out_sharding=col)
        table = self.position_embedding_table[...]
        safe = jnp.clip(position_ids, 0, table.shape[1] - 1)
        position_embedding = table[0].at[safe[..., 0]].get(out_sharding=col) + table[1].at[
            safe[..., 1]
        ].get(out_sharding=col)
        valid = jnp.all(position_ids != POSITIONS_PAD_VALUE, axis=-1, keepdims=True)
        return hidden + jnp.where(valid, position_embedding, 0).astype(hidden.dtype)


class Gemma4VisionAttention(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        self.hidden_size = int(config.hidden_size)
        self.dtype = dtype
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        if self.num_heads % self.num_kv_heads:
            raise ValueError("Gemma 4 vision heads must be divisible by KV heads")
        if specs.tp and self.num_heads % int(mesh.shape["tensor"]):
            raise ValueError(
                f"vision num_heads={self.num_heads} must be divisible by "
                f"tp={mesh.shape['tensor']}"
            )
        self.specs = specs
        self.q_proj = LinearBase(
            self.hidden_size,
            self.num_heads * self.head_dim,
            mesh,
            use_bias=False,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.k_proj = LinearBase(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            mesh,
            use_bias=False,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.v_proj = LinearBase(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            mesh,
            use_bias=False,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.o_proj = LinearBase(
            self.num_heads * self.head_dim,
            self.hidden_size,
            mesh,
            use_bias=False,
            kernel_axes=specs.row_kernel_axes,
            params_dtype=dtype,
        )
        epsilon = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = GemmaRMSNorm(self.head_dim, epsilon=epsilon, add_unit_offset=False)
        self.k_norm = GemmaRMSNorm(self.head_dim, epsilon=epsilon, add_unit_offset=False)
        self.v_norm = RMSNorm(
            self.head_dim,
            epsilon=epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=False,
        )
        rope_parameters = getattr(config, "rope_parameters", {}) or {}
        if "full_attention" in rope_parameters:
            rope_parameters = rope_parameters["full_attention"]
        self.rope_theta = float(rope_parameters.get("rope_theta", 100.0))
        self.backend = make_vision_attention_backend(
            mesh,
            sm_scale=1.0,
            causal=False,
            head_tp=specs.tp,
            use_varlen=True,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        position_ids: jax.Array,
        attention: VisionAttentionMetadata,
    ) -> jax.Array:
        batch, length, _ = hidden_states.shape
        col = self.specs.sharding(
            self.specs.batch_axis,
            None,
            self.specs.tensor_axis,
        )
        head_sharding = self.specs.sharding(
            self.specs.batch_axis,
            None,
            self.specs.tensor_axis,
            None,
        )
        q, _ = self.q_proj(hidden_states, out_sharding=col)
        k, _ = self.k_proj(hidden_states, out_sharding=col)
        v, _ = self.v_proj(hidden_states, out_sharding=col)
        q = q.reshape(batch, length, self.num_heads, self.head_dim, out_sharding=head_sharding)
        k = k.reshape(
            batch,
            length,
            self.num_kv_heads,
            self.head_dim,
            out_sharding=head_sharding,
        )
        v = v.reshape(
            batch,
            length,
            self.num_kv_heads,
            self.head_dim,
            out_sharding=head_sharding,
        )
        q = apply_multidimensional_rope(self.q_norm(q), position_ids, self.rope_theta).astype(
            self.dtype
        )
        k = apply_multidimensional_rope(self.k_norm(k), position_ids, self.rope_theta).astype(
            self.dtype
        )
        v = self.v_norm(v).astype(self.dtype)
        output = self.backend(q, k, v, attention)
        output = output.reshape(
            batch,
            length,
            self.num_heads * self.head_dim,
            out_sharding=col,
        )
        return self.o_proj(
            output,
            out_sharding=self.specs.sharding(self.specs.batch_axis),
        )[0]


class Gemma4VisionMLP(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.specs = specs
        self.gate_proj = LinearBase(
            hidden_size,
            intermediate_size,
            mesh,
            use_bias=False,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.up_proj = LinearBase(
            hidden_size,
            intermediate_size,
            mesh,
            use_bias=False,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.down_proj = LinearBase(
            intermediate_size,
            hidden_size,
            mesh,
            use_bias=False,
            kernel_axes=specs.row_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        col = self.specs.sharding(
            self.specs.batch_axis,
            None,
            self.specs.tensor_axis,
        )
        gate, _ = self.gate_proj(hidden_states, out_sharding=col)
        up, _ = self.up_proj(hidden_states, out_sharding=col)
        return self.down_proj(
            jax.nn.gelu(gate, approximate=True) * up,
            out_sharding=self.specs.sharding(self.specs.batch_axis),
        )[0]


class Gemma4VisionEncoderLayer(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        epsilon = float(getattr(config, "rms_norm_eps", 1e-6))
        norm = lambda: GemmaRMSNorm(
            int(config.hidden_size),
            epsilon=epsilon,
            add_unit_offset=False,
        )
        self.input_layernorm = norm()
        self.self_attn = Gemma4VisionAttention(config, dtype, mesh, specs)
        self.post_attention_layernorm = norm()
        self.pre_feedforward_layernorm = norm()
        self.mlp = Gemma4VisionMLP(config, dtype, mesh, specs)
        self.post_feedforward_layernorm = norm()

    def __call__(self, hidden_states, position_ids, attention):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            attention,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(hidden_states)


class Gemma4MultimodalProjector(nnx.Module):
    def __init__(self, vision_hidden, text_hidden, epsilon, dtype, mesh, specs):
        self.specs = specs
        self.pre_projection_norm = RMSNorm(
            vision_hidden,
            epsilon=epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=False,
        )
        self.embedding_projection = LinearBase(
            vision_hidden,
            text_hidden,
            mesh,
            use_bias=False,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, hidden_states):
        return self.embedding_projection(
            self.pre_projection_norm(hidden_states),
            out_sharding=self.specs.sharding(
                self.specs.batch_axis,
                None,
                self.specs.tensor_axis,
            ),
        )[0]


class Gemma4VisionModel(nnx.Module):
    """Gemma 4 patch encoder using SGL-JAX lane packing and vision sharding."""

    def __init__(
        self,
        config,
        text_hidden_size: int,
        dtype,
        rngs,
        mesh: Mesh,
        vision_tp: bool,
        input_buckets: tuple[int, ...] | None = None,
    ):
        if mesh is None:
            raise ValueError("Gemma 4 vision requires a device mesh")
        self.config = config
        self.mesh = mesh
        self.vision_tp = vision_tp
        self.specs = VisionShardSpecs(mesh, vision_tp)
        self.dtype = dtype
        self.pooling_kernel_size = int(config.pooling_kernel_size)
        self.pooling_unit = self.pooling_kernel_size**2
        default_capacity = int(config.default_output_length) * self.pooling_unit
        buckets = input_buckets or (default_capacity, 2 * default_capacity)
        self.input_buckets = tuple(
            sorted(
                {
                    math.ceil(int(capacity) / self.pooling_unit) * self.pooling_unit
                    for capacity in buckets
                    if int(capacity) > 0
                }
            )
        )
        if not self.input_buckets:
            raise ValueError("Gemma 4 vision requires at least one positive input bucket")
        self.patch_dim = 3 * int(config.patch_size) ** 2
        self.patch_embedder = Gemma4VisionPatchEmbedder(
            config,
            dtype,
            rngs,
            mesh,
            self.specs,
        )
        self.layers = nnx.List(
            [
                Gemma4VisionEncoderLayer(config, dtype, mesh, self.specs)
                for _ in range(int(config.num_hidden_layers))
            ]
        )
        self.standardize = bool(getattr(config, "standardize", False))
        if self.standardize:
            self.std_bias = nnx.Param(jnp.zeros((int(config.hidden_size),), dtype=dtype))
            self.std_scale = nnx.Param(jnp.ones((int(config.hidden_size),), dtype=dtype))
        else:
            self.std_bias = None
            self.std_scale = None
        self.projector = Gemma4MultimodalProjector(
            int(config.hidden_size),
            int(text_hidden_size),
            float(getattr(config, "rms_norm_eps", 1e-6)),
            dtype,
            mesh,
            self.specs,
        )

    def _pool(self, hidden_states, pool_indices):
        batch, _, hidden_size = hidden_states.shape
        output_length = hidden_states.shape[1] // self.pooling_unit
        valid = pool_indices >= 0
        safe_indices = jnp.where(valid, pool_indices, 0)
        index_sharding = self.specs.sharding(self.specs.batch_axis, None)
        batch_indices = jnp.broadcast_to(
            jnp.arange(batch)[:, None],
            safe_indices.shape,
            out_sharding=index_sharding,
        )
        values = jnp.where(valid[..., None], hidden_states.astype(jnp.float32), 0)
        pooled_sharding = self.specs.sharding(
            self.specs.batch_axis,
            None,
            self.specs.tensor_axis,
        )
        pooled = jnp.zeros(
            (batch, output_length, hidden_size),
            dtype=jnp.float32,
            out_sharding=pooled_sharding,
        )
        pooled = pooled.at[batch_indices, safe_indices].add(
            values,
            out_sharding=pooled_sharding,
        )
        pooled /= self.pooling_unit
        mask = jnp.zeros(
            (batch, output_length),
            dtype=jnp.int32,
            out_sharding=index_sharding,
        )
        mask = mask.at[batch_indices, safe_indices].add(
            valid.astype(jnp.int32),
            out_sharding=index_sharding,
        )
        return pooled * math.sqrt(hidden_size), mask > 0

    def __call__(self, patches, position_ids, patch_counts):
        return self.encode(patches, position_ids, patch_counts)

    def _forward(self, patches, metadata: Gemma4VisionMetadata):
        hidden_states = self.patch_embedder(patches, metadata.position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                metadata.position_ids,
                metadata.attention,
            )
        hidden_states, output_mask = self._pool(hidden_states, metadata.pool_indices)
        if self.standardize:
            hidden_states = (
                hidden_states - self.std_bias[...].astype(jnp.float32)
            ) * self.std_scale[...].astype(jnp.float32)
        hidden_states = self.projector(hidden_states.astype(self.dtype))
        return jnp.where(output_mask[..., None], hidden_states, 0)

    @staticmethod
    def _pool_indices(position_ids: np.ndarray, kernel_size: int) -> np.ndarray:
        if position_ids.ndim != 2 or position_ids.shape[1] != 2:
            raise ValueError(
                f"pixel_position_ids must have shape [patches,2], got {position_ids.shape}"
            )
        if position_ids.size == 0 or np.any(position_ids < 0):
            raise ValueError("packed Gemma 4 pixel positions must be non-negative")
        unit = kernel_size**2
        if position_ids.shape[0] % unit:
            raise ValueError(
                f"Gemma 4 patch count {position_ids.shape[0]} must be divisible by {unit}"
            )
        max_x = int(position_ids[:, 0].max()) + 1
        if max_x % kernel_size:
            raise ValueError(f"Gemma 4 patch-grid width {max_x} must be divisible by {kernel_size}")
        pooled_width = max_x // kernel_size
        kernel_indices = position_ids // kernel_size
        result = kernel_indices[:, 1] * pooled_width + kernel_indices[:, 0]
        output_length = position_ids.shape[0] // unit
        counts = np.bincount(result, minlength=output_length)
        if result.max(initial=-1) >= output_length or not np.all(counts == unit):
            raise ValueError("Gemma 4 pixel positions do not form complete pooling windows")
        return result.astype(np.int32)

    def _build_metadata(
        self,
        position_ids: np.ndarray | jax.Array,
        patch_counts: np.ndarray | jax.Array,
    ) -> Gemma4VisionMetadata:
        position_ids = np.asarray(jax.device_get(position_ids), dtype=np.int32)
        patch_counts = np.asarray(jax.device_get(patch_counts), dtype=np.int32)
        batch, capacity = position_ids.shape[:2]
        pool_indices = np.full(
            (batch, capacity),
            POSITIONS_PAD_VALUE,
            dtype=np.int32,
        )
        cu_seqlens = np.zeros(
            (batch, capacity // self.pooling_unit + 1),
            dtype=np.int32,
        )
        for lane_index, counts in enumerate(patch_counts):
            patch_offset = 0
            output_offset = 0
            boundary_offset = 1
            for count in counts[counts > 0]:
                end = patch_offset + int(count)
                local_pool = self._pool_indices(
                    position_ids[lane_index, patch_offset:end], self.pooling_kernel_size
                )
                pool_indices[lane_index, patch_offset:end] = local_pool + output_offset
                cu_seqlens[lane_index, boundary_offset] = end
                patch_offset = end
                output_offset += int(count) // self.pooling_unit
                boundary_offset += 1
            cu_seqlens[lane_index, boundary_offset:] = patch_offset
        return Gemma4VisionMetadata(
            position_ids=position_ids,
            pool_indices=pool_indices,
            attention=VisionAttentionMetadata(cu_seqlens, max_seq_len=capacity),
        )

    @jax.jit
    def _encode_jit(self, patches, metadata):
        return self._forward(patches, metadata)

    def encode(self, patches, position_ids, patch_counts):
        batch_sharding = self.specs.sharding(self.specs.batch_axis)
        patches = jax.device_put(patches, batch_sharding)
        metadata = jax.device_put(
            self._build_metadata(position_ids, patch_counts),
            batch_sharding,
        )
        if self.mesh is None:
            return self._encode_jit(patches, metadata)
        with jax.set_mesh(self.mesh):
            return self._encode_jit(patches, metadata)

    def get_packed_capacities(self) -> tuple[int, ...]:
        rows = encoder_num_lanes(self.mesh, self.vision_tp)
        return tuple(rows * capacity // self.pooling_unit for capacity in self.input_buckets)

    def precompile(self) -> None:
        precompile_mrope_vision_model(
            self,
            mesh=self.mesh,
            num_lanes=encoder_num_lanes(self.mesh, self.vision_tp),
            buckets=self.input_buckets,
            patch_dim=self.patch_dim,
            merge_unit=self.pooling_unit,
            rope_type="rope_2d_packed",
        )
