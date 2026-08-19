"""MiMoV2 multimodal (VLM) model for SGLang-JAX.

The text/MoE backbone is reused verbatim from :mod:`mimo_v2_flash` /
:mod:`mimo_v2_pro`; this module adds the MiMoV2 vision tower and wires it onto
the in-model multimodal contract (:class:`InModelMultimodalContract`).

The vision compute (GQA attention with optional attention sinks, per-layer
windowed *band* attention with column-major reordering, SwiGLU MLP, spatial
patch merger) mirrors the HF reference ``MiMoVisionTransformer``. The packing
and encode glue follows the same item-ordered array contract as
``qwen2_5_vl.py``. Audio follows the same contract as vision.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.models.mimo_v2_flash import MiMoV2FlashForCausalLM
from sgl_jax.srt.models.mimo_v2_pro import MiMoV2ForCausalLM
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    pack_lanes,
    precompile_mrope_vision_model,
    restore_encoder_output,
    run_mrope_vision_model,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import (
    VisionShardSpecs,
    apply_data_sharding,
    resolve_encoder_tp,
)
from sgl_jax.srt.utils.common_utils import resolve_vision_patch_buckets
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

_init_fn = nnx.initializers.uniform()

ConfigLike = Mapping[str, Any] | SimpleNamespace | object


def _value(config: ConfigLike | None, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


# ---------------------------------------------------------------------------
# Vision attention primitives
# ---------------------------------------------------------------------------


def _apply_rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    """RoPE for the ViT: *x* is ``[B, T, heads, head_dim]``, *freqs* ``[B, T, head_dim]``."""
    original_dtype = x.dtype
    x = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    rotated = jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    cos = jnp.cos(freqs)[:, :, None, :]
    sin = jnp.sin(freqs)[:, :, None, :]
    return (x * cos + rotated * sin).astype(original_dtype)


def _take_units(x: jax.Array, index: jax.Array, unit: int) -> jax.Array:
    """Reorder *x* ``[B, L, ...]`` in blocks of ``unit`` rows by ``index`` ``[B, L/unit]``."""
    batch, length = x.shape[:2]
    tail = x.shape[2:]
    x = x.reshape(batch, length // unit, unit, *tail)
    gather = index.reshape(batch, index.shape[1], *([1] * (x.ndim - 2)))
    gather = jnp.broadcast_to(gather, (batch, index.shape[1], *x.shape[2:]))
    return jnp.take_along_axis(x, gather, axis=1).reshape(batch, length, *tail)


# ---------------------------------------------------------------------------
# Per-resolution vision metadata
# ---------------------------------------------------------------------------


@register_pytree_node_class
class _MiMoVisionMetadata:
    """Packed per-lane vision metadata (host arrays or device arrays)."""

    def __init__(self, col_index, rotary_freqs, cu_seqlens):
        self.col_index = col_index
        self.rotary_freqs = rotary_freqs
        self.cu_seqlens = cu_seqlens

    def tree_flatten(self):
        return (self.col_index, self.rotary_freqs, self.cu_seqlens), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


class MiMoVisionPatchEmbed(nnx.Module):
    """3D (temporal × spatial) patch embedding conv."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp):
        self.temporal_patch_size = int(_value(config, "temporal_patch_size", 2))
        self.patch_size = int(_value(config, "patch_size", 16))
        self.in_channels = int(_value(config, "in_channels", None) or _value(config, "in_chans", 3))
        self.hidden_size = int(_value(config, "hidden_size"))
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)
        self.proj = nnx.Conv(
            in_features=self.in_channels,
            out_features=self.hidden_size,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            strides=(self.temporal_patch_size, self.patch_size, self.patch_size),
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, S, _ = x.shape
        C, T, P = self.in_channels, self.temporal_patch_size, self.patch_size
        x = x.reshape(B, S, C, T, P, P)
        if self.mesh is not None:
            x = apply_data_sharding(x, self.mesh, PartitionSpec(self.specs.batch_axis))
        x = jnp.transpose(x, (0, 1, 3, 4, 5, 2))  # [B, S, T, P, P, C]

        sh = None
        if self.mesh is not None and "data" in self.mesh.abstract_mesh.explicit_axes:
            sh = self.specs.sharding(self.specs.batch_axis)
        x = x.reshape(B * S, T, P, P, C, out_sharding=sh)
        x = self.proj(x, out_sharding=sh)
        x = x.reshape(B, S, self.hidden_size, out_sharding=sh)
        return x


class MiMoVisionMLP(nnx.Module):
    """SwiGLU MLP with bias."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp):
        hidden = int(_value(config, "hidden_size"))
        intermediate = int(_value(config, "intermediate_size"))
        self.specs = VisionShardSpecs(mesh, vision_tp)
        act = _value(config, "hidden_act", "silu")
        self.act_fn = jax.nn.silu if act == "silu" else jax.nn.gelu
        self.gate_proj = LinearBase(
            hidden,
            intermediate,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.up_proj = LinearBase(
            hidden,
            intermediate,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.down_proj = LinearBase(
            intermediate,
            hidden,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        specs = self.specs
        col = specs.sharding(specs.batch_axis, None, specs.tensor_axis)
        row = specs.sharding(specs.batch_axis)
        gate, _ = self.gate_proj(x, out_sharding=col)
        up, _ = self.up_proj(x, out_sharding=col)
        out, _ = self.down_proj(self.act_fn(gate) * up, out_sharding=row)
        return out


class MiMoVisionAttention(nnx.Module):
    """ViT self-attention: split QKV, GQA, RoPE, packed-cu sparse attention."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp, use_sinks):
        hidden = int(_value(config, "hidden_size"))
        self.num_heads = int(_value(config, "num_heads"))
        self.num_kv_heads = int(
            _value(config, "num_key_value_heads", self.num_heads) or self.num_heads
        )
        self.head_dim = int(_value(config, "qk_channels", 64))
        if self.num_heads % self.num_kv_heads:
            raise ValueError("MiMoV2 vision num_heads must be divisible by num_key_value_heads.")
        if self.head_dim % 4:
            raise ValueError("MiMoV2 vision head_dim must be divisible by 4.")
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)

        self.q_proj = LinearBase(
            hidden,
            self.num_heads * self.head_dim,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.k_proj = LinearBase(
            hidden,
            self.num_kv_heads * self.head_dim,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.v_proj = LinearBase(
            hidden,
            self.num_kv_heads * self.head_dim,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.proj = LinearBase(
            self.num_heads * self.head_dim,
            hidden,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )
        sink_spec = PartitionSpec(self.specs.tensor_axis)
        self.sinks = (
            nnx.Param(
                jnp.zeros(
                    (self.num_heads,),
                    dtype=dtype,
                    out_sharding=(NamedSharding(mesh, sink_spec) if mesh is not None else None),
                )
            )
            if use_sinks
            else None
        )

        if mesh is not None and jax.default_backend() != "cpu":
            from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
                VisionVarlenAttentionBackend,
            )

            self.attn_backend = VisionVarlenAttentionBackend(
                mesh,
                sm_scale=1.0 / math.sqrt(self.head_dim),
                head_tp=self.specs.tp,
            )
        else:
            self.attn_backend = None

    def __call__(self, x, freqs, cu_seqlens, window_size) -> jax.Array:
        B, T, _ = x.shape
        specs = self.specs
        col = specs.sharding(specs.batch_axis, None, specs.tensor_axis)
        hs = specs.sharding(specs.batch_axis, None, specs.tensor_axis, None)

        q, _ = self.q_proj(x, out_sharding=col)
        k, _ = self.k_proj(x, out_sharding=col)
        v, _ = self.v_proj(x, out_sharding=col)
        q = q.reshape(B, T, self.num_heads, self.head_dim, out_sharding=hs)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim, out_sharding=hs)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim, out_sharding=hs)

        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)

        if isinstance(window_size, int):
            ragged_window = (-1, -1) if window_size <= 0 else (window_size, window_size)
        else:
            ragged_window = window_size
        sinks = None if self.sinks is None else self.sinks[...]
        if self.attn_backend is None:
            from sgl_jax.srt.multimodal.kernels.varlen_attention import (
                ref_varlen_attention,
                varlen_attention,
            )

            attention = ref_varlen_attention if jax.default_backend() == "cpu" else varlen_attention

            out = jnp.stack(
                [
                    attention(
                        q[index],
                        k[index],
                        v[index],
                        cu_seqlens[index],
                        jnp.sum(jnp.diff(cu_seqlens[index]) > 0, dtype=jnp.int32).reshape(1),
                        sm_scale=1.0 / math.sqrt(self.head_dim),
                        window_size=ragged_window,
                        attention_sink=sinks,
                    )
                    for index in range(B)
                ]
            )
        else:
            out = self.attn_backend(
                q,
                k,
                v,
                cu_seqlens,
                sinks,
                window_size=ragged_window,
            )

        out = out.reshape(B, T, self.num_heads * self.head_dim, out_sharding=col)
        out, _ = self.proj(out, out_sharding=specs.sharding(specs.batch_axis))
        return out


class MiMoVisionBlock(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, vision_tp, use_sinks):
        hidden = int(_value(config, "hidden_size"))
        eps = float(_value(config, "rms_norm_eps", 1e-6))
        _rngs = rngs or nnx.Rngs(0)
        self.norm1 = nnx.RMSNorm(hidden, epsilon=eps, dtype=dtype, param_dtype=dtype, rngs=_rngs)
        self.norm2 = nnx.RMSNorm(hidden, epsilon=eps, dtype=dtype, param_dtype=dtype, rngs=_rngs)
        self.attn = MiMoVisionAttention(config, dtype, rngs, mesh, vision_tp, use_sinks)
        self.mlp = MiMoVisionMLP(config, dtype, rngs, mesh, vision_tp)

    def __call__(self, x, freqs, cu_seqlens, window_size) -> jax.Array:
        x = x + self.attn(self.norm1(x), freqs, cu_seqlens, window_size)
        x = x + self.mlp(self.norm2(x))
        return x


class MiMoVisionPatchMerger(nnx.Module):
    """LayerNorm → reshape(sms²) → 2-layer MLP → [B, T/sms², out_hidden]."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp):
        context = int(_value(config, "hidden_size"))
        self.unit = int(_value(config, "spatial_merge_size", 2)) ** 2
        self.hidden_size = context * self.unit
        self.specs = VisionShardSpecs(mesh, vision_tp)
        _rngs = rngs or nnx.Rngs(0)
        self.ln_q = nnx.LayerNorm(
            context,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=dtype,
            use_fast_variance=False,
            rngs=_rngs,
        )
        self.mlp_fc1 = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.mlp_fc2 = LinearBase(
            self.hidden_size,
            int(_value(config, "out_hidden_size")),
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        specs = self.specs
        row = specs.sharding(specs.batch_axis)
        x = self.ln_q(x)
        x = x.reshape(x.shape[0], -1, self.hidden_size, out_sharding=row)
        x, _ = self.mlp_fc1(
            x, out_sharding=specs.sharding(specs.batch_axis, None, specs.tensor_axis)
        )
        x = jax.nn.gelu(x, approximate=False)
        x, _ = self.mlp_fc2(x, out_sharding=row)
        return x


class MiMoVisionTransformer(nnx.Module):
    """MiMoV2 ViT: patch embed → windowed/full blocks (col reorder) → merge."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp, input_buckets):
        self.config = config
        self.mesh = mesh
        self.vision_tp = vision_tp
        self.specs = VisionShardSpecs(mesh, vision_tp)
        self.dtype = dtype

        self.spatial_merge_size = int(_value(config, "spatial_merge_size", 2))
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.input_buckets = tuple(input_buckets)
        if any(b <= 0 or b % self.spatial_merge_unit for b in self.input_buckets):
            raise ValueError(
                f"vision patch buckets must be positive multiples of {self.spatial_merge_unit}"
            )

        self.patch_size = int(_value(config, "patch_size", 16))
        self.temporal_patch_size = int(_value(config, "temporal_patch_size", 2))
        self.in_channels = int(_value(config, "in_channels", None) or _value(config, "in_chans", 3))
        self.patch_dim = self.in_channels * self.temporal_patch_size * self.patch_size**2
        self.head_dim = int(_value(config, "qk_channels", 64))
        self.theta = float(_value(config, "rope_theta", 10000.0))

        depth = int(_value(config, "depth"))
        full = tuple(int(i) for i in (_value(config, "fullatt_block_indexes", ()) or ()))
        self.full_blocks = frozenset(full)
        self.window_types = tuple(_value(config, "vit_window_attn_types", None) or [-1] * depth)
        if len(self.window_types) != depth:
            raise ValueError("vit_window_attn_types must have one entry per vision block.")
        use_sink = bool(_value(config, "use_sink", False))
        self.window_size = int(_value(config, "visual_token_window_size", -1))

        self.patch_embed = MiMoVisionPatchEmbed(config, dtype, rngs, mesh, vision_tp)
        self.blocks = nnx.List(
            [
                MiMoVisionBlock(
                    config, dtype, rngs, mesh, vision_tp, use_sink and i not in self.full_blocks
                )
                for i in range(depth)
            ]
        )
        self.merger = MiMoVisionPatchMerger(config, dtype, rngs, mesh, vision_tp)

        self._metadata_cache: dict[tuple[int, int, int], _MiMoVisionMetadata] = {}

    # -- forward ----------------------------------------------------------

    def __call__(self, patches, grid_thw) -> jax.Array:
        return self.encode(patches, grid_thw)

    def _forward(self, patches, meta: _MiMoVisionMetadata, valid) -> jax.Array:
        col_index = jnp.asarray(meta.col_index)
        rotary_freqs = jnp.asarray(meta.rotary_freqs)
        cu_seqlens = jnp.asarray(meta.cu_seqlens)

        hidden = self.patch_embed(patches)
        col_freqs = _take_units(rotary_freqs, col_index, self.spatial_merge_unit)
        reverse_col_index = jnp.argsort(col_index, axis=1)

        for index, block in enumerate(self.blocks):
            col = self.window_types[index] == 1
            previous_col = index > 0 and self.window_types[index - 1] == 1
            if col and not previous_col:
                hidden = _take_units(hidden, col_index, self.spatial_merge_unit)
            elif previous_col and not col:
                hidden = _take_units(hidden, reverse_col_index, self.spatial_merge_unit)
            freqs = col_freqs if col else rotary_freqs
            window = -1 if index in self.full_blocks else self.window_size
            hidden = block(hidden, freqs, cu_seqlens, window)

        output = self.merger(hidden)
        output_valid = valid // self.spatial_merge_unit
        return jnp.where(
            jnp.arange(output.shape[1])[None, :, None] < output_valid[:, None, None],
            output,
            0,
        )

    # -- metadata ---------------------------------------------------------

    def _metadata_for_grid(self, grid: tuple[int, int, int]) -> _MiMoVisionMetadata:
        cached = self._metadata_cache.get(grid)
        if cached is not None:
            return cached
        t, h, w = grid
        merge = self.spatial_merge_size
        if min(grid) <= 0 or h % merge or w % merge:
            raise ValueError(
                f"MiMoV2 vision grid {grid} must be positive and divisible by {merge}."
            )
        h_pos, w_pos = np.indices((h, w))
        shape = (h // merge, merge, w // merge, merge)
        h_pos = h_pos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        w_pos = w_pos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        pos = np.tile(np.stack((h_pos, w_pos), axis=-1), (t, 1))
        inv = 1.0 / (
            self.theta
            ** (np.arange(0, self.head_dim // 2, 2, dtype=np.float32) / (self.head_dim // 2))
        )
        table = np.outer(np.arange(max(h, w), dtype=np.float32), inv)
        freqs = table[pos].reshape(pos.shape[0], -1)
        freqs = np.concatenate((freqs, freqs), axis=-1).astype(np.float32)

        units = np.arange(t * (h // merge) * (w // merge), dtype=np.int32)
        col_index = units.reshape(t, h // merge, w // merge).transpose(0, 2, 1).reshape(-1)
        cu_seqlens = np.arange(0, (t + 1) * h * w, h * w, dtype=np.int32)
        meta = _MiMoVisionMetadata(col_index, freqs, cu_seqlens)
        if len(self._metadata_cache) >= 64:
            self._metadata_cache.pop(next(iter(self._metadata_cache)))
        self._metadata_cache[grid] = meta
        return meta

    def _pack_metadata(self, grids: list[tuple[int, int, int]]) -> _MiMoVisionMetadata:
        col_indices, freqs = [], []
        cu_seqlens = [0]
        unit_offset = 0
        patch_offset = 0
        for grid in grids:
            meta = self._metadata_for_grid(grid)
            col_indices.append(meta.col_index + unit_offset)
            freqs.append(meta.rotary_freqs)
            cu_seqlens.extend((meta.cu_seqlens[1:] + patch_offset).tolist())
            unit_offset += meta.col_index.size
            patch_offset += int(np.prod(grid))
        return _MiMoVisionMetadata(
            np.concatenate(col_indices),
            np.concatenate(freqs),
            np.asarray(cu_seqlens, dtype=np.int32),
        )

    def _empty_metadata(self, input_capacity: int) -> _MiMoVisionMetadata:
        merge = self.spatial_merge_size
        return self._metadata_for_grid((1, merge, input_capacity // merge))

    def _pad_metadata(self, meta: _MiMoVisionMetadata, input_capacity: int) -> _MiMoVisionMetadata:
        units = input_capacity // self.spatial_merge_unit
        col_index = np.arange(units, dtype=np.int32)
        col_index[: meta.col_index.size] = meta.col_index
        freqs = np.zeros((input_capacity, self.head_dim), dtype=np.float32)
        freqs[: meta.rotary_freqs.shape[0]] = meta.rotary_freqs
        boundary_capacity = units + 1
        cu_seqlens = np.full(boundary_capacity, meta.cu_seqlens[-1], dtype=np.int32)
        cu_seqlens[: meta.cu_seqlens.size] = meta.cu_seqlens
        return _MiMoVisionMetadata(col_index, freqs, cu_seqlens)

    @nnx.jit
    def _encode_jit(self, patches, meta, valid) -> jax.Array:
        features = self._forward(patches, meta, valid)
        if self.mesh is None:
            return features
        return jax.sharding.reshard(
            features,
            NamedSharding(self.mesh, PartitionSpec(*([None] * features.ndim))),
        )

    def encode(self, patches, grid_thw: np.ndarray | jax.Array) -> jax.Array:
        batch_sharding = self.specs.sharding(self.specs.batch_axis)
        patches = jax.device_put(patches, batch_sharding)
        meta, valid = self._build_metadata(grid_thw, patches.shape[1])
        meta = jax.device_put(meta, batch_sharding)
        valid = jax.device_put(valid, batch_sharding)
        if self.mesh is None:
            return self._encode_jit(patches, meta, valid)
        with jax.set_mesh(self.mesh):
            return self._encode_jit(patches, meta, valid)

    def _build_metadata(self, grid_thw: np.ndarray | jax.Array, capacity: int):
        grid_thw = np.asarray(jax.device_get(grid_thw), dtype=np.int32)
        if grid_thw.ndim == 2:
            grid_thw = grid_thw[None]
        empty = self._pad_metadata(self._empty_metadata(capacity), capacity)
        metadata = []
        valid = np.zeros(len(grid_thw), dtype=np.int32)
        for lane_index, lane in enumerate(grid_thw):
            grids = [tuple(map(int, grid)) for grid in lane if np.any(grid)]
            metadata.append(
                self._pad_metadata(self._pack_metadata(grids), capacity) if grids else empty
            )
            valid[lane_index] = sum(int(np.prod(grid)) for grid in grids)
        return jax.tree.map(lambda *values: np.stack(values), *metadata), valid

    def precompile(self) -> None:
        precompile_mrope_vision_model(
            self,
            mesh=self.mesh,
            num_lanes=encoder_num_lanes(self.mesh, self.vision_tp),
            buckets=self.input_buckets,
            patch_dim=self.patch_dim,
            merge_unit=self.spatial_merge_unit,
            rope_type="rope_3d",
        )


# ---------------------------------------------------------------------------
# Audio tower (speech codes → per-channel embed → local transformer → project)
# ---------------------------------------------------------------------------


def _int_list(value, length: int) -> list[int]:
    if isinstance(value, str):
        values = [int(x) for x in value.split("-")]
    elif isinstance(value, int):
        values = [value]
    else:
        values = [int(x) for x in value]
    if len(values) == 1:
        values *= length
    if len(values) != length:
        raise ValueError(f"Expected {length} audio values, got {len(values)}.")
    return values


class MiMoAudioCodeEmbedding(nnx.Module):
    """Per-channel speech-code embedding lookup (padding index handled upstream)."""

    def __init__(self, size, features, dtype, mesh, specs):
        self.mesh = mesh
        self.specs = specs
        self.embedding = nnx.Param(
            jnp.zeros(
                (size, features),
                dtype=dtype,
                out_sharding=(
                    NamedSharding(mesh, PartitionSpec(None, None)) if mesh is not None else None
                ),
            )
        )

    def __call__(self, indices: jax.Array) -> jax.Array:
        emb = self.embedding[...]
        if self.mesh is None:
            return emb[indices]
        sh = NamedSharding(
            self.mesh, PartitionSpec(self.specs.batch_axis, *([None] * (indices.ndim - 1)))
        )
        return emb.at[indices].get(out_sharding=sh)


class MiMoAudioAttention(nnx.Module):
    """Full or causal self-attention with partial RoPE (no GQA), audio-local."""

    def __init__(self, config, dtype, mesh, specs):
        hidden = int(_value(config, "input_local_dim"))
        self.heads = int(_value(config, "input_local_attn_heads"))
        self.head_dim = int(_value(config, "input_local_head_dim", hidden // self.heads))
        self.rotary_dim = int(self.head_dim * float(_value(config, "partial_rotary_factor", 1.0)))
        if self.rotary_dim % 2:
            raise ValueError("MiMoV2 audio rotary dimension must be even.")
        self.theta = float(_value(config, "rope_theta", 640000.0))
        self.full_attention = bool(_value(config, "input_full_attention", True))
        self.specs = specs
        proj = lambda bias: LinearBase(
            hidden,
            self.heads * self.head_dim,
            mesh=mesh,
            use_bias=bias,
            kernel_axes=(None, None),
            params_dtype=dtype,
        )
        self.q_proj, self.k_proj, self.v_proj = proj(True), proj(True), proj(True)
        self.o_proj = LinearBase(
            self.heads * self.head_dim,
            hidden,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
        )

    def __call__(self, hidden: jax.Array) -> jax.Array:
        B, T = hidden.shape[:2]
        row = self.specs.sharding(self.specs.batch_axis)
        q, _ = self.q_proj(hidden, out_sharding=row)
        k, _ = self.k_proj(hidden, out_sharding=row)
        v, _ = self.v_proj(hidden, out_sharding=row)
        q, k, v = (x.reshape(B, T, self.heads, self.head_dim) for x in (q, k, v))
        positions = jnp.arange(T, dtype=jnp.float32)
        inv = 1.0 / (
            self.theta ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        angles = jnp.outer(positions, inv)
        freqs = jnp.concatenate((angles, angles), axis=-1)[None]
        q = jnp.concatenate(
            (_apply_rope(q[..., : self.rotary_dim], freqs), q[..., self.rotary_dim :]), axis=-1
        )
        k = jnp.concatenate(
            (_apply_rope(k[..., : self.rotary_dim], freqs), k[..., self.rotary_dim :]), axis=-1
        )
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)
        if not self.full_attention:
            scores = jnp.where(
                jnp.arange(T)[:, None] >= jnp.arange(T)[None, :],
                scores,
                jnp.finfo(scores.dtype).min,
            )
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(hidden.dtype)
        out = jnp.einsum("bhts,bshd->bthd", probs, v).reshape(B, T, self.heads * self.head_dim)
        return self.o_proj(out, out_sharding=row)[0]


class MiMoAudioMLP(nnx.Module):
    """Audio-local SwiGLU MLP (no bias)."""

    def __init__(self, config, dtype, mesh, specs):
        hidden = int(_value(config, "input_local_dim"))
        intermediate = int(_value(config, "input_local_intermediate_size"))
        self.specs = specs
        linear = lambda i, o: LinearBase(
            i, o, mesh=mesh, use_bias=False, kernel_axes=(None, None), params_dtype=dtype
        )
        self.gate_proj = linear(hidden, intermediate)
        self.up_proj = linear(hidden, intermediate)
        self.down_proj = linear(intermediate, hidden)

    def __call__(self, hidden: jax.Array) -> jax.Array:
        row = self.specs.sharding(self.specs.batch_axis)
        gate, _ = self.gate_proj(hidden, out_sharding=row)
        up, _ = self.up_proj(hidden, out_sharding=row)
        return self.down_proj(jax.nn.silu(gate) * up, out_sharding=row)[0]


class MiMoAudioBlock(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        hidden = int(_value(config, "input_local_dim"))
        eps = float(_value(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = nnx.RMSNorm(hidden, epsilon=eps, param_dtype=dtype, rngs=nnx.Rngs(0))
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden, epsilon=eps, param_dtype=dtype, rngs=nnx.Rngs(0)
        )
        self.self_attn = MiMoAudioAttention(config, dtype, mesh, specs)
        self.mlp = MiMoAudioMLP(config, dtype, mesh, specs)

    def __call__(self, hidden: jax.Array) -> jax.Array:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class MiMoAudioTransformer(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        self.layers = nnx.List(
            [
                MiMoAudioBlock(config, dtype, mesh, specs)
                for _ in range(int(_value(config, "input_local_layers")))
            ]
        )
        self.norm = (
            nnx.RMSNorm(
                int(_value(config, "input_local_dim")),
                epsilon=float(_value(config, "rms_norm_eps", 1e-6)),
                param_dtype=dtype,
                rngs=nnx.Rngs(0),
            )
            if bool(_value(config, "add_post_norm", True))
            else None
        )

    def __call__(self, hidden: jax.Array) -> jax.Array:
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden) if self.norm is not None else hidden


class MiMoAudioEncoder(nnx.Module):
    """Speech codes ``[B, cap, C]`` → grouped embed → local transformer → project."""

    def __init__(self, config, dtype, mesh, encoder_tp):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.encoder_tp = encoder_tp
        self.specs = VisionShardSpecs(mesh, encoder_tp)
        self.channels = int(_value(config, "audio_channels"))
        self.group_size = int(_value(config, "group_size"))
        self.local_dim = int(_value(config, "input_local_dim"))
        self.out_hidden_size = int(_value(config, "out_hidden_size"))
        vocab_sizes = _int_list(_value(config, "speech_vocab_size"), self.channels)
        self.zero_ids = tuple(
            _int_list(
                _value(config, "speech_zeroemb_idx", _value(config, "zeroemb_idx")),
                self.channels,
            )
        )
        # Bucket ladder (multiples of group_size); oversize falls back to pow2.
        self.input_buckets = tuple(self.group_size * n for n in (16, 64, 256, 1024))

        self.speech_embeddings = nnx.List(
            [
                MiMoAudioCodeEmbedding(size, self.local_dim, dtype, mesh, self.specs)
                for size in vocab_sizes
            ]
        )
        self.transformer = MiMoAudioTransformer(config, dtype, mesh, self.specs)
        projection_layers = int(_value(config, "projection_layers", 2))
        projection_input = self.local_dim * self.group_size
        linear = lambda i, o: LinearBase(
            i, o, mesh=mesh, use_bias=False, kernel_axes=(None, None), params_dtype=dtype
        )
        if projection_layers == 1:
            self.proj_fc1 = linear(projection_input, self.out_hidden_size)
            self.proj_fc2 = None
        elif projection_layers == 2:
            self.proj_fc1 = linear(projection_input, projection_input * 4)
            self.proj_fc2 = linear(projection_input * 4, self.out_hidden_size)
        else:
            raise ValueError(f"Unsupported MiMoV2 audio projection_layers={projection_layers}.")

    def __call__(self, codes: jax.Array, valid: jax.Array) -> jax.Array:
        codes = codes.astype(jnp.int32)
        position_valid = jnp.arange(codes.shape[1])[None] < valid[:, None]
        zero_ids = jnp.asarray(self.zero_ids, dtype=jnp.int32)
        codes = jnp.where(position_valid[:, :, None], codes, zero_ids)
        B, T = codes.shape[:2]
        groups = T // self.group_size
        codes = codes.reshape(
            B,
            groups,
            self.group_size,
            self.channels,
            out_sharding=self.specs.sharding(self.specs.batch_axis, None, None, None),
        )
        hidden = jnp.zeros((B, groups, self.group_size, self.local_dim), dtype=self.dtype)
        for channel, embedding in enumerate(self.speech_embeddings):
            hidden += embedding(codes[..., channel])
        hidden = hidden.reshape(
            B * groups,
            self.group_size,
            self.local_dim,
            out_sharding=self.specs.sharding(self.specs.batch_axis, None, None),
        )
        hidden = self.transformer(hidden)
        hidden = hidden.reshape(
            B,
            groups,
            self.group_size * self.local_dim,
            out_sharding=self.specs.sharding(self.specs.batch_axis, None, None),
        )
        row = self.specs.sharding(self.specs.batch_axis)
        hidden, _ = self.proj_fc1(hidden, out_sharding=row)
        if self.proj_fc2 is not None:
            hidden = jax.nn.gelu(hidden, approximate=False)
            hidden, _ = self.proj_fc2(hidden, out_sharding=row)
        output_valid = valid // self.group_size
        return jnp.where(
            jnp.arange(groups)[None, :, None] < output_valid[:, None, None],
            hidden,
            0,
        )

    @nnx.jit
    def _encode_jit(self, codes, valid) -> jax.Array:
        features = self(codes, valid)
        if self.mesh is None:
            return features
        return jax.sharding.reshard(
            features,
            NamedSharding(self.mesh, PartitionSpec(*([None] * features.ndim))),
        )

    def encode(self, codes, valid) -> jax.Array:
        if self.mesh is None:
            return self._encode_jit(codes, valid)
        with jax.set_mesh(self.mesh):
            return self._encode_jit(codes, valid)

    def precompile(self) -> None:
        num_lanes = encoder_num_lanes(self.mesh, self.encoder_tp)
        batch_sharding = self.specs.sharding(self.specs.batch_axis)
        for capacity in self.input_buckets:
            codes = jax.device_put(
                np.zeros((num_lanes, capacity, self.channels), dtype=np.int32),
                batch_sharding,
            )
            valid = np.zeros(num_lanes, dtype=np.int32)
            valid[0] = capacity
            valid = jax.device_put(valid, batch_sharding)
            jax.block_until_ready(self.encode(codes, valid))


# ---------------------------------------------------------------------------
# Top-level VLM (backbone + vision tower + in-model contract)
# ---------------------------------------------------------------------------


class _MiMoV2MultimodalMixin(InModelMultimodalContract):
    """Wire the MiMoV2 vision + audio towers onto the in-model multimodal contract."""

    def __init__(self, config, mesh: Mesh | None = None, dtype: jnp.dtype = jnp.bfloat16):
        if mesh is None:
            raise ValueError("MiMoV2 multimodal models require a device mesh.")
        super().__init__(config, mesh, dtype)

        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        vision_config = _value(config, "vision_config", None)
        audio_config = _value(config, "audio_config", None)
        if vision_config is None and audio_config is None:
            raise ValueError("MiMoV2 VLM requires config.vision_config or config.audio_config.")

        self.encoder_tp = resolve_encoder_tp(
            mesh, global_server_args_dict.get("vision_encoder_parallel", "dp")
        )
        self.vision_tp = self.encoder_tp
        self.visual = None
        if vision_config is not None:
            input_buckets = tuple(
                resolve_vision_patch_buckets(
                    global_server_args_dict.get("precompile_vision_patch_paddings")
                )
            )
            self.visual = MiMoVisionTransformer(
                vision_config,
                self.dtype,
                nnx.Rngs(0),
                mesh,
                self.encoder_tp,
                input_buckets,
            )
        self.audio_encoder = (
            MiMoAudioEncoder(audio_config, self.dtype, mesh, self.encoder_tp)
            if audio_config is not None
            else None
        )

    # -- contract ---------------------------------------------------------

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.get_input_embeddings()

    def precompile_multimodal(self) -> None:
        if self.visual is not None:
            self.visual.precompile()
        if self.audio_encoder is not None:
            self.audio_encoder.precompile()

    def get_multimodal_embedding_packed_capacities(self) -> tuple[int, ...]:
        capacities: list[int] = []
        if self.visual is not None:
            rows = encoder_num_lanes(self.mesh, self.visual.vision_tp)
            unit = self.visual.spatial_merge_unit
            capacities.extend(rows * bucket // unit for bucket in self.visual.input_buckets)
        if self.audio_encoder is not None:
            rows = encoder_num_lanes(self.mesh, self.audio_encoder.encoder_tp)
            group = self.audio_encoder.group_size
            capacities.extend(rows * bucket // group for bucket in self.audio_encoder.input_buckets)
        return tuple(capacities)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        return self._get_visual_feature(items)

    def get_video_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        return self._get_visual_feature(items)

    def _get_visual_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        num_lanes = encoder_num_lanes(self.mesh, self.visual.vision_tp)
        return run_mrope_vision_model(
            self.visual,
            items,
            mesh=self.mesh,
            num_lanes=num_lanes,
            buckets=self.visual.input_buckets,
            merge_unit=self.visual.spatial_merge_unit,
            rope_type="rope_3d",
        )

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        encoder = self.audio_encoder
        packed = pack_lanes(
            items,
            encoder_num_lanes(self.mesh, encoder.encoder_tp),
            buckets=encoder.input_buckets,
            merge_unit=encoder.group_size,
            dtype=np.int32,
        )
        batch_sharding = encoder.specs.sharding(encoder.specs.batch_axis)
        output = encoder.encode(
            jax.device_put(packed.features, batch_sharding),
            jax.device_put(packed.valid, batch_sharding),
        )
        return restore_encoder_output(output, packed.output_indices, self.mesh)

    def get_multimodal_encode_funcs(self):
        funcs = {}
        if self.visual is not None:
            funcs[Modality.IMAGE] = self.get_image_feature
            funcs[Modality.MULTI_IMAGES] = self.get_image_feature
            funcs[Modality.VIDEO] = self.get_video_feature
        if self.audio_encoder is not None:
            funcs[Modality.AUDIO] = self.get_audio_feature
        return funcs

    # -- weights ----------------------------------------------------------

    def load_weights(self, model_config: ModelConfig) -> None:
        super().load_weights(model_config)
        vision = _value(self.config, "vision_config", None)
        heads = int(_value(vision, "num_heads", 1))
        kv_heads = int(_value(vision, "num_key_value_heads", heads) or heads)
        head_dim = int(_value(vision, "qk_channels", 64))
        tower_config = SimpleNamespace(
            model_path=model_config.model_path,
            quantization_config=None,
            hf_config=self.config,
            hf_text_config=SimpleNamespace(head_dim=head_dim, v_head_dim=head_dim),
            num_attention_heads=heads,
            hidden_size=heads * head_dim,
            get_total_num_kv_heads=lambda: kv_heads,
            _dummy_mode=getattr(model_config, "_dummy_mode", False),
        )
        loader = WeightLoader(self, tower_config, self.mesh, self.dtype)
        mappings: dict[str, WeightMapping] = {}
        if self.visual is not None:
            mappings.update(self._vision_weight_mappings())
        if self.audio_encoder is not None:
            mappings.update(self._audio_weight_mappings())
        if not mappings:
            return
        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(mappings)
        else:
            loader.load_weights_from_safetensors(mappings)
        logger.info("MiMoV2 multimodal tower weights loaded.")

    @staticmethod
    def _linear_mappings(source, target, sharding) -> dict[str, WeightMapping]:
        return {
            f"{source}.weight": WeightMapping(
                target_path=f"{target}.weight", sharding=sharding, transpose=True
            ),
            f"{source}.bias": WeightMapping(
                target_path=f"{target}.bias", sharding=(sharding[-1],), transpose=False
            ),
        }

    def _vision_weight_mappings(self) -> dict[str, WeightMapping]:
        specs = self.visual.specs
        col, row = specs.col_kernel_axes, specs.row_kernel_axes
        mappings: dict[str, WeightMapping] = {
            "visual.patch_embed.proj.weight": WeightMapping(
                target_path="visual.patch_embed.proj.kernel",
                sharding=(None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "visual.merger.ln_q.weight": WeightMapping(
                target_path="visual.merger.ln_q.scale", sharding=(None,), transpose=False
            ),
            "visual.merger.ln_q.bias": WeightMapping(
                target_path="visual.merger.ln_q.bias", sharding=(None,), transpose=False
            ),
        }
        mappings.update(self._linear_mappings("visual.merger.mlp.0", "visual.merger.mlp_fc1", col))
        mappings.update(self._linear_mappings("visual.merger.mlp.2", "visual.merger.mlp_fc2", row))
        for index, block in enumerate(self.visual.blocks):
            src = tgt = f"visual.blocks.{index}"
            for norm in ("norm1", "norm2"):
                mappings[f"{src}.{norm}.weight"] = WeightMapping(
                    target_path=f"{tgt}.{norm}.scale", sharding=(None,), transpose=False
                )
            mappings[f"{src}.attn.qkv.weight"] = WeightMapping(
                target_path=[f"{tgt}.attn.{n}_proj.weight" for n in ("q", "k", "v")],
                sharding=col,
                transpose=True,
            )
            mappings[f"{src}.attn.qkv.bias"] = WeightMapping(
                target_path=[f"{tgt}.attn.{n}_proj.bias" for n in ("q", "k", "v")],
                sharding=(col[-1],),
                transpose=False,
            )
            mappings.update(self._linear_mappings(f"{src}.attn.proj", f"{tgt}.attn.proj", row))
            for name in ("gate_proj", "up_proj"):
                mappings.update(
                    self._linear_mappings(f"{src}.mlp.{name}", f"{tgt}.mlp.{name}", col)
                )
            mappings.update(
                self._linear_mappings(f"{src}.mlp.down_proj", f"{tgt}.mlp.down_proj", row)
            )
            if block.attn.sinks is not None:
                mappings[f"{src}.attn.sinks"] = WeightMapping(
                    target_path=f"{tgt}.attn.sinks",
                    sharding=(specs.tensor_axis,),
                    transpose=False,
                )
        return mappings

    def _audio_weight_mappings(self) -> dict[str, WeightMapping]:
        encoder = self.audio_encoder
        mappings: dict[str, WeightMapping] = {}
        # Per-channel speech code embeddings live at top level in the checkpoint.
        for index in range(encoder.channels):
            mappings[f"speech_embeddings.{index}.weight"] = WeightMapping(
                target_path=f"audio_encoder.speech_embeddings.{index}.embedding",
                sharding=(None, None),
                transpose=False,
            )
        src_root = "audio_encoder.input_local_transformer"
        tgt_root = "audio_encoder.transformer"
        if encoder.transformer.norm is not None:
            mappings[f"{src_root}.norm.weight"] = WeightMapping(
                target_path=f"{tgt_root}.norm.scale", sharding=(None,), transpose=False
            )
        for index in range(len(encoder.transformer.layers)):
            src = f"{src_root}.layers.{index}"
            tgt = f"{tgt_root}.layers.{index}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                mappings[f"{src}.{norm}.weight"] = WeightMapping(
                    target_path=f"{tgt}.{norm}.scale", sharding=(None,), transpose=False
                )
            for name in ("q_proj", "k_proj", "v_proj"):
                mappings.update(
                    self._linear_mappings(
                        f"{src}.self_attn.{name}", f"{tgt}.self_attn.{name}", (None, None)
                    )
                )
            mappings[f"{src}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{tgt}.self_attn.o_proj.weight", sharding=(None, None), transpose=True
            )
            for name in ("gate_proj", "up_proj", "down_proj"):
                mappings[f"{src}.mlp.{name}.weight"] = WeightMapping(
                    target_path=f"{tgt}.mlp.{name}.weight", sharding=(None, None), transpose=True
                )
        if encoder.proj_fc2 is None:
            mappings["audio_encoder.projection.weight"] = WeightMapping(
                target_path="audio_encoder.proj_fc1.weight", sharding=(None, None), transpose=True
            )
        else:
            mappings["audio_encoder.projection.mlp.0.weight"] = WeightMapping(
                target_path="audio_encoder.proj_fc1.weight", sharding=(None, None), transpose=True
            )
            mappings["audio_encoder.projection.mlp.2.weight"] = WeightMapping(
                target_path="audio_encoder.proj_fc2.weight", sharding=(None, None), transpose=True
            )
        return mappings


class MiMoV2FlashForConditionalGeneration(_MiMoV2MultimodalMixin, MiMoV2FlashForCausalLM):
    pass


class MiMoV2ForConditionalGeneration(_MiMoV2MultimodalMixin, MiMoV2ForCausalLM):
    pass


EntryClass = [MiMoV2FlashForConditionalGeneration, MiMoV2ForConditionalGeneration]
