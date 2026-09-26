"""MiMoV2 vision tower."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.configs.mimo import config_value
from sgl_jax.srt.layers.activation import ACT2FN
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.in_model.lane_packing import encoder_num_lanes
from sgl_jax.srt.multimodal.layers.vision_sharding import VisionShardSpecs


def _apply_rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    """Apply RoPE in float32, broadcasting over heads."""
    original_dtype = x.dtype
    x = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    rotated = jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    cos = jnp.cos(freqs)[..., None, :]
    sin = jnp.sin(freqs)[..., None, :]
    return (x * cos + rotated * sin).astype(original_dtype)


class MiMoVisionPatchEmbed(nnx.Module):
    """3D (temporal × spatial) patch embedding conv."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp):
        self.temporal_patch_size = int(config_value(config, "temporal_patch_size", 2))
        self.patch_size = int(config_value(config, "patch_size", 16))
        self.in_channels = int(config_value(config, "in_channels", 3))
        self.hidden_size = int(config_value(config, "hidden_size"))
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
        tokens = x.shape[0]
        C, T, P = self.in_channels, self.temporal_patch_size, self.patch_size
        sharding = self.specs.sharding(self.specs.batch_axis)
        x = x.reshape(tokens, C, T, P, P, out_sharding=sharding)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x, out_sharding=sharding)
        return x.reshape(tokens, self.hidden_size, out_sharding=sharding)


class MiMoVisionMLP(nnx.Module):
    """SwiGLU MLP with bias."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp):
        hidden = int(config_value(config, "hidden_size"))
        intermediate = int(config_value(config, "intermediate_size"))
        self.specs = VisionShardSpecs(mesh, vision_tp)
        act = config_value(config, "hidden_act", "silu")
        self.act_fn = ACT2FN[act]

        def linear(i, o, axes):
            return LinearBase(i, o, mesh=mesh, use_bias=True, kernel_axes=axes, params_dtype=dtype)

        self.gate_proj = linear(hidden, intermediate, self.specs.col_kernel_axes)
        self.up_proj = linear(hidden, intermediate, self.specs.col_kernel_axes)
        self.down_proj = linear(intermediate, hidden, self.specs.row_kernel_axes)

    def __call__(self, x: jax.Array) -> jax.Array:
        specs = self.specs
        col = specs.sharding(specs.batch_axis, specs.tensor_axis)
        row = specs.sharding(specs.batch_axis)
        gate, _ = self.gate_proj(x, out_sharding=col)
        up, _ = self.up_proj(x, out_sharding=col)
        out, _ = self.down_proj(self.act_fn(gate) * up, out_sharding=row)
        return out


class MiMoVisionAttention(nnx.Module):
    """ViT self-attention: split QKV, GQA, RoPE, packed-cu sparse attention."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp, use_sinks):
        hidden = int(config_value(config, "hidden_size"))
        self.num_heads = int(config_value(config, "num_heads"))
        self.num_kv_heads = int(
            config_value(config, "num_key_value_heads", self.num_heads) or self.num_heads
        )
        self.head_dim = int(config_value(config, "qk_channels", 64))
        if self.num_heads % self.num_kv_heads:
            raise ValueError("MiMoV2 vision num_heads must be divisible by num_key_value_heads.")
        if self.head_dim % 4:
            raise ValueError("MiMoV2 vision head_dim must be divisible by 4.")
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)

        def projection(heads):
            return LinearBase(
                hidden,
                heads * self.head_dim,
                mesh=mesh,
                use_bias=True,
                kernel_axes=self.specs.col_kernel_axes,
                params_dtype=dtype,
            )

        self.q_proj = projection(self.num_heads)
        self.k_proj = projection(self.num_kv_heads)
        self.v_proj = projection(self.num_kv_heads)
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
                    out_sharding=NamedSharding(mesh, sink_spec),
                )
            )
            if use_sinks
            else None
        )

        # make_vision_attention_backend(use_varlen=True) falls back to
        # VisionFlashAttentionBackend on CPU, which accepts neither attention_sink
        # nor window_size. Select TPU varlen and our CPU reference explicitly so
        # both paths preserve MiMo's sink and sliding-window semantics.
        if mesh.devices.flat[0].platform == "tpu":
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

    def __call__(self, x, freqs, cu_seqlens, window_size, *, max_seq_len):
        tokens = x.shape[0]
        specs = self.specs
        col = specs.sharding(specs.batch_axis, specs.tensor_axis)
        q, _ = self.q_proj(x, out_sharding=col)
        k, _ = self.k_proj(x, out_sharding=col)
        v, _ = self.v_proj(x, out_sharding=col)
        q = q.reshape(tokens, self.num_heads, self.head_dim, out_sharding=col)
        k = k.reshape(tokens, self.num_kv_heads, self.head_dim, out_sharding=col)
        v = v.reshape(tokens, self.num_kv_heads, self.head_dim, out_sharding=col)
        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)
        # SGLang disables local attention when the block has no sink.
        window = (-1, -1) if self.sinks is None or window_size < 0 else (window_size, window_size)
        sinks = self.sinks[...] if self.sinks is not None else None
        if self.attn_backend is None:
            out = self._reference_attention(q, k, v, cu_seqlens, window, sinks)
        else:
            out = self.attn_backend(
                q,
                k,
                v,
                cu_seqlens,
                attention_sink=sinks,
                window_size=window,
                max_seq_len=max_seq_len,
            )
        out = out.reshape(tokens, self.num_heads * self.head_dim, out_sharding=col)
        out, _ = self.proj(out, out_sharding=specs.sharding(specs.batch_axis))
        return out

    def _reference_attention(self, q, k, v, cu_seqlens, window, sinks=None):
        """Dense CPU reference with the same per-shard contract as TPU varlen."""
        from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
            vision_segment_ids_from_cu_seqlens,
        )

        def attend(q, k, v, cu, sink):
            k = jnp.repeat(k, q.shape[1] // k.shape[1], axis=1)
            v = jnp.repeat(v, q.shape[1] // v.shape[1], axis=1)
            ids = vision_segment_ids_from_cu_seqlens(cu[None], q.shape[0]).q[0]
            positions = jnp.arange(q.shape[0])
            mask = (ids[:, None] == ids[None, :]) & (ids[None, :] >= 0)
            if window[0] >= 0:
                mask &= positions[None, :] >= positions[:, None] - window[0]
            if window[1] >= 0:
                mask &= positions[None, :] <= positions[:, None] + window[1]
            # Varlen never reads padding; the dense reference must mask values
            # explicitly because the packer leaves unused feature rows undefined.
            v = jnp.where((ids >= 0)[:, None, None], v, 0)
            logits = jnp.einsum("thd,shd->hts", q.astype(jnp.float32), k.astype(jnp.float32))
            logits *= 1.0 / math.sqrt(self.head_dim)
            logits = jnp.where(mask[None], logits, -jnp.inf)
            if sink is not None:
                # The sink is a value-less softmax slot, never a bias on a real key.
                sink_logits = jnp.broadcast_to(sink[:, None, None], (*logits.shape[:-1], 1))
                logits = jnp.concatenate((logits, sink_logits.astype(jnp.float32)), axis=-1)
            probs = jax.nn.softmax(logits, axis=-1)[..., : q.shape[0]]
            probs = jnp.where((ids >= 0)[None, :, None], probs, 0)
            return jnp.einsum("hts,shd->thd", probs, v.astype(jnp.float32)).astype(q.dtype)

        spec = PartitionSpec(self.specs.batch_axis, self.specs.tensor_axis)
        return jax.shard_map(
            attend,
            mesh=self.mesh,
            in_specs=(
                spec,
                spec,
                spec,
                PartitionSpec(self.specs.batch_axis),
                PartitionSpec(self.specs.tensor_axis) if sinks is not None else None,
            ),
            out_specs=spec,
            check_vma=False,
        )(q, k, v, cu_seqlens, sinks)


class MiMoVisionBlock(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, vision_tp, use_sinks):
        hidden = int(config_value(config, "hidden_size"))
        eps = float(config_value(config, "rms_norm_eps", 1e-6))
        _rngs = rngs or nnx.Rngs(0)
        self.norm1 = nnx.RMSNorm(hidden, epsilon=eps, dtype=dtype, param_dtype=dtype, rngs=_rngs)
        self.norm2 = nnx.RMSNorm(hidden, epsilon=eps, dtype=dtype, param_dtype=dtype, rngs=_rngs)
        self.attn = MiMoVisionAttention(config, dtype, rngs, mesh, vision_tp, use_sinks)
        self.mlp = MiMoVisionMLP(config, dtype, rngs, mesh, vision_tp)

    def __call__(self, x, freqs, cu_seqlens, window_size, *, max_seq_len):
        x = x + self.attn(
            self.norm1(x),
            freqs,
            cu_seqlens,
            window_size,
            max_seq_len=max_seq_len,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MiMoVisionPatchMerger(nnx.Module):
    """RMSNorm → spatial merge → MLP over flat output tokens."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp):
        context = int(config_value(config, "hidden_size"))
        self.unit = int(config_value(config, "spatial_merge_size", 2)) ** 2
        self.hidden_size = context * self.unit
        self.specs = VisionShardSpecs(mesh, vision_tp)
        _rngs = rngs or nnx.Rngs(0)
        self.ln_q = nnx.RMSNorm(
            context,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.mlp_fc1 = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=False,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.mlp_fc2 = LinearBase(
            self.hidden_size,
            int(config_value(config, "out_hidden_size")),
            mesh=mesh,
            use_bias=False,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        specs = self.specs
        row = specs.sharding(specs.batch_axis)
        x = self.ln_q(x)
        x = x.reshape(-1, self.hidden_size, out_sharding=row)
        x, _ = self.mlp_fc1(x, out_sharding=specs.sharding(specs.batch_axis, specs.tensor_axis))
        x = jax.nn.gelu(x, approximate=False)
        x, _ = self.mlp_fc2(x, out_sharding=row)
        return x


class MiMoVisionTransformer(nnx.Module):
    """MiMoV2 ViT: patch embed → windowed/full blocks (col reorder) → merge."""

    def __init__(self, config, dtype, rngs, mesh, vision_tp):
        self.config = config
        self.mesh = mesh
        self.vision_tp = vision_tp
        self.specs = VisionShardSpecs(mesh, vision_tp)
        self.dtype = dtype

        self.spatial_merge_size = int(config_value(config, "spatial_merge_size", 2))
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.patch_size = int(config_value(config, "patch_size", 16))
        self.temporal_patch_size = int(config_value(config, "temporal_patch_size", 2))
        self.in_channels = int(config_value(config, "in_channels", 3))
        self.patch_dim = self.in_channels * self.temporal_patch_size * self.patch_size**2
        self.head_dim = int(config_value(config, "qk_channels", 64))
        self.theta = float(config_value(config, "rope_theta", 10000.0))

        depth = int(config_value(config, "depth"))
        full = tuple(int(i) for i in (config_value(config, "fullatt_block_indexes", ()) or ()))
        self.full_blocks = frozenset(full)
        self.window_types = tuple(
            config_value(config, "vit_window_attn_types", None) or [-1] * depth
        )
        if len(self.window_types) != depth:
            raise ValueError("vit_window_attn_types must have one entry per vision block.")
        use_sink = bool(config_value(config, "use_sink", False))
        self.window_size = int(config_value(config, "visual_token_window_size", -1))

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

        self._metadata_cache: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}

    def __call__(self, patches, **metadata) -> jax.Array:
        return self.encode(patches, **metadata)

    def _reorder_units(self, x, indices):
        """Reorder spatial merge units within each device's lane, as in Qwen2.5-VL."""
        shape = x.shape
        sharding = self.specs.sharding(self.specs.batch_axis)
        units = x.reshape(-1, self.spatial_merge_unit, *shape[1:], out_sharding=sharding)
        spec = PartitionSpec(self.specs.batch_axis)
        units = jax.shard_map(
            lambda values, order: values[order],
            mesh=self.mesh,
            in_specs=(spec, spec),
            out_specs=spec,
            check_vma=False,
        )(units, indices)
        return units.reshape(shape, out_sharding=sharding)

    def _forward(self, patches, meta: dict[str, jax.Array]) -> jax.Array:
        col_index = jnp.asarray(meta["col_index"])
        reverse_col_index = jnp.asarray(meta["reverse_col_index"])
        rotary_freqs = jnp.asarray(meta["rotary_freqs"])
        cu_seqlens = jnp.asarray(meta["cu_seqlens"])

        hidden = self.patch_embed(patches)
        col_freqs = self._reorder_units(rotary_freqs, col_index)

        capacity = patches.shape[0] // encoder_num_lanes(self.mesh, self.vision_tp)
        for index, block in enumerate(self.blocks):
            col = self.window_types[index] == 1
            previous_col = index > 0 and self.window_types[index - 1] == 1
            if col and not previous_col:
                hidden = self._reorder_units(hidden, col_index)
            elif previous_col and not col:
                hidden = self._reorder_units(hidden, reverse_col_index)
            freqs = col_freqs if col else rotary_freqs
            window = -1 if index in self.full_blocks else self.window_size
            hidden = block(hidden, freqs, cu_seqlens, window, max_seq_len=capacity)

        return self.merger(hidden)

    def _metadata_for_grid(self, grid: tuple[int, int, int]) -> dict[str, np.ndarray]:
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
        reverse_col_index = np.argsort(col_index).astype(np.int32, copy=False)
        cu_seqlens = np.arange(0, (t + 1) * h * w, h * w, dtype=np.int32)
        meta = {
            "col_index": col_index,
            "reverse_col_index": reverse_col_index,
            "rotary_freqs": freqs,
            "cu_seqlens": cu_seqlens,
        }
        if len(self._metadata_cache) >= 64:
            self._metadata_cache.pop(next(iter(self._metadata_cache)))
        self._metadata_cache[grid] = meta
        return meta

    def _pack_metadata(self, grids: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
        col_indices, reverse_col_indices, freqs = [], [], []
        cu_seqlens = [0]
        unit_offset = 0
        patch_offset = 0
        for grid in grids:
            meta = self._metadata_for_grid(grid)
            col_indices.append(meta["col_index"] + unit_offset)
            reverse_col_indices.append(meta["reverse_col_index"] + unit_offset)
            freqs.append(meta["rotary_freqs"])
            cu_seqlens.extend((meta["cu_seqlens"][1:] + patch_offset).tolist())
            unit_offset += meta["col_index"].size
            patch_offset += int(np.prod(grid))
        return {
            "col_index": np.concatenate(col_indices),
            "reverse_col_index": np.concatenate(reverse_col_indices),
            "rotary_freqs": np.concatenate(freqs),
            "cu_seqlens": np.asarray(cu_seqlens, dtype=np.int32),
        }

    def _pad_metadata(
        self, meta: dict[str, np.ndarray], input_capacity: int
    ) -> dict[str, np.ndarray]:
        units = input_capacity // self.spatial_merge_unit
        col_index = np.arange(units, dtype=np.int32)
        col_index[: meta["col_index"].size] = meta["col_index"]
        reverse_col_index = np.arange(units, dtype=np.int32)
        reverse_col_index[: meta["reverse_col_index"].size] = meta["reverse_col_index"]
        freqs = np.zeros((input_capacity, self.head_dim), dtype=np.float32)
        freqs[: meta["rotary_freqs"].shape[0]] = meta["rotary_freqs"]
        boundary_capacity = units + 1
        cu_seqlens = np.full(boundary_capacity, meta["cu_seqlens"][-1], dtype=np.int32)
        cu_seqlens[: meta["cu_seqlens"].size] = meta["cu_seqlens"]
        return {
            "col_index": col_index,
            "reverse_col_index": reverse_col_index,
            "rotary_freqs": freqs,
            "cu_seqlens": cu_seqlens,
        }

    @jax.jit
    def encode(self, patches, *, meta) -> jax.Array:
        token_sharding = self.specs.sharding(self.specs.batch_axis)
        patches = patches.reshape(-1, self.patch_dim, out_sharding=token_sharding)
        return self._forward(patches.astype(self.dtype), meta)

    def prepare_metadata(self, grid_thw, capacity: int, *, sharding: NamedSharding):
        grid_thw = np.asarray(grid_thw, dtype=np.int32)
        if grid_thw.ndim == 2:
            grid_thw = grid_thw[None]
        meta = self._build_metadata(grid_thw, capacity)
        return jax.device_put({"meta": meta}, sharding)

    def _build_metadata(self, grid_thw: np.ndarray, capacity: int):
        metadata = []
        for lane in grid_thw:
            grids = [tuple(map(int, grid)) for grid in lane if np.any(grid)]
            if grids:
                meta = self._pad_metadata(self._pack_metadata(grids), capacity)
            else:
                merge = self.spatial_merge_size
                meta = self._pad_metadata(
                    self._metadata_for_grid((1, merge, capacity // merge)), capacity
                )
                meta["cu_seqlens"].fill(0)
            metadata.append(meta)
        # Indices and cumulative lengths remain lane-local, just as in Qwen-VL.
        return jax.tree.map(lambda *values: np.concatenate(values), *metadata)
