import glob
import math
import os
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.lax import Precision
from safetensors import safe_open
from transformers import modeling_flax_utils

if TYPE_CHECKING:
    from sgl_jax.srt.utils.weight_utils import WeightMapping



def apply_rotary_pos_emb_vision(
    query: jax.Array,
    key: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    original_query_dtype = query.dtype
    original_key_dtype = key.dtype
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    if cos.shape[-1] * 2 == query.shape[-1]:
        cos = jnp.concatenate([cos, cos], axis=-1)
        sin = jnp.concatenate([sin, sin], axis=-1)
    cos = cos[:, None, :].astype(jnp.float32)
    sin = sin[:, None, :].astype(jnp.float32)

    def rotate_half(x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query.astype(original_query_dtype), key.astype(original_key_dtype)


def mimo_vision_rot_pos_emb(
    grid_thw: tuple[tuple[int, int, int], ...],
    spatial_merge_size: int,
    rotary_pos_emb: Callable[[int], jax.Array],
) -> jax.Array:
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids, wpos_ids = jnp.indices((h, w))
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = jnp.transpose(hpos_ids, (0, 2, 1, 3)).flatten()
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = jnp.transpose(wpos_ids, (0, 2, 1, 3)).flatten()
        pos_ids.append(jnp.tile(jnp.stack([hpos_ids, wpos_ids], axis=-1), (t, 1)))

    pos_ids = jnp.concatenate(pos_ids, axis=0)
    max_grid_size = max(max(h, w) for _, h, w in grid_thw)
    rotary_pos_emb_full = rotary_pos_emb(max_grid_size)
    return rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)


def mimo_vision_get_window_index_1d(
    grid_thw: tuple[tuple[int, int, int], ...],
    spatial_merge_size: int,
    col: bool = True,
) -> jax.Array:
    window_index = []
    window_index_id = 0
    for grid_t, grid_h, grid_w in grid_thw:
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = jnp.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t,
            llm_grid_h,
            llm_grid_w,
        )
        index_new = jnp.transpose(index, (0, 2, 1)).reshape(-1) if col else index.reshape(-1)
        window_index.append(index_new + window_index_id)
        window_index_id += grid_t * llm_grid_h * llm_grid_w
    return jnp.concatenate(window_index, axis=0)


def mimo_vision_apply_index(
    tensor: jax.Array,
    index: jax.Array,
    spatial_merge_size: int,
) -> jax.Array:
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    tensor = tensor.reshape(-1, spatial_merge_unit, *tensor.shape[1:])
    tensor = tensor[index]
    return tensor.reshape(-1, *tensor.shape[2:])


class MiMoVisionPatchEmbed(nnx.Module):
    """MiMo vision 3D patch embedding.

    The input is already pre-extracted into flattened patch rows:
        [num_patches, C * temporal_patch_size * patch_size * patch_size]

    This mirrors the Hugging Face MiMo Conv3d patch embed while adapting the
    tensor layout for Flax's channels-last Conv.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size

        _rngs = rngs or nnx.Rngs(0)
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=_rngs,
            precision=Precision.HIGHEST,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        patch_dim = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        if x.shape[-1] != patch_dim:
            raise ValueError(
                f"Expected flattened patch dim {patch_dim}, got input shape {x.shape}."
            )

        x = x.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        return x.reshape(-1, self.embed_dim)


class MiMoVisionRotaryEmbedding(nnx.Module):

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seq_len: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seq_len, dtype=jnp.float32)
        return jnp.outer(seq, inv_freq)


class MiMoVisionAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_sink: bool = False,
        window_size: int | tuple[int, int] = -1,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_sink = use_sink
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(head_dim)

        _rngs = rngs or nnx.Rngs(0)
        qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv = nnx.Linear(
            hidden_size,
            qkv_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.proj = nnx.Linear(
            num_heads * head_dim,
            hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.sinks = nnx.Param(jnp.zeros((num_heads,), dtype=dtype)) if use_sink else None

    def __call__(
        self,
        hidden_states: jax.Array,
        cu_seqlens: jax.Array,
        position_embeddings: tuple[jax.Array, jax.Array],
        full_attn: bool = False,
    ) -> jax.Array:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query = qkv[:, :q_size].reshape(seq_len, self.num_heads, self.head_dim)
        key = qkv[..., q_size : q_size + kv_size].reshape(seq_len, self.num_kv_heads, self.head_dim)
        value = qkv[..., q_size + kv_size :].reshape(seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)

        lengths = np.asarray(cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        query_chunks = jnp.split(query, np.cumsum(lengths)[:-1], axis=0)
        key_chunks = jnp.split(key, np.cumsum(lengths)[:-1], axis=0)
        value_chunks = jnp.split(value, np.cumsum(lengths)[:-1], axis=0)

        outputs = []
        for query_chunk, key_chunk, value_chunk in zip(query_chunks, key_chunks, value_chunks):
            query_chunk = jnp.transpose(query_chunk[None, ...], (0, 2, 1, 3))
            key_chunk = jnp.transpose(key_chunk[None, ...], (0, 2, 1, 3))
            value_chunk = jnp.transpose(value_chunk[None, ...], (0, 2, 1, 3))

            if self.num_heads != self.num_kv_heads:
                num_groups = self.num_heads // self.num_kv_heads
                key_chunk = jnp.repeat(key_chunk, num_groups, axis=1)
                value_chunk = jnp.repeat(value_chunk, num_groups, axis=1)

            attn_weights = jnp.einsum("bnth,bnsh->bnts", query_chunk, key_chunk) * self.scale
            window_size = (
                self.window_size[0] if isinstance(self.window_size, tuple) else self.window_size
            )
            if not full_attn and window_size > 0:
                chunk_len = query_chunk.shape[2]
                positions = jnp.arange(chunk_len)
                distance = jnp.abs(positions[:, None] - positions[None, :])
                mask = distance > window_size
                attn_weights = jnp.where(
                    mask[None, None, :, :],
                    jnp.finfo(attn_weights.dtype).min,
                    attn_weights,
                )
            if self.use_sink and self.sinks is not None:
                sink = jnp.broadcast_to(
                    self.sinks[...][None, :, None].astype(attn_weights.dtype),
                    (1, self.num_heads, query_chunk.shape[2]),
                )
                attn_weights = attn_weights.at[..., 0].add(sink)
            attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(
                hidden_states.dtype
            )
            attn_output = jnp.einsum("bnts,bnsh->bnth", attn_weights, value_chunk)
            outputs.append(jnp.transpose(attn_output[0], (1, 0, 2)))

        attn_output = jnp.concatenate(outputs, axis=0)
        attn_output = attn_output.reshape(seq_len, self.num_heads * self.head_dim)
        return self.proj(attn_output)


class MiMoVisionSwiGLUMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.act_fn = modeling_flax_utils.ACT2FN[hidden_act]

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiMoVisionBlock(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_act: str = "silu",
        norm_eps: float = 1e-6,
        use_sink: bool = False,
        window_size: int | tuple[int, int] = -1,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        self.norm1 = nnx.RMSNorm(
            hidden_size,
            epsilon=norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.norm2 = nnx.RMSNorm(
            hidden_size,
            epsilon=norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.attn = MiMoVisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_sink=use_sink,
            window_size=window_size,
            dtype=dtype,
            rngs=_rngs,
        )
        self.mlp = MiMoVisionSwiGLUMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            dtype=dtype,
            rngs=_rngs,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        cu_seqlens: jax.Array,
        position_embeddings: tuple[jax.Array, jax.Array],
        full_attn: bool = False,
    ) -> jax.Array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            full_attn=full_attn,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiMoVisionPatchMerger(nnx.Module):
    def __init__(
        self,
        out_hidden_size: int,
        context_dim: int,
        norm_eps: float,
        spatial_merge_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        _rngs = rngs or nnx.Rngs(0)

        self.ln_q = nnx.LayerNorm(
            context_dim,
            epsilon=norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            rngs=_rngs,
        )
        self.mlp_fc1 = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = nnx.Linear(
            self.hidden_size,
            out_hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.ln_q(x)
        x = x.reshape(-1, self.hidden_size)
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        return self.mlp_fc2(x)


class MiMoVisionTransformer(nnx.Module):
    def __init__(
        self,
        config,
        norm_eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.spatial_merge_size = int(config.spatial_merge_size)
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.fullatt_block_indexes = set(getattr(config, "fullatt_block_indexes", []) or [])
        self.vit_window_attn_types = list(
            getattr(config, "vit_window_attn_types", [0] * int(config.depth))
        )
        self.visual_token_window_size = int(getattr(config, "visual_token_window_size", -1))

        num_heads = int(config.num_heads)
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = int(config.qk_channels)
        self.rotary_pos_emb = MiMoVisionRotaryEmbedding(head_dim // 2)

        self.patch_embed = MiMoVisionPatchEmbed(
            in_channels=int(config.in_channels),
            embed_dim=self.hidden_size,
            patch_size=int(config.patch_size),
            temporal_patch_size=int(config.temporal_patch_size),
            dtype=dtype,
            rngs=_rngs,
        )
        use_sink = bool(getattr(config, "use_sink", False))
        self.blocks = nnx.List(
            [
                MiMoVisionBlock(
                    hidden_size=self.hidden_size,
                    intermediate_size=int(config.intermediate_size),
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    hidden_act=getattr(config, "hidden_act", "silu"),
                    norm_eps=norm_eps,
                    use_sink=use_sink and i not in self.fullatt_block_indexes,
                    window_size=self.visual_token_window_size,
                    dtype=dtype,
                    rngs=_rngs,
                )
                for i in range(int(config.depth))
            ]
        )
        self.merger = MiMoVisionPatchMerger(
            out_hidden_size=int(config.out_hidden_size),
            context_dim=self.hidden_size,
            norm_eps=norm_eps,
            spatial_merge_size=self.spatial_merge_size,
            dtype=dtype,
            rngs=_rngs,
        )

    def _prepare_forward(self, pixel_values: jax.Array, grid_thw: tuple[tuple[int, int, int], ...]):
        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb = mimo_vision_rot_pos_emb(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            rotary_pos_emb=self.rotary_pos_emb,
        )
        col_index = mimo_vision_get_window_index_1d(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            col=True,
        )
        reverse_col_index = jnp.argsort(col_index)
        seq_lens = jnp.asarray([h * w for t, h, w in grid_thw for _ in range(t)], dtype=jnp.int32)
        cu_seqlens = jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(seq_lens, dtype=jnp.int32)]
        )
        row_embeddings = (jnp.cos(rotary_pos_emb), jnp.sin(rotary_pos_emb))
        col_rotary_pos_emb = mimo_vision_apply_index(
            rotary_pos_emb,
            col_index,
            spatial_merge_size=self.spatial_merge_size,
        )
        col_embeddings = (jnp.cos(col_rotary_pos_emb), jnp.sin(col_rotary_pos_emb))
        return (
            hidden_states,
            row_embeddings,
            col_embeddings,
            col_index,
            reverse_col_index,
            cu_seqlens,
        )

    def run_blocks(
        self,
        hidden_states: jax.Array,
        row_embeddings: tuple[jax.Array, jax.Array],
        col_embeddings: tuple[jax.Array, jax.Array],
        col_index: jax.Array,
        reverse_col_index: jax.Array,
        cu_seqlens: jax.Array,
    ) -> jax.Array:
        for layer_idx, block in enumerate(self.blocks):
            window_attn_type = self.vit_window_attn_types[layer_idx]
            if window_attn_type == 1 and (
                layer_idx == 0 or self.vit_window_attn_types[layer_idx - 1] != 1
            ):
                hidden_states = mimo_vision_apply_index(
                    hidden_states,
                    col_index,
                    spatial_merge_size=self.spatial_merge_size,
                )
            if (
                layer_idx > 0
                and window_attn_type != 1
                and self.vit_window_attn_types[layer_idx - 1] == 1
            ):
                hidden_states = mimo_vision_apply_index(
                    hidden_states,
                    reverse_col_index,
                    spatial_merge_size=self.spatial_merge_size,
                )
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=col_embeddings if window_attn_type == 1 else row_embeddings,
                full_attn=layer_idx in self.fullatt_block_indexes,
            )
        return self.merger(hidden_states)

    def __call__(self, pixel_values: jax.Array, grid_thw: tuple[tuple[int, int, int], ...]):
        return self.run_blocks(*self._prepare_forward(pixel_values, grid_thw))

    def load_weights_from_safetensors(self, model_path: str, config=None) -> None:
        load_weights_from_safetensors(self, model_path, config or self.config)




def load_weights_from_safetensors(model: nnx.Module, model_path: str, config) -> None:
    from sgl_jax.srt.multimodal.models.mimo_v2_5.weights_mapping import (
        create_mimo_vision_weight_mappings,
    )

    weight_index = _index_safetensors(model_path)
    mappings = create_mimo_vision_weight_mappings(config)
    zero_filled: list[str] = []
    non_bias_missing: list[str] = []
    for hf_key, mapping in mappings.items():
        if hf_key not in weight_index:
            if not _has_param(model, mapping.target_path):
                continue
            if not hf_key.endswith(".bias"):
                non_bias_missing.append(hf_key)
                continue
            _zero_param(model, mapping.target_path)
            zero_filled.append(hf_key)
            continue
        weight = _load_weight(weight_index[hf_key], hf_key)
        if mapping.transpose_axes is not None:
            weight = jnp.transpose(weight, mapping.transpose_axes)
        elif mapping.transpose:
            weight = jnp.transpose(weight, (1, 0))
        _set_param(model, mapping.target_path, weight)

    if non_bias_missing:
        raise AssertionError(
            f"Missing non-bias MiMo vision weights from checkpoint: {non_bias_missing}"
        )
    if zero_filled:
        warnings.warn(
            f"MiMo vision: zero-filled {len(zero_filled)} missing bias weights "
            f"to mirror HF behavior: {zero_filled}",
            stacklevel=2,
        )


def _index_safetensors(model_path: str) -> dict[str, str]:
    index = {}
    for filename in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(filename, framework="np", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                index[key] = filename
    return index


def _load_weight(filename: str, key: str) -> jnp.ndarray:
    with safe_open(filename, framework="np", device="cpu") as handle:
        return jnp.asarray(handle.get_tensor(key))


def _set_param(model: nnx.Module, target_path: str | list[str], weight: jnp.ndarray) -> None:
    target = _resolve_param(model, target_path)
    target[...] = weight.astype(target.dtype)


def _zero_param(model: nnx.Module, target_path: str | list[str]) -> None:
    target = _resolve_param(model, target_path)
    target[...] = jnp.zeros_like(target[...])


def _resolve_param(model: nnx.Module, target_path: str | list[str]):
    if not isinstance(target_path, str):
        raise TypeError(f"MiMo vision loader expects a single target path, got {target_path}")

    target = model
    for part in target_path.split("."):
        target = target[int(part)] if part.isdigit() else getattr(target, part)

    if not isinstance(target, nnx.Variable):
        raise TypeError(f"{target_path} does not point to an NNX variable")
    return target


def _has_param(model: nnx.Module, target_path: str | list[str]) -> bool:
    try:
        _resolve_param(model, target_path)
    except (AttributeError, TypeError, IndexError, KeyError):
        return False
    return True
