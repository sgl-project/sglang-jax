import logging
import math
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import modeling_flax_utils

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import MRotaryEmbedding, ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.managers.mm_utils import mm_encode
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2Model
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import (
    QwenVLModelVitConfig,
)
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_FLASH_MHA = None


def _get_flash_mha():
    global _FLASH_MHA
    if _FLASH_MHA is None:
        from flash_attn_jax import flash_mha as _FLASH_MHA
    return _FLASH_MHA


init_fn = nnx.initializers.uniform()


def apply_rotary_pos_emb_vision(x: jax.Array, rotary_pos_emb: jax.Array) -> jax.Array:
    _, _, _, H = x.shape
    half_dim = H // 2

    x_real = x[..., :half_dim]
    x_imag = x[..., half_dim:]

    cos_emb = jnp.cos(rotary_pos_emb)
    sin_emb = jnp.sin(rotary_pos_emb)

    cos_emb = cos_emb[None, :, None, :]
    sin_emb = sin_emb[None, :, None, :]

    x_rotated_real = x_real * cos_emb - x_imag * sin_emb
    x_rotated_imag = x_real * sin_emb + x_imag * cos_emb

    return jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)


def vision_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    window_size: int = -1,
    valid_token_count: jax.Array | None = None,
) -> jax.Array:
    """
    Compute vision attention using flash attention on GPU or native attention on TPU.

    This is a simple attention function for vision models (no KV cache, no causal masking).

    Args:
        q, k, v: Input tensors of shape [B, T, N, H] (batch, seq_len, num_heads, head_dim)
        scale: Attention scale factor (1/sqrt(head_dim))
        window_size: Window size for local attention. -1 means full attention.

    Returns:
        Output tensor of shape [B, T, N, H]
    """
    if not is_tpu_runtime() and valid_token_count is None:
        # GPU: use flash_mha
        flash_mha = _get_flash_mha()
        original_dtype = q.dtype
        if q.dtype not in [jnp.bfloat16, jnp.float16]:
            q = q.astype(jnp.bfloat16)
            k = k.astype(jnp.bfloat16)
            v = v.astype(jnp.bfloat16)

        if window_size > 0:
            output = flash_mha(
                q,
                k,
                v,
                softmax_scale=scale,
                is_causal=False,
                window_size=(window_size, window_size),
            )
        else:
            output = flash_mha(q, k, v, softmax_scale=scale, is_causal=False)

        if output.dtype != original_dtype:
            output = output.astype(original_dtype)
        return output
    else:
        # TPU: native attention
        B, T, N, H = q.shape
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, N, T, H]
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        attn_weights = jnp.einsum("bnth,bnsh->bnts", q, k) * scale

        if window_size > 0:
            # Create window mask for local attention
            positions = jnp.arange(T)
            distance = jnp.abs(positions[:, None] - positions[None, :])
            window_mask = distance > window_size
            attn_weights = jnp.where(
                window_mask[None, None, :, :], jnp.finfo(attn_weights.dtype).min, attn_weights
            )

        if valid_token_count is not None:
            valid_token_count = jnp.reshape(valid_token_count, ())[()]
            positions = jnp.arange(T)
            valid_queries = positions < valid_token_count
            valid_keys = positions < valid_token_count
            safe_key0 = positions == 0
            padding_mask = jnp.where(
                valid_queries[:, None],
                valid_keys[None, :],
                safe_key0[None, :],
            )
            attn_weights = jnp.where(
                padding_mask[None, None, :, :],
                attn_weights,
                jnp.finfo(attn_weights.dtype).min,
            )

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        output = jnp.einsum("bnts,bnsh->bnth", attn_weights, v)
        return jnp.transpose(output, (0, 2, 1, 3))  # [B, T, N, H]


class Qwen2_5_VisionPatchEmbed(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        kernel_size = (temporal_patch_size, patch_size, patch_size)

        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),  # Use dummy rngs if None (for eval_shape)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is (L, C * T * H * W)
        L, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size * self.patch_size)
        # Reshape to (L, T, H, W, C) for Conv3D with channels_last
        x = x.reshape(L, C, self.temporal_patch_size, self.patch_size, self.patch_size)
        # L,T,H,W,C
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        # After conv, shape is (L, T_out, H_out, W_out, C_out)
        # With stride=kernel_size, T_out=H_out=W_out=1.
        # So shape is (L, 1, 1, 1, hidden_size)
        x = x.reshape(L, self.hidden_size)
        return x


class Qwen2_5_VisionRotaryEmbedding(nnx.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seq_len: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)


class Qwen2_5_VLMLP(nnx.Module):
    def __init__(self, config: QwenVLModelVitConfig, dtype: jnp.dtype, rngs: nnx.Rngs = None):
        in_features = config.hidden_size
        hidden_features = config.intermediate_size
        act_fn = modeling_flax_utils.ACT2FN[config.hidden_act]

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.gate_proj = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.up_proj = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.down_proj = nnx.Linear(
            hidden_features,
            in_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.act_fn = act_fn

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        return self.down_proj(fuse)


class Qwen2_5_VisionAttention(nnx.Module):
    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self._window_token_size = (
            (config.window_size // config.spatial_merge_size // config.patch_size) ** 2
        ) * (config.spatial_merge_size**2)

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.qkv_proj = nnx.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

        self.proj = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_window_seqlens: jax.Array | None = None,
        use_fullattn: bool = True,
        valid_token_count: jax.Array | None = None,
    ) -> jax.Array:
        T, B, D = x.shape
        assert B == 1, "Vision attention currently only supports batch size 1"

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape: [T, B, D] -> [B, T, N, H]
        q = q.reshape(T, B, self.num_heads, self.head_dim).transpose(1, 0, 2, 3)
        k = k.reshape(T, B, self.num_heads, self.head_dim).transpose(1, 0, 2, 3)
        v = v.reshape(T, B, self.num_heads, self.head_dim).transpose(1, 0, 2, 3)

        # Apply rotary embeddings
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # Use static window size derived from config for JIT compatibility.
        window_size = -1
        if not use_fullattn and cu_window_seqlens is not None:
            window_size = self._window_token_size

        # Compute attention using the backend function
        output = vision_attention(
            q,
            k,
            v,
            self.scale,
            window_size,
            valid_token_count=valid_token_count,
        )

        # Reshape back: [B, T, N, H] -> [T, B, D]
        output = output.transpose(1, 0, 2, 3).reshape(T, B, D)

        return self.proj(output)


class Qwen2_5_VisionBlock(nnx.Module):
    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
    ):
        dim = config.hidden_size
        norm_layer = partial(
            nnx.RMSNorm,
            epsilon=config.rms_norm_eps,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=_rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=_rngs)
        self.attn = Qwen2_5_VisionAttention(config=config, dtype=dtype, rngs=rngs, mesh=mesh)
        self.mlp = Qwen2_5_VLMLP(config=config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_window_seqlens: jax.Array | None = None,
        use_fullattn: bool = True,
        valid_token_count: jax.Array | None = None,
    ) -> jax.Array:
        x = x + self.attn(
            self.norm1(x),
            rotary_pos_emb,
            cu_window_seqlens,
            use_fullattn,
            valid_token_count=valid_token_count,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchMerger(nnx.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable,
        spatial_merge_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.ln_q = norm_layer(
            context_dim, dtype=dtype, rngs=_rngs, scale_init=nnx.with_partitioning(init_fn, (None,))
        )
        self.mlp_fc1 = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = nnx.Linear(
            self.hidden_size,
            d_model,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.ln_q(x)
        x = x.reshape(-1, self.hidden_size)
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)
        return x


class Qwen2_5_VisionTransformer(nnx.Module):

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
    ):
        self.config = config
        self.dtype = dtype

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            dtype=dtype,
            rngs=rngs,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nnx.List(
            [
                Qwen2_5_VisionBlock(
                    config=config,
                    dtype=dtype,
                    rngs=rngs,
                    mesh=mesh,
                )
                for _ in range(config.depth)
            ]
        )

        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=config.out_hidden_size,
            context_dim=config.hidden_size,
            norm_layer=partial(nnx.RMSNorm, epsilon=norm_eps),
            spatial_merge_size=config.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
        )

        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

    def rotary_pos_emb_thw(self, t, h, w):
        hpos_ids, wpos_ids = jnp.indices((h, w))
        hpos_ids = (
            hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )
        wpos_ids = (
            wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )
        pos_ids = jnp.stack([hpos_ids, wpos_ids], axis=-1)
        pos_ids = jnp.tile(pos_ids, (t, 1))

        max_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)

        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )

        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size

        index = jnp.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)

        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = jnp.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
        )
        index_padded = jnp.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        # The number of valid indices is static because grid_t, grid_h, grid_w
        # are static.
        num_valid_indices = grid_t * llm_grid_h * llm_grid_w
        valid_indices = jnp.nonzero(index_padded != -100, size=num_valid_indices)[0]
        index_new = index_padded[valid_indices]
        cu_seqlens_tmp = jnp.cumsum(seqlens) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.astype(jnp.int32)

        # NOTE (wenlong): Pytorch code uses this to reduce replication,
        # but I don't think there is a need here, plus it would cause problem in JIT
        # Please refer here if there is a problem down-stream
        # cu_seqlens_tmp = jnp.unique(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seq_lens_window_thw = self.get_window_index_thw(t, h, w)

        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)

        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.reshape(-1, rotary_pos_emb_thw.shape[-1])
        cu_seq_lens_thw = jnp.full(t, h * w, dtype=jnp.int32)

        return rotary_pos_emb_thw, window_index_thw, cu_seq_lens_window_thw, cu_seq_lens_thw

    def compute_aux_arrays(self, grid_thw: tuple[tuple[int, int, int]]):
        num_grids = len(grid_thw)

        rotary_pos_emb = []
        window_index: list = []
        cu_window_seqlens: list = [jnp.array([0], dtype=jnp.int32)]
        cu_seqlens: list = []

        window_index_id = 0
        cu_window_seqlens_last = 0
        for i in range(num_grids):
            t, h, w = grid_thw[i]

            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = jnp.concatenate(rotary_pos_emb, axis=0)
        window_index = jnp.concatenate(window_index, axis=0)
        cu_window_seqlens = jnp.concatenate(cu_window_seqlens, axis=0)

        cu_seqlens = jnp.concatenate(cu_seqlens, axis=0)
        cu_seqlens = jnp.cumsum(cu_seqlens, axis=0, dtype=jnp.int32)
        cu_seqlens = jnp.pad(cu_seqlens, ((1, 0),), mode="constant", constant_values=0)
        return window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens

    def compute_hidden_states(
        self,
        x: jax.Array,
        window_index: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_seqlens: jax.Array,
        cu_window_seqlens: jax.Array,
        valid_patch_rows: jax.Array | None = None,
    ) -> jax.Array:
        hidden_states = self.patch_embed(x)

        # num of patches
        seq_len = x.shape[0]

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        hidden_states = jnp.expand_dims(hidden_states, axis=1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                hidden_states = blk(
                    hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    cu_window_seqlens=cu_seqlens,
                    use_fullattn=True,
                    valid_token_count=valid_patch_rows,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    cu_window_seqlens=cu_window_seqlens,
                    use_fullattn=False,
                    valid_token_count=valid_patch_rows,
                )

        # adapter
        hidden_states = self.merger(hidden_states)
        # JIT-safe argsort (numpy would break under JIT).
        reverse_indices = jnp.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def __call__(self, x: jax.Array, grid_thw: tuple[tuple[int, int, int]]) -> jax.Array:
        # x: pixel_values: jax.Array
        # """Shape:
        # `(num_patches, num_channels * patch_size * patch_size)`
        # """

        # grid_thw: image_grid_thw: jax.Array
        # """Shape: `(num_images, 3)`
        # This should be in `(grid_t, grid_h, grid_w)` format.
        # """
        # Run in eager mode (no JIT) to avoid kernel cache issues
        # Vision encoding happens once during prefill, performance isn't critical
        window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens = self.compute_aux_arrays(
            grid_thw
        )
        return self.compute_hidden_states(
            x, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens
        )


def _get_visual_config(config: Any) -> QwenVLModelVitConfig:
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        vision_config = getattr(config, "vision_config_dict", None)
    if vision_config is None:
        return QwenVLModelVitConfig()
    if isinstance(vision_config, QwenVLModelVitConfig):
        return vision_config
    if isinstance(vision_config, dict):
        config_obj = QwenVLModelVitConfig()
        for key, value in vision_config.items():
            setattr(config_obj, key, value)
        return config_obj
    return vision_config


class Qwen2_5_VL_Model(Qwen2Model):
    """Qwen2Model with MRoPE support for Qwen2.5-VL.

    Kept as a subclass (B): the base ``Qwen2Model`` rope is not MRoPE-aware
    (hardcoded ``RotaryEmbedding`` that flattens positions), so MRoPE is isolated
    here -- ``__init__`` swaps each layer's ``rotary_emb`` to ``MRotaryEmbedding``
    and ``__call__`` feeds 3-D ``forward_batch.mrope_positions``. Base ``qwen2.py``
    is untouched.
    """

    def __init__(
        self,
        config,
        mesh,
        dtype=jnp.bfloat16,
    ):
        super().__init__(config=config, mesh=mesh, dtype=dtype)
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self._mrope_section = rope_scaling.get("mrope_section")
        self._mrope_interleaved = rope_scaling.get("mrope_interleaved", False)
        if self._mrope_section:
            rope_theta = getattr(config, "rope_theta", 1000000)
            max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
            for layer in self.layers:
                head_dim = layer.self_attn.head_dim
                layer.self_attn.rotary_emb = MRotaryEmbedding(
                    head_size=head_dim,
                    rotary_dim=head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=rope_theta,
                    is_neox_style=True,
                    dtype=dtype,
                    mrope_section=self._mrope_section,
                    mrope_interleaved=self._mrope_interleaved,
                )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = None
        input_embeds = (
            forward_batch.input_embedding
            if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            else None
        )
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids) if input_embeds is None else input_embeds
        )
        rope_positions = (
            forward_batch.mrope_positions
            if self._mrope_section and forward_batch.mrope_positions is not None
            else forward_batch.positions
        )
        layers_kv_fused = []
        layers_callback_flag = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, callback_flag = layer(
                rope_positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_callback_flag


class Qwen2_5_VLForConditionalGeneration(nnx.Module):
    """In-model Qwen2.5-VL (single-file): vision tower + Qwen2 backbone (+ MRoPE)
    + lm_head. The visual encode/merge surfaces stay outside the backbone JIT;
    MRoPE lives in ``Qwen2_5_VL_Model``.
    """

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16

        # Language backbone (Qwen2 + MRoPE) + lm_head + logits.
        self.model = Qwen2_5_VL_Model(self.text_config, mesh=mesh, dtype=self.dtype)
        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=self.mesh)
        self.image_token_id = getattr(self.config, "image_token_id", None)
        self.video_token_id = getattr(self.config, "video_token_id", None)

        # Vision tower. `self.visual` IS the ViT; the in-model embedder is
        # `get_image_feature` (resolved by `embed_mm_inputs` via
        # `getattr(model, "get_image_feature")`, no `mm_embedders` dict).
        self.visual_config = _get_visual_config(config)
        self.visual = Qwen2_5_VisionTransformer(
            config=self.visual_config,
            dtype=self.dtype,
            rngs=rngs,
            mesh=mesh,
            norm_eps=getattr(self.visual_config, "rms_norm_eps", 1e-6),
        )

    def get_input_embeddings(self):
        """Return the token-embedding module (callable on input_ids).

        Used by `general_mm_embed_routine`/`embed_mm_inputs` to seed `running`
        once, outside the round loop, before merging vision features.
        """
        return self.model.embed_tokens

    def get_image_feature(self, enc):
        """Image embedder (= upstream `get_image_feature`, models/qwen2_5_vl.py:643;
        resolved by `embed_mm_inputs` via `getattr(model, "get_image_feature")`).

        aux 落点 = Design X: compute the ViT aux arrays HERE (host, per rank from
        `enc.grid`), reshard to `P("data")`, then run the JIT(1) encode shard_map.
        `enc` is a `VisionEncodeInputs` for one DP round. Returns
        `[dp*out_rows, H]` as `P("data", None)`.
        """
        aux = self._compute_round_aux(enc.grid)
        return mm_encode(self.mesh, self._vision_encode_body, enc.pixels, aux, enc.valid)

    def _compute_round_aux(self, grid):
        """Per-round host aux (Design X; computed here from ``grid`` instead of
        being stored in the plan).

        `grid` is `[dp, 3]`; per rank call `compute_aux_arrays(grid[r])` with a
        static python-int single-image grid, pad/stack the 4 aux leaves across
        ranks to the cross-rank max, then reshard each to `P("data")`. aux arrays
        are `jnp` (already on device) -> this is a `P("data")` reshard, NOT an H2D
        transfer. Dummy ranks (`grid == 0`) get a zero lane. Returns the 4-tuple
        of stacked+sharded aux leaves, or `None` if all ranks are dummy.
        """
        visual = self.visual  # ViT tower (compute_aux_arrays lives here)
        # grid is host (static; init_new does NOT device_put it). np.asarray is a
        # no-op on host numpy and syncs if a device array ever slips through.
        grid_host = np.asarray(grid)
        dp_size = int(grid_host.shape[0])

        per_rank_aux: list = []  # over ranks: 4-tuple of jnp aux, or None (dummy)
        for r in range(dp_size):
            g = tuple(int(v) for v in grid_host[r])
            if g == (0, 0, 0):
                per_rank_aux.append(None)
                continue
            per_rank_aux.append(tuple(visual.compute_aux_arrays((g,))))

        present = [a for a in per_rank_aux if a is not None]
        if not present:
            return None

        num_leaves = len(present[0])
        max_rows = [max(int(a[i].shape[0]) for a in present) for i in range(num_leaves)]
        stacked_leaves = []
        for i in range(num_leaves):
            ref = present[0][i]
            tail_shape = ref.shape[1:]
            # aux[0] is window_index (a permutation of [0, F_r)); pad it by
            # EXTENDING the permutation so compute_hidden_states' argsort stays a
            # valid inverse. Other leaves zero-pad (rotary padded rows are
            # attention-masked by valid_patch_rows; cu_seqlens values are unused
            # by the distance-band attention).
            pad = self._pad_window_index if i == 0 else self._pad_rows_to
            rank_arrs = []
            for a in per_rank_aux:
                leaf = a[i] if a is not None else jnp.zeros((0, *tail_shape), dtype=ref.dtype)
                rank_arrs.append(pad(leaf, max_rows[i]))
            stacked = jnp.stack(rank_arrs, axis=0)  # [dp, max_rows_i, ...]
            spec = PartitionSpec("data", *([None] * (stacked.ndim - 1)))
            stacked_leaves.append(self._device_put_data_axis(stacked, spec))
        return tuple(stacked_leaves)

    def _vision_encode_body(self, pixels, aux, valid):
        """Single-image ViT body for one DP rank (runs INSIDE the JIT(1) encode
        shard_map). `aux` is the 4-tuple `(window_index, rotary_pos_emb,
        cu_seqlens, cu_window_seqlens)` computed host-side by `get_image_feature`
        (`_compute_round_aux`). Returns `[out_rows, H]` for this rank's image.
        """
        return self.visual.compute_hidden_states(pixels, *aux, valid_patch_rows=valid)

    def load_weights(self, model_config: ModelConfig):
        # Backbone (text) weights -- folded in from former Qwen2_5_VL_Generation.
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(self._create_qwen2_weight_mappings())
        logger.info("Qwen2.5-VL (LLM) weights loaded successfully!")
        # Vision (ViT) weights -- second WeightLoader pass mapping only visual.*.
        # The vision loader reads ONLY `model_path` (WeightLoader's safetensors
        # glob); the head/kv block in weight_utils is skipped for the
        # string-target vision mappings, so no other config field is needed.
        visual_loader_config = SimpleNamespace(model_path=model_config.model_path)
        self._load_vision_weights(visual_loader_config)

    def _load_vision_weights(self, model_config) -> None:
        """Load the ViT (``self.visual``) weights from safetensors.

        Folded in from the former ``Qwen2_5_VL_VisionModel.load_weights``. The
        backbone (text) weights are loaded by ``super().load_weights`` above; this
        is a second WeightLoader pass over the same safetensors that maps only the
        ``visual.*`` keys. No ``text_embed`` -- the LM owns token embedding.
        """
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_qwen2_5_vl_vision_weight_mappings()
        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(weight_mappings)
        else:
            loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen2.5-VL ViT weights loaded successfully!")

    def _create_qwen2_5_vl_vision_weight_mappings(self) -> dict:
        # Vision layers use replicated weights (no tensor parallelism). Targets are
        # relative to `self` (the wrapper), whose `self.visual` IS the ViT tower.
        mappings = {
            # Patch embed Conv3D: PyTorch [out,in,kd,kh,kw] -> JAX [kd,kh,kw,in,out]
            "visual.patch_embed.proj.weight": WeightMapping(
                target_path="visual.patch_embed.proj.kernel",
                sharding=(None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "visual.merger.ln_q.weight": WeightMapping(
                target_path="visual.merger.ln_q.scale",
                sharding=(None,),
                transpose=False,
            ),
            "visual.merger.mlp.0.weight": WeightMapping(
                target_path="visual.merger.mlp_fc1.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            "visual.merger.mlp.0.bias": WeightMapping(
                target_path="visual.merger.mlp_fc1.bias",
                sharding=(None,),
                transpose=False,
            ),
            "visual.merger.mlp.2.weight": WeightMapping(
                target_path="visual.merger.mlp_fc2.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            "visual.merger.mlp.2.bias": WeightMapping(
                target_path="visual.merger.mlp_fc2.bias",
                sharding=(None,),
                transpose=False,
            ),
        }
        num_vision_layers = getattr(self.visual_config, "depth", 0)
        for layer_idx in range(num_vision_layers):
            mappings.update(self._create_vision_layer_mappings(layer_idx))
        return mappings

    def _create_vision_layer_mappings(self, layer_idx: int) -> dict:
        # Qwen2.5-VL uses visual.blocks.{i}.* for vision layers (replicated).
        prefix = f"visual.blocks.{layer_idx}"
        target_prefix = f"visual.blocks.{layer_idx}"
        return {
            f"{prefix}.norm1.weight": WeightMapping(
                target_path=f"{target_prefix}.norm1.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.norm2.weight": WeightMapping(
                target_path=f"{target_prefix}.norm2.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.attn.qkv.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.attn.qkv.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.attn.proj.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.attn.proj.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.mlp.gate_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
        }

    def _device_put_data_axis(self, value, spec):
        if getattr(self, "mesh", None) is None:
            return value
        return jax.device_put(value, NamedSharding(self.mesh, spec))

    @staticmethod
    def _pad_rows_to(value, target_rows: int):
        """Zero-pad `value` (a `[rows, ...]` array) up to `target_rows` rows.

        Used by `_compute_round_aux` to pad each rank's aux leaf to the cross-rank
        max before stacking; shapes are static under trace.
        """
        rows = int(value.shape[0])
        if rows == target_rows:
            return value
        pad_shape = (target_rows - rows, *value.shape[1:])
        padding = jnp.zeros(pad_shape, dtype=value.dtype)
        return jnp.concatenate([value, padding], axis=0)

    @staticmethod
    def _pad_window_index(window_index, target_rows: int):
        """Pad ``window_index`` (a permutation of ``[0, rows)``) up to
        ``target_rows`` by EXTENDING the permutation with
        ``arange(rows, target_rows)``.

        Unlike zero-padding, this keeps it a valid permutation of
        ``[0, target_rows)`` so ``compute_hidden_states``'s
        ``argsort(window_index)`` stays a correct inverse for padded ranks --
        zero-padding would collide the value-0 entries and scramble the padded
        rank's feature rows. Dummy ranks (``rows == 0``) get the full identity
        permutation.
        """
        rows = int(window_index.shape[0])
        if rows >= target_rows:
            return window_index
        ext = jnp.arange(rows, target_rows, dtype=window_index.dtype)
        return jnp.concatenate([window_index, ext], axis=0)

    def _create_qwen2_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.text_config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            mappings.update(self._create_layer_mappings(layer_idx))

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        if getattr(self.text_config, "attention_bias", True):
            mappings.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=False,
                    ),
                    f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                }
            )

        return mappings

    def get_embed_and_head(self):
        if getattr(self.text_config, "tie_word_embeddings", False):
            weight = self.model.embed_tokens.embedding.value
            return (weight, weight)
        return (self.model.embed_tokens.embedding.value, self.lm_head.embedding.value)

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        token_to_kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool
        )
        if not getattr(self.text_config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None


EntryClass = Qwen2_5_VLForConditionalGeneration
