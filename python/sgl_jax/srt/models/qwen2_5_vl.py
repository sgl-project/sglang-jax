import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from jax.tree_util import register_pytree_node_class
from transformers import modeling_flax_utils

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.managers.mm_utils import jitted_mm_encode
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2Model
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import (
    QwenVLModelVitConfig,
)
from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

init_fn = nnx.initializers.uniform()


def apply_rotary_pos_emb_vision(x: jax.Array, rotary_pos_emb: jax.Array) -> jax.Array:
    # x: [dp, T, N, H]; rotary_pos_emb: [dp, T, rot] (per-image, dp-leading).
    _, _, _, H = x.shape
    half_dim = H // 2

    x_real = x[..., :half_dim]
    x_imag = x[..., half_dim:]

    cos_emb = jnp.cos(rotary_pos_emb)
    sin_emb = jnp.sin(rotary_pos_emb)

    # rope already carries the dp (batch) axis -> only insert the heads (N) axis.
    cos_emb = cos_emb[:, :, None, :]  # [dp, T, 1, rot]
    sin_emb = sin_emb[:, :, None, :]

    x_rotated_real = x_real * cos_emb - x_imag * sin_emb
    x_rotated_imag = x_real * sin_emb + x_imag * cos_emb

    return jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)


def vision_attention(
    backend,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seg: jax.Array,
) -> jax.Array:
    """In-model ViT attention via ``VisionFlashAttentionBackend`` (block-diagonal).

    Batched, dp-leading (spec §2.4 / §3.10): ``q/k/v`` are ``[dp, T, N, H]`` and
    ``seg`` is ``[dp, T]``. Uses ``VisionFlashAttentionBackend`` (DP-only) -- it
    wraps the segment-flash pallas kernel in a DP-only shard_map
    (qkv ``P("data",None,None,None)`` / seg ``P("data",None)``), NO head-TP. Since
    ``jitted_mm_encode`` is now a pure jit, calling the backend's
    ``jax.jit(shard_map(...))`` is jit-in-jit (inlined; NOT shard_map nesting).

    Layout: the kernel wants ``[dp, N, T, H]`` so we ``transpose(0,2,1,3)``. T is
    padded to a multiple of 128 (pallas block req); padding rows get a sentinel
    segment (``-1``, distinct from every real segment -> masked as q and kv), then
    sliced ``[:, :, :T, :]`` back. ``causal=False`` (carried by the backend).
    """
    dp, T, N, H = q.shape

    # [dp, T, N, H] -> [dp, N, T, H] (kernel layout).
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # Pad seq (T) up to a multiple of 128 (kernel block requirement). Padding
    # rows get a sentinel segment so they never share a segment with real patches.
    T_aligned = ((T + 127) // 128) * 128
    pad = T_aligned - T
    if pad > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
        seg = jnp.pad(seg, ((0, 0), (0, pad)), constant_values=-1)  # [dp, T_aligned]

    segment_ids = SegmentIds(q=seg, kv=seg)

    output = backend(q, k, v, segment_ids)  # [dp, N, T_aligned, H]

    output = output[:, :, :T, :]  # slice back to real seq
    # [dp, N, T, H] -> [dp, T, N, H]
    return jnp.transpose(output, (0, 2, 1, 3))


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
        # x is (dp, seq_len, C * T * H * W) -- dp-leading batched (spec §3.10);
        # seq_len == the per-image patch count (== patch_k in the plan).
        dp, seq_len, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size * self.patch_size)
        # Fold [dp, seq_len] into ONE conv batch axis: conv is per-element over
        # the batch, so dp*seq_len patches together == per-image (no cross-image
        # mixing).
        x = x.reshape(dp * seq_len, C, self.temporal_patch_size, self.patch_size, self.patch_size)
        # (dp*seq_len), T, H, W, C
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        # After conv (stride=kernel_size): (dp*seq_len, 1, 1, 1, hidden_size)
        x = x.reshape(dp, seq_len, self.hidden_size)  # unfold dp
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

        # DP-only vision attention backend (spec §2.4 / §3.10): reused across all
        # in-model VLMs. Lazy import avoids a module-level import cycle
        # (flash_attention_backend -> schedule_batch -> models). ``mesh`` is None
        # only during eval_shape (which never calls __call__), so guard it.
        if mesh is not None:
            from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
                VisionFlashAttentionBackend,
            )

            self.attn_backend = VisionFlashAttentionBackend(mesh, sm_scale=self.scale, causal=False)
        else:
            self.attn_backend = None

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu: jax.Array,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        """ViT attention via block-diagonal segment flash (dp-leading, spec §3.10).

        ``x`` is ``[dp, T, D]``. ``cu`` is this block's cumulative segment
        boundaries PER IMAGE: full-att blocks pass ``cu_full = [dp, 1]`` (= valid),
        windowed blocks pass ``cu_window_seqlens`` ``[dp, max_windows]`` -- SAME
        function, only the ``cu`` differs ("swap the cu, not the function"). The
        per-patch segment ids are derived here via a batched ``searchsorted``-
        equivalent broadcast; round-loop cross-rank padding patches
        (``pos >= valid``) get a sentinel segment (``-1``).
        """
        dp, T, D = x.shape

        # cu -> per-patch segment ids (window order), batched over dp. This is the
        # broadcast form of searchsorted(side="right"): count of cu <= pos.
        positions = jnp.arange(T)
        seg = (cu[:, None, :] <= positions[None, :, None]).sum(-1).astype(jnp.int32)  # [dp, T]
        if valid is not None:
            is_real = positions[None, :] < jnp.reshape(valid, (dp, 1))  # [dp, T]
            seg = jnp.where(is_real, seg, jnp.full_like(seg, -1))

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [dp, T, 3D]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [dp, T, D] each

        # [dp, T, D] -> [dp, T, N, H] (dp-leading: NO transpose, already kernel-ready)
        q = q.reshape(dp, T, self.num_heads, self.head_dim)
        k = k.reshape(dp, T, self.num_heads, self.head_dim)
        v = v.reshape(dp, T, self.num_heads, self.head_dim)

        # Apply rotary embeddings (rope is per-image [dp, T, rot])
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # Block-diagonal segment flash attention via the DP-only backend.
        output = vision_attention(self.attn_backend, q, k, v, seg)  # [dp, T, N, H]

        # [dp, T, N, H] -> [dp, T, D]
        output = output.reshape(dp, T, D)

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
        cu: jax.Array,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb, cu, valid)
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
        # x: [dp, T, ctx] (dp-leading, spec §3.10 #11).
        x = self.ln_q(x)
        dp = x.shape[0]
        # Keep dp on axis 0: the sms² spatial-merge stays WITHIN each image.
        # ``reshape(-1, ...)`` here would interleave T and dp and silently mix
        # across images (the one silent-corruption point, spec §3.10 manual-verify②).
        x = x.reshape(dp, -1, self.hidden_size)  # [dp, T/sms², ctx*sms²]
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)
        return x  # [dp, T/sms², d_model]


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
        cu_window_seqlens: jax.Array,
        rotary_pos_emb: jax.Array,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        """In-model single-image ViT forward (segment-flash, spec §3.9).

        Consumes the scheduler-built ``VisionMetadata`` (``window_index`` /
        ``cu_window_seqlens`` / ``rotary_pos_emb``). Order: patch_embed ->
        unit-granularity ``hidden[window_index]`` (window order) -> blocks
        (unified cu path) -> merger (window order spatial-merge) ->
        ``argsort(window_index)`` inverse (raster order).

        Unified cu path ("swap the cu, not the function", §3.9): full-att blocks
        pass the degenerate single-segment ``cu_full = [valid]`` (image-internal
        full attention), windowed blocks pass ``cu_window_seqlens``; the block
        derives per-patch segment ids (``searchsorted``) + padding sentinel from
        ``(cu, valid)`` internally.

        ``rotary_pos_emb`` arrives ALREADY gathered into window order by the
        builder -> do NOT re-gather. ``valid`` (real patch count) masks
        round-loop cross-rank padding patches.
        """
        # x: [dp, seq, dim_in] (dp-leading batched, spec §3.10).
        hidden_states = self.patch_embed(x)  # [dp, seq, D]
        dp = x.shape[0]
        seq_len = x.shape[1]
        u = self.spatial_merge_unit

        hidden_states = hidden_states.reshape(dp, seq_len // u, u, -1)  # [dp, seq//u, u, D]
        # window 正排 (per-image gather on the unit axis): take_along_axis(axis=1).
        gather_idx = jnp.broadcast_to(window_index[:, :, None, None], hidden_states.shape)
        hidden_states = jnp.take_along_axis(hidden_states, gather_idx, axis=1)
        hidden_states = hidden_states.reshape(dp, seq_len, -1)  # [dp, T, D]

        # full-att blocks: degenerate single segment cu_full = [dp, 1] (= valid);
        # windowed blocks: cu_window_seqlens. Block derives seg + sentinel itself.
        if valid is None:
            cu_full = jnp.full((dp, 1), seq_len, dtype=cu_window_seqlens.dtype)
        else:
            cu_full = jnp.reshape(valid, (dp, 1)).astype(cu_window_seqlens.dtype)

        for layer_num, blk in enumerate(self.blocks):
            cu = cu_full if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            hidden_states = blk(hidden_states, rotary_pos_emb, cu, valid)

        # adapter (merger): [dp, T, D] -> [dp, T/sms², d_model]
        hidden_states = self.merger(hidden_states)
        # 反排 (per-image): argsort along axis=1, then take_along_axis(axis=1).
        reverse_indices = jnp.argsort(window_index, axis=1)  # [dp, seq//u]
        rev_idx = jnp.broadcast_to(reverse_indices[:, :, None], hidden_states.shape)
        hidden_states = jnp.take_along_axis(hidden_states, rev_idx, axis=1)
        return hidden_states  # [dp, out_rows, H]

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


@register_pytree_node_class
@dataclass
class VisionMetadata:
    """Per-round vision aux, scheduler-computed, threaded into the encode JIT.

    Per-arch registered pytree (Qwen2.5-VL): defined HERE in the model file (not
    in the modality-general ``mm_plan``), since common code treats the encode
    ``meta`` as an OPAQUE pytree and never names these fields. ``get_metadata`` /
    ``stack_metadata`` (below) construct it; only the ViT encode body interprets
    it.

    aux 落点 = Design B -- the per-arch ``get_metadata`` builder computes these
    ViT aux arrays host-side in the scheduler (config-only, from ``grid``), so
    the plan carries the derived aux directly and the encode body never recomputes
    it. Because this payload crosses the encode JIT boundary, it is a registered
    pytree (children flatten order is fixed: ``window_index``,
    ``cu_window_seqlens``, ``rotary_pos_emb``).

    Fields hold one round's pad-stacked-across-ranks ViT aux:

    - ``window_index``:      ``[dp, merge_units]`` int -- unit-granularity
      window-order permutation (identity-padded across ranks so ``argsort``
      un-permute holds).
    - ``cu_window_seqlens``: ``[dp, max_windows]`` int -- cumulative window
      boundaries on the window-reordered layout (no leading 0; last real value =
      true patch count; sentinel-padded to ``max_windows``). The per-patch
      segment ids are computed inside the ViT forward via
      ``searchsorted(cu_window_seqlens, arange(seq))`` (host no longer
      pre-derives them).
    - ``rotary_pos_emb``:    ``[dp, patch_k, rot_dim]`` -- per-patch 2D rope
      (zero-padded).
    """

    window_index: Any
    cu_window_seqlens: Any
    rotary_pos_emb: Any

    def tree_flatten(self):
        children = (self.window_index, self.cu_window_seqlens, self.rotary_pos_emb)
        aux_data = {}  # static sizes/roles may live here; runtime does not depend on it
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class VisionMetadataBuilder:
    """Per-arch, config-only ViT aux builder (spec §3.2 / decision 2.1).

    Pure numpy, NO weights, NO model instance: the scheduler instantiates this
    from ``model_config.vision_config`` and calls :meth:`get_metadata` once per
    image to produce a :class:`VisionMetadata` (``window_index`` /
    ``cu_window_seqlens`` / ``rotary_pos_emb``). Reuses the same window/rope algorithm the
    ViT used host-side (ported off ``jnp`` to ``np``); ``cu_window_seqlens`` is
    carried as-is (cumulative window boundaries) and converted to per-patch
    segment ids INSIDE the ViT forward (``searchsorted``), not here.

    Cross-rank pad/stack of the produced arrays into the round-loop bucket is the
    scheduler's job; this builder only ever produces SINGLE-image metadata.
    """

    def __init__(self, vision_cfg):
        self.vision_cfg = vision_cfg
        self.patch_size = int(getattr(vision_cfg, "patch_size", 14))
        self.window_size = int(getattr(vision_cfg, "window_size", 112))
        self.spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))
        self.fullatt_block_indexes = list(
            getattr(vision_cfg, "fullatt_block_indexes", [7, 15, 23, 31])
        )
        num_heads = int(getattr(vision_cfg, "num_heads", 16))
        hidden_size = int(getattr(vision_cfg, "hidden_size", 1280))
        head_dim = hidden_size // num_heads
        # rotary dim = head_dim // 2 (matches Qwen2_5_VisionRotaryEmbedding(head_dim//2))
        self.rotary_dim = head_dim // 2
        self.theta = float(getattr(vision_cfg, "rope_theta", 10000.0))
        self.spatial_merge_unit = self.spatial_merge_size**2

    # ---- ported host-only algorithms (numpy) ----------------------------------
    def _rotary_pos_emb_full(self, seq_len: int) -> np.ndarray:
        # mirrors Qwen2_5_VisionRotaryEmbedding.__call__ (dim = self.rotary_dim)
        inv_freq = 1.0 / (
            self.theta ** (np.arange(0, self.rotary_dim, 2, dtype=np.float32) / self.rotary_dim)
        )
        seq = np.arange(seq_len, dtype=np.float32)
        return np.outer(seq, inv_freq)  # [seq_len, rotary_dim//2]

    def _rotary_pos_emb_thw(self, t, h, w) -> np.ndarray:
        sms = self.spatial_merge_size
        hpos_ids, wpos_ids = np.indices((h, w))
        hpos_ids = hpos_ids.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
        wpos_ids = wpos_ids.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
        pos_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
        pos_ids = np.tile(pos_ids, (t, 1))

        max_size = max(h, w)
        rotary_pos_emb_full = self._rotary_pos_emb_full(max_size)

        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        return rotary_pos_emb  # [merge_units, sms^2, rot_dim]

    def _window_index_thw(self, grid_t, grid_h, grid_w):
        sms = self.spatial_merge_size
        vit_merger_window_size = self.window_size // sms // self.patch_size

        llm_grid_h = grid_h // sms
        llm_grid_w = grid_w // sms

        index = np.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)

        pad_h = (
            vit_merger_window_size - llm_grid_h % vit_merger_window_size
        ) % vit_merger_window_size
        pad_w = (
            vit_merger_window_size - llm_grid_w % vit_merger_window_size
        ) % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = np.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum(axis=(2, 3)).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        cu_seqlens_window = np.cumsum(seqlens) * self.spatial_merge_unit
        cu_seqlens_window = cu_seqlens_window.astype(np.int32)
        return index_new.astype(np.int32), cu_seqlens_window

    def get_metadata(self, image_grid) -> VisionMetadata:
        """Single-image ``(t, h, w)`` -> :class:`VisionMetadata`.

        - ``window_index``: unit-granularity window-order permutation (length
          ``t * (h//sms) * (w//sms)`` = merge_units).
        - ``rotary_pos_emb``: per-patch 2D rope, ALREADY gathered into window order
          (``rope[window_index]`` flattened), length ``t*h*w`` patches.
        - ``cu_window_seqlens``: cumulative window boundaries on the
          window-reordered layout (``cumsum(seqlens) * spatial_merge_unit``; no
          leading 0; last value = true patch count). The per-patch segment ids
          (``searchsorted(cu_window_seqlens, patch, side="right")``) are NOT
          pre-derived here -- they are computed inside
          ``compute_hidden_states`` (in the encode JIT), right before the
          flash kernel.

        ``cu_seqlens`` (whole-image boundary) is NOT produced: round-loop single
        image -> full-att = full attention + ``valid`` padding mask (spec §2.3).
        """
        t, h, w = (int(image_grid[0]), int(image_grid[1]), int(image_grid[2]))

        window_index, cu_window_seqlens = self._window_index_thw(t, h, w)

        # rope: gather into window order (mirrors get_rope_by_thw).
        rope_units = self._rotary_pos_emb_thw(t, h, w)  # [merge_units, sms^2, rot_dim]
        rope = rope_units[window_index, :, :]
        rope = rope.reshape(-1, rope.shape[-1]).astype(np.float32)  # [patches, rot_dim]

        return VisionMetadata(
            window_index=window_index.astype(np.int32),
            cu_window_seqlens=cu_window_seqlens.astype(np.int32),
            rotary_pos_emb=rope,
        )

    def stack_metadata(self, metas, patch_k):
        """Cross-rank pad-by-role + stack of single-image metas -> VisionMetadata[dp, ...].

        ``metas[r]`` is this rank's round-k native-size :class:`VisionMetadata`,
        or ``None`` for a dummy lane (rank owns < k+1 images). ``patch_k`` is the
        round's cross-rank max patch-row bucket (drives the cu sentinel and rope
        pad length). Bucket sizes are the cross-rank max of each role's native
        length:

        - ``window_index`` is a PERMUTATION over ``units_k`` (= merge_units):
          identity-fill pad slots with ``arange(native_units, units_k)`` so the
          full row stays a valid permutation (the ViT ``argsort``-un-permutes; a
          non-permutation would corrupt that reverse-scatter). Dummy lanes get a
          full ``arange(units_k)`` identity row.
        - ``cu_window_seqlens`` (cumulative boundaries): sentinel = ``patch_k``
          for pad slots. cu's last real value = true patch count <= ``patch_k``,
          so a ``patch_k`` sentinel keeps the row non-decreasing AND > every real
          patch index, hence the forward's ``searchsorted(side="right")`` never
          counts a sentinel window for any real patch. Dummy lanes get an
          all-``patch_k`` row (irrelevant -- masked by ``valid`` in the forward).
        - ``rotary_pos_emb`` (per-patch values): 0 for pad patches; rope is
          padded to ``patch_k`` rows.
        """
        present = [m for m in metas if m is not None]
        units_k = max(int(m.window_index.shape[0]) for m in present)
        win_k = max(int(m.cu_window_seqlens.shape[0]) for m in present)
        rot_dim = int(present[0].rotary_pos_emb.shape[-1])
        wi, cu, rope = [], [], []
        for m in metas:
            if m is None:  # dummy lane
                wi.append(np.arange(units_k, dtype=np.int32))  # identity perm (un-permute safe)
                cu.append(np.full(win_k, patch_k, dtype=np.int32))  # all sentinel
                rope.append(np.zeros((patch_k, rot_dim), dtype=np.float32))
            else:
                # window_index: true values + arange continuation for the pad tail.
                w = np.arange(units_k, dtype=np.int32)
                n_units = int(m.window_index.shape[0])
                w[:n_units] = np.asarray(m.window_index, dtype=np.int32)
                # cu_window_seqlens: true values + patch_k sentinel tail.
                c = np.full(win_k, patch_k, dtype=np.int32)
                n_win = int(m.cu_window_seqlens.shape[0])
                c[:n_win] = np.asarray(m.cu_window_seqlens, dtype=np.int32)
                # rotary_pos_emb: true rows + zero pad to patch_k.
                r = np.zeros((patch_k, rot_dim), dtype=np.float32)
                rp = np.asarray(m.rotary_pos_emb, dtype=np.float32)
                r[: int(rp.shape[0])] = rp
                wi.append(w)
                cu.append(c)
                rope.append(r)
        return VisionMetadata(np.stack(wi), np.stack(cu), np.stack(rope))


def resolve_vision_metadata_builder(arch_or_config):
    """Resolve the per-arch :class:`VisionMetadataBuilder` class (spec §3.2).

    Accepts an arch name string or an object carrying ``.arch``. The scheduler
    then instantiates the returned class with ``model_config.vision_config`` and
    calls ``get_metadata`` per image -- NO model instance required.

    Resolution mirrors the repo's ``ModelRegistry``/``EntryClass`` mechanism: the
    arch must map (via ``EntryClass``) to this module's embedder. Currently the
    only vision-metadata arch is ``Qwen2_5_VLForConditionalGeneration``.
    """
    arch = (
        arch_or_config if isinstance(arch_or_config, str) else getattr(arch_or_config, "arch", None)
    )
    if arch == "Qwen2_5_VLForConditionalGeneration":
        return VisionMetadataBuilder
    raise ValueError(
        f"No VisionMetadataBuilder registered for arch={arch!r}; "
        "expected 'Qwen2_5_VLForConditionalGeneration'."
    )


class Qwen2_5_VLForConditionalGeneration(nnx.Module):
    """In-model Qwen2.5-VL (single-file): vision tower + Qwen2 backbone (+ MRoPE)
    + lm_head. The visual encode/merge surfaces stay outside the backbone JIT;
    MRoPE is handled transparently by the plain ``Qwen2Model`` (mrope-aware
    ``get_rope`` + 3-D ``forward_batch.mrope_positions``), so no backbone subclass.
    """

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16

        # Language backbone (Qwen2 + MRoPE) + lm_head + logits.
        self.model = Qwen2Model(self.text_config, mesh=mesh, dtype=self.dtype)
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

    def get_image_feature(self, enc):
        """Image embedder (= upstream ``get_image_feature``; resolved by
        ``embed_mm_inputs`` via ``getattr(model, "get_image_feature")``).

        aux 落点 = Design B: the ViT aux (``window_index`` /
        ``cu_window_seqlens`` / ``rotary_pos_emb``) is computed host-side in the scheduler
        (``VisionMetadataBuilder``) and carried on ``enc.meta``; this embedder no
        longer recomputes it. It
        ``nnx.split``s the ViT (graphdef static + state dynamic, an explicit JIT
        operand -- reload-safe, no compile-cache clear) and runs the JIT(1) encode
        shard_map. ``enc`` is a ``VisionEncodeInputs`` for one DP round. Returns
        ``[dp*out_rows, H]`` as ``P("data", None)``.
        """
        graphdef, state = nnx.split(self.visual)
        return jitted_mm_encode(
            self.mesh, self._vision_encode_body, graphdef, state, enc.pixels, enc.meta, enc.valid
        )

    @staticmethod
    def _vision_encode_body(visual, pixels, meta, valid):
        """Single-image ViT body for one DP rank (runs INSIDE the JIT(1) encode
        shard_map). ``visual`` is the ViT tower merged inside ``jitted_mm_encode``
        from the nnx graphdef/state passed in -- weights are an explicit JIT
        operand, not closure-captured. ``meta`` is the ``VisionMetadata`` pytree
        (``window_index`` / ``cu_window_seqlens`` / ``rotary_pos_emb``) from ``enc.meta``
        (Design B; scheduler-built). Returns ``[out_rows, H]`` for this rank's
        image.
        """
        return visual.compute_hidden_states(
            pixels, meta.window_index, meta.cu_window_seqlens, meta.rotary_pos_emb, valid
        )

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
