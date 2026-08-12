import logging
import math
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import modeling_flax_utils

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2Model, create_qwen2_weight_mappings
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import (
    QwenVLModelVitConfig,
)
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    pack_vision_inputs,
    put_sharded_batch,
    restore_encoder_output,
    run_dp_sharded_encoder,
)
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    VisionAttentionMetadata,
    make_vision_attention_backend,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import (
    VisionShardSpecs,
    apply_data_sharding,
    resolve_encoder_tp,
)
from sgl_jax.srt.utils.common_utils import resolve_vision_patch_buckets
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_init_fn = nnx.initializers.uniform()


def _apply_rotary_pos_emb_vision(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply precomputed vision RoPE to ``x[B, T, heads, head_dim]``."""
    half_dim = x.shape[-1] // 2
    x_real, x_imag = x[..., :half_dim], x[..., half_dim:]
    return jnp.concatenate(
        [x_real * cos - x_imag * sin, x_real * sin + x_imag * cos],
        axis=-1,
    ).astype(x.dtype)


class Qwen2_5_VisionPatchEmbed(nnx.Module):
    """3D (temporal × spatial) patch embedding conv."""

    def __init__(
        self,
        rngs: nnx.Rngs = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: Mesh = None,
        vision_tp: bool = False,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)

        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """*x*: ``[B, S, C·T·H·W]`` → ``[B, S, hidden_size]``."""
        B, S, D = x.shape
        C = D // (self.temporal_patch_size * self.patch_size * self.patch_size)
        x = x.reshape(B, S, C, self.temporal_patch_size, self.patch_size, self.patch_size)
        if self.mesh is not None:
            x = apply_data_sharding(x, self.mesh, PartitionSpec(self.specs.batch_axis))

        # [B, S, C, T, H, W] → [B, S, T, H, W, C]
        x = jnp.transpose(x, (0, 1, 3, 4, 5, 2))

        sh = None
        if self.mesh is not None and "data" in self.mesh.abstract_mesh.explicit_axes:
            sh = self.specs.sharding(self.specs.batch_axis)

        x = x.reshape(
            B * S,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
            C,
            out_sharding=sh,
        )
        x = self.proj(x, out_sharding=sh)
        x = x.reshape(B, S, 1, 1, 1, self.hidden_size, out_sharding=sh)
        return jnp.squeeze(x, axis=(2, 3, 4))


class Qwen2_5_VLMLP(nnx.Module):
    """ViT MLP: gate/up → SiLU gate → down."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        vision_tp: bool = False,
    ):
        self.specs = VisionShardSpecs(mesh, vision_tp)
        self.act_fn = modeling_flax_utils.ACT2FN[config.hidden_act]

        self.gate_proj = LinearBase(
            config.hidden_size,
            config.intermediate_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.up_proj = LinearBase(
            config.hidden_size,
            config.intermediate_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.down_proj = LinearBase(
            config.intermediate_size,
            config.hidden_size,
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


class Qwen2_5_VisionAttention(nnx.Module):
    """ViT self-attention with fused QKV, RoPE, and block-diagonal flash attn."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        vision_tp: bool = False,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)

        if self.specs.tp:
            tp_size = int(mesh.shape["tensor"]) if mesh is not None else 1
            assert (
                self.num_heads % tp_size == 0
            ), f"vision num_heads={self.num_heads} must be divisible by tp={tp_size}"

        self.q_proj = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.k_proj = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.v_proj = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.proj = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

        if mesh is not None:
            self.attn_backend = make_vision_attention_backend(
                mesh,
                sm_scale=1.0 / math.sqrt(self.head_dim),
                causal=False,
                head_tp=self.specs.tp,
                use_varlen=True,
            )
        else:
            self.attn_backend = None

    def __call__(
        self,
        x: jax.Array,
        rotary_cos: jax.Array,
        rotary_sin: jax.Array,
        metadata: VisionAttentionMetadata,
    ) -> jax.Array:
        B, T, D = x.shape
        specs = self.specs
        col = specs.sharding(specs.batch_axis, None, specs.tensor_axis)

        # Project Q, K, V separately (TP-safe: each is independently column-parallel).
        q, _ = self.q_proj(x, out_sharding=col)
        k, _ = self.k_proj(x, out_sharding=col)
        v, _ = self.v_proj(x, out_sharding=col)

        hs = specs.sharding(specs.batch_axis, None, specs.tensor_axis, None)
        q = q.reshape(B, T, self.num_heads, self.head_dim, out_sharding=hs)
        k = k.reshape(B, T, self.num_heads, self.head_dim, out_sharding=hs)
        v = v.reshape(B, T, self.num_heads, self.head_dim, out_sharding=hs)

        q = _apply_rotary_pos_emb_vision(q, rotary_cos, rotary_sin)
        k = _apply_rotary_pos_emb_vision(k, rotary_cos, rotary_sin)

        out = self.attn_backend(q, k, v, metadata)
        out = out.reshape(B, T, D, out_sharding=col)
        out, _ = self.proj(out, out_sharding=specs.sharding(specs.batch_axis))
        return out


class Qwen2_5_VisionBlock(nnx.Module):
    """One ViT transformer block: attn (pre-norm) + MLP (pre-norm)."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
        vision_tp: bool = False,
    ):
        _rngs = rngs or nnx.Rngs(0)
        norm = partial(
            nnx.RMSNorm, epsilon=norm_eps, scale_init=nnx.with_partitioning(_init_fn, (None,))
        )

        self.norm1 = norm(config.hidden_size, dtype=dtype, rngs=_rngs)
        self.norm2 = norm(config.hidden_size, dtype=dtype, rngs=_rngs)
        self.attn = Qwen2_5_VisionAttention(
            config,
            dtype,
            rngs=rngs,
            mesh=mesh,
            vision_tp=vision_tp,
        )
        self.mlp = Qwen2_5_VLMLP(config, dtype, rngs=rngs, mesh=mesh, vision_tp=vision_tp)

    def __call__(
        self,
        x: jax.Array,
        rotary_cos: jax.Array,
        rotary_sin: jax.Array,
        metadata: VisionAttentionMetadata,
    ) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_cos, rotary_sin, metadata)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchMerger(nnx.Module):
    """Spatial merge: LN → reshape(sms²) → 2-layer MLP → [B, T/sms², d_model]."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable,
        spatial_merge_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        vision_tp: bool = False,
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)
        _rngs = rngs or nnx.Rngs(0)

        self.ln_q = norm_layer(
            context_dim,
            dtype=dtype,
            rngs=_rngs,
            scale_init=nnx.with_partitioning(_init_fn, (None,)),
        )
        self.mlp_fc1 = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = LinearBase(
            self.hidden_size,
            d_model,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        specs = self.specs
        row = specs.sharding(specs.batch_axis)
        x = self.ln_q(x)
        B = x.shape[0]
        x = x.reshape(B, -1, self.hidden_size, out_sharding=row)
        x, _ = self.mlp_fc1(
            x, out_sharding=specs.sharding(specs.batch_axis, None, specs.tensor_axis)
        )
        x = self.mlp_act(x)
        x, _ = self.mlp_fc2(x, out_sharding=row)
        return x


class Qwen2_5_VisionTransformer(nnx.Module):
    """Qwen2.5-VL ViT: patch embed → window / full-attn blocks → merge → reorder."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
        vision_tp: bool = False,
        input_buckets: tuple[int, ...] | None = None,
    ):
        self.mesh = mesh
        self.vision_tp = vision_tp
        self.specs = VisionShardSpecs(mesh, vision_tp)
        self.input_buckets = input_buckets or tuple(resolve_vision_patch_buckets(None))
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        if any(bucket <= 0 or bucket % self.spatial_merge_unit for bucket in self.input_buckets):
            raise ValueError(
                f"vision patch buckets must be positive multiples of {self.spatial_merge_unit}"
            )

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
            vision_tp=vision_tp,
        )
        self.blocks = nnx.List(
            [
                Qwen2_5_VisionBlock(
                    config,
                    dtype,
                    rngs=rngs,
                    mesh=mesh,
                    norm_eps=norm_eps,
                    vision_tp=vision_tp,
                )
                for i in range(config.depth)
            ]
        )
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=config.out_hidden_size,
            context_dim=config.hidden_size,
            norm_layer=partial(nnx.RMSNorm, epsilon=norm_eps),
            spatial_merge_size=config.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
            vision_tp=vision_tp,
        )

        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.patch_size = config.patch_size
        self.patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.window_size = config.window_size
        self.rotary_dim = config.hidden_size // config.num_heads // 2
        self.theta = float(getattr(config, "rope_theta", 10000.0))
        self.rot_dim = 2 * len(range(0, self.rotary_dim, 2))

    def __call__(
        self,
        patches: jax.Array,
        grid_thw: np.ndarray | jax.Array,
    ) -> jax.Array:
        return self.encode(patches, grid_thw)

    def _forward(
        self,
        patches: jax.Array,
        indices: jax.Array,
        position_ids: jax.Array,
        window_attn: VisionAttentionMetadata,
        full_attn: VisionAttentionMetadata,
    ) -> jax.Array:
        B, S = patches.shape[:2]
        u = self.spatial_merge_unit
        n_units = S // u
        window_index, reverse_indices = indices[:, :, 0], indices[:, :, 1]
        inv_freq = 1.0 / (
            self.theta ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        rotary_pos_emb = (position_ids[..., None].astype(jnp.float32) * inv_freq).reshape(
            B, S, self.rot_dim
        )
        rotary_cos = jnp.cos(rotary_pos_emb)[:, :, None, :]
        rotary_sin = jnp.sin(rotary_pos_emb)[:, :, None, :]

        x = self.patch_embed(patches)
        x = x.reshape(B, n_units, u, -1)

        # Window reorder (batch axis stays on 0).
        x = jnp.take_along_axis(x, window_index[:, :, None, None], axis=1)
        x = x.reshape(B, S, -1)

        # Select the pre-planned metadata per block: full-frame for the layers in
        # ``fullatt_block_indexes``, otherwise the local-window layout.
        layout_metadata = (window_attn, full_attn)
        for i, blk in enumerate(self.blocks):
            block_meta = layout_metadata[int(i in self.fullatt_block_indexes)]
            x = blk(x, rotary_cos, rotary_sin, block_meta)

        x = self.merger(x)
        return jnp.take_along_axis(x, reverse_indices[:, :, None], axis=1)

    def encode(
        self,
        patches: jax.Array,
        grid_thw: np.ndarray | jax.Array,
    ) -> jax.Array:
        patches = put_sharded_batch(patches, self.mesh, self.specs.batch_axis)
        metadata = self._build_metadata(grid_thw, patches.shape[1])
        metadata = put_sharded_batch(metadata, self.mesh, self.specs.batch_axis)
        if self.mesh is None:
            return self._encode_jit(patches, *metadata)
        with jax.set_mesh(self.mesh):
            return self._encode_jit(patches, *metadata)

    def precompile(self) -> None:
        num_lanes = encoder_num_lanes(self.mesh, self.vision_tp)
        for capacity in self.input_buckets:
            patches = np.zeros(
                (num_lanes, capacity, self.patch_dim),
                dtype=np.float32,
            )
            grid = (1, self.spatial_merge_size, capacity // self.spatial_merge_size)
            grid_thw = np.zeros((num_lanes, 1, 3), dtype=np.int32)
            grid_thw[0, 0] = grid
            jax.block_until_ready(self.encode(patches, grid_thw))

    def _build_metadata(
        self,
        lane_grids: np.ndarray | jax.Array,
        capacity: int,
    ) -> tuple[np.ndarray, np.ndarray, VisionAttentionMetadata, VisionAttentionMetadata]:
        lane_grids = np.asarray(jax.device_get(lane_grids), dtype=np.int32)
        if lane_grids.ndim == 2:
            lane_grids = lane_grids[None]
        batch = len(lane_grids)
        merge = self.spatial_merge_size
        unit = self.spatial_merge_unit
        num_units = capacity // unit
        window = self.window_size // merge // self.patch_size
        unit_range = np.arange(num_units, dtype=np.int32)
        indices = np.broadcast_to(unit_range[None, :, None], (batch, num_units, 2)).copy()
        position_ids = np.zeros((batch, capacity, 2), dtype=np.int32)
        cu_seqlens = np.zeros((batch, 2, num_units + 1), dtype=np.int32)

        def grid_layout(t: int, h: int, w: int):
            grid_h, grid_w = h // merge, w // merge
            index = np.arange(t * grid_h * grid_w).reshape(t, grid_h, grid_w)
            pad_h, pad_w = (-grid_h) % window, (-grid_w) % window
            windows_h, windows_w = (grid_h + pad_h) // window, (grid_w + pad_w) // window
            index = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-1)
            index = index.reshape(t, windows_h, window, windows_w, window)
            index = index.transpose(0, 1, 3, 2, 4).reshape(-1, window, window)
            window_lengths = (index != -1).sum(axis=(1, 2)).astype(np.int32) * unit
            index = index.reshape(-1)
            index = index[index != -1].astype(np.int32)

            y, x = np.indices((h, w))
            coords = np.stack((y, x), axis=-1)
            coords = coords.reshape(grid_h, merge, grid_w, merge, 2)
            coords = coords.transpose(0, 2, 1, 3, 4).reshape(h * w, 2)
            coords = np.tile(coords, (t, 1))
            coords = coords.reshape(-1, unit, 2)[index].reshape(t * h * w, 2)
            return index, window_lengths, coords

        for lane, grids in enumerate(lane_grids):
            patch_offset = unit_offset = 0
            window_ends = []
            frame_ends = []
            for grid in grids:
                if not np.any(grid):
                    continue
                t, h, w = map(int, grid)
                patch_count = t * h * w
                window_index, window_lengths, coords = grid_layout(t, h, w)
                unit_count = patch_count // unit
                patch_slice = slice(patch_offset, patch_offset + patch_count)
                unit_slice = slice(unit_offset, unit_offset + unit_count)
                indices[lane, unit_slice, 0] = window_index + unit_offset
                position_ids[lane, patch_slice] = coords
                window_ends.extend(patch_offset + np.cumsum(window_lengths))
                frame_ends.extend(patch_offset + np.arange(1, t + 1, dtype=np.int32) * h * w)
                patch_offset += patch_count
                unit_offset += unit_count
            indices[lane, :, 1] = np.argsort(indices[lane, :, 0]).astype(np.int32)
            for layout, ends in enumerate((window_ends, frame_ends)):
                count = len(ends)
                cu_seqlens[lane, layout, 1 : count + 1] = ends
                cu_seqlens[lane, layout, count + 1 :] = patch_offset
        # cu_seqlens[:, 0] is the window layout, [:, 1] the full-frame layout.
        window_cu_seqlens = cu_seqlens[:, 0]
        full_cu_seqlens = cu_seqlens[:, 1]
        # Static bounds must depend only on the compile bucket, not the request's
        # exact segment values, so requests in one bucket share a compilation.
        window_max_seq_len = min(capacity, window * window * unit)
        return (
            indices,
            position_ids,
            VisionAttentionMetadata(
                window_cu_seqlens,
                max_seq_len=window_max_seq_len,
            ),
            VisionAttentionMetadata(
                full_cu_seqlens,
                max_seq_len=capacity,
            ),
        )

    @jax.jit
    def _encode_jit(
        self,
        patches: jax.Array,
        indices: jax.Array,
        position_ids: jax.Array,
        window_attn: VisionAttentionMetadata,
        full_attn: VisionAttentionMetadata,
    ) -> jax.Array:
        features = self._forward(patches, indices, position_ids, window_attn, full_attn)
        if self.mesh is None:
            return features
        # Keep the DP lane-to-replicated transition inside the compiled encode.
        # An eager reshard of a multi-device result can otherwise stage through
        # the host when no source device owns the complete array.
        return jax.sharding.reshard(
            features,
            NamedSharding(
                self.mesh,
                PartitionSpec(*([None] * features.ndim)),
            ),
        )


class Qwen2_5_VLForConditionalGeneration(nnx.Module, InModelMultimodalContract):
    """Qwen2.5-VL: vision tower + Qwen2 backbone (+ MRoPE) + lm_head.

    The visual encode stays outside the backbone JIT.  MRoPE is handled
    transparently by ``Qwen2Model`` (mrope-aware RoPE + 3-D positions).
    """

    mrope_position_axes = 3

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16

        # Language backbone.
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

        # Vision tower.
        self.visual_config = config.vision_config

        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        vision_tp = resolve_encoder_tp(
            mesh, global_server_args_dict.get("vision_encoder_parallel", "dp")
        )
        self.visual = Qwen2_5_VisionTransformer(
            config=self.visual_config,
            dtype=self.dtype,
            rngs=rngs,
            mesh=mesh,
            norm_eps=getattr(self.visual_config, "rms_norm_eps", 1e-6),
            vision_tp=vision_tp,
            input_buckets=tuple(
                resolve_vision_patch_buckets(
                    global_server_args_dict.get("precompile_vision_patch_paddings")
                )
            ),
        )

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.get_input_embeddings()

    def precompile_multimodal(self) -> None:
        self.visual.precompile()

    def get_multimodal_embedding_packed_capacities(self) -> tuple[int, ...]:
        rows = encoder_num_lanes(self.mesh, self.visual.vision_tp)
        unit = self.visual.spatial_merge_unit
        return tuple(rows * bucket // unit for bucket in self.visual.input_buckets)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        return self._get_visual_feature(items)

    def get_video_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        return self._get_visual_feature(items)

    def _get_visual_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        num_lanes = encoder_num_lanes(self.mesh, self.visual.vision_tp)
        if not self.visual.vision_tp:
            return run_dp_sharded_encoder(
                self.visual,
                items,
                num_lanes=num_lanes,
                buckets=self.visual.input_buckets,
                merge_unit=self.visual.spatial_merge_unit,
            )

        patches, grid_thw, output_indices = pack_vision_inputs(
            items,
            num_lanes=num_lanes,
            buckets=self.visual.input_buckets,
            merge_unit=self.visual.spatial_merge_unit,
        )
        output = self.visual(patches, grid_thw)
        return restore_encoder_output(output, output_indices, self.mesh)

    def get_multimodal_encode_funcs(self):
        return {
            Modality.IMAGE: self.get_image_feature,
            Modality.MULTI_IMAGES: self.get_image_feature,
            Modality.VIDEO: self.get_video_feature,
        }

    def load_weights(self, model_config: ModelConfig) -> None:
        # Text backbone + lm_head.
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        loader.load_weights_from_safetensors(create_qwen2_weight_mappings(self.text_config))
        logger.info("Qwen2.5-VL (LLM) weights loaded.")
        # ViT weights — carry vision head info so _split_qkv_weight can slice the
        # fused ``qkv.weight`` / ``qkv.bias`` into q_proj, k_proj, v_proj.
        vc = self.visual_config
        vision_model_config = SimpleNamespace(
            model_path=model_config.model_path,
            num_attention_heads=vc.num_heads,
            hidden_size=vc.hidden_size,
            get_total_num_kv_heads=lambda: vc.num_heads,  # no GQA in ViT
        )
        self._load_vision_weights(vision_model_config)

    def _load_vision_weights(self, model_config) -> None:
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = self._vision_weight_mappings()
        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(mappings)
        else:
            loader.load_weights_from_safetensors(mappings)
        logger.info("Qwen2.5-VL ViT weights loaded.")

    def _vision_weight_mappings(self) -> dict:
        tp = self.visual.specs.tp
        col = (None, "tensor") if tp else (None, None)
        row = ("tensor", None) if tp else (None, None)

        mappings = {
            # Patch embed Conv3D: PyTorch [out,in,kd,kh,kw] → JAX [kd,kh,kw,in,out].
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
            **self._merger_mlp_mappings(col, row),
        }
        for i in range(getattr(self.visual_config, "depth", 0)):
            mappings.update(self._block_mappings(i, col, row))
        return mappings

    @staticmethod
    def _merger_mlp_mappings(col, row) -> dict:
        """Weight mappings for the patch merger MLP (mlp.0 / mlp.2 in HF)."""
        return {
            "visual.merger.mlp.0.weight": WeightMapping(
                target_path="visual.merger.mlp_fc1.weight", sharding=col, transpose=True
            ),
            "visual.merger.mlp.0.bias": WeightMapping(
                target_path="visual.merger.mlp_fc1.bias", sharding=(None,), transpose=False
            ),
            "visual.merger.mlp.2.weight": WeightMapping(
                target_path="visual.merger.mlp_fc2.weight", sharding=row, transpose=True
            ),
            "visual.merger.mlp.2.bias": WeightMapping(
                target_path="visual.merger.mlp_fc2.bias", sharding=(None,), transpose=False
            ),
        }

    @staticmethod
    def _block_mappings(layer_idx: int, col, row) -> dict:
        """Weight mappings for one ViT block (``visual.blocks.{i}.*``).

        The fused ``qkv.weight`` / ``qkv.bias`` are split into separate
        q/k/v projections so column-parallel sharding is TP-safe (each
        projection independently stripe-interleaves its own output slice).
        """
        p = f"visual.blocks.{layer_idx}"
        return {
            f"{p}.norm1.weight": WeightMapping(
                target_path=f"{p}.norm1.scale", sharding=(None,), transpose=False
            ),
            f"{p}.norm2.weight": WeightMapping(
                target_path=f"{p}.norm2.scale", sharding=(None,), transpose=False
            ),
            f"{p}.attn.qkv.weight": WeightMapping(
                target_path=[
                    f"{p}.attn.q_proj.weight",
                    f"{p}.attn.k_proj.weight",
                    f"{p}.attn.v_proj.weight",
                ],
                sharding=col,
                transpose=True,
            ),
            f"{p}.attn.qkv.bias": WeightMapping(
                target_path=[
                    f"{p}.attn.q_proj.bias",
                    f"{p}.attn.k_proj.bias",
                    f"{p}.attn.v_proj.bias",
                ],
                sharding=(None,),
                transpose=False,
            ),
            f"{p}.attn.proj.weight": WeightMapping(
                target_path=f"{p}.attn.proj.weight", sharding=row, transpose=True
            ),
            f"{p}.attn.proj.bias": WeightMapping(
                target_path=f"{p}.attn.proj.bias", sharding=(None,), transpose=False
            ),
            f"{p}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{p}.mlp.gate_proj.weight", sharding=col, transpose=True
            ),
            f"{p}.mlp.gate_proj.bias": WeightMapping(
                target_path=f"{p}.mlp.gate_proj.bias", sharding=(None,), transpose=False
            ),
            f"{p}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{p}.mlp.up_proj.weight", sharding=col, transpose=True
            ),
            f"{p}.mlp.up_proj.bias": WeightMapping(
                target_path=f"{p}.mlp.up_proj.bias", sharding=(None,), transpose=False
            ),
            f"{p}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{p}.mlp.down_proj.weight", sharding=row, transpose=True
            ),
            f"{p}.mlp.down_proj.bias": WeightMapping(
                target_path=f"{p}.mlp.down_proj.bias", sharding=(None,), transpose=False
            ),
        }

    def get_embed_and_head(self):
        if getattr(self.text_config, "tie_word_embeddings", False):
            w = self.model.embed_tokens.embedding.value
            return (w, w)
        return (self.model.embed_tokens.embedding.value, self.lm_head.embedding.value)

    def set_embed_and_head(
        self, embed_weight: jax.Array | None = None, head_weight: jax.Array | None = None
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
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, memory_pools.token_to_kv_pool
        )
        head = (
            self.model.embed_tokens
            if getattr(self.text_config, "tie_word_embeddings", False)
            else self.lm_head
        )
        output = self.logits_processor(hidden_states, head, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None


EntryClass = Qwen2_5_VLForConditionalGeneration
