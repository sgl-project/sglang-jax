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
from numba import njit, types
from transformers import modeling_flax_utils

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2Model
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import (
    QwenVLModelVitConfig,
)
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    precompile_mrope_vision_model,
    run_mrope_vision_model,
)
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
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
    """Apply precomputed vision RoPE to ``x[tokens, heads, head_dim]``."""
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
        mesh: Mesh,
        rngs: nnx.Rngs = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
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
        """*x*: ``[tokens, C·T·H·W]`` → ``[tokens, hidden_size]``."""
        tokens, D = x.shape
        C = D // (self.temporal_patch_size * self.patch_size * self.patch_size)
        x = x.reshape(tokens, C, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = apply_data_sharding(x, self.mesh, PartitionSpec(self.specs.batch_axis))

        # Each token is one independent convolution input patch.
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        sh = None
        if "data" in self.mesh.abstract_mesh.explicit_axes:
            sh = self.specs.sharding(self.specs.batch_axis)

        x = self.proj(x, out_sharding=sh)
        return x.reshape(tokens, self.hidden_size, out_sharding=sh)


class Qwen2_5_VLMLP(nnx.Module):
    """ViT MLP: gate/up → SiLU gate → down."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh,
        rngs: nnx.Rngs = None,
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
        col = specs.sharding(specs.batch_axis, specs.tensor_axis)
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
        mesh: Mesh,
        rngs: nnx.Rngs = None,
        vision_tp: bool = False,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)

        if self.specs.tp:
            tp_size = int(mesh.shape["tensor"])
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

        self.attn_backend = make_vision_attention_backend(
            mesh,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            causal=False,
            head_tp=self.specs.tp,
            use_varlen=True,
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_cos: jax.Array,
        rotary_sin: jax.Array,
        cu_seqlens: jax.Array,
        *,
        max_seq_len: int,
    ) -> jax.Array:
        tokens, D = x.shape
        specs = self.specs
        col = specs.sharding(specs.batch_axis, specs.tensor_axis)

        # Project Q, K, V separately (TP-safe: each is independently column-parallel).
        q, _ = self.q_proj(x, out_sharding=col)
        k, _ = self.k_proj(x, out_sharding=col)
        v, _ = self.v_proj(x, out_sharding=col)

        hs = specs.sharding(specs.batch_axis, specs.tensor_axis, None)
        q = q.reshape(tokens, self.num_heads, self.head_dim, out_sharding=hs)
        k = k.reshape(tokens, self.num_heads, self.head_dim, out_sharding=hs)
        v = v.reshape(tokens, self.num_heads, self.head_dim, out_sharding=hs)

        q = _apply_rotary_pos_emb_vision(q, rotary_cos, rotary_sin)
        k = _apply_rotary_pos_emb_vision(k, rotary_cos, rotary_sin)

        out = self.attn_backend(q, k, v, cu_seqlens, max_seq_len=max_seq_len)
        out = out.reshape(tokens, D, out_sharding=col)
        out, _ = self.proj(out, out_sharding=specs.sharding(specs.batch_axis))
        return out


class Qwen2_5_VisionBlock(nnx.Module):
    """One ViT transformer block: attn (pre-norm) + MLP (pre-norm)."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh,
        rngs: nnx.Rngs = None,
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
        cu_seqlens: jax.Array,
        *,
        max_seq_len: int,
    ) -> jax.Array:
        x = x + self.attn(
            self.norm1(x), rotary_cos, rotary_sin, cu_seqlens, max_seq_len=max_seq_len
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchMerger(nnx.Module):
    """Spatial merge: LN → reshape(sms²) → MLP → [tokens/sms², d_model]."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable,
        spatial_merge_size: int,
        dtype: jnp.dtype,
        mesh: Mesh,
        rngs: nnx.Rngs = None,
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
        x = x.reshape(-1, self.hidden_size, out_sharding=row)
        x, _ = self.mlp_fc1(x, out_sharding=specs.sharding(specs.batch_axis, specs.tensor_axis))
        x = self.mlp_act(x)
        x, _ = self.mlp_fc2(x, out_sharding=row)
        return x


class Qwen2_5_VisionTransformer(nnx.Module):
    """Qwen2.5-VL ViT: patch embed → window / full-attn blocks → merge → reorder."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh,
        rngs: nnx.Rngs = None,
        norm_eps: float = 1e-6,
        vision_tp: bool = False,
        input_buckets: tuple[int, ...] | None = None,
    ):
        self.mesh = mesh
        self.dtype = dtype
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

    # Compile (or load the cache) before serving, with dimensions as runtime values.
    # Accept strided and read-only grids without additional specializations.
    @staticmethod
    @njit(
        types.int32[::1](
            types.Array(types.int32, 3, "A", readonly=True),
            types.int64,
            types.int64,
            types.int64,
        ),
        nogil=True,
        cache=True,
    )
    def _build_metadata(grid_thw, capacity, merge, window):
        """Write lane-local permutations, positions and boundaries in one pass.

        The flat buffer matches ``Qwen2_5_VisionTransformer._metadata_views``.
        Visit only valid merge units in window order, avoiding padded indices,
        coordinate grids, gathers and per-image temporary arrays.
        """
        if merge <= 0 or window <= 0 or capacity < 0:
            raise ValueError("merge/window must be positive and capacity non-negative")
        unit = merge * merge
        if capacity % unit:
            raise ValueError("capacity must be divisible by the spatial merge unit")
        if grid_thw.shape[2] != 3:
            raise ValueError("grid_thw must have three coordinates per grid")
        num_lanes = grid_thw.shape[0]
        num_units = capacity // unit
        positions_start = 2 * num_units
        window_start = positions_start + 2 * capacity
        full_start = window_start + num_units + 1
        metadata = np.zeros((num_lanes, full_start + num_units + 1), dtype=np.int32)

        for lane in range(num_lanes):
            # Padding units retain the identity permutation and zero positions.
            for index in range(num_units):
                metadata[lane, 2 * index] = index
                metadata[lane, 2 * index + 1] = index
            patch_offset = unit_offset = 0
            window_segment = frame_segment = 1
            for image in range(grid_thw.shape[1]):
                t, h, w = grid_thw[lane, image]
                if t == 0 and h == 0 and w == 0:
                    continue
                # Validate before writing: Numba does not bounds-check array access.
                if t <= 0 or h <= 0 or w <= 0 or h % merge or w % merge:
                    raise ValueError("grid dimensions must be positive and spatially merge-aligned")
                if t * h * w > capacity - patch_offset:
                    raise ValueError("vision grids exceed the lane capacity")
                grid_h, grid_w = h // merge, w // merge
                image_unit_offset = unit_offset
                for frame in range(t):
                    frame_unit_offset = image_unit_offset + frame * grid_h * grid_w
                    for window_y in range(0, grid_h, window):
                        for window_x in range(0, grid_w, window):
                            for y in range(window_y, min(window_y + window, grid_h)):
                                for x in range(window_x, min(window_x + window, grid_w)):
                                    source_unit = frame_unit_offset + y * grid_w + x
                                    metadata[lane, 2 * unit_offset] = source_unit
                                    metadata[lane, 2 * source_unit + 1] = unit_offset
                                    for dy in range(merge):
                                        for dx in range(merge):
                                            position = positions_start + 2 * patch_offset
                                            metadata[lane, position] = y * merge + dy
                                            metadata[lane, position + 1] = x * merge + dx
                                            patch_offset += 1
                                    unit_offset += 1
                            metadata[lane, window_start + window_segment] = patch_offset
                            window_segment += 1
                    metadata[lane, full_start + frame_segment] = patch_offset
                    frame_segment += 1
            # Repeated final ends describe empty segments in the bucket padding.
            metadata[lane, window_start + window_segment : full_start] = patch_offset
            metadata[lane, full_start + frame_segment :] = patch_offset
        return metadata.reshape(-1)

    def prepare_metadata(
        self, grid_thw: np.ndarray, capacity: int, *, sharding: NamedSharding
    ) -> jax.Array:
        """Build host attention metadata and upload it with the input sharding."""
        with jax.profiler.TraceAnnotation("encoder_metadata_host_build"):
            grid_thw = np.asarray(grid_thw, dtype=np.int32)
            if grid_thw.ndim == 2:
                grid_thw = grid_thw[None]
            if grid_thw.ndim != 3 or grid_thw.shape[-1] != 3:
                raise ValueError("grid_thw must have shape [items, 3] or [lanes, items, 3]")
            metadata = self._build_metadata(
                grid_thw,
                capacity,
                self.spatial_merge_size,
                self.window_size // self.spatial_merge_size // self.patch_size,
            )
        with jax.profiler.TraceAnnotation("encoder_metadata_device_put"):
            return jax.device_put(metadata, sharding)

    def _metadata_views(self, metadata: np.ndarray | jax.Array, capacity: int):
        """Views of the same buffer on the host and inside the ViT JIT."""
        leading_shape = metadata.shape[:-1]
        num_units = capacity // self.spatial_merge_unit
        indices_end = 2 * num_units
        positions_end = indices_end + 2 * capacity
        window_end = positions_end + num_units + 1
        return (
            metadata[..., :indices_end].reshape(*leading_shape, num_units, 2),
            metadata[..., indices_end:positions_end].reshape(*leading_shape, capacity, 2),
            metadata[..., positions_end:window_end],
            metadata[..., window_end:],
        )

    def _reorder(self, x: jax.Array, indices: jax.Array) -> jax.Array:
        """Gather within each device's lane, using lane-local unit indices."""
        spec = PartitionSpec(self.specs.batch_axis)
        return jax.shard_map(
            lambda values, order: values[order],
            mesh=self.mesh,
            in_specs=(spec, spec),
            out_specs=spec,
            check_vma=False,
        )(x, indices)

    def __call__(
        self,
        patches: jax.Array,
        metadata: jax.Array,
    ) -> jax.Array:
        return self.encode(patches, metadata)

    def _forward(
        self,
        patches: jax.Array,
        indices: jax.Array,
        position_ids: jax.Array,
        window_cu_seqlens: jax.Array,
        full_cu_seqlens: jax.Array,
    ) -> jax.Array:
        """Run the ViT over flat tokens with lane-local indices and boundaries."""
        tokens = patches.shape[0]
        capacity = tokens // encoder_num_lanes(self.mesh, self.vision_tp)
        u = self.spatial_merge_unit
        window_index, reverse_indices = indices[:, 0], indices[:, 1]
        inv_freq = 1.0 / (
            self.theta ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        rotary_pos_emb = (position_ids[..., None].astype(jnp.float32) * inv_freq).reshape(
            tokens, self.rot_dim
        )
        rotary_cos = jnp.cos(rotary_pos_emb)[:, None, :]
        rotary_sin = jnp.sin(rotary_pos_emb)[:, None, :]

        x = self.patch_embed(patches)
        x = x.reshape(tokens // u, u, -1)
        x = self._reorder(x, window_index).reshape(tokens, -1)

        # Select the pre-planned metadata per block: full-frame for the layers in
        # ``fullatt_block_indexes``, otherwise the local-window layout.
        cu_seqlens = (window_cu_seqlens, full_cu_seqlens)
        window = self.window_size // self.spatial_merge_size // self.patch_size
        # Static bounds depend only on the compile bucket and model config so
        # different segment values with the same shapes share a compilation.
        max_seq_lens = (min(capacity, window * window * u), capacity)
        for i, blk in enumerate(self.blocks):
            layout = int(i in self.fullatt_block_indexes)
            x = blk(x, rotary_cos, rotary_sin, cu_seqlens[layout], max_seq_len=max_seq_lens[layout])

        x = self.merger(x)
        return self._reorder(x, reverse_indices)

    @jax.jit
    def encode(
        self,
        patches: jax.Array,
        metadata: jax.Array,
    ) -> jax.Array:
        """Encode flat lane-major patch and metadata buffers."""
        num_lanes = encoder_num_lanes(self.mesh, self.vision_tp)
        capacity = patches.size // (num_lanes * self.patch_dim)
        token_sharding = self.specs.sharding(self.specs.batch_axis)
        patches = patches.reshape(-1, self.patch_dim, out_sharding=token_sharding)
        spec = PartitionSpec(self.specs.batch_axis)
        metadata_views = jax.shard_map(
            partial(self._metadata_views, capacity=capacity),
            mesh=self.mesh,
            in_specs=spec,
            out_specs=(spec,) * 4,
            check_vma=False,
        )(metadata)
        patches = patches.astype(self.dtype)
        return self._forward(patches, *metadata_views)

    def precompile(self) -> None:
        precompile_mrope_vision_model(
            self,
            mesh=self.mesh,
            num_lanes=encoder_num_lanes(self.mesh, self.vision_tp),
            buckets=self.input_buckets,
            patch_dim=self.patch_dim,
            merge_unit=self.spatial_merge_unit,
            rope_type="rope_3d",
            input_sharding=self.specs.sharding(self.specs.batch_axis),
            output_sharding=self.specs.sharding(),
        )


class Qwen2_5_VLForConditionalGeneration(nnx.Module, InModelMultimodalContract):
    """Qwen2.5-VL: vision tower + Qwen2 backbone (+ MRoPE) + lm_head.

    The visual encode stays outside the backbone JIT. MRoPE positions are
    selected by this wrapper and passed to the shared ``Qwen2Model`` backbone.
    """

    mrope_position_axes = 3

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16
        self.is_mrope_enabled = "mrope_section" in (
            getattr(self.text_config, "rope_scaling", None) or {}
        )

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

        vision_tp = resolve_encoder_tp(mesh, getattr(config, "vision_encoder_parallel", "dp"))
        self.visual = Qwen2_5_VisionTransformer(
            config=self.visual_config,
            dtype=self.dtype,
            rngs=rngs,
            mesh=mesh,
            norm_eps=getattr(self.visual_config, "rms_norm_eps", 1e-6),
            vision_tp=vision_tp,
            input_buckets=tuple(
                resolve_vision_patch_buckets(
                    getattr(config, "precompile_vision_patch_paddings", None)
                )
            ),
        )

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.embed_tokens

    def precompile_multimodal(self) -> None:
        self.visual.precompile()

    def get_multimodal_embedding_packed_capacities(self) -> tuple[int, ...]:
        rows = encoder_num_lanes(self.mesh, self.visual.vision_tp)
        unit = self.visual.spatial_merge_unit
        return tuple(rows * bucket // unit for bucket in self.visual.input_buckets)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        num_lanes = encoder_num_lanes(self.mesh, self.visual.vision_tp)
        return run_mrope_vision_model(
            self.visual,
            items,
            mesh=self.mesh,
            num_lanes=num_lanes,
            buckets=self.visual.input_buckets,
            merge_unit=self.visual.spatial_merge_unit,
            rope_type="rope_3d",
            input_sharding=self.visual.specs.sharding(self.visual.specs.batch_axis),
            output_sharding=self.visual.specs.sharding(),
        )

    def get_video_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        return self.get_image_feature(items)

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
        loader.load_weights_from_safetensors(self._language_weight_mappings())
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

    def _language_weight_mappings(self) -> dict:
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

        for layer_idx in range(self.text_config.num_hidden_layers):
            mappings.update(self._language_layer_mappings(layer_idx))

        return mappings

    def _language_layer_mappings(self, layer_idx: int) -> dict:
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
                    f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.o_proj.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                }
            )

        return mappings

    def _load_vision_weights(self, model_config) -> None:
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = self._vision_weight_mappings()
        with self.mesh:
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
        positions = (
            forward_batch.mrope_positions if self.is_mrope_enabled else forward_batch.positions
        )
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, memory_pools.token_to_kv_pool, positions=positions
        )
        head = (
            self.model.embed_tokens
            if getattr(self.text_config, "tie_word_embeddings", False)
            else self.lm_head
        )
        output = self.logits_processor(hidden_states, head, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None


EntryClass = Qwen2_5_VLForConditionalGeneration
