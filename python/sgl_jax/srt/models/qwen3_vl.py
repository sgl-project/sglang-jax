import logging
from collections.abc import Callable
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen3 import QWen3Model, create_qwen3_weight_mappings
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    precompile_mrope_vision_model,
    run_mrope_vision_model,
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


def _merge_order(x: np.ndarray, t: int, h: int, w: int, merge: int) -> np.ndarray:
    return (
        np.broadcast_to(x, (t, *x.shape))
        .reshape(t, h // merge, merge, w // merge, merge, *x.shape[2:])
        .transpose(0, 1, 3, 2, 4, *range(5, x.ndim + 3))
        .reshape(t * h * w, *x.shape[2:])
    )


def _rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    left, right = x[..., :half], x[..., half:]
    cos, sin = jnp.cos(freqs)[:, :, None], jnp.sin(freqs)[:, :, None]
    return jnp.concatenate((left * cos - right * sin, left * sin + right * cos), axis=-1).astype(
        x.dtype
    )


class Qwen3VLPatchEmbed(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp):
        self.channels = config.in_channels
        self.temporal = config.temporal_patch_size
        self.patch = config.patch_size
        self.hidden = config.hidden_size
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, tp)
        self.proj = nnx.Conv(
            self.channels,
            self.hidden,
            (self.temporal, self.patch, self.patch),
            strides=(self.temporal, self.patch, self.patch),
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        batch, length, _ = x.shape
        sh = self.specs.sharding(self.specs.batch_axis)
        x = x.reshape(
            batch * length,
            self.channels,
            self.temporal,
            self.patch,
            self.patch,
            out_sharding=sh,
        )
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x, out_sharding=sh).reshape(batch, length, self.hidden, out_sharding=sh)
        if self.mesh is not None:
            x = apply_data_sharding(x, self.mesh, PartitionSpec(self.specs.batch_axis))
        return x


class Qwen3VLVisionMLP(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        self.fc1 = LinearBase(
            config.hidden_size,
            config.intermediate_size,
            mesh,
            use_bias=True,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.fc2 = LinearBase(
            config.intermediate_size,
            config.hidden_size,
            mesh,
            use_bias=True,
            kernel_axes=specs.row_kernel_axes,
            params_dtype=dtype,
        )
        self.specs = specs
        self.approximate = config.hidden_act == "gelu_pytorch_tanh"

    def __call__(self, x):
        specs = self.specs
        x, _ = self.fc1(x, out_sharding=specs.sharding(specs.batch_axis, None, specs.tensor_axis))
        x = jax.nn.gelu(x, approximate=self.approximate)
        return self.fc2(x, out_sharding=specs.sharding(specs.batch_axis))[0]


class Qwen3VLVisionAttention(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        self.hidden = config.hidden_size
        self.heads = config.num_heads
        self.head_dim = self.hidden // self.heads
        self.specs = specs
        if specs.tp:
            assert (
                self.heads % int(mesh.shape["tensor"]) == 0
            ), f"vision num_heads={self.heads} must be divisible by tp={mesh.shape['tensor']}"
        linear = lambda: LinearBase(
            self.hidden,
            self.hidden,
            mesh,
            use_bias=True,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.q_proj, self.k_proj, self.v_proj = linear(), linear(), linear()
        self.proj = LinearBase(
            self.hidden,
            self.hidden,
            mesh,
            use_bias=True,
            kernel_axes=specs.row_kernel_axes,
            params_dtype=dtype,
        )
        self.backend = make_vision_attention_backend(
            mesh,
            sm_scale=self.head_dim**-0.5,
            causal=False,
            head_tp=specs.tp,
            # Qwen3-VL vision uses one block-diagonal full-frame layout for every
            # layer, so route it through the packed varlen kernel.
            use_varlen=True,
        )

    def __call__(self, x, freqs, metadata):
        batch, length, _ = x.shape
        specs = self.specs
        col = specs.sharding(specs.batch_axis, None, specs.tensor_axis)
        q, k, v = (
            layer(x, out_sharding=col)[0]
            for layer in (
                self.q_proj,
                self.k_proj,
                self.v_proj,
            )
        )
        sharding = specs.sharding(specs.batch_axis, None, specs.tensor_axis, None)
        q, k, v = (
            value.reshape(batch, length, self.heads, self.head_dim, out_sharding=sharding)
            for value in (q, k, v)
        )
        output = self.backend(_rope(q, freqs), _rope(k, freqs), v, metadata)
        output = output.reshape(batch, length, self.hidden, out_sharding=col)
        return self.proj(output, out_sharding=specs.sharding(specs.batch_axis))[0]


class Qwen3VLVisionBlock(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp):
        specs = VisionShardSpecs(mesh, tp)
        norm = lambda: nnx.LayerNorm(
            config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=dtype,
            use_fast_variance=False,
            rngs=rngs,
        )
        self.norm1, self.norm2 = norm(), norm()
        self.attn = Qwen3VLVisionAttention(config, dtype, mesh, specs)
        self.mlp = Qwen3VLVisionMLP(config, dtype, mesh, specs)

    def __call__(self, x, freqs, metadata):
        x = x + self.attn(self.norm1(x), freqs, metadata)
        return x + self.mlp(self.norm2(x))


class Qwen3VLPatchMerger(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp, postshuffle):
        self.hidden = config.hidden_size * config.spatial_merge_size**2
        self.postshuffle = postshuffle
        self.specs = VisionShardSpecs(mesh, tp)
        self.norm = nnx.LayerNorm(
            self.hidden if postshuffle else config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=dtype,
            use_fast_variance=False,
            rngs=rngs,
        )
        self.fc1 = LinearBase(
            self.hidden,
            self.hidden,
            mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.fc2 = LinearBase(
            self.hidden,
            config.out_hidden_size,
            mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, x):
        specs = self.specs
        sharding = specs.sharding(specs.batch_axis)
        if self.postshuffle:
            x = self.norm(x.reshape(x.shape[0], -1, self.hidden, out_sharding=sharding))
        else:
            x = self.norm(x).reshape(x.shape[0], -1, self.hidden, out_sharding=sharding)
        x, _ = self.fc1(x, out_sharding=specs.sharding(specs.batch_axis, None, specs.tensor_axis))
        x = jax.nn.gelu(x, approximate=False)
        return self.fc2(x, out_sharding=sharding)[0]


class Qwen3VLVisionModel(nnx.Module):
    def __init__(
        self,
        config,
        dtype,
        rngs=None,
        mesh=None,
        tp=False,
        input_buckets: tuple[int, ...] | None = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.mesh = mesh
        self.vision_tp = tp
        self.specs = VisionShardSpecs(mesh, tp)
        self.input_buckets = input_buckets or tuple(resolve_vision_patch_buckets(None))
        self.merge = int(config.spatial_merge_size)
        self.spatial_merge_unit = self.merge**2
        if any(bucket <= 0 or bucket % self.spatial_merge_unit for bucket in self.input_buckets):
            raise ValueError(
                f"vision patch buckets must be positive multiples of {self.spatial_merge_unit}"
            )
        self.num_grid = int(config.num_position_embeddings**0.5)
        self.rotary_dim = int(config.hidden_size) // int(config.num_heads) // 2
        self.patch_embed = Qwen3VLPatchEmbed(config, dtype, rngs, mesh, tp)
        self.pos_embed = Embed(
            config.num_position_embeddings,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=(None, None),
            mesh=mesh,
        )
        self.blocks = nnx.List(
            [Qwen3VLVisionBlock(config, dtype, rngs, mesh, tp) for _ in range(config.depth)]
        )
        self.deepstack_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_mergers = nnx.List(
            [
                Qwen3VLPatchMerger(config, dtype, rngs, mesh, tp, True)
                for _ in self.deepstack_indexes
            ]
        )
        self.merger = Qwen3VLPatchMerger(config, dtype, rngs, mesh, tp, False)
        self.patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2

    def _lane_metadata(self, grids: list[tuple[int, int, int]], capacity: int):
        """Build one lane's padded metadata once on the host."""
        pos_indices = np.zeros((4, capacity), dtype=np.int32)
        pos_weights = np.zeros((4, capacity), dtype=np.float32)
        position_ids = np.zeros((capacity, 2), dtype=np.int32)
        cu_seqlens = np.zeros(capacity // self.spatial_merge_unit + 1, dtype=np.int32)
        patch_offset, boundary_offset = 0, 1
        for t, h, w in grids:
            end = patch_offset + t * h * w
            ys = np.linspace(0, self.num_grid - 1, h, dtype=np.float32)
            xs = np.linspace(0, self.num_grid - 1, w, dtype=np.float32)
            y0, x0 = ys.astype(np.int32), xs.astype(np.int32)
            y1, x1 = np.minimum(y0 + 1, self.num_grid - 1), np.minimum(x0 + 1, self.num_grid - 1)
            dy, dx = ys - y0, xs - x0
            indices = (
                y0[:, None] * self.num_grid + x0[None],
                y0[:, None] * self.num_grid + x1[None],
                y1[:, None] * self.num_grid + x0[None],
                y1[:, None] * self.num_grid + x1[None],
            )
            weights = (
                (1 - dy[:, None]) * (1 - dx[None]),
                (1 - dy[:, None]) * dx[None],
                dy[:, None] * (1 - dx[None]),
                dy[:, None] * dx[None],
            )
            pos_indices[:, patch_offset:end] = np.stack(
                [_merge_order(x[..., None], t, h, w, self.merge)[:, 0] for x in indices]
            )
            pos_weights[:, patch_offset:end] = np.stack(
                [_merge_order(x[..., None], t, h, w, self.merge)[:, 0] for x in weights]
            )
            rows, cols = np.indices((h, w))
            coords = _merge_order(np.stack((rows, cols), axis=-1), t, h, w, self.merge)
            position_ids[patch_offset:end] = coords
            cu_seqlens[boundary_offset : boundary_offset + t] = patch_offset + np.arange(
                1, t + 1, dtype=np.int32
            ) * (h * w)
            patch_offset, boundary_offset = end, boundary_offset + t
        cu_seqlens[boundary_offset:] = patch_offset
        return pos_indices, pos_weights, position_ids, cu_seqlens

    def __call__(
        self,
        patches: jax.Array,
        grid_thw: np.ndarray | jax.Array,
    ) -> jax.Array:
        return self.encode(patches, grid_thw)

    def _forward(
        self,
        patches: jax.Array,
        pos_indices: jax.Array,
        pos_weights: jax.Array,
        position_ids: jax.Array,
        metadata: VisionAttentionMetadata,
    ) -> tuple[jax.Array, jax.Array]:
        inv_freq = 1.0 / (
            10000.0 ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        rotary_pos_emb = jnp.concatenate(
            (
                position_ids[..., :1].astype(jnp.float32) * inv_freq,
                position_ids[..., 1:].astype(jnp.float32) * inv_freq,
            ),
            axis=-1,
        )
        x = self.patch_embed(patches)
        pos = self.pos_embed.embedding.at[pos_indices].get(
            out_sharding=self.specs.sharding(self.specs.batch_axis)
        )
        x += jnp.sum(pos * pos_weights[..., None].astype(pos.dtype), axis=1).astype(x.dtype)
        # One block-diagonal (full-frame) layout, planned on the host and shared
        # by every block.
        deepstack = []
        for index, block in enumerate(self.blocks):
            x = block(x, rotary_pos_emb, metadata)
            if index in self.deepstack_indexes:
                merger = self.deepstack_mergers[self.deepstack_indexes.index(index)]
                deepstack.append(merger(x))
        merged = self.merger(x)
        deepstack = (
            jnp.stack(deepstack, axis=2)
            if deepstack
            else jnp.empty((x.shape[0], merged.shape[1], 0, merged.shape[2]), x.dtype)
        )
        return merged, deepstack

    @jax.jit
    def _encode_jit(
        self,
        patches: jax.Array,
        pos_indices: jax.Array,
        pos_weights: jax.Array,
        position_ids: jax.Array,
        metadata: VisionAttentionMetadata,
    ) -> jax.Array:
        output, deepstack = self._forward(
            patches,
            pos_indices,
            pos_weights,
            position_ids,
            metadata,
        )
        if self.mesh is not None:
            output = apply_data_sharding(output, self.mesh, PartitionSpec(self.specs.batch_axis))
            deepstack = apply_data_sharding(
                deepstack, self.mesh, PartitionSpec(self.specs.batch_axis)
            )
        # Concatenate deepstack planes onto the trailing feature axis so the merge
        # gathers one [rows, cap, (1+D)*H] tensor. D is static at trace time.
        deepstack_dim = deepstack.shape[2]
        if deepstack_dim:
            b, cap, h = output.shape
            output = jnp.concatenate(
                [output, deepstack.reshape(b, cap, deepstack_dim * h)], axis=-1
            )
        return output

    def encode(self, patches: jax.Array, grid_thw: np.ndarray | jax.Array) -> jax.Array:
        batch_sharding = self.specs.sharding(self.specs.batch_axis)
        patches = jax.device_put(patches, batch_sharding)
        metadata = jax.device_put(
            self._build_metadata(grid_thw, patches.shape[1]),
            batch_sharding,
        )
        if self.mesh is None:
            return self._encode_jit(patches, *metadata)
        with jax.set_mesh(self.mesh):
            return self._encode_jit(patches, *metadata)

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

    def _build_metadata(self, grid_thw: np.ndarray | jax.Array, capacity: int):
        grid_thw = np.asarray(jax.device_get(grid_thw), dtype=np.int32)
        if grid_thw.ndim == 2:
            grid_thw = grid_thw[None]
        lane_metadata = [
            self._lane_metadata(
                [tuple(map(int, grid)) for grid in lane if np.any(grid)],
                capacity,
            )
            for lane in grid_thw
        ]
        pos_indices, pos_weights, position_ids, cu_seqlens = (
            np.stack(values) for values in zip(*lane_metadata, strict=True)
        )
        metadata = VisionAttentionMetadata(
            cu_seqlens,
            max_seq_len=capacity,
        )
        return pos_indices, pos_weights, position_ids, metadata


class Qwen3VLForConditionalGeneration(nnx.Module, InModelMultimodalContract):
    mrope_position_axes = 3

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16
        rope = getattr(self.text_config, "rope_parameters", None)
        if rope:
            self.text_config.rope_theta = rope.get(
                "rope_theta", getattr(self.text_config, "rope_theta", 5_000_000)
            )
            self.text_config.rope_scaling = {
                "rope_type": rope.get("rope_type", "default"),
                "mrope_section": rope.get("mrope_section", [24, 20, 20]),
                "mrope_interleaved": True,
            }
        elif not getattr(self.text_config, "rope_scaling", None):
            self.text_config.rope_scaling = {
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
                "mrope_interleaved": True,
            }
        self.model = QWen3Model(self.text_config, mesh=mesh, dtype=self.dtype)
        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
                mesh=mesh,
            )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=mesh)
        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        encoder_tp = resolve_encoder_tp(
            mesh, global_server_args_dict.get("vision_encoder_parallel", "dp")
        )
        self.visual = Qwen3VLVisionModel(
            config.vision_config,
            self.dtype,
            rngs,
            mesh,
            encoder_tp,
            tuple(
                resolve_vision_patch_buckets(
                    global_server_args_dict.get("precompile_vision_patch_paddings")
                )
            ),
        )
        self.deepstack_visual_layers = len(self.visual.deepstack_indexes)

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
        return run_mrope_vision_model(
            self.visual,
            items,
            mesh=self.mesh,
            num_lanes=num_lanes,
            buckets=self.visual.input_buckets,
            merge_unit=self.visual.spatial_merge_unit,
            rope_type="rope_3d",
        )

    def get_multimodal_encode_funcs(self):
        return {
            Modality.IMAGE: self.get_image_feature,
            Modality.MULTI_IMAGES: self.get_image_feature,
            Modality.VIDEO: self.get_video_feature,
        }

    def load_weights(self, model_config: ModelConfig) -> None:
        text_loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        text_loader.load_weights_from_safetensors(
            create_qwen3_weight_mappings(
                self.text_config, source_prefix="model.language_model", target_prefix="model"
            )
        )
        config = self.config.vision_config
        vision_config = SimpleNamespace(
            model_path=model_config.model_path,
            num_attention_heads=config.num_heads,
            hidden_size=config.hidden_size,
            get_total_num_kv_heads=lambda: config.num_heads,
        )
        WeightLoader(self, vision_config, self.mesh, self.dtype).load_weights_from_safetensors(
            self._vision_weight_mappings()
        )
        logger.info("Qwen3-VL weights loaded successfully")

    @classmethod
    def create_vision_weight_mappings(cls, config, visual):
        specs = visual.specs
        col, row = specs.col_kernel_axes, specs.row_kernel_axes
        mappings = {
            "model.visual.patch_embed.proj.weight": WeightMapping(
                "visual.patch_embed.proj.kernel",
                (None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "model.visual.patch_embed.proj.bias": WeightMapping(
                "visual.patch_embed.proj.bias", (None,), transpose=False
            ),
            "model.visual.pos_embed.weight": WeightMapping(
                "visual.pos_embed.embedding", (None, None), transpose=False
            ),
        }
        for index in range(config.vision_config.depth):
            source, target = f"model.visual.blocks.{index}", f"visual.blocks.{index}"
            mappings.update(cls._block_mappings(source, target, col, row))
        mappings.update(cls._merger_mappings("model.visual.merger", "visual.merger", col, row))
        for index, _ in enumerate(visual.deepstack_indexes):
            mappings.update(
                cls._merger_mappings(
                    f"model.visual.deepstack_merger_list.{index}",
                    f"visual.deepstack_mergers.{index}",
                    col,
                    row,
                )
            )
        return mappings

    def _vision_weight_mappings(self):
        return self.create_vision_weight_mappings(self.config, self.visual)

    @staticmethod
    def _linear(source, target, sharding):
        return {
            f"{source}.weight": WeightMapping(target + ".weight", sharding, transpose=True),
            f"{source}.bias": WeightMapping(target + ".bias", (None,), transpose=False),
        }

    @classmethod
    def _block_mappings(cls, source, target, col, row):
        mappings = {}
        for name in ("norm1", "norm2"):
            mappings[f"{source}.{name}.weight"] = WeightMapping(
                f"{target}.{name}.scale", (None,), transpose=False
            )
            mappings[f"{source}.{name}.bias"] = WeightMapping(
                f"{target}.{name}.bias", (None,), transpose=False
            )
        mappings[f"{source}.attn.qkv.weight"] = WeightMapping(
            [f"{target}.attn.{name}_proj.weight" for name in "qkv"], col, transpose=True
        )
        mappings[f"{source}.attn.qkv.bias"] = WeightMapping(
            [f"{target}.attn.{name}_proj.bias" for name in "qkv"], (None,), transpose=False
        )
        mappings.update(cls._linear(f"{source}.attn.proj", f"{target}.attn.proj", row))
        mappings.update(cls._linear(f"{source}.mlp.linear_fc1", f"{target}.mlp.fc1", col))
        mappings.update(cls._linear(f"{source}.mlp.linear_fc2", f"{target}.mlp.fc2", row))
        return mappings

    @classmethod
    def _merger_mappings(cls, source, target, col, row):
        mappings = {
            f"{source}.norm.weight": WeightMapping(
                f"{target}.norm.scale", (None,), transpose=False
            ),
            f"{source}.norm.bias": WeightMapping(f"{target}.norm.bias", (None,), transpose=False),
        }
        mappings.update(cls._linear(f"{source}.linear_fc1", f"{target}.fc1", col))
        mappings.update(cls._linear(f"{source}.linear_fc2", f"{target}.fc2", row))
        return mappings

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.embedding.value
        return (
            (embed, embed)
            if getattr(self.text_config, "tie_word_embeddings", False)
            else (
                embed,
                self.lm_head.embedding.value,
            )
        )

    def set_embed_and_head(self, embed_weight=None, head_weight=None):
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None and not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden, aux, kv, callbacks = self.model(forward_batch, memory_pools.token_to_kv_pool)
        head = (
            self.model.embed_tokens
            if getattr(self.text_config, "tie_word_embeddings", False)
            else self.lm_head
        )
        output = self.logits_processor(hidden, head, logits_metadata, aux_hidden_states=aux)
        return output, {"token_to_kv_pool": kv}, callbacks, None


EntryClass = Qwen3VLForConditionalGeneration
