import logging
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.multimodal.common.modality_enum import MultimodalDataItem
from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import KimiK25ModelVitConfig
from sgl_jax.srt.multimodal.in_model.lane_packing import get_grid_thw
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    make_vision_attention_backend,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import (
    VisionShardSpecs,
    apply_data_sharding,
)
from sgl_jax.srt.utils.weight_utils import WeightMapping

init_fn = nnx.initializers.uniform()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Learnable2DInterPosEmbDivided_fixed(nnx.Module):

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
        rngs: nnx.Rngs = None,
    ) -> None:
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode

        _rngs = rngs or nnx.Rngs(0)
        self.weight = nnx.Param(nnx.initializers.normal()(_rngs.params(), (height, width, dim)))

    def __call__(self, grid_thws: jax.Array) -> jax.Array:

        pos_embs = []
        for t, h, w in grid_thws:
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.reshape(-1, self.weight.shape[-1])
            else:
                pos_emb_2d = jax.image.resize(
                    self.weight,
                    shape=(h, w, self.dim),
                    method="bicubic",
                ).reshape(-1, self.dim)

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = jnp.tile(jnp.expand_dims(pos_emb_2d, axis=0), (t, 1, 1))

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = jnp.concatenate(pos_embs, axis=0)
        return out


class Rope2DPosEmbRepeated(nnx.Module):

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: int = 10000,
    ):
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _get_freqs_cis(self, grid_thws):
        half_dim = self.dim // 2
        inv_freq = 1.0 / (
            self.theta_base ** (jnp.arange(0, half_dim, 2, dtype=jnp.float32) / half_dim)
        )
        h_pos = jnp.arange(self.max_height, dtype=jnp.float32)
        w_pos = jnp.arange(self.max_width, dtype=jnp.float32)
        freqs_h = jnp.outer(h_pos, inv_freq)
        freqs_w = jnp.outer(w_pos, inv_freq)

        freqs_h_exp = jnp.tile(freqs_h[:, None, :], (1, self.max_width, 1))
        freqs_w_exp = jnp.tile(freqs_w[None, :, :], (self.max_height, 1, 1))
        freqs = jnp.stack([freqs_w_exp, freqs_h_exp], axis=-1).reshape(
            self.max_height, self.max_width, -1
        )

        cos_table = jnp.cos(freqs)
        sin_table = jnp.sin(freqs)

        results = []
        for t, h, w in grid_thws:
            cos_hw = cos_table[:h, :w, :].reshape(h * w, self.dim // 2)
            sin_hw = sin_table[:h, :w, :].reshape(h * w, self.dim // 2)
            results.append(jnp.tile(jnp.stack([cos_hw, sin_hw], axis=0), (1, t, 1)))
        return jnp.concatenate(results, axis=1)


class KimiK25VisionPatchEmbed(nnx.Module):

    def __init__(
        self,
        rngs: nnx.Rngs | None = None,
        patch_size: int = 14,
        in_channels: int = 3,
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.kernel_size = (patch_size, patch_size)

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterPosEmbDivided_fixed(
                height=pos_emb_height, width=pos_emb_width, num_frames=pos_emb_time, dim=hidden_size
            )
        else:
            raise NotImplementedError(f"No support for pos_emb_type: {pos_emb_type}")

        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=self.kernel_size,
            strides=self.kernel_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.proj(x)

        x = x.reshape(-1, self.hidden_size)
        return x


def apply_2d_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    x_real = x[..., 0::2]
    x_imag = x[..., 1::2]
    x_rot_real = x_real * cos[:, None, :] - x_imag * sin[:, None, :]
    x_rot_imag = x_real * sin[:, None, :] + x_imag * cos[:, None, :]
    return jnp.stack([x_rot_real, x_rot_imag], axis=-1).reshape(x.shape)


class KimiK25VisionAttention(nnx.Module):
    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh,
        rngs: nnx.Rngs | None = None,
        vision_tp: bool = True,
    ):
        assert mesh is not None, "KimiK25VisionAttention requires a sharding Mesh"
        self.mesh = mesh
        self.hidden_size = config.vt_hidden_size
        self.num_heads = config.vt_num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.specs = VisionShardSpecs(mesh, vision_tp)

        # ``head_tp`` keeps every token on every shard and splits the heads. The
        # data-parallel alternative shards the token axis instead, which lane
        # packing now makes representable because a shard holds whole items;
        # it is left off until it has been validated on hardware.
        self.attn_backend = make_vision_attention_backend(
            mesh,
            sm_scale=self.scale,
            causal=False,
            head_tp=vision_tp,
            use_varlen=True,
        )

        _rngs = rngs or nnx.Rngs(0)

        self.qkv_proj = nnx.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        cu_seqlens: jax.Array,
        position_embeddings: jax.Array,
    ) -> jax.Array:
        sum_seq_len, D = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(sum_seq_len, self.num_heads, self.head_dim)
        k = k.reshape(sum_seq_len, self.num_heads, self.head_dim)
        v = v.reshape(sum_seq_len, self.num_heads, self.head_dim)

        cos_emb, sin_emb = position_embeddings[0], position_embeddings[1]
        q = apply_2d_rope(q, cos_emb, sin_emb)
        k = apply_2d_rope(k, cos_emb, sin_emb)

        # The backend's shard_map declares P(batch_axis, head_axis, None); an
        # explicit-axis mesh rejects replicated inputs against that spec, so the
        # layout is stated here rather than inferred.
        qkv_spec = PartitionSpec(self.specs.batch_axis, self.specs.tensor_axis, None)
        q = apply_data_sharding(q, self.mesh, qkv_spec)
        k = apply_data_sharding(k, self.mesh, qkv_spec)
        v = apply_data_sharding(v, self.mesh, qkv_spec)
        cu_seqlens = apply_data_sharding(
            cu_seqlens, self.mesh, PartitionSpec(self.specs.batch_axis)
        )

        # The varlen kernel consumes cumulative lengths directly, so the dense
        # kernel's per-token segment ids are no longer built here.
        output = self.attn_backend(q, k, v, cu_seqlens)

        # Head sharding is internal to the attention step. Downstream ops -- in
        # particular the temporal-merge gather -- cannot infer an output sharding
        # from a partitioned operand, so the layout is collapsed back here.
        output = apply_data_sharding(output, self.mesh, PartitionSpec())

        return output.reshape(sum_seq_len, D)


class KimiK25VisionMLP(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs | None = None,
    ):
        in_features = config.vt_hidden_size
        intermediate_size = config.vt_intermediate_size

        _rngs = rngs or nnx.Rngs(0)

        self.up_proj = nnx.Linear(
            in_features,
            intermediate_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

        self.down_proj = nnx.Linear(
            intermediate_size,
            in_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        up = jax.nn.gelu(self.up_proj(x), approximate=True)
        return self.down_proj(up)


class KimiK25VisionBlock(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh | None = None,
        norm_eps: float = 1e-6,
        rngs: nnx.Rngs | None = None,
        vision_tp: bool = True,
    ):
        assert mesh is not None, "KimiK25VisionBlock requires a sharding Mesh"

        self.attn = KimiK25VisionAttention(config, dtype, mesh, rngs, vision_tp)
        self.mlp = KimiK25VisionMLP(config, dtype, rngs)

        _rngs = rngs or nnx.Rngs(0)

        self.pre_norm = nnx.LayerNorm(config.vt_hidden_size, param_dtype=dtype, rngs=_rngs)

        self.proj = nnx.Linear(
            config.vt_hidden_size,
            config.vt_hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

        self.post_norm = nnx.LayerNorm(config.vt_hidden_size, param_dtype=dtype, rngs=_rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        cu_seqlens: jax.Array,
        rope_freqs_cis: jax.Array,
    ):
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=rope_freqs_cis,
        )
        hidden_states = self.proj(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VisionTowerEncoder(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh | None = None,
        rngs: nnx.Rngs | None = None,
        video_attn_type: str = "spatial_temporal",
        vision_tp: bool = True,
    ):
        self.config = config
        self.dtype = dtype

        assert (
            video_attn_type == "spatial_temporal"
        ), f'video_attn_type must be "spatial_temporal", got {video_attn_type}'

        self.blocks = nnx.List(
            [
                KimiK25VisionBlock(
                    config=config,
                    dtype=dtype,
                    rngs=rngs,
                    mesh=mesh,
                    vision_tp=vision_tp,
                )
                for _ in range(config.vt_num_hidden_layers)
            ]
        )

        _rngs = rngs or nnx.Rngs(0)

        self.final_layernorm = nnx.LayerNorm(config.vt_hidden_size, param_dtype=dtype, rngs=_rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        rope_freqs_cis: jax.Array,
        cu_seqlens: jax.Array,
    ) -> jax.Array:

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rope_freqs_cis=rope_freqs_cis,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def build_lane_merge_plan(
    grid_thws,
    merge_kernel_size,
    *,
    base_offset: int,
    output_cap: int,
    max_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lane-local sd2_tpool merge plan, padded to a fixed ``output_cap``.

    The merger groups ``merge_h * merge_w`` neighbouring patches and averages
    over the temporal axis, so a ``t``-frame item collapses to the same token
    count as a single frame.

    Three properties are required by ``lane_packing``:

    * Patch indices are offset by ``base_offset`` (``lane_index * capacity``)
      because the encoder sees one flat ``[num_lanes * capacity, hidden]``
      buffer rather than a dense per-batch concatenation.
    * Exactly ``output_cap`` rows are emitted so every lane has the same shape
      and the lanes can be concatenated. Padding rows gather index
      ``base_offset`` with weight 0, so they stay in bounds and contribute
      nothing; ``restore_encoder_output`` discards them.
    * ``max_t`` is supplied by the caller rather than derived, so that lanes
      holding different frame counts still agree on the temporal axis. Items
      with ``t < max_t`` duplicate their first frame into the padded slots,
      which carry weight 0, keeping each item's mean exact.
    """
    merge_h, merge_w = merge_kernel_size
    merge_unit = merge_h * merge_w

    indices = np.full((output_cap, merge_unit, max_t), base_offset, dtype=np.int32)
    weights = np.zeros((output_cap, max_t), dtype=np.float32)

    patch_offset = 0
    token_offset = 0
    for t, h, w in grid_thws:
        if h % merge_h or w % merge_w:
            raise ValueError(
                f"grid_thw {(t, h, w)} is not divisible by merge kernel {(merge_h, merge_w)}"
            )
        new_h, new_w = h // merge_h, w // merge_w
        rows = new_h * new_w
        if token_offset + rows > output_cap:
            raise ValueError(
                f"lane merge plan overflow: {token_offset + rows} output rows "
                f"exceed capacity {output_cap}."
            )

        item = np.arange(
            base_offset + patch_offset,
            base_offset + patch_offset + t * h * w,
            dtype=np.int32,
        )
        item = item.reshape(t, new_h, merge_h, new_w, merge_w)
        item = item.transpose(1, 3, 2, 4, 0).reshape(rows, merge_unit, t)

        indices[token_offset : token_offset + rows, :, :t] = item
        if t < max_t:
            # Duplicate the first frame into the padded slots. They carry weight
            # 0, so they never contribute; a valid index keeps the gather in
            # bounds.
            indices[token_offset : token_offset + rows, :, t:] = item[:, :, :1]
        weights[token_offset : token_offset + rows, :t] = 1.0 / t

        patch_offset += t * h * w
        token_offset += rows

    return indices, weights


class VisionTower(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs | None = None,
        mesh: Mesh | None = None,
        vision_tp: bool = True,
    ):
        self.config = config
        self.dtype = dtype
        self.vision_tp = vision_tp
        self.specs = VisionShardSpecs(mesh, vision_tp) if mesh is not None else None

        self.merge_kernel_size = config.merge_kernel_size

        self.rope_2d = Rope2DPosEmbRepeated(
            config.vt_hidden_size // config.vt_num_attention_heads, 512, 512
        )

        _rngs = rngs or nnx.Rngs(0)

        self.patch_embed = KimiK25VisionPatchEmbed(
            rngs,
            config.patch_size,
            config.in_channels,
            config.init_pos_emb_height,
            config.init_pos_emb_width,
            config.init_pos_emb_time,
            config.pos_emb_type,
            config.vt_hidden_size,
            dtype,
        )

        self.encoder = VisionTowerEncoder(config, dtype, mesh, rngs, vision_tp=vision_tp)

    def compute_hidden_states(
        self,
        pixel_values: jax.Array,
        abs_pos_embs: jax.Array,
        rope_freq_cis: jax.Array,
        cu_seqlens: jax.Array,
        merge_indices: jax.Array,
        merge_weights: jax.Array | None = None,
    ) -> jax.Array:
        """Run the ViT body and merge patches into merge_h * merge_w groups.

        merge_weights averages every item over its own frame count. It is
        required whenever a batch mixes items with different t (an image plus
        a video, or two videos with different frame counts). When omitted, all
        temporal slots are averaged uniformly, which matches the behavior of
        batches where every item has the same t.
        """

        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + abs_pos_embs

        hidden_states = self.encoder(hidden_states, rope_freq_cis, cu_seqlens)

        # Both operand and indices are sharded along the batch axis once the
        # inputs are lane-packed. Every lane's indices are confined to that
        # lane's own slice of the buffer (build_lane_merge_plan offsets them by
        # ``lane_index * capacity``), so the gather is shard-local -- but that
        # is a property of the index values, which JAX cannot see, so it
        # refuses to infer an output sharding. State it explicitly.
        merged_states = hidden_states.at[merge_indices].get(
            out_sharding=self.specs.sharding(self.specs.batch_axis) if self.specs else None
        )

        if merge_weights is None:
            merged_states = merged_states.mean(axis=2)
        else:
            weights = jnp.asarray(merge_weights, dtype=merged_states.dtype)
            merged_states = (merged_states * weights[:, None, :, None]).sum(axis=2)

        return merged_states


class Kimi_K25_MultiModalProjector(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
    ):
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = config.vt_hidden_size * merge_h * merge_w

        _rngs = rngs or nnx.Rngs(0)
        self.pre_norm = nnx.LayerNorm(
            config.vt_hidden_size,
            epsilon=config.projector_ln_eps,
            param_dtype=dtype,
            rngs=_rngs,
        )

        self.proj_0 = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

        self.proj_1 = nnx.Linear(
            self.hidden_size,
            config.text_hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(
        self,
        image_features: jax.Array,
    ) -> jax.Array:
        hidden_states = self.pre_norm(image_features).reshape(-1, self.hidden_size)
        hidden_states = self.proj_0(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states, approximate=False)
        return self.proj_1(hidden_states)


class Kimi_K25_VisionModel(nnx.Module):
    """Vision tower plus multimodal projector.

    Held as ``self.visual`` by ``KimiK25ForConditionalGeneration``; weights are
    loaded by that class, not here.
    """

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: Mesh | None = None,
        vision_tp: bool = True,
    ) -> None:

        self.vision_tower = VisionTower(config, dtype, rngs, mesh, vision_tp)
        self.mm_projector = Kimi_K25_MultiModalProjector(config, dtype, rngs)
        self.merge_kernel_size = config.merge_kernel_size
        self.merge_unit = config.merge_kernel_size[0] * config.merge_kernel_size[1]
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size

        logger.info("Kimi K2.5 Vision Model initialized with dtype %s", dtype)

    @staticmethod
    def vision_output_length(item: MultimodalDataItem, merge_unit: int) -> int:
        """Encoder output tokens for one item, as required by ``lane_packing``.

        The ``sd2_tpool`` merge averages over the temporal axis, so a ``t``-frame
        item collapses to the same token count as a single frame. The shared
        default (``t*h*w // merge_unit``) would over-count by a factor of ``t``.
        """
        _, height, width = get_grid_thw(item)
        return (height * width) // merge_unit

    def prepare_metadata(
        self,
        grid_thw: np.ndarray,
        capacity: int,
        *,
        sharding,
    ) -> dict:
        """Build the lane-padded aux arrays the tower needs, on the host.

        ``grid_thw`` arrives lane-major as ``[lanes, items, 3]`` from
        ``lane_packing``; all-zero rows are padding. Every array is allocated at
        ``capacity`` (or ``capacity // merge_unit`` on the output side) so the
        encoder sees one static shape per bucket and compiles once.
        """
        grid_thw = np.asarray(grid_thw, dtype=np.int32)
        if grid_thw.ndim == 2:
            grid_thw = grid_thw[None]
        if grid_thw.ndim != 3 or grid_thw.shape[-1] != 3:
            raise ValueError("grid_thw must have shape [items, 3] or [lanes, items, 3]")

        tower = self.vision_tower
        output_cap = capacity // self.merge_unit
        lane_grids = [
            [tuple(int(v) for v in grid) for grid in lane if np.any(grid)] for lane in grid_thw
        ]
        # A single temporal axis is shared by every lane once they are
        # concatenated, so max_t must be taken across the whole batch.
        max_t = max((t for lane in lane_grids for t, _, _ in lane), default=1)

        rope_lanes, pos_lanes, cu_lanes, index_lanes, weight_lanes = [], [], [], [], []
        for lane_index, grids in enumerate(lane_grids):
            base_offset = lane_index * capacity
            patches = sum(t * h * w for t, h, w in grids)
            if patches > capacity:
                raise ValueError(f"lane {lane_index} holds {patches} patches > capacity {capacity}")

            rope = np.zeros((2, capacity, tower.rope_2d.dim // 2), dtype=np.float32)
            pos = np.zeros((capacity, tower.config.vt_hidden_size), dtype=np.float32)
            if grids:
                rope[:, :patches] = np.asarray(tower.rope_2d._get_freqs_cis(grid_thws=grids))
                pos[:patches] = np.asarray(tower.patch_embed.pos_emb(np.asarray(grids)))

            # One segment per item, lane-local. The tail repeats the final
            # offset so the padding slots form zero-length segments and are
            # never attended to.
            cu = np.zeros(output_cap + 1, dtype=np.int32)
            offset = 0
            for item_index, (t, h, w) in enumerate(grids):
                offset += t * h * w
                cu[item_index + 1] = offset
            cu[len(grids) + 1 :] = offset

            indices, weights = build_lane_merge_plan(
                grids,
                self.merge_kernel_size,
                base_offset=base_offset,
                output_cap=output_cap,
                max_t=max_t,
            )

            rope_lanes.append(rope)
            pos_lanes.append(pos)
            cu_lanes.append(cu)
            index_lanes.append(indices)
            weight_lanes.append(weights)

        lane_major = {
            "abs_pos_embs": np.concatenate(pos_lanes, axis=0),
            "cu_seqlens": np.concatenate(cu_lanes),
            "merge_indices": np.concatenate(index_lanes, axis=0),
            "merge_weights": np.concatenate(weight_lanes, axis=0),
        }
        metadata = jax.device_put(lane_major, sharding)
        # rope_freq_cis is [2, tokens, dim/2] -- axis 0 selects cos/sin, so the
        # token axis is 1 and the lane-major spec has to be shifted right.
        rope_sharding = NamedSharding(sharding.mesh, PartitionSpec(None, *tuple(sharding.spec)))
        metadata["rope_freq_cis"] = jax.device_put(
            np.concatenate(rope_lanes, axis=1), rope_sharding
        )
        return metadata

    def __call__(self, patches: jax.Array, **metadata: jax.Array) -> jax.Array:
        """Encode lane-packed patches into projected multimodal tokens."""
        # lane_packing hands over one flat buffer; Kimi's patch embed consumes
        # [patches, channels, patch, patch] rather than a flattened patch_dim.
        specs = self.vision_tower.specs
        patches = patches.reshape(
            -1,
            self.in_channels,
            self.patch_size,
            self.patch_size,
            out_sharding=specs.sharding(specs.batch_axis),
        )
        hidden_states = self.vision_tower.compute_hidden_states(
            patches.astype(self.vision_tower.dtype),
            metadata["abs_pos_embs"],
            metadata["rope_freq_cis"],
            metadata["cu_seqlens"],
            metadata["merge_indices"],
            metadata["merge_weights"],
        )
        return self.mm_projector(hidden_states)


def create_kimi_vision_weight_mappings(
    num_hidden_layers: int,
    target_prefix: str = "",
) -> dict:
    """Map checkpoint keys for the Kimi vision tower and projector.

    target_prefix names where the tower is mounted in the JAX module tree. The
    in-model VLM nests it under ``visual.``; the checkpoint-side keys are the
    same wherever it is mounted.
    """
    mappings = {
        "vision_tower.patch_embed.pos_emb.weight": WeightMapping(
            target_path=f"{target_prefix}vision_tower.patch_embed.pos_emb.weight",
            sharding=(None,),
            transpose=False,
        ),
        "vision_tower.patch_embed.proj.weight": WeightMapping(
            target_path=f"{target_prefix}vision_tower.patch_embed.proj.kernel",
            sharding=(None, None, None, None),
            transpose_axes=(2, 3, 1, 0),
        ),
        "vision_tower.patch_embed.proj.bias": WeightMapping(
            target_path=f"{target_prefix}vision_tower.patch_embed.proj.bias",
            sharding=(None,),
            transpose=False,
        ),
        "vision_tower.encoder.final_layernorm.weight": WeightMapping(
            target_path=f"{target_prefix}vision_tower.encoder.final_layernorm.scale",
            sharding=(None,),
            transpose=False,
        ),
        "vision_tower.encoder.final_layernorm.bias": WeightMapping(
            target_path=f"{target_prefix}vision_tower.encoder.final_layernorm.bias",
            sharding=(None,),
            transpose=False,
        ),
        "mm_projector.pre_norm.bias": WeightMapping(
            target_path=f"{target_prefix}mm_projector.pre_norm.bias",
            sharding=(None,),
            transpose=False,
        ),
        "mm_projector.pre_norm.weight": WeightMapping(
            target_path=f"{target_prefix}mm_projector.pre_norm.scale",
            sharding=(None,),
            transpose=False,
        ),
        "mm_projector.proj.0.weight": WeightMapping(
            target_path=f"{target_prefix}mm_projector.proj_0.kernel",
            sharding=(None,),
            transpose=True,
        ),
        "mm_projector.proj.0.bias": WeightMapping(
            target_path=f"{target_prefix}mm_projector.proj_0.bias",
            sharding=(None,),
            transpose=False,
        ),
        "mm_projector.proj.2.weight": WeightMapping(
            target_path=f"{target_prefix}mm_projector.proj_1.kernel",
            sharding=(None,),
            transpose=True,
        ),
        "mm_projector.proj.2.bias": WeightMapping(
            target_path=f"{target_prefix}mm_projector.proj_1.bias",
            sharding=(None,),
            transpose=False,
        ),
    }

    for layer_idx in range(num_hidden_layers):
        mappings.update(create_kimi_vision_layer_mappings(layer_idx, target_prefix))

    return mappings


def create_kimi_vision_layer_mappings(layer_idx: int, target_prefix: str = "") -> dict:
    source = f"vision_tower.encoder.blocks.{layer_idx}"
    target = f"{target_prefix}vision_tower.encoder.blocks.{layer_idx}"

    return {
        f"{source}.wqkv.weight": WeightMapping(
            target_path=f"{target}.attn.qkv_proj.kernel",
            sharding=(None,),
            transpose=True,
        ),
        f"{source}.wqkv.bias": WeightMapping(
            target_path=f"{target}.attn.qkv_proj.bias",
            sharding=(None,),
            transpose=False,
        ),
        f"{source}.wo.weight": WeightMapping(
            target_path=f"{target}.proj.kernel",
            sharding=(None,),
            transpose=True,
        ),
        f"{source}.wo.bias": WeightMapping(
            target_path=f"{target}.proj.bias",
            sharding=(None,),
            transpose=False,
        ),
        f"{source}.mlp.fc0.weight": WeightMapping(
            target_path=f"{target}.mlp.up_proj.kernel",
            sharding=(None,),
            transpose=True,
        ),
        f"{source}.mlp.fc0.bias": WeightMapping(
            target_path=f"{target}.mlp.up_proj.bias",
            sharding=(None,),
            transpose=False,
        ),
        f"{source}.mlp.fc1.weight": WeightMapping(
            target_path=f"{target}.mlp.down_proj.kernel",
            sharding=(None,),
            transpose=True,
        ),
        f"{source}.mlp.fc1.bias": WeightMapping(
            target_path=f"{target}.mlp.down_proj.bias",
            sharding=(None,),
            transpose=False,
        ),
        f"{source}.norm0.weight": WeightMapping(
            target_path=f"{target}.pre_norm.scale",
            sharding=(None,),
            transpose=False,
        ),
        f"{source}.norm0.bias": WeightMapping(
            target_path=f"{target}.pre_norm.bias",
            sharding=(None,),
            transpose=False,
        ),
        f"{source}.norm1.weight": WeightMapping(
            target_path=f"{target}.post_norm.scale",
            sharding=(None,),
            transpose=False,
        ),
        f"{source}.norm1.bias": WeightMapping(
            target_path=f"{target}.post_norm.bias",
            sharding=(None,),
            transpose=False,
        ),
    }
