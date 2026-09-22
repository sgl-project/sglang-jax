import logging
import math

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import KimiK25ModelVitConfig
from sgl_jax.srt.multimodal.kernels.flash_attention import (
    BlockSizes,
    SegmentIds,
    flash_attention,
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


def align_to(x, a):
    return pl.cdiv(x, a) * a


# The Pallas kernel's smallest addressable tile along the KV axis.
_KV_MIN_BLOCK = 128

# Cost of one KV token inside the kernel's VMEM window: the 72-wide heads pad up
# to a full 128-lane register, the window is f32, and Pallas double-buffers it.
_KV_WINDOW_BYTES_PER_TOKEN = 128 * 4 * 2

# Q, the output tile and the softmax scratch all have to share VMEM with the KV
# window, so only a fraction of the 64 MB budget is spent here.
_KV_WINDOW_BUDGET_BYTES = 8 * 1024 * 1024


def _vmem_safe_kv_block(kv_seq_len: int) -> int:
    """Largest KV block the flash-attention kernel can hold in VMEM.

    The kernel walks the KV axis in kv_seq_len // block_k_major steps, so the
    block has to divide the sequence exactly, and it must also be a multiple of
    the kernel's 128-lane minimum. kv_seq_len is returned unchanged when the
    whole sequence already fits, which keeps short items -- images and brief
    clips -- on the kernel's single-step fast path.
    """
    max_block = _KV_WINDOW_BUDGET_BYTES // _KV_WINDOW_BYTES_PER_TOKEN
    if kv_seq_len <= max_block:
        return kv_seq_len

    # Callers align the sequence to 256, so 128 always divides it and the loop
    # is guaranteed to terminate on a valid block.
    for block in range(max_block - max_block % _KV_MIN_BLOCK, 0, -_KV_MIN_BLOCK):
        if kv_seq_len % block == 0:
            return block
    return _KV_MIN_BLOCK


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
    ):
        assert mesh is not None, "KimiK25VisionAttention requires a sharding Mesh"
        self.mesh = mesh
        self.hidden_size = config.vt_hidden_size
        self.num_heads = config.vt_num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

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

        indices = jnp.arange(sum_seq_len)
        item_ids = jnp.sum(indices[:, None] >= cu_seqlens[1:][None, :], axis=-1) + 1

        output = self._flash_attention(q, k, v, item_ids)

        return output.reshape(sum_seq_len, D)

    def _flash_attention(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        item_ids: jax.Array,
    ) -> jax.Array:
        seq_len = q.shape[0]
        align_seq_len = align_to(seq_len, 256)
        pad_len = align_seq_len - seq_len

        if pad_len:
            padding = ((0, pad_len), (0, 0), (0, 0))
            q = jnp.pad(q, padding)
            k = jnp.pad(k, padding)
            v = jnp.pad(v, padding)

        seg = jnp.pad(item_ids, (0, pad_len))
        segment_ids = SegmentIds(q=seg[None, :], kv=seg[None, :])

        pad_q = jnp.transpose(q, (1, 0, 2))[None, ...]
        pad_k = jnp.transpose(k, (1, 0, 2))[None, ...]
        pad_v = jnp.transpose(v, (1, 0, 2))[None, ...]

        # For any kv_seq_len <= 92800 the kernel forces
        # block_k_major = block_k = kv_seq_len to reach its single-step fast
        # path, which asks for a VMEM window of kv_seq_len * 128 lanes * 4 B * 2 buffers.
        block_sizes = None
        kv_block = _vmem_safe_kv_block(align_seq_len)
        if kv_block < align_seq_len:
            block_sizes = BlockSizes(
                block_q=min(256, align_seq_len),
                block_k_major=kv_block,
                block_k=kv_block,
                block_b=1,
            )

        def local_flash_attention(q, k, v, segment_ids):
            return flash_attention(
                q,
                k,
                v,
                segment_ids=segment_ids,
                causal=False,
                sm_scale=self.scale,
                block_sizes=block_sizes,
            )

        in_specs = (
            jax.sharding.PartitionSpec(None, None, None, None),
            jax.sharding.PartitionSpec(None, None, None, None),
            jax.sharding.PartitionSpec(None, None, None, None),
            jax.sharding.PartitionSpec(),
        )

        output = jax.shard_map(
            local_flash_attention,
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=jax.sharding.PartitionSpec(None, None, None, None),
            check_vma=False,
        )(pad_q, pad_k, pad_v, segment_ids)

        return jnp.transpose(output[0], (1, 0, 2))[:seq_len, :, :]


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
    ):
        assert mesh is not None, "KimiK25VisionBlock requires a sharding Mesh"

        self.attn = KimiK25VisionAttention(config, dtype, mesh, rngs)
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


def build_temporal_merge_plan(
    grid_thws,
    merge_kernel_size,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan the sd2_tpool merge for a batch of image and/or video items.

    The merger groups merge_h * merge_w neighboring patches and averages over
    the temporal axis. Items in a batch can have different frame counts (an image
    has t == 1, a video has t > 1), so the temporal axis is padded to the
    batch-wide maximum with duplicated, zero-weighted slots. That keeps a single
    static-shaped gather for the whole batch while leaving each item's mean exact,
    including when t does not divide max_t.

    Args:
        grid_thws: Sequence of (t, h, w) patch grids, one per item, matching
            the order of the flattened patch sequence.
        merge_kernel_size: (merge_h, merge_w) spatial merge kernel.

    Returns:
        merge_indices shaped [num_output_tokens, merge_h * merge_w, max_t]
        with indices into the flat patch sequence, and merge_weights shaped
        [num_output_tokens, max_t] holding 1 / t for real frames and 0 for
        padded slots.
    """
    if len(grid_thws) == 0:
        raise ValueError("grid_thws must contain at least one (t, h, w) grid")

    merge_h, merge_w = merge_kernel_size
    grids = [(int(t), int(h), int(w)) for t, h, w in grid_thws]
    max_t = max(t for t, _, _ in grids)

    all_merge_indices = []
    all_merge_weights = []
    token_offset = 0

    for t, h, w in grids:
        if h % merge_h or w % merge_w:
            raise ValueError(
                f"grid_thw {(t, h, w)} is not divisible by merge kernel " f"{(merge_h, merge_w)}"
            )

        new_h, new_w = h // merge_h, w // merge_w

        indices = np.arange(token_offset, token_offset + t * h * w, dtype=np.int32)
        indices = indices.reshape(t, new_h, merge_h, new_w, merge_w)
        indices = indices.transpose(1, 3, 2, 4, 0)
        indices = indices.reshape(new_h * new_w, merge_h * merge_w, t)

        if t < max_t:
            # Duplicate the first frame into the padded slots. They carry weight 0,
            # so they never contribute; duplicating a valid index just keeps the
            # gather in bounds.
            padding = np.repeat(indices[:, :, :1], max_t - t, axis=2)
            indices = np.concatenate([indices, padding], axis=2)

        weights = np.zeros((new_h * new_w, max_t), dtype=np.float32)
        weights[:, :t] = 1.0 / t

        all_merge_indices.append(indices)
        all_merge_weights.append(weights)
        token_offset += t * h * w

    return (
        np.concatenate(all_merge_indices, axis=0),
        np.concatenate(all_merge_weights, axis=0),
    )


class VisionTower(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs | None = None,
        mesh: Mesh | None = None,
    ):
        self.config = config
        self.dtype = dtype

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

        self.encoder = VisionTowerEncoder(config, dtype, mesh, rngs)

    def compute_aux_arrays(
        self,
        grid_thws,
    ):
        """Build the host-side arrays the ViT body needs for one batch.

        Args:
            grid_thws: Sequence of (t, h, w) patch grids, one per item.
                t == 1 is a still image; t > 1 is a video.

        Returns:
            (rope_freqs_cis, cu_seqlens, abs_pos_embs, merge_indices,
            merge_weights). The last two are consumed by compute_hidden_states
            to pool merge_h * merge_w patches spatially and t frames temporally.
        """
        merge_indices, merge_weights = build_temporal_merge_plan(grid_thws, self.merge_kernel_size)

        rope_freqs_cis = self.rope_2d._get_freqs_cis(grid_thws=grid_thws)

        grid_thws = np.array(grid_thws)
        lengths = jnp.concatenate(
            (
                jnp.zeros(1, dtype=jnp.int32),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )

        cu_seqlens = lengths.cumsum(axis=0, dtype=jnp.int32)

        abs_pos_embs = self.patch_embed.pos_emb(grid_thws)

        return rope_freqs_cis, cu_seqlens, abs_pos_embs, merge_indices, merge_weights

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

        merged_states = hidden_states[merge_indices]

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
    ) -> None:

        self.vision_tower = VisionTower(config, dtype, rngs, mesh)
        self.mm_projector = Kimi_K25_MultiModalProjector(config, dtype, rngs)

        logger.info("Kimi K2.5 Vision Model initialized with dtype %s", dtype)


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
