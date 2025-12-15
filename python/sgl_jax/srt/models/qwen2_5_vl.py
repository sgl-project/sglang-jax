import math
from functools import partial
from typing import List, Callable, NamedTuple, Optional, Any
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers import modeling_flax_utils
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental import shard_map
from flax import nnx

from sgl_jax.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sgl_jax.srt.models.qwen2 import Qwen2Model
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
from sgl_jax.utils import logger
from sgl_jax.srt.kernels.flash_attention import flash_attention


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

  SegmentIds are used to generate segment mask, which prevents attention between
  different segments in the input sequence. Each array is a list of ids
  (integers).
  Only the token with the same id can attend to each other.

  Attributes:
    q: segment ids along the Q sequence.
    kv: segment ids along the KV sequence.
  """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


def apply_rotary_pos_emb_vision(x: jax.Array,
                                rotary_pos_emb: jax.Array) -> jax.Array:
    # x: [B, T, N, H]
    # rotary_pos_emb: [T, H//2]
    _, _, _, H = x.shape
    half_dim = H // 2

    # [B, T, N, H//2]
    x_real = x[..., :half_dim]
    x_imag = x[..., half_dim:]

    # [T, H//2]
    cos_emb = jnp.cos(rotary_pos_emb)
    sin_emb = jnp.sin(rotary_pos_emb)

    # [1, T, 1, H//2]
    cos_emb = cos_emb[None, :, None, :]
    sin_emb = sin_emb[None, :, None, :]

    # [B, T, N, H//2]
    x_rotated_real = x_real * cos_emb - x_imag * sin_emb
    x_rotated_imag = x_real * sin_emb + x_imag * cos_emb

    # [B, T, N, H]
    x_rotated = jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)

    return x_rotated


def generate_window_segment_ids(cu_seqlens: jax.Array, seq_len: int,
                                padded_seq_len: int) -> SegmentIds:
    """Generates segment IDs for windowed attention

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths for each window.
            e.g., [0, len_win0, len_win0+len_win1, ...]

    Returns:
        A SegmentIds object for flash_attention.
    """
    indices = jnp.arange(seq_len, dtype=jnp.int32)
    segment_ids = jnp.searchsorted(cu_seqlens[1:], indices, side='right') + 1
    padding_segment_ids = jnp.zeros(padded_seq_len - seq_len, dtype=jnp.int32)
    segment_ids = jnp.concatenate([segment_ids, padding_segment_ids])
    segment_ids = segment_ids.reshape(1, -1)

    return SegmentIds(q=segment_ids, kv=segment_ids)


def get_padded_num_heads(num_heads: int, sharding_size: int) -> int:
    if num_heads >= sharding_size:
        assert num_heads % sharding_size == 0
    else:
        assert sharding_size % num_heads == 0
        num_heads = sharding_size
    return num_heads


class Qwen2_5_VisionMLP(nnx.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None
    ):
        self.gate_proj = LinearBase(
            hidden_size,
            intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.up_proj = LinearBase(
            hidden_size,
            intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            intermediate_size,
            hidden_size,
            kernel_axes=("tensor", None),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.act_fn = jax.nn.swish

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x)[0])
        up = self.up_proj(x)[0]
        fuse = gate * up
        result = self.down_proj(fuse)[0]
        return result


class Attention(nnx.Module):

    def __init__(
            self,
            mesh: Mesh = None,
            scale: Optional[float] = None,
    ):
        self.mesh = mesh
        self.scale = scale

    def __call__(self, q, k, v, mask=None):
        attn_logits = jnp.einsum("bnth,bnsh->bnts", q, k) * self.scale

        # Apply appropriate masking
        if mask is not None:
            mask_value = jnp.finfo(attn_logits.dtype).min
            attn_logits = jnp.where(mask, attn_logits, mask_value)

        # Softmax
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        attn_output = jnp.matmul(attn_weights, v)
        return attn_output

def sharded_flash_attention(
    mesh: Mesh,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    vmem_limit_bytes: int | None = None,
) -> Callable[..., Any]:
    in_specs = (
        P("data", "tensor", None, None),  # q
        P("data", "tensor", None, None),  # k
        P("data", "tensor", None, None),  # v
        P(),  # segment_ids
    )
    out_specs = P("data", "tensor", None, None)

    def _flash_attention(q, k, v, segment_ids):
        return flash_attention(q,
                               k,
                               v,
                               segment_ids=segment_ids,
                               sm_scale=sm_scale,
                               causal=causal,
                               vmem_limit_bytes=vmem_limit_bytes)

    return jax.jit(
        shard_map.shard_map(_flash_attention,
                            mesh=mesh,
                            in_specs=in_specs,
                            out_specs=out_specs,
                            check_rep=False))

class Qwen2_5_VisionAttention(nnx.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            rope_theta: float = 1000000,
            rope_scaling: dict[str, Any] | None = None,
            head_dim: int | None = None,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = self.num_heads
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        sharding_size = mesh.shape["tensor"]
        self.num_heads = get_padded_num_heads(self.num_heads,
                                              sharding_size)
        self.num_kv_heads = get_padded_num_heads(self.num_kv_heads,
                                                 sharding_size)

        # TODO: Consider padding in future
        self.head_dim = head_dim or hidden_size // self.num_heads

        self.mesh = mesh

        self.qkv_proj = LinearBase(
            hidden_size,
            3 * hidden_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.o_proj = LinearBase(
            hidden_size,
            hidden_size,
            kernel_axes=("tensor", None),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.flash_attention = sharded_flash_attention(
            mesh=mesh,
            causal=False,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            vmem_limit_bytes=128 * 1024 * 1024,
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_window_seqlens: Optional[jax.Array] = None,
        use_fullattn: bool = True,
    ) -> jax.Array:
        T, B, D = x.shape
        assert B == 1, "Vision attention currently only supports batch size 1"
        # [T, B, D] -> [T, B, 3 * D]
        qkv, _ = self.qkv_proj(x)

        # Split into Q, K, V.
        # NOTE: simplified from vLLM's split_qkv,
        # may need to revisit for tp>1
        # [T, B, 3 * D] -> 3 *[T, B, D]
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # [T, B, N, H]
        q = q.reshape(T, B, self.num_heads, self.head_dim)
        k = k.reshape(T, B, self.num_heads, self.head_dim)
        v = v.reshape(T, B, self.num_heads, self.head_dim)

        # [T, B, N, H] -> [B, T, N, H]
        q = jnp.transpose(q, (1, 0, 2, 3))
        k = jnp.transpose(k, (1, 0, 2, 3))
        v = jnp.transpose(v, (1, 0, 2, 3))

        # rotary_pos_emb shape: (T, H)
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # NOTE: an extra transpose because we need to
        # align the correctness with vLLM's design.
        # Might be able to remove one once implemented.
        # [B, T, N, H] -> [B, N, T, H]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Pad the sequence length to be a multiple of 128 for flash_attention
        block_k_major = 128
        T_attn = q.shape[2]
        padded_T = (T_attn + block_k_major -
                    1) // block_k_major * block_k_major
        pad_width = ((0, 0), (0, 0), (0, padded_T - T_attn), (0, 0))

        q = jnp.pad(q, pad_width, 'constant')
        k = jnp.pad(k, pad_width, 'constant')
        v = jnp.pad(v, pad_width, 'constant')

        segment_ids = generate_window_segment_ids(cu_window_seqlens, T_attn,
                                                  padded_T)

        # TODO (jacobplatin): add support for quantized KV cache?
        output = self.flash_attention(q, k, v, segment_ids)

        # Unpad the output
        output = output[:, :, :T_attn, :]

        # [B, N, T, H] -> [T, B, N, H]
        output = jnp.transpose(output, (2, 0, 1, 3))

        output = output.reshape(T, B, D)

        output = self.o_proj(output)

        return output[0]


class Qwen2_5_VisionBlock(nnx.Module):

    def __init__(
            self,
            config: Qwen2_5_VLConfig,
            norm_eps: float = 1e-6,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None,
    ):
        dim = config.hidden_size
        norm_layer = partial(nnx.RMSNorm,
                             epsilon=norm_eps,
                             scale_init=nnx.with_partitioning(
                                 nnx.initializers.uniform(), (None, )))

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.attn = Qwen2_5_VisionAttention(hidden_size=config.vision_config.hidden_size,
                                            num_heads=config.vision_config.num_heads,
                                            rope_theta=config.rope_theta,
                                            rope_scaling=config.rope_scaling,
                                            head_dim=config.vision_config.hidden_size // config.vision_config.num_heads,
                                            dtype=dtype,
                                            rngs=rngs,
                                            mesh=mesh)
        self.mlp = Qwen2_5_VisionMLP(hidden_size=config.vision_config.hidden_size,
                                     intermediate_size=config.vision_config.intermediate_size,
                                     dtype=dtype,
                                     rngs=rngs,
                                     mesh=mesh)

    def __call__(self,
                 x: jax.Array,
                 rotary_pos_emb: jax.Array,
                 cu_window_seqlens: Optional[jax.Array] = None,
                 use_fullattn: bool = True) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb, cu_window_seqlens,
                          use_fullattn)
        x = x + self.mlp(self.norm2(x))

        return x


class Qwen2_5_VisionPatchEmbed(nnx.Module):

    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nnx.Conv(in_features=in_channels,
                             out_features=hidden_size,
                             kernel_size=kernel_size,
                             strides=kernel_size,
                             use_bias=False,
                            #  padding="VALID",
                             param_dtype=dtype,
                             kernel_init=nnx.with_partitioning(
                                 nnx.initializers.uniform(),
                                 (None, None, None, None, "tensor")
                             ),
                             rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is (L, C * T * H * W)
        L, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size *
                    self.patch_size)
        # Reshape to (L, C, T, H, W) first
        x = x.reshape(L, C, self.temporal_patch_size, self.patch_size,
                      self.patch_size)
        # Transpose to (L, T, H, W, C) for Conv3D with channels_last format
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        # After conv, shape is (L, T_out, H_out, W_out, C_out)
        # With stride=kernel_size, T_out=H_out=W_out=1.
        # So shape is (L, 1, 1, 1, hidden_size)
        x = x.reshape(L, self.hidden_size)
        return x


class Qwen2_5_VisionPatchMerger(nnx.Module):

    def __init__(
            self,
            d_model: int,
            context_dim: int,
            norm_layer: Callable,
            spatial_merge_size: int,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = norm_layer(context_dim,
                               dtype=dtype,
                               rngs=rngs,
                               scale_init=nnx.with_partitioning(
                                   nnx.initializers.uniform(),
                                   (None, )
                               ))
        self.mlp_fc1 = LinearBase(
            self.hidden_size,
            self.hidden_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = LinearBase(
            self.hidden_size,
            d_model,
            kernel_axes=("tensor", None),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.ln_q(x)
        x = x.reshape(-1, self.hidden_size)
        x = self.mlp_fc1(x)[0]
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)[0]
        return x


class Qwen2_5_VisionRotaryEmbedding(nnx.Module):

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta**(
            jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)


class Qwen2_5_VisionTransformer(nnx.Module):

    def __init__(self,
                 config: Qwen2_5_VLConfig,
                 norm_eps: float = 1e-6,
                 dtype: jnp.dtype = jnp.bfloat16,
                 rngs: nnx.Rngs = None,
                 mesh: Mesh = None):
        self.dtype = dtype
        
        # args for get_window_index_thw
        self.window_size = config.vision_config.window_size
        self.patch_size = config.vision_config.patch_size
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.fullatt_block_indexes = config.vision_config.fullatt_block_indexes
        self.spatial_merge_unit = config.vision_config.spatial_merge_size**2

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.vision_config.patch_size,
            temporal_patch_size=config.vision_config.temporal_patch_size,
            in_channels=config.vision_config.in_channels,
            hidden_size=config.vision_config.hidden_size,
            dtype=dtype,
            rngs=rngs)

        head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nnx.data([
            Qwen2_5_VisionBlock(
                config=config,
                norm_eps=norm_eps,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            ) for _ in range(config.vision_config.depth)
        ])
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=config.vision_config.out_hidden_size,
            context_dim=config.vision_config.hidden_size,
            norm_layer=partial(nnx.RMSNorm, epsilon=norm_eps),
            spatial_merge_size=config.vision_config.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

    def rotary_pos_emb_thw(self, t, h, w):
        # hpos_ids: [h, w], wpos_ids: [h, w]
        hpos_ids, wpos_ids = jnp.indices((h, w))
        # hpos_ids: [h, w] -> [(h / spatial_merge_size) *
        #                      (w / spatial_merge_size) *
        #                      spatial_merge_size       *
        #                      spatial_merge_size]
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).transpose(0, 2, 1, 3).flatten()
        # wpos_ids: [h, w] -> [(h / spatial_merge_size) *
        #                      (w / spatial_merge_size) *
        #                      spatial_merge_size       *
        #                      spatial_merge_size]
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).transpose(0, 2, 1, 3).flatten()
        # pos_ids: [(h / spatial_merge_size) *
        #           (w / spatial_merge_size) *
        #           spatial_merge_size       *
        #           spatial_merge_size, 2]
        pos_ids = jnp.stack([hpos_ids, wpos_ids], axis=-1)
        # pos_ids: [t * (h / spatial_merge_size) *
        #           (w / spatial_merge_size) *
        #           spatial_merge_size       *
        #           spatial_merge_size, 2]
        pos_ids = jnp.tile(pos_ids, (t, 1))

        max_size = max(h, w)
        # rotary_pos_emb_full: [max_size, head_dim // 4]
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)
        # rotary_pos_emb: [t * h * w, head_dim // 2]
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(
            pos_ids.shape[0], -1)
        # rotary_pos_emb: [t * h * w / (spatial_merge_size*spatial_merge_size),
        #                  spatial_merge_size*spatial_merge_size,
        #                  head_dim // 2]
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit, -1)

        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size

        index = jnp.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w)

        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = jnp.pad(index, ((0, 0), (0, pad_h), (0, pad_w)),
                               constant_values=-100)
        index_padded = index_padded.reshape(grid_t, num_windows_h,
                                            vit_merger_window_size,
                                            num_windows_w,
                                            vit_merger_window_size)
        index_padded = jnp.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
            vit_merger_window_size)
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        # The number of valid indices is static because grid_t, grid_h, grid_w
        # are static.
        num_valid_indices = grid_t * llm_grid_h * llm_grid_w
        valid_indices = jnp.nonzero(index_padded != -100,
                                    size=num_valid_indices)[0]
        index_new = index_padded[valid_indices]
        cu_seqlens_tmp = jnp.cumsum(seqlens) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.astype(jnp.int32)

        # NOTE (wenlong): Pytorch code uses this to reduce replication,
        # but I don't think there is a need here, plus it would cause problem in JIT
        # Please refer here if there is a problem down-stream
        # cu_seqlens_tmp = jnp.unique(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(
            t, h, w)

        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)

        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.reshape(
            -1, rotary_pos_emb_thw.shape[-1])
        cu_seqlens_thw = jnp.full(t, h * w, dtype=jnp.int32)

        return (rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw,
                cu_seqlens_thw)

    def __call__(self, x: jax.Array, grid_thw: tuple[tuple[int, int,
                                                           int]]) -> jax.Array:
        # x: pixel_values: jax.Array
        # """Shape:
        # `(num_patches, num_channels * patch_size * patch_size)`
        # """

        # grid_thw: image_grid_thw: jax.Array
        # """Shape: `(num_images, 3)`
        # This should be in `(grid_t, grid_h, grid_w)` format.
        # """
        hidden_states = self.patch_embed(x)

        # num of patches
        seq_len = x.shape[0]
        # num of images/videoes
        num_grids = len(grid_thw)

        rotary_pos_emb = []
        window_index = []
        cu_window_seqlens = [jnp.array([0], dtype=jnp.int32)]
        cu_seqlens = []

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:

            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += (t * llm_h * llm_w)

            cu_seqlens_window_thw += cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = jnp.concatenate(rotary_pos_emb, axis=0)
        window_index = jnp.concatenate(window_index, axis=0)
        cu_window_seqlens = jnp.concatenate(cu_window_seqlens, axis=0)

        cu_seqlens = jnp.concatenate(cu_seqlens, axis=0)
        cu_seqlens = jnp.cumsum(cu_seqlens, axis=0, dtype=jnp.int32)
        cu_seqlens = jnp.pad(cu_seqlens, ((1, 0), ),
                             mode='constant',
                             constant_values=0)

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = jnp.expand_dims(hidden_states, axis=1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                hidden_states = blk(hidden_states,
                                    rotary_pos_emb=rotary_pos_emb,
                                    cu_window_seqlens=cu_seqlens,
                                    use_fullattn=True)
            else:
                hidden_states = blk(hidden_states,
                                    rotary_pos_emb=rotary_pos_emb,
                                    cu_window_seqlens=cu_window_seqlens,
                                    use_fullattn=False)

        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = jnp.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states


class Qwen2_5_VLForConditionalGeneration(nnx.Module):

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rng_key: jax.Array | None = None,
        mesh: Mesh = None,
    ) -> None:

        self.config = config
        self.rng = nnx.Rngs(rng_key)
        self.dtype = dtype
        self.mesh = mesh

        self.visual = Qwen2_5_VisionTransformer(
            config=config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            dtype=dtype,
            rngs=self.rng,
            mesh=mesh,
        )

        self.model = Qwen2Model(
            config=config,
            dtype=dtype,
            rngs=self.rng,
            mesh=mesh,
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            rngs=self.rng,
        )

        self.is_mrope_enabled = "mrope_section" in config.rope_scaling

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> jax.Array:
        # in qwen-vl, last dim is the same
        pixel_values = jnp.concatenate([item.feature for item in items], axis=0).astype(
            self.visual.dtype
        )
        # Collect image_grid_thw as Python tuples directly from items
        grid_thw_list = []
        for item in items:
            if hasattr(item, "image_grid_thw") and item.image_grid_thw is not None:
                # Convert to Python tuple immediately
                grid_thw = item.image_grid_thw
                if isinstance(grid_thw, jax.Array) or hasattr(grid_thw, '__array__'):
                    # Convert JAX/NumPy array to tuple
                    import numpy as np
                    grid_np = np.asarray(grid_thw)
                    grid_tuple = tuple(tuple(int(x) for x in row) for row in grid_np)
                else:
                    # Already a tuple or list
                    grid_tuple = tuple(tuple(row) for row in grid_thw)
                grid_thw_list.append(grid_tuple)
            else:
                # Provide a default value if image_grid_thw is not available
                grid_thw_list.append(((1, 1, 1),))
        
        # Flatten the list of tuples into a single tuple
        grid_thw_tuple = tuple(item for sublist in grid_thw_list for item in sublist)
        
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw_tuple)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> jax.Array:
        # in qwen-vl, last dim is the same
        pixel_values = jnp.concatenate([item.feature for item in items], axis=0).astype(
            self.visual.dtype
        )
        # Collect video_grid_thw as Python tuples directly from items
        grid_thw_list = []
        for item in items:
            if hasattr(item, 'video_grid_thw') and item.video_grid_thw is not None:
                # Convert to Python tuple immediately
                grid_thw = item.video_grid_thw
                if hasattr(grid_thw, 'detach'):  # PyTorch tensor
                    grid_thw = grid_thw.detach().cpu().numpy()
                
                if isinstance(grid_thw, jax.Array) or hasattr(grid_thw, '__array__'):
                    # Convert JAX/NumPy array to tuple
                    import numpy as np
                    grid_np = np.asarray(grid_thw)
                    grid_tuple = tuple(tuple(int(x) for x in row) for row in grid_np)
                else:
                    # Already a tuple or list
                    grid_tuple = tuple(tuple(row) for row in grid_thw)
                grid_thw_list.append(grid_tuple)
            else:
                raise ValueError(f"MultimodalDataItem missing video_grid_thw attribute")
        
        # Flatten the list of tuples into a single tuple
        grid_thw_tuple = tuple(item for sublist in grid_thw_list for item in sublist)
        
        video_embeds = self.visual(pixel_values, grid_thw=grid_thw_tuple)
        return video_embeds

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        """Run forward pass for Qwen2_5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.shape[0] == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.shape}"
                )

        hidden_states, layers_kv_fused, layers_callback_flag = general_mm_embed_routine(
            forward_batch=forward_batch,
            language_model=self.model,
            token_to_kv_pool=token_to_kv_pool,
            multimodal_model=self,
        )
        
        return self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata), layers_kv_fused, layers_callback_flag

    def load_weights(self, model_config, rng_key: jax.Array):
        """Load weights for Qwen2.5-VL model.
        
        Args:
            model_config: Model configuration containing model path and settings
            rng_key: JAX random key for initialization
        """
        self.rng = nnx.Rngs(rng_key)
        
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        
        weight_mappings = self._create_qwen2_5_vl_weight_mappings()
        
        self._load_weights_with_custom_conv3d(loader, weight_mappings)
        logger.info("Qwen2.5-VL weights loaded successfully!")
    
    def _load_weights_with_custom_conv3d(self, loader: WeightLoader, weight_mappings: dict):
        """Custom weight loading that handles Conv3D weight permutation."""
        import copy
        
        # Create a copy of mappings for modification
        modified_mappings = copy.deepcopy(weight_mappings)
        
        # Remove the Conv3D weight from normal loading
        conv3d_mapping = modified_mappings.pop("visual.patch_embed.proj.weight", None)
        
        # Load all other weights normally
        loader.load_weights_from_safetensors(modified_mappings)
        
        # Handle Conv3D weight separately if it exists
        if conv3d_mapping:
            self._load_conv3d_weight(loader, conv3d_mapping)
    
    def _load_conv3d_weight(self, loader: WeightLoader, mapping: WeightMapping):
        """Load and properly permute Conv3D weight."""
        # Get the model parameters
        params = nnx.state(self)
        
        # Find the Conv3D weight in the safetensors files
        for hf_key, hf_weight in loader._iterate_weights():
            if hf_key == "visual.patch_embed.proj.weight":
                # HuggingFace format: (out_channels, in_channels, T, H, W)
                # Flax format: (T, H, W, in_channels, out_channels)
                # Permute from (0,1,2,3,4) to (2,3,4,1,0)
                permuted_weight = jnp.transpose(hf_weight, (2, 3, 4, 1, 0))
                
                # Apply sharding
                sharded_weight = loader._shard_weight(permuted_weight, mapping.sharding)
                
                # Get the target parameter and assign
                target_param = loader._get_param(params, mapping.target_path)
                target_param.value = sharded_weight.astype(target_param.value.dtype)
                
                logger.debug(
                    "Loaded Conv3D weight: %s -> %s, original shape: %s, permuted shape: %s",
                    hf_key,
                    mapping.target_path,
                    hf_weight.shape,
                    permuted_weight.shape,
                )
                break
        
        # Update the model with modified parameters
        nnx.update(self, params)
    
    def _create_qwen2_5_vl_weight_mappings(self) -> dict:
        """Create weight mappings for Qwen2.5-VL model.
        
        Returns:
            Dictionary mapping HuggingFace weight names to model parameter paths
        """        
        mappings = {}
        
        # Vision transformer weights
        mappings.update(self._create_vision_transformer_mappings())
        
        # Language model embeddings
        mappings["model.embed_tokens.weight"] = WeightMapping(
            target_path="model.embed_tokens.embedding",
            sharding=("tensor", None),
            transpose=False,
        )
        
        # Language model norm
        mappings["model.norm.weight"] = WeightMapping(
            target_path="model.norm.scale",
            sharding=(None,),
            transpose=False,
        )
        
        # LM head
        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )
        
        # Language model layers
        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)
        
        return mappings
    
    def _create_vision_transformer_mappings(self) -> dict:
        """Create weight mappings for the vision transformer.
        
        Returns:
            Dictionary mapping vision transformer weight names to model paths
        """        
        mappings = {}
        
        # Vision embeddings
        mappings["visual.patch_embed.proj.weight"] = WeightMapping(
            target_path="visual.patch_embed.proj.kernel",
            sharding=(None, None, None, None, "tensor"),
            transpose=False,
        )
        
        # Note: In the model definition, use_bias=False is set for the proj Conv layer
        # So we don't need to map the bias parameter
        
        # Add merger mappings
        mappings["visual.merger.ln_q.weight"] = WeightMapping(
            target_path="visual.merger.ln_q.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings["visual.merger.mlp.0.weight"] = WeightMapping(
            target_path="visual.merger.mlp_fc1.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings["visual.merger.mlp.0.bias"] = WeightMapping(
            target_path="visual.merger.mlp_fc1.bias",
            sharding=("tensor",),
            transpose=False,
        )
        mappings["visual.merger.mlp.2.weight"] = WeightMapping(
            target_path="visual.merger.mlp_fc2.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings["visual.merger.mlp.2.bias"] = WeightMapping(
            target_path="visual.merger.mlp_fc2.bias",
            sharding=(None,),
            transpose=False,
        )
        
        # Vision transformer layers
        if hasattr(self.config, "vision_config"):
            num_vision_layers = getattr(self.config.vision_config, "depth", 0)
            for layer_idx in range(num_vision_layers):
                vision_layer_mappings = self._create_vision_layer_mappings(layer_idx)
                mappings.update(vision_layer_mappings)
        
        return mappings
    
    def _create_vision_layer_mappings(self, layer_idx: int) -> dict:
        """Create weight mappings for a single vision transformer layer.
        
        Args:
            layer_idx: Index of the vision layer
            
        Returns:
            Dictionary mapping vision layer weight names to model paths
        """
        from sgl_jax.srt.utils.weight_utils import WeightMapping
        
        prefix = f"visual.blocks.{layer_idx}"
        target_prefix = f"visual.blocks.{layer_idx}"
        
        mappings = {
            # Attention norm
            f"{prefix}.norm1.weight": WeightMapping(
                target_path=f"{target_prefix}.norm1.scale",
                sharding=(None,),
                transpose=False,
            ),
            # Attention QKV projection
            f"{prefix}.attn.qkv.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.attn.qkv.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.bias",
                sharding=("tensor",),
                transpose=False,
            ),
            # Attention output projection
            f"{prefix}.attn.proj.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.attn.proj.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.o_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            # MLP norm
            f"{prefix}.norm2.weight": WeightMapping(
                target_path=f"{target_prefix}.norm2.scale",
                sharding=(None,),
                transpose=False,
            ),
            # MLP gate projection
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.gate_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.bias",
                sharding=("tensor",),
                transpose=False,
            ),
            # MLP up projection
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.bias",
                sharding=("tensor",),
                transpose=False,
            ),
            # MLP down projection
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
        }
        
        return mappings
    
    def _create_layer_mappings(self, layer_idx: int) -> dict:
        """Create weight mappings for a single language model layer.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            Dictionary mapping layer weight names to model paths
        """
        from sgl_jax.srt.utils.weight_utils import WeightMapping
        
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
        
        # Add bias mappings if attention_bias is enabled
        if getattr(self.config, "attention_bias", True):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=("tensor",),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=("tensor",),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=("tensor",),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
            }
            mappings.update(bias_mappings)
        
        return mappings


EntryClass = [Qwen2_5_VLForConditionalGeneration]
