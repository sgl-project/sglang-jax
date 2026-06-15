import logging
import math
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import modeling_flax_utils

from sgl_jax.srt.utils.jax_utils import is_tpu_runtime

_FLASH_MHA = None


def _get_flash_mha():
    global _FLASH_MHA
    if _FLASH_MHA is None:
        from flash_attn_jax import flash_mha as _FLASH_MHA
    return _FLASH_MHA


init_fn = nnx.initializers.uniform()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    valid_mask: jax.Array | None = None,
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
    if not is_tpu_runtime():
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

        # V-2 bucketing: mask out padding-patch keys (so real patches never attend to the
        # padding added to reach a canonical bucket grid). valid_mask is [T] bool (in the
        # block-processing / window-permuted order); None when bucketing is off (0-diff).
        if valid_mask is not None:
            attn_weights = jnp.where(
                ~valid_mask[None, None, None, :], jnp.finfo(attn_weights.dtype).min, attn_weights
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


class Qwen2_5_VisionMLP(nnx.Module):
    def __init__(self, config, dtype: jnp.dtype, rngs: nnx.Rngs = None):
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
        config,
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
        valid_mask: jax.Array | None = None,
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

        # Compute attention using the backend function (valid_mask masks V-2 padding patches)
        output = vision_attention(q, k, v, self.scale, window_size, valid_mask=valid_mask)

        # Reshape back: [B, T, N, H] -> [T, B, D]
        output = output.transpose(1, 0, 2, 3).reshape(T, B, D)

        return self.proj(output)


class Qwen2_5_VisionBlock(nnx.Module):
    def __init__(
        self,
        config,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
    ):
        dim = config.hidden_size
        norm_layer = partial(
            nnx.RMSNorm,
            epsilon=norm_eps,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=_rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=_rngs)
        self.attn = Qwen2_5_VisionAttention(config=config, dtype=dtype, rngs=rngs, mesh=mesh)
        self.mlp = Qwen2_5_VisionMLP(config=config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_window_seqlens: jax.Array | None = None,
        use_fullattn: bool = True,
        valid_mask: jax.Array | None = None,
    ) -> jax.Array:
        x = x + self.attn(
            self.norm1(x), rotary_pos_emb, cu_window_seqlens, use_fullattn, valid_mask=valid_mask
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


class Qwen2_5_VL_VisionTransformer(nnx.Module):

    def __init__(
        self,
        config,
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
                    norm_eps=norm_eps,
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
        valid_mask: jax.Array | None = None,
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
                    valid_mask=valid_mask,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    cu_window_seqlens=cu_window_seqlens,
                    use_fullattn=False,
                    valid_mask=valid_mask,
                )

        # adapter
        hidden_states = self.merger(hidden_states)
        # JIT-safe argsort (numpy would break under JIT).
        reverse_indices = jnp.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def _compute_valid_units(
        self,
        grid_thw: tuple[tuple[int, int, int]],
        real_llm_dims: jax.Array,
    ) -> jax.Array:
        """V-2 bucketing: per-merge-unit bool mask (in canonical (t, llm_h, llm_w) row-major
        order) marking which units are real vs. bucket padding. A unit at LLM-grid (row, col)
        is real iff row < real_llm_h and col < real_llm_w. real_llm_dims is a *traced* [num_grids, 2]
        array (real_llm_h, real_llm_w per grid) -- traced, NOT static, so the encode jit compiles
        only on the *padded* (canonical) grid, never on the real size (that is the whole point of
        bucketing: bound the recompiles)."""
        m = self.spatial_merge_size
        valid_units = []
        for i, (t, h, w) in enumerate(grid_thw):
            llm_h, llm_w = h // m, w // m
            real_llm_h = real_llm_dims[i, 0]
            real_llm_w = real_llm_dims[i, 1]
            rows = jnp.arange(llm_h)[:, None] < real_llm_h
            cols = jnp.arange(llm_w)[None, :] < real_llm_w
            unit = jnp.broadcast_to((rows & cols)[None, :, :], (t, llm_h, llm_w)).reshape(
                -1
            )  # (t, llm_h, llm_w) row-major -> matches window_index unit ordering
            valid_units.append(unit)
        return jnp.concatenate(valid_units, axis=0)  # [sum_i t*llm_h*llm_w] bool

    def encode_bucketed(
        self,
        x: jax.Array,
        grid_thw: tuple[tuple[int, int, int]],
        real_llm_dims: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """V-2 bucketing entry point. ``x``/``grid_thw`` are padded to a canonical bucket grid
        (static); ``real_llm_dims`` (traced [num_grids, 2]) gives each image's true LLM-grid size.
        Returns ``(hidden_padded [num_units, out], valid_units [num_units] bool)`` in canonical
        (t, llm_h, llm_w) row-major order. Bucket-padding units are masked out of the ViT attention
        (so real patches never attend to them) and flagged in ``valid_units`` so the caller can
        compact them out before merge. Window tiling is anchored at origin in fixed wsize-blocks,
        so a real patch keeps its exact window membership in the padded grid; with masked keys
        contributing exp(finfo.min)=0 this is mathematically equal to no-bucketing, and in practice
        numerically equivalent at the float-rounding level (XLA's shape-dependent reduction order
        may differ by ULPs -- not guaranteed bit-identical; validated by identical greedy outputs).
        ``__call__`` (the non-bucketed path) is untouched."""
        window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens = self.compute_aux_arrays(
            grid_thw
        )
        valid_units = self._compute_valid_units(grid_thw, real_llm_dims)  # canonical order
        # Expand per-unit validity to per-patch (window order). Use broadcast+reshape, NOT
        # jnp.repeat: under the Explicit-axis AR mesh jnp.repeat demands an out_sharding, whereas
        # broadcast_to (new replicated axis) + reshape propagate the replicated sharding cleanly.
        vu = valid_units[window_index]  # [num_units] bool, window order
        valid_mask = jnp.broadcast_to(vu[:, None], (vu.shape[0], self.spatial_merge_unit)).reshape(
            -1
        )  # per-patch [seq]
        hidden_states = self.compute_hidden_states(
            x, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens, valid_mask=valid_mask
        )
        return hidden_states, valid_units

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
