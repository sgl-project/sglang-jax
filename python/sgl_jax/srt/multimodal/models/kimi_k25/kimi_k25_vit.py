import logging
import math
from collections.abc import Callable
from functools import partial
from typing import Literal, TypedDict

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from transformers import modeling_flax_utils

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import (
    KimiK25ModelVitConfig,
)
from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds, flash_attention
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping


init_fn = nnx.initializers.uniform()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def tpool_patch_merger(
    x: jax.Array,
    grid_thws: jax.Array,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[jax.Array]:

    d_model = x.shape[-1]

    outputs = []
    pre_sum = 0

    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]

        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width

        reshaped_seq = seq.reshape(
            t, new_height, kernel_height, new_width, kernel_width, d_model
        )

        reshaped_seq = (
            reshaped_seq.transpose(0, 1, 3, 2, 4, 5).mean(axis=0)
        )

        padded_seq = reshaped_seq.reshape(
            new_height * new_width, kernel_height * kernel_width, -1
        )

        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs

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
        self.weight = nnx.Param(
            nnx.initializers.normal()(_rngs.params(), (height, width, dim))
        )

    def __call__(
        self,
        x: jax.Array,
        grid_thws: jax.Array
    ) -> jax.Array:

        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.reshape(-1, self.weight.shape[-1])
            else:
                pos_emb_2d = self.weight[:h, :w, :].reshape(-1, self.weight.shape[-1])
                
            # TODO: Implement interpolation mode

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    jnp.expand_dims(pos_emb_2d, axis=0).repeat(t, 1, 1) # Add self.time_weight[0:t] for temporal axis
                )

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + jnp.concatenate(pos_embs, axis=0)
        return out

class Rope2DPosEmbRepeated(nnx.Module):

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: int = 10000, # Verify this value with config
    ):
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self) -> jax.Array:
        N = self.max_height * self.max_width
        flat_pos = jnp.arange(0, N).astype(jnp.float32)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width

        dim_range = (
            jnp.arange(0, self.dim, 4)[: (self.dim // 4)].astype(jnp.float32)
        )

        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = jnp.outer(x_pos, freqs).astype(jnp.float32)
        y_freqs = jnp.outer(y_pos, freqs).astype(jnp.float32)

        cos_x = jnp.cos(x_freqs)
        sin_x = jnp.sin(x_freqs)
        cos_y = jnp.cos(y_freqs)
        sin_y = jnp.sin(y_freqs)

        cos_emb = jnp.stack([cos_x, cos_y], axis=-1).reshape(N, -1)
        sin_emb = jnp.stack([sin_x, sin_y], axis=-1).reshape(N, -1)

        freqs_cis = jnp.stack([cos_emb, sin_emb], axis=0)
        freqs_cis = freqs_cis.reshape(2, self.max_height, self.max_width, -1)
        return freqs_cis

    def _get_freqs_cis(
        self,
        grid_thws: jax.Array,
    ) -> jax.Array:
        freqs_cis = self._precompute_freqs_cis()

        shapes = grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )

        freqs_cis = jnp.concatenate(
            [
                freqs_cis[:, :h, :w].reshape(2, -1, self.dim // 2).repeat(t, axis=1)
                for t, h, w in shapes
            ], 
            axis=1
        )

        return freqs_cis


class KimiK25VisionPatchEmbed(nnx.Module):

    def __init__(
        self,
        rngs: nnx.Rngs = None,
        patch_size: int = 14,
        in_channels: int = 3, # TODO: check which model config this corresponds to
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
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=hidden_size
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

    def __call__(self, x: jax.Array, grid_thws: jax.Array) -> jax.Array:
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.proj(x)

        # After conv, shape is (L, T_out, H_out, W_out, C_out)
        # With stride=kernel_size, T_out=H_out=W_out=1.
        # So shape is (L, 1, 1, 1, hidden_size)
        x = x.reshape(-1, self.hidden_size)
        return self.pos_emb(x, grid_thws)


def align_to(x, a):
    return pl.cdiv(x, a) * a


class KimiK25VisionAttention(nnx.Module):
    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh = None,
        rngs: nnx.Rngs = None,
    ):
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

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape: [S, D] -> [S, N, H_D]
        q = q.reshape(sum_seq_len, self.num_heads, self.head_dim)
        k = k.reshape(sum_seq_len, self.num_heads, self.head_dim)
        v = v.reshape(sum_seq_len, self.num_heads, self.head_dim)

        # Apply 2D RoPE
        cos_emb, sin_emb = position_embeddings[0], position_embeddings[1]

        q_real = q[..., 0::2]
        q_imag = q[..., 1::2]
        q_rot_real = q_real * cos_emb[:, None, :] - q_imag * sin_emb[:, None, :]
        q_rot_imag = q_real * sin_emb[:, None, :] + q_imag * cos_emb[:, None, :]
        q = jnp.stack([q_rot_real, q_rot_imag], axis=-1).reshape(sum_seq_len, self.num_heads, self.head_dim)

        k_real = k[..., 0::2]
        k_imag = k[..., 1::2]
        k_rot_real = k_real * cos_emb[:, None, :] - k_imag * sin_emb[:, None, :]
        k_rot_imag = k_real * sin_emb[:, None, :] + k_imag * cos_emb[:, None, :]
        k = jnp.stack([k_rot_real, k_rot_imag], axis=-1).reshape(sum_seq_len, self.num_heads, self.head_dim)

        # TPU Path: Segmented TPU Pallas FlashAttention
        
        # 1. Pad sequence length to multiple of 256
        align_seq_len = align_to(sum_seq_len, 256)
        
        pad_q = q
        pad_k = k
        pad_v = v
        
        seg_q = None
        seg_kv = None
        segment_ids = None
        
        if sum_seq_len != align_seq_len:
            pad_q = jnp.pad(q, ((0, align_seq_len - sum_seq_len), (0, 0), (0, 0)))
            pad_k = jnp.pad(k, ((0, align_seq_len - sum_seq_len), (0, 0), (0, 0)))
            pad_v = jnp.pad(v, ((0, align_seq_len - sum_seq_len), (0, 0), (0, 0)))
            
            # Generate segment IDs: valid tokens have positive indices, padding has 0
            indices = jnp.arange(sum_seq_len)
            item_ids = jnp.sum(indices[:, None] >= cu_seqlens[1:][None, :], axis=-1) + 1
            
            seg_q = jnp.pad(item_ids, (0, align_seq_len - sum_seq_len))
            seg_kv = jnp.pad(item_ids, (0, align_seq_len - sum_seq_len))
            
            segment_ids = SegmentIds(q=seg_q[None, :], kv=seg_kv[None, :])

        # Reshape to batch-format expected by Pallas kernel: [B=1, H, S, H_D]
        pad_q = jnp.transpose(pad_q, (1, 0, 2))[None, ...]
        pad_k = jnp.transpose(pad_k, (1, 0, 2))[None, ...]
        pad_v = jnp.transpose(pad_v, (1, 0, 2))[None, ...]

        # Execute TPU Pallas FlashAttention kernel
        output = flash_attention(
            pad_q,
            pad_k,
            pad_v,
            segment_ids=segment_ids,
            causal=False,
            sm_scale=self.scale,
        )

        # Reshape back: [B=1, H, S, H_D] -> [S, H, H_D] -> slice back to sum_seq_len -> [S, D]
        output = jnp.transpose(output[0], (1, 0, 2))
        output = output[:sum_seq_len, :, :].reshape(sum_seq_len, D)

        return output



class KimiK25VisionMLP(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh = None,
        rngs: nnx.Rngs = None,
    ):
        in_features = config.vt_hidden_size
        intermediate_size = config.vt_intermediate_size

        _rngs = rngs or nnx.Rngs(0)

        self.mesh = mesh

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

        self.act_fn = modeling_flax_utils.ACT2FN[config.projector_hidden_act] # TODO: Verify if this is the right param

    def __call__(self, x: jax.Array) -> jax.Array:
        up = self.act_fn(self.up_proj(x))
        return self.down_proj(up)


class KimiK25VisionBlock(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
        rngs: nnx.Rngs = None,
    ):

        self.attn = KimiK25VisionAttention(config, dtype, mesh, rngs) # TODO: Investigate the working of the Attention with RoPE

        self.mlp = KimiK25VisionMLP(config, dtype, mesh, rngs)

        _rngs = rngs or nnx.Rngs(0)
        self.pre_norm = nnx.LayerNorm(config.vt_hidden_size, param_dtype=dtype, rngs=_rngs)

        self.proj = nnx.Linear( # TODO: Add activation function too
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
        max_seqlen: int,
        rope_freqs_cis: jax.Array
    ):
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states)

        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=rope_freqs_cis,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VisionTowerEncoder(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
        rngs: nnx.Rngs = None,
        video_attn_type: str = "spatial_temporal",
    ):
        self.config = config
        self.dtype = dtype

        assert (video_attn_type == "spatial_temporal"), f'video_attn_type must be "spatial_temporal", got {video_attn_type}'

        self.rope_2d = Rope2DPosEmbRepeated(config.vt_hidden_size // config.vt_num_attention_heads, 512, 512) 

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
        grid_thws: jax.Array,
    ) -> jax.Array:

        rope_freqs_cis = self.rope_2d._get_freqs_cis(grid_thws=grid_thws)

        lengths = jnp.concatenate(
            (
                jnp.zeros(1, dtype=grid_thws.dtype),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )

        max_seqlen = lengths.max()
        cu_seqlens = lengths.cumsum(axis=0, dtype=jnp.int32)

        for block in self.blocks:
            hidden_states = block(
                hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class VisionTower(nnx.Module):
    
    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
    ):
        self.config = config
        self.dtype = dtype

        self.merge_kernel_size = config.merge_kernel_size

        self.patch_embed = KimiK25VisionPatchEmbed(
            rngs,
            config.patch_size,
            3,
            config.init_pos_emb_height, 
            config.init_pos_emb_width,
            config.init_pos_emb_time,
            config.pos_emb_type,
            config.vt_hidden_size, 
            dtype)

        self.encoder = VisionTowerEncoder(config, dtype, mesh, norm_eps, rngs)

    def __call__(
        self,
        pixel_values: jax.Array,
        grid_thws: jax.Array,
    ) -> jax.Array:

        # TODO: Add assertions
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)

        hidden_states = tpool_patch_merger(
            hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size
        )

        return hidden_states

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
        self.pre_norm = nnx.LayerNorm(config.vt_hidden_size, param_dtype=dtype, rngs=_rngs)

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

        self.act = nnx.GELU()

    def __call__(
        self,
        image_features: jax.Array,
    ) -> jax.Array:
        hidden_states = self.pre_norm(image_features).reshape(-1, self.hidden_size)
        hidden_states = self.proj_0(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.proj_1(hidden_states)



class Kimi_K25_VisionModel(nnx.Module):
    '''
    Placeholder model class for the ViT stage.
    '''

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None
    ) -> None:

        self.config = config
        self.dtype = dtype
        self.mesh = mesh

        self.vision_tower = VisionTower(config, dtype, rngs, mesh, config.projector_ln_eps)

        logger.info("Kimi K2.5 Vision Model initialized with dtype %s", dtype)


    def load_weights(self, model_config: KimiK25ModelVitConfig) -> None:
        '''Load model weights with JAX distributed loading support'''

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_kimi_k25_vision_tower_weight_mappings()

        if self.mesh is not None:
            with self.mesh: # TODO: Understand this
                loader.load_weights_from_safetensors(weight_mappings)
        else:
            loader.load_weights_from_safetensors(weight_mappings)

        logger.info("Kimi-K2.5 - ViT stage weights loaded successfully!")

    def _create_kimi_k25_vision_tower_weight_mappings(self) -> dict:
        mappings = {}

        mappings.update(
            {
                "vision_tower.patch_embed.pos_emb.weight": WeightMapping(
                    target_path="vision_tower.patch_embed.pos_emb.weight",
                    sharding=(None,),
                    transpose=False,
                ),
                "vision_tower.patch_embed.proj.weight": WeightMapping(
                    target_path="vision_tower.patch_embed.proj.kernel",
                    sharding=(None, None, None, None),
                    transpose_axes=(2, 3, 1, 0),
                ),
                "vision_tower.patch_embed.proj.bias": WeightMapping(
                    target_path="vision_tower.patch_embed.proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
                "vision_tower.encoder.final_layernorm.weight": WeightMapping(
                    target_path="vision_tower.encoder.final_layernorm.scale",
                    sharding=(None,),
                    transpose=False,
                ),
                "vision_tower.encoder.final_layernorm.bias": WeightMapping(
                    target_path="vision_tower.encoder.final_layernorm.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            })

        for layer_idx in range(self.config.vt_num_hidden_layers):
            vision_layer_mappings = self._create_vision_layer_mappings(layer_idx)
            mappings.update(vision_layer_mappings)

        return mappings

    def _create_vision_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"vision_tower.encoder.blocks.{layer_idx}"

        mappings = {
            f"{prefix}.wqkv.weight": WeightMapping(
                target_path=f"{prefix}.attn.qkv_proj.kernel",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.wqkv.bias": WeightMapping(
                target_path=f"{prefix}.attn.qkv_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.wo.weight": WeightMapping(
                target_path=f"{prefix}.proj.kernel",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.wo.bias": WeightMapping(
                target_path=f"{prefix}.proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.fc0.weight": WeightMapping(
                target_path=f"{prefix}.mlp.up_proj.kernel",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.fc0.bias": WeightMapping(
                target_path=f"{prefix}.mlp.up_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.fc1.weight": WeightMapping(
                target_path=f"{prefix}.mlp.down_proj.kernel",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.fc1.bias": WeightMapping(
                target_path=f"{prefix}.mlp.down_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.norm0.weight": WeightMapping(
                target_path=f"{prefix}.pre_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.norm0.bias": WeightMapping(
                target_path=f"{prefix}.pre_norm.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.norm1.weight": WeightMapping(
                target_path=f"{prefix}.post_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.norm1.bias": WeightMapping(
                target_path=f"{prefix}.post_norm.bias",
                sharding=(None,),
                transpose=False,
            ),
        }

        return mappings
