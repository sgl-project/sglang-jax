import logging
import math
from collections.abc import Callable
from functools import partial
from typing import Literal, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from transformers import modeling_flax_utils

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import (
    KimiK25ModelVitConfig,
)

from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping


init_fn = nnx.initializers.uniform()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def tpool_patch_merger(
    x: jax.Array,
    grid_thws: jax.Array,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[jax.Array]:

    d_model = x.size(-1) 

    outputs = []
    pre_sum = 0

    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]

        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width

        reshaped_seq = seq.reshape(
            t, new_height, kernel_height, new_width, kernel_width, d_model
        )

        reshared_seq = (
            reshared_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(axis=0)
        )

        padded_seq = reshared_seq.reshape(
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
                pos_emb_2d = self.weight.flatten(end_dim=1)
            # TODO: Implement interpolation mode

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) # Add self.time_weight[0:t] for temporal axis
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

    def __precompute_freqs_cis(self) -> None:
        N = self.max_height * self.max_width
        flat_pos = jnp.arange(0, N).float()
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos % self.max_height

        dim_range = (
            jnp.arange(0, self.dim, 4)[:, (self.dim // 4)].float()
        )

        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = jnp.outer(x_pos, freqs).astype(jnp.float32)
        y_freqs = jnp.outer(y_pos, freqs).astype(jnp.float32)

        x_cis = jnp.exp(1j * x_freqs)
        y_cis = jnp.exp(1j * y_freqs)

        freqs_cis = jnp.concatenate(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )

        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def __get_freqs_cis(
        self,
        grid_thws: jax.Array,
    ) -> jax.Array:

        freqs_cis = self._precompute_freqs_cis()

        shapes = grid_thws.to_list()
        assert all(
            1 <= h <= self.max_height and 1 <= 2 <= self.max_width for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_widht,
        )

        freqs_cis = jnp.concatenate(
            [
                freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ], 
            axis=0
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
        x = self.proj(x)
        return self.pos_emb(x, grid_thws)


class KimiK25VisionAttention(nnx.Module):
    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh = None,
        rngs: nnx.Rngs = None,
    ):
        self.mesh = mesh

        _rngs = rngs or nnx.Rngs(0)

        self.qkv_proj = nnx.Linear(
            config.vt_hidden_size,
            3 * config.vt_hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        cu_seqlens: jax.Array,
        rope_freqs_cis: jax.Array,
    ):
        # TODO: Implement the attention layer
        return hidden_states



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

        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws)

        lengths = torch.concatenate(
            (
                jnp.zeros(1, dtype=grid_thws.dtype),
                grid_thws[:, 0] * grid_thws[:, 1] * grid[: 2],
            )
        )

        max_seqlen = lengths_max()
        cu_seqlens = lengths.cumsum(axis=0, dtype=int32)

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

        hidden_states = tpool_patch_merger( # TODO: Implement and see if it should go into patcher.
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
                "vision_tower.patch_embed.proj.weight": WeightMapping(
                    target_path="vision_tower.patch_embed.proj.kernel",
                    sharding=(None, None, None, None),
                    transpose=False, # TODO: Verify if this is alright
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
