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

class KimiK25VisionPatchEmbed(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs = None,
        patch_size: int = 14,
        in_channels: int = 3, # TODO: check which model config this corresponds to
        temporal_patch_size: int = 2,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.kernel_size = (patch_size, patch_size) # TODO: Verify if the temporal dimension to include
        self.temporal_patch_size = temporal_patch_size

        self.pos_emb = None # TODO: Implement position embedding
        
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        L, dim = x.shape
        channels = dim // (self.patch_size**2 * self.temporal_patch_size)

        x = x.reshape(L, channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        x = self.proj(x)
        return x.reshape(L, self.hidden_size)

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
            self.vt_hidden_size,
            3 * self.vt_hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )


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

        norm_layer = partial(
            nnx.RMSNorm,
            epsilon=config.projector_ln_eps,
            scale_init=nnx.with_partitioning(init_fn, (None, )), # TODO: validate this initialize
        )

        self.attn = KimiK25VisionAttention(config, dtype, mesh, rngs)

        self.mlp = KimiK25VisionMLP(config, dtype, mesh, rngs)

        _rngs = rngs or nnx.Rngs(0)
        self.pre_norm = norm_layer(config.vt_hidden_size, dtype=dtype, rngs=_rngs)

        self.proj = nnx.Linear(
            self.vt_hidden_size,
            self.vt_hidden_size,
            use_bias=True,
            param_dtpe=dtype,
            rngs=_rngs,
        )

        self.post_norm = norm_layer(config.vt_hidden_size, dtype=dtype, rngs=_rngs)

    def __call__(self, x:jnp.Array):


class VisionTowerEncoder(nnx.Module):

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.dtype = dtype

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

        norm_layer = partial(
            nnx.RMSNorm,
            epsilon=config.projector_ln_eps,
            scale_init=nnx.with_partitioning(init_fn, (None, )), # TODO: validate this initialize
        )

        _rngs = rngs or nnx.Rngs(0)

        self.final_layernorm = norm_layer(config.vt_hidden_size, dtype=dtype, rngs=_rngs)

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

        self.patch_embed = KimiK25PatchEmbed(rngs, config.patch_size, 3, config.vt_hidden_size, dtype) 

        self.encoder = VisionTowerEncoder(config, dtype, mesh, norm_eps, rngs)

class Kimi_K25_VisionModel(nnx.Module):
    '''
    Placeholder model class for the ViT stage.
    - Call encode_vision() to get vision embeddings
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

        for layer_idx in range(self.config):
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

                    

            


























