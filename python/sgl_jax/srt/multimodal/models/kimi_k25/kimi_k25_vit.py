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

from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import (
    KimiK25ModelVitConfig,
)

init_fn = nnx.initializers.uniform()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VisionTower(nnx.Module):
    '''
    Placeholder class for Vision Tower for Kimi K2.5
    '''

    def __init__(
        self,
        config: KimiK25ModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
    ):
        self.config = config
        self.dtype = dtype

    def __call__(
        self,
        pixel_values: jax.Array,
        grid_thws: jax.Array,
    ) -> jax.Array:
        ''' 
        Returns zeros of shape (n_visual_tokens, hidden_size).
        grid_thws: numpy array or tuple of (t, h, w) tuples.
        ''' 
        grid_arr = np.array(grid_thws)
        merge_kernel_size = getattr(self.config, "merge_kernel_size", 2)

        n_visual = int(np.prod(grid_arr, axis=-1).sum()) // (merge_kernel_size ** 2)
        return jnp.zeros((n_visual, self.config.vt_hidden_size), dtype=self.dtype)


class Kimi_K25_VisionModel(nnx.Module):
    '''
    Placeholder class for Vision Model for Kimi K2.5
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
        self._hidden_size = getattr(config, "vt_hidden_size", 7168)

        self.vision_tower = VisionTower(config, dtype, rngs, mesh)

        logger.info("Kimi K2.5 Vision Model initialized with dtype %s", dtype)

    def text_embed(self, input_ids: jax.Array) -> jax.Array:
        """
        Placeholder for Text Embed. 
        Returns zeros of shape (seq_len, hidden_size).
        """
        return jnp.zeros((input_ids.shape[0], self._hidden_size), dtype=self.dtype)


    def load_weights(self, model_config: KimiK25ModelVitConfig) -> None:
        '''Placeholder for loading model weights with JAX distributed loading support'''

        logger.warning("KimiK25VisionModel: dummy load_weights — no weights loaded")

