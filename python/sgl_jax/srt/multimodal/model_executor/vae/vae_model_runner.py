from functools import partial
import glob
import json
import os

import jax
from flax import nnx
from safetensors import safe_open

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.configs.config_registry import get_vae_config
from sgl_jax.srt.server_args import ServerArgs


class VaeModelRunner(BaseModelRunner):
    def __init__(
        self, server_args: ServerArgs = None, mesh: jax.sharding.Mesh = None, model_class=None
    ):
        self.mesh = mesh
        self.model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=model_class,
                sub_dir="vae",
            ),
            mesh=self.mesh,
        )
        self.model_class = model_class
        self.server_args = server_args
        self.initialize()

    def initialize(self):
        self.load_model()
        self.initialize_jit()

    def load_model(self):
        self.model_config = get_vae_config(self.server_args.model_path)
        self.model_config.model_path = self.server_args.model_path
        self.model_config.model_class = self.model_class
        self.model = self.model_loader.load_model(
            model_config=self.model_config,
        )
        # self.model = self.model_class(WanVAEConfig(), rngs=nnx.Rngs(0))

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        @partial(
            jax.jit,
            static_argnames=["model_state_def"],
        )
        def encode(
            model_def,
            model_state_def,
            model_state_leaves,
            x,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.encode(x)

        @partial(
            jax.jit,
            static_argnames=["model_state_def"],
        )
        def decode(
            model_def,
            model_state_def,
            model_state_leaves,
            x,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.decode(x)

        def encode_wrapper(x: jax.Array):
            return encode(model_def, model_state_def, model_state_leaves, x)

        def decode_wrapper(x: jax.Array):
            return decode(model_def, model_state_def, model_state_leaves, x)

        self.jitted_encode = encode_wrapper
        self.jitted_decode = decode_wrapper

    def forward(self, x: jax.Array, mode: str):
        cache_miss_count = 0
        import jax._src.test_util as jtu

        with jtu.count_pjit_cpp_cache_miss() as count:
            if mode == "encode":
                output = self.jitted_encode(x)
            elif mode == "decode":
                output = self.jitted_decode(x)
            cache_miss_count = count()
        return output, cache_miss_count
