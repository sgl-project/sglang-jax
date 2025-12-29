from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.multimodal.configs.models.vaes.wanvae import WanVAEConfig
from sgl_jax.srt.multimodal.models.vaes.wanvae import AutoencoderKLWan
from sgl_jax.srt.server_args import ServerArgs


class VaeModelRunner(BaseModelRunner):
    def __init__(self, server_args: ServerArgs = None, mesh: jax.sharding.Mesh = None):
        # self.model_loader = get_model_loader(
        #     load_config = LoadConfig(
        #         load_format = server_args.load_format,
        #         perdownload_dir = server_args.download_dir,
        #     ),
        #     mesh = self.mesh,
        # )
        self.initialize()

    def initialize(self):
        self.load_model()
        self.initialize_jit()

    def load_model(self):
        # self.model = self.model_loader.load_model(
        #     model_config = self.model_config,
        # )
        self.model = AutoencoderKLWan(WanVAEConfig(), rngs=nnx.Rngs(0))

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


if __name__ == "__main__":
    runner = VaeModelRunner()
    x = jnp.array(np.arange(1 * 5 * 192 * 192 * 3), dtype=jnp.float32).reshape(1, 5, 192, 192, 3)
    y, cache_miss = runner.forward(x, "encode")
    print(f"encode cache_miss {cache_miss}")
    y, cache_miss = runner.forward(x, "encode")
    print(f"encode cache_miss {cache_miss}")

    x = jnp.array(np.arange(1 * 5 * 3 * 4 * 16), dtype=jnp.float32).reshape(1, 5, 3, 4, 16)
    y, cache_miss = runner.forward(x, "decode")
    print(f"decode cache_miss {cache_miss}")
    y, cache_miss = runner.forward(x, "decode")
    print(f"decode cache_miss {cache_miss}")
