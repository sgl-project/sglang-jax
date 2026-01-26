from functools import partial

import jax
from flax import nnx

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.configs.config_registry import get_qwen_vl_config

from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req

class VitModelRunner(BaseModelRunner):
    """Runner shell for ViT stage execution."""

    def __init__(
            self,
            server_args: MultimodalServerArgs = None,
            mesh: jax.sharding.Mesh = None,
            model_class=None
    ):
        self.mesh = mesh
        self.model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=model_class
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
        self.model_config = get_qwen_vl_config(self.server_args.model_path)
        self.model_config.model_path = self.server_args.model_path
        self.model_config.model_class = self.model_class
        self.model = self.model_loader.load_model(
            model_config=self.model_config,
        )

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        @partial(
            jax.jit,
            static_argnames=["model_state_def"],
        )
        def encode_vision(
                model_def,
                model_state_def,
                model_state_leaves,
                x,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.encode_vision(x)

        def encode_vision_wrapper(x: jax.Array):
            return encode_vision(model_def, model_state_def, model_state_leaves, x)

        self.jitted_encode_vision = encode_vision_wrapper

    def forward(self, batch: Req, mesh: jax.sharding.Mesh):
        return batch
