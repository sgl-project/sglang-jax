import logging
from contextlib import nullcontext

import jax
from flax import nnx
from jax.experimental.pjit import pjit
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.configs.config_registry import get_vae_config
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class VaeModelRunner(BaseModelRunner):
    def __init__(
        self,
        server_args: ServerArgs = None,
        mesh: jax.sharding.Mesh = None,
        model_class=None,
        stage_sub_dir: str | None = None,
    ):
        self.mesh = mesh
        self.replicated_sharding = NamedSharding(self.mesh, P()) if self.mesh is not None else None
        self.decode_batch_axis = self._get_decode_batch_axis()
        self.decode_batch_axis_size = (
            self.mesh.shape[self.decode_batch_axis] if self.decode_batch_axis is not None else 1
        )
        self.decode_input_sharding = (
            NamedSharding(self.mesh, P(self.decode_batch_axis, None, None, None, None))
            if self.decode_batch_axis is not None
            else self.replicated_sharding
        )
        load_sub_dir = "vae" if stage_sub_dir is None else stage_sub_dir
        if load_sub_dir == "":
            load_sub_dir = None
        self.model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=model_class,
                sub_dir=load_sub_dir,
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

    def _get_decode_batch_axis(self) -> str | None:
        if self.mesh is None:
            return None

        for axis_name in ("data", "tensor"):
            if axis_name in self.mesh.axis_names and self.mesh.shape.get(axis_name, 1) > 1:
                return axis_name

        for axis_name in self.mesh.axis_names:
            if self.mesh.shape.get(axis_name, 1) > 1:
                return axis_name

        return None

    def _get_mesh_context(self):
        if self.mesh is None:
            return nullcontext()
        try:
            return jax.sharding.use_mesh(self.mesh)
        except AttributeError:
            try:
                return jax.set_mesh(self.mesh)
            except AttributeError:
                return self.mesh

    def should_use_spmd_decode(self, batch_size: int) -> bool:
        return self.decode_batch_axis is not None and batch_size >= self.decode_batch_axis_size

    def get_decode_input_sharding(self, batch_size: int) -> NamedSharding | None:
        if self.should_use_spmd_decode(batch_size):
            return self.decode_input_sharding
        return self.replicated_sharding

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        def encode_impl(model_def, model_state_def, model_state_leaves, x):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.encode(x)

        def decode_impl(model_def, model_state_def, model_state_leaves, x):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.decode(x)

        encode = jax.jit(encode_impl, static_argnames=["model_state_def"])
        decode = jax.jit(decode_impl, static_argnames=["model_state_def"])
        spmd_decode = None
        if self.decode_batch_axis is not None:
            model_state_shardings = jax.tree_util.tree_map(
                lambda _: self.replicated_sharding, model_state_leaves
            )
            spmd_decode = pjit(
                decode_impl,
                in_shardings=(None, None, model_state_shardings, self.decode_input_sharding),
                out_shardings=self.decode_input_sharding,
                static_argnames=["model_state_def"],
            )

        def encode_wrapper(x: jax.Array):
            return encode(model_def, model_state_def, model_state_leaves, x)

        def decode_wrapper(x: jax.Array):
            return decode(model_def, model_state_def, model_state_leaves, x)

        def spmd_decode_wrapper(x: jax.Array):
            return spmd_decode(model_def, model_state_def, model_state_leaves, x)

        self.jitted_encode = encode_wrapper
        self.jitted_decode = decode_wrapper
        self.spmd_decode = spmd_decode_wrapper if spmd_decode is not None else None

    def forward(self, x: jax.Array, mode: str):
        cache_miss_count = 0
        import jax._src.test_util as jtu

        with self._get_mesh_context():
            with jtu.count_pjit_cpp_cache_miss() as count:
                if mode == "encode":
                    output = self.jitted_encode(x)
                elif mode == "decode":
                    if self.should_use_spmd_decode(x.shape[0]) and self.spmd_decode is not None:
                        output = self.spmd_decode(x)
                    else:
                        output = self.jitted_decode(x)
                cache_miss_count = count()
        return output, cache_miss_count
