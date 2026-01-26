import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.configs.config_registry import get_qwen_vl_config
from sgl_jax.srt.multimodal.manager.schedule_batch import Req


class VitModelRunner(BaseModelRunner):
    """Runner shell for ViT stage execution."""

    def __init__(
        self,
        server_args: MultimodalServerArgs = None,
        mesh: jax.sharding.Mesh = None,
        model_class=None,
    ):
        self.mesh = mesh
        self.model_loader = get_model_loader(
            load_config=LoadConfig(model_class=model_class),
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

        def _encode_vision_impl(
            model_def,
            model_state_def,
            model_state_leaves,
            pixel_values,
            window_index,
            rotary_pos_emb,
            cu_seqlens,
            cu_window_seqlens,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.visual.compute_hidden_states(
                pixel_values, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens
            )

        encode_vision = jax.jit(_encode_vision_impl, static_argnames=["model_state_def"])

        def _to_static_grid(grid_thw):
            if grid_thw is None:
                return None
            if isinstance(grid_thw, tuple):
                return tuple(tuple(int(x) for x in row) for row in grid_thw)
            grid = np.asarray(grid_thw)
            if grid.size == 0:
                return None
            return tuple(tuple(int(x) for x in row) for row in grid.tolist())

        def encode_vision_wrapper(pixel_values: jax.Array, image_grid_thw, video_grid_thw):
            image_grid_thw = _to_static_grid(image_grid_thw)
            video_grid_thw = _to_static_grid(video_grid_thw)
            combined_grid_thw = []
            if image_grid_thw:
                combined_grid_thw.extend(image_grid_thw)
            if video_grid_thw:
                combined_grid_thw.extend(video_grid_thw)
            if not combined_grid_thw:
                return jnp.zeros((0, self.model.config.hidden_size), dtype=pixel_values.dtype)
            combined_grid_thw = tuple(combined_grid_thw)
            window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens = (
                self.model.visual.compute_aux_arrays(combined_grid_thw)
            )
            return encode_vision(
                model_def,
                model_state_def,
                model_state_leaves,
                pixel_values,
                window_index,
                rotary_pos_emb,
                cu_seqlens,
                cu_window_seqlens,
            )

        self.jitted_encode_vision = encode_vision_wrapper

    def forward(self, batch: Req, mesh: jax.sharding.Mesh):
        batch.multimodal_embeddings = self.jitted_encode_vision(
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
            video_grid_thw=batch.video_grid_thw,
        )
        # Cache for downstream LLM stage (auto_regressive) to avoid re-encoding.
        batch.cached_vision_embeds = batch.multimodal_embeddings
        if hasattr(batch, "vlm_inputs") and isinstance(batch.vlm_inputs, dict):
            batch.vlm_inputs["cached_vision_embeds"] = batch.multimodal_embeddings
        print(
            f"VitModelRunner forward multimodal_embeddings shape: {batch.multimodal_embeddings.shape}"
        )
        return batch
