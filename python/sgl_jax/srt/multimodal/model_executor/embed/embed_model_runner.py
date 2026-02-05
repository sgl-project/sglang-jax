from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoConfig

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req


class EmbedModelRunner(BaseModelRunner):
    """Runner shell for Omni Embedding stage execution."""

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
        model_config = AutoConfig.from_pretrained(
            self.server_args.model_path,
            trust_remote_code=True,
        )
        self.model_config = model_config.thinker_config
        self.model_config.revision = None
        self.model_config.dtype = jnp.bfloat16
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
        def forward_model(
            model_def,
            model_state_def,
            model_state_leaves,
            input_ids,
            input_features,
            audio_feature_lengths,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model(
                input_ids=input_ids,
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

        def forward_wrapper(
            input_ids: jax.Array,
            input_features: jax.Array | None = None,
            audio_feature_lengths: jax.Array | None = None,
            pixel_values: jax.Array | None = None,
            pixel_values_videos: jax.Array | None = None,
            image_grid_thw: jax.Array | None = None,
            video_grid_thw: jax.Array | None = None,
        ):
            return forward_model(
                model_def,
                model_state_def,
                model_state_leaves,
                input_ids,
                input_features,
                audio_feature_lengths,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
            )

        self.jitted_embedding = forward_wrapper

    def _prepare_input(self, batch: Req):
        """Prepare all input parameters required by jitted_embedding.

        Args:
            batch: Input batch request

        Returns:
            dict: Dictionary containing all parameters needed by jitted_embedding
        """
        # Extract input_ids
        input_ids = batch.input_ids or batch.origin_input_ids
        input_ids = jnp.asarray(input_ids) if input_ids is not None else None

        # Extract multimodal data from omni_inputs
        omni_inputs = batch.omni_inputs if isinstance(batch.omni_inputs, dict) else None

        # Initialize all input parameters
        audio_features = None
        audio_feature_lengths = None
        pixel_values = None
        pixel_values_videos = None
        image_grid_thw = None
        video_grid_thw = None

        if omni_inputs is not None:
            # Audio features
            audio_features = (
                jnp.asarray(batch.audio_features) if batch.audio_features is not None else None
            )

            # Audio feature lengths
            audio_feature_lengths = omni_inputs.get("audio_feature_lengths")
            if audio_feature_lengths is not None:
                audio_feature_lengths = jnp.asarray(audio_feature_lengths)

            # Image pixel values
            pixel_values = (
                jnp.asarray(batch.pixel_values_images)
                if batch.pixel_values_images is not None
                else None
            )

            # Video pixel values
            pixel_values_videos = (
                jnp.asarray(batch.pixel_values_videos)
                if batch.pixel_values_videos is not None
                else None
            )

            # Image grid THW - convert from tuple(tuple()) to 2D JAX tensor
            image_grid_thw = batch.image_grid_thw
            if image_grid_thw is not None:
                image_grid_thw = jnp.array(image_grid_thw)

            # Video grid THW - convert from tuple(tuple()) to 2D JAX tensor
            video_grid_thw = batch.video_grid_thw
            if video_grid_thw is not None:
                video_grid_thw = jnp.array(video_grid_thw)

        return {
            "input_ids": input_ids,
            "input_features": audio_features,
            "audio_feature_lengths": audio_feature_lengths,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
        }

    def forward(self, batch: Req, mesh: jax.sharding.Mesh):
        # Prepare inputs
        inputs = self._prepare_input(batch)

        # Call jitted_embedding
        input_embeds, visual_embeds_multiscale, visual_pos_masks = self.jitted_embedding(**inputs)

        mm_inputs = batch.omni_inputs if isinstance(batch.omni_inputs, dict) else None
        if mm_inputs is not None:
            mm_inputs["multimodal_embedding"] = input_embeds
            mm_inputs["deepstack_visual_embedding"] = visual_embeds_multiscale
            mm_inputs["deepstack_visual_pos_mask"] = visual_pos_masks

        return batch
