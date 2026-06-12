import logging

import jax
from flax import nnx
from jax import numpy as jnp
from transformers import Qwen3OmniMoeThinkerConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.audio_encoder import (
    Qwen3OmniMoeAudioEncoder,
)
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.vision_encoder import (
    Qwen3OmniMoeVisionEncoder,
)
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.weights_mapping import (
    create_audio_tower_weight_mappings,
    create_visual_weight_mappings,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class Qwen3OmniMoeThinkerEmbedding(nnx.Module):
    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mesh = mesh
        self.dtype = dtype
        self.config = config
        self.audio_tower = Qwen3OmniMoeAudioEncoder(
            config.audio_config, mesh=mesh, dtype=dtype, rngs=rngs
        )
        self.visual = Qwen3OmniMoeVisionEncoder(
            config.vision_config, mesh=mesh, dtype=dtype, rngs=rngs
        )
        self.text_embed_tokens = Embed(
            num_embeddings=config.text_config.vocab_size,
            features=config.text_config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = {
            **create_audio_tower_weight_mappings(self.config.audio_config),
            **create_visual_weight_mappings(self.config.vision_config),
            **self._create_text_embed_tokens_mappings(self.config.text_config),
        }

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3OmniMoeThinkerEmbedding weights loaded successfully!")

    @staticmethod
    def _create_text_embed_tokens_mappings(config: Qwen3OmniMoeAudioEncoderConfig) -> dict:
        mappings = {}
        prefix = "thinker.model"
        target_prefix = "text_embed_tokens"

        mappings[f"{prefix}.embed_tokens.weight"] = WeightMapping(
            target_path=f"{target_prefix}.embedding",
            sharding=("tensor", None),
        )

        return mappings

    def get_placeholder_mask(
        self,
        input_ids: jnp.ndarray,
        input_embeds: jnp.ndarray,
        image_features: jnp.ndarray | None = None,
        video_features: jnp.ndarray | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `input_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = jnp.sum(special_image_mask)
        special_image_mask = jnp.broadcast_to(
            jnp.expand_dims(special_image_mask, axis=-1), input_embeds.shape
        )
        if (
            image_features is not None
            and input_embeds[special_image_mask].size != image_features.size
        ):
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = jnp.sum(special_video_mask)
        special_video_mask = jnp.broadcast_to(
            jnp.expand_dims(special_video_mask, axis=-1), input_embeds.shape
        )
        if (
            video_features is not None
            and input_embeds[special_video_mask].size != video_features.size
        ):
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        special_audio_mask = jnp.broadcast_to(
            jnp.expand_dims(special_audio_mask, axis=-1), input_embeds.shape
        )

        return special_image_mask, special_video_mask, special_audio_mask

    def __call__(
        self,
        input_ids: jax.Array,
        input_features=None,
        audio_feature_lengths=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """

        # 1. Extract the input embeddings
        input_embeds = self.text_embed_tokens(input_ids)

        visual_embeds_multiscale = None
        visual_pos_masks = None

        audio_embeds = None
        image_embeds = None
        video_embeds = None

        # Merge text , audios , image and video
        if input_features is not None:
            audio_embeds = self.audio_tower(
                input_features.astype(self.dtype),
                feature_lens=audio_feature_lengths,
            )

        if pixel_values is not None:
            image_features = self.visual(pixel_values.astype(self.dtype), image_grid_thw)
            image_embeds, image_embeds_multiscale = (
                image_features["pooler_output"],
                image_features["deepstack_features"],
            )
            visual_embeds_multiscale = image_embeds_multiscale

        if pixel_values_videos is not None:
            video_features = self.visual(pixel_values_videos.astype(self.dtype), video_grid_thw)
            video_embeds, video_embeds_multiscale = (
                video_features["pooler_output"],
                video_features["deepstack_features"],
            )
            if visual_embeds_multiscale is None:
                visual_embeds_multiscale = video_embeds_multiscale

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(
            input_ids,
            input_embeds=input_embeds,
            image_features=image_embeds,
            video_features=video_embeds,
        )
        if audio_embeds is not None:
            input_embeds = input_embeds.at[audio_mask].set(jnp.ravel(audio_embeds))
        if image_embeds is not None:
            input_embeds = input_embeds.at[image_mask].set(jnp.ravel(image_embeds))
        if video_embeds is not None:
            input_embeds = input_embeds.at[video_mask].set(jnp.ravel(video_embeds))

        # for image and video mask
        if pixel_values is not None and pixel_values_videos is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask | image_mask
            visual_embeds_multiscale_joint = ()
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(image_embeds_multiscale, video_embeds_multiscale):
                embed_joint = jnp.zeros(
                    (visual_pos_masks.sum(), img_embed.shape[-1]), dtype=img_embed.dtype
                )
                embed_joint = embed_joint.at[image_mask_joint, :].set(img_embed)
                embed_joint = embed_joint.at[video_mask_joint, :].set(vid_embed)
                visual_embeds_multiscale_joint = visual_embeds_multiscale_joint + (embed_joint,)
            visual_embeds_multiscale = visual_embeds_multiscale_joint
        elif pixel_values is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
        elif pixel_values_videos is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
        return input_embeds, visual_embeds_multiscale, visual_pos_masks
