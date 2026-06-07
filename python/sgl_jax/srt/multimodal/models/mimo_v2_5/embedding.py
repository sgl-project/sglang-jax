"""MiMo-V2.5 embed-stage model.

Builds the first-token embedding sequence for MiMo-V2.5 omni requests by embedding
text and scattering each modality's understanding-tower features into their
placeholder positions.

Interface is modality-uniform and forward-looking:

- Towers mirror the HF checkpoint's own attribute names: ``self.audio_encoder``
  (wired) and ``self.visual`` (the MiMoVL ViT shared by image+video, reserved).
- ``__call__`` keeps the full keyword surface (audio / image / video inputs) and
  runs the same ``_encode_<modality>`` -> ``_scatter_modality`` flow for each.
- To add vision later: implement the ViT (``mimo_v2_5/vision_encoder.py``),
  instantiate it as ``self.visual`` in ``__init__``, and add its weight mappings
  in ``weights_mapping.py``. ``_encode_image`` / ``_encode_video`` then start
  returning features with no further interface change.

``MiMoV25AudioUnderstandingEncoder`` is re-exported here for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.multimodal.models.mimo_v2_5.audio_encoder import (
    MiMoV25AudioUnderstandingEncoder,
)
from sgl_jax.srt.multimodal.models.mimo_v2_5.weights_mapping import (
    build_embedding_weight_mappings,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader

logger = logging.getLogger(__name__)

__all__ = ["EmbedOutput", "MiMoV2_5Embedding", "MiMoV25AudioUnderstandingEncoder"]


class EmbedOutput(NamedTuple):
    input_embeds: jax.Array
    # Reserved for towers that emit deepstack features (MiMo-V2.5 has none, so these
    # stay None; the MiMoVL ViT has a self-contained merger and no deepstack either).
    deepstack_embeds: jax.Array | None = None
    deepstack_pos_mask: jax.Array | None = None


class MiMoV2_5Embedding(nnx.Module):
    # Capability flag consumed by EmbedModelRunner: when this model receives audio,
    # the runner applies MiMo-V2.5 host-side audio payload/codes validation. Audio is
    # OPTIONAL — text-only / vision-only / any-subset requests skip the audio path.
    uses_mimo_v25_audio_contract: bool = True

    @staticmethod
    def _get_config_value(config: PretrainedConfig, key: str, default=None):
        # MiMo-V2.5 keeps several token ids (e.g. audio_token_id) only inside
        # processor_config, so fall back to it when the top-level attr is missing.
        value = getattr(config, key, None)
        if value is not None:
            return value
        processor_config = getattr(config, "processor_config", None)
        if isinstance(processor_config, dict):
            return processor_config.get(key, default)
        if processor_config is not None:
            return getattr(processor_config, key, default)
        return default

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.hidden_size = int(getattr(config, "hidden_size", 4096))
        self.text_embed_tokens = Embed(
            num_embeddings=int(getattr(config, "vocab_size", 152576)),
            features=self.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        # --- Modality understanding towers (names mirror the HF checkpoint) ---
        audio_config = getattr(config, "audio_config", config)
        self.audio_encoder = MiMoV25AudioUnderstandingEncoder(
            audio_config,
            mesh=mesh,
            dtype=dtype,
            rngs=rngs,
        )
        # Reserved: MiMoVL ViT, shared by image + video (HF `self.visual`).
        # Implement in mimo_v2_5/vision_encoder.py and instantiate here to enable.
        self.visual = None

        # --- Per-modality scatter token ids (image/video reserved) ---
        audio_token_id = self._get_config_value(config, "audio_token_id", 151669)
        self.audio_token_id = int(audio_token_id) if audio_token_id is not None else None
        self.image_token_id = self._get_config_value(config, "image_token_id")
        self.video_token_id = self._get_config_value(config, "video_token_id")

    @classmethod
    def get_embed_model_config(cls, model_config: PretrainedConfig) -> PretrainedConfig:
        return model_config

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        # When self.visual is wired, extend build_embedding_weight_mappings with the
        # ViT mappings (keyed off self.visual being present).
        mappings = build_embedding_weight_mappings(
            num_audio_channels=self.audio_encoder.audio_channels,
            num_input_local_layers=len(self.audio_encoder.input_local_transformer.layers),
        )
        # Hard-fail on a missing audio-tower key (review R2-5). load_weights_from_safetensors
        # only logs+skips a missing HF key, which would leave the audio tower at random
        # init and produce silently-wrong audio embeddings. Since the HF key prefixes here
        # are not yet verified against a real checkpoint, refuse to load rather than run
        # garbage: every speech_embeddings / input_local / projection key must exist.
        self._assert_audio_tower_weights_present(loader, mappings)
        loader.load_weights_from_safetensors(mappings)
        logger.info("MiMoV2_5Embedding weights loaded successfully!")

    @staticmethod
    def _assert_audio_tower_weights_present(loader, mappings: dict) -> None:
        audio_prefixes = ("speech_embeddings.", "audio_encoder.")
        missing = [
            hf_key
            for hf_key in mappings
            if hf_key.startswith(audio_prefixes) and not loader.has_weight_on_disk(hf_key)
        ]
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(
                "MiMo-V2.5 audio tower weights missing from checkpoint "
                f"({len(missing)} keys, e.g. {preview}). The HF key prefixes in "
                "weights_mapping.py do not match this checkpoint; loading would leave the "
                "audio tower randomly initialized. Reconcile weights_mapping.py with the "
                "real tensor names before serving audio."
            )

    # ---- per-modality encoders: return [N, hidden] features, or None if absent ----
    def _encode_audio(self, *, input_features, audio_feature_lengths, audio_codes):
        return self.audio_encoder(
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_codes=audio_codes,
        )

    def _encode_image(self, *, pixel_values, image_grid_thw):
        if pixel_values is None:
            return None
        if self.visual is None:
            raise NotImplementedError(
                "MiMo-V2.5 image tower (self.visual / MiMoVL ViT) is not wired yet "
                "in this round; see design §1.3 / mimo_v2_5/vision_encoder.py."
            )
        return self.visual(pixel_values.astype(self.dtype), image_grid_thw)

    def _encode_video(self, *, pixel_values_videos, video_grid_thw):
        if pixel_values_videos is None:
            return None
        if self.visual is None:
            raise NotImplementedError(
                "MiMo-V2.5 video tower (shared MiMoVL ViT) is not wired yet in this "
                "round; see design §4.2 / mimo_v2_5/vision_encoder.py."
            )
        return self.visual(pixel_values_videos.astype(self.dtype), video_grid_thw)

    @staticmethod
    def _scatter_modality(
        input_ids: jax.Array,
        input_embeds: jax.Array,
        modality_embeds: jax.Array | None,
        token_id: int | None,
    ) -> jax.Array:
        """Scatter ``[N, hidden]`` modality features into their placeholder rows.

        Generic over modality: the host-side contract (validated pre-JIT by
        ``EmbedModelRunner._validate_audio_placeholder_contract``) guarantees
        ``#placeholders == modality_embeds.shape[0]`` so each modality's disjoint
        token-id positions are filled independently. No-op when there are no features
        or no token id.

        Defense-in-depth for a contract violation that slips past the host guard
        (review R2-6): ``jnp.nonzero(size=N, fill_value=L)`` pads surplus slots with an
        out-of-range index ``L`` (= seq len), and the matching ``mode="drop"`` makes
        those writes no-ops instead of silently overwriting token 0. Likewise, padding
        rows beyond the real placeholder count never get written because their target
        index stays the drop sentinel.
        """
        if modality_embeds is None or token_id is None:
            return input_embeds
        seq_len = input_embeds.shape[0]
        mask = input_ids == token_id
        # fill_value = seq_len → an always-out-of-bounds row index; combined with
        # mode="drop" below, extra (unmatched) slots scatter nowhere instead of to row 0.
        positions = jnp.nonzero(mask, size=modality_embeds.shape[0], fill_value=seq_len)[0]
        return input_embeds.at[positions, :].set(modality_embeds, mode="drop")

    def __call__(
        self,
        input_ids: jax.Array,
        input_features=None,
        audio_feature_lengths=None,
        audio_codes=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
    ) -> EmbedOutput:
        input_embeds = self.text_embed_tokens(input_ids)

        audio_embeds = self._encode_audio(
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_codes=audio_codes,
        )
        image_embeds = self._encode_image(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        video_embeds = self._encode_video(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

        input_embeds = self._scatter_modality(
            input_ids, input_embeds, audio_embeds, self.audio_token_id
        )
        input_embeds = self._scatter_modality(
            input_ids, input_embeds, image_embeds, self.image_token_id
        )
        input_embeds = self._scatter_modality(
            input_ids, input_embeds, video_embeds, self.video_token_id
        )
        return EmbedOutput(input_embeds=input_embeds)
