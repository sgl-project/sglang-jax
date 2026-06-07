from __future__ import annotations

import inspect
import math

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoConfig

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.mm_assembly import assemble_mm_inputs
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
        if hasattr(self.model_class, "get_embed_model_config"):
            model_config = self.model_class.get_embed_model_config(model_config)
        elif hasattr(model_config, "thinker_config"):
            model_config = model_config.thinker_config
        self.model_config = model_config
        self.model_config.revision = None
        self.model_config.dtype = jnp.bfloat16
        self.model_config.model_path = self.server_args.model_path
        self.model_config.model_class = self.model_class
        self.model = self.model_loader.load_model(
            model_config=self.model_config,
        )
        self.embed_input_keys = self._get_embed_input_keys()

    def _get_embed_input_keys(self) -> tuple[str, ...]:
        if hasattr(self.model_class, "get_embed_input_keys"):
            return tuple(self.model_class.get_embed_input_keys())
        signature = inspect.signature(self.model.__call__)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        ):
            return (
                "input_ids",
                "input_features",
                "audio_feature_lengths",
                "audio_codes",
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            )
        return tuple(signature.parameters)

    @staticmethod
    def _validate_audio_placeholder_contract(input_ids, omni_inputs, audio_codes) -> None:
        if input_ids is None or audio_codes is None or not isinstance(omni_inputs, dict):
            return

        audio_token_id = omni_inputs.get("audio_token_id")
        if audio_token_id is None:
            # Audio codes present but no scatter target: the embedding model would
            # silently drop the audio (scatter is a no-op when token_id is None), so
            # the audio would never reach the AR stage. Fail loudly (review R2-6).
            raise ValueError(
                "MiMo-V2.5 received audio codes but omni_inputs has no audio_token_id; "
                "cannot place audio embeddings. Check that the model config / processor "
                "resolves audio_token_id (expected 151669)."
            )

        items = omni_inputs.get("mm_items", []) or []
        expected = 0
        for raw_item in items:
            if isinstance(raw_item, dict):
                modality = raw_item.get("modality")
                modality_name = str(modality).lower()
                is_audio = modality_name == "audio" or modality_name.endswith("audio")
                meta = raw_item.get("model_specific_data") or {}
                feature = raw_item.get("feature")
            else:
                is_audio = raw_item.is_audio() if hasattr(raw_item, "is_audio") else False
                meta = getattr(raw_item, "model_specific_data", None) or {}
                feature = getattr(raw_item, "feature", None)
            if not is_audio:
                continue
            if not (meta.get("is_codes") or "token_lengths" in meta):
                continue
            token_lengths = meta.get("token_lengths")
            if token_lengths is not None:
                expected += sum(int(length) for length in token_lengths)
                continue
            group_size = int(meta.get("group_size") or 1)
            if feature is None:
                continue
            shape = getattr(feature, "shape", ())
            if len(shape) == 2:
                expected += math.ceil(int(shape[0]) / group_size)

        if expected == 0:
            return

        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        actual = sum(1 for token in ids if int(token) == int(audio_token_id))
        if actual != expected:
            raise ValueError(
                "MiMo-V2.5 audio placeholder count mismatch before embed stage: "
                f"found {actual} audio pad tokens, expected {expected} from mm_items "
                f"token_lengths. input_ids_len={len(ids)} audio_codes_shape="
                f"{getattr(audio_codes, 'shape', None)}"
            )

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        def forward_model(
            model_def,
            model_state_def,
            model_state_leaves,
            embed_input_keys,
            input_ids,
            input_features,
            audio_feature_lengths,
            audio_codes,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            inputs = {
                "input_ids": input_ids,
                "input_features": input_features,
                "audio_feature_lengths": audio_feature_lengths,
                "audio_codes": audio_codes,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
            }
            return model(**{key: inputs[key] for key in embed_input_keys})

        def forward_wrapper(
            input_ids: jax.Array,
            input_features: jax.Array | None = None,
            audio_feature_lengths: jax.Array | None = None,
            audio_codes: jax.Array | None = None,
            pixel_values: jax.Array | None = None,
            pixel_values_videos: jax.Array | None = None,
            image_grid_thw: jax.Array | None = None,
            video_grid_thw: jax.Array | None = None,
        ):
            return forward_model(
                model_def,
                model_state_def,
                model_state_leaves,
                self.embed_input_keys,
                input_ids,
                input_features,
                audio_feature_lengths,
                audio_codes,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
            )

        self.jitted_embedding = forward_wrapper

    def _prepare_input(self, batch: Req):
        """Assemble jitted_embedding inputs from the request's mm_items (P1).

        `mm_items` is the single source of truth for multimodal features; the shared
        assembler turns them into per-modality kwargs. No model-specific logic here —
        audio is routed to discrete `audio_codes` or continuous `input_features` by a
        generic per-item flag, and each model owns its own forward.
        """
        input_ids = batch.input_ids or batch.origin_input_ids
        input_ids = jnp.asarray(input_ids) if input_ids is not None else None

        omni_inputs = batch.omni_inputs if isinstance(batch.omni_inputs, dict) else None
        assembled = assemble_mm_inputs(omni_inputs)

        def _arr(value):
            return jnp.asarray(value) if value is not None else None

        audio_codes = _arr(assembled["audio_codes"])
        audio_features = _arr(assembled["audio_features"])
        self._validate_audio_placeholder_contract(input_ids, omni_inputs, audio_codes)
        audio_feature_lengths = None
        # Continuous-audio models (e.g. Qwen3-Omni mel) densify via feature_attention_mask;
        # discrete-codes audio sets audio_features=None and skips this generically.
        feature_attention_mask = assembled["audio_feature_attention_mask"]
        if audio_features is not None and feature_attention_mask is not None:
            feature_attention_mask = jnp.asarray(feature_attention_mask)
            audio_feature_lengths = jnp.asarray(feature_attention_mask.sum(axis=1))
            audio_features = audio_features.transpose(0, 2, 1)[
                feature_attention_mask.astype(jnp.bool)
            ].transpose(1, 0)

        pixel_values = _arr(assembled["pixel_values_images"])
        pixel_values_videos = _arr(assembled["pixel_values_videos"])
        image_grid_thw = (
            jnp.array(assembled["image_grid_thw"])
            if assembled["image_grid_thw"] is not None
            else None
        )
        video_grid_thw = (
            jnp.array(assembled["video_grid_thw"])
            if assembled["video_grid_thw"] is not None
            else None
        )

        return {
            "input_ids": input_ids,
            "input_features": audio_features,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_codes": audio_codes,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
        }

    def forward(self, batch: Req, mesh: jax.sharding.Mesh):
        # Prepare inputs
        inputs = self._prepare_input(batch)

        # Call jitted_embedding
        output = self.jitted_embedding(**inputs)
        input_embeds = getattr(output, "input_embeds", None)
        visual_embeds_multiscale = getattr(output, "deepstack_embeds", None)
        visual_pos_masks = getattr(output, "deepstack_pos_mask", None)
        if input_embeds is None:
            input_embeds, visual_embeds_multiscale, visual_pos_masks = output

        mm_inputs = batch.omni_inputs if isinstance(batch.omni_inputs, dict) else None
        if mm_inputs is not None:
            mm_inputs["multimodal_embedding"] = input_embeds
            if visual_embeds_multiscale is None:
                mm_inputs["deepstack_visual_embedding"] = jnp.zeros((3, 1, input_embeds.shape[-1]))
                mm_inputs["deepstack_visual_pos_mask"] = jnp.array([1], dtype=jnp.int32)
            else:
                mm_inputs["deepstack_visual_embedding"] = jnp.array(visual_embeds_multiscale)
                mm_inputs["deepstack_visual_pos_mask"] = visual_pos_masks
        return batch
