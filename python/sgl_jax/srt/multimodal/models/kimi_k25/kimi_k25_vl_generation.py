from __future__ import annotations

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3ForCausalLM
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import KimiK25ModelVitConfig
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    run_mrope_vision_model,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import resolve_encoder_tp
from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vit import (
    Kimi_K25_VisionModel,
    create_kimi_vision_weight_mappings,
)
from sgl_jax.srt.utils.common_utils import resolve_vision_patch_buckets
from sgl_jax.srt.utils.weight_utils import WeightLoader

logger = logging.getLogger(__name__)


class KimiK25ForConditionalGeneration(DeepseekV3ForCausalLM, InModelMultimodalContract):
    """Kimi-K2.5 on the in-model multimodal path.

    The vision tower lives inside this class as self.visual. The engine calls
    get_input_embeddings() to embed text and get_multimodal_encode_funcs() to
    encode media, then merges the two itself.
    """

    def __init__(
        self,
        config=None,
        dtype=None,
        mesh=None,
        rngs: nnx.Rngs | None = None,
    ):
        self.text_config = get_hf_text_config(config) or config
        # The server policy lives on the outer config; the base model reads the text config.
        self.text_config.enable_dp_lm_head = getattr(config, "enable_dp_lm_head", False)
        self.dtype = dtype or jnp.bfloat16
        self.mesh = mesh

        # text_config.quantization_config is a raw dict from the JSON.
        # ModelConfig already handles quantization at the top-level hf_config, but
        # due to quantization config being nested in text_config, it still in JSON
        # Clear it here so FusedMoE doesn't receive a raw dict as the config contains 'pack-quantized'
        # format which isn't yet supported.
        if isinstance(getattr(self.text_config, "quantization_config", None), dict):
            self.text_config.quantization_config = None

        super().__init__(
            config=self.text_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        self.hf_weight_prefix = "language_model."

        self.vision_config = KimiK25ModelVitConfig()
        # Head-parallel encoder. Lane packing makes the data-parallel mode
        # representable too, but it is left off until validated on hardware.
        # resolve_encoder_tp returns False on meshes without a usable tensor
        # axis, where the tower simply stays replicated.
        vision_tp = resolve_encoder_tp(self.mesh, "tp") if self.mesh is not None else False
        self.visual = Kimi_K25_VisionModel(
            self.vision_config,
            dtype=self.dtype,
            rngs=rngs or nnx.Rngs(0),
            mesh=self.mesh,
            vision_tp=vision_tp,
        )
        # Lane-packing compile buckets. _bucket_capacity silently skips buckets
        # that are not a multiple of the merge unit, so filter them out here to
        # keep the effective bucket list explicit.
        merge_unit = self.visual.merge_unit
        self.vision_buckets = tuple(
            bucket for bucket in resolve_vision_patch_buckets(None) if bucket % merge_unit == 0
        )

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.embed_tokens

    def get_multimodal_encode_funcs(self):
        # Kimi's processor reports images and video chunks in one grid_thws
        # tensor and routes both through the same tower, so every visual
        # modality maps to the same encoder. IMAGE is what the processor tags
        # today; the other two are registered so a future tagging change cannot
        # silently drop embeddings.
        return {
            Modality.IMAGE: self.encode_vision_items,
            Modality.MULTI_IMAGES: self.encode_vision_items,
            Modality.VIDEO: self.encode_vision_items,
        }

    def encode_vision_items(self, items: list[MultimodalDataItem]) -> jax.Array:
        """Encode media items into one item-ordered [tokens, hidden] array.

        Runs through the shared lane-packing orchestrator, which balances items
        over the encoder lanes, pads to a compile bucket, runs the tower, and
        restores item order. ``sd2_tpool`` pools away the temporal axis,
        so each item emits h*w/merge_unit tokens regardless of its frame count.
        """
        if not items:
            return jnp.zeros((0, self.vision_config.text_hidden_size), dtype=self.dtype)

        specs = self.visual.vision_tower.specs
        return run_mrope_vision_model(
            self.visual,
            items,
            mesh=self.mesh,
            num_lanes=encoder_num_lanes(self.mesh, self.visual.vision_tower.vision_tp),
            buckets=self.vision_buckets,
            merge_unit=self.visual.merge_unit,
            rope_type="rope_2d",
            pool_temporal_dimension=True,
            input_sharding=specs.sharding(specs.batch_axis),
            output_sharding=specs.sharding(),
        )

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings(model_config)

        weight_mappings.update(
            create_kimi_vision_weight_mappings(
                self.vision_config.vt_num_hidden_layers,
                target_prefix="visual.",
            )
        )
        loader.load_weights_from_safetensors(weight_mappings)

        for layer in self.model.layers:
            layer.self_attn.post_load_weights()
        logger.info("Kimi K2.5 language model and vision tower weights loaded successfully!")


EntryClass = KimiK25ForConditionalGeneration
