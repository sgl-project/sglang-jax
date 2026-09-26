"""Top-level MiMoV2 multimodal model composition."""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.configs.mimo import config_value
from sgl_jax.srt.models.mimo_v2_audio import MiMoAudioEncoder
from sgl_jax.srt.models.mimo_v2_pro import MiMoV2ForCausalLM
from sgl_jax.srt.models.mimo_v2_vision import MiMoVisionTransformer
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.interface import (
    AudioCodeInputSpec,
    InModelMultimodalContract,
    VisionInputSpec,
)
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    pack_lanes,
    restore_encoder_output,
    run_mrope_vision_model,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import resolve_encoder_tp
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class MiMoV2ForConditionalGeneration(InModelMultimodalContract, MiMoV2ForCausalLM):
    """Wire the MiMoV2 vision + audio towers onto the in-model multimodal contract."""

    # Like SGLang, apply the checkpoint quantization only to the language model.
    unquantized_modules = ("visual", "audio_encoder")

    def __init__(self, config, mesh: Mesh | None = None, dtype: jnp.dtype = jnp.bfloat16):
        if mesh is None:
            raise ValueError("MiMoV2 multimodal models require a device mesh.")
        super().__init__(config, mesh, dtype)

        vision_config = config_value(config, "vision_config", None)
        audio_config = config_value(config, "audio_config", None)
        if vision_config is None and audio_config is None:
            raise ValueError("MiMoV2 VLM requires config.vision_config or config.audio_config.")

        self.encoder_tp = resolve_encoder_tp(mesh, getattr(config, "vision_encoder_parallel", "dp"))
        self.vision_tp = self.encoder_tp
        if vision_config is None:
            self.visual = None
        else:
            self.visual = MiMoVisionTransformer(
                vision_config,
                self.dtype,
                nnx.Rngs(0),
                mesh,
                self.encoder_tp,
            )
        self.audio_encoder = (
            MiMoAudioEncoder(audio_config, self.dtype, mesh, self.encoder_tp)
            if audio_config is not None
            else None
        )

        if self.visual is not None:
            self.vision_input_spec = VisionInputSpec(
                patch_dim=self.visual.patch_dim,
                spatial_merge_size=self.visual.spatial_merge_size,
            )
        if self.audio_encoder is not None:
            self.audio_input_spec = AudioCodeInputSpec(
                channels=self.audio_encoder.channels,
                group_size=self.audio_encoder.group_size,
            )

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.embed_tokens

    def get_image_feature(self, items_per_lane: list[list[MultimodalDataItem]]) -> jax.Array:
        visual = self.visual
        num_lanes = encoder_num_lanes(self.mesh, visual.vision_tp)
        return run_mrope_vision_model(
            visual,
            items_per_lane,
            mesh=self.mesh,
            num_lanes=num_lanes,
            merge_unit=visual.spatial_merge_unit,
            rope_type="rope_3d",
            input_sharding=visual.specs.sharding(visual.specs.batch_axis),
            output_sharding=visual.specs.sharding(),
        )

    def get_audio_feature(self, items_per_lane: list[list[MultimodalDataItem]]) -> jax.Array:
        encoder = self.audio_encoder
        num_lanes = encoder_num_lanes(self.mesh, encoder.encoder_tp)
        if len(items_per_lane) != num_lanes:
            raise ValueError("item lane count does not match the encoder topology")
        batch_sharding = encoder.specs.sharding(encoder.specs.batch_axis)
        codes, output_indices, _ = pack_lanes(
            items_per_lane,
            merge_unit=encoder.group_size,
            input_sharding=batch_sharding,
            dtype=np.int32,
        )
        valid = np.asarray(
            [sum(item.feature.shape[0] for item in lane) for lane in items_per_lane],
            dtype=np.int32,
        )
        with jax.set_mesh(self.mesh):
            output = encoder.encode(
                codes.reshape(num_lanes, -1, encoder.channels, out_sharding=batch_sharding),
                jax.device_put(valid, batch_sharding),
            )
            return restore_encoder_output(output, output_indices, encoder.specs.sharding())

    def get_multimodal_encode_funcs(self):
        funcs = {}
        if self.visual is not None:
            funcs[Modality.IMAGE] = self.get_image_feature
            funcs[Modality.MULTI_IMAGES] = self.get_image_feature
            funcs[Modality.VIDEO] = self.get_image_feature
        if self.audio_encoder is not None:
            funcs[Modality.AUDIO] = self.get_audio_feature
        return funcs

    def load_weights(self, model_config: ModelConfig) -> None:
        """Load the text backbone, then load each modality tower.

        The text loader replaces graph state, so towers must be detached during
        that step and restored even if loading fails.
        """
        visual = self.visual
        audio_encoder = self.audio_encoder
        del self.visual, self.audio_encoder
        try:
            super().load_weights(model_config)
        finally:
            self.visual, self.audio_encoder = visual, audio_encoder

        vision = config_value(self.config, "vision_config", None)
        heads = int(config_value(vision, "num_heads"))
        kv_heads = int(config_value(vision, "num_key_value_heads", heads) or heads)
        head_dim = int(config_value(vision, "qk_channels", 64))
        tower_config = SimpleNamespace(
            model_path=model_config.model_path,
            quantization_config=None,
            hf_config=self.config,
            hf_text_config=SimpleNamespace(head_dim=head_dim, v_head_dim=head_dim),
            num_attention_heads=heads,
            hidden_size=heads * head_dim,
            get_total_num_kv_heads=lambda: kv_heads,
            _dummy_mode=getattr(model_config, "_dummy_mode", False),
        )
        if self.visual is not None:
            loader = WeightLoader(self.visual, tower_config, self.mesh, self.dtype)
            with self.mesh:
                loader.load_weights_from_safetensors(self._vision_weight_mappings())
            logger.info("MiMoV2 vision tower weights loaded.")
        if self.audio_encoder is not None:
            audio_config = SimpleNamespace(
                model_path=model_config.model_path,
                quantization_config=None,
                _dummy_mode=getattr(model_config, "_dummy_mode", False),
            )
            loader = WeightLoader(self.audio_encoder, audio_config, self.mesh, self.dtype)
            with self.mesh:
                loader.load_weights_from_safetensors(self._audio_weight_mappings())
            logger.info("MiMoV2 audio tower weights loaded.")

    @staticmethod
    def _linear_mappings(
        source, target, sharding=(None, None), *, bias=True
    ) -> dict[str, WeightMapping]:
        mappings = {
            f"{source}.weight": WeightMapping(
                target_path=f"{target}.weight", sharding=sharding, transpose=True
            ),
        }
        if bias:
            mappings[f"{source}.bias"] = WeightMapping(
                target_path=f"{target}.bias", sharding=(sharding[-1],), transpose=False
            )
        return mappings

    def _vision_weight_mappings(self) -> dict[str, WeightMapping]:
        specs = self.visual.specs
        col, row = specs.col_kernel_axes, specs.row_kernel_axes
        mappings: dict[str, WeightMapping] = {
            "visual.patch_embed.proj.weight": WeightMapping(
                target_path="patch_embed.proj.kernel",
                sharding=(None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "visual.merger.ln_q.weight": WeightMapping(
                target_path="merger.ln_q.scale", sharding=(None,), transpose=False
            ),
            "visual.merger.mlp.0.weight": WeightMapping(
                target_path="merger.mlp_fc1.weight", sharding=col, transpose=True
            ),
            "visual.merger.mlp.2.weight": WeightMapping(
                target_path="merger.mlp_fc2.weight", sharding=row, transpose=True
            ),
        }
        for index, block in enumerate(self.visual.blocks):
            src = f"visual.blocks.{index}"
            tgt = f"blocks.{index}"
            for norm in ("norm1", "norm2"):
                mappings[f"{src}.{norm}.weight"] = WeightMapping(
                    target_path=f"{tgt}.{norm}.scale", sharding=(None,), transpose=False
                )
            mappings[f"{src}.attn.qkv.weight"] = WeightMapping(
                target_path=[f"{tgt}.attn.{n}_proj.weight" for n in ("q", "k", "v")],
                sharding=col,
                transpose=True,
            )
            mappings[f"{src}.attn.qkv.bias"] = WeightMapping(
                target_path=[f"{tgt}.attn.{n}_proj.bias" for n in ("q", "k", "v")],
                sharding=(col[-1],),
                transpose=False,
            )
            mappings.update(self._linear_mappings(f"{src}.attn.proj", f"{tgt}.attn.proj", row))
            for name in ("gate_proj", "up_proj"):
                mappings.update(
                    self._linear_mappings(f"{src}.mlp.{name}", f"{tgt}.mlp.{name}", col)
                )
            mappings.update(
                self._linear_mappings(f"{src}.mlp.down_proj", f"{tgt}.mlp.down_proj", row)
            )
            if block.attn.sinks is not None:
                mappings[f"{src}.attn.sinks"] = WeightMapping(
                    target_path=f"{tgt}.attn.sinks",
                    sharding=(specs.tensor_axis,),
                    transpose=False,
                )
        return mappings

    def _audio_weight_mappings(self) -> dict[str, WeightMapping]:
        encoder = self.audio_encoder
        mappings: dict[str, WeightMapping] = {}
        # Per-channel speech code embeddings live at top level in the checkpoint.
        for index in range(encoder.channels):
            mappings[f"speech_embeddings.{index}.weight"] = WeightMapping(
                target_path=f"speech_embeddings.{index}.embedding",
                sharding=(None, None),
                transpose=False,
            )
        src_root = "audio_encoder.input_local_transformer"
        tgt_root = "transformer"
        if encoder.transformer.norm is not None:
            mappings[f"{src_root}.norm.weight"] = WeightMapping(
                target_path=f"{tgt_root}.norm.scale", sharding=(None,), transpose=False
            )
        for index in range(len(encoder.transformer.layers)):
            src = f"{src_root}.layers.{index}"
            tgt = f"{tgt_root}.layers.{index}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                mappings[f"{src}.{norm}.weight"] = WeightMapping(
                    target_path=f"{tgt}.{norm}.scale", sharding=(None,), transpose=False
                )
            for name, bias in (
                ("self_attn.q_proj", True),
                ("self_attn.k_proj", True),
                ("self_attn.v_proj", True),
                ("self_attn.o_proj", False),
                ("mlp.gate_proj", False),
                ("mlp.up_proj", False),
                ("mlp.down_proj", False),
            ):
                mappings.update(self._linear_mappings(f"{src}.{name}", f"{tgt}.{name}", bias=bias))
        projections = (
            [("projection", "proj_fc1")]
            if encoder.proj_fc2 is None
            else [("projection.mlp.0", "proj_fc1"), ("projection.mlp.2", "proj_fc2")]
        )
        for source, target in projections:
            mappings.update(self._linear_mappings(f"audio_encoder.{source}", target, bias=False))
        return mappings


EntryClass = MiMoV2ForConditionalGeneration
