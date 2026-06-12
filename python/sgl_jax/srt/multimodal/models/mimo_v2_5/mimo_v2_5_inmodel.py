"""In-model MiMo-V2.5 (vision + RVQ-codes audio understanding, text-out) — refactor M4.

Mirrors the in-model template (qwen3_omni_inmodel.py / qwen2_5_vl.py): the towers are encoded via
embed_mm() + mm_core.merge() on the standard srt control plane -- no staged GlobalScheduler / embed
stage. Under C-1 (design §5.2) the encode runs ONCE per req on the host (model_runner.encode_mm_reqs)
and the result is sliced per chunk; there is no in-forward encode.

MiMo-V2.5 vs the Qwen3-Omni template:
  - NO deepstack (the MiMoVL ViT has a self-contained merger) -> embed_mm returns (fused, None, None).
  - AUDIO = RVQ DISCRETE codes (audio_kind="codes"), not continuous mel: encode_audio consumes an
    int `audio_codes` array via the RVQ codebook tower (per-channel Embed lookup + group transformer
    + projection), so the encode pass carries `mm_audio_codes` (int) rather than mm_audio_features.
  - FP8 MoE AR backbone (MiMoV2ForCausalLM): the AR self-loads + post-load-dequants its FP8 weights
    via its own load_weights; the vision/audio towers load separately as bf16 (G2-b: the quant path
    only touches model.layers.* / lm_head, never the towers).
  - V-4: the MiMoVL ViT attention was made jit-safe (segment-masked single pass; see vision_encoder).

Resolution: hf_config.architectures is ["MiMoV2ForCausalLM"] (the backbone). model_config.py remaps
it to "MiMoV2_5ForConditionalGeneration" when vision_config is present (multimodal checkpoint) and
not a draft model, so the standard ModelRegistry resolves THIS wrapper instead of the bare backbone.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.mm_core.merge import merge
from sgl_jax.srt.models.mimo_v2_pro import MiMoV2ForCausalLM
from sgl_jax.srt.multimodal.models.mimo_v2_5.audio_encoder import (
    MiMoV25AudioUnderstandingEncoder,
)
from sgl_jax.srt.multimodal.models.mimo_v2_5.config_utils import (
    get_config_value,
    normalize_vision_config,
)
from sgl_jax.srt.multimodal.models.mimo_v2_5.vision_encoder import MiMoVisionTransformer

logger = logging.getLogger(__name__)


class MiMoV2_5ForConditionalGeneration(nnx.Module):
    """MiMo-V2.5 image/video + RVQ-audio understanding, text-out, in-model on the standard scheduler."""

    audio_kind = "codes"  # RVQ discrete codes (vs Qwen3-Omni's continuous "features")
    has_deepstack = False

    def __init__(self, config=None, dtype=None, mesh=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.dtype = dtype or jnp.bfloat16
        rngs = nnx.Rngs(0)

        # AR body = the FP8 MoE backbone (embed -> AR -> logits; reads forward_batch.input_embedding
        # in extend mode, mimo_v2_flash.py:510-519). The top-level config carries the AR fields +
        # quantization_config (fp8); the backbone reads them.
        self.model = MiMoV2ForCausalLM(config, mesh=mesh, dtype=self.dtype)

        # RVQ-codes audio tower (HF `audio_encoder.*`).
        audio_config = getattr(config, "audio_config", config)
        self.audio_encoder = MiMoV25AudioUnderstandingEncoder(
            audio_config, mesh=mesh, dtype=self.dtype, rngs=rngs
        )
        # MiMoVL ViT shared by image + video (HF `visual.*`); normalize config (in_chans->in_channels,
        # qk_channels default 64) via the shared config helper to stay in sync with the checkpoint.
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            self.visual = MiMoVisionTransformer(
                normalize_vision_config(vision_config),
                dtype=self.dtype,
                rngs=rngs,
            )
        else:
            self.visual = None

        # Per-modality placeholder token ids (audio_token_id may live only in processor_config).
        self.audio_token_id = get_config_value(config, "audio_token_id", 151669)
        self.audio_token_id = int(self.audio_token_id) if self.audio_token_id is not None else None
        self.image_token_id = get_config_value(config, "image_token_id")
        self.video_token_id = get_config_value(config, "video_token_id")

    # ---- per-modality encoders (model-owned towers) ----

    def encode_image(self, pixel_values: jax.Array, grid_thw):
        if self.visual is None:
            raise NotImplementedError(
                "MiMo-V2.5 got vision input but checkpoint has no vision_config"
            )
        # grid_thw is a static tuple-of-(t,h,w) (the ViT iterates it / uses it as a static arg).
        grid = tuple(tuple(int(x) for x in row) for row in grid_thw)
        return self.visual(pixel_values.astype(self.dtype), grid)

    def encode_audio(self, audio_codes: jax.Array):
        """RVQ discrete codes -> [N_audio, hidden] via the codebook tower (input_features=None)."""
        return self.audio_encoder(
            input_features=None, audio_feature_lengths=None, audio_codes=audio_codes
        )

    def embed_mm(
        self,
        input_ids,
        mm_pixel_values=None,
        mm_grid_thw=None,
        mm_pixel_values_videos=None,
        mm_video_grid_thw=None,
        mm_audio_features=None,
        mm_audio_feature_lengths=None,
        mm_audio_codes=None,
        mm_real_llm_dims=None,
        mm_real_video_llm_dims=None,
    ):
        """C-1 (design §5.2): full-sequence text-embed + tower encode + merge. Returns the uniform
        ``(fused [seq, hidden], None, None)`` (MiMo has no deepstack). Scheme B: input_ids is clean;
        merge keys by the raw image/video/audio token id. (mm_audio_features / *_lengths accepted for
        signature uniformity with Qwen3-Omni but unused -- MiMo audio is discrete codes.)"""
        text_embed = self.model.model.embed_tokens(input_ids)
        mod_embeds = []
        # Order must match the modalities' appearance order in input_ids (merge scatters
        # concat(mod_embeds) into the sorted placeholder positions). Single-modality requests
        # (the common case) are always aligned regardless of order.
        if mm_audio_codes is not None:
            mod_embeds.append(self.encode_audio(mm_audio_codes))
        if mm_pixel_values is not None:
            mod_embeds.append(self.encode_image(mm_pixel_values, mm_grid_thw))
        if mm_pixel_values_videos is not None:
            mod_embeds.append(self.encode_image(mm_pixel_values_videos, mm_video_grid_thw))
        placeholder_ids = [
            t
            for t in (self.audio_token_id, self.image_token_id, self.video_token_id)
            if t is not None
        ]
        fused = merge(text_embed, mod_embeds, placeholder_ids, input_ids, mesh=self.mesh).embed
        return fused, None, None

    def __call__(self, forward_batch, memory_pools, logits_metadata):
        # AR-only (C-1): the fused embedding is produced once per req by the host encode pass and
        # sliced per chunk into forward_batch.input_embedding; the backbone reads it in extend mode.
        return self.model(forward_batch, memory_pools, logits_metadata)

    def load_weights(self, model_config):
        """Two-pass load: (1) the FP8 AR backbone self-loads + post-load-dequants in place; (2) the
        vision + audio towers load separately as bf16, fully replicated under the AR mesh (§3.3.5 /
        §5.7 G2-a). G2-b: the FP8 quant path lives entirely in the AR's load_weights and only targets
        model.layers.* / lm_head -- the towers (loaded here as plain .weight, no weight_q/scale) are
        never wrapped in QuantizedLinear."""
        import copy

        from sgl_jax.srt.mm_core.weights import assert_replicated, replicate_mappings
        from sgl_jax.srt.multimodal.models.mimo_v2_5.weights_mapping import (
            build_input_local_mapping,
            build_projection_mapping,
            build_speech_embeddings_mapping,
            create_mimo_vision_weight_mappings,
        )
        from sgl_jax.srt.utils.weight_utils import WeightLoader

        # (1) FP8 AR backbone (model.* / lm_head.*) -- self-managed load + dequant.
        self.model.load_weights(model_config)

        # (2) Towers (bf16, replicated). Skip build_text_embed_mapping -- the AR already provides
        # model.embed_tokens (text_embed reuses it in embed_mm). Audio mappings target audio_encoder.*
        # and vision visual.*, both of which are this wrapper's submodules.
        tower_mappings = {}
        tower_mappings.update(build_speech_embeddings_mapping(self.audio_encoder.audio_channels))
        tower_mappings.update(
            build_input_local_mapping(len(self.audio_encoder.input_local_transformer.layers))
        )
        tower_mappings.update(build_projection_mapping())
        if self.visual is not None:
            tower_mappings.update(
                create_mimo_vision_weight_mappings(
                    self.visual.config, source_prefix="visual", target_prefix="visual."
                )
            )
        tower_mappings = replicate_mappings(tower_mappings)
        assert_replicated(tower_mappings, where="MiMo-V2.5 vision + audio tower")

        # Strip the top-level fp8 quantization_config for the tower pass: the towers are bf16 and
        # leaving it set would make the loader treat .is_static_checkpoint on a plain dict / try fp8.
        tower_config = copy.copy(model_config)
        if getattr(tower_config, "quantization_config", None) is not None:
            tower_config.quantization_config = None
        loader = WeightLoader(
            model=self, model_config=tower_config, mesh=self.mesh, dtype=self.dtype
        )
        loader.load_weights_from_safetensors(tower_mappings)
        logger.info(
            "MiMo-V2.5 (in-model) weights loaded: AR (FP8) + %d tower mappings", len(tower_mappings)
        )

        # G2-b assertion (design §3.3.3): the towers must stay bf16 (no FP8/QuantizedLinear). The
        # two-pass load guarantees it structurally (the FP8 path is entirely inside the AR's
        # load_weights, scoped to self.model.layers.* / lm_head; the towers loaded above use plain
        # .weight mappings + a quant-stripped config). Assert it on the mappings + a representative
        # loaded weight to catch future drift, best-effort (never crash the load on introspection).
        self._assert_towers_bf16(tower_mappings)

    def _assert_towers_bf16(self, tower_mappings) -> None:
        # (1) No tower mapping pulls an FP8 weight_q / weight_scale source key.
        fp8_srcs = [k for k in tower_mappings if k.endswith("weight_q") or "weight_scale" in k]
        assert (
            not fp8_srcs
        ), f"G2-b violation: tower mapping references FP8 source keys: {fp8_srcs[:8]}"
        # (2) A representative tower weight loaded as a floating dtype (not an int FP8 container).
        try:
            kernel = self.visual.patch_embed.proj.kernel.value if self.visual is not None else None
            if kernel is not None and not jnp.issubdtype(kernel.dtype, jnp.floating):
                raise AssertionError(
                    f"G2-b violation: vision patch_embed kernel is {kernel.dtype} (expected float; "
                    "the FP8 quant path must not touch the towers)"
                )
        except AttributeError:
            logger.debug(
                "G2-b dtype check skipped (tower param path changed); mapping check passed"
            )


EntryClass = [MiMoV2_5ForConditionalGeneration]
