"""In-model Qwen3-Omni Thinker (understanding, text-out) — refactor M5.

Mirrors the Qwen2.5-VL in-model template (srt/multimodal/models/qwen2_5VL/qwen2_5_vl.py):
vision (and audio) are encoded via embed_mm() + mm_core.merge() on the standard srt control
plane -- no staged GlobalScheduler / embed stage. Under C-1 (design §5.2) the encode runs ONCE
per req on the host (model_runner.encode_mm_reqs) and the result is sliced per chunk; there is
no in-forward encode.

What's new vs Qwen2.5-VL (see tmp/refactor/m5-qwen3omni-map.md):
  - DEEPSTACK: the vision encoder emits 3 multi-scale feature levels; embed_mm returns them
    SPARSE ([num_levels, num_visual, hidden]) + a visual placeholder mask, which the encode pass
    attaches to req.deepstack_visual_embedding / deepstack_visual_pos_mask. ScheduleBatch.
    _merge_multimodal densifies them per chunk into forward_batch.deepstack_visual_embedding +
    apply_for_deepstack, and the AR body adds level i after layer i (Qwen3OmniMoeThinkerTextModel
    already does this, so no AR change). (Deepstack densify lives only here -- merge() does not
    densify; that avoids a second device-side implementation.)
  - self.model is the complete Qwen3OmniMoeThinkerTextForConditionalGeneration (embed -> AR ->
    logits, already reads input_embedding / deepstack / mrope), so the wrapper only stamps the
    ForwardBatch fields and passes through.
  - MoE AR body (launch with --moe-backend epmoe; moe_intermediate_size=768 crashes fused).

NOTE (validation status): structure mirrors the staged stage + the validated Qwen2.5-VL
template; construct(eval_shape)/weight-load/forward are validated incrementally on the TPU
dev pod. Audio (continuous-mel tower, audio_kind="features") is built for weight-load but the
forward audio path is a follow-up that needs ForwardBatch audio plumbing (mm_audio_features).
"""

from __future__ import annotations

import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.mm_core.merge import merge
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.audio_encoder import (
    Qwen3OmniMoeAudioEncoder,
)
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.qwen3_omni_thinker import (
    Qwen3OmniMoeThinkerTextForConditionalGeneration,
)
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.vision_encoder import (
    Qwen3OmniMoeVisionEncoder,
)
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.weights_mapping import (
    create_audio_tower_weight_mappings,
    create_visual_weight_mappings,
)

logger = logging.getLogger(__name__)


def _thinker_config(config):
    """The HF Qwen3-Omni config nests everything under thinker_config."""
    return getattr(config, "thinker_config", config)


class Qwen3OmniMoeForConditionalGeneration(nnx.Module):
    """Image/video (+audio) understanding, text-out, in-model on the standard scheduler.

    Named to match hf_config.architectures (['Qwen3OmniMoeForConditionalGeneration']) so the
    standard ModelRegistry resolves it. Serves the Thinker (understanding / text-out) path;
    the Talker (speech-out) is generation-plane and out of scope here.
    """

    audio_kind = "features"  # continuous mel (vs MiMo's "codes")
    has_deepstack = True

    def __init__(self, config=None, dtype=None, mesh=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.dtype = dtype or jnp.bfloat16
        thinker = _thinker_config(config)
        self.thinker_config = thinker

        # The Qwen3-Omni vision/audio towers' norm layers call rngs.params() unconditionally
        # (unlike the Qwen2.5-VL ViT which tolerates None), so pass a real Rngs. eval_shape
        # abstracts the actual init; nnx.Rngs(0) is the same fallback the as-is towers use.
        rngs = nnx.Rngs(0)
        # Vision tower -> {pooler_output, deepstack_features: [tuple of per-level [N, hidden]]}.
        self.visual = Qwen3OmniMoeVisionEncoder(
            thinker.vision_config, mesh=mesh, dtype=self.dtype, rngs=rngs
        )
        # Audio tower (continuous mel). Built for weight-load; forward audio path is a follow-up.
        self.audio_tower = Qwen3OmniMoeAudioEncoder(
            thinker.audio_config, mesh=mesh, dtype=self.dtype, rngs=rngs
        )
        # AR body = the complete Thinker ForCausalLM (embed -> AR -> logits; reads
        # input_embedding / deepstack_visual_embedding / apply_for_deepstack / mrope).
        self.model = Qwen3OmniMoeThinkerTextForConditionalGeneration(
            thinker.text_config, mesh=mesh, dtype=self.dtype
        )

        self.image_token_id = getattr(thinker, "image_token_id", None)
        self.video_token_id = getattr(thinker, "video_token_id", None)
        self.audio_token_id = getattr(thinker, "audio_token_id", None)

    # ---- per-modality encoders (model-owned towers) ----

    def encode_image(self, pixel_values: jax.Array, grid_thw):
        """[total_patches, in_dim] + grid_thw -> (pooler [N, hidden], [per-level [N, hidden]]).

        The Qwen3-Omni ViT indexes grid_thw as a 2D array (grid_thw[:, 1]); ForwardBatch carries
        it as a static tuple-of-tuples, so materialize a concrete [num_images, 3] np array.
        """
        out = self.visual(pixel_values.astype(self.dtype), np.asarray(grid_thw))
        return out["pooler_output"], list(out["deepstack_features"])

    def encode_audio(self, input_features: jax.Array, feature_lens):
        """Continuous-mel audio -> [N_audio, hidden]. Follow-up: needs ForwardBatch audio fields."""
        return self.audio_tower(input_features, feature_lens)

    def embed_mm(
        self,
        input_ids,
        mm_pixel_values=None,
        mm_grid_thw=None,
        mm_pixel_values_videos=None,
        mm_video_grid_thw=None,
        mm_audio_features=None,
        mm_audio_feature_lengths=None,
    ):
        """C-1 (design §5.2): full-sequence encode for the host encode pass. Returns the uniform
        tuple ``(fused [seq, hidden], deepstack_sparse [num_levels, num_visual, hidden] or None,
        visual_pos_mask [seq] bool or None)``. The deepstack is returned SPARSE (the per-level
        visual features, NOT densified) plus a full-prompt visual placeholder mask, exactly the
        format ScheduleBatch._merge_multimodal densifies per chunk via req.deepstack_visual_embedding
        + req.deepstack_visual_pos_mask. Scheme B: input_ids clean; merge keys by token id."""
        text_embed = self.model.model.embed_tokens(input_ids)
        mod_embeds = []
        multiscale = None
        if mm_pixel_values is not None:
            pool, ds_levels = self.encode_image(mm_pixel_values, mm_grid_thw)
            mod_embeds.append(pool)
            multiscale = jnp.stack(ds_levels, axis=0)
        if mm_pixel_values_videos is not None:
            vpool, vds = self.encode_image(mm_pixel_values_videos, mm_video_grid_thw)
            mod_embeds.append(vpool)
            vstack = jnp.stack(vds, axis=0)
            multiscale = (
                vstack if multiscale is None else jnp.concatenate([multiscale, vstack], axis=1)
            )
        if mm_audio_features is not None:
            mod_embeds.append(
                self.encode_audio(mm_audio_features, np.asarray(mm_audio_feature_lengths))
            )
        placeholder_ids = [
            t
            for t in (self.image_token_id, self.video_token_id, self.audio_token_id)
            if t is not None
        ]
        fused = merge(text_embed, mod_embeds, placeholder_ids, input_ids, mesh=self.mesh).embed
        deepstack = visual_pos_mask = None
        if multiscale is not None:
            # A12(5) guard: the AR injects deepstack level i after layer i, deriving the level
            # count from this tensor's leading dim. Assert it matches the configured target layers
            # so an encoder/config drift fails loudly instead of silently mis-injecting.
            expected_levels = len(self.thinker_config.vision_config.deepstack_visual_indexes)
            assert multiscale.shape[0] == expected_levels, (
                f"deepstack level count mismatch: encoder emitted {multiscale.shape[0]} levels, "
                f"vision_config.deepstack_visual_indexes has {expected_levels}"
            )
            visual_ids = jnp.asarray(
                [t for t in (self.image_token_id, self.video_token_id) if t is not None],
                dtype=input_ids.dtype,
            )
            visual_pos_mask = jnp.isin(input_ids, visual_ids)
            deepstack = multiscale  # [num_levels, num_visual, hidden], sparse
        return fused, deepstack, visual_pos_mask

    def __call__(self, forward_batch, memory_pools, logits_metadata):
        # In-model multimodal (C-1, design §5.2): the fused embedding AND the sparse deepstack
        # (+ visual pos mask) are produced once per req by the host-side encode pass
        # (model_runner.encode_mm_reqs -> embed_mm) and held on req.{multimodal_embedding,
        # deepstack_visual_embedding, deepstack_visual_pos_mask}; ScheduleBatch._merge_multimodal
        # slices/densifies them per chunk into forward_batch.{input_embedding,
        # deepstack_visual_embedding}, which the AR reads. There is NO in-forward encode here --
        # the earlier per-chunk in-forward path re-encoded every chunk and stamped a dense
        # chunk-keyed deepstack that did NOT slice the chunk window (B1/B2 misalignment for both
        # the embed and the deepstack); C-1 supersedes it. embed_mm above is the single encode
        # source, invoked only by the encode pass.
        return self.model(forward_batch, memory_pools, logits_metadata)

    def load_weights(self, model_config):
        """Compose the 3 towers' weight mappings: visual.* + audio_tower.* (reused from the
        staged stage, prefixes already match) + the AR ForCausalLM's mappings under `model.`.
        """
        from sgl_jax.srt.mm_core.weights import assert_replicated, replicate_mappings
        from sgl_jax.srt.utils.weight_utils import WeightLoader

        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = {}

        # The towers run fully replicated under the multi-chip AR mesh (design §3.3.5 / §5.7
        # G2-a), so load their weights replicated via the CORE replicate_mappings() helper
        # (overrides any TP sharding to all-None). With replicated kernels + the ViT's
        # replicated out_sharding, the whole vision/audio compute stays replicated -- no
        # mid-tower reshards. assert_replicated() guards against a tower mapping that slips
        # through with a TP axis. The staged path loaded these on a 1-device mesh where the
        # original sharding was already trivial.
        tower_mappings = replicate_mappings(
            {
                **create_visual_weight_mappings(self.thinker_config.vision_config),
                **create_audio_tower_weight_mappings(self.thinker_config.audio_config),
            }
        )
        assert_replicated(tower_mappings, where="Qwen3-Omni visual + audio tower")
        mappings.update(tower_mappings)
        # AR mappings target paths relative to the ForCausalLM (model.* / lm_head.*); from this
        # wrapper self.model IS that module, so prepend "model.". For MoE mappings the loader
        # treats target_path as [model_target, *source_hf_keys] (weight_utils uses
        # target_path[1:] as the checkpoint keys), so prefix ONLY index 0 -- the source keys
        # must stay as-is. Plain str targets are prefixed whole.
        for src, m in self.model._create_qwen3_omni_moe_weight_mappings().items():
            tp = m.target_path
            new_tp = "model." + tp if isinstance(tp, str) else type(tp)(["model." + tp[0], *tp[1:]])
            mappings[src] = dataclasses.replace(m, target_path=new_tp)

        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(mappings)
        else:
            loader.load_weights_from_safetensors(mappings)
        logger.info("Qwen3-Omni Thinker (in-model) weights loaded: %d mappings", len(mappings))


EntryClass = [Qwen3OmniMoeForConditionalGeneration]
