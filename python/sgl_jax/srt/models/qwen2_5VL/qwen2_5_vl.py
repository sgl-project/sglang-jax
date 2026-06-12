"""Qwen2.5-VL in-model (refactor M3).

A single model registered on the standard srt LLM runtime (no staged ViT/AR pipeline).
Its multimodal embedding (encode + merge) runs ONCE PER REQUEST on the host encode pass
(C-1, ``ModelRunner.encode_mm_reqs`` -> ``embed_mm``), NOT inside the forward; the AR forward
then reads the prepared ``input_embedding``:

    text_embed = self.model.embed_tokens(input_ids)
    mod_embeds = encode_image/encode_video(self.visual, pixel_values, grid_thw)
    fused = mm_core.merge(text_embed, mod_embeds, pad_values, input_ids)   # single mask-scatter
    forward_batch.input_embedding = fused                                  # AR body reads this
    -> self.model(forward_batch) -> logits

Reuses the existing ViT tower (`Qwen2_5_VL_VisionTransformer`) and the AR LLM body
(`Qwen2_5_VL_Model`), replacing the staged `vit_model_runner` cumsum-gather with merge().
See design doc §3.3 and tmp/refactor/m3-plan.md.

NOTE (validation status): import / construct(eval_shape) / weight-load are validated on
the TPU dev pod. The vision config is derived from the checkpoint's vision_config via
qwen_vl_vision_config_from_hf (hidden_size=1280 for 7B; the bare QwenVLModelVitConfig()
default 3584 is the post-merger LLM dim and breaks the patch_embed reshape). ViT
precompile/HBM behaviour (m3-plan risk 1) is observed via the encode+merge forward smoke.
"""

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.configs.qwen_vl.config_helpers import qwen_vl_vision_config_from_hf
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.mm_core.merge import merge
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2_5VL.qwen2_5_vit import Qwen2_5_VL_VisionTransformer
from sgl_jax.srt.models.qwen2_5VL.qwen2_5_vl_generation import Qwen2_5_VL_Model
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class Qwen2_5_VLForConditionalGeneration(nnx.Module):
    """In-model Qwen2.5-VL: owns the ViT + LLM and merges in-forward (refactor M3)."""

    # Multimodal capability declarations (mm_core.capability / U3).
    audio_kind = None
    has_deepstack = False

    def __init__(self, config=None, dtype=None, mesh=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16

        # ViT tower (reused as-is). Vision config is derived from the checkpoint's
        # vision_config (hidden_size=1280 for 7B); the bare QwenVLModelVitConfig() default
        # (hidden_size=3584) is the post-merger LLM dim and breaks the patch_embed reshape.
        self.vision_config = qwen_vl_vision_config_from_hf(config)
        self.visual = Qwen2_5_VL_VisionTransformer(
            config=self.vision_config,
            dtype=self.dtype,
            rngs=None,
            mesh=mesh,
            norm_eps=getattr(self.vision_config, "rms_norm_eps", 1e-6),
        )

        # AR LLM body (reused as-is; reads forward_batch.input_embedding).
        self.model = Qwen2_5_VL_Model(self.text_config, mesh=mesh, dtype=self.dtype)
        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=mesh)

        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)
        self.spatial_merge_size = getattr(self.vision_config, "spatial_merge_size", 2)

    # ---- per-modality encoders (model-owned tower; capability U3) ----

    def _encode(
        self, pixel_values: jax.Array, grid_thw: tuple, real_llm_dims: jax.Array | None = None
    ):
        """Encode one modality's pixels, one ViT call per item. Returns ``(hidden [N, out], valid)``
        where ``valid`` is None unless V-2 bucketing is active. When ``real_llm_dims`` (traced
        [num_items, 2] of real (llm_h, llm_w)) is given, grid_thw/pixel_values are padded to a
        canonical bucket; the ViT masks the bucket padding and returns it + a per-unit valid mask
        (the caller compacts the padding out before merge). The compile keys only on the canonical
        (padded) grid, never on the real size -> bounded recompiles."""
        embeds = []
        valids = []
        cur = 0
        for idx, thw in enumerate(grid_thw):
            t, h, w = thw
            size = int(t) * int(h) * int(w)
            px = pixel_values[cur : cur + size, :]
            cur += size
            if real_llm_dims is not None:
                hidden, valid = self.visual.encode_bucketed(
                    px, (thw,), real_llm_dims[idx : idx + 1]
                )
                embeds.append(hidden)
                valids.append(valid)
            else:
                embeds.append(self.visual(px, (thw,)))
        hidden = jnp.concatenate(embeds, axis=0)
        valid = jnp.concatenate(valids, axis=0) if real_llm_dims is not None else None
        return hidden, valid

    def encode_image(
        self, pixel_values: jax.Array, grid_thw: tuple, real_llm_dims: jax.Array | None = None
    ):
        return self._encode(pixel_values, grid_thw, real_llm_dims)

    def encode_video(
        self, pixel_values: jax.Array, grid_thw: tuple, real_llm_dims: jax.Array | None = None
    ):
        return self._encode(pixel_values, grid_thw, real_llm_dims)

    # ---- embed_mm: host-side once-per-req encode + merge (C-1); AR body reuse ----

    def embed_mm(
        self,
        input_ids,
        mm_pixel_values=None,
        mm_grid_thw=None,
        mm_pixel_values_videos=None,
        mm_video_grid_thw=None,
        mm_audio_features=None,
        mm_audio_feature_lengths=None,
        mm_audio_codes=None,  # uniform embed_mm contract; Qwen2.5-VL has no audio -> always None
        mm_real_llm_dims=None,
        mm_real_video_llm_dims=None,
    ):
        """Full-sequence text-embed + ViT encode + merge (C-1, design §5.2). Returns the uniform
        encode-pass tuple ``(fused [seq, hidden], deepstack_sparse_or_None, visual_pos_mask_or_None)``
        (Qwen2.5-VL has no deepstack -> last two are None). The single source of truth for the
        in-model encode+merge: called once-per-req by the host-side encode pass
        (model_runner.encode_mm_reqs) -- NOT by ``__call__`` -- which runs it over the FULL
        input_ids+pixels and
        holds the result on ``req.multimodal_embedding`` so the scheduler slices it per chunk -- no
        per-chunk re-encode, no chunk-boundary merge misalignment (B1/B2/B8). Scheme B: input_ids
        is clean; merge keys by the raw image/video token id. (mm_audio_* accepted for a uniform
        signature with the audio models; unused here.) mm_real_llm_dims / mm_real_video_llm_dims
        carry the *traced* real (llm_h, llm_w) per item when V-2 bucketing pads pixels to a
        canonical bucket (None = off); the padded visual rows are then compacted valid-to-front
        so merge fills the real placeholders and drops the trailing bucket padding."""
        text_embed = self.model.embed_tokens(input_ids)
        visuals = []
        valids = []
        if mm_pixel_values is not None:
            hidden, valid = self.encode_image(
                mm_pixel_values, mm_grid_thw, real_llm_dims=mm_real_llm_dims
            )
            visuals.append(hidden)
            valids.append(valid)
        if mm_pixel_values_videos is not None:
            hidden, valid = self.encode_video(
                mm_pixel_values_videos, mm_video_grid_thw, real_llm_dims=mm_real_video_llm_dims
            )
            visuals.append(hidden)
            valids.append(valid)
        mod_embeds = []
        if visuals:
            all_v = jnp.concatenate(visuals, axis=0)
            # V-2 bucketing: if any modality was padded to a bucket, compact the real rows to the
            # front (stable) so merge -- whose scatter is sized to the padded row count and uses
            # mode="drop" -- fills the real placeholders in order and discards the trailing bucket
            # padding. Off (no real dims): all-True masks -> identity, byte-equivalent.
            if any(v is not None for v in valids):
                masks = [
                    v if v is not None else jnp.ones((h.shape[0],), dtype=bool)
                    for h, v in zip(visuals, valids)
                ]
                all_m = jnp.concatenate(masks, axis=0)
                n = all_m.shape[0]
                order = jnp.argsort(jnp.where(all_m, jnp.arange(n), n + jnp.arange(n)))
                all_v = all_v[order]
            mod_embeds = [all_v]
        placeholder_ids = [t for t in (self.image_token_id, self.video_token_id) if t is not None]
        fused = merge(text_embed, mod_embeds, placeholder_ids, input_ids, mesh=self.mesh).embed
        return fused, None, None

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        # In-model multimodal (C-1, design §5.2): the fused embedding is produced once per req by
        # the host-side encode pass (model_runner.encode_mm_reqs -> embed_mm) and sliced per chunk
        # into forward_batch.input_embedding by ScheduleBatch._merge_multimodal; the AR body reads
        # it in extend mode. There is NO in-forward encode here -- the earlier per-chunk in-forward
        # path re-encoded every chunk and misaligned the merge at chunk boundaries (B1/B2),
        # superseded by C-1's single full-sequence encode + per-chunk slice. (embed_mm above is
        # the single encode source, invoked by the encode pass.)
        token_to_kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool
        )
        if not getattr(self.text_config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None

    # ---- weight loading: merge ViT (visual.*) + LLM (model.* / lm_head) mappings ----

    def load_weights(self, model_config: ModelConfig):
        from sgl_jax.srt.mm_core.weights import assert_replicated, replicate_mappings

        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = {}
        # ViT runs fully replicated under the AR mesh (design §3.3.5 / §5.7 G2-a): force the
        # visual.* mappings replicated via the CORE helper and assert it, so a future mapping
        # that forgets all-None can't silently TP-shard the tower.
        vit_mappings = replicate_mappings(self._vit_weight_mappings())
        assert_replicated(vit_mappings, where="Qwen2.5-VL visual tower")
        mappings.update(vit_mappings)
        mappings.update(self._llm_weight_mappings())
        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(mappings)
        else:
            loader.load_weights_from_safetensors(mappings)
        logger.info("Qwen2.5-VL (in-model) weights loaded: %d mappings", len(mappings))

    def _vit_weight_mappings(self) -> dict:
        # visual.* keys from the HF checkpoint -> self.visual.* (replicated; no TP).
        m = {
            "visual.patch_embed.proj.weight": WeightMapping(
                target_path="visual.patch_embed.proj.kernel",
                sharding=(None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "visual.merger.ln_q.weight": WeightMapping(
                target_path="visual.merger.ln_q.scale", sharding=(None,), transpose=False
            ),
            "visual.merger.mlp.0.weight": WeightMapping(
                target_path="visual.merger.mlp_fc1.kernel", sharding=(None, None), transpose=True
            ),
            "visual.merger.mlp.0.bias": WeightMapping(
                target_path="visual.merger.mlp_fc1.bias", sharding=(None,), transpose=False
            ),
            "visual.merger.mlp.2.weight": WeightMapping(
                target_path="visual.merger.mlp_fc2.kernel", sharding=(None, None), transpose=True
            ),
            "visual.merger.mlp.2.bias": WeightMapping(
                target_path="visual.merger.mlp_fc2.bias", sharding=(None,), transpose=False
            ),
        }
        for i in range(getattr(self.vision_config, "depth", 0)):
            p = f"visual.blocks.{i}"
            m.update(
                {
                    f"{p}.norm1.weight": WeightMapping(
                        target_path=f"{p}.norm1.scale", sharding=(None,), transpose=False
                    ),
                    f"{p}.norm2.weight": WeightMapping(
                        target_path=f"{p}.norm2.scale", sharding=(None,), transpose=False
                    ),
                    f"{p}.attn.qkv.weight": WeightMapping(
                        target_path=f"{p}.attn.qkv_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{p}.attn.qkv.bias": WeightMapping(
                        target_path=f"{p}.attn.qkv_proj.bias", sharding=(None,), transpose=False
                    ),
                    f"{p}.attn.proj.weight": WeightMapping(
                        target_path=f"{p}.attn.proj.kernel", sharding=(None, None), transpose=True
                    ),
                    f"{p}.attn.proj.bias": WeightMapping(
                        target_path=f"{p}.attn.proj.bias", sharding=(None,), transpose=False
                    ),
                    f"{p}.mlp.gate_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.gate_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{p}.mlp.gate_proj.bias": WeightMapping(
                        target_path=f"{p}.mlp.gate_proj.bias", sharding=(None,), transpose=False
                    ),
                    f"{p}.mlp.up_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.up_proj.kernel", sharding=(None, None), transpose=True
                    ),
                    f"{p}.mlp.up_proj.bias": WeightMapping(
                        target_path=f"{p}.mlp.up_proj.bias", sharding=(None,), transpose=False
                    ),
                    f"{p}.mlp.down_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.down_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{p}.mlp.down_proj.bias": WeightMapping(
                        target_path=f"{p}.mlp.down_proj.bias", sharding=(None,), transpose=False
                    ),
                }
            )
        return m

    def _llm_weight_mappings(self) -> dict:
        tc = self.text_config
        m = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }
        if not getattr(tc, "tie_word_embeddings", False):
            m["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )
        attn_bias = getattr(tc, "attention_bias", True)
        for i in range(tc.num_hidden_layers):
            p = f"model.layers.{i}"
            m.update(
                {
                    f"{p}.input_layernorm.weight": WeightMapping(
                        target_path=f"{p}.input_layernorm.scale", sharding=(None,), transpose=False
                    ),
                    f"{p}.post_attention_layernorm.weight": WeightMapping(
                        target_path=f"{p}.post_attention_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{p}.self_attn.q_proj.weight": WeightMapping(
                        target_path=f"{p}.self_attn.q_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        head_dim_padding=True,
                    ),
                    f"{p}.self_attn.k_proj.weight": WeightMapping(
                        target_path=f"{p}.self_attn.k_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{p}.self_attn.v_proj.weight": WeightMapping(
                        target_path=f"{p}.self_attn.v_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{p}.self_attn.o_proj.weight": WeightMapping(
                        target_path=f"{p}.self_attn.o_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                        head_dim_padding=True,
                    ),
                    f"{p}.mlp.gate_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.gate_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{p}.mlp.up_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.up_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{p}.mlp.down_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.down_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                }
            )
            if attn_bias:
                for proj, kvpad in (("q", False), ("k", True), ("v", True)):
                    m[f"{p}.self_attn.{proj}_proj.bias"] = WeightMapping(
                        target_path=f"{p}.self_attn.{proj}_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=kvpad,
                    )
        return m
