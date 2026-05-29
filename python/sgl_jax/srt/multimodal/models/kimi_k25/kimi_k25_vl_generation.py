import logging

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from sgl_jax.srt.configs.model_config import AttentionArch, ModelConfig
from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Model
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)


class KimiDeepseekV3Model(DeepseekV3Model):
    
    def __init__(
        self,
        config,
        mesh,
        dtype=jnp.bfloat16,
    ):
        super().__init__(config=config, mesh=mesh, dtype=dtype)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        input_embeds = (
            forward_batch.input_embedding
            if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            else None
        )

        hidden_states = (
            self.embed_tokens(forward_batch.input_ids) if input_embeds is None else input_embeds
        )

        residual = None
        layers_kv_fused = []
        layers_topk_ids = []

        for layer in self.layers:
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, layers_topk_ids



class KimiK25ForConditionalGeneration(nnx.Module):

    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        # MLA: Q/K head dim is qk_nope + qk_rope; V head dim differs (v_head_dim).
        # Override the generic head_dim so the attention backend and KV pool
        # allocate MLA-shaped buffers.
        mc.attention_arch = AttentionArch.MLA
        qk_nope = getattr(mc.hf_text_config, "qk_nope_head_dim", 0)
        qk_rope = getattr(mc.hf_text_config, "qk_rope_head_dim", 0)
        if qk_nope and qk_rope:
            mc.head_dim = qk_nope + qk_rope

    def __init__(
        self,
        config=None,
        dtype=None,
        mesh=None,
    ):
        super().__init__()

        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16
        self.mesh = mesh

        self.model = KimiDeepseekV3Model(self.text_config, mesh=mesh, dtype=self.dtype)

        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
            )

        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=mesh)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        token_to_kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch, token_to_kv_pool
        )

        if not getattr(self.text_config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        return output, layers_kv_fused, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings(model_config)
        loader.load_weights_from_safetensors(weight_mappings)
        # Absorbed-MLA path: pre-split kv_b_proj into w_uk/w_uv and drop the
        # original projection (sglang parity, deepseek_weight_loader.post_process).
        for layer in self.model.layers:
            layer.self_attn.post_load_weights()
        logger.info("Kimi K2.5 Languge model weights loaded successfully!")

    def _create_weight_mappings(self, model_config: ModelConfig) -> dict:
        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint

        mappings = {
            "language_model.model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "language_model.model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            "language_model.lm_head.weight": WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
        }

        n_routed_experts = getattr(self.text_config, "n_routed_experts", None)
        first_k_dense_replace = getattr(self.text_config, "first_k_dense_replace", 0)
        moe_layer_freq = getattr(self.text_config, "moe_layer_freq", 1)
        moe_backend = getattr(self.text_config, "moe_backend", "epmoe")
        use_fused = moe_backend == "fused"

        for layer_idx in range(self.text_config.num_hidden_layers):
            is_moe = (
                n_routed_experts is not None
                and layer_idx >= first_k_dense_replace
                and layer_idx % moe_layer_freq == 0
            )
            layer_mappings = self._create_layer_mappings(
                layer_idx, is_moe, moe_backend, use_fused, is_static_quant
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(
        self,
        layer_idx: int,
        is_moe: bool,
        moe_backend: str,
        use_fused: bool,
        is_static_quant: bool = False,
    ) -> dict:
        prefix = f"language_model.model.layers.{layer_idx}"
        target = f"model.layers.{layer_idx}"
        mappings: dict = {}

        def add_linear(hf_prefix: str, target_prefix: str, sharding_std: tuple):
            # HF weights are `[out, in]`.
            #   Unquantized: loaded into LinearBase.weight `[in, out]` so
            #   `transpose=True` and sharding is the LinearBase kernel_axes.
            #   Static FP8: loaded into QuantizedLinear.weight_q `[out, in]`
            #   directly; sharding is kernel_axes swapped. Also register the
            #   `weight_scale_inv` sidecar into `weight_scale`.
            if not is_static_quant:
                mappings[f"{hf_prefix}.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.weight",
                    sharding=sharding_std,
                    transpose=True,
                )
                return
            sharding_quant = (sharding_std[1], sharding_std[0])
            mappings[f"{hf_prefix}.weight"] = WeightMapping(
                target_path=f"{target_prefix}.weight_q",
                sharding=sharding_quant,
                transpose=False,
            )
            # HF block scale is `[out_blocks, in_blocks]`; WeightLoader's
            # _maybe_expand_linear_block_scale expands to `[in_blocks, 1, n_out]`.
            # QuantizedLinear.__call__ expects sharding P(kernel_axes[0], None,
            # kernel_axes[1]) on the 3D result, which maps back to the 2D
            # checkpoint as sharding_quant = (out_blocks_axis, in_blocks_axis):
            #   col-parallel: ("tensor", None) → 3D axis 2 sharded.
            #   row-parallel: (None, "tensor") → 3D axis 0 sharded.
            mappings[f"{hf_prefix}.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.weight_scale",
                sharding=sharding_quant,
                transpose=False,
            )

        # Layer norms
        mappings[f"{prefix}.input_layernorm.weight"] = WeightMapping(
            target_path=f"{target}.input_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{prefix}.post_attention_layernorm.weight"] = WeightMapping(
            target_path=f"{target}.post_attention_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )

        # MLA attention. HF 'kv_a_proj_with_mqa' -> JAX 'kv_a_proj'.
        ap = f"{prefix}.self_attn"
        tp = f"{target}.self_attn"
        q_lora_rank = getattr(self.text_config, "q_lora_rank", None)
        if q_lora_rank is None:
            add_linear(f"{ap}.q_proj", f"{tp}.q_proj", (None, "tensor"))
        else:
            add_linear(f"{ap}.q_a_proj", f"{tp}.q_a_proj", (None, None))
            mappings[f"{ap}.q_a_layernorm.weight"] = WeightMapping(
                target_path=f"{tp}.q_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            )
            add_linear(f"{ap}.q_b_proj", f"{tp}.q_b_proj", (None, "tensor"))
        add_linear(f"{ap}.kv_a_proj_with_mqa", f"{tp}.kv_a_proj", (None, None))
        mappings[f"{ap}.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{tp}.kv_a_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        add_linear(f"{ap}.kv_b_proj", f"{tp}.kv_b_proj", (None, "tensor"))
        add_linear(f"{ap}.o_proj", f"{tp}.o_proj", ("tensor", None))

        if not is_moe:
            # Dense MLP (first K layers)
            for proj, sharding in [
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ]:
                add_linear(f"{prefix}.mlp.{proj}", f"{target}.mlp.{proj}", sharding)
            return mappings

        # MoE gate (router) — NOT quantized in HF FP8 checkpoint.
        mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{target}.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
        )
        if getattr(self.text_config, "topk_method", "noaux_tc") == "noaux_tc":
            mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target}.moe_gate.bias",
                sharding=(None,),
                transpose=False,
            )

        metadata = get_global_expert_location_metadata()
        phy_to_log = None
        if metadata is not None:
            physical_to_logical_map = np.array(jax.device_get(metadata.physical_to_logical_map))
            phy_to_log = physical_to_logical_map[layer_idx]

        moe_mappings = create_moe_weights_mapping(
            prefix=prefix,
            target_prefix=target,
            num_experts=self.text_config.n_routed_experts,
            expert_type_names=("gate_proj", "up_proj", "down_proj"),
            moe_backend=moe_backend,
            physical_to_logical_map=phy_to_log,
        )
        mappings.update(moe_mappings)

        # Routed expert weight-scale sidecars (static FP8, non-fused only).
        # Fused MoE static-FP8 placeholder shapes are (1,) today — loading
        # block scales would need a dedicated fix in fused_moe.py. Skip.
        #
        # For each expert weight group (emitted by create_moe_weights_mapping
        # as `__MOE_EXPERTS__<target_base>`) register a parallel scale group
        # whose HF keys are the per-expert `*.weight_scale_inv` tensors and
        # whose target is `<target_base>_scale` (e.g. `wi_0_scale`). After
        # stacking, WeightLoader's _maybe_convert_epmoe_scale_for_kernel
        # converts the `[E, out_blocks, in_blocks]` layout into the
        # kernel-ready `[E, k_blocks, 1, n_out]` expected by GMM.
        if is_static_quant and not use_fused:
            for moe_key, wm in moe_mappings.items():
                if not moe_key.startswith("__MOE_EXPERTS__"):
                    continue
                target_base = wm.target_path[0]
                expert_scale_keys = [
                    k.replace(".weight", ".weight_scale_inv") for k in wm.target_path[1:]
                ]
                scale_target = f"{target_base}_scale"
                # Stacked checkpoint scale is `[E, out_blocks, in_blocks]`. Load
                # replicated on the block dims; _maybe_convert_epmoe_scale_for_kernel
                # expands via jnp.take, which fails if the gathered axis is
                # tensor-sharded (ambiguous output sharding). The converter reshards
                # to model_param.value.sharding at the end.
                mappings[f"__MOE_EXPERTS__{scale_target}"] = WeightMapping(
                    target_path=[scale_target] + expert_scale_keys,
                    sharding=("expert", None, None),
                    transpose=False,
                    physical_to_logical_map=wm.physical_to_logical_map,
                )

        # Shared expert weights
        n_shared_experts = getattr(self.text_config, "n_shared_experts", None)
        if n_shared_experts and n_shared_experts > 0:
            if use_fused:
                # Fused backend stores shared-expert weights as bare nnx.Param
                # (no QuantizedLinear swap). Static-FP8 scale mappings for the
                # fused path are TODO (requires fixed placeholder shapes in
                # FusedEPMoE.quantize_weights(is_static=True)).
                for hf_name, target_name in [
                    ("gate_proj", "w1_shared"),
                    ("up_proj", "w3_shared"),
                    ("down_proj", "w2_shared"),
                ]:
                    mappings[f"{prefix}.mlp.shared_experts.{hf_name}.weight"] = WeightMapping(
                        target_path=f"{target}.mlp.{target_name}",
                        sharding=(None, None),
                        transpose=True,
                    )
            else:
                for proj, sharding in [
                    ("gate_proj", (None, "tensor")),
                    ("up_proj", (None, "tensor")),
                    ("down_proj", ("tensor", None)),
                ]:
                    add_linear(
                        f"{prefix}.mlp.shared_experts.{proj}",
                        f"{target}.shared_experts.{proj}",
                        sharding,
                    )

        return mappings

