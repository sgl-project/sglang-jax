from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig
from sgl_jax.srt.configs.model_config import AttentionArch, ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.attention.fla.gated_rmsnorm import GatedRMSNorm
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Attention as KimiMLAAttention
from sgl_jax.srt.utils.debug_utils import begin_forward, dump_array
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class KimiMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        output, _ = self.down_proj(jax.nn.silu(gate) * up)
        return output


class KimiDeltaAttention(nnx.Module):
    """Standalone Kimi Delta Attention layer.

    This intentionally implements only the KDA attention module. Full
    KimiDecoderLayer/KimiLinearModel assembly is handled by the model skeleton
    integration work.
    """

    def __init__(
        self,
        config,
        layer_idx: int = 0,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh

        linear_config = config.linear_attn_config
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = linear_config["short_conv_kernel_size"]
        self.head_dim = linear_config["head_dim"]
        self.k_head_dim = self.head_dim
        self.v_head_dim = getattr(config, "v_head_dim", None) or self.head_dim
        self.num_heads = linear_config["num_heads"]
        self.num_k_heads = self.num_heads
        self.num_v_heads = self.num_heads
        self.projection_k_size = self.num_k_heads * self.k_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.q_proj = LinearBase(
            self.hidden_size,
            self.projection_k_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            self.hidden_size,
            self.projection_k_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            self.hidden_size,
            self.projection_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="v_proj",
        )

        # Depthwise short-conv weights stored in ``[D, K]`` directly — that's
        # the layout ``short_convolution`` consumes. We use LinearBase only as
        # a parameter container (never call it); ``input_size=D``,
        # ``output_size=K`` makes ``weight.value`` ship out as ``[D, K]`` with
        # D sharded across "tensor".
        self.q_conv1d = LinearBase(
            self.projection_k_size,
            self.conv_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="q_conv1d",
        )
        self.k_conv1d = LinearBase(
            self.projection_k_size,
            self.conv_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="k_conv1d",
        )
        self.v_conv1d = LinearBase(
            self.projection_size,
            self.conv_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="v_conv1d",
        )

        self.A_log = nnx.Param(jnp.zeros((1, 1, self.num_heads, 1), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.zeros((self.projection_size,), dtype=jnp.float32))

        self.f_a_proj = LinearBase(
            self.hidden_size,
            self.head_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
            scope_name="f_a_proj",
        )
        self.f_b_proj = LinearBase(
            self.head_dim,
            self.projection_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="f_b_proj",
        )
        self.b_proj = LinearBase(
            self.hidden_size,
            self.num_heads,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="b_proj",
        )
        self.g_a_proj = LinearBase(
            self.hidden_size,
            self.head_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
            scope_name="g_a_proj",
        )
        self.g_b_proj = LinearBase(
            self.head_dim,
            self.projection_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="g_b_proj",
        )
        self.o_norm = GatedRMSNorm(self.head_dim, epsilon=self.rms_norm_eps)
        self.o_proj = LinearBase(
            self.projection_size,
            self.hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="o_proj",
        )

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.num_k_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.k_head_dim,
            head_k_dim=self.k_head_dim,
            head_v_dim=self.v_head_dim,
            q_conv1d=self.q_conv1d,
            k_conv1d=self.k_conv1d,
            v_conv1d=self.v_conv1d,
            bias=None,
            activation=jax.nn.silu,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def __call__(
        self,
        positions: jax.Array | None,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
    ) -> tuple[jax.Array, object]:
        del positions

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        raw_gate, _ = self.f_b_proj(self.f_a_proj(hidden_states)[0])
        raw_gate = raw_gate.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)
        beta = jax.nn.sigmoid(self.b_proj(hidden_states)[0].astype(jnp.float32))

        o, recurrent_state_pool = self.attn(
            forward_batch,
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state_pool,
        )
        o = o.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)

        g_a, _ = self.g_a_proj(hidden_states)
        output_gate, _ = self.g_b_proj(g_a)
        output_gate = output_gate.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)
        o = self.o_norm(o, output_gate).reshape(hidden_states.shape[0], self.projection_size)
        o, _ = self.o_proj(o)

        return o, recurrent_state_pool


class KimiDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        mesh: jax.sharding.Mesh,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_kda = config.is_kda_layer(layer_idx)

        # Attention
        if self.is_kda:
            self.self_attn = KimiDeltaAttention(
                config=config,
                mesh=mesh,
                dtype=dtype,
                layer_idx=layer_idx,
            )
        else:
            self.self_attn = KimiMLAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                mesh=mesh,
                layer_id=layer_idx,
                dtype=dtype,
                use_absorbed=getattr(config, "use_absorbed_mla", True),
                skip_rope=config.mla_use_nope,
            )

        # FFN
        is_moe = (
            config.is_moe
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.is_moe_layer = is_moe
        self.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
        self.use_fused = self.moe_backend == "fused"

        if not is_moe:
            self.mlp = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                dtype=dtype,
            )
            self.moe_gate = None
        else:
            # Gate
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.num_experts,
                enable_expert_bias=True,
                weight_dtype=dtype,
                score_func=config.moe_router_activation_func,
            )

            # TopK routing
            self.topk = TopK(
                topk=config.num_experts_per_token,
                renormalize=config.moe_renormalize,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                routed_scaling_factor=config.routed_scaling_factor,
                layer_id=layer_idx,
            )

            if self.use_fused:
                self.block_sparse_moe = FusedEPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_token,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_idx,
                    ep_size=config.ep_size,
                    activation_fn=config.moe_router_activation_func,
                    renormalize_topk_logits=config.moe_renormalize,
                    routed_scaling_factor=config.routed_scaling_factor,
                    use_grouped_topk=config.num_expert_group > 0,
                    num_groups=config.num_expert_group,
                    top_k_groups=config.topk_group,
                    num_shared_experts=config.num_shared_experts,
                    moe_shared_expert_intermediate_size=config.moe_intermediate_size,
                )
            else:
                self.block_sparse_moe = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_token,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_idx,
                    ep_size=config.ep_size,
                )

            # Shared experts
            if config.num_shared_experts > 0:
                self.shared_experts = KimiMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                    mesh=mesh,
                    dtype=dtype,
                )
            else:
                self.shared_experts = None

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        memory_pools,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
        tag = f"layer_{self.layer_idx:02d}"

        # Pre-norm residual pattern
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        dump_array(f"{tag}_input_layernorm", hidden_states)

        # Attention
        if self.is_kda:
            kv_pool = memory_pools.recurrent_state_pool
        else:
            kv_pool = memory_pools.token_to_kv_pool

        hidden_states, kv_fused = self.self_attn(
            positions,
            hidden_states,
            forward_batch,
            kv_pool,
        )
        dump_array(f"{tag}_attn_out", hidden_states)
        # Post-attention residual + norm
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        dump_array(f"{tag}_post_attention_layernorm", hidden_states)

        # MLP (MoE or dense)
        if self.is_moe_layer:
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
                dump_array(f"{tag}_shared_experts", shared_output)
            else:
                shared_output = None

            router_logits = self.moe_gate(hidden_states)
            dump_array(f"{tag}_moe_gate", router_logits)
            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(
                router_logits, correction_bias, dispatch_info=dispatch_info
            )
            dump_array(f"{tag}_topk_weights", topk_weights)
            dump_array(f"{tag}_topk_ids", topk_ids)

            if self.use_fused:
                token_valid_mask = forward_batch.get_token_valid_mask(hidden_states.shape[0])
                topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)

            hidden_states = self.block_sparse_moe(hidden_states, topk_weights, topk_ids)
            dump_array(f"{tag}_block_sparse_moe", hidden_states)

            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            dump_array(f"{tag}_mlp", hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, topk_ids


class KimiModel(nnx.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                KimiDecoderLayer(
                    config=config,
                    mesh=mesh,
                    layer_idx=i,
                    dtype=dtype,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scope_name="norm",
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools,
    ) -> tuple[jax.Array, list, list, list]:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        dump_array("embed_tokens", hidden_states)

        residual = None
        layers_kv_fused = []
        layers_recurrent_buffers = []
        layers_conv_buffers = []
        layers_topk_ids = []

        for i, layer in enumerate(self.layers):
            hidden_states, residual, attn_state, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                memory_pools,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            dump_array(f"layer_{i:02d}_out", hidden_states)
            if layer.is_kda:
                rec_buf, conv_buf_list = attn_state
                layers_recurrent_buffers.append(rec_buf)
                layers_conv_buffers.append(conv_buf_list)
            else:
                layers_kv_fused.append(attn_state)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        dump_array("model_norm", hidden_states)
        return (
            hidden_states,
            layers_kv_fused,
            (layers_recurrent_buffers, layers_conv_buffers),
            layers_topk_ids,
        )


class KimiLinearForCausalLM(nnx.Module):
    @classmethod
    def patch_model_config(cls, config: ModelConfig) -> None:
        config.attention_arch = AttentionArch.MLA
        qk_nope = getattr(config.hf_text_config, "qk_nope_head_dim", 0)
        qk_rope = getattr(config.hf_text_config, "qk_rope_head_dim", 0)
        if qk_nope and qk_rope:
            config.head_dim = qk_nope + qk_rope

    def __init__(
        self, config: KimiLinearConfig, mesh: jax.sharding.Mesh, dtype: jnp.dtype = jnp.bfloat16
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = KimiModel(
            config=config,
            mesh=mesh,
            dtype=dtype,
        )

        if not getattr(config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools,
        logits_metadata: LogitsMetadata,
    ):
        begin_forward(
            "prefill" if forward_batch.forward_mode.is_prefill() else "decode"
        )
        hidden_states, layers_kv_fused, layers_recurrent_state, layers_topk_ids = self.model(
            forward_batch,
            memory_pools,
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        dump_array("logits", output.next_token_logits)
        return (
            output,
            {
                "token_to_kv_pool": layers_kv_fused,
                "recurrent_state_pool": layers_recurrent_state,
            },
            True,
            layers_topk_ids,
        )

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        for layer in self.model.layers:
            if not layer.is_kda:
                layer.self_attn.post_load_weights()
        logger.info("Weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        num_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        moe_backend = getattr(self.config, "moe_backend", "epmoe")

        for layer_idx in range(num_layers):
            is_dense = layer_idx < first_k_dense_replace
            is_kda = self.config.is_kda_layer(layer_idx)
            layer_mappings = self._create_layer_mappings(
                layer_idx,
                is_dense=is_dense,
                is_kda=is_kda,
                moe_backend=moe_backend,
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(
        self, layer_idx: int, *, is_dense: bool, is_kda: bool, moe_backend: str
    ) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            # Layer norms (all layers)
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        # --- Attention mappings ---
        if is_kda:
            attn_target = f"{target_prefix}.self_attn"
            for proj_name in ("q_proj", "k_proj", "v_proj", "f_b_proj", "b_proj", "g_b_proj"):
                mappings[f"{prefix}.self_attn.{proj_name}.weight"] = WeightMapping(
                    target_path=f"{attn_target}.{proj_name}.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
            for proj_name in ("f_a_proj", "g_a_proj"):
                mappings[f"{prefix}.self_attn.{proj_name}.weight"] = WeightMapping(
                    target_path=f"{attn_target}.{proj_name}.weight",
                    sharding=(None, None),
                    transpose=True,
                )
            mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )
            # Conv1d weights: HF shape (projection_size, 1, conv_size) -> JAX (conv_size, projection_size) -> (projection_size, conv_size)
            # These live under self_attn.attn (RadixLinearAttention), not self_attn directly.
            conv_size = self.config.linear_attn_config["short_conv_kernel_size"]
            num_heads = self.config.linear_attn_config["num_heads"]
            head_dim = self.config.linear_attn_config["head_dim"]
            projection_size = num_heads * head_dim
            for conv_name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                mappings[f"{prefix}.self_attn.{conv_name}.weight"] = WeightMapping(
                    target_path=f"{attn_target}.attn.{conv_name}.weight",
                    sharding=("tensor", None),
                    transpose=False,
                    # transpose_axes=(2, 0, 1),
                    reshape=(projection_size, conv_size),
                )
            mappings[f"{prefix}.self_attn.o_norm.weight"] = WeightMapping(
                target_path=f"{attn_target}.o_norm.weight",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.dt_bias"] = WeightMapping(
                target_path=f"{attn_target}.attn.dt_bias",
                sharding=("tensor",),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.A_log"] = WeightMapping(
                target_path=f"{attn_target}.A_log",
                sharding=(None, None, "tensor", None),
                transpose=False,
            )
        else:
            # MLA layer — MLAAttention is directly self.self_attn
            attn_target = f"{target_prefix}.self_attn"
            mappings[f"{prefix}.self_attn.q_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )
            mappings[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = WeightMapping(
                target_path=f"{attn_target}.kv_a_proj.weight",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.self_attn.kv_a_layernorm.weight"] = WeightMapping(
                target_path=f"{attn_target}.kv_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.kv_b_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.kv_b_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )

        # --- FFN mappings ---
        if is_dense:
            # Dense MLP
            for proj_name, sharding in [
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ]:
                mappings[f"{prefix}.mlp.{proj_name}.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.{proj_name}.weight",
                    sharding=sharding,
                    transpose=True,
                )
        else:
            # MoE — gate/topk are flat on decoder layer, experts in block_sparse_moe

            # Gate
            mappings[f"{prefix}.block_sparse_moe.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.block_sparse_moe.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.bias",
                sharding=(None,),
            )

            # Expert weights
            num_logical_experts = self.config.num_experts

            from sgl_jax.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            phy_to_log = None
            if metadata is not None:
                physical_to_logical_map = np.array(jax.device_get(metadata.physical_to_logical_map))
                phy_to_log = physical_to_logical_map[layer_idx]

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=num_logical_experts,
                expert_type_names=("w1", "w3", "w2"),
                moe_backend=moe_backend,
                moe_path="block_sparse_moe",
                physical_to_logical_map=phy_to_log,
            )
            mappings.update(moe_mappings)

            # Shared experts
            for proj_name, sharding in [
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ]:
                mappings[f"{prefix}.block_sparse_moe.shared_experts.{proj_name}.weight"] = (
                    WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.{proj_name}.weight",
                        sharding=sharding,
                        transpose=True,
                    )
                )

        return mappings


EntryClass = KimiLinearForCausalLM
