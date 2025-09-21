import logging
from typing import Optional, Tuple

import jax
import jax.lax
from flax import nnx
from jax import numpy as jnp
from jax._src.mesh import get_abstract_mesh, use_abstract_mesh
from transformers import PretrainedConfig

from sgl_jax.srt.layers.activation import GeluAndMul
from sgl_jax.srt.layers.embeddings import (
    Embed,
    ParallelLMHead,
    RotaryEmbedding,
    ScalingRotaryEmbedding,
)
from sgl_jax.srt.layers.layernorm import RMSNorm, dual_rmsnorm_forward
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


def get_rope_scaling(config):
    rope_type = getattr(config, "rope_type", None)
    if rope_type:
        original_max_position_embeddings = getattr(
            config, "original_max_position_embeddings", None
        )
        scaling_factor = getattr(config, "scaling_factor", None)
        extrapolation_factor = getattr(config, "extrapolation_factor", 1.0)
        attn_factor = getattr(config, "attn_factor", 1.0)
        beta_fast = getattr(config, "beta_fast", 32)
        beta_slow = getattr(config, "beta_slow", 1)
        rope_scaling = {
            "extra_method": rope_type,
            "max_position_embeddings": original_max_position_embeddings,
            "scaling_factor": scaling_factor,
            "extrapolation_factor": extrapolation_factor,
            "attn_factor": attn_factor,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "dtype": jnp.float32,
        }
        return rope_scaling
    else:
        return None


class Grok1MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        super().__init__()

        self.gate_up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size * 2,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            rngs=rngs,
        )
        self.act_fn = GeluAndMul(approximate="tanh")
        self.layer_id = layer_id

    def __call__(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Grok1MoE(nnx.Module):
    """A tensor-parallel MoE implementation for Grok1 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        rngs: Optional[nnx.Rngs] = None,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        super().__init__()

        if mesh is None:
            mesh = get_abstract_mesh()
        # Gate always runs at full precision for stability (see https://arxiv.org/pdf/2101.03961)
        self.gate = LinearBase(
            input_size=config.hidden_size,
            output_size=config.num_local_experts,
            use_bias=False,
            params_dtype=jnp.float32,
            kernel_axes=None,
            rngs=rngs,
        )

        self.router_logit_softcapping = getattr(config, "router_logit_softcapping", 30)

        expert_parallel_size = mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1)
        with use_abstract_mesh(mesh):
            self.experts = EPMoE(
                config=config,
                num_experts=config.num_local_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                expert_parallel_size=expert_parallel_size,
                mesh=mesh,
                intermediate_dim=config.moe_intermediate_size,
                dtype=dtype,
                activation="gelu",
                layer_id=layer_id,
                rngs=rngs,
            )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        router_logits, _ = self.gate(hidden_states)
        if self.router_logit_softcapping != 0:
            router_logits = router_logits / self.router_logit_softcapping
            router_logits = jax.nn.tanh(router_logits) * self.router_logit_softcapping
        router_logits = jax.nn.softmax(router_logits, axis=-1)
        return self.experts(hidden_states, router_logits)


class Grok1Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        reduce_results: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = getattr(config, "head_dim", 128)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        rope_scaling = get_rope_scaling(config)

        num_heads = self.total_num_heads + self.total_num_kv_heads
        self.qkv_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
        )
        self.o_proj = LinearBase(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=False,
            # reduce_results=reduce_results,
            kernel_axes=("tensor", None),
            rngs=rngs,
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
            dtype=jnp.float32,
        )

        self.rope_rotate_half_dims = getattr(config, "rope_rotate_half_dims", False)

        if rope_scaling is not None:
            self.rotary_emb = ScalingRotaryEmbedding(
                self.head_dim,
                rotary_dim=(
                    self.head_dim
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2
                ),
                base=int(self.rope_theta),
                is_neox_style=True,
                **rope_scaling,
            )
            pos_encoding_mode = "NONE"
        else:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                rotary_dim=(
                    self.head_dim
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2
                ),
                max_position=max_position,
                base=int(self.rope_theta),
                is_neox_style=True,
            )
            pos_encoding_mode = "NONE"

        logit_cap = max(getattr(config, "attn_logit_softcapping", 30.0), 0.0)
        logit_capping_method = getattr(config, "attn_logit_softcapping_method", "tanh")

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
            pos_encoding_mode=pos_encoding_mode,
            logit_capping_method=logit_capping_method,
            xai_temperature_len=getattr(self.config, "attn_temperature_len", -1),
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states

        qkv, _ = self.qkv_proj(hidden_states)

        q, k, v = jnp.split(qkv, [self.q_size, self.q_size + self.kv_size], axis=-1)

        num_tokens = q.shape[0]
        q = q.reshape(num_tokens, self.num_heads, self.head_dim)
        k = k.reshape(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.reshape(num_tokens, self.num_kv_heads, self.head_dim)
        
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)

        output, _ = self.o_proj(attn_output)
        return output


class Grok1DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        rngs: Optional[nnx.Rngs] = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.residual_moe = getattr(config, "residual_moe", False)
        self.layer_id = layer_id

        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = Grok1Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=(
                config.context_len
                if hasattr(config, "context_len")
                else config.max_position_embeddings
            ),
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            reduce_results=False,
            rngs=rngs,
        )

        split_gate_up = not getattr(config, "merge_gate_up", True)
        if self.num_experts > 0:
            self.block_sparse_moe = Grok1MoE(
                config=config,
                layer_id=layer_id,
                # reduce_results=not self.residual_moe,
                # inplace=False,  # not self.residual_moe,
                # no_combine=False,  # self.residual_moe,  # just a suggestion to not combine topk
                rngs=rngs,
                mesh=mesh,
            )
            if self.residual_moe:
                self.mlp = Grok1MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    # reduce_results=False,
                    layer_id=layer_id,
                    rngs=rngs,
                )
        else:
            raise NotImplementedError()

        self.pre_attn_norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.post_attn_norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.pre_moe_norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.post_moe_norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )

        if self.num_experts > 0:
            if self.residual_moe:
                # NOTE: self.block_sparse_moe modifies the input in-place,
                # so we have to call it later. Be aware of any possible related errors.
                if mesh.size > 1:
                    self.ffn = lambda x: jax.lax.psum(
                        self.moe_with_rmoe(x), axis_name="tensor"
                    )
                else:
                    self.ffn = self.moe_with_rmoe
            else:
                self.ffn = self.block_sparse_moe
        else:
            raise NotImplementedError()

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        residual: Optional[jax.Array] = None,
        deferred_norm: Optional[RMSNorm] = None,
    ) -> Tuple[jax.Array, jax.Array, RMSNorm]:

        hidden_states_original = hidden_states
        residual_original = residual

        # Self Attention
        if deferred_norm is not None:
            assert residual is not None
            # here hidden_states is output of ffn, residual is residual from after previous attn layer
            hidden_states, residual = dual_rmsnorm_forward(
                hidden_states,
                residual,
                deferred_norm.weight,
                self.pre_attn_norm.weight,
                deferred_norm.variance_epsilon,
            )
        else:
            # here hidden_states is the residual
            hidden_states, residual = self.pre_attn_norm(hidden_states), hidden_states

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states = jax.lax.psum(hidden_states, axis_name="tensor")

        hidden_states, residual = dual_rmsnorm_forward(
            hidden_states,
            residual,
            self.post_attn_norm.weight,
            self.pre_moe_norm.weight,
            self.post_attn_norm.variance_epsilon,
        )

        # Fully Connected
        hidden_states = self.ffn(hidden_states)
        return hidden_states, residual, self.post_moe_norm  # defer layernorm

    def moe_with_rmoe(self, x):
        mlp_result = self.mlp(x)
        moe_result = self.block_sparse_moe(x)
        return (mlp_result + moe_result) / 1.4142135623730951


class Grok1Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        rngs: Optional[nnx.Rngs] = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            # enable_tp=not replicate_embedding,
        )

        self.layers = [
            Grok1DecoderLayer(config=config, layer_id=i, mesh=mesh, rngs=rngs)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
        input_embeds: jax.Array = None,
    ) -> jax.Array:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states.mul_(self.config.embedding_multiplier_scale)
        else:
            hidden_states = input_embeds

        residual, deferred_norm = None, None
        for i in range(len(self.layers)):
            hidden_states, residual, deferred_norm = self.layers[i](
                positions, hidden_states, forward_batch, residual, deferred_norm
            )

        hidden_states, _ = dual_rmsnorm_forward(
            hidden_states,
            residual,
            deferred_norm.weight,
            self.norm.weight,
            deferred_norm.variance_epsilon,
        )

        return hidden_states


class Grok1ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        rngs: Optional[nnx.Rngs] = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ) -> None:
        super().__init__()
        self.config = config
        if mesh is None:
            mesh = get_abstract_mesh()

        self.model = Grok1Model(config, rngs=rngs, mesh=mesh)

        lm_head_params_dtype = None
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            param_dtype=lm_head_params_dtype,
            rngs=rngs,
        )
        self.logits_processor = LogitsProcessor(config, lm_head=self.lm_head, mesh=mesh)

        self.loaded_param_names = set()

    def forward(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
        input_embeds: Optional[jax.Array] = None,
    ) -> jax.Array:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)
        loader = WeightLoader(
            model=self, model_config=self.config, mesh=self.mesh, dtype=self.dtype
        )
        weight_mappings = self._create_grok_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Grok weights loaded successfully!")

    def _create_moe_weight_mappings(
        self, layer_idx: int, state_name: str, model_name: str
    ):
        return {
            f"__MOE_EXPERTS__{model_name}": WeightMapping(
                target_path=[
                    f"model.layers.{layer_idx}.block_sparse_moe.experts.{model_name}.weight",
                ]
                + [
                    f"model.layers.{layer_idx}.block_sparse_moe.experts.{eid}.{state_name}.weight"
                    for eid in range(self.config.hf_config.num_local_experts)
                ],
                sharding=(("data", "tensor"), None, None),
                transpose=False,
            )
        }

    def _create_grok_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=(None, None),
                transpose=False,
            ),
        }

        if not getattr(self.config.hf_config, "tie_word_embeddings", True):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=(None, None), transpose=False
            )

        num_layers = self.config.hf_config.num_hidden_layers
        for layer_idx in range(num_layers):
            mappings[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = WeightMapping(
                target_path=f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                sharding=(None, None),
                transpose=False,
            )
            mappings[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = WeightMapping(
                target_path=f"model.layers.{layer_idx}.mlp.down_proj.weight",
                sharding=(None, None),
                transpose=False,
            )
            mappings[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = WeightMapping(
                target_path=f"model.layers.{layer_idx}.mlp.up_proj.weight",
                sharding=(None, None),
                transpose=False,
            )
            mappings.update(self._create_moe_weight_mappings(layer_idx, "w1", "wi_0"))
            mappings.update(self._create_moe_weight_mappings(layer_idx, "w2", "wi_1"))
            mappings.update(self._create_moe_weight_mappings(layer_idx, "w3", "wo"))
            mappings[f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"] = (
                WeightMapping(
                    target_path=f"model.layers.{layer_idx}.block_sparse_moe.gate.weight",
                    sharding=(None, None),
                    transpose=False,
                )
            )

            for key in ("q_proj", "k_proj", "v_proj"):
                mappings[f"model.layers.{layer_idx}.self_attn.{key}.weight"] = (
                    WeightMapping(
                        target_path=f"model.layers.{layer_idx}.attn.{key}.weight",
                        sharding=(None, "tensor"),
                        transpose=False,
                    )
                )
            mappings[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = (
                WeightMapping(
                    target_path=f"model.layers.{layer_idx}.attn.o_proj.weight",
                    sharding=("tensor", None),
                    transpose=False,
                )
            )

            for key in (
                "pre_attn_norm",
                "post_attn_norm",
                "pre_moe_norm",
                "post_moe_norm",
            ):
                mappings[f"model.layers.{layer_idx}.{key}.weight"] = WeightMapping(
                    target_path=f"model.layers.{layer_idx}.{key}.weight",
                    sharding=(None, None),
                    transpose=False,
                )

        return mappings
