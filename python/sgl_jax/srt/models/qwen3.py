import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import (
    Embed,
    MRotaryEmbedding,
    ParallelLMHead,
    RotaryEmbedding,
)
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


def create_qwen3_weight_mappings(
    config,
    source_prefix: str = "model",
    target_prefix: str = "model",
) -> dict:
    mappings = {
        f"{source_prefix}.embed_tokens.weight": WeightMapping(
            target_path=f"{target_prefix}.embed_tokens.embedding",
            sharding=("tensor", None),
            transpose=False,
        ),
        f"{source_prefix}.norm.weight": WeightMapping(
            target_path=f"{target_prefix}.norm.scale", sharding=(None,), transpose=False
        ),
    }
    if not getattr(config, "tie_word_embeddings", False):
        mappings["lm_head.weight"] = WeightMapping(
            target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
        )
    for layer_idx in range(config.num_hidden_layers):
        mappings.update(
            create_qwen3_layer_mappings(config, layer_idx, source_prefix, target_prefix)
        )
    return mappings


def create_qwen3_layer_mappings(
    config,
    layer_idx: int,
    source_prefix: str = "model",
    target_prefix: str = "model",
) -> dict:
    source = f"{source_prefix}.layers.{layer_idx}"
    target = f"{target_prefix}.layers.{layer_idx}"
    mappings = {
        f"{source}.input_layernorm.weight": WeightMapping(
            target_path=f"{target}.input_layernorm.scale", sharding=(None,), transpose=False
        ),
        f"{source}.post_attention_layernorm.weight": WeightMapping(
            target_path=f"{target}.post_attention_layernorm.scale",
            sharding=(None,),
            transpose=False,
        ),
        f"{source}.self_attn.q_proj.weight": WeightMapping(
            target_path=f"{target}.self_attn.q_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=False,
        ),
        f"{source}.self_attn.k_proj.weight": WeightMapping(
            target_path=f"{target}.self_attn.k_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=True,
        ),
        f"{source}.self_attn.v_proj.weight": WeightMapping(
            target_path=f"{target}.self_attn.v_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=True,
        ),
        f"{source}.self_attn.o_proj.weight": WeightMapping(
            target_path=f"{target}.self_attn.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
            kv_head_padding=False,
        ),
        f"{source}.self_attn.q_norm.weight": WeightMapping(
            target_path=f"{target}.self_attn.q_norm.scale", sharding=(None,), transpose=False
        ),
        f"{source}.self_attn.k_norm.weight": WeightMapping(
            target_path=f"{target}.self_attn.k_norm.scale", sharding=(None,), transpose=False
        ),
        f"{source}.mlp.gate_proj.weight": WeightMapping(
            target_path=f"{target}.mlp.gate_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        ),
        f"{source}.mlp.up_proj.weight": WeightMapping(
            target_path=f"{target}.mlp.up_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        ),
        f"{source}.mlp.down_proj.weight": WeightMapping(
            target_path=f"{target}.mlp.down_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        ),
    }
    if getattr(config, "attention_bias", False):
        for name, padding in (("q_proj", False), ("k_proj", True), ("v_proj", True)):
            mappings[f"{source}.self_attn.{name}.bias"] = WeightMapping(
                target_path=f"{target}.self_attn.{name}.bias",
                sharding=(None,),
                transpose=False,
                kv_head_padding=padding,
            )
        mappings[f"{source}.self_attn.o_proj.bias"] = WeightMapping(
            target_path=f"{target}.self_attn.o_proj.bias",
            sharding=(None,),
            transpose=False,
        )
    return mappings


class QWen3Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.mesh = mesh

        self.q_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm"
        )
        self.k_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm"
        )

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )
        if rope_scaling and "mrope_section" in rope_scaling:
            self.rotary_emb = MRotaryEmbedding(
                self.head_dim,
                self.head_dim,
                max_position_embeddings,
                rope_theta,
                True,
                dtype,
                mrope_section=rope_scaling["mrope_section"],
                mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                self.head_dim,
                max_position_embeddings,
                rope_theta,
                True,
                dtype,
            )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        output, _ = self.o_proj(attn_output, out_sharding=out_sharding)
        return output, kv_fused


class Qwen3MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id

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

        self.act_fn = jax.nn.silu

    @named_scope
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel, out_sharding=out_sharding)
        return output


class QWen3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = QWen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            layer_id=layer_id,
            attention_bias=config.attention_bias,
            dtype=dtype,
            mesh=mesh,
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )
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
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ):
        layer_callback_flag = []
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        layer_norm_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "input_layernorm_output", "INPUT_LAYERNORM", self.layer_id
        )
        layer_callback_flag.append(layer_norm_callback_flag)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        attn_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "self_attn_output", "SELF_ATTN", self.layer_id
        )
        layer_callback_flag.append(attn_callback_flag)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        mlp_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "mlp_output", "MLP", self.layer_id
        )
        layer_callback_flag.append(mlp_callback_flag)

        return hidden_states, residual, kv_fused, layer_callback_flag


class QWen3Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                QWen3DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="norm",
        )
        # For EAGLE3 support
        self.layers_to_capture = []

    def get_input_embeddings(self):
        return self.embed_tokens

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = None
        input_embedding = getattr(forward_batch, "input_embedding", None)
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids)
            if input_embedding is None
            else input_embedding
        )
        layers_kv_fused = []
        layers_callback_flag = []
        aux_hidden_states = []
        rope_positions = getattr(forward_batch, "mrope_positions", None)
        rope_positions = forward_batch.positions if rope_positions is None else rope_positions
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual, kv_fused, callback_flag = layer(
                rope_positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            deepstack = getattr(forward_batch, "deepstack_visual_embedding", None)
            if deepstack is not None and layer_id < deepstack.shape[0]:
                hidden_states += deepstack[layer_id].astype(hidden_states.dtype)
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "transformer_output", "TRANSFORMER"
        )
        layers_callback_flag.append(callback_flag)
        return hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag


class Qwen3ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("QWen3ForCausalLMModel config dtype: %s", self.dtype)
        self.model = QWen3Model(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_qwen3_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3 weights loaded successfully!")

    def _create_qwen3_weight_mappings(self) -> dict:
        return create_qwen3_weight_mappings(self.config)

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        return create_qwen3_layer_mappings(self.config, layer_idx)

    def get_embed_and_head(self):
        return (
            self.model.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        """Set word embedding and LM Head weights.

        Args:
            embed_weight: Embedding matrix with shape [vocab_size, hidden_size].
            head_weight:  LM Head matrix with shape [vocab_size, hidden_size].
        """
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight

        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None):

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, kv_pool
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata, aux_hidden_states=aux_hidden_states
            )
        else:
            output = self.logits_processor(
                hidden_states,
                self.model.embed_tokens,
                logits_metadata,
                aux_hidden_states=aux_hidden_states,
            )

        return output, {"token_to_kv_pool": layers_kv_fused}, layers_callback_flag, None


EntryClass = Qwen3ForCausalLM
