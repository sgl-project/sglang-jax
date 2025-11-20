import contextlib
import json
import logging
import os

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.models.qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class MiMoMTPLayer(nnx.Module):
    """Container for MiMo multi-token prediction (MTP) block weights."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.layer_id = layer_id
        hidden_size = getattr(config, "mtp_hidden_size", config.hidden_size)
        rope_theta = getattr(config, "mtp_rope_theta", getattr(config, "rope_theta", 1000000))
        rope_scaling = getattr(config, "mtp_rope_scaling", getattr(config, "rope_scaling", None))
        max_position_embeddings = getattr(
            config, "mtp_max_position_embeddings", getattr(config, "max_position_embeddings", 32768)
        )
        head_dim = getattr(config, "mtp_head_dim", getattr(config, "head_dim", None))
        num_heads = getattr(config, "mtp_num_attention_heads", config.num_attention_heads)
        num_kv_heads = getattr(
            config,
            "mtp_num_key_value_heads",
            getattr(config, "num_key_value_heads", config.num_attention_heads),
        )
        intermediate_size = getattr(config, "mtp_intermediate_size", config.intermediate_size)

        self.input_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.input_layernorm = RMSNorm(hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.hidden_layernorm = RMSNorm(hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)
        self.token_layernorm = RMSNorm(hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)
        self.final_layernorm = RMSNorm(hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)

        self.self_attn = Qwen2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            layer_id=layer_id,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.mlp = Qwen2MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("MiMo MTP layers are not yet integrated into inference.")


class MiMoModel(Qwen2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__(config=config, dtype=dtype, rngs=rngs, mesh=mesh)
        self._mtp_layers_initialized = False
        self.mtp_layers = nnx.data([])

    def ensure_mtp_layers(
        self,
        num_layers: int,
        config: PretrainedConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs | None,
        mesh: jax.sharding.Mesh | None,
    ):
        if num_layers <= 0:
            return
        if self._mtp_layers_initialized and len(self.mtp_layers) == num_layers:
            return

        mesh_context = jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()

        with mesh_context:
            mtp_layers = [
                MiMoMTPLayer(
                    config=config,
                    layer_id=layer_id,
                    dtype=dtype,
                    rngs=rngs,
                    mesh=mesh,
                )
                for layer_id in range(num_layers)
            ]
        self.mtp_layers = nnx.data(mtp_layers)
        self._mtp_layers_initialized = True


class MiMoForCausalLM(Qwen2ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__(config, dtype, rngs, mesh)
        self.model = MiMoModel(config, dtype=self.dtype, rngs=rngs, mesh=mesh)

    def load_weights(self, model_config: ModelConfig, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)
        self._weight_map = self._load_weight_map(model_config.model_path)
        self._num_mtp_layers = self._infer_mtp_layer_count()
        if self._num_mtp_layers:
            self.model.ensure_mtp_layers(
                num_layers=self._num_mtp_layers,
                config=self.config,
                dtype=self.dtype,
                rngs=self.rng,
                mesh=self.mesh,
            )

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_mimo_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("MiMo weights loaded successfully!")

    def _create_mimo_weight_mappings(self) -> dict:
        mappings = super()._create_qwen2_weight_mappings()

        num_mtp_layers = getattr(self, "_num_mtp_layers", 0)
        if not num_mtp_layers:
            num_mtp_layers = len(getattr(self.model, "mtp_layers", []))

        for layer_idx in range(num_mtp_layers):
            mappings.update(self._create_mtp_layer_mappings(layer_idx))

        return mappings

    def _create_mtp_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.mtp_layers.{layer_idx}"
        target_prefix = prefix

        mappings = {
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
            f"{prefix}.hidden_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.hidden_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.token_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.token_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.final_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.final_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.input_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.input_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        if getattr(self.config, "attention_bias", True):
            mappings.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                    ),
                    f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                }
            )

        return mappings

    def _load_weight_map(self, model_path: str) -> dict:
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            return {}
        try:
            with open(index_path, "r", encoding="utf-8") as index_file:
                data = json.load(index_file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read MiMo weight map from %s: %s", index_path, exc)
            return {}
        return data.get("weight_map", {})

    def _infer_mtp_layer_count(self) -> int:
        weight_map = getattr(self, "_weight_map", {}) or {}
        layer_ids = set()
        for key in weight_map.keys():
            if not key.startswith("model.mtp_layers."):
                continue
            parts = key.split(".")
            if len(parts) < 3:
                continue
            layer_index = parts[2]
            if layer_index.isdigit():
                layer_ids.add(int(layer_index))
        return len(layer_ids)


EntryClass = MiMoForCausalLM
