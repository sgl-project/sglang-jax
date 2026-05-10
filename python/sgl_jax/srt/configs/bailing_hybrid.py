from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import PretrainedConfig


class BailingHybridConfig(PretrainedConfig):
    """Minimal Bailing hybrid config for Ling/Ring 2.5 linear-attention models."""

    model_type = "bailing_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 157184,
        hidden_size: int = 2048,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        use_qkv_bias: bool = False,
        use_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
        max_position_embeddings: int = 32768,
        rope_theta: float = 600000.0,
        rope_scaling: dict[str, Any] | None = None,
        pad_token_id: int = 156892,
        eos_token_id: int = 156892,
        num_experts: int = 256,
        num_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        moe_intermediate_size: int = 512,
        first_k_dense_replace: int = 1,
        head_dim: int | None = 128,
        use_qk_norm: bool = True,
        moe_router_enable_expert_bias: bool = True,
        norm_topk_prob: bool = False,
        routed_scaling_factor: float = 1.0,
        score_function: str = "sigmoid",
        router_dtype: str | None = None,
        layer_group_size: int = 1,
        layers_block_type: list[str] | None = None,
        group_norm_size: int = 1,
        linear_silu: bool = False,
        use_linear_silu: bool | None = None,
        linear_rope: bool = True,
        full_attention_type: str = "mla",
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = None,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        v_head_dim: int = 128,
        rope_interleave: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.use_qk_norm = use_qk_norm

        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.score_function = score_function
        self.router_dtype = router_dtype

        self.layer_group_size = layer_group_size
        self._layers_block_type = list(layers_block_type) if layers_block_type is not None else None
        self.group_norm_size = group_norm_size
        self.linear_silu = linear_silu if use_linear_silu is None else use_linear_silu
        self.use_linear_silu = self.linear_silu
        self.linear_rope = linear_rope
        self.num_linear_key_value_heads = num_attention_heads
        self.full_attention_type = full_attention_type

        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_interleave = rope_interleave

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self) -> list[str]:
        if self._layers_block_type is not None:
            return self._layers_block_type
        if self.layer_group_size <= 0:
            raise ValueError(f"layer_group_size must be positive, got {self.layer_group_size}")
        return [
            "attention" if (layer_id + 1) % self.layer_group_size == 0 else "linear_attention"
            for layer_id in range(self.num_hidden_layers)
        ]

    @layers_block_type.setter
    def layers_block_type(self, value: list[str] | None) -> None:
        self._layers_block_type = list(value) if value is not None else None

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            i
            for i, block_type in enumerate(self.layers_block_type)
            if str(block_type).lower() == "linear_attention"
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            i
            for i, block_type in enumerate(self.layers_block_type)
            if str(block_type).lower() in {"attention", "full_attention"}
        ]


@dataclass(frozen=True)
class BailingHybridLinearConfig:
    num_hidden_layers: int
    num_attention_heads: int
    num_linear_key_value_heads: int
    head_dim: int
    linear_layer_ids: list[int]
    full_attention_layer_ids: list[int]

    @property
    def linear_state_params(self):
        from sgl_jax.srt.mem_cache.recurrent_state_pool import (
            LinearRecurrentStateParams,
            recurrent_state_dtype,
        )

        return LinearRecurrentStateParams(
            layers=self.linear_layer_ids,
            num_heads=self.num_linear_key_value_heads,
            head_dim=self.head_dim,
            conv_kernel_size=1,
            dtype=recurrent_state_dtype(),
        )

    @property
    def linear_attn_config(self) -> dict[str, Any]:
        return {
            "kda_layers": self.linear_layer_ids,
            "num_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "short_conv_kernel_size": 1,
        }


def get_bailing_hybrid_config(hf_config: Any) -> BailingHybridLinearConfig | None:
    if not _is_bailing_hybrid_config(hf_config):
        return None

    num_hidden_layers = int(hf_config.num_hidden_layers)
    num_attention_heads = int(hf_config.num_attention_heads)
    num_linear_key_value_heads = int(
        getattr(hf_config, "num_linear_key_value_heads", num_attention_heads)
    )
    head_dim = int(hf_config.head_dim)
    linear_layer_ids, full_attention_layer_ids = _get_layer_ids(hf_config, num_hidden_layers)
    if not linear_layer_ids:
        return None

    return BailingHybridLinearConfig(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_linear_key_value_heads=num_linear_key_value_heads,
        head_dim=head_dim,
        linear_layer_ids=linear_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
    )


def _is_bailing_hybrid_config(hf_config: Any) -> bool:
    if getattr(hf_config, "model_type", None) == "bailing_hybrid":
        return True
    architectures = getattr(hf_config, "architectures", None) or []
    return any(str(arch) == "BailingMoeV2_5ForCausalLM" for arch in architectures)


def _get_layer_ids(hf_config: Any, num_hidden_layers: int) -> tuple[list[int], list[int]]:
    layers_block_type = getattr(hf_config, "layers_block_type", None)
    if layers_block_type is not None:
        if len(layers_block_type) != num_hidden_layers:
            raise ValueError(
                f"layers_block_type length ({len(layers_block_type)}) must match "
                f"num_hidden_layers ({num_hidden_layers})"
            )
        linear_layer_ids = []
        full_attention_layer_ids = []
        for layer_id, block_type in enumerate(layers_block_type):
            normalized = str(block_type).lower()
            if normalized == "linear_attention":
                linear_layer_ids.append(layer_id)
            elif normalized in {"attention", "full_attention"}:
                full_attention_layer_ids.append(layer_id)
            else:
                raise ValueError(f"Unsupported Bailing hybrid layer block type: {block_type}")
        return linear_layer_ids, full_attention_layer_ids

    layer_group_size = int(hf_config.layer_group_size)
    if layer_group_size <= 0:
        raise ValueError(f"layer_group_size must be positive, got {layer_group_size}")

    linear_layer_ids = []
    full_attention_layer_ids = []
    for layer_id in range(num_hidden_layers):
        if (layer_id + 1) % layer_group_size == 0:
            full_attention_layer_ids.append(layer_id)
        else:
            linear_layer_ids.append(layer_id)
    return linear_layer_ids, full_attention_layer_ids


__all__ = [
    "BailingHybridConfig",
    "BailingHybridLinearConfig",
    "get_bailing_hybrid_config",
]
