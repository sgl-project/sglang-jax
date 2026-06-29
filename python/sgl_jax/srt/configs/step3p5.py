"""Step 3.5 Flash config (sgl-jax local definition).

Step3p5Config mirrors the HF ``configuration_step3p5.Step3p5Config``
schema. We define it locally so the model can be loaded without
``trust_remote_code=True`` and to expose the ``num_key_value_heads``
alias expected by ``ModelConfig`` (HF uses ``num_attention_groups``).

Key fields consumed by sgl-jax:
- ``layer_types``: per-layer list of "full_attention" / "sliding_attention"
- ``rope_theta``: per-layer list of floats (one per hidden layer)
- ``partial_rotary_factors``: per-layer list of floats
- ``attention_other_setting``: dict describing the sliding-attention head config
- ``swiglu_limits`` / ``swiglu_limits_shared``: per-layer SwiGLU gate limits
- ``moe_layers_enum``: comma-separated string or list of MoE layer indices
- ``head_dim``: Q/K/V head dimension (128)
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

__all__ = ["Step3p5Config"]


class Step3p5Config(PretrainedConfig):
    """Local config class for Step 3.5 Flash.

    HF field ``num_attention_groups`` is surfaced as ``num_key_value_heads``
    so that ``ModelConfig`` picks up the correct KV-head count without
    needing a special-case.
    """

    model_type = "step3p5"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11264,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_attention_groups: int = 8,
        head_dim: int = 128,
        vocab_size: int = 128896,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 262144,
        rope_theta: float | list[float] = 5000000.0,
        rope_scaling: dict[str, Any] | None = None,
        layer_types: list[str] | None = None,
        partial_rotary_factors: list[float] | None = None,
        attention_other_setting: dict[str, Any] | None = None,
        swiglu_limits: list[float] | None = None,
        swiglu_limits_shared: list[float] | None = None,
        moe_layers_enum: str | list[int] | None = None,
        moe_num_experts: int = 288,
        moe_top_k: int = 8,
        moe_intermediate_size: int = 1280,
        share_expert_dim: int = 1280,
        share_expert_dims: int | None = None,
        moe_router_activation: str = "sigmoid",
        moe_router_scaling_factor: float = 3.0,
        norm_expert_weight: bool = True,
        use_moe_router_bias: bool = True,
        use_qk_norm: bool = True,
        use_head_wise_attn_gate: bool = True,
        sliding_window: int | None = 512,
        att_impl_type: str = "GQA",
        num_nextn_predict_layers: int = 3,
        use_rope_layers: list[int] | None = None,
        yarn_only_types: list[str] | None = None,
        need_fp32_gate: bool = True,
        sink: bool = False,
        zero_centered: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        # Expose as num_key_value_heads so ModelConfig picks up GQA ratio.
        self.num_key_value_heads = num_attention_groups
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.layer_types = list(layer_types) if layer_types is not None else None
        self.partial_rotary_factors = (
            list(partial_rotary_factors) if partial_rotary_factors is not None else None
        )
        self.attention_other_setting = attention_other_setting
        self.swiglu_limits = list(swiglu_limits) if swiglu_limits is not None else None
        self.swiglu_limits_shared = (
            list(swiglu_limits_shared) if swiglu_limits_shared is not None else None
        )
        self.moe_layers_enum = moe_layers_enum
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        # Canonical aliases for shared infra that reads the standard names:
        # EPLB expert_location reads ``num_experts``; the routed-experts capturer
        # (tp_worker) reads ``num_experts_per_tok``. (Same pattern as
        # num_key_value_heads = num_attention_groups above.)
        self.num_experts = moe_num_experts
        self.num_experts_per_tok = moe_top_k
        self.moe_intermediate_size = moe_intermediate_size
        # `share_expert_dims` (plural) is the canonical name in the HF checkpoint
        # config.json and upstream sglang's Step3p5Config; it wins when present.
        # `share_expert_dim` (singular) is kept for internal/test callers. Both map
        # to self.share_expert_dim (mirrors upstream's self.share_expert_dim = share_expert_dims).
        self.share_expert_dim = (
            share_expert_dims if share_expert_dims is not None else share_expert_dim
        )
        self.moe_router_activation = moe_router_activation
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.norm_expert_weight = norm_expert_weight
        self.use_moe_router_bias = use_moe_router_bias
        self.use_qk_norm = use_qk_norm
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.sliding_window = sliding_window
        self.att_impl_type = att_impl_type
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.use_rope_layers = list(use_rope_layers) if use_rope_layers is not None else []
        self.yarn_only_types = list(yarn_only_types) if yarn_only_types is not None else []
        self.need_fp32_gate = need_fp32_gate
        self.sink = sink
        self.zero_centered = zero_centered

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
