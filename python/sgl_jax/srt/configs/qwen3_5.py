"""Qwen3.5 hybrid-attention MoE config (sgl-jax local definition).

We define the config from ``PretrainedConfig`` directly rather than
inheriting from ``transformers.Qwen3_5MoeConfig`` so the code works
across the transformers versions seen in production (4.57+) and dev
(5.3+ which ships a built-in ``qwen3_5_moe`` family).

HF stores RoPE under ``text_config.rope_parameters``; sgl-jax
consumers (``get_hf_text_config``, RoPE builders, model classes)
expect flat ``rope_scaling`` / ``rope_theta`` /
``partial_rotary_factor``. We flatten on construction so downstream
helpers don't need to change.

``Qwen3_5HybridConfig`` mirrors ``BailingHybridConfig`` /
``KimiLinearConfig`` enough that ``model_runner_kv_cache_mixin``
helpers (``_linear_state_params_from_config`` etc.) can consume it
via the same duck-typing.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

__all__ = ["Qwen3_5HybridConfig", "get_qwen3_5_hybrid_config"]


class _Qwen3_5TextConfig(PretrainedConfig):
    """Text-side sub-config for Qwen3.5 hybrid MoE.

    Mirrors the HF 35B-A3B ``config.json -> text_config`` schema; only
    fields used by sgl-jax (model construction, KV pool sizing, GDN
    recurrent state) are declared. Anything else is absorbed into
    ``**kwargs`` so we forward-compat with new HF fields.
    """

    model_type = "qwen3_5_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 2048,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attn_output_gate: bool = True,
        rope_parameters: dict | None = None,
        full_attention_interval: int = 4,
        layer_types: list[str] | None = None,
        # GDN (Gated DeltaNet) linear-attn fields.
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        # MoE fields.
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        # MTP (multi-token prediction) — not used by base inference but
        # surfaced for ``adjust_layer_num`` / weight-loader whitelist.
        mtp_num_hidden_layers: int = 1,
        mtp_use_dedicated_embeddings: bool = False,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attn_output_gate = attn_output_gate

        # Flatten RoPE alias once at construction time. HF nests under
        # ``rope_parameters`` (5.x) or ships flat (4.x).
        self.rope_parameters = rope_parameters
        if rope_parameters is not None:
            self.rope_scaling = {
                "rope_type": rope_parameters["rope_type"],
                "mrope_section": rope_parameters["mrope_section"],
                "mrope_interleaved": rope_parameters["mrope_interleaved"],
            }
            self.rope_theta = rope_parameters["rope_theta"]
            self.partial_rotary_factor = rope_parameters["partial_rotary_factor"]
        else:
            # Fallback: HF 4.x flat layout.
            self.rope_scaling = kwargs.pop("rope_scaling", None)
            self.rope_theta = kwargs.pop("rope_theta", 1.0e7)
            self.partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.25)

        # Layer-type schedule. HF ships an explicit ``layer_types`` list
        # in 35B-A3B's config.json; older revisions only had the
        # ``full_attention_interval`` regular schedule. Honor the explicit
        # list when present, else synthesize from the interval.
        self.full_attention_interval = full_attention_interval
        if layer_types is not None:
            assert len(layer_types) == num_hidden_layers
            self.layer_types = list(layer_types)
        else:
            self.layer_types = [
                "linear_attention" if (i + 1) % full_attention_interval else "full_attention"
                for i in range(num_hidden_layers)
            ]

        # GDN.
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE.
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = list(mlp_only_layers) if mlp_only_layers else []

        # MTP.
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            i
            for i, t in enumerate(self.layer_types)
            if str(t).lower() in {"full_attention", "attention"}
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if str(t).lower() == "linear_attention"]

    @property
    def linear_state_params(self):
        """Sizing block for ``RecurrentStatePool`` over GDN layers.

        Mirrors ``BailingHybridConfig.linear_state_params``; populates
        the K-side fields so the pool tracks GDN's asymmetric Q/K widths
        (``conv_dim = 2·key_dim + value_dim``).
        """
        from sgl_jax.srt.mem_cache.recurrent_state_pool import (
            LinearRecurrentStateParams,
            recurrent_state_dtype,
        )

        return LinearRecurrentStateParams(
            layers=self.linear_layer_ids,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            conv_kernel_size=self.linear_conv_kernel_dim,
            dtype=recurrent_state_dtype(),
            num_k_heads=self.linear_num_key_heads,
            head_k_dim=self.linear_key_head_dim,
        )


class Qwen3_5HybridConfig(PretrainedConfig):
    """Root config for Qwen3.5-35B-A3B (hybrid attention + MoE).

    HF 35B-A3B carries a vision sub-config; M1 is text-only, so the
    vision side is accepted but unused. ``AutoConfig.register(...,
    exist_ok=True)`` is needed when running against transformers 5.3+
    (which ships its own ``qwen3_5_moe`` class).
    """

    model_type = "qwen3_5_moe"
    sub_configs = {"text_config": _Qwen3_5TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    # 35B-A3B disk layout is pre-fused (gate_up + experts merged at export).
    # Flip if a future variant ships per-expert weights.
    moe_pre_fused: bool = True

    def __init__(
        self,
        text_config: dict | None = None,
        vision_config: dict | None = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config = _Qwen3_5TextConfig(**text_config)
        self.text_config = text_config

        # Keep vision_config as a plain dict — base inference path doesn't
        # construct vision layers, so we don't need a typed sub-config.
        self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def get_qwen3_5_hybrid_config(hf_config: Any) -> Qwen3_5HybridConfig | None:
    """Return the hf_config cast to ``Qwen3_5HybridConfig``, else ``None``.

    Mirrors ``get_kimi_linear_config`` / ``get_bailing_hybrid_config`` so
    the runner can dispatch via duck-typing.
    """
    if getattr(hf_config, "model_type", None) == "qwen3_5_moe":
        return hf_config
    return None
