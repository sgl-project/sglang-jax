"""Config for GLM-5.3-Flash (`glm5_next`) and relatives.

GLM-5.3-Flash is a hybrid model: 45 layers of `[3x KDA linear attention, 1x NoPE sparse MLA]`, a 288-expert
MoE (top-8 + 1 shared), four-stream manifold-constrained hyper-connections (mHC), and a K-pool-indexed DSA
sparse attention. It reuses sglang-jax's KDA backend and DSA sparse backend.

This config is deliberately shaped like `BailingHybridConfig` / `KimiLinearConfig` so the existing hybrid
plumbing (`attn_backend_windows`, `RecurrentStatePool`, `HybridLinearKVPool`) works unchanged:

  * `use_kda` / `linear_layer_ids` / `full_attention_layer_ids` / `is_kda_layer`
  * `linear_state_params` (drives `RecurrentStatePool`)
  * `linear_attn_config` (read by `KDAAttnBackend` via the layer, not directly)

Detection: `glm5_next` model_type, or an architecture beginning with `Glm5Next.

Config facts (zai-org/GLM-5.3-Flash):
  hidden 4096, 45 layers, first_k_dense_replace 3, 288 experts top-8, moe_inter 2048,
  q_lora 1536, kv_lora 512, qk_nope 256, qk_rope 0 (NoPE), v_head 256, heads 64,
  KDA heads 64 head_dim 128 conv_k 4 gate_lower_bound -5.0,
  index heads 32 head_dim 128 topk 2048 kpool 4, hc_mult 4, swiglu_limit 10.0.
"""

from __future__ import annotations

import inspect
from typing import Any

from transformers import PretrainedConfig


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        n_routed_experts: int = 288,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        scoring_func: str = "sigmoid",
        first_k_dense_replace: int = 3,
        rms_norm_eps: float = 1e-5,
        swiglu_limit: float = 10.0,
        # attention layer schedule
        layer_types: list[str] | None = None,
        indexer_types: list[str] | None = None,
        # MLA (NoPE)
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        # Like GLM-5 (#1729): the checkpoint has no q_norm/k_norm tensors and never applies them.
        use_qk_norm: bool = False,
        # KDA (KimiDeltaAttention reads num_heads/head_dim/short_conv_kernel_size/gate_lower_bound)
        linear_attn_config: dict[str, Any] | None = None,
        # DSA indexer / K-pool
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        index_kpool: int = 4,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
        # mHC
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        rope_theta: float = 800000.0,
        max_position_embeddings: int = 1048576,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.first_k_dense_replace = first_k_dense_replace
        self.rms_norm_eps = rms_norm_eps
        self.swiglu_limit = swiglu_limit
        self.layer_types = list(layer_types) if layer_types else None
        # Every DSA layer carries a full indexer unless the checkpoint says otherwise.
        self.indexer_types = list(indexer_types or ["full"] * num_hidden_layers)
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.use_qk_norm = use_qk_norm
        self.linear_attn_config = linear_attn_config or {
            "num_heads": num_attention_heads,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        }
        self.use_kda = True
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        # Only the KPool indexer is implemented (as in PyTorch SGLang's GLM-5.3 indexer).
        if not (index_kpool > 1 and index_kpool_compress and index_kpool_always_select_tail):
            raise ValueError(
                "GLM-5.3-Flash requires the KPool indexer: index_kpool > 1, "
                "index_kpool_compress and index_kpool_always_select_tail"
            )
        self.index_kpool = index_kpool
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # ── layer taxonomy (mirrors BailingHybridConfig) ──
    def is_kda_layer(self, layer_idx: int) -> bool:
        if self.layer_types:
            return str(self.layer_types[layer_idx]).lower() == "linear_attention"
        return layer_idx % 4 != 3  # [3 KDA, 1 DSA]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if self.is_kda_layer(i)]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)]

    @property
    def head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def linear_state_params(self):
        from sgl_jax.srt.mem_cache.recurrent_state_pool import (
            LinearRecurrentStateParams,
            recurrent_state_dtype,
        )

        la = self.linear_attn_config
        return LinearRecurrentStateParams(
            layers=self.linear_layer_ids,
            num_heads=la["num_heads"],
            head_dim=la["head_dim"],
            conv_kernel_size=la["short_conv_kernel_size"],
            dtype=recurrent_state_dtype(),
        )


# Text-config fields copied into Glm5NextConfig when the checkpoint sets them.
_FIELDS = tuple(
    name
    for name, param in inspect.signature(Glm5NextConfig.__init__).parameters.items()
    if name != "self" and param.kind is not param.VAR_KEYWORD
)


def get_glm5_next_config(hf_config: Any) -> Glm5NextConfig | None:
    """Return a normalized Glm5NextConfig for an HF glm5_next config, else None.

    ``hf_config`` is the checkpoint root config (``text_config`` nested) or the text
    config itself; fields it leaves unset keep the Glm5NextConfig defaults.
    """
    model_type = getattr(hf_config, "model_type", None)
    archs = getattr(hf_config, "architectures", None) or []
    if model_type not in ("glm5_next", "glm5_next_text") and not any(
        str(a).startswith("Glm5Next") for a in archs
    ):
        return None
    text = getattr(hf_config, "text_config", hf_config)
    fields = {name: getattr(text, name, None) for name in _FIELDS}
    return Glm5NextConfig(**{k: v for k, v in fields.items() if v is not None})


class Glm5NextTextHFConfig(PretrainedConfig):
    """Checkpoint ``text_config`` (``glm5_next_text``) for transformers releases without it.

    Fields are stored as written in config.json; ``get_glm5_next_config`` normalizes them.
    """

    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]


class Glm5NextHFConfig(PretrainedConfig):
    """Checkpoint root config (``glm5_next``) for transformers releases without it.

    Only the text model is served; ``vision_config`` is kept as written.
    """

    model_type = "glm5_next"
    sub_configs = {"text_config": Glm5NextTextHFConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, text_config: dict | None = None, vision_config: dict | None = None, **kwargs
    ):
        if isinstance(text_config, dict):
            text_config = Glm5NextTextHFConfig(**text_config)
        self.text_config = text_config if text_config is not None else Glm5NextTextHFConfig()
        self.vision_config = vision_config
        super().__init__(**kwargs)
