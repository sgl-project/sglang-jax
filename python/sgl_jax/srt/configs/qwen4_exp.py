"""Qwen4Exp (Qwen3.8-Flash-Next) hybrid-attention MoE config.

Defined from ``PretrainedConfig`` directly rather than inheriting from a
Qwen3.5 / Qwen3-Next config, for two reasons: the repo's other hybrid
families (bailing, kimi, qwen3_5, gemma4) each own their config outright,
and Qwen calls Flash-Next a prototype of the next-generation Qwen4 series,
so it is expected to diverge.

The backbone matches Qwen3.5 -- GDN linear attention on most layers, full
attention every ``full_attention_interval``, MoE FFN -- so ``layer_types``,
the RoPE flattening and ``linear_state_params`` mirror
``configs/qwen3_5.py`` and are consumed by the same
``model_runner_kv_cache_mixin`` helpers via duck typing.

Three mechanisms have no Qwen3.5 counterpart and are surfaced here for the
layers that will consume them: hyper connections (``hc_*``), the N-gram
embedding layer (``ple_*`` / ``ngram_*``), and Qwen Sparse Attention
(``indexer_*``). Validation mirrors the upstream vLLM config so a bad
checkpoint fails here rather than deep in a kernel.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

__all__ = ["Qwen4ExpConfig", "get_qwen4_exp_config"]

# QSA is configured as a block: either every field is present or none are.
_QSA_FIELDS = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)

# The block-level top-k the QSA kernels are compiled for.
_QSA_ALLOWED_BLOCK_TOPK = (512, 2048)

_OUTPUT_GATE_TYPES = ("silu", "swish", "sigmoid")


class _Qwen4ExpTextConfig(PretrainedConfig):
    """Text-side sub-config for Qwen4Exp.

    Mirrors the HF ``config.json -> text_config`` schema. Unlike upstream
    vLLM, the QSA and N-gram sharding fields are declared explicitly rather
    than absorbed into ``**kwargs``, so the interface is self-documenting
    and the validators read off named attributes.
    """

    model_type = "qwen4_exp_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 2560,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 24,
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
        rope_parameters: dict | None = None,
        full_attention_interval: int = 4,
        layer_types: list[str] | None = None,
        # GDN (Gated DeltaNet) linear-attn fields.
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 48,
        mamba_ssm_dtype: str = "float32",
        # Output gate on the GDN branch. Qwen3.5 ships "swish", Flash-Next
        # "sigmoid"; the repo has always hardcoded silu, which is only
        # accidentally right for Qwen3.5. Consumed by the model module.
        output_gate_type: str = "sigmoid",
        # Hyper connections. The inter-block residual carries hc_count
        # parallel streams, so backbone hidden states are hidden_size *
        # hc_count wide. See layers/gated_residual.py.
        hc_count: int = 4,
        hc_lowrank: int = 320,
        # N-gram embedding ("ple" in the config and the checkpoint).
        ple_layer_ids: list[int] | None = None,
        ple_embed_dim: int | None = None,
        ple_conv_kernel_size: int = 4,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20_000_000,
        make_ngram_vocab_size_divisible_by: int = 128,
        split_ngram_parts: int = 128,
        # QSA (Qwen Sparse Attention). All-or-nothing; see _validate_qsa.
        indexer_budget: int | None = None,
        indexer_compress_ratio: int | None = None,
        indexer_head_dim: int | None = None,
        indexer_n_heads: int | None = None,
        indexer_kv_heads: int | None = None,
        # FFN: MoE ships the moe_* / num_experts* block, dense would ship
        # intermediate_size. Both default to None so is_moe is unambiguous.
        intermediate_size: int | None = None,
        moe_intermediate_size: int | None = None,
        shared_expert_intermediate_size: int | None = None,
        num_experts_per_tok: int | None = None,
        num_experts: int | None = None,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        # MTP — out of scope for inference, surfaced for the weight-loader
        # whitelist. `mtp` is HF's nested block and carries its own
        # layer_types, separate from the backbone list above.
        mtp: dict | None = None,
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

        # Flatten the RoPE alias once at construction. HF nests it under
        # ``rope_parameters`` (5.x) or ships it flat (4.x).
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
            self.rope_scaling = kwargs.pop("rope_scaling", None)
            self.rope_theta = kwargs.pop("rope_theta", 1.0e7)
            self.partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.25)

        # Layer-type schedule. Flash-Next ships the explicit list; honor it
        # when present, else synthesize from the interval. The assert keeps a
        # future variant from folding MTP layers into the backbone list,
        # which would give full_attention_layer_ids phantom entries and
        # over-allocate the KV pool.
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
        self.mamba_ssm_dtype = mamba_ssm_dtype
        self.output_gate_type = self._normalize_output_gate_type(output_gate_type)

        # Hyper connections.
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank

        # N-gram embedding.
        self.ple_layer_ids = list(ple_layer_ids) if ple_layer_ids else []
        self.ple_embed_dim = hidden_size if ple_embed_dim is None else ple_embed_dim
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.split_ngram_parts = split_ngram_parts

        # QSA.
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio
        self.indexer_head_dim = indexer_head_dim
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads

        # FFN.
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = list(mlp_only_layers) if mlp_only_layers else []

        # MTP.
        self.mtp = mtp
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

        self._validate_hyper_connections()
        self._validate_ngram()
        self._validate_qsa()

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @staticmethod
    def _normalize_output_gate_type(output_gate_type: str) -> str:
        if output_gate_type not in _OUTPUT_GATE_TYPES:
            raise ValueError(
                f"output_gate_type must be one of {_OUTPUT_GATE_TYPES}, got {output_gate_type!r}"
            )
        return "silu" if output_gate_type == "swish" else output_gate_type

    def _validate_hyper_connections(self) -> None:
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4Exp requires hc_count > 1, got {self.hc_count}")
        if self.hc_lowrank <= 0:
            raise ValueError(f"hc_lowrank must be positive, got {self.hc_lowrank}")

    def _validate_ngram(self) -> None:
        if not self.ple_layer_ids:
            return
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be positive, got {self.heads_per_ngram}")
        if self.ple_conv_kernel_size <= 0:
            raise ValueError(
                f"ple_conv_kernel_size must be positive, got {self.ple_conv_kernel_size}"
            )
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ple_embed_dim <= 0 or self.ple_embed_dim % ngram_heads:
            raise ValueError(
                "ple_embed_dim must be positive and divisible by the total ngram heads: "
                f"{self.ple_embed_dim} % {ngram_heads} != 0"
            )
        # ple_layer_ids are 1-based in the checkpoint; see short_conv_layer_ids.
        invalid = [i for i in self.ple_layer_ids if not 1 <= int(i) <= self.num_hidden_layers]
        if invalid:
            raise ValueError(
                "ple_layer_ids are 1-based and must refer to an existing layer; "
                f"got {invalid} for {self.num_hidden_layers} layers"
            )

    def _validate_qsa(self) -> None:
        configured = {name: getattr(self, name) for name in _QSA_FIELDS}
        if all(value is None for value in configured.values()):
            return
        missing = [name for name, value in configured.items() if value is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}")
        if any(int(value) <= 0 for value in configured.values()):
            raise ValueError(f"QSA config values must be positive: {configured}")
        if int(self.indexer_kv_heads) != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if int(self.indexer_budget) % int(self.indexer_compress_ratio):
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        if self.indexer_block_topk not in _QSA_ALLOWED_BLOCK_TOPK:
            raise ValueError(
                "QSA requires indexer_budget / indexer_compress_ratio to be one of "
                f"{_QSA_ALLOWED_BLOCK_TOPK}, got {self.indexer_block_topk}"
            )
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim > int(self.indexer_head_dim):
            raise ValueError(
                "QSA indexer_head_dim must cover the attention rotary dimension, got "
                f"{self.indexer_head_dim} < {rotary_dim}"
            )

    @property
    def is_moe(self) -> bool:
        """Single dense/MoE discriminator, as in Qwen3.5."""
        return self.num_experts is not None

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
    def hyper_hidden_size(self) -> int:
        """Width of the inter-block residual, which carries hc_count streams."""
        return self.hidden_size * self.hc_count

    @property
    def indexer_block_topk(self) -> int | None:
        """Blocks each QSA query attends, the static shape of the QSA kernels."""
        if self.indexer_budget is None or self.indexer_compress_ratio is None:
            return None
        return int(self.indexer_budget) // int(self.indexer_compress_ratio)

    @property
    def ngram_context_len(self) -> int:
        """Previous token ids the N-gram layer keeps to form its keys."""
        return max(self.ngram_size - 1, 0) if self.ple_layer_ids else 0

    @property
    def short_conv_layer_ids(self) -> list[int]:
        """N-gram layers as 0-based indices (``ple_layer_ids`` is 1-based)."""
        return sorted({int(i) - 1 for i in self.ple_layer_ids})

    @property
    def short_conv_state_shape(self) -> tuple[int, int] | None:
        """Conv state carried by the N-gram layer, separate from GDN's.

        The channel count happens to equal GDN's ``conv_dim`` for this
        checkpoint (2*16*128 + 48*128 == 2560*4 == 10240) but the two are
        unrelated: this conv is dilated by ``ngram_size`` and holds
        ``(kernel - 1) * ngram_size`` positions, GDN's holds ``kernel - 1``.
        """
        if not self.short_conv_layer_ids:
            return None
        state_len = (self.ple_conv_kernel_size - 1) * self.ngram_size
        return self.hidden_size * self.hc_count, state_len

    @property
    def linear_state_params(self):
        """Sizing block for ``RecurrentStatePool`` over the GDN layers."""
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


class Qwen4ExpConfig(PretrainedConfig):
    """Root config for Qwen3.8-Flash-Next.

    The checkpoint carries a vision sub-config; text-only inference does not
    construct vision layers, so it is accepted as a plain dict.
    """

    model_type = "qwen4_exp"
    sub_configs = {"text_config": _Qwen4ExpTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    # Experts are merged at export, as in Qwen3.5.
    moe_pre_fused: bool = True

    def __init__(
        self,
        text_config: dict | None = None,
        vision_config: dict | None = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        language_model_only: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config = _Qwen4ExpTextConfig(**text_config)
        self.text_config = text_config

        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.language_model_only = language_model_only

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def get_qwen4_exp_config(hf_config: Any) -> Qwen4ExpConfig | None:
    """Return the hf_config cast to ``Qwen4ExpConfig``, else ``None``.

    Mirrors ``get_qwen3_5_hybrid_config`` so the runner can dispatch by duck
    typing rather than by isinstance.
    """
    if getattr(hf_config, "model_type", None) == "qwen4_exp":
        return hf_config
    return None
