# DeepSeek-V4 configuration and the single source of truth for layer typing.
#
# Every consumer that needs to know "what kind of layer is layer i" must call
# ``classify_layers`` here. Three separate places need it -- the model
# (M1), the KV/state pools (C1.1), and the attention metadata derivation
# (M2.1) -- and if each re-derives it from ``compress_ratios`` they will drift.
from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Any

from transformers.configuration_utils import PretrainedConfig

__all__ = [
    "DeepseekV4Config",
    "DeepseekV4LayerType",
    "classify_layers",
    "hash_moe_layer_flags",
    "layer_type_for_compress_ratio",
    "mhc_param_shapes",
    "mix_hc_width",
    "trunk_compress_ratios",
]


class DeepseekV4LayerType(enum.Enum):
    """Attention flavour of a trunk layer, keyed off its compression ratio.

    Names mirror upstream vLLM's ``_layer_type_for`` (``swaonly`` / ``c4a`` /
    ``c128a``) so cross-reading the two stacks stays mechanical.
    """

    # No compressed history: sliding-window attention only.
    SWA_ONLY = "swaonly"
    # Compression ratio 4: CSA -- sparse MLA driven by the indexer's top-k.
    C4A = "c4a"
    # Compression ratio 128: HCA -- dense attention over compressed history.
    C128A = "c128a"


def layer_type_for_compress_ratio(compress_ratio: int) -> DeepseekV4LayerType:
    """Map one ``compress_ratios`` entry to a layer type.

    Ratio 0 and 1 both mean "no compressed history"; the shipped Flash 0731
    config uses 0 for its first two and last trunk layers, while vLLM's default
    for a model without ``compress_ratios`` at all is 1.

    Raises on any other ratio on purpose: a silent fallback would mean a layer
    quietly attends over the wrong cache.
    """
    if compress_ratio <= 1:
        return DeepseekV4LayerType.SWA_ONLY
    if compress_ratio == 4:
        return DeepseekV4LayerType.C4A
    if compress_ratio == 128:
        return DeepseekV4LayerType.C128A
    raise ValueError(
        f"Unsupported DeepSeek-V4 compress_ratio={compress_ratio}; expected 0, 1, 4, or 128."
    )


def trunk_compress_ratios(config: Any) -> tuple[int, ...]:
    """The ``num_hidden_layers`` ratios that belong to the trunk.

    ``compress_ratios`` in the shipped Flash 0731 checkpoint is **longer than
    the trunk** (46 entries for 43 layers); the tail belongs to other config
    surfaces, not to extra decoder layers. Truncating here is what keeps those
    trailing entries from being mistaken for layers.
    """
    num_layers = int(config.num_hidden_layers)
    ratios = getattr(config, "compress_ratios", None)
    if ratios is None:
        # A model with no compressed history at all: every layer is SWA-only.
        return (1,) * num_layers
    if not isinstance(ratios, Sequence) or isinstance(ratios, (str, bytes)):
        raise TypeError(f"compress_ratios must be a sequence, got {type(ratios).__name__}")
    if len(ratios) < num_layers:
        raise ValueError(
            f"compress_ratios has {len(ratios)} entries but num_hidden_layers="
            f"{num_layers}; cannot classify every trunk layer."
        )
    return tuple(int(r) for r in ratios[:num_layers])


def classify_layers(config: Any) -> tuple[DeepseekV4LayerType, ...]:
    """Per-trunk-layer attention type. Length is exactly ``num_hidden_layers``."""
    return tuple(layer_type_for_compress_ratio(r) for r in trunk_compress_ratios(config))


def hash_moe_layer_flags(config: Any) -> tuple[bool, ...]:
    """Per-trunk-layer flag: does this layer route experts by token id?

    The first ``num_hash_layers`` layers use a vocab-indexed routing table
    instead of gate logits. Flash 0731 ships ``num_hash_layers=3``.
    """
    num_layers = int(config.num_hidden_layers)
    num_hash = int(getattr(config, "num_hash_layers", 0) or 0)
    if num_hash < 0:
        raise ValueError(f"num_hash_layers must be >= 0, got {num_hash}")
    if num_hash > num_layers:
        raise ValueError(f"num_hash_layers={num_hash} exceeds num_hidden_layers={num_layers}.")
    return tuple(i < num_hash for i in range(num_layers))


def mix_hc_width(hc_mult: int) -> int:
    """Rows of an mHC mixing matrix: pre(m) + post(m) + comb(m*m).

    Must stay equal to ``sgl_jax.srt.kernels.mhc.mix_hc_width``; duplicated
    rather than imported so the config module carries no Pallas dependency.
    ``test_deepseek_v4_config.py`` pins the two together.
    """
    return (2 + hc_mult) * hc_mult


def mhc_param_shapes(config: Any) -> dict[str, tuple[int, ...]]:
    """Shapes of the mHC parameters, for weight mapping (M1.1b) and the layer (M1.2).

    All of these are float32 in the checkpoint and must stay float32 -- they are
    gate/mixing coefficients fed to a Sinkhorn normalisation, not projections,
    so they do not follow the model's bf16 activation dtype.
    """
    hc_mult = int(config.hc_mult)
    hidden_size = int(config.hidden_size)
    mix_hc = mix_hc_width(hc_mult)
    hc_dim = hc_mult * hidden_size
    return {
        # Per-layer, one set for the attention sublayer and one for the FFN.
        "fn": (mix_hc, hc_dim),
        "base": (mix_hc,),
        "scale": (3,),
        # Model-level, applied once after the last layer.
        "head_fn": (hc_mult, hc_dim),
        "head_base": (hc_mult,),
        "head_scale": (1,),
    }


# The shipped Flash 0731 YaRN setting. Kept as the default because transformers'
# rope validation rewrites a ``None`` into ``{"rope_type": "default"}``, which
# would silently drop the 16x context extension.
_FLASH_0731_ROPE_SCALING: dict = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "type": "yarn",
}

# The shipped Flash 0731 list: two uncompressed layers, then 4/128 alternating,
# then a trailing 4 -- that is 43 trunk layers -- followed by three entries that
# are NOT layers. 46 total against ``num_hidden_layers=43``; see
# ``trunk_compress_ratios`` for why the tail must be truncated.
# ``test_deepseek_v4_config.py`` pins this against the real config.json.
_FLASH_0731_COMPRESS_RATIOS: list[int] = [0, 0] + [4, 128] * 20 + [4] + [0, 0, 0]


class DeepseekV4Config(PretrainedConfig):
    """DeepSeek-V4 (Flash 0731) config.

    Defaults are the shipped ``deepseek-ai/DeepSeek-V4-Flash-0731`` values so a
    bare ``DeepseekV4Config()`` is a usable fixture. Fields the first version
    does not consume (``dspark_*``, ``num_nextn_predict_layers``) are still
    accepted and stored, because dropping them would make a round-trip through
    this class lossy.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1024,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        compress_rope_theta: float = 160000.0,
        sliding_window: int = 128,
        compress_ratios: Sequence[int] | None = None,
        # mHC
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # MoE
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 6,
        n_shared_experts: int = 1,
        moe_intermediate_size: int = 2048,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        swiglu_limit: float | None = 10.0,
        num_hash_layers: int = 3,
        # indexer (CSA top-k)
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        # weights
        expert_dtype: str = "fp4",
        tie_word_embeddings: bool = False,
        # accepted but not consumed by the first version
        num_nextn_predict_layers: int = 1,
        dspark_block_size: int | None = 5,
        dspark_noise_token_id: int | None = 128799,
        dspark_target_layer_ids: Sequence[int] | None = None,
        dspark_markov_rank: int | None = 256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = (
            dict(rope_scaling) if rope_scaling is not None else dict(_FLASH_0731_ROPE_SCALING)
        )
        self.compress_rope_theta = compress_rope_theta
        self.sliding_window = sliding_window
        self.compress_ratios = (
            list(compress_ratios) if compress_ratios is not None else _FLASH_0731_COMPRESS_RATIOS
        )

        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.swiglu_limit = swiglu_limit
        self.num_hash_layers = num_hash_layers

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        self.expert_dtype = expert_dtype

        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = (
            list(dspark_target_layer_ids) if dspark_target_layer_ids is not None else [40, 41, 42]
        )
        self.dspark_markov_rank = dspark_markov_rank

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # -- convenience wrappers so callers can go through the config object ----
    #
    # Deliberately NOT named ``layer_types``: transformers' PretrainedConfig
    # owns that name and rewrites it during validation (it expects HF's
    # "sliding_attention"/"full_attention" strings), and sgl-jax's
    # ``ModelConfig.get_hybrid_layer_counts`` reads it with those semantics.

    @property
    def attention_layer_types(self) -> tuple[DeepseekV4LayerType, ...]:
        return classify_layers(self)

    @property
    def hash_moe_layers(self) -> tuple[bool, ...]:
        return hash_moe_layer_flags(self)
