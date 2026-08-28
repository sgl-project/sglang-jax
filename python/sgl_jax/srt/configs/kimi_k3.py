"""Kimi-K3 config.

K3 is Kimi-Linear plus two architectural additions, so this subclasses ``KimiLinearConfig``
rather than restating it. Everything K3 shares with Kimi-Linear -- ``is_kda_layer``, the
``linear_attn_config`` KDA block, the MoE fields (``num_experts``, ``num_experts_per_token``,
``num_shared_experts``, ``first_k_dense_replace``, ``moe_layer_freq``, grouped-topk), and the MLA
fields (``kv_lora_rank``, ``qk_nope_head_dim``, ``qk_rope_head_dim``, ``mla_use_nope``) -- is
inherited unchanged.

The K3-only fields, derived by diffing every ``config.*`` access in
the K3 torch reference against ``KimiLinearConfig``:

===============================  ====================================================
``activation_situ_beta``         SITU gate soft-clip bound (``beta``)
``activation_situ_linear_beta``  SITU *up*-branch soft-clip; ``None`` disables it
``attn_res_block_size``          AttnRes checkpoint period; ``None`` disables AttnRes
                                 entirely and the layer falls back to a plain
                                 additive residual
``latent_moe_use_norm``          normalize the LatentMoE output
``mla_use_output_gate``          K3's MLA adds an output gate Kimi-Linear lacks
``media_placeholder_token_id``   multimodal; unused on the text-only path
===============================  ====================================================
"""

from __future__ import annotations

from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig


class KimiK3Config(KimiLinearConfig):
    """Config for Moonshot Kimi-K3 (2.8T total / 104B active, 93 layers = 69 KDA + 24 MLA)."""

    model_type = "kimi_k3"

    def __init__(
        self,
        *args,
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        attn_res_block_size: int | None = None,
        latent_moe_use_norm: bool = False,
        mla_use_output_gate: bool = False,
        media_placeholder_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.attn_res_block_size = attn_res_block_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.mla_use_output_gate = mla_use_output_gate
        self.media_placeholder_token_id = media_placeholder_token_id

    @property
    def uses_attn_res(self) -> bool:
        """AttnRes is active only when a block size is configured.

        The reference keys the entire two-AttnRes-per-layer path off
        ``attn_res_block_size is not None`` and otherwise runs an ordinary pre-norm residual, so
        this is the single switch between the two decoder-layer forward paths.
        """
        return self.attn_res_block_size is not None

    @property
    def uses_situ(self) -> bool:
        return getattr(self, "hidden_act", "silu") == "situ"

    def n_attn_res_candidates(self, layer_idx: int) -> int:
        """Candidates the AttnRes softmax spans entering ``layer_idx`` (shape oracle)."""
        if not self.uses_attn_res:
            return 0
        return layer_idx // self.attn_res_block_size + 1
