# Adapted from inclusionAI/Ling-V2 reference modeling code
# (configuration_bailing_moe_v3.py shipped with the Ling3 ckpt).
from transformers.configuration_utils import PretrainedConfig


class BailingMoeV3Config(PretrainedConfig):
    model_type = "bailing_moe_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=1536,
        intermediate_size=4608,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        head_dim=128,
        rms_norm_eps=1e-6,
        use_qkv_bias=False,
        use_bias=False,
        use_qk_norm=True,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleave=True,
        partial_rotary_factor=0.5,
        rotary_dim=64,
        # MLA
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        use_mla_nope=False,
        gated_attention_proj_granularity_type="head_wise",
        # KDA
        layer_group_size=4,
        no_kda_lora=True,
        kda_lower_bound=-5.0,
        short_conv_kernel_size=4,
        # MoE
        num_experts=128,
        num_experts_per_tok=8,
        num_shared_experts=1,
        moe_intermediate_size=512,
        moe_shared_expert_intermediate_size=512,
        first_k_dense_replace=1,
        moe_router_enable_expert_bias=True,
        norm_topk_prob=True,
        n_group=8,
        topk_group=4,
        topk_method="noaux_tc",
        score_function="sigmoid",
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        router_dtype="fp32",
        # MTP (skipped at inference for Ling3-Tiny)
        num_nextn_predict_layers=1,
        mtp_loss_scaling_factor=0,
        mtp_use_kda=False,
        # tokenizer
        pad_token_id=156892,
        eos_token_id=156892,
        tie_word_embeddings=False,
        # plumbing
        initializer_range=0.02,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.use_qk_norm = use_qk_norm
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleave = rope_interleave
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = rotary_dim

        # MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.mla_use_nope = use_mla_nope
        self.gated_attention_proj_granularity_type = (
            gated_attention_proj_granularity_type
        )

        # KDA
        self.layer_group_size = layer_group_size
        self.no_kda_lora = no_kda_lora
        self.kda_lower_bound = kda_lower_bound
        self.short_conv_kernel_size = short_conv_kernel_size

        # MoE — keep both ckpt-side and sglang-jax-side aliases populated
        self.n_routed_experts = self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts_per_token = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.norm_topk_prob = norm_topk_prob
        self.moe_renormalize = norm_topk_prob
        self.n_group = n_group
        self.num_expert_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.score_function = score_function
        self.scoring_func = scoring_func
        self.moe_router_activation_func = score_function
        self.routed_scaling_factor = routed_scaling_factor
        self.router_dtype = router_dtype

        # MTP
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.mtp_use_kda = mtp_use_kda

        # plumbing
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Derive the hybrid layer pattern (1-based, matches sglang-jax convention
        # in `configs/kimi_linear.py::is_kda_layer`).
        # Within each group of `layer_group_size` consecutive layers, the LAST
        # one is a full-attention (MLA) layer; the others are KDA.
        # For Ling3-Tiny: layer_group_size=4 → MLA at 0-based [3,7,11,15,19,23],
        # KDA at 0-based [0,1,2,4,5,6,8,9,10,12,...] = 1-based [1,2,3,5,6,7,...].
        kda_layers = [
            i
            for i in range(1, num_hidden_layers + 1)
            if i % layer_group_size != 0
        ]
        full_attn_layers = [
            i
            for i in range(1, num_hidden_layers + 1)
            if i % layer_group_size == 0
        ]
        self.linear_attn_config = {
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
            "head_dim": head_dim,
            "num_heads": num_attention_heads,
            "short_conv_kernel_size": short_conv_kernel_size,
        }

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self) -> bool:
        return True

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None and self.num_experts > 0

    @property
    def is_linear_attn(self) -> bool:
        return bool(self.linear_attn_config["kda_layers"])

    def is_kda_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) in self.linear_attn_config["kda_layers"]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if self.is_kda_layer(i)]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)
        ]
