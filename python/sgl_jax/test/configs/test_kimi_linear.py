"""Tests for KimiLinearConfig (mirrors upstream sglang)."""

from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig


class TestKimiLinearConfigBasics:
    def test_model_type_attribute(self):
        assert KimiLinearConfig.model_type == "kimi_linear"

    def test_default_construction(self):
        cfg = KimiLinearConfig()
        # head_dim auto-derived: 4096 // 32 = 128
        assert cfg.head_dim == 128
        assert cfg.linear_attn_config is None
        assert cfg.is_linear_attn is False
        assert cfg.is_mla is False
        assert cfg.is_moe is False


class TestIsKdaLayer:
    def _make_cfg(self):
        # 1-indexed kda_layers: layer indices 1 and 3 (since (1+1)=2, (3+1)=4 ∈ kda_layers)
        return KimiLinearConfig(
            num_hidden_layers=4,
            linear_attn_config={"kda_layers": [2, 4], "full_attn_layers": [1, 3]},
        )

    def test_kda_vs_full_layers_1_indexed(self):
        cfg = self._make_cfg()
        assert cfg.is_kda_layer(0) is False
        assert cfg.is_kda_layer(1) is True  # 1+1=2 ∈ {2,4}
        assert cfg.is_kda_layer(2) is False
        assert cfg.is_kda_layer(3) is True  # 3+1=4 ∈ {2,4}

    def test_full_attention_layer_ids_property(self):
        cfg = self._make_cfg()
        assert cfg.full_attention_layer_ids == [0, 2]

    def test_linear_layer_ids_property(self):
        cfg = self._make_cfg()
        assert cfg.linear_layer_ids == [1, 3]

    def test_no_linear_attn_config(self):
        cfg = KimiLinearConfig(num_hidden_layers=2)
        assert cfg.is_linear_attn is False
        assert cfg.is_kda_layer(0) is False
        assert cfg.is_kda_layer(1) is False
        assert cfg.full_attention_layer_ids == [0, 1]
        assert cfg.linear_layer_ids == []

    def test_is_linear_attn_empty_kda_list(self):
        cfg = KimiLinearConfig(
            num_hidden_layers=1,
            linear_attn_config={"kda_layers": [], "full_attn_layers": [1]},
        )
        # Empty kda_layers list → not really linear-attn-based
        assert cfg.is_linear_attn is False


class TestIsMla:
    def test_q_lora_rank_set(self):
        assert KimiLinearConfig(q_lora_rank=64).is_mla is True

    def test_kv_lora_rank_set(self):
        assert KimiLinearConfig(kv_lora_rank=128).is_mla is True

    def test_qk_nope_head_dim_set(self):
        assert KimiLinearConfig(qk_nope_head_dim=64).is_mla is True

    def test_qk_rope_head_dim_set(self):
        assert KimiLinearConfig(qk_rope_head_dim=32).is_mla is True

    def test_v_head_dim_set(self):
        assert KimiLinearConfig(v_head_dim=128).is_mla is True

    def test_mla_use_nope_true(self):
        assert KimiLinearConfig(mla_use_nope=True).is_mla is True


class TestIsMoe:
    def test_num_experts_set(self):
        assert KimiLinearConfig(num_experts=8).is_moe is True

    def test_no_experts(self):
        assert KimiLinearConfig().is_moe is False


class TestAutoConfigRegistration:
    def test_auto_config_resolves_kimi_linear(self):
        # Importing model_config triggers AutoConfig.register as a side effect.
        from transformers import AutoConfig

        import sgl_jax.srt.configs.model_config  # noqa: F401

        cfg = AutoConfig.for_model(
            "kimi_linear",
            num_hidden_layers=2,
            linear_attn_config={"kda_layers": [2], "full_attn_layers": [1]},
        )
        assert isinstance(cfg, KimiLinearConfig)
        assert cfg.is_kda_layer(1) is True  # 1+1=2 ∈ {2}
        assert cfg.is_kda_layer(0) is False
