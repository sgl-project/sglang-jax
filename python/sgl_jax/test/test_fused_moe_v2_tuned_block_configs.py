import jax.numpy as jnp

from sgl_jax.srt.kernels.fused_moe.v2 import tuned_block_configs
from sgl_jax.srt.kernels.fused_moe.v2.kernel import FusedMoEBlockConfig


def test_glm52_ep16_16_tokens_uses_tuned_v7_config(monkeypatch):
    monkeypatch.setattr(tuned_block_configs, "get_device_name", lambda: "TPU v7")

    config = tuned_block_configs.get_tuned_fused_moe_v2_block_config(
        num_tokens=16,
        num_experts=256,
        top_k=8,
        hidden_size=6144,
        intermediate_size=2048,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float8_e4m3fn,
        ep_size=16,
        use_shared_expert=True,
        use_grouped_topk=False,
        enable_act_quant=True,
    )

    assert config == FusedMoEBlockConfig(
        bt=8,
        bf=512,
        btc=8,
        bse=128,
        bts=8,
    )


def test_glm52_ep16_per_channel_uses_distinct_tuned_v7_config(monkeypatch):
    monkeypatch.setattr(tuned_block_configs, "get_device_name", lambda: "TPU v7")

    per_channel = tuned_block_configs.get_tuned_fused_moe_v2_block_config(
        num_tokens=128,
        num_experts=256,
        top_k=8,
        hidden_size=6144,
        intermediate_size=2048,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float8_e4m3fn,
        ep_size=16,
        use_shared_expert=True,
        use_grouped_topk=False,
        enable_act_quant=True,
        quant_mode="per_channel",
    )
    blockwise = tuned_block_configs.get_tuned_fused_moe_v2_block_config(
        num_tokens=128,
        num_experts=256,
        top_k=8,
        hidden_size=6144,
        intermediate_size=2048,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float8_e4m3fn,
        ep_size=16,
        use_shared_expert=True,
        use_grouped_topk=False,
        enable_act_quant=True,
        quant_mode="blockwise",
    )

    assert per_channel == FusedMoEBlockConfig(bt=8, bf=512, btc=8, bse=512, bts=8)
    assert blockwise == FusedMoEBlockConfig(bt=8, bf=1024, btc=8, bse=128, bts=8)


def test_glm52_ep16_w8a16_per_channel_uses_hot_bucket_configs(monkeypatch):
    monkeypatch.setattr(tuned_block_configs, "get_device_name", lambda: "TPU v7")

    def lookup(num_tokens):
        return tuned_block_configs.get_tuned_fused_moe_v2_block_config(
            num_tokens=num_tokens,
            num_experts=256,
            top_k=8,
            hidden_size=6144,
            intermediate_size=2048,
            dtype=jnp.bfloat16,
            weight_dtype=jnp.float8_e4m3fn,
            ep_size=16,
            use_shared_expert=True,
            use_grouped_topk=False,
            enable_act_quant=False,
            quant_mode="per_channel",
        )

    assert lookup(32) == FusedMoEBlockConfig(bt=8, bf=1024, btc=8, bse=512, bts=8)
    assert lookup(32768) == FusedMoEBlockConfig(
        bt=128,
        bf=1024,
        btc=128,
        bse=1024,
        bts=128,
    )


def test_glm52_ep32_w8a16_per_channel_uses_hot_bucket_configs(monkeypatch):
    monkeypatch.setattr(tuned_block_configs, "get_device_name", lambda: "TPU v7")

    def lookup(num_tokens):
        return tuned_block_configs.get_tuned_fused_moe_v2_block_config(
            num_tokens=num_tokens,
            num_experts=256,
            top_k=8,
            hidden_size=6144,
            intermediate_size=2048,
            dtype=jnp.bfloat16,
            weight_dtype=jnp.float8_e4m3fn,
            ep_size=32,
            use_shared_expert=True,
            use_grouped_topk=False,
            enable_act_quant=False,
            quant_mode="per_channel",
        )

    assert lookup(64) == FusedMoEBlockConfig(
        bt=8,
        bf=512,
        btc=16,
        bse=128,
        bts=16,
    )
    assert lookup(65536) == FusedMoEBlockConfig(
        bt=128,
        bf=1024,
        btc=32,
        bse=1024,
        bts=160,
    )


def test_glm52_ep32_64k_uses_confirmed_v7_config(monkeypatch):
    monkeypatch.setattr(tuned_block_configs, "get_device_name", lambda: "TPU v7")

    config = tuned_block_configs.get_tuned_fused_moe_v2_block_config(
        num_tokens=65536,
        num_experts=256,
        top_k=8,
        hidden_size=6144,
        intermediate_size=2048,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float8_e4m3fn,
        ep_size=32,
        use_shared_expert=True,
        use_grouped_topk=False,
        enable_act_quant=True,
    )

    assert config == FusedMoEBlockConfig(
        bt=128,
        bf=1024,
        btc=32,
        bse=1024,
        bts=160,
    )


def test_glm52_ep32_per_channel_w8a8_uses_tuned_configs(monkeypatch):
    monkeypatch.setattr(tuned_block_configs, "get_device_name", lambda: "TPU v7")

    def lookup(num_tokens):
        return tuned_block_configs.get_tuned_fused_moe_v2_block_config(
            num_tokens=num_tokens,
            num_experts=256,
            top_k=8,
            hidden_size=6144,
            intermediate_size=2048,
            dtype=jnp.bfloat16,
            weight_dtype=jnp.float8_e4m3fn,
            ep_size=32,
            use_shared_expert=True,
            use_grouped_topk=False,
            enable_act_quant=True,
            quant_mode="per_channel",
        )

    # ep32 W8A8 per-channel now has its own key (previously fell back to the
    # 6-tuple blockwise table). Decode self-selects bse=512; the 65536 prefill
    # (C64 = 64 concurrency x 1K extend) tuned to btc=128.
    assert lookup(32) == FusedMoEBlockConfig(bt=8, bf=512, btc=8, bse=512, bts=8)
    assert lookup(512) == FusedMoEBlockConfig(bt=16, bf=512, btc=32, bse=512, bts=32)
    assert lookup(65536) == FusedMoEBlockConfig(
        bt=64, bf=1024, btc=128, bse=1024, bts=128
    )

