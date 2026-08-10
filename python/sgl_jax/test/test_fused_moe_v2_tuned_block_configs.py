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


def test_ep32_interleave_policy_switches_after_32k_tokens():
    assert tuned_block_configs.should_interleave_fused_moe_v2_bt(
        num_tokens=32768,
        ep_size=32,
    )
    assert not tuned_block_configs.should_interleave_fused_moe_v2_bt(
        num_tokens=65536,
        ep_size=32,
    )
