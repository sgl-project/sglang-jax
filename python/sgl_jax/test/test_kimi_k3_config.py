"""KimiK3Config tests, driven by the REAL config.json from the released checkpoint.

The reference json is vendored next to this test so the expectations are grounded in the actual
model rather than in a guess about K3's layer pattern.
"""
import json, pathlib, pytest
from sgl_jax.srt.configs.kimi_k3 import KimiK3Config

REF = json.loads((pathlib.Path(__file__).parent / "kimi_k3_config_reference.json").read_text())
T = REF.get("text_config", REF)


def _cfg():
    return KimiK3Config(
        num_hidden_layers=T["num_hidden_layers"],
        hidden_size=T["hidden_size"],
        intermediate_size=T["intermediate_size"],
        moe_intermediate_size=T["moe_intermediate_size"],
        num_experts=T["num_experts"],
        num_experts_per_token=T["num_experts_per_token"],
        num_shared_experts=T["num_shared_experts"],
        first_k_dense_replace=T["first_k_dense_replace"],
        kv_lora_rank=T["kv_lora_rank"],
        qk_nope_head_dim=T["qk_nope_head_dim"],
        qk_rope_head_dim=T["qk_rope_head_dim"],
        mla_use_nope=T["mla_use_nope"],
        linear_attn_config=T["linear_attn_config"],
        hidden_act=T["hidden_act"],
        activation_situ_beta=T["activation_situ_beta"],
        activation_situ_linear_beta=T["activation_situ_linear_beta"],
        attn_res_block_size=T["attn_res_block_size"],
        mla_use_output_gate=T["mla_use_output_gate"],
        latent_moe_use_norm=T["latent_moe_use_norm"],
    )


def test_layer_split_is_69_kda_24_mla():
    """K3's documented architecture: 93 layers = 69 KDA + 24 gated MLA (3:1)."""
    c = _cfg()
    kda = [i for i in range(c.num_hidden_layers) if c.is_kda_layer(i)]
    mla = [i for i in range(c.num_hidden_layers) if not c.is_kda_layer(i)]
    assert (len(kda), len(mla)) == (69, 24), (len(kda), len(mla))
    assert len(kda) + len(mla) == 93


def test_kda_mla_pattern_is_three_to_one():
    """Every 4th layer (1-based) is MLA -- the 3:1 interleave."""
    c = _cfg()
    for i in range(0, 88):
        assert c.is_kda_layer(i) == ((i + 1) % 4 != 0), i


def test_situ_is_the_live_activation_path():
    c = _cfg()
    assert c.uses_situ, "K3 ships hidden_act='situ'; SituAndMul must be on the live path"
    assert c.activation_situ_beta == 4.0
    assert c.activation_situ_linear_beta == 25.0


def test_attn_res_enabled_with_block_size_12():
    c = _cfg()
    assert c.uses_attn_res and c.attn_res_block_size == 12


def test_attn_res_candidate_growth_over_full_depth():
    """block=12 over 93 layers => 8 checkpoints, so the softmax spans at most 8 candidates."""
    c = _cfg()
    counts = [c.n_attn_res_candidates(i) for i in range(c.num_hidden_layers)]
    assert counts[0] == 1 and counts[11] == 1 and counts[12] == 2
    assert max(counts) == 8, max(counts)
    assert sum(1 for i in range(c.num_hidden_layers) if i % c.attn_res_block_size == 0) == 8


def test_attn_res_disabled_when_block_size_absent():
    """The whole two-AttnRes path keys off this single switch."""
    c = KimiK3Config(num_hidden_layers=4, hidden_size=64, attn_res_block_size=None)
    assert not c.uses_attn_res and c.n_attn_res_candidates(3) == 0
