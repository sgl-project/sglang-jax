from pathlib import Path

SOURCE = Path(__file__).parents[1] / "srt" / "models" / "bailing_moe.py"


def _source() -> str:
    return SOURCE.read_text()


def test_bailing_moe_fused_v2_folds_shared_expert_into_kernel():
    src = _source()

    assert "use_inkernel_se = self.moe_backend == MoEBackend.FUSED_V2" in src
    assert "self.use_inkernel_se = use_inkernel_se" in src
    assert "num_shared_experts=num_shared_experts if use_inkernel_se else 0" in src
    assert "moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size" in src
    assert "num_shared_experts=0" in src
    assert "num_shared_experts=num_shared_experts," not in src
    assert "if num_shared_experts > 0 and not use_inkernel_se:" in src


def test_bailing_moe_shared_weight_mapping_matches_fused_backend_semantics():
    src = _source()

    assert 'use_fused_shared = moe_backend == "fused_v2" and num_shared > 0' in src
    assert 'target_path = f"{target_prefix}.mlp.{target_name}"' in src
    assert 'target_base = f"{target_prefix}.shared_experts.{target_name}"' in src
    assert 'use_fused_shared = moe_backend == "fused"' not in src


def test_bailing_moe_fused_static_scale_mapping_supports_per_channel_quantization():
    src = _source()

    assert "is_per_channel = _wbs is None" in src
    assert "num_blocks = 1 if is_per_channel else in_dim // BLOCK_SIZE" in src
    assert "scale_repeat = None if is_per_channel else (1, num_blocks)" in src
