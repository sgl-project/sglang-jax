"""K3 must resolve to the K3 class, never to Kimi-Linear.

This guards a specific silent failure: K3's text_config declares
model_type "kimi_linear" / architectures ["KimiLinearForCausalLM"], so any routing that consults
the text config loads a model with NO AttnRes and NO SITU. It would load K3's weights, run, and
produce fluent output with two architectural components missing.
"""
import json, pathlib, pytest
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.models.kimi_k3 import KimiK3ForCausalLM, KimiK3ForConditionalGeneration

def _resolve_config() -> pathlib.Path:
    """The released config, from wherever the checkpoint is staged."""
    import os

    model_dir = os.environ.get("KIMI_K3_MODEL_DIR", "/dev/shm/k3_4l")
    for candidate in (os.path.join(model_dir, "config.json"), "/tmp/k3_config.json"):
        if pathlib.Path(candidate).exists():
            return pathlib.Path(candidate)
    return pathlib.Path("/nonexistent")


CFG = _resolve_config()


def test_top_level_arch_resolves_to_k3():
    archs = ModelRegistry.get_supported_archs()
    assert "KimiK3ForConditionalGeneration" in archs
    cls, _ = ModelRegistry.resolve_model_cls(["KimiK3ForConditionalGeneration"])
    assert cls is KimiK3ForConditionalGeneration
    assert issubclass(cls, KimiK3ForCausalLM)


@pytest.mark.skipif(not CFG.exists(), reason="K3 config not staged")
def test_released_config_top_level_arch_is_the_k3_one():
    c = json.loads(CFG.read_text())
    assert c["architectures"] == ["KimiK3ForConditionalGeneration"], c["architectures"]


@pytest.mark.skipif(not CFG.exists(), reason="K3 config not staged")
def test_text_config_would_misroute_if_used():
    """Documents the trap: the text config points at Kimi-Linear."""
    c = json.loads(CFG.read_text())
    t = c["text_config"]
    assert t["model_type"] == "kimi_linear"
    assert t["architectures"] == ["KimiLinearForCausalLM"]
    # ...and Kimi-Linear has neither of K3's additions, which is why routing on it is wrong
    assert "attn_res_block_size" in t and t["hidden_act"] == "situ"


def test_k3_class_is_not_kimi_linear():
    from sgl_jax.srt.models.kimi_linear import KimiLinearForCausalLM
    assert not issubclass(KimiK3ForConditionalGeneration, KimiLinearForCausalLM)
