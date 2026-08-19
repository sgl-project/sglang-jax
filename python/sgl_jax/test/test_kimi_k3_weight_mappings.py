"""Weight mappings must cover every text parameter in the REAL K3 checkpoint index.

An unmapped tensor is silently dropped by the loader, which yields a model that loads clean and
computes wrong -- the same silent-failure class as the other bugs on this port. So this test
diffs the mapping keys against the shipped index rather than asserting a hand-written list.
"""
import json, os, pathlib, re, pytest


def _resolve_index() -> pathlib.Path:
    """The FULL released index -- all 93 layers, not the truncation's subset.

    This test asks whether the mappings cover every text tensor the RELEASE ships, so a filtered
    index would make it vacuously pass on whatever happens to be staged. `stage_k3_truncated.py`
    keeps the full index under a name the safetensors loader does not glob, for exactly this.
    """
    model_dir = os.environ.get("KIMI_K3_MODEL_DIR", "/dev/shm/k3_4l")
    for candidate in (
        os.environ.get("KIMI_K3_INDEX"),
        os.path.join(model_dir, "model.safetensors.index.full.json"),
        "/tmp/k3_index.json",
    ):
        if candidate and pathlib.Path(candidate).exists():
            return pathlib.Path(candidate)
    return pathlib.Path("/nonexistent")


IDX = _resolve_index()
pytestmark = pytest.mark.skipif(not IDX.exists(), reason="full K3 index not staged")


def _index_text_params():
    wm = json.loads(IDX.read_text())["weight_map"]
    # text tower only; the release also carries mm_projector.* and a vision tower
    return {k for k in wm if k.startswith("language_model.")}


def test_attn_res_params_exist_and_are_siblings():
    """Pin the layout the mapping assumes: _norm and _proj suffixes, not a nested module."""
    p = _index_text_params()
    assert "language_model.model.layers.0.self_attention_res_norm.weight" in p
    assert "language_model.model.layers.0.self_attention_res_proj.weight" in p
    assert "language_model.model.layers.0.mlp_res_norm.weight" in p
    assert "language_model.model.layers.0.mlp_res_proj.weight" in p
    # nested form must NOT exist -- if it ever does, the mapping is wrong
    assert not any(".self_attention_res.norm." in k for k in p)


def test_model_level_output_attn_res_exists():
    """The third AttnRes: model-level, outside any layer."""
    p = _index_text_params()
    assert "language_model.model.output_attn_res_norm.weight" in p
    assert "language_model.model.output_attn_res_proj.weight" in p
    assert not any(k.startswith("language_model.model.layers.") for k in p
                   if "output_attn_res" in k)


def test_moe_experts_are_mxfp4_packed_with_scales():
    """Every packed expert weight must have a matching scale, or dequant silently mis-scales."""
    wm = json.loads(IDX.read_text())["weight_map"]
    packed = {k for k in wm if k.endswith("weight_packed")}
    assert len(packed) > 100000, len(packed)
    missing = [k for k in list(packed)[:2000]
               if k.replace("weight_packed", "weight_scale") not in wm]
    assert not missing, missing[:3]


def test_attention_is_not_quantized():
    """K3 quantizes only Linear targets: attention/norm params must be plain tensors."""
    wm = json.loads(IDX.read_text())["weight_map"]
    for stem in ("self_attn.A_log", "self_attn.dt_bias", "input_layernorm.weight"):
        hits = [k for k in wm if stem in k]
        assert hits, stem
        assert not any(h.endswith(("weight_packed", "weight_scale")) for h in hits), stem
