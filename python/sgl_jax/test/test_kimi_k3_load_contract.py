"""KimiK3ForCausalLM construction and its weight-loading contract with the shared loader.

The unit tests elsewhere build K3's pieces; this one builds the whole causal LM, so constructor
drift in shared layers (e.g. a changed ParallelLMHead signature) fails here rather than at the
first multi-host launch. It then checks the mapping against the model it will load into: every
WeightSpec must target a parameter that exists, and the MLA layers' absorbed-projection hook must
actually rewrite their kv_b_proj entries (the hook returns the mappings unchanged, silently, when
its target paths do not match).
"""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from flax import nnx

from sgl_jax.srt.model_loader.weights import WeightSpec
from sgl_jax.srt.models.kimi_k3 import KimiK3ForCausalLM
from sgl_jax.test.test_kimi_k3_model import _in_mesh, _tiny_cfg


def _targets(spec):
    return spec.target_path if isinstance(spec.target_path, list) else [spec.target_path]


MXFP4 = {"format": "mxfp4-pack-quantized"}


def _build(quantization_config=MXFP4, attn_res=2):
    cfg = _tiny_cfg(attn_res=attn_res)
    cfg.quantization_config = quantization_config
    with _in_mesh() as mesh:
        model = KimiK3ForCausalLM(cfg, mesh, dtype=jnp.float32)
        mappings = model._create_weight_mappings()
    return model, mappings


@pytest.fixture(autouse=True)
def _no_streaming_uri(monkeypatch):
    # KIMI_K3_WEIGHTS_URI implies an MXFP4 release; the tests pick the format explicitly.
    monkeypatch.delenv("KIMI_K3_WEIGHTS_URI", raising=False)
    monkeypatch.delenv("KIMI_K3_MOE_FP4", raising=False)


def _expert_readers(mappings):
    return [
        src
        for k, spec in mappings.items()
        for src in (k, *spec.sources)
        if ".block_sparse_moe.experts." in src
    ]


def _moe_layers(model):
    return [
        layer.block_sparse_moe
        for layer in model.model.layers
        if hasattr(layer, "routed_expert_down_proj")
    ]


def test_full_model_builds_and_every_mapping_targets_a_real_param():
    model, mappings = _build()
    assert mappings
    assert all(isinstance(spec, WeightSpec) for spec in mappings.values())
    params = {".".join(map(str, path)) for path, _ in nnx.to_flat_state(nnx.state(model))}
    missing = sorted(t for spec in mappings.values() for t in _targets(spec) if t not in params)
    assert not missing, f"{len(missing)} mapping targets are not model params: {missing[:8]}"


def test_mla_absorption_hook_rewrites_kv_b_proj():
    model, mappings = _build()
    mla = [
        (".".join(map(str, path)), module)
        for path, module in nnx.iter_graph(model)
        if module is not model and hasattr(module, "prepare_weight_loading")
    ]
    assert mla, "no module exposes prepare_weight_loading; MLA layers were not built"
    loader = SimpleNamespace(model_config=SimpleNamespace(quantization_config=None))
    absorbed = [prefix for prefix, module in mla if getattr(module, "use_absorbed", False)]
    for prefix, module in mla:
        mappings = module.prepare_weight_loading(loader, mappings, prefix)
    for prefix in absorbed:
        targets = {t for spec in mappings.values() for t in _targets(spec)}
        assert not any(t.startswith(prefix + ".kv_b_proj.") for t in targets), prefix
        assert prefix + ".w_uk" in targets and prefix + ".w_uv" in targets, prefix


def test_mxfp4_checkpoint_drops_the_bf16_expert_groups():
    """The released K3 experts are MXFP4 and loaded by _fixup_moe_mxfp4. A spec reading bf16
    experts.N.w{1,2,3}.weight makes the shared loader's planning step fail on every rank."""
    model, mappings = _build(quantization_config=MXFP4)
    assert not _expert_readers(mappings), _expert_readers(mappings)[:4]
    assert _moe_layers(model) and all(moe.fp4 for moe in _moe_layers(model))


def test_bf16_checkpoint_keeps_the_expert_groups():
    """A bf16 checkpoint through this class (e.g. Kimi-Linear) loads its experts only through
    the inherited groups, and must not get fp4 expert params."""
    model, mappings = _build(quantization_config=None)
    readers = _expert_readers(mappings)
    assert any(r.endswith(".experts.0.w1.weight") for r in readers)
    assert _moe_layers(model) and not any(moe.fp4 for moe in _moe_layers(model))


def test_streamed_expert_uri_implies_mxfp4(monkeypatch):
    monkeypatch.setenv("KIMI_K3_WEIGHTS_URI", "gs://bucket/release")
    _, mappings = _build(quantization_config=None)
    assert not _expert_readers(mappings)


def test_disabling_attn_res_removes_only_the_attn_res_mappings():
    """With AttnRes off, the full-rank KDA gate, MLA and LatentMoE mappings must be unchanged;
    exactly the AttnRes pairs go away, and nothing targets a param the model did not build."""
    _, with_res = _build(attn_res=2)
    model, without_res = _build(attn_res=None)
    removed = set(with_res) - set(without_res)
    assert removed and all(
        any(tag in k for tag in ("self_attention_res_", "mlp_res_", "output_attn_res_"))
        for k in removed
    ), sorted(removed)[:8]
    assert set(without_res) <= set(with_res)
    params = {".".join(map(str, path)) for path, _ in nnx.to_flat_state(nnx.state(model))}
    missing = sorted(t for spec in without_res.values() for t in _targets(spec) if t not in params)
    assert not missing, missing[:8]


@pytest.mark.parametrize("quantization_config", [MXFP4, None], ids=["mxfp4", "bf16"])
def test_every_checkpoint_name_carries_the_text_prefix(quantization_config):
    model, mappings = _build(quantization_config=quantization_config)
    prefix = model.TEXT_PREFIX
    unprefixed = [
        src
        for k, spec in mappings.items()
        for src in (k, *spec.sources)
        if not src.startswith(prefix)
    ]
    assert not unprefixed, unprefixed[:4]
