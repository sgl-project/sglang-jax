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
from flax import nnx

from sgl_jax.srt.model_loader.weights import WeightSpec
from sgl_jax.srt.models.kimi_k3 import KimiK3ForCausalLM
from sgl_jax.test.test_kimi_k3_model import _in_mesh, _tiny_cfg


def _targets(spec):
    return spec.target_path if isinstance(spec.target_path, list) else [spec.target_path]


def _build():
    with _in_mesh() as mesh:
        model = KimiK3ForCausalLM(_tiny_cfg(), mesh, dtype=jnp.float32)
        mappings = model._create_weight_mappings()
    return model, mappings


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
