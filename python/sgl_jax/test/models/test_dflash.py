from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.mesh import AxisType
from jax.sharding import Mesh


def _mesh():
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devices, ("data", "tensor"), axis_types=(AxisType.Explicit, AxisType.Explicit))


def _tiny_config(**overrides):
    cfg = dict(
        architectures=["DFlashDraftModel"],
        model_type="qwen3",
        vocab_size=64,
        hidden_size=16,
        target_hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        rope_theta=1000000,
        rms_norm_eps=1e-6,
        attention_bias=False,
        block_size=4,
        target_layer_ids=[0, 1],
        dflash_config={"target_layer_ids": [0, 1], "mask_token_id": 63, "block_size": 4},
        tie_word_embeddings=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_dflash_architecture_is_registered():
    from sgl_jax.srt.models.registry import ModelRegistry

    model_cls, arch = ModelRegistry.resolve_model_cls(["DFlashDraftModel"])
    assert arch == "DFlashDraftModel"
    assert model_cls.__name__ == "DFlashDraftModel"


def test_dflash_weight_mapping_covers_tiny_config():
    from sgl_jax.srt.models.dflash import DFlashDraftModel

    cfg = _tiny_config(num_hidden_layers=2)
    model = SimpleNamespace(config=cfg)
    mappings = DFlashDraftModel._create_weight_mappings(model)

    expected = {
        "fc.weight",
        "hidden_norm.weight",
        "model.norm.weight",
    }
    for layer_idx in range(2):
        prefix = f"model.layers.{layer_idx}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )

    assert set(mappings) == expected
    assert mappings["fc.weight"].target_path == "fc.weight"
    assert mappings["fc.weight"].transpose is True


def test_project_target_hidden_shape():
    from sgl_jax.srt.models.dflash import DFlashDraftModel

    mesh = _mesh()
    cfg = _tiny_config()
    with jax.set_mesh(mesh):
        model = DFlashDraftModel(cfg, mesh=mesh, dtype=jnp.bfloat16)

    target_hidden = jnp.ones((3, 2 * cfg.hidden_size), dtype=jnp.bfloat16)
    projected = model.project_target_hidden(target_hidden)
    assert projected.shape == (3, cfg.hidden_size)
    assert projected.dtype == jnp.bfloat16


def test_dflash_materialize_kv_shapes():
    from sgl_jax.srt.models.dflash import DFlashDraftModel

    mesh = _mesh()
    cfg = _tiny_config()
    with jax.set_mesh(mesh):
        model = DFlashDraftModel(cfg, mesh=mesh, dtype=jnp.bfloat16)

    target_hidden = jnp.ones((3, 2 * cfg.hidden_size), dtype=jnp.bfloat16)
    positions = jnp.arange(3, dtype=jnp.int32)
    kv = model.materialize_kv(target_hidden, positions)

    assert len(kv) == cfg.num_hidden_layers
    for k, v in kv:
        assert k.shape == (3, cfg.num_key_value_heads, cfg.head_dim)
        assert v.shape == (3, cfg.num_key_value_heads, cfg.head_dim)


def test_qwen3_dflash_capture_sets_explicit_layers():
    from sgl_jax.srt.models.qwen3 import Qwen3ForCausalLM

    mesh = _mesh()
    cfg = _tiny_config(num_hidden_layers=6, target_layer_ids=[1, 3])
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(cfg, mesh=mesh, dtype=jnp.bfloat16)

    assert model.capture_aux_hidden_states is False
    model.set_dflash_layers_to_capture([1, 3])
    assert model.capture_aux_hidden_states is True
    assert model.model.layers_to_capture == [2, 4]
