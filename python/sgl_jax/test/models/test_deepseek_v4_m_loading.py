"""M loading tests with explicit substitutes for the unmerged E/B/H boundaries."""

import json
import struct
import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors.numpy import save_file

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config
from sgl_jax.srt.model_loader.weights.source import LocalSource
from sgl_jax.srt.model_loader.weights.specs import WeightSpec
from sgl_jax.srt.models import deepseek_v4 as model_module
from sgl_jax.srt.models.deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4ForCausalLM,
    Disposition,
    classify_checkpoint,
    expected_trunk_keys,
)


@pytest.fixture
def mesh():
    return Mesh(
        np.asarray(jax.devices()).reshape(1, -1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )


@pytest.mark.parametrize("borrowed", [True, False])
@pytest.mark.parametrize("outcome", ["missing", "payload_error", "success", "bad_report"])
def test_checkpoint_source_lifetime_and_ownership(monkeypatch, mesh, borrowed, outcome):
    from sgl_jax.srt.model_loader.weights import source as weight_source

    config = DeepseekV4Config(
        num_hidden_layers=1, n_routed_experts=2, num_hash_layers=1, compress_ratios=[0]
    )
    required = expected_trunk_keys(config)
    facts = classify_checkpoint(config, required)
    m_keys = {key for key in required if facts[key].disposition is Disposition.M_OWNED}
    e_keys = required - m_keys
    inventory = {key: [{}] for key in required}
    if outcome == "missing":
        del inventory["embed.weight"]
    events = []

    class Source:
        metadata = inventory

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("close")

        def prefetch(self):
            events.append("prefetch")

    source = Source()

    def create_source(model_config, *, warmup=False):
        assert not borrowed, "M must reuse the injected source"
        assert warmup
        return source

    monkeypatch.setattr(weight_source, "LocalSource", create_source)
    # E owns numerical expert loading. Pin its current interface here so M
    # remains compatible without importing E's runtime integration branch.
    e_loader = ModuleType("sgl_jax.srt.layers.deepseek_v4_moe_loader")
    e_loader.STATIC_EXPERT_FORMAT = "sglang-jax-deepseek-v4-expert-fp8-per-channel-v1"
    e_loader.expected_moe_keys = lambda mlp: e_keys
    monkeypatch.setitem(sys.modules, e_loader.__name__, e_loader)

    class PayloadError(Exception):
        pass

    def load_regular(info, *, weight_source):
        assert weight_source is source
        assert info is inventory
        assert events == ["prefetch"]
        events.append("m_payload")
        if outcome == "payload_error":
            raise PayloadError
        return m_keys

    def load_experts(assigned, *, expert_format):
        assert set(assigned) == e_keys
        assert all(assigned[key] is inventory[key] for key in e_keys)
        assert expert_format is None
        events.append("e_payload")
        return SimpleNamespace(consumed_keys=set() if outcome == "bad_report" else e_keys)

    monkeypatch.setattr(
        model_module,
        "_rope_cache",
        lambda config, ratio: jnp.zeros((2, config.qk_rope_head_dim), jnp.float32),
    )
    backbone = SimpleNamespace(
        layers=[
            SimpleNamespace(
                mlp=SimpleNamespace(layer_id=0, load_owned_weights=load_experts),
                self_attn=SimpleNamespace(prepare_grouped_wo_a=lambda **kwargs: None),
            )
        ],
        **{
            name: nnx.Variable(None)
            for name in (
                "rope_plain",
                "rope_compressed",
                "rope_compressed_cos",
                "rope_compressed_sin",
            )
        },
    )
    model = SimpleNamespace(
        config=config, mesh=mesh, model=backbone, _load_regular_weights=load_regular
    )
    model_config = SimpleNamespace(
        model_path="/unused", _weight_source=source if borrowed else None
    )
    if outcome == "success":
        DeepseekV4ForCausalLM.load_weights(model, model_config)
    else:
        error = PayloadError if outcome == "payload_error" else ValueError
        match = {"missing": "missing", "bad_report": "consumed-key"}.get(outcome)
        with pytest.raises(error, match=match):
            DeepseekV4ForCausalLM.load_weights(model, model_config)
    expected = [] if outcome == "missing" else ["prefetch", "m_payload"]
    if outcome in ("success", "bad_report"):
        expected.append("e_payload")
    assert events == expected + ([] if borrowed else ["close"])


@pytest.mark.parametrize("abstract", [False, True])
@pytest.mark.parametrize("fused_wo_a", [False, True])
def test_dummy_post_load_preparation(monkeypatch, mesh, abstract, fused_wo_a):
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=128,
        num_hidden_layers=3,
        compress_ratios=[0, 4, 128],
        num_attention_heads=2,
        head_dim=128,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=2,
        index_n_heads=2,
        index_head_dim=128,
        max_position_embeddings=256,
    )
    monkeypatch.setenv("DSV4_FUSED_WO_A", str(int(fused_wo_a)))
    monkeypatch.setattr(model_module, "_HCA_FUSED_PROJ", True)
    monkeypatch.setitem(sys.modules, "sgl_jax.srt.layers.deepseek_v4_moe_loader", None)
    expert_mesh = Mesh(
        np.asarray(jax.devices()).reshape(-1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    ratios = []

    def rope_cache(config, ratio):
        # Test M's mesh boundary without requiring B's compressor import.
        active = jax.sharding.get_abstract_mesh() if abstract else jax.sharding.get_mesh()
        assert active == (mesh.abstract_mesh if abstract else mesh)
        ratios.append(ratio)
        rows = jnp.arange(config.max_position_embeddings, dtype=jnp.float32)
        return jnp.broadcast_to(rows[:, None], (rows.size, config.qk_rope_head_dim))

    monkeypatch.setattr(model_module, "_rope_cache", rope_cache)

    class Experts(nnx.Module):
        def __init__(self):
            self.moe_mesh = expert_mesh
            with jax.sharding.use_abstract_mesh(expert_mesh.abstract_mesh):
                self.weight = nnx.Param(
                    jnp.zeros(
                        (4, 4),
                        jnp.bfloat16,
                        out_sharding=NamedSharding(expert_mesh, P("data", None)),
                    )
                )

    class MoE(nnx.Module):
        def __init__(self):
            self.experts = Experts()
            self.is_hash_layer = False

    class Layer(nnx.Module):
        def __init__(self, layer_id):
            self.self_attn = DeepseekV4Attention(config, mesh, layer_id, jnp.bfloat16)
            self.mlp = MoE()

    class Backbone(nnx.Module):
        def __init__(self):
            self.layers = nnx.List([Layer(i) for i in range(3)])
            self.rope_plain = nnx.Variable(jnp.zeros((256, config.qk_rope_head_dim)))
            self.rope_compressed = nnx.Variable(jnp.zeros((256, config.qk_rope_head_dim)))
            self.rope_compressed_cos = nnx.Variable(jnp.zeros((256, 32)))
            self.rope_compressed_sin = nnx.Variable(jnp.zeros((256, 32)))

    class Model(nnx.Module):
        def __init__(self):
            self.config = config
            self.mesh = mesh
            self.model = Backbone()

        load_weights = DeepseekV4ForCausalLM.load_weights

    def initialize():
        model = Model()
        if abstract:
            model.load_weights(SimpleNamespace(_dummy_mode=True, _abstract_mode=True))
        return model

    context = jax.sharding.use_abstract_mesh(mesh.abstract_mesh) if abstract else jax.set_mesh(mesh)
    with context:
        model = nnx.eval_shape(initialize)
    if not abstract:
        ratios.clear()
        model.load_weights(SimpleNamespace(_dummy_mode=True))
    assert ratios[-2:] == [0, 4]
    tables = [
        model.model.rope_plain.value,
        model.model.rope_compressed.value,
        model.model.rope_compressed_cos.value,
        model.model.rope_compressed_sin.value,
    ]
    projections = []
    for layer in model.model.layers:
        attention = layer.self_attn
        projection = attention.wo_a_fused if fused_wo_a else attention.wo_a_grouped
        assert projection.value.shape == ((128, 256) if fused_wo_a else (2, 128, 128))
        projections.append(projection.value)
        expert = layer.mlp.experts.weight.value
        assert expert.sharding.spec == P("data", None)
        if abstract:
            assert isinstance(expert, jax.ShapeDtypeStruct)
        else:
            assert expert.sharding.mesh == expert_mesh
            expert.block_until_ready()
    fused = model.model.layers[2].self_attn.compressor.fused_proj.value
    assert fused.shape == (128, 256)
    assert fused.dtype == jnp.bfloat16
    for value in tables + projections + [fused]:
        if abstract:
            assert isinstance(value, jax.ShapeDtypeStruct)
        else:
            assert isinstance(value, jax.Array)
            value.block_until_ready()
            assert value.sharding.mesh == mesh
            assert np.isfinite(np.asarray(value)).all()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("scale_code", [128, 255])
def test_m_source_dtypes_and_e8m0_validation(tmp_path, monkeypatch, mesh, shared, scale_code):
    values = {
        "plain.weight": np.arange(12, dtype=np.float32).reshape(4, 3).astype(ml_dtypes.bfloat16),
        "quant.weight": np.ones((128, 256), ml_dtypes.float8_e4m3fn),
        "quant.scale": np.full((1, 2), scale_code, np.uint8).view(ml_dtypes.float8_e8m0fnu),
    }
    filename = tmp_path / "model.safetensors"
    save_file(values, str(filename))
    # The base branch's shared metadata scanner needs E's E8M0 extension.
    # Supply valid byte-range metadata here to isolate M's decoder contract.
    data = filename.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_size])
    info = {
        key: [
            {
                "file": str(filename),
                "shape": entry["shape"],
                "dtype": entry["dtype"],
                "byte_offset": 8 + header_size + entry["data_offsets"][0],
                "byte_size": entry["data_offsets"][1] - entry["data_offsets"][0],
            }
        ]
        for key, entry in header.items()
    }

    def param(shape, dtype):
        return nnx.Param(jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, P())))

    model = SimpleNamespace(
        config=SimpleNamespace(),
        mesh=mesh,
        plain=param((3, 4), jnp.bfloat16),
        quant=SimpleNamespace(
            weight_q=param((128, 256), jnp.float8_e4m3fn),
            weight_scale=param((2, 1, 128), jnp.float32),
        ),
    )
    mappings = {
        "plain.weight": WeightSpec("plain", transpose=True),
        "quant.weight": WeightSpec("quant.weight_q"),
        "quant.scale": WeightSpec("quant.weight_scale"),
    }
    monkeypatch.setattr(model_module, "build_weight_mappings", lambda *args: mappings)
    with LocalSource() as source:
        if scale_code == 255:
            with pytest.raises(ValueError, match="reserved E8M0"):
                DeepseekV4ForCausalLM._load_regular_weights(
                    model, info, weight_source=source if shared else None
                )
        else:
            consumed = DeepseekV4ForCausalLM._load_regular_weights(
                model, info, weight_source=source if shared else None
            )
            assert consumed == set(values)
            np.testing.assert_array_equal(np.asarray(model.plain.value), values["plain.weight"].T)
            np.testing.assert_array_equal(np.asarray(model.quant.weight_q.value, np.float32), 1)
            np.testing.assert_array_equal(np.asarray(model.quant.weight_scale.value), 2)
        assert bool(source.handles) == shared
