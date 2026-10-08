from types import SimpleNamespace

import pytest

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config
from sgl_jax.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    Disposition,
    build_weight_mappings,
    classify_checkpoint,
    expected_trunk_keys,
)


def test_flash_inventory_partitions_exactly_between_m_and_e():
    config = DeepseekV4Config()
    required = expected_trunk_keys(config)
    facts = classify_checkpoint(config, required | {"mtp.2.dspark.weight"})
    m_keys = {key for key, fact in facts.items() if fact.disposition is Disposition.M_OWNED}
    e_keys = {key for key, fact in facts.items() if fact.disposition is Disposition.E_OWNED}

    assert m_keys == set(build_weight_mappings(config))
    assert not m_keys & e_keys
    assert m_keys | e_keys == required
    assert facts["mtp.2.dspark.weight"].disposition is Disposition.DROPPED
    assert "layers.0.ffn.gate.tid2eid" in e_keys
    assert "layers.3.ffn.gate.bias" in e_keys


@pytest.mark.parametrize(
    "key",
    [
        "layers.0.attn.compressor.wkv.weight",
        "layers.2.ffn.gate.bias",
        "layers.3.ffn.gate.tid2eid",
        "layers.3.attn.unknown.weight",
        "layers.43.attn_norm.weight",
    ],
)
def test_invalid_checkpoint_key_is_rejected(key):
    with pytest.raises(ValueError):
        classify_checkpoint(DeepseekV4Config(), [key])


@pytest.mark.parametrize("borrowed", [True, False])
@pytest.mark.parametrize("missing", [True, False])
def test_checkpoint_source_prefetch_and_lifetime(monkeypatch, borrowed, missing):
    # Import the loader before replacing its LocalSource dependency: its
    # restricted-source adapter subclasses the real implementation.
    from sgl_jax.srt.layers import deepseek_v4_moe_loader  # noqa: F401
    from sgl_jax.srt.model_loader.weights import source as weight_source

    config = DeepseekV4Config(
        num_hidden_layers=1, n_routed_experts=2, num_hash_layers=1, compress_ratios=[0]
    )
    inventory = {key: [{}] for key in expected_trunk_keys(config)}
    if missing:
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
        assert not borrowed, "V4 must reuse the framework's injected source"
        assert warmup, "standalone V4 loading must configure filesystem prefetch"
        return source

    monkeypatch.setattr(weight_source, "LocalSource", create_source)

    class ReachedPayloadLoading(Exception):
        pass

    def load_regular(info, *, weight_source):
        assert weight_source is source
        assert info is inventory
        assert events == ["prefetch"]
        events.append("payload")
        # Stop before numerical loading: this case verifies the real model
        # entry's source lifecycle and validation-before-I/O boundary.
        raise ReachedPayloadLoading

    model = SimpleNamespace(config=config, _load_regular_weights=load_regular)
    model_config = SimpleNamespace(
        model_path="/unused", _weight_source=source if borrowed else None
    )
    with pytest.raises(ValueError if missing else ReachedPayloadLoading):
        DeepseekV4ForCausalLM.load_weights(model, model_config)
    assert events == ([] if missing else ["prefetch", "payload"]) + ([] if borrowed else ["close"])


def test_post_load_tables_materialize_without_ambient_device_mesh(monkeypatch):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from jax.sharding import AxisType, Mesh

    from sgl_jax.srt.models import deepseek_v4 as model_module

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
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=1,
        moe_intermediate_size=128,
        index_n_heads=2,
        index_head_dim=128,
        max_position_embeddings=256,
    )
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    # Match JAXModelLoader: shape-only graph construction, then load with no
    # surrounding concrete device mesh. TPU cannot lower new RoPE iotas under
    # an AbstractMesh alone; CPU lowering can hide that mistake.
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: DeepseekV4ForCausalLM(config, mesh, jnp.bfloat16))
    assert isinstance(model.model.rope_plain.value, jax.ShapeDtypeStruct)
    compute_rope = model_module._rope_cache
    ratios = []

    def concrete_rope(config, ratio):
        assert jax.sharding.get_mesh() == mesh
        ratios.append(ratio)
        return compute_rope(config, ratio)

    monkeypatch.setattr(model_module, "_rope_cache", concrete_rope)
    model.load_weights(SimpleNamespace(_dummy_mode=True))
    assert ratios == [0, 4]
    for table in (
        model.model.rope_plain,
        model.model.rope_compressed,
        model.model.rope_compressed_cos,
        model.model.rope_compressed_sin,
    ):
        value = table.value
        assert isinstance(value, jax.Array)
        value.block_until_ready()
        assert np.isfinite(np.asarray(value)).all()
    for layer in model.model.layers:
        assert hasattr(layer.self_attn, "wo_a_grouped") or hasattr(layer.self_attn, "wo_a_fused")
    assert model.model.layers[2].self_attn.compressor.fused_proj.value.is_fully_addressable


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("scale_code", [128, 255])
def test_m_source_reuse_preserves_checkpoint_dtypes_and_scale_validation(
    tmp_path, monkeypatch, borrowed, scale_code
):
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from flax import nnx
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from safetensors.numpy import save_file

    from sgl_jax.srt.model_loader.weights.source import LocalSource
    from sgl_jax.srt.model_loader.weights.specs import WeightSpec
    from sgl_jax.srt.models import deepseek_v4 as model_module

    values = {
        "plain.weight": np.arange(12, dtype=np.float32).reshape(4, 3).astype(ml_dtypes.bfloat16),
        "quant.weight": np.ones((128, 256), ml_dtypes.float8_e4m3fn),
        "quant.scale": np.full((1, 2), scale_code, np.uint8).view(ml_dtypes.float8_e8m0fnu),
    }
    save_file(values, str(tmp_path / "model.safetensors"))
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )

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
    with LocalSource(SimpleNamespace(model_path=str(tmp_path))) as source:
        if scale_code == 255:
            with pytest.raises(ValueError, match="reserved E8M0"):
                DeepseekV4ForCausalLM._load_regular_weights(
                    model, source.metadata, weight_source=source if borrowed else None
                )
        else:
            consumed = DeepseekV4ForCausalLM._load_regular_weights(
                model, source.metadata, weight_source=source if borrowed else None
            )
            assert consumed == set(values)
            np.testing.assert_array_equal(np.asarray(model.plain.value), values["plain.weight"].T)
            np.testing.assert_array_equal(np.asarray(model.quant.weight_q.value, np.float32), 1)
            np.testing.assert_array_equal(np.asarray(model.quant.weight_scale.value), 2)
        assert bool(source.handles) == borrowed
