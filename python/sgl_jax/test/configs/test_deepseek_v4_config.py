"""M1.1a: DeepSeek-V4 config and the layer-classification contract.

These tests are the guard on the one thing three separate subsystems share --
the mapping from ``compress_ratios`` to a per-layer attention type. The model
(M1), the KV/state pools (C1.1) and the attention metadata (M2.1) all consume
``classify_layers``; if it drifts they drift silently.
"""

import json
import pathlib

import pytest

from sgl_jax.srt.configs.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4LayerType,
    classify_layers,
    hash_moe_layer_flags,
    layer_type_for_compress_ratio,
    mhc_param_shapes,
    mix_hc_width,
    trunk_compress_ratios,
)

# The shipped deepseek-ai/DeepSeek-V4-Flash-0731 config, pinned at the revision
# M0.1 fixed. Written out longhand here (rather than reusing the module's own
# construction) so a wrong edit to the module cannot make this test agree with it.
FLASH_0731_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"

FLASH_0731_RATIOS = [
    0, 0,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4,
    0, 0, 0,
]  # fmt: skip


def test_shipped_ratios_are_longer_than_the_trunk():
    """The whole point of ``trunk_compress_ratios``.

    46 entries, 43 layers. A consumer that iterates the raw list gets three
    phantom SWA-only layers.
    """
    cfg = DeepseekV4Config()
    assert len(cfg.compress_ratios) == 46
    assert cfg.num_hidden_layers == 43
    assert cfg.compress_ratios == FLASH_0731_RATIOS
    assert len(trunk_compress_ratios(cfg)) == 43
    assert trunk_compress_ratios(cfg) == tuple(FLASH_0731_RATIOS[:43])


def test_classification_of_the_shipped_config():
    cfg = DeepseekV4Config()
    types = classify_layers(cfg)
    assert len(types) == 43
    # Layers 0 and 1 carry no compressed history.
    assert types[0] is DeepseekV4LayerType.SWA_ONLY
    assert types[1] is DeepseekV4LayerType.SWA_ONLY
    # Then 4/128 alternate, starting at 4 on the first even layer.
    assert types[2] is DeepseekV4LayerType.C4A
    assert types[3] is DeepseekV4LayerType.C128A
    # The last trunk layer is C4A, not SWA-only -- that only looks like
    # SWA-only if you read past index 42 into the tail.
    assert types[42] is DeepseekV4LayerType.C4A
    counts = {t: types.count(t) for t in DeepseekV4LayerType}
    assert counts == {
        DeepseekV4LayerType.SWA_ONLY: 2,
        DeepseekV4LayerType.C4A: 21,
        DeepseekV4LayerType.C128A: 20,
    }


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (0, DeepseekV4LayerType.SWA_ONLY),
        (1, DeepseekV4LayerType.SWA_ONLY),
        (4, DeepseekV4LayerType.C4A),
        (128, DeepseekV4LayerType.C128A),
    ],
)
def test_layer_type_for_supported_ratios(ratio, expected):
    assert layer_type_for_compress_ratio(ratio) is expected


@pytest.mark.parametrize("ratio", [2, 3, 8, 64, 127, 129, 256])
def test_layer_type_rejects_unknown_ratio(ratio):
    """A silent fallback would make a layer attend over the wrong cache."""
    with pytest.raises(ValueError, match="Unsupported DeepSeek-V4 compress_ratio"):
        layer_type_for_compress_ratio(ratio)


def test_empty_ratio_list_is_an_error_not_absence():
    cfg = DeepseekV4Config(num_hidden_layers=5, compress_ratios=[])
    # An explicit empty list is a short list, not "absent".
    with pytest.raises(ValueError, match="cannot classify every trunk layer"):
        classify_layers(cfg)


def test_absent_compress_ratios_attribute_defaults_to_swa_only():
    class Bare:
        num_hidden_layers = 5

    assert classify_layers(Bare()) == (DeepseekV4LayerType.SWA_ONLY,) * 5


def test_short_ratio_list_is_an_error_not_a_pad():
    cfg = DeepseekV4Config(num_hidden_layers=10, compress_ratios=[0, 0, 4, 128])
    with pytest.raises(ValueError, match="4 entries but num_hidden_layers=10"):
        trunk_compress_ratios(cfg)


def test_hash_moe_flags():
    cfg = DeepseekV4Config()
    flags = hash_moe_layer_flags(cfg)
    assert len(flags) == 43
    # Flash 0731 ships num_hash_layers=3: the first three layers route by
    # token id, everything after routes on gate logits.
    assert flags[:4] == (True, True, True, False)
    assert sum(flags) == 3


def test_hash_moe_flags_bounds():
    with pytest.raises(ValueError, match="exceeds num_hidden_layers"):
        hash_moe_layer_flags(DeepseekV4Config(num_hidden_layers=2, num_hash_layers=3))
    assert sum(hash_moe_layer_flags(DeepseekV4Config(num_hash_layers=0))) == 0


def test_mhc_param_shapes_match_the_reference_layout():
    cfg = DeepseekV4Config()
    shapes = mhc_param_shapes(cfg)
    # hc_mult=4 -> mix_hc = (2+4)*4 = 24; hc_dim = 4*4096 = 16384.
    assert shapes["fn"] == (24, 16384)
    assert shapes["base"] == (24,)
    assert shapes["scale"] == (3,)
    assert shapes["head_fn"] == (4, 16384)
    assert shapes["head_base"] == (4,)
    assert shapes["head_scale"] == (1,)


def test_mix_hc_width_agrees_with_the_kernel():
    """The config module duplicates this to stay Pallas-free; pin them together."""
    from sgl_jax.srt.kernels.mhc import mhc as mhc_kernels

    for hc_mult in (1, 2, 4, 8):
        assert mix_hc_width(hc_mult) == mhc_kernels.mix_hc_width(hc_mult)


def test_rope_scaling_default_is_not_flattened_away():
    """transformers rewrites a ``None`` rope_scaling into ``rope_type: default``,
    which would silently drop the 16x YaRN context extension."""
    cfg = DeepseekV4Config()
    assert cfg.rope_scaling["type"] == "yarn"
    assert cfg.rope_scaling["factor"] == 16
    assert cfg.rope_scaling["original_max_position_embeddings"] == 65536
    assert cfg.max_position_embeddings == 1048576


def test_our_classification_does_not_shadow_transformers_layer_types():
    """``layer_types`` belongs to transformers' PretrainedConfig, which rewrites
    it during validation, and sgl-jax's ``ModelConfig.get_hybrid_layer_counts``
    reads it with HF's "sliding_attention" string semantics. Ours must live
    under a different name -- constructing the config at all fails otherwise."""
    cfg = DeepseekV4Config()
    assert cfg.attention_layer_types == classify_layers(cfg)
    # Whatever transformers puts on ``layer_types``, it is not our enum tuple.
    assert not any(
        isinstance(x, DeepseekV4LayerType) for x in (getattr(cfg, "layer_types", None) or ())
    )


def test_registered_with_autoconfig():
    from transformers import AutoConfig

    import sgl_jax.srt.hf_transformers_utils  # noqa: F401  (performs the registration)

    cfg = AutoConfig.for_model("deepseek_v4")
    assert isinstance(cfg, DeepseekV4Config)


def test_round_trips_through_json_without_losing_unconsumed_fields():
    """dspark_* / num_nextn_predict_layers are not consumed by the first version
    but must survive a save/load, or a converted checkpoint loses them."""
    cfg = DeepseekV4Config()
    reloaded = DeepseekV4Config(**json.loads(cfg.to_json_string()))
    assert reloaded.dspark_target_layer_ids == [40, 41, 42]
    assert reloaded.num_nextn_predict_layers == 1
    assert reloaded.compress_ratios == cfg.compress_ratios
    assert classify_layers(reloaded) == classify_layers(cfg)


def test_defaults_match_the_pinned_upstream_config_if_present():
    """If a copy of the real config.json is available, no default may disagree.

    Skipped when the file is absent so the suite stays offline-clean.
    """
    path = pathlib.Path(__file__).with_name("deepseek_v4_flash_0731_config.json")
    if not path.exists():
        pytest.skip(f"no pinned config copy at {path}")
    real = json.loads(path.read_text())
    cfg = DeepseekV4Config()
    ignored = {
        "architectures",
        "model_type",
        "quantization_config",
        "torch_dtype",
        "transformers_version",
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "eos_token_id",
        "initializer_range",
        "use_cache",
    }
    disagreements = {}
    for key, want in real.items():
        if key in ignored:
            continue
        got = getattr(cfg, key, "<<missing>>")
        if isinstance(got, tuple):
            got = list(got)
        if isinstance(want, dict) and isinstance(got, dict):
            # transformers' rope validation *adds* normalised keys (rope_type,
            # rope_theta). Extra keys are fine; disagreeing values are not.
            got = {k: v for k, v in got.items() if k in want}
        if got != want:
            disagreements[key] = (want, got)
    assert not disagreements, disagreements
