"""GLM-5.3-Flash (glm5_next) config parsing without native transformers support."""

from transformers import AutoConfig

from sgl_jax.srt import hf_transformers_utils  # noqa: F401  (registers glm5_next)
from sgl_jax.srt.configs.glm5_next import Glm5NextHFConfig, get_glm5_next_config


def _text_config(num_layers=8):
    layer_types = [
        "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention" for i in range(num_layers)
    ]
    return {
        "model_type": "glm5_next_text",
        "num_hidden_layers": num_layers,
        "num_attention_heads": 64,
        "hidden_size": 4096,
        "layer_types": layer_types,
        "indexer_types": ["full"] * num_layers,
        "linear_attn_config": {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        "qk_rope_head_dim": 0,
        "index_kpool": 4,
        "index_kpool_compress": True,
        "hc_mult": 4,
    }


def test_glm5_next_model_type_is_registered():
    assert AutoConfig.for_model("glm5_next").model_type == "glm5_next"


def test_root_config_normalizes_text_config():
    hf = Glm5NextHFConfig(
        text_config=_text_config(),
        vision_config={"model_type": "glm5_next_vision"},
        architectures=["Glm5NextForConditionalGeneration"],
    )
    assert hf.text_config.num_attention_heads == 64

    cfg = get_glm5_next_config(hf)
    assert cfg.num_hidden_layers == 8
    assert cfg.linear_layer_ids == [0, 1, 2, 4, 5, 6]
    assert cfg.full_attention_layer_ids == [3, 7]
    assert cfg.qk_rope_head_dim == 0
    assert cfg.index_kpool == 4
    assert cfg.hc_mult == 4
    # The checkpoint ships no q_norm/k_norm tensors (as for GLM-5, #1729).
    assert cfg.use_qk_norm is False
