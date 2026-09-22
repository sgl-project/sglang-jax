"""Offline regressions for the Transformers v5 configuration/tokenizer boundary."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer
from transformers import AutoConfig

from sgl_jax.srt.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_tokenizer,
)
from sgl_jax.srt.layers.activation import ACT2FN


@pytest.mark.parametrize(
    "model_type",
    ["llama", "qwen2", "qwen3", "qwen2_moe", "qwen3_moe", "deepseek_v3", "glm4_moe", "glm_moe_dsa"],
)
def test_rope_config_roundtrip(model_type, tmp_path):
    config = AutoConfig.for_model(
        model_type,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 123456.0,
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
        },
        max_position_embeddings=16384,
    )
    config.save_pretrained(tmp_path)
    restored = get_config(str(tmp_path), trust_remote_code=False, local_files_only=True)
    assert restored.rope_parameters == config.rope_parameters
    assert get_context_length(restored) == 16384


def test_nested_rope_and_context():
    full = {"rope_type": "yarn", "rope_theta": 1000000, "factor": 4}
    config = SimpleNamespace(
        rope_parameters={
            "full_attention": full,
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
        },
        max_position_embeddings=4096,
        num_attention_heads=8,
    )
    assert get_context_length(SimpleNamespace(text_config=config)) == 16384


@pytest.mark.parametrize("name", ACT2FN)
def test_jax_activations_match_hf(name):
    import torch
    from transformers.activations import ACT2FN as HF_ACT2FN

    x = np.linspace(-5, 5, 101, dtype=np.float32)
    expected = HF_ACT2FN[name](torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(ACT2FN[name](jnp.asarray(x)), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "model_type,class_name",
    [
        ("llama", "LlamaDecoderLayer"),
        ("qwen2", "Qwen2DecoderLayer"),
        ("qwen3", "QWen3DecoderLayer"),
        ("qwen2_moe", "Qwen2MoeDecoderLayer"),
        ("qwen3_moe", "QWen3MoeDecoderLayer"),
    ],
)
def test_decoder_uses_checkpoint_rope(model_type, class_name):
    import importlib

    import jax
    from flax import nnx

    from sgl_jax.srt.layers.embeddings import YarnRotaryEmbedding
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    config = AutoConfig.for_model(
        model_type,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        max_position_embeddings=128,
        ep_size=1,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 123456.0,
            "factor": 4.0,
            "original_max_position_embeddings": 32,
        },
    )
    mesh = create_device_mesh(
        ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
    )
    cls = getattr(importlib.import_module(f"sgl_jax.srt.models.{model_type}"), class_name)
    with jax.set_mesh(mesh):
        layer = nnx.eval_shape(lambda: cls(config, mesh, dtype=jnp.float32))
    rope = layer.self_attn.rotary_emb
    assert isinstance(rope, YarnRotaryEmbedding)
    assert rope.base == 123456.0
    assert rope.scaling_factor == 4.0


@pytest.mark.parametrize("tied", [True, False])
def test_qwen25_vl_uses_root_embedding_tie(tied):
    import jax
    from flax import nnx

    from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    config = AutoConfig.for_model("qwen2_5_vl", tie_word_embeddings=tied)
    mesh = create_device_mesh(
        ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
    )
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: Qwen2_5_VLForConditionalGeneration(config, mesh=mesh))
    assert hasattr(model, "lm_head") is not tied
    assert ("lm_head.weight" in model._language_weight_mappings()) is not tied


@pytest.mark.parametrize(
    "field", ["pad_token_id", "bos_token_id", "eos_token_id", "tie_word_embeddings"]
)
def test_composite_config_preserves_shared_fields(field):
    from sgl_jax.srt.hf_transformers_utils import get_hf_text_config

    text = SimpleNamespace(num_attention_heads=8)
    parent = SimpleNamespace(text_config=text, **{field: 7})
    assert get_hf_text_config(parent) is text
    assert getattr(text, field) == 7
    setattr(text, field, None)
    get_hf_text_config(parent)
    assert getattr(text, field) is None  # An explicit value must not be overwritten.
    delattr(parent, field)
    get_hf_text_config(parent)
    assert getattr(parent, field) is None


@pytest.mark.parametrize("tokenizer_class", ["LlamaTokenizerFast", "Qwen2TokenizerFast"])
def test_checkpoint_bytelevel_tokenizer(tokenizer_class, tmp_path):
    import json

    from tokenizers import decoders, models, pre_tokenizers, processors, trainers

    raw = Tokenizer(models.BPE(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    texts = ["hello world", "hello 世界", " hello!"]
    raw.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<s>", "</s>"],
        ),
    )
    raw.post_processor = processors.ByteLevel(trim_offsets=False)
    raw.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": tokenizer_class,
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "add_bos_token": True,
            }
        )
    )
    tokenizer = get_tokenizer(str(tmp_path), local_files_only=True)
    prefix = [raw.token_to_id("<s>")] if tokenizer_class == "LlamaTokenizerFast" else []
    for text in texts:
        ids = raw.encode(text).ids
        assert tokenizer.encode(text, add_special_tokens=False) == ids
        assert tokenizer.encode(text) == prefix + ids
        assert tokenizer.decode(ids) == text
    without_bos = get_tokenizer(str(tmp_path), local_files_only=True, add_bos_token=False)
    assert without_bos.encode(texts[0]) == raw.encode(texts[0]).ids
