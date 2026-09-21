"""Run with JAX_NUM_CPU_DEVICES=8 for multi-device sharding coverage."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.lm_head_parallel import prepare_weight, weight_spec
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode


@pytest.fixture(params=[(1, 8), (4, 2), (8, 1)])
def mesh(request):
    if len(jax.devices()) < 8:
        pytest.skip("Requires 8 devices; set JAX_NUM_CPU_DEVICES=8")
    return Mesh(
        np.array(jax.devices()[:8]).reshape(request.param),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


class _HeadModel(nnx.Module):
    def __init__(self, mesh, vocab=32, dp_head=False):
        self.lm_head = ParallelLMHead(
            vocab, 12, param_dtype=jnp.float32, mesh=mesh, enable_dp_lm_head=dp_head
        )
        self.logits_processor = LogitsProcessor(vocab, mesh, enable_dp_lm_head=dp_head)


def _projection(mesh, dp_head, weight, soft_cap=None):
    vocab, hidden = weight.shape
    head = ParallelLMHead(
        vocab,
        hidden,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        mesh=mesh,
        enable_dp_lm_head=dp_head,
    )
    head.embedding.value = prepare_weight(
        jax.device_put(weight, NamedSharding(mesh, P())), mesh, dp_head
    )
    proc = LogitsProcessor(vocab, mesh, soft_cap=soft_cap, enable_dp_lm_head=dp_head)
    return head, proc


@pytest.mark.parametrize("dp_head", [False, True])
@pytest.mark.parametrize("vocab", [32, 30, 29])
def test_projection_and_dp_local_selection(mesh, dp_head, vocab):
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(16, 12)).astype(np.float32)
    weight = rng.normal(size=(vocab, 12)).astype(np.float32)
    h = jax.device_put(hidden, NamedSharding(mesh, P("data", None)))
    with jax.set_mesh(mesh):
        head, proc = _projection(mesh, dp_head, weight)
        w = head.embedding.value
        expected_logits = hidden @ weight.T
        out = jax.jit(lambda x, y: proc._get_logits(x, y))(h, head)
        np.testing.assert_allclose(out, expected_logits, rtol=2e-5, atol=2e-5)
        expected = P("data", "tensor") if vocab % mesh.shape["tensor"] == 0 else P("data", None)
        assert out.sharding.spec == expected
        assert w.sharding.spec == weight_spec(dp_head)
        parts = mesh.shape["tensor"] * (1 if dp_head else mesh.shape["data"])
        assert w.addressable_shards[0].data.shape == ((vocab + parts - 1) // parts, 12)
        # Choose one different local row per DP group, as prefill/logprob do.
        dp = mesh.shape["data"]
        idx_np = np.arange(dp, dtype=np.int32) % (16 // dp)
        idx = jax.device_put(idx_np, NamedSharding(mesh, P("data")))
        selected = jax.jit(proc._select_logits)(out, idx)
        rows = np.arange(dp) * (16 // dp) + idx_np
        np.testing.assert_allclose(selected, expected_logits[rows], rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(jnp.argmax(out, -1), np.argmax(expected_logits, -1))


@pytest.mark.parametrize("dp_head", [False, True])
def test_loading_and_shared_draft_head(mesh, dp_head):
    with jax.set_mesh(mesh):
        model, draft = _HeadModel(mesh, dp_head=dp_head), _HeadModel(mesh, dp_head=dp_head)
        mapping = model.lm_head.weight_mapping("lm_head.embedding")
        assert mapping.sharding == tuple(weight_spec(dp_head))
        original = np.asarray(model.lm_head.embedding.value)
        draft.lm_head.embedding.value = model.lm_head.embedding.value
        assert draft.logits_processor.enable_dp_lm_head == dp_head
        assert draft.lm_head.embedding.value is model.lm_head.embedding.value
        np.testing.assert_array_equal(model.lm_head.embedding.value, original)
        assert model.lm_head.embedding.value.sharding.spec == weight_spec(dp_head)


def test_tied_embedding_keeps_input_layout(mesh):
    class Model(nnx.Module):
        def __init__(self):
            self.embed = Embed(32, 12)
            self.lm_head = ParallelLMHead(32, 12).tie_weights(self.embed)
            self.logits_processor = LogitsProcessor(32, mesh)

    with jax.set_mesh(mesh):
        model = Model()
        original = model.embed.embedding.value
        assert model.lm_head.weight_mapping("lm_head.embedding").sharding == model.embed.kernel_axes
        assert model.embed.embedding.value is original
        assert model.lm_head.embedding is model.embed.embedding


@pytest.mark.parametrize("dp_head", [False, True])
@pytest.mark.parametrize("vocab", [32, 29])
@pytest.mark.parametrize("dummy", [False, True])
def test_weight_loader(mesh, dp_head, vocab, dummy, tmp_path):
    from types import SimpleNamespace

    from safetensors.numpy import save_file

    from sgl_jax.srt.utils.weight_utils import WeightLoader

    original = np.arange(vocab * 12, dtype=np.float32).reshape(vocab, 12) / 100
    save_file({"lm_head.weight": original}, tmp_path / "model.safetensors")
    config = SimpleNamespace(
        model_path=str(tmp_path),
        hf_config=SimpleNamespace(enable_dp_lm_head=dp_head),
        _dummy_mode=dummy,
    )
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: _HeadModel(mesh, vocab, dp_head))
        loader = WeightLoader(model, config, mesh, dtype=jnp.float32)
        loader.load_weights_from_safetensors(
            {"lm_head.weight": model.lm_head.weight_mapping("lm_head.embedding")}
        )
        # Divisible heads must already be sharded directly by the loader.
        if vocab == 32:
            assert model.lm_head.embedding.value.sharding.spec == weight_spec(dp_head)
        np.testing.assert_array_equal(
            np.asarray(model.lm_head.embedding.value)[:vocab],
            np.zeros_like(original) if dummy else original,
        )
        assert model.lm_head.embedding.value.sharding.spec == weight_spec(dp_head)


def test_cli_flag():
    import argparse

    from sgl_jax.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    assert not parser.parse_args(["--model-path", "/unused"]).enable_dp_lm_head
    assert parser.parse_args(["--model-path", "/unused", "--enable-dp-lm-head"]).enable_dp_lm_head


@pytest.mark.parametrize("dp_head", [False, True])
@pytest.mark.parametrize("mode", ["EXTEND", "DECODE", "TARGET_VERIFY", "DRAFT_EXTEND"])
def test_forward_modes(mesh, dp_head, mode):
    rng = np.random.default_rng(17)
    hidden = rng.normal(size=(16, 12)).astype(np.float32)
    weight = rng.normal(size=(32, 12)).astype(np.float32)
    dp = mesh.shape["data"]
    per_dp = 16 // dp
    with jax.set_mesh(mesh):
        head, proc = _projection(mesh, dp_head, weight, soft_cap=3.0)
        md = LogitsMetadata(
            forward_mode=ForwardMode[mode],
            capture_hidden_mode=CaptureHiddenMode.FULL,
            logits_indices=jax.device_put(
                np.full(dp, per_dp - 1, np.int32), NamedSharding(mesh, P("data"))
            ),
        )
        rows = np.arange(16)
        if mode == "EXTEND":
            rows = np.arange(dp) * per_dp + per_dp - 1
        elif mode == "DRAFT_EXTEND":
            md.extend_seq_lens = jax.device_put(
                np.full(dp, per_dp, np.int32), NamedSharding(mesh, P("data"))
            )
            md.accept_lens = jax.device_put(np.ones(dp, np.int32), NamedSharding(mesh, P("data")))
            rows = np.arange(dp) * per_dp
        result = jax.jit(lambda h, w, m: proc(h, w, m))(
            jax.device_put(hidden, NamedSharding(mesh, P("data", None))), head, md
        )
        expected = 3.0 * np.tanh((hidden[rows] @ weight.T) / 3.0)
        np.testing.assert_allclose(result.next_token_logits, expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(result.hidden_states, hidden)


@pytest.mark.parametrize("dp_head", [False, True])
@pytest.mark.parametrize("vocab", [32, 29])
@pytest.mark.parametrize("mode", ["TARGET_VERIFY", "DRAFT_EXTEND", "DECODE", "EXTEND"])
def test_greedy_projection_preserves_vocab_shards_and_dp_ids(mesh, dp_head, vocab, mode):
    from sgl_jax.srt.layers.lm_head_parallel import argmax_with_dp_sharding

    # All real logits are negative, so padded zero weights must never win.
    # Equal maxima straddle vocabulary shards, testing global first-index ties.
    hidden = np.ones((16, 4), np.float32)
    weights = np.full((vocab, 4), -2.0, np.float32)
    weights[1] = weights[vocab - 1] = -1.0
    dp = mesh.shape["data"]
    with jax.set_mesh(mesh):
        head, proc = _projection(mesh, dp_head, weights, soft_cap=3.0)
        md = LogitsMetadata(
            forward_mode=ForwardMode[mode],
            capture_hidden_mode=CaptureHiddenMode.NULL,
            preserve_vocab_sharding=True,
            logits_indices=jax.device_put(
                np.full(dp, 16 // dp - 1, np.int32), NamedSharding(mesh, P("data"))
            ),
        )

        @jax.jit
        def run(h, w, metadata):
            out = proc(h, w, metadata).next_token_logits
            return out, argmax_with_dp_sharding(out)

        logits, ids = run(jax.device_put(hidden, NamedSharding(mesh, P("data", None))), head, md)
        rows = 16 if mode in ("TARGET_VERIFY", "DECODE") else dp
        np.testing.assert_array_equal(ids, np.ones(rows, np.int32))
        assert ids.sharding.spec == P("data")
        if not dp_head and vocab == 32:
            assert logits.sharding.spec == P(None, ("data", "tensor"))
        else:
            expected = P("data", "tensor") if vocab % mesh.shape["tensor"] == 0 else P("data", None)
            assert logits.sharding.spec == expected
        expected_logits = 3.0 * np.tanh((hidden[:rows] @ weights.T) / 3.0)
        np.testing.assert_allclose(logits, expected_logits, rtol=2e-5, atol=2e-5)


def test_fused_greedy_consumers_keep_row_order_and_map_draft_vocab(mesh):
    from sgl_jax.srt.speculative.draft_extend_fused import (
        _eagle3_raw_and_mapped_token_from_logits,
        _topk1_index_from_logits,
    )

    logits_np = np.random.default_rng(42).normal(size=(16, 32)).astype(np.float32)
    expected = np.argmax(logits_np, axis=-1)
    mapping_np = np.arange(32, dtype=np.int32)[::-1].copy() * 7
    with jax.set_mesh(mesh):
        logits = jax.device_put(logits_np, NamedSharding(mesh, P(None, ("data", "tensor"))))
        mapping = jax.device_put(mapping_np, NamedSharding(mesh, P()))
        indices = jax.jit(_topk1_index_from_logits)(logits)
        raw, mapped = jax.jit(_eagle3_raw_and_mapped_token_from_logits)(logits, mapping)
        np.testing.assert_array_equal(indices[:, 0], expected)
        np.testing.assert_array_equal(raw, expected)
        np.testing.assert_array_equal(mapped, mapping_np[expected])
        assert raw.sharding.spec == P("data")
        assert mapped.sharding.spec == P("data")


@pytest.mark.parametrize("dummy", [False, True])
def test_multimodal_config_without_hf_config_loads_weights(tmp_path, dummy):
    from safetensors.numpy import save_file

    from sgl_jax.srt.configs.load_config import LoadConfig
    from sgl_jax.srt.model_loader.loader import JAXModelLoader
    from sgl_jax.srt.multimodal.configs.vaes.wan_vae_config import WanVAEConfig
    from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

    class Encoder(nnx.Module):
        def __init__(self, config, dtype, mesh):
            self.mesh = mesh
            self.weight = nnx.Param(jnp.zeros((4, 4), dtype=dtype))

        def load_weights(self, config):
            WeightLoader(self, config, self.mesh, dtype=config.dtype).load_weights_from_safetensors(
                {"weight": WeightMapping("weight", sharding=(None, None))}, dummy=dummy
            )

    original = np.arange(16, dtype=np.float32).reshape(4, 4)
    save_file({"weight": original}, tmp_path / "model.safetensors")
    config = WanVAEConfig(model_path=str(tmp_path), dtype=jnp.float32)
    assert not hasattr(config, "hf_config")
    mesh = Mesh(np.array(jax.devices()[:1]), ("encoder",))
    model = JAXModelLoader(LoadConfig(), mesh)._get_model(Encoder, config)
    np.testing.assert_array_equal(
        np.asarray(model.weight.value), np.zeros_like(original) if dummy else original
    )
    assert model.weight.value.sharding.spec == P(None, None)


@pytest.mark.parametrize("dp_head", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_direct_construction_and_load(mesh, dp_head, dummy, tied, tmp_path):
    """The model's own load_weights must suffice without an outer model loader."""
    from types import SimpleNamespace

    from safetensors.numpy import save_file
    from transformers import Qwen3Config

    from sgl_jax.srt.models.qwen3 import Qwen3ForCausalLM

    # Divisible by attention TP; deliberately needs global-TP padding at DP > 1.
    vocab = 5 * mesh.shape["tensor"]
    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=16,
        num_hidden_layers=0,
        num_attention_heads=8,
        num_key_value_heads=8,
        tie_word_embeddings=tied,
    )
    config.enable_dp_lm_head = dp_head
    embed = np.arange(vocab * 16, dtype=np.float32).reshape(vocab, 16) / 100
    weight = embed + 1
    checkpoint = {"model.embed_tokens.weight": embed, "model.norm.weight": np.ones(16, np.float32)}
    if not tied:
        checkpoint["lm_head.weight"] = weight
    save_file(checkpoint, tmp_path / "model.safetensors")
    load_config = SimpleNamespace(model_path=str(tmp_path), hf_config=config, _dummy_mode=dummy)
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, mesh, dtype=jnp.float32)
        assert model.logits_processor.enable_dp_lm_head == dp_head
        if not tied:
            assert model.lm_head.embedding.value.sharding.spec == weight_spec(dp_head)
        model.load_weights(load_config)
        head = model.model.embed_tokens if tied else model.lm_head
        expected_weight = embed if tied else weight
        if dummy:
            expected_weight = np.zeros_like(expected_weight)
        np.testing.assert_array_equal(np.asarray(head.embedding.value)[:vocab], expected_weight)
        expected_spec = P("tensor", None) if tied else weight_spec(dp_head)
        assert head.embedding.value.sharding.spec == expected_spec
        if not tied:
            np.testing.assert_array_equal(np.asarray(head.embedding.value)[vocab:], 0)
        h = jax.device_put(np.ones((16, 16), np.float32), NamedSharding(mesh, P("data", None)))
        result = jax.jit(lambda x, w: model.logits_processor._get_logits(x, w))(h, head)
        np.testing.assert_allclose(result, np.ones((16, 16)) @ expected_weight.T, rtol=2e-5)
        # Projection must not replace the tied input embedding with a padded head.
        assert model.model.embed_tokens.embedding.value.shape == (vocab, 16)
