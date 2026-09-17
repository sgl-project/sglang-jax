"""Run with JAX_NUM_CPU_DEVICES=8 for multi-device sharding coverage."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.lm_head_parallel import (
    configure_lm_heads,
    lm_head_load_shardings,
    prepare_weight,
    weight_spec,
)
from sgl_jax.srt.layers.logits_processor import LogitsProcessor


@pytest.fixture(params=[(1, 8), (4, 2), (8, 1)])
def mesh(request):
    if len(jax.devices()) < 8:
        pytest.skip("Requires 8 devices; set JAX_NUM_CPU_DEVICES=8")
    return Mesh(
        np.array(jax.devices()[:8]).reshape(request.param),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


@pytest.mark.parametrize("dp_head", [False, True])
@pytest.mark.parametrize("vocab", [32, 30, 29])
def test_projection_and_dp_local_selection(mesh, dp_head, vocab):
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(16, 12)).astype(np.float32)
    weight = rng.normal(size=(vocab, 12)).astype(np.float32)
    h = jax.device_put(hidden, NamedSharding(mesh, P("data", None)))
    w = prepare_weight(jax.device_put(weight, NamedSharding(mesh, P())), mesh, dp_head)
    with jax.set_mesh(mesh):
        head = ParallelLMHead(vocab, 12, dtype=jnp.float32, param_dtype=jnp.float32)
        head.embedding.value = w
        proc = LogitsProcessor(vocab, mesh)
        proc.enable_dp_lm_head = dp_head
        out = jax.jit(lambda x, y: proc._get_logits(x, y))(h, head)
        np.testing.assert_allclose(out, hidden @ weight.T, rtol=2e-5, atol=2e-5)
        expected = (
            P("data", "tensor")
            if vocab % mesh.shape["tensor"] == 0
            else P("data", None)
        )
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
        np.testing.assert_allclose(
            selected, (hidden @ weight.T)[rows], rtol=2e-5, atol=2e-5
        )
        np.testing.assert_array_equal(
            jnp.argmax(out, -1), np.argmax(hidden @ weight.T, -1)
        )


@pytest.mark.parametrize("dp_head", [False, True])
def test_loading_and_shared_draft_head(mesh, dp_head):
    class Model(nnx.Module):
        def __init__(self):
            self.lm_head = ParallelLMHead(32, 12, param_dtype=jnp.float32)
            self.logits_processor = LogitsProcessor(32, mesh)

    with jax.set_mesh(mesh):
        model, draft = Model(), Model()
        mappings = lm_head_load_shardings(model, mesh, dp_head)
        assert mappings == {"lm_head.embedding": tuple(weight_spec(dp_head))}
        original = np.asarray(model.lm_head.embedding.value)
        configure_lm_heads(model, mesh, dp_head)
        configure_lm_heads(draft, mesh, dp_head)
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
        assert lm_head_load_shardings(model, mesh, False) == {}
        configure_lm_heads(model, mesh, False)
        assert model.embed.embedding.value is original
        assert model.lm_head.embedding is model.embed.embedding


@pytest.mark.parametrize("dp_head", [False, True])
@pytest.mark.parametrize("vocab", [32, 29])
@pytest.mark.parametrize("dummy", [False, True])
def test_weight_loader(mesh, dp_head, vocab, dummy, tmp_path):
    from types import SimpleNamespace

    from safetensors.numpy import save_file
    from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

    class Model(nnx.Module):
        def __init__(self):
            self.lm_head = ParallelLMHead(vocab, 12, param_dtype=jnp.float32)
            self.logits_processor = LogitsProcessor(vocab, mesh)

    original = np.arange(vocab * 12, dtype=np.float32).reshape(vocab, 12) / 100
    save_file({"lm_head.weight": original}, tmp_path / "model.safetensors")
    config = SimpleNamespace(
        model_path=str(tmp_path),
        hf_config=SimpleNamespace(enable_dp_lm_head=dp_head),
        _dummy_mode=dummy,
    )
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(Model)
        loader = WeightLoader(model, config, mesh, dtype=jnp.float32)
        loader.load_weights_from_safetensors(
            {
                "lm_head.weight": WeightMapping(
                    "lm_head.embedding", sharding=("tensor", None)
                )
            }
        )
        # Divisible heads must already be sharded directly by the loader.
        if vocab == 32:
            assert model.lm_head.embedding.value.sharding.spec == weight_spec(dp_head)
        configure_lm_heads(model, mesh, dp_head)
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
    assert parser.parse_args(
        ["--model-path", "/unused", "--enable-dp-lm-head"]
    ).enable_dp_lm_head


@pytest.mark.parametrize("dp_head", [False, True])
@pytest.mark.parametrize("mode", ["EXTEND", "DECODE", "TARGET_VERIFY", "DRAFT_EXTEND"])
def test_forward_modes(mesh, dp_head, mode):
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    rng = np.random.default_rng(17)
    hidden = rng.normal(size=(16, 12)).astype(np.float32)
    weight = rng.normal(size=(32, 12)).astype(np.float32)
    dp = mesh.shape["data"]
    per_dp = 16 // dp
    with jax.set_mesh(mesh):
        head = ParallelLMHead(32, 12, dtype=jnp.float32, param_dtype=jnp.float32)
        head.embedding.value = prepare_weight(
            jax.device_put(weight, NamedSharding(mesh, P())), mesh, dp_head
        )
        proc = LogitsProcessor(32, mesh, soft_cap=3.0)
        proc.enable_dp_lm_head = dp_head
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
            md.accept_lens = jax.device_put(
                np.ones(dp, np.int32), NamedSharding(mesh, P("data"))
            )
            rows = np.arange(dp) * per_dp
        result = jax.jit(lambda h, w, m: proc(h, w, m))(
            jax.device_put(hidden, NamedSharding(mesh, P("data", None))), head, md
        )
        expected = 3.0 * np.tanh((hidden[rows] @ weight.T) / 3.0)
        np.testing.assert_allclose(
            result.next_token_logits, expected, rtol=2e-5, atol=2e-5
        )
        np.testing.assert_array_equal(result.hidden_states, hidden)


def test_model_without_lm_head_accepts_other_mesh_axes():
    # The common loader is also used by vision/audio models, whose meshes do
    # not necessarily have the language model's data/tensor axis names.
    class Encoder(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jnp.ones((4, 4)))

    mesh = Mesh(np.array(jax.devices()[:1]), ("encoder",))
    model = Encoder()
    original = model.weight.value
    assert lm_head_load_shardings(model, mesh, False) == {}
    configure_lm_heads(model, mesh, False)
    assert model.weight.value is original
