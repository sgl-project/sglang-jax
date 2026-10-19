"""Tiny numerical tests; CPU GMM interpreter and reference attention, not TPU validation."""

import asyncio
import copy
import importlib.util
import os
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from safetensors.numpy import save_file
from transformers import Llama4Config, Llama4TextConfig

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ref_ragged_paged_attention,
    same_attention_chunk,
)
from sgl_jax.srt.models.llama4 import Llama4ForCausalLM, Llama4ForConditionalGeneration


def tiny_config(**kwargs):
    values = dict(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=128,
        intermediate_size_mlp=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_local_experts=3,
        num_experts_per_tok=1,
        moe_layers=[1],
        no_rope_layers=[1, 0],
        use_qk_norm=False,
        attention_chunk_size=4,
        max_position_embeddings=65536,
        floor_scale=4,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
    )
    values.update(kwargs)
    return Llama4TextConfig(**values)


def parameter(model, path):
    for part in path.split("."):
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    return model


def random_checkpoint(model):
    rng = np.random.default_rng(42)
    tensors = {}
    for key, spec in model._create_llama_weight_mappings().items():
        paths = spec.target_path if isinstance(spec.target_path, list) else [spec.target_path]
        values = []
        for path in paths:
            shape = parameter(model, path).value.shape
            value = rng.normal(0, 0.06, shape).astype(np.float32)
            if path.endswith(".scale"):
                value = np.ones(shape, np.float32)
            values.append(value)
        value = np.concatenate(values, axis=spec.split_axis) if len(values) > 1 else values[0]
        tensors[key] = value.T if spec.transpose else value
    return tensors


class ReferencePagedAttention:
    """Test-only backend exercising the repository's JAX paged reference and KV reuse."""

    supports_attention_chunk_size = True

    def __init__(self):
        self.cache = {}

    def __call__(self, q, k, v, layer, forward_batch, token_to_kv_pool):
        del token_to_kv_pool
        padded_count = q.shape[0]
        count = forward_batch.num_tokens
        q, k, v = q[:count], k[:count], v[:count]
        if layer.layer_id in self.cache:
            old_k, old_v = self.cache[layer.layer_id]
            k, v = jnp.concatenate((old_k, k)), jnp.concatenate((old_v, v))
        self.cache[layer.layer_id] = (k, v)
        count = k.shape[0]
        replicated = NamedSharding(jax.sharding.get_mesh(), PartitionSpec())
        q, k_ref, v_ref = (jax.device_put(value, replicated) for value in (q, k, v))
        out = ref_ragged_paged_attention(
            q,
            k_ref[:, None],
            v_ref[:, None],
            jnp.array([count]),
            jnp.arange(count)[None],
            jnp.array([0, q.shape[0]]),
            jnp.array([1]),
            sm_scale=layer.scaling,
            attention_chunk_size=layer.attention_chunk_size,
        )
        out = jnp.pad(out, ((0, padded_count - q.shape[0]), (0, 0), (0, 0)))
        return out.reshape(padded_count, -1), (k, v)


class TestLlama4(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh(
            np.array(jax.devices()[:1]).reshape(1, 1),
            ("data", "tensor"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )
        self.mesh_context = jax.set_mesh(self.mesh)
        self.mesh_context.__enter__()
        self.addCleanup(self.mesh_context.__exit__, None, None, None)
        self.folder = Path.cwd() / f".llama4-test-weights-{uuid.uuid4().hex}"
        self.folder.mkdir()
        self.addCleanup(shutil.rmtree, self.folder)

    def load(self, model, tensors):
        save_file(
            {name: np.ascontiguousarray(value) for name, value in tensors.items()},
            str(self.folder / "model.safetensors"),
        )
        config = model.config
        mc = SimpleNamespace(
            model_path=str(self.folder),
            revision=None,
            hf_config=config,
            hf_text_config=config,
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            get_total_num_kv_heads=lambda: config.num_key_value_heads,
            needs_kv_head_replication=lambda _: False,
        )
        model.load_weights(mc)

    def model(self, config=None):
        return Llama4ForCausalLM(config or tiny_config(), self.mesh, dtype=jnp.float32)

    def logits(self, model, ids, backend, start=0):
        # Match serving's padded token batches and the existing GMM tile alignment.
        padded_count = (len(ids) + 127) // 128 * 128
        batch = SimpleNamespace(
            input_ids=jnp.pad(jnp.asarray(ids, jnp.int32), (0, padded_count - len(ids))),
            positions=jnp.pad(jnp.arange(start, start + len(ids)), (0, padded_count - len(ids))),
            num_tokens=len(ids),
            attn_backend=backend,
        )
        hidden, _, _, _ = model.model(batch, None)
        head = model.model.embed_tokens if model.config.tie_word_embeddings else model.lm_head
        return model.logits_processor._get_logits(hidden, head)[: len(ids)]

    def test_strict_checkpoint_names_layouts_and_missing(self):
        for conditional in (False, True):
            with self.subTest(conditional=conditional):
                config = tiny_config()
                model = (
                    Llama4ForConditionalGeneration(
                        Llama4Config(text_config=config), self.mesh, dtype=jnp.float32
                    )
                    if conditional
                    else self.model(config)
                )
                tensors = random_checkpoint(model)
                if conditional:
                    tensors["vision_model.unused.weight"] = np.ones((2,), np.float32)
                self.load(model, tensors)
                prefix = model.checkpoint_prefix + "model.layers.1.feed_forward.experts."
                fused = tensors[prefix + "gate_up_proj"]
                experts = model.model.layers[1].mlp.experts
                np.testing.assert_array_equal(experts.wi_0.value, fused[:, :, :128])
                np.testing.assert_array_equal(experts.wi_1.value, fused[:, :, 128:])
                np.testing.assert_array_equal(experts.wo.value, tensors[prefix + "down_proj"])
                qkey = model.checkpoint_prefix + "model.layers.0.self_attn.q_proj.weight"
                np.testing.assert_array_equal(
                    model.model.layers[0].self_attn.q_proj.weight.value, tensors[qkey].T
                )
                tensors[prefix + "gate_up_proj"] = np.concatenate((fused, fused[:1]), axis=0)
                with self.assertRaisesRegex(ValueError, "Invalid Llama 4 expert shape"):
                    self.load(model, tensors)
                tensors[prefix + "gate_up_proj"] = fused
                del tensors[prefix + "down_proj"]
                with self.assertRaisesRegex(ValueError, "Missing checkpoint inputs"):
                    self.load(model, tensors)

    def test_router_and_input_scaled_moe(self):
        model = self.model(tiny_config(num_experts_per_tok=2))
        self.load(model, random_checkpoint(model))
        moe = model.model.layers[1].mlp
        x = jnp.asarray(np.random.default_rng(7).normal(size=(128, 128)), jnp.float32)
        logits = x @ moe.router.weight.value
        scores, ids = moe.topk(logits)
        weights = jax.nn.sigmoid(scores)
        np.testing.assert_array_equal(ids, np.argsort(-np.asarray(logits), axis=-1)[:, :2])
        np.testing.assert_allclose(
            weights,
            1 / (1 + np.exp(-np.take_along_axis(np.asarray(logits), np.asarray(ids), axis=1))),
            rtol=1e-6,
        )
        self.assertGreater(float(jnp.max(jnp.abs(weights.sum(axis=1) - 1))), 0.05)
        shared = moe.shared_expert(x)
        expected = shared
        wrong = shared
        for choice in range(2):
            w0 = np.asarray(moe.experts.wi_0.value)[np.asarray(ids[:, choice])]
            w1 = np.asarray(moe.experts.wi_1.value)[np.asarray(ids[:, choice])]
            wo = np.asarray(moe.experts.wo.value)[np.asarray(ids[:, choice])]

            def expert(inputs, w0=w0, w1=w1, wo=wo):
                gate = jnp.einsum("th,thi->ti", inputs, w0)
                up = jnp.einsum("th,thi->ti", inputs, w1)
                return jnp.einsum("ti,tih->th", jax.nn.silu(gate) * up, wo)

            expected = expected + expert(x * weights[:, choice, None])
            wrong = wrong + expert(x) * weights[:, choice, None]
        actual = moe(x)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        self.assertGreater(float(jnp.max(jnp.abs(actual - wrong))), 1e-3)
        self.assertGreater(float(jnp.max(jnp.abs(actual - shared))), 1e-3)

    def test_rotary_norm_temperature_and_schedule(self):
        model = self.model(tiny_config(use_qk_norm=True))
        self.load(model, random_checkpoint(model))
        x = jnp.asarray(np.random.default_rng(3).normal(size=(4, 128)), jnp.float32)
        positions = jnp.array([0, 3, 4, 32768])
        for index in range(2):
            attn = model.model.layers[index].self_attn
            q, k, _ = attn.prepare_qkv(positions, x)
            raw_q = np.asarray(x @ attn.q_proj.weight.value).reshape(4, 2, 64)
            if index == 0:
                inv_freq = 1 / (500000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / 64))
                angle = np.asarray(positions, dtype=np.float32)[:, None] * inv_freq
                pairs = raw_q.reshape(4, 2, 32, 2)
                z = (pairs[..., 0] + 1j * pairs[..., 1]) * np.exp(1j * angle[:, None])
                ref = np.stack((z.real, z.imag), axis=-1).reshape(raw_q.shape)
                ref /= np.sqrt(np.mean(ref**2, axis=-1, keepdims=True) + model.config.rms_norm_eps)
                self.assertEqual(attn.attn.attention_chunk_size, 4)
                np.testing.assert_allclose(np.mean(np.asarray(k) ** 2, axis=-1), 1, atol=1e-3)
            else:
                scales = 1 + 0.1 * np.log1p(np.floor((np.asarray(positions) + 1) / 4))
                ref = raw_q * scales[:, None, None]
                self.assertIsNone(attn.attn.attention_chunk_size)
            np.testing.assert_allclose(q, ref, rtol=2e-4, atol=2e-4)
        self.assertFalse(self.model().model.layers[0].self_attn.use_qk_norm)
        self.assertEqual(tiny_config(moe_layers=None, interleave_moe_layer_step=2).moe_layers, [1])
        self.assertEqual(tiny_config(moe_layers=[]).moe_layers, [])

    def test_chunk_mask_long_positions(self):
        for chunk in (4, 8192):
            positions = jnp.array([chunk - 1, chunk, chunk + 1, 32767, 32768, 65536])
            actual = jax.jit(same_attention_chunk, static_argnums=2)(
                positions[:, None], positions[None, :], chunk
            )
            expected = (
                np.asarray(positions)[:, None] // chunk == np.asarray(positions)[None, :] // chunk
            )
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(bool(actual[0, 1]))
            self.assertTrue(bool(actual[1, 2]))

    def test_prefill_decode_and_prefix_reuse(self):
        model = self.model()
        self.load(model, random_checkpoint(model))
        ids = [1, 7, 2, 9, 3, 5, 4, 6, 8]
        full = self.logits(model, ids, ReferencePagedAttention())
        backend = ReferencePagedAttention()
        pieces = [self.logits(model, ids[:3], backend)]
        fork = copy.deepcopy(backend.cache)
        pieces.append(self.logits(model, ids[3:6], backend, 3))
        for start in range(6, 9):
            pieces.append(self.logits(model, ids[start : start + 1], backend, start))
        np.testing.assert_allclose(jnp.concatenate(pieces), full, atol=3e-5, rtol=3e-5)
        branch = ReferencePagedAttention()
        branch.cache = fork
        np.testing.assert_allclose(
            self.logits(model, ids[3:], branch, 3), full[3:], atol=3e-5, rtol=3e-5
        )

    def test_config_and_image_rejection(self):
        from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sgl_jax.srt.managers.io_struct import GenerateReqInput
        from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
        from sgl_jax.srt.model_loader.arch import get_model_architecture

        for architecture, expected in (
            ("Llama4ForCausalLM", Llama4ForCausalLM),
            ("Llama4ForConditionalGeneration", Llama4ForConditionalGeneration),
        ):
            config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture]), model_impl="auto"
            )
            self.assertIs(get_model_architecture(config)[0], expected)

        mc = SimpleNamespace(is_multimodal=True)
        Llama4ForConditionalGeneration.patch_model_config(mc)
        self.assertFalse(mc.is_multimodal)
        manager = SimpleNamespace(model_config=mc)
        for field in ("image_data", "video_data", "audio_data"):
            request = GenerateReqInput(text="hello", **{field: "not-downloaded"})
            with self.assertRaisesRegex(ValueError, "text-only"):
                asyncio.run(TokenizerManager._tokenize_one_request(manager, request))
        serving = SimpleNamespace(tokenizer_manager=manager)
        chat = SimpleNamespace(
            messages=[
                SimpleNamespace(
                    content=[{"type": "image_url", "image_url": {"url": "not-downloaded"}}]
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "text-only"):
            OpenAIServingChat._convert_to_internal_request(serving, chat)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional HF PyTorch reference")
    def test_huggingface_logits(self):
        import torch
        from transformers import LlamaConfig
        from transformers.models.llama4.modeling_llama4 import (
            Llama4ForCausalLM as HFModel,
        )
        from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFLlama

        from sgl_jax.srt.models.llama import LlamaForCausalLM

        llama_config = LlamaConfig(
            vocab_size=32,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
        )
        ids = [1, 7, 2, 9, 3, 5, 4, 6, 8]
        for config, hf_class, native_class in (
            (tiny_config(), HFModel, Llama4ForCausalLM),
            (tiny_config(use_qk_norm=True), HFModel, Llama4ForCausalLM),
            (llama_config, HFLlama, LlamaForCausalLM),
        ):
            with self.subTest(
                model=config.model_type, qk_norm=getattr(config, "use_qk_norm", False)
            ):
                torch.manual_seed(31)
                config._attn_implementation = "eager"
                hf = hf_class(config).eval()
                native = native_class(config, self.mesh, dtype=jnp.float32)
                self.load(
                    native,
                    {key: value.detach().numpy().copy() for key, value in hf.state_dict().items()},
                )
                with torch.no_grad():
                    expected = hf(torch.tensor([ids]), use_cache=False).logits[0].numpy()
                actual = self.logits(native, ids, ReferencePagedAttention())
                np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
