from __future__ import annotations

import json
import os
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from safetensors import safe_open

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _tpu_env in ("TPU_ACCELERATOR_TYPE", "TPU_WORKER_HOSTNAMES"):
    os.environ.pop(_tpu_env, None)

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

if jax is not None:
    from sgl_jax.srt.multimodal.configs.dits.flux_model_config import FluxModelConfig
    from sgl_jax.srt.multimodal.models.dits.flux import (
        FluxTransformer2DModel,
    )


MODEL_ROOT = Path(os.environ.get("FLUX_MODEL_PATH", "/models/FLUX1.0"))
TRANSFORMER_PATH = MODEL_ROOT / "transformer"
CONFIG_PATH = TRANSFORMER_PATH / "config.json"
INDEX_PATH = TRANSFORMER_PATH / "diffusion_pytorch_model.safetensors.index.json"


def _make_mesh():
    devices = np.array(jax.devices("cpu")[:1]).reshape((1, 1))
    try:
        return jax.sharding.Mesh(
            devices,
            ("data", "tensor"),
            axis_types=(
                jax.sharding.AxisType.Explicit,
                jax.sharding.AxisType.Explicit,
            ),
        )
    except TypeError:
        return jax.sharding.Mesh(devices, ("data", "tensor"))


def _mesh_context(mesh):
    try:
        return jax.sharding.use_mesh(mesh)
    except AttributeError:
        try:
            return jax.set_mesh(mesh)
        except AttributeError:
            return nullcontext()


def _load_flux_config() -> FluxModelConfig:
    with CONFIG_PATH.open() as f:
        config_json = json.load(f)

    init_kwargs = {k: v for k, v in config_json.items() if not k.startswith("_")}
    return FluxModelConfig(
        **init_kwargs,
        model_path=str(TRANSFORMER_PATH),
        dtype=jnp.bfloat16,
        weights_dtype=jnp.bfloat16,
        attention_impl="sdpa",
    )


def _load_hf_tensor(hf_key: str) -> np.ndarray:
    with INDEX_PATH.open() as f:
        weight_map = json.load(f)["weight_map"]

    if hf_key not in weight_map:
        raise KeyError(f"Weight {hf_key!r} not found in {INDEX_PATH}")

    tensor_file = TRANSFORMER_PATH / weight_map[hf_key]
    with safe_open(tensor_file, framework="np", device="cpu") as f:
        return np.asarray(f.get_tensor(hf_key))


def _to_numpy(x) -> np.ndarray:
    return np.asarray(jax.device_get(x), dtype=np.float32)


@unittest.skipIf(jax is None, "jax not installed")
class TestFluxTransformer2DWeightLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TRANSFORMER_PATH.exists():
            raise unittest.SkipTest(f"Model path not found: {TRANSFORMER_PATH}")
        if not CONFIG_PATH.exists():
            raise unittest.SkipTest(f"Config not found: {CONFIG_PATH}")
        if not INDEX_PATH.exists():
            raise unittest.SkipTest(f"Safetensors index not found: {INDEX_PATH}")

        cls.mesh = _make_mesh()
        cls.config = _load_flux_config()

        with _mesh_context(cls.mesh):
            cls.model = FluxTransformer2DModel(
                cls.config,
                dtype=cls.config.dtype,
                mesh=cls.mesh,
            )
            cls.model.load_weights(str(TRANSFORMER_PATH))

    def _assert_weight_loaded(
        self,
        hf_key: str,
        jax_value,
        *,
        transpose: bool = False,
        name: str | None = None,
    ) -> None:
        expected = _load_hf_tensor(hf_key)
        if transpose:
            expected = expected.T

        actual = _to_numpy(jax_value)
        expected = np.asarray(expected, dtype=np.float32)

        self.assertEqual(
            actual.shape,
            expected.shape,
            f"{name or hf_key}: shape mismatch {actual.shape} != {expected.shape}",
        )
        np.testing.assert_array_equal(actual, expected, err_msg=f"{name or hf_key}: value mismatch")

    def test_load_weights_matches_hf_safetensors(self):
        model = self.model

        checks = [
            (
                "time_text_embed.guidance_embedder.linear_1.weight",
                model.time_text_embed.guidance_embedder.linear_1.weight[...],
                True,
            ),
            (
                "time_text_embed.guidance_embedder.linear_2.bias",
                model.time_text_embed.guidance_embedder.linear_2.bias[...],
                False,
            ),
            ("context_embedder.weight", model.context_embedder.weight[...], True),
            (
                "transformer_blocks.0.norm1.linear.bias",
                model.transformer_blocks[0].norm1.linear.bias[...],
                False,
            ),
            (
                "transformer_blocks.0.attn.to_q.weight",
                model.transformer_blocks[0].attn.to_q.weight[...],
                True,
            ),
            (
                "transformer_blocks.0.attn.norm_added_q.weight",
                model.transformer_blocks[0].attn.norm_added_q.scale[...],
                False,
            ),
            (
                "transformer_blocks.0.ff.net.0.proj.weight",
                model.transformer_blocks[0].ff.net[0].weight[...],
                True,
            ),
            (
                "single_transformer_blocks.0.proj_out.weight",
                model.single_transformer_blocks[0].proj_out.weight[...],
                True,
            ),
            ("norm_out.linear.weight", model.norm_out.linear.weight[...], True),
            ("proj_out.bias", model.proj_out.bias[...], False),
        ]

        for hf_key, jax_value, transpose in checks:
            with self.subTest(weight=hf_key):
                self._assert_weight_loaded(hf_key, jax_value, transpose=transpose)


if __name__ == "__main__":
    unittest.main()
